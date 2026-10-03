"""
api.py — Hamilton's public interface.

Everything else in this project (main.py, raytracer.py, ionosphere.py,
igrf.py, mc_scattering.py, link_budget.py, absorption.py, geometry.py,
visualization.py, ECHAIM/) is the engine: correct, validated, but built as
a set of cooperating pieces you have to wire together yourself (build an
Ionosphere, build an IgrfField over the same patch, build a Ray from both,
hand that to a RayTracer or MCScatteringTracer, unpack a TraceResult...).
This module is the assembled version of that wiring, behind one class.

    from api import Link

    link = Link(57.0, 10.0, frequency_mhz=8.0, dt=datetime(2024, 6, 21, 14, 0))
    ray = link.trace(elevation_deg=28.0)
    print(ray)                          # human-readable one-shot summary
    ray.plot().write_html("ray.html")   # interactive 3-D, opens in a browser

    ensemble = link.scatter(elevation_deg=28.0, n_rays=200)
    budget = link.link_budget(elevation_deg=28.0)

A `Link` is a transmitter: a fixed place, a fixed moment in time (which
pins down the ionosphere and geomagnetic field it sees), and a default
carrier frequency and magneto-ionic mode. Everything expensive -- the
E-CHAIM/IRI-2020 electron-density query, the IGRF-13 field query -- happens
once, in the constructor, over a patch sized generously enough to cover
every ray you go on to ask for. Tracing individual rays afterward (even at
a different frequency or mode, or in a fan of many launch angles) is cheap,
since it reuses that same patch.

Units, deliberately not the engine's raw SI throughout: distances in km,
frequency in MHz, angles in degrees, loss in dB, delay in ms -- the units
someone doing HF link planning actually thinks in. Geocentric (ECEF)
positions are the one exception, kept in metres to match geometry.py and
so RayPath/ScatterResult can be handed straight to visualization.plot_ray
with no conversion.

Public names
------------
    Link              -- the entry point (see above)
    RayPath           -- result of Link.trace() / one member of a ScatterResult
    ScatterResult      -- result of Link.scatter()
    LinkBudget         -- result of Link.link_budget()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence, Union

import numpy as np

from constants import C
from geometry import xyz_m_to_latlon, latlon_to_xyz_m, ellipsoid_outward_normal_m
from main import build_ionosphere_for_ray, Ray
from igrf import build_igrf_for_ionosphere, IgrfField
from ionosphere import Ionosphere
from raytracer import RayTracer, TraceResult
from mc_scattering import IrregularitySpectrum, MCScatteringTracer
from link_budget import compute_receiver_statistics
from visualization import plot_ray as _plot_ray

__all__ = ["Link", "RayPath", "ScatterResult", "LinkBudget"]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

_MODE_TO_INT = {"o": 1, "x": -1, 1: 1, -1: -1}
_INT_TO_MODE = {1: "O", -1: "X"}


def _normalize_mode(mode: Union[str, int]) -> tuple[int, str]:
    key = mode.lower() if isinstance(mode, str) else mode
    if key not in _MODE_TO_INT:
        raise ValueError(f"mode must be 'O', 'X', +1, or -1 -- got {mode!r}")
    m = _MODE_TO_INT[key]
    return m, _INT_TO_MODE[m]


def _altitude_km(path_m: np.ndarray) -> np.ndarray:
    """(n,3) geocentric metres -> (n,) WGS84 altitude, km."""
    _, _, alt_m = xyz_m_to_latlon(np.asarray(path_m, dtype=float).T)
    return alt_m / 1e3


def _latlon_deg(path_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n,3) geocentric metres -> ((n,) lat deg, (n,) lon deg)."""
    lat, lon, _ = xyz_m_to_latlon(np.asarray(path_m, dtype=float).T)
    return lat, lon


def _arrival_elevation_deg(r_final: np.ndarray, u_final: np.ndarray) -> float:
    """Angle of arrival above the true (WGS84) local horizontal, degrees."""
    up = ellipsoid_outward_normal_m(r_final)
    u_hat = u_final / np.linalg.norm(u_final)
    return float(np.degrees(np.arcsin(np.clip(-np.dot(u_hat, up), -1.0, 1.0))))


# ---------------------------------------------------------------------------
# RayPath
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RayPath:
    """
    One traced HF ray, from launch to wherever it stopped.

    Geometry (`path`, `bounce_points`, `launch_point`, `final_point`,
    `final_direction`) is geocentric Cartesian (ECEF) metres -- Hamilton's
    native frame, and what visualization.plot_ray expects, so a RayPath
    can be plotted directly (`ray.plot()`, or pass several to
    `Link.plot(...)`). Everything else is in the units an HF link budget
    is normally read in: km, dB, ms, degrees.

    `status` is the quick answer to "what happened": "landed" (came back
    to the WGS84 ellipsoid), "escaped" (crossed the model's altitude
    ceiling, ~800 km, and kept going), or "incomplete" (used up its path
    budget, or genuinely stalled, without doing either -- rare; means the
    ray needs a longer max_path_km).
    """

    # --- what was asked for ---
    frequency_mhz: float
    mode: str                              # "O" or "X"
    elevation_deg: float
    azimuth_deg: float

    # --- outcome ---
    landed: bool
    escaped: bool
    hops_completed: int

    # --- geometry, ECEF metres (matches geometry.py / visualization.plot_ray) ---
    path: np.ndarray = field(repr=False)           # (n, 3)
    arc_length: np.ndarray = field(repr=False)      # (n,) m, matches `path` row for row
    bounce_points: np.ndarray = field(repr=False)   # (hops_completed, 3)
    launch_point: np.ndarray = field(repr=False)    # (3,)
    final_point: np.ndarray = field(repr=False)     # (3,)
    final_direction: np.ndarray = field(repr=False) # (3,) wave-normal, unit-ish

    # --- summary quantities ---
    ground_range_km: float
    arc_length_km: float
    phase_path_km: float
    group_path_km: float
    group_delay_ms: float                  # group_path_km / c -- the physical delay
    absorption_db: float
    arrival_elevation_deg: Optional[float] # None unless landed
    sigma_phi2: Optional[float]            # rad^2, None unless phase_variance=True was requested

    # --- advanced / optional ---
    tangent_map: Optional[np.ndarray] = field(default=None, repr=False)  # (6,6), only if tangent_map=True

    @property
    def status(self) -> str:
        if self.landed:
            return "landed"
        if self.escaped:
            return "escaped"
        return "incomplete"

    def altitude_km(self) -> np.ndarray:
        """WGS84 altitude, km, one value per row of `path`."""
        return _altitude_km(self.path)

    def latlon_deg(self) -> tuple[np.ndarray, np.ndarray]:
        """Geodetic (lat, lon) in degrees, one pair per row of `path`."""
        return _latlon_deg(self.path)

    def plot(self, **kwargs):
        """Interactive 3-D plot (plotly). See visualization.plot_ray for kwargs."""
        return _plot_ray(self, **kwargs)

    def __str__(self) -> str:
        lat, lon = _latlon_deg(self.final_point[None, :])
        arrival = f"{self.arrival_elevation_deg:.1f}deg" if self.arrival_elevation_deg is not None else "n/a"
        return (
            f"RayPath({self.frequency_mhz:g} MHz, {self.mode}-mode, "
            f"el={self.elevation_deg:g}deg az={self.azimuth_deg:g}deg)\n"
            f"  status         : {self.status} ({self.hops_completed} hop(s))\n"
            f"  landing        : {lat[0]:.2f}N {lon[0]:.2f}E, ground range {self.ground_range_km:.1f} km\n"
            f"  arrival angle  : {arrival}\n"
            f"  arc length     : {self.arc_length_km:.1f} km\n"
            f"  group path     : {self.group_path_km:.1f} km  (delay {self.group_delay_ms:.3f} ms)\n"
            f"  absorption     : {self.absorption_db:.2f} dB"
        )


def _build_ray_path(result: TraceResult, *, frequency_mhz: float, mode: str,
                     elevation_deg: float, azimuth_deg: float) -> RayPath:
    arrival = _arrival_elevation_deg(result.r_final, result.u_final) if result.landed else None
    return RayPath(
        frequency_mhz=frequency_mhz, mode=mode,
        elevation_deg=elevation_deg, azimuth_deg=azimuth_deg,
        landed=result.landed, escaped=result.escaped, hops_completed=result.hops_completed,
        path=result.path, arc_length=result.s_path,
        bounce_points=result.bounce_points,
        launch_point=result.path[0], final_point=result.r_final,
        final_direction=result.u_final,
        ground_range_km=result.ground_range_total_m / 1e3,
        arc_length_km=result.s_total / 1e3,
        phase_path_km=result.phase_path_total / 1e3,
        group_path_km=result.group_path_total / 1e3,
        group_delay_ms=1e3 * result.group_path_total / C,
        absorption_db=result.absorption_db_total,
        arrival_elevation_deg=arrival,
        sigma_phi2=(result.sigma_phi2 if result.sigma_phi2 else None),
        tangent_map=result.tangent_map_final,
    )


# ---------------------------------------------------------------------------
# ScatterResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScatterResult:
    """
    A Monte Carlo ensemble of scattered rays (mc_scattering.py's Langevin
    model of small-scale field-aligned ionospheric irregularities) around
    one deterministic reference ray.

    `reference` is the unscattered ray (same as Link.trace() would give
    for this launch geometry). `rays` are the ensemble members that
    landed (n_rays were launched; rays that didn't come down -- escaped,
    or ran out of path budget -- are excluded from both `rays` and every
    statistic below, matching mc_scattering.summarize()'s own convention).

    `coherent_fraction` = exp(-mean(sigma_phi2)/2) is the fraction of
    signal power that stays in a coherent wavefront rather than being
    scattered into phase noise -- 1.0 is undisturbed, well below that
    means the irregularities are strong enough to matter for anything
    phase-sensitive (Doppler, coherent integration, ionosonde autoscaling).
    """

    reference: RayPath
    rays: list[RayPath]
    n_rays: int
    n_landed: int

    range_mean_km: float
    range_std_km: float
    cross_range_std_km: float
    elevation_mean_deg: float
    elevation_std_deg: float
    group_path_mean_km: float
    group_delay_spread_ms: float
    absorption_mean_db: float
    absorption_std_db: float
    sigma_phi2_mean: float
    coherent_fraction: float

    def plot(self, **kwargs):
        """Interactive 3-D plot: reference ray highlighted, ensemble behind it."""
        return _plot_ray([self.reference] + self.rays, highlight=self.reference, **kwargs)

    def __str__(self) -> str:
        return (
            f"ScatterResult({self.reference.frequency_mhz:g} MHz, {self.reference.mode}-mode, "
            f"el={self.reference.elevation_deg:g}deg): {self.n_landed}/{self.n_rays} landed\n"
            f"  landing spread : {self.range_std_km:.2f} km down-range, "
            f"{self.cross_range_std_km:.2f} km cross-range (1 sigma)\n"
            f"  arrival angle  : {self.elevation_mean_deg:.2f} +/- {self.elevation_std_deg:.2f} deg\n"
            f"  delay spread   : {self.group_delay_spread_ms*1e3:.2f} us (1 sigma)\n"
            f"  absorption     : {self.absorption_mean_db:.2f} +/- {self.absorption_std_db:.2f} dB\n"
            f"  coherent frac. : {self.coherent_fraction:.3e}  (<E>/E0 = exp(-sigma_phi2/2))"
        )


# ---------------------------------------------------------------------------
# LinkBudget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinkBudget:
    """
    A relative dB report for one ray: how much the ionosphere focused or
    defocused it (ray-tube area, via the tangent map), how much power
    absorption cost it, and how much coherence it lost to small-scale
    scattering -- combined into `total_db`.

    This is RELATIVE only (see link_budget.py): there's no transmitter
    EIRP, antenna gain, or receiver sensitivity model anywhere in
    Hamilton, so getting from `total_db` to an actual received power is
    on you. `focusing_gain_db` (and therefore `total_db`) is None if the
    ray didn't land -- there's no receiver, and no ray tube to measure,
    for a ray that escaped or ran out of budget.

    `ray` is the full underlying RayPath (with its tangent map attached),
    for anything not surfaced here directly.
    """

    landed: bool
    escaped: bool
    ground_range_km: float

    focusing_gain_db: Optional[float]
    absorption_db: float
    coherence_loss_db: float
    total_db: Optional[float]

    sigma_phi2: float
    coherent_fraction: float

    ray: RayPath

    def __str__(self) -> str:
        foc = f"{self.focusing_gain_db:+.2f} dB" if self.focusing_gain_db is not None else "n/a"
        tot = f"{self.total_db:+.2f} dB" if self.total_db is not None else "n/a"
        return (
            f"LinkBudget({self.ray.frequency_mhz:g} MHz, {self.ray.mode}-mode, "
            f"el={self.ray.elevation_deg:g}deg): {'landed' if self.landed else 'did not land'}, "
            f"{self.ground_range_km:.1f} km\n"
            f"  focusing gain  : {foc}\n"
            f"  absorption     : -{self.absorption_db:.2f} dB\n"
            f"  coherence loss : -{self.coherence_loss_db:.2f} dB\n"
            f"  total          : {tot}  (relative only -- see class docstring)"
        )


# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------

class Link:
    """
    An HF transmitter: a fixed place and moment in time, ready to trace
    rays at whatever launch angles (and, per call if you like, frequency
    or mode) you ask for.

    Construction does the expensive part once: an E-CHAIM+IRI-2020
    electron-density patch and an IGRF-13 field patch, both centred on
    (lat_deg, lon_deg) and sized generously enough (see max_path_km,
    max_nhops) to cover every ray you'll go on to trace. Everything after
    that -- trace(), scatter(), link_budget(), the sweep_*() helpers --
    reuses that same patch and is correspondingly cheap.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Geodetic transmitter location (degrees).
    alt_m : float
        Transmitter altitude above the WGS84 ellipsoid (metres). Default 0.
    frequency_mhz : float
        Default carrier frequency (MHz). Override per call via the
        `frequency_mhz=` argument of trace()/scatter()/link_budget() --
        cheap, since it's the ionosphere/field query that's expensive,
        not building a Ray for a new frequency.
    dt : datetime or None
        UTC date/time -- determines solar/geomagnetic indices, season,
        and local time per longitude for both background models. Default:
        now (UTC). Pass this explicitly for reproducible results.
    mode : "O", "X", +1, or -1
        Default magneto-ionic mode (Appleton-Hartree branch). Override
        per call the same way as frequency_mhz.
    max_path_km : float
        The longest per-hop path length you intend to trace. Sizes the
        ionosphere/field patch (along with max_nhops) -- calls to
        trace()/scatter()/link_budget() default to this, and may use a
        *smaller* max_path_km, but asking for a longer one than the patch
        was built for will silently run past its edge (Ne/B both fall
        back to boundary values there, which is not what you want).
        Default 2000 km.
    max_nhops : int
        The most ground bounces you intend to request in a single trace()
        call. Used only to size the patch (multiplied against
        max_path_km). Default 1.
    use_echaim : bool
        Prefer E-CHAIM over IRI-2020 wherever E-CHAIM is valid (AACGM
        geomagnetic latitude >= 45 degrees); see ionosphere.py. Default
        True.
    n_ang, dr_km, dang_deg : int, float, float
        Ionosphere query/grid resolution -- see ionosphere.py /
        main.py::build_ionosphere_for_ray. Defaults 15, 2.0, 1.0.
    igrf_n_ang, igrf_n_r : int
        IGRF query grid resolution -- see igrf.py. Defaults 5, 5.
    xi_rms, L0_km, p_index, axial_ratio : float
        Default field-aligned irregularity spectrum, used by scatter()
        and link_budget() (mc_scattering.py's IrregularitySpectrum).
        Defaults: 5e-3 (quiet-to-moderate), 5 km, 4.0, 10.0.
    absorption_model : str or callable(alt_km) -> nu_eff
        Collision-frequency profile for absorption -- see absorption.py.
        Default "default" (illustrative D/E-region profile).
    verbose : bool
        Print progress (ionosphere/field build, per-ray scatter progress)
        to stdout. Default True.
    """

    def __init__(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float = 0.0,
        *,
        frequency_mhz: float,
        dt: Optional[datetime] = None,
        mode: Union[str, int] = "O",
        max_path_km: float = 2000.0,
        max_nhops: int = 1,
        use_echaim: bool = True,
        n_ang: int = 15,
        dr_km: float = 2.0,
        dang_deg: float = 1.0,
        igrf_n_ang: int = 5,
        igrf_n_r: int = 5,
        xi_rms: float = 5e-3,
        L0_km: float = 5.0,
        p_index: float = 4.0,
        axial_ratio: float = 10.0,
        absorption_model: str = "default",
        verbose: bool = True,
    ):
        if not (-90.0 <= lat_deg <= 90.0):
            raise ValueError(f"lat_deg must be in [-90, 90] -- got {lat_deg}")
        if frequency_mhz <= 0.0:
            raise ValueError(f"frequency_mhz must be positive -- got {frequency_mhz}")
        if max_path_km <= 0.0:
            raise ValueError(f"max_path_km must be positive -- got {max_path_km}")
        if max_nhops < 1:
            raise ValueError(f"max_nhops must be >= 1 -- got {max_nhops}")
        _normalize_mode(mode)  # validate early

        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.alt_m = float(alt_m)
        self.frequency_mhz = float(frequency_mhz)
        self.dt = dt if dt is not None else datetime.now(timezone.utc)
        self.mode = mode
        self.max_path_km = float(max_path_km)
        self.max_nhops = int(max_nhops)
        self.use_echaim = use_echaim
        self.absorption_model = absorption_model
        self.xi_rms, self.L0_km = xi_rms, L0_km
        self.p_index, self.axial_ratio = p_index, axial_ratio
        self.verbose = verbose

        self.r0 = latlon_to_xyz_m(self.lat_deg, self.lon_deg, self.alt_m)
        self._up, self._north, self._east = _enu_frame(self.lat_deg, self.lon_deg)

        # Placeholder launch direction ("up") for sizing the patch: for a
        # ground/near-ground transmitter the ray always starts INSIDE the
        # ionosphere grid (build_ionosphere_for_ray's simpler branch),
        # where the resulting patch is centred on the transmitter and
        # sized from max_path_m alone -- so which direction is passed
        # here doesn't affect the result, only alt0_m being inside the
        # grid does.
        if self.verbose:
            print(f"[Link] building ionosphere + field patch for "
                  f"{lat_deg:.2f}N {lon_deg:.2f}E, {self.dt.isoformat()} ...")
        self._ionosphere: Ionosphere = build_ionosphere_for_ray(
            lat0_deg=self.lat_deg, lon0_deg=self.lon_deg, alt0_m=self.alt_m,
            direction=self._up, max_path_m=self.max_path_km * self.max_nhops * 1e3,
            dt=self.dt, n_ang=n_ang, use_echaim=self.use_echaim,
            dr_km=dr_km, dang_deg=dang_deg,
        )
        self._field: IgrfField = build_igrf_for_ionosphere(
            self._ionosphere, n_ang=igrf_n_ang, n_r=igrf_n_r,
        )
        if self.verbose:
            self._ionosphere.grid_info()

        self._ray_cache: dict[tuple[float, int], Ray] = {}
        self._spectrum_cache: dict[float, IrregularitySpectrum] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _direction(self, elevation_deg: float, azimuth_deg: float) -> np.ndarray:
        if not (0.0 <= elevation_deg <= 90.0):
            raise ValueError(f"elevation_deg must be in [0, 90] -- got {elevation_deg}")
        el, az = np.deg2rad(elevation_deg), np.deg2rad(azimuth_deg)
        horiz = np.cos(az) * self._north + np.sin(az) * self._east
        d = np.cos(el) * horiz + np.sin(el) * self._up
        return d / np.linalg.norm(d)

    def _resolve_ray(self, frequency_mhz: Optional[float], mode: Optional[Union[str, int]]) -> Ray:
        freq = self.frequency_mhz if frequency_mhz is None else float(frequency_mhz)
        if freq <= 0.0:
            raise ValueError(f"frequency_mhz must be positive -- got {freq}")
        mode_int, _ = _normalize_mode(self.mode if mode is None else mode)
        key = (freq, mode_int)
        if key not in self._ray_cache:
            omega = 2.0 * np.pi * freq * 1e6
            self._ray_cache[key] = Ray(
                omega=omega, MODE=mode_int,
                electron_density=self._ionosphere.electron_density_xyz_m,
                B_field=self._field.B_vec_xyz_m,
            )
        return self._ray_cache[key]

    def _resolve_spectrum(self, frequency_mhz: Optional[float]) -> IrregularitySpectrum:
        freq = self.frequency_mhz if frequency_mhz is None else float(frequency_mhz)
        if freq not in self._spectrum_cache:
            spec = IrregularitySpectrum(
                xi_rms=self.xi_rms, L0_m=self.L0_km * 1e3,
                p_index=self.p_index, axial_ratio=self.axial_ratio,
            )
            k0 = 2.0 * np.pi * freq * 1e6 / C
            spec.tabulate(k0=k0)
            self._spectrum_cache[freq] = spec
        return self._spectrum_cache[freq]

    def _path_budget_m(self, max_path_km: Optional[float]) -> float:
        return (self.max_path_km if max_path_km is None else float(max_path_km)) * 1e3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace(
        self,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        *,
        max_path_km: Optional[float] = None,
        nhops: int = 1,
        method: str = "fixed",
        frequency_mhz: Optional[float] = None,
        mode: Optional[Union[str, int]] = None,
        absorption: bool = True,
        group_path: bool = True,
        phase_variance: bool = False,
        tangent_map: bool = False,
        ds_step_km: float = 2.0,
    ) -> RayPath:
        """
        Trace one deterministic ray.

        elevation_deg, azimuth_deg : launch angles above the local
            horizontal and clockwise from true north, degrees.
        max_path_km, nhops : path budget per hop, and how many ground
            bounces to allow (specular reflection off the true WGS84
            ellipsoid). Default max_path_km is this Link's own
            max_path_km; don't exceed what the Link was built to cover
            (see Link's max_path_km/max_nhops docs).
        method : "fixed" (default -- fixed-step RK4, fast, what the
            Monte Carlo ensemble also uses) or "adaptive" (solve_ivp
            RK45 with tight error control -- slower, and required for
            tangent_map=True; requested automatically if you set
            tangent_map=True without also setting method="adaptive").
        frequency_mhz, mode : override this Link's defaults for just
            this ray (cheap -- reuses the already-built ionosphere/field).
        absorption, group_path : include non-deviative D/E-region
            absorption (absorption.py) and the group-path/group-index
            calculation (raytracer.py) -- both on by default, each is a
            real (if small) per-step cost if you don't need them.
        phase_variance : accumulate sigma_phi2, the deterministic
            phase-variance rate from the same irregularity spectrum
            scatter() draws kicks from -- gives you RayPath.sigma_phi2
            without running a full ensemble. Off by default.
        tangent_map : carry the 6x6 ray-tube tangent map (sensitivity of
            landing state to launch angle) -- gives you
            RayPath.tangent_map. This is what link_budget() uses
            internally for focusing_gain_db; call link_budget() directly
            if that's what you actually want. Off by default (real cost:
            forces method="adaptive" and a second integration pass).
        ds_step_km : integration step for method="fixed" only.

        Returns
        -------
        RayPath
        """
        ray = self._resolve_ray(frequency_mhz, mode)
        spectrum = self._resolve_spectrum(frequency_mhz) if phase_variance else None
        tracer = RayTracer(ray, spectrum=spectrum, ds_step_m=ds_step_km * 1e3,
                            absorption_model=self.absorption_model)

        if tangent_map and method != "adaptive":
            method = "adaptive"

        u0 = self._direction(elevation_deg, azimuth_deg)
        result = tracer.trace(
            self.r0, u0, self._path_budget_m(max_path_km),
            nhops=nhops, method=method, carry_tangent_map=tangent_map,
            compute_group_path=group_path, compute_absorption=absorption,
            compute_phase_variance=phase_variance,
        )
        _, mode_label = _normalize_mode(self.mode if mode is None else mode)
        freq = self.frequency_mhz if frequency_mhz is None else frequency_mhz
        return _build_ray_path(result, frequency_mhz=freq, mode=mode_label,
                                elevation_deg=elevation_deg, azimuth_deg=azimuth_deg)

    def scatter(
        self,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        *,
        n_rays: int = 200,
        max_path_km: Optional[float] = None,
        nhops: int = 1,
        frequency_mhz: Optional[float] = None,
        mode: Optional[Union[str, int]] = None,
        ds_step_km: float = 2.0,
        seed: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> ScatterResult:
        """
        Trace a Monte Carlo ensemble of n_rays scattered rays (Langevin
        kicks drawn from this Link's irregularity spectrum -- see
        mc_scattering.py) around the deterministic ray for this launch
        geometry, plus that deterministic ray itself as `.reference`.

        seed : RNG seed, for a reproducible ensemble. None (default) is
            fresh randomness each call.
        verbose : print "ray i/n_rays" progress every 25 rays. Defaults
            to this Link's own verbose setting.

        Returns
        -------
        ScatterResult
        """
        ray = self._resolve_ray(frequency_mhz, mode)
        spectrum = self._resolve_spectrum(frequency_mhz)
        mc = MCScatteringTracer(ray, spectrum, ds_step_m=ds_step_km * 1e3,
                                 seed=seed, record_every=5)

        u0 = self._direction(elevation_deg, azimuth_deg)
        budget_m = self._path_budget_m(max_path_km)
        verbose = self.verbose if verbose is None else verbose

        det = mc.trace(self.r0, u0, budget_m, scatter=False, nhops=nhops)
        results = mc.ensemble(self.r0, u0, n_rays=n_rays, max_path_m=budget_m,
                               nhops=nhops, verbose_every=(25 if verbose else 0))
        summ = mc.summarize(results, self.r0)

        _, mode_label = _normalize_mode(self.mode if mode is None else mode)
        freq = self.frequency_mhz if frequency_mhz is None else frequency_mhz

        def _rp(res) -> RayPath:
            return RayPath(
                frequency_mhz=freq, mode=mode_label,
                elevation_deg=elevation_deg, azimuth_deg=azimuth_deg,
                landed=res.landed, escaped=res.escaped, hops_completed=res.hops_completed,
                path=res.path, arc_length=res.s_path, bounce_points=res.bounce_points,
                launch_point=res.path[0], final_point=res.r_final, final_direction=res.u_final,
                ground_range_km=res.ground_range_total_m / 1e3,
                arc_length_km=res.s_total / 1e3,
                phase_path_km=res.phase_path_total / 1e3,
                group_path_km=res.group_path_total / 1e3,
                group_delay_ms=1e3 * res.group_path_total / C,
                absorption_db=res.absorption_db_total,
                arrival_elevation_deg=(_arrival_elevation_deg(res.r_final, res.u_final) if res.landed else None),
                sigma_phi2=(res.sigma_phi2 if res.sigma_phi2 else None),
                tangent_map=None,
            )

        reference = _rp(det)
        if summ.get("n_landed", 0) == 0:
            return ScatterResult(
                reference=reference, rays=[], n_rays=n_rays, n_landed=0,
                range_mean_km=float("nan"), range_std_km=float("nan"), cross_range_std_km=float("nan"),
                elevation_mean_deg=float("nan"), elevation_std_deg=float("nan"),
                group_path_mean_km=float("nan"), group_delay_spread_ms=float("nan"),
                absorption_mean_db=float("nan"), absorption_std_db=float("nan"),
                sigma_phi2_mean=float("nan"), coherent_fraction=float("nan"),
            )

        landed_rays = [_rp(r) for r in results if r.landed]
        return ScatterResult(
            reference=reference, rays=landed_rays, n_rays=n_rays, n_landed=summ["n_landed"],
            range_mean_km=summ["range_mean_km"], range_std_km=summ["range_std_km"],
            cross_range_std_km=summ["cross_std_km"],
            elevation_mean_deg=summ["elev_mean_deg"], elevation_std_deg=summ["elev_std_deg"],
            group_path_mean_km=summ["group_path_mean_km"],
            group_delay_spread_ms=1e3 * summ["group_path_std_km"] * 1e3 / C,
            absorption_mean_db=summ["absorption_db_mean"], absorption_std_db=summ["absorption_db_std"],
            sigma_phi2_mean=summ["sigma_phi2_mean"], coherent_fraction=summ["coherent_fraction"],
        )

    def link_budget(
        self,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        *,
        max_path_km: Optional[float] = None,
        nhops: int = 1,
        frequency_mhz: Optional[float] = None,
        mode: Optional[Union[str, int]] = None,
    ) -> LinkBudget:
        """
        Relative dB link-budget report for one ray: focusing gain (from
        the ray-tube tangent map), absorption, and coherence loss,
        combined into total_db. See LinkBudget's docstring for what
        "relative" means here. Runs a single adaptive trace with the
        tangent map, group/phase path, absorption, and phase variance
        all enabled -- see link_budget.py::compute_receiver_statistics.

        Returns
        -------
        LinkBudget
        """
        ray = self._resolve_ray(frequency_mhz, mode)
        spectrum = self._resolve_spectrum(frequency_mhz)
        tracer = RayTracer(ray, spectrum=spectrum, absorption_model=self.absorption_model)

        u0 = self._direction(elevation_deg, azimuth_deg)
        stats = compute_receiver_statistics(tracer, self.r0, u0, self._path_budget_m(max_path_km),
                                             nhops=nhops)

        _, mode_label = _normalize_mode(self.mode if mode is None else mode)
        freq = self.frequency_mhz if frequency_mhz is None else frequency_mhz
        ray_path = _build_ray_path(stats.result, frequency_mhz=freq, mode=mode_label,
                                    elevation_deg=elevation_deg, azimuth_deg=azimuth_deg)
        return LinkBudget(
            landed=stats.landed, escaped=stats.escaped,
            ground_range_km=stats.ground_range_total_m / 1e3,
            focusing_gain_db=stats.focusing_gain_db,
            absorption_db=stats.absorption_db,
            coherence_loss_db=stats.coherence_loss_db,
            total_db=stats.total_db,
            sigma_phi2=stats.sigma_phi2, coherent_fraction=stats.coherent_fraction,
            ray=ray_path,
        )

    def sweep_elevation(self, elevations_deg: Sequence[float], azimuth_deg: float = 0.0,
                         **trace_kwargs) -> list[RayPath]:
        """trace() at each of several elevation angles. **trace_kwargs forwarded as-is."""
        return [self.trace(el, azimuth_deg, **trace_kwargs) for el in elevations_deg]

    def sweep_azimuth(self, elevation_deg: float, azimuths_deg: Sequence[float],
                       **trace_kwargs) -> list[RayPath]:
        """trace() at each of several azimuths, fixed elevation. **trace_kwargs forwarded as-is."""
        return [self.trace(elevation_deg, az, **trace_kwargs) for az in azimuths_deg]

    def sweep_frequency(self, elevation_deg: float, frequencies_mhz: Sequence[float],
                         azimuth_deg: float = 0.0, **trace_kwargs) -> list[RayPath]:
        """
        trace() at each of several frequencies, fixed launch geometry.
        Reuses this Link's ionosphere/field patch -- only a new Ray (cheap)
        is built per frequency. **trace_kwargs forwarded as-is.
        """
        return [self.trace(elevation_deg, azimuth_deg, frequency_mhz=f, **trace_kwargs)
                for f in frequencies_mhz]

    def plot(self, *rays, highlight=None, title: Optional[str] = None, **kwargs):
        """
        Interactive 3-D plot (plotly) of one or more RayPath/ScatterResult
        objects (or bare lists of them). A ScatterResult contributes its
        reference ray (as the highlight, unless you pass your own) plus
        its full ensemble. See visualization.plot_ray for other kwargs
        (show_bounce_points, ensemble_stride, ...).
        """
        flat: list[RayPath] = []
        for r in rays:
            if isinstance(r, ScatterResult):
                flat.append(r.reference)
                flat.extend(r.rays)
                if highlight is None:
                    highlight = r.reference
            elif isinstance(r, (list, tuple)):
                flat.extend(r)
            else:
                flat.append(r)
        return _plot_ray(flat, highlight=highlight, title=title, **kwargs)

    def grid_info(self) -> None:
        """Print a summary of the ionosphere patch this Link built (see Ionosphere.grid_info)."""
        self._ionosphere.grid_info()

    @property
    def ionosphere(self) -> Ionosphere:
        """The underlying E-CHAIM+IRI-2020 Ionosphere patch, for direct inspection/plotting."""
        return self._ionosphere

    @property
    def magnetic_field(self) -> IgrfField:
        """The underlying IGRF-13 field patch, for direct inspection."""
        return self._field

    def __repr__(self) -> str:
        return (f"Link({self.lat_deg:.2f}, {self.lon_deg:.2f}, "
                f"frequency_mhz={self.frequency_mhz:g}, mode={self.mode!r}, "
                f"dt={self.dt.isoformat()!r})")


def _enu_frame(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local (up, north, east) unit vectors in ECEF, at a given geodetic (lat, lon)."""
    lat_r, lon_r = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    up = np.array([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    north = np.array([-np.sin(lat_r) * np.cos(lon_r), -np.sin(lat_r) * np.sin(lon_r), np.cos(lat_r)])
    east = np.array([-np.sin(lon_r), np.cos(lon_r), 0.0])
    return up, north, east