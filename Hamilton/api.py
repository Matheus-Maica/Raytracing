"""
api.py -- Hamilton's public interface.

The rest of the project (core/, geophysics/, raytracing/, propagation/, stochastic/, echoes/, receiver/, plotting/;
see README.md) is the engine. This module wires it together behind one class, `Link`: an HF transmitter at a fixed
place and time, whose ionosphere (E-CHAIM + IRI-2020) and magnetic field (IGRF) are built once, in the constructor.
Everything after that reuses them.

    from datetime import datetime
    from api import Link

    link = Link(57.0, 10.0, frequency_mhz=8.0, dt=datetime(2024, 6, 21, 14, 0))
    ray = link.trace(elevation_deg=28.0)                  # RayPath
    print(ray)
    fan = link.sweep_elevation(range(5, 50, 4))           # [RayPath, ...]
    budget = link.link_budget(elevation_deg=28.0)         # LinkBudget: one-way dB budget of the leg
    channel = link.channel(ray)                           # ChannelParams: HiCIRF stochastic channel of the leg

    link.plot(fan, highlight=ray).savefig("rays.png")     # PHaRLAP-style figure + the red ray's HiCIRF channel
    link.plot_3d(fan)                                     # PHaRLAP's 3-D lat/lon/height figure
    link.plot_scattering_function(ray)                    # HiCIRF delay-Doppler scattering function
    link.plot_html(fan).write_html("rays.html")           # interactive 3-D (Plotly)

Public names
------------
    Link               the entry point
    RayPath            one traced ray (Link.trace, the sweep_* methods)
    LinkBudget         one-way budget of one leg (Link.link_budget)
    ScatterResult      Monte Carlo ensemble (Link.scatter -- the deprecated Monte Carlo layer, decision D6)
    ChannelParams      HiCIRF channel of one leg (Link.channel, Link.leg_channel), from core.records
    ChannelOptions     the channel options the spec leaves open, from stochastic.channel
    IrregularityInputs the irregularity inputs of a Link, from core.inputs

Units: km, MHz, degrees, dB, ms -- the units of HF link planning -- except ECEF positions (RayPath.path,
launch_point, ...), which stay in metres, as core.geometry and plotting/ take them.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence, Union

import numpy as np

from core.constants import C
from core.geometry import ellipsoid_outward_normal_m, latlon_to_xyz_m, xyz_m_to_latlon
from core.inputs import IrregularityInputs
from core.records import ChannelParams, RaySamples
from geophysics import environment
from geophysics.igrf import IgrfField, build_igrf_for_ionosphere
from geophysics.ionosphere import Ionosphere
from propagation.link_budget import one_way_budget
from raytracing.batch import BatchTracer, samples_from_record
from raytracing.main import Ray, build_ionosphere_for_ray
from raytracing.mc_scattering import IrregularitySpectrum, MCScatteringTracer
from stochastic.channel import ChannelOptions, channel_parameters
from stochastic.leg_screens import screens_for_leg

__all__ = ["Link", "RayPath", "LinkBudget", "ScatterResult", "ChannelParams", "ChannelOptions",
           "IrregularityInputs"]

Mode = Union[str, int]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

_MODE_TO_INT = {"o": 1, "x": -1, 1: 1, -1: -1}
_INT_TO_MODE = {1: "O", -1: "X"}


def _normalize_mode(mode: Mode) -> tuple[int, str]:
    """(+1 / -1, "O" / "X") from "O", "X" (any case), +1 or -1."""
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


def _enu_frame(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local (up, north, east) unit vectors in ECEF, at a given geodetic (lat, lon)."""
    lat_r, lon_r = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    up = np.array([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    north = np.array([-np.sin(lat_r) * np.cos(lon_r), -np.sin(lat_r) * np.sin(lon_r), np.cos(lat_r)])
    east = np.array([-np.sin(lon_r), np.cos(lon_r), 0.0])
    return up, north, east


# ---------------------------------------------------------------------------
# RayPath
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RayPath:
    """
    One traced HF ray, from launch to wherever it stopped.

    Geometry (`path`, `bounce_points`, `launch_point`, `final_point`, `final_direction`) is geocentric Cartesian
    (ECEF) metres, Hamilton's native frame; everything else is in km, dB, ms and degrees.

    `status` is the quick answer to "what happened": "landed" (came back to the WGS84 ellipsoid), "escaped" (crossed
    the model's ceiling, ~800 km, and kept going) or "incomplete" (used up its path budget first: give it a longer
    max_path_km). `hops_completed` counts finished hops (a landed one-hop ray has 1).

    `samples` is the full per-step record of the leg (core.records.RaySamples: position, directions, phase and
    group path, Ne, B, the transported frame ...). It is what Link.channel() builds the leg's phase screens from, so
    the HiCIRF channel is always that of this very ray. Rays from the deprecated Link.scatter() have none.

    Equality is identity (the fields are arrays); compare `path` etc. explicitly if needed.
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

    # --- geometry, ECEF metres ---
    path: np.ndarray = field(repr=False)            # (n, 3)
    arc_length: np.ndarray = field(repr=False)      # (n,) m, row for row with `path`
    bounce_points: np.ndarray = field(repr=False)   # (number of bounces, 3)
    launch_point: np.ndarray = field(repr=False)    # (3,)
    final_point: np.ndarray = field(repr=False)     # (3,)
    final_direction: np.ndarray = field(repr=False)  # (3,) wave normal, unit-ish

    # --- summary quantities ---
    ground_range_km: float
    arc_length_km: float
    phase_path_km: float
    group_path_km: float
    group_delay_ms: float                  # group_path_km / c: the physical delay
    absorption_db: float
    arrival_elevation_deg: Optional[float]  # None unless landed
    sigma_phi2: Optional[float]            # rad^2 from the leg's phase screens; None unless trace(phase_variance=True)

    # --- the traced leg itself ---
    samples: Optional[RaySamples] = field(default=None, repr=False)

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
        """PHaRLAP-style range-height figure of this ray (matplotlib Figure; plotting.figures.ray_slice_figure).
        A RayPath carries no ionosphere, so the background is empty, as in PHaRLAP's ray-only plots; Link.plot(ray)
        draws it over the Link's ionosphere, with its HiCIRF channel."""
        from plotting.figures import ray_slice_figure
        return ray_slice_figure([self], highlight=self, **kwargs)

    def plot_3d(self, **kwargs):
        """PHaRLAP's 3-D latitude/longitude/height figure (plotting.figures.rays_3d_figure)."""
        from plotting.figures import rays_3d_figure
        return rays_3d_figure([self], **kwargs)

    def plot_html(self, **kwargs):
        """The interactive 3-D Plotly scene (plotting.plotly_3d.plot_ray); .write_html(path) saves it."""
        from plotting.plotly_3d import plot_ray
        return plot_ray(self, **kwargs)

    def __str__(self) -> str:
        lat, lon = _latlon_deg(self.final_point[None, :])
        arrival = f"{self.arrival_elevation_deg:.1f}deg" if self.arrival_elevation_deg is not None else "n/a"
        lines = [
            f"RayPath({self.frequency_mhz:g} MHz, {self.mode}-mode, "
            f"el={self.elevation_deg:g}deg az={self.azimuth_deg:g}deg)",
            f"  status         : {self.status} ({self.hops_completed} hop(s))",
            f"  end point      : {lat[0]:.2f}N {lon[0]:.2f}E, ground range {self.ground_range_km:.1f} km",
            f"  arrival angle  : {arrival}",
            f"  arc length     : {self.arc_length_km:.1f} km",
            f"  group path     : {self.group_path_km:.1f} km  (delay {self.group_delay_ms:.3f} ms)",
            f"  absorption     : {self.absorption_db:.2f} dB",
        ]
        if self.sigma_phi2 is not None:
            lines.append(f"  phase variance : {self.sigma_phi2:.3g} rad^2")
        return "\n".join(lines)


def _ray_path_from_batch(res, samples, *, frequency_mhz: float, mode: str, elevation_deg: float,
                         azimuth_deg: float, sigma_phi2: Optional[float]) -> RayPath:
    """RayPath of one ray from a BatchResult (single ray) and its RaySamples (batch.samples_from_record)."""
    landed = bool(res.landed[0])
    arrival = _arrival_elevation_deg(res.r_final[0], res.u_final[0]) if landed else None
    bounces = np.asarray(res.bounce_points[0], dtype=float).reshape(-1, 3)
    return RayPath(
        frequency_mhz=frequency_mhz, mode=mode,
        elevation_deg=elevation_deg, azimuth_deg=azimuth_deg,
        landed=landed, escaped=bool(res.escaped[0]), hops_completed=int(res.hops[0]),
        path=np.asarray(samples.r, dtype=float), arc_length=np.asarray(samples.s, dtype=float),
        bounce_points=bounces,
        launch_point=np.asarray(samples.r[0], dtype=float), final_point=np.asarray(res.r_final[0], dtype=float),
        final_direction=np.asarray(res.u_final[0], dtype=float),
        ground_range_km=float(res.ground_range_m[0]) / 1e3,
        arc_length_km=float(res.s_total[0]) / 1e3,
        phase_path_km=float(res.phase_path[0]) / 1e3,
        group_path_km=float(res.group_path[0]) / 1e3,
        group_delay_ms=1e3 * float(res.group_path[0]) / C,
        absorption_db=float(res.absorption_db[0]),
        arrival_elevation_deg=arrival,
        sigma_phi2=sigma_phi2,
        samples=samples,
    )


# ---------------------------------------------------------------------------
# LinkBudget
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class LinkBudget:
    """
    The one-way budget of one leg (spec §6.2-§6.4, eq. 6.2-6.3): flux-tube focusing with the caustic cap (§4.5),
    absorption, ground losses at every bounce, and the 3.01 dB polarization loss. There is no coherence term (spec
    §6.3); the coherent/diffuse split is the HiCIRF channel's (Link.channel).

    `total_db` is relative: focusing gain minus the losses. Hamilton has no transmitter EIRP, antenna gain or
    receiver model, so `received_dBW` is set only when the caller gives Pt_dBW (and gains). `focusing_gain_db` and
    `total_db` are None if the ray didn't land. `doppler_hz` is NaN: a single-epoch ionosphere has no Doppler.

    `ray` is the full underlying RayPath, for anything not surfaced here directly.
    """

    landed: bool
    ground_range_km: float
    delay_ms: float
    doppler_hz: float
    focusing_gain_db: Optional[float]
    caustic: bool
    absorption_db: float
    ground_loss_db: float
    polarization_loss_db: float
    total_db: Optional[float]
    received_dBW: Optional[float]

    ray: RayPath

    def __str__(self) -> str:
        foc = "n/a"
        if self.focusing_gain_db is not None:
            foc = f"{self.focusing_gain_db:+.2f} dB" + (" (caustic cap)" if self.caustic else "")
        tot = f"{self.total_db:+.2f} dB" if self.total_db is not None else "n/a"
        rec = f"{self.received_dBW:.1f} dBW" if self.received_dBW is not None else "not computed (no Pt)"
        dop = "n/a (single epoch)" if self.doppler_hz != self.doppler_hz else f"{self.doppler_hz:+.2f} Hz"
        return (
            f"LinkBudget({self.ray.frequency_mhz:g} MHz, {self.ray.mode}-mode, "
            f"el={self.ray.elevation_deg:g}deg): {'landed' if self.landed else 'did not land'}, "
            f"{self.ground_range_km:.1f} km, delay {self.delay_ms:.3f} ms, Doppler {dop}\n"
            f"  focusing gain  : {foc}\n"
            f"  absorption     : -{self.absorption_db:.2f} dB\n"
            f"  ground loss    : -{self.ground_loss_db:.2f} dB\n"
            f"  polarization   : -{self.polarization_loss_db:.2f} dB\n"
            f"  total          : {tot}  (relative only -- see class docstring)\n"
            f"  received power : {rec}"
        )


# ---------------------------------------------------------------------------
# ScatterResult (deprecated Monte Carlo layer)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ScatterResult:
    """
    A Monte Carlo ensemble of scattered rays (mc_scattering.py's Langevin model of small-scale field-aligned
    irregularities) around one deterministic reference ray. Result of Link.scatter(), the deprecated Monte Carlo
    layer (decision D6); Link.channel() gives the HiCIRF statistics of a leg instead.

    `reference` is the unscattered ray. `rays` are the ensemble members that landed (n_rays were launched; rays
    that did not come down are excluded from `rays` and from every statistic below).

    `coherent_fraction` = exp(-mean(sigma_phi2)/2) is the fraction of signal power that stays in a coherent
    wavefront: 1.0 is undisturbed; well below that, the irregularities matter for anything phase-sensitive.
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
        """PHaRLAP-style range-height figure: the ensemble in white, the reference ray in red."""
        from plotting.figures import ray_slice_figure
        return ray_slice_figure([self.reference] + list(self.rays), highlight=self.reference, **kwargs)

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
# Link
# ---------------------------------------------------------------------------

class Link:
    """
    An HF transmitter: a fixed place and moment in time, ready to trace rays at whatever launch angles (and, per
    call, frequency or mode) you ask for, and to give their budget, their HiCIRF stochastic channel and their
    PHaRLAP-style figures.

    Construction does the expensive part once: an E-CHAIM + IRI-2020 electron-density patch and an IGRF field patch,
    centred on (lat_deg, lon_deg) and sized (max_path_km x max_nhops) to cover every ray you will trace.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Geodetic transmitter location (degrees).
    alt_m : float
        Transmitter altitude above the WGS84 ellipsoid (metres). Default 0.
    frequency_mhz : float
        Default carrier frequency (MHz); every method takes `frequency_mhz=` to override it per call (cheap: only a
        new Ray is built, the patch is reused).
    dt : datetime or None
        UTC date and time: solar and geomagnetic indices, season, local time. Default: now (UTC). Pass it explicitly
        for reproducible results.
    mode : "O", "X", +1 or -1
        Default magneto-ionic mode; override per call with `mode=`. Stored as "O" or "X".
    max_path_km, max_nhops : float, int
        The longest per-hop path and the most hops you intend to trace; they size the patch. A ray may use less;
        a longer one runs past the patch edge (density and field fall back to boundary values). Defaults 2000 km, 1.
        Note: E-CHAIM has been seen to crash building patches wider than about 60 deg (roughly max_path_km x
        max_nhops > 2500 km at 57 deg latitude).
    use_echaim : bool
        Prefer E-CHAIM over IRI-2020 where it is valid (AACGM latitude >= 45 deg). Default True.
    n_ang, dr_km, dang_deg : int, float, float
        Ionosphere query and grid resolution (geophysics/ionosphere.py). Defaults 15, 2.0, 1.0.
    igrf_n_ang, igrf_n_r : int
        IGRF query grid resolution (geophysics/igrf.py). Defaults 5, 5.
    irregularity : core.inputs.IrregularityInputs, optional
        The irregularity inputs of the HiCIRF channel when no irregularity_model / drift_model is given: xi, L0,
        p, the two axial ratios, the drift and its spread. Default: IrregularityInputs() (placeholders).
    xi_rms, L0_km, p_index, axial_ratio : float, optional
        Shortcuts that override single fields of `irregularity` (axial_ratio sets both axial ratios). They also
        define the spectrum of the deprecated scatter().
    irregularity_model, drift_model : environment.IrregularityModel / DriftModel, optional
        Position- and time-varying models (e.g. geophysics.wbmod.CrudeWBMOD, geophysics.plasma_drift.WBMODStyleDrift),
        queried per ionospheric sample as model.at(r_ecef, dt). When given they replace the corresponding inputs.
    coarsen_elevation_deg : float, optional
        Place a phase screen only every this many degrees of ray-direction change (HiCIRF paragraph 17 uses 5).
        Default None: one screen per ionospheric sample.
    absorption_model : str or callable(alt_km) -> nu
        Collision-frequency profile for absorption (raytracing/absorption.py). Default "default" (illustrative).
    verbose : bool
        Print progress (ionosphere and field build, scatter progress). Default True.
    """

    def __init__(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float = 0.0,
        *,
        frequency_mhz: float,
        dt: Optional[datetime] = None,
        mode: Mode = "O",
        max_path_km: float = 2000.0,
        max_nhops: int = 1,
        use_echaim: bool = True,
        n_ang: int = 15,
        dr_km: float = 2.0,
        dang_deg: float = 1.0,
        igrf_n_ang: int = 5,
        igrf_n_r: int = 5,
        irregularity: Optional[IrregularityInputs] = None,
        xi_rms: Optional[float] = None,
        L0_km: Optional[float] = None,
        p_index: Optional[float] = None,
        axial_ratio: Optional[float] = None,
        irregularity_model: Optional[environment.IrregularityModel] = None,
        drift_model: Optional[environment.DriftModel] = None,
        coarsen_elevation_deg: Optional[float] = None,
        absorption_model: Union[str, Callable] = "default",
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

        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.alt_m = float(alt_m)
        self.frequency_mhz = float(frequency_mhz)
        self.dt = dt if dt is not None else datetime.now(timezone.utc)
        self.mode = _normalize_mode(mode)[1]
        self.max_path_km = float(max_path_km)
        self.max_nhops = int(max_nhops)
        self.use_echaim = use_echaim
        self.absorption_model = absorption_model

        overrides = {}
        if xi_rms is not None:
            overrides["xi"] = float(xi_rms)
        if L0_km is not None:
            overrides["L0_m"] = float(L0_km) * 1e3
        if p_index is not None:
            overrides["p"] = float(p_index)
        if axial_ratio is not None:
            overrides.update(axial_ratio_B=float(axial_ratio), axial_ratio_sheet=float(axial_ratio))
        self.irregularity = dataclasses.replace(irregularity or IrregularityInputs(), **overrides)
        self.irregularity_model = irregularity_model
        self.drift_model = drift_model
        self.coarsen_elevation_deg = coarsen_elevation_deg
        self.verbose = verbose

        self.r0 = latlon_to_xyz_m(self.lat_deg, self.lon_deg, self.alt_m)
        self._up, self._north, self._east = _enu_frame(self.lat_deg, self.lon_deg)

        # A ground or near-ground transmitter starts inside the ionosphere grid, so build_ionosphere_for_ray centres
        # the patch on it and sizes it from the path budget alone: the direction passed ("up") does not matter.
        if self.verbose:
            print(f"[Link] building ionosphere + field patch for {lat_deg:.2f}N {lon_deg:.2f}E, "
                  f"{self.dt.isoformat()} ...")
        self._ionosphere: Ionosphere = build_ionosphere_for_ray(
            lat0_deg=self.lat_deg, lon0_deg=self.lon_deg, alt0_m=self.alt_m,
            direction=self._up, max_path_m=self.max_path_km * self.max_nhops * 1e3,
            dt=self.dt, n_ang=n_ang, use_echaim=self.use_echaim,
            dr_km=dr_km, dang_deg=dang_deg,
        )
        self._field: IgrfField = build_igrf_for_ionosphere(self._ionosphere, n_ang=igrf_n_ang, n_r=igrf_n_r)
        if self.verbose:
            self._ionosphere.grid_info()

        self._ray_cache: dict[tuple[float, int], Ray] = {}
        self._spectrum_cache: dict[float, IrregularitySpectrum] = {}

    # Scalar views of the irregularity inputs (also the spectrum of the deprecated scatter()).
    @property
    def xi_rms(self) -> float:
        return self.irregularity.xi

    @property
    def L0_km(self) -> float:
        return self.irregularity.L0_m / 1e3

    @property
    def p_index(self) -> float:
        return self.irregularity.p

    @property
    def axial_ratio(self) -> float:
        return self.irregularity.axial_ratio_B

    def __repr__(self) -> str:
        return (f"Link({self.lat_deg:.2f}, {self.lon_deg:.2f}, frequency_mhz={self.frequency_mhz:g}, "
                f"mode={self.mode!r}, dt={self.dt.isoformat()!r})")

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

    def _frequency(self, frequency_mhz: Optional[float]) -> float:
        freq = self.frequency_mhz if frequency_mhz is None else float(frequency_mhz)
        if freq <= 0.0:
            raise ValueError(f"frequency_mhz must be positive -- got {freq}")
        return freq

    def _resolve_ray(self, frequency_mhz: Optional[float], mode: Optional[Mode]) -> Ray:
        freq = self._frequency(frequency_mhz)
        mode_int, _ = _normalize_mode(self.mode if mode is None else mode)
        key = (freq, mode_int)
        if key not in self._ray_cache:
            self._ray_cache[key] = Ray(omega=2.0 * np.pi * freq * 1e6, MODE=mode_int,
                                       electron_density=self._ionosphere.electron_density_xyz_m,
                                       B_field=self._field.B_vec_xyz_m)
        return self._ray_cache[key]

    def _resolve_spectrum(self, frequency_mhz: Optional[float]) -> IrregularitySpectrum:
        freq = self._frequency(frequency_mhz)
        if freq not in self._spectrum_cache:
            spec = IrregularitySpectrum(xi_rms=self.xi_rms, L0_m=self.L0_km * 1e3, p_index=self.p_index,
                                        axial_ratio=self.axial_ratio)
            spec.tabulate(k0=2.0 * np.pi * freq * 1e6 / C)
            self._spectrum_cache[freq] = spec
        return self._spectrum_cache[freq]

    def _path_budget_m(self, max_path_km: Optional[float]) -> float:
        return (self.max_path_km if max_path_km is None else float(max_path_km)) * 1e3

    def _drift_ecef(self) -> np.ndarray:
        ve, vn = self.irregularity.drift_east_north_mps
        return ve * self._east + vn * self._north

    def _screens(self, samples: RaySamples, ray: Ray):
        """Phase screens of a traced leg (stochastic.leg_screens), from this Link's irregularity inputs or models."""
        irr = self.irregularity
        return screens_for_leg(samples, ray, xi=irr.xi, L0=irr.L0_m, p=irr.p, a=irr.axial_ratio_B,
                               b=irr.axial_ratio_sheet, v_d=self._drift_ecef(), sigma_v_frac=irr.sigma_v_frac,
                               irregularity_model=self.irregularity_model, drift_model=self.drift_model, t=self.dt,
                               coarsen_elevation_deg=self.coarsen_elevation_deg)

    def _leg_screens_and_channel(self, ray: RayPath, options: Optional[ChannelOptions] = None):
        """(ScreenSet, ChannelParams) of a traced ray, from its own samples."""
        if ray.samples is None:
            raise ValueError("this RayPath has no per-step samples (only rays from Link.trace() and the sweep_* "
                             "methods have them)")
        Pg = np.asarray(ray.samples.Pg, dtype=float)
        if not np.all(np.isfinite(Pg)) or not np.any(np.diff(Pg) > 0.0):
            raise ValueError("the channel needs the group path: trace the ray with group_path=True")
        scr = self._screens(ray.samples, self._resolve_ray(ray.frequency_mhz, ray.mode))
        return scr, channel_parameters(scr, ray.frequency_mhz * 1e6, options or ChannelOptions())

    @staticmethod
    def _flatten_rays(rays, highlight):
        """RayPaths from RayPath / ScatterResult / list arguments; a ScatterResult's reference is the highlight
        unless one is given."""
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
        return flat, highlight

    # ------------------------------------------------------------------
    # Rays
    # ------------------------------------------------------------------

    def trace(
        self,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        *,
        nhops: int = 1,
        max_path_km: Optional[float] = None,
        frequency_mhz: Optional[float] = None,
        mode: Optional[Mode] = None,
        absorption: bool = True,
        group_path: bool = True,
        phase_variance: bool = False,
        ds_step_km: float = 2.0,
    ) -> RayPath:
        """
        Trace one ray (fixed-step RK4 with step rejection, raytracing.batch.BatchTracer).

        elevation_deg, azimuth_deg : launch angles above the local horizontal and clockwise from true north, degrees.
        nhops, max_path_km : number of hops (specular ground bounces on the WGS84 ellipsoid) and the path budget per
            hop (default: the Link's max_path_km).
        frequency_mhz, mode : override this Link's defaults for this ray only.
        absorption : non-deviative absorption (raytracing/absorption.py), on by default.
        group_path : the exact group path P' (spec §2.7), on by default; Link.channel() needs it.
        phase_variance : also fill RayPath.sigma_phi2, the phase variance of the leg's phase screens (the same as
            Link.channel(ray).sigma_phi2).
        ds_step_km : integration step.
        """
        ray = self._resolve_ray(frequency_mhz, mode)
        tracer = BatchTracer(ray, ds_m=ds_step_km * 1e3, absorption_model=self.absorption_model)
        u0 = self._direction(elevation_deg, azimuth_deg)
        res = tracer.trace(self.r0, u0[None], self._path_budget_m(max_path_km), nhops=nhops,
                           want_group=group_path, want_absorption=absorption, record_states=True)
        samples = samples_from_record(tracer, res.records[0], transport=True)
        sigma_phi2 = float(np.sum(self._screens(samples, ray).sigma2)) if phase_variance else None
        return _ray_path_from_batch(res, samples, frequency_mhz=self._frequency(frequency_mhz),
                                    mode=_normalize_mode(self.mode if mode is None else mode)[1],
                                    elevation_deg=elevation_deg, azimuth_deg=azimuth_deg, sigma_phi2=sigma_phi2)

    def sweep_elevation(self, elevations_deg: Sequence[float], azimuth_deg: float = 0.0,
                        **trace_kwargs) -> list[RayPath]:
        """trace() at each of several elevations (a PHaRLAP fan). **trace_kwargs go to trace()."""
        return [self.trace(el, azimuth_deg, **trace_kwargs) for el in elevations_deg]

    def sweep_azimuth(self, elevation_deg: float, azimuths_deg: Sequence[float], **trace_kwargs) -> list[RayPath]:
        """trace() at each of several azimuths, fixed elevation. **trace_kwargs go to trace()."""
        return [self.trace(elevation_deg, az, **trace_kwargs) for az in azimuths_deg]

    def sweep_frequency(self, elevation_deg: float, frequencies_mhz: Sequence[float], azimuth_deg: float = 0.0,
                        **trace_kwargs) -> list[RayPath]:
        """trace() at each of several frequencies, fixed launch geometry (only a new Ray per frequency; the patch
        is reused). **trace_kwargs go to trace()."""
        return [self.trace(elevation_deg, azimuth_deg, frequency_mhz=f, **trace_kwargs) for f in frequencies_mhz]

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def link_budget(
        self,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        *,
        nhops: int = 1,
        max_path_km: Optional[float] = None,
        frequency_mhz: Optional[float] = None,
        mode: Optional[Mode] = None,
        Pt_dBW: Optional[float] = None,
        Gt_dB: float = 0.0,
        Gr_dB: float = 0.0,
    ) -> LinkBudget:
        """
        One-way dB budget of one leg (spec §6.2-§6.4): flux-tube focusing with the caustic cap (landing mini-fan,
        propagation/legs.py), absorption, ground losses and the 3.01 dB polarization loss. No coherence term. Pass
        Pt_dBW (and Gt_dB, Gr_dB) for an absolute received power.
        """
        ray = self._resolve_ray(frequency_mhz, mode)
        tracer = BatchTracer(ray, absorption_model=self.absorption_model)
        bud = one_way_budget(tracer, self.r0, self._up, self._north, self._east, elevation_deg, azimuth_deg,
                             self._path_budget_m(max_path_km), nhops=nhops, Pt_dBW=Pt_dBW, Gt_dB=Gt_dB,
                             Gr_dB=Gr_dB)
        ray_path = self.trace(elevation_deg, azimuth_deg, max_path_km=max_path_km, nhops=nhops,
                              frequency_mhz=frequency_mhz, mode=mode)
        total = None
        if bud.landed and bud.focusing_gain_db is not None:
            total = bud.focusing_gain_db - bud.total_loss_db
        return LinkBudget(
            landed=bud.landed, ground_range_km=bud.ground_range_m / 1e3, delay_ms=bud.delay_s * 1e3,
            doppler_hz=bud.doppler_hz, focusing_gain_db=bud.focusing_gain_db, caustic=bud.caustic,
            absorption_db=bud.absorption_db, ground_loss_db=bud.ground_loss_db,
            polarization_loss_db=bud.polarization_loss_db, total_db=total, received_dBW=bud.received_dBW,
            ray=ray_path,
        )

    # ------------------------------------------------------------------
    # HiCIRF channel
    # ------------------------------------------------------------------

    def channel(self, ray: RayPath, *, options: Optional[ChannelOptions] = None) -> ChannelParams:
        """
        The HiCIRF stochastic channel of a traced ray's leg (spec §10, Algorithm 5): its phase screens from this
        Link's irregularity inputs or models (stochastic/leg_screens.py), then sigma_phi^2, the coherence ellipse
        (l_x, l_y), tau_o, the apparent velocity, C_xt, C_yt, the mean excess delay, omega_coh, the wander alpha,
        S4 and the Rician split (A_c, A_d). Uses the ray's own samples: the channel is that of this very ray.
        """
        return self._leg_screens_and_channel(ray, options)[1]

    def leg_channel(self, elevation_deg: float, azimuth_deg: float = 0.0, *, nhops: int = 1,
                    max_path_km: Optional[float] = None, frequency_mhz: Optional[float] = None,
                    mode: Optional[Mode] = None, options: Optional[ChannelOptions] = None) -> ChannelParams:
        """channel() of the ray traced at these launch angles (a shortcut for channel(trace(...)))."""
        ray = self.trace(elevation_deg, azimuth_deg, nhops=nhops, max_path_km=max_path_km,
                         frequency_mhz=frequency_mhz, mode=mode, absorption=False)
        return self.channel(ray, options=options)

    # ------------------------------------------------------------------
    # Figures (plotting/)
    # ------------------------------------------------------------------

    def plot(self, *rays, highlight: Optional[RayPath] = None, title: Optional[str] = None, hicirf: bool = True,
             background: bool = True, options: Optional[ChannelOptions] = None, **kwargs):
        """
        PHaRLAP-style range-height figure (matplotlib Figure; plotting.figures.ray_slice_figure): the plasma
        frequency of this Link's ionosphere along the bearing of the highlighted (else first) ray, the rays over it
        in PHaRLAP's curved-Earth frame, and a UTC / frequency / bearing note.

        rays : RayPath, ScatterResult, or lists of them. A single ray is highlighted (red, with its Tx elevation);
            with several, pass highlight= to pick one, otherwise they are drawn as a plain PHaRLAP fan.
        hicirf : for the highlighted ray, draw its phase screens and its channel (channel(); options= are its
            ChannelOptions). Skipped for rays without samples (Monte Carlo rays).
        background : False leaves the slice empty.
        Other kwargs go to ray_slice_figure: start_range, end_range, end_height, cmap ('MuellerMap', 'parula', ...),
        startup=False for MATLAB's factory style, fig_width_px, ...
        """
        from plotting.figures import ray_slice_figure

        flat, highlight = self._flatten_rays(rays, highlight)
        if not flat:
            raise ValueError("nothing to plot: pass at least one RayPath")
        if highlight is None and len(flat) == 1:
            highlight = flat[0]
        hic = None
        if hicirf and highlight is not None and highlight.samples is not None:
            try:
                scr, P = self._leg_screens_and_channel(highlight, options)
                hic = dict(samples=highlight.samples, screens=scr, channel=P, absorption_db=highlight.absorption_db)
            except ValueError as exc:            # e.g. a ray that never enters the ionosphere has no screens
                hic = dict(error=str(exc))
        return ray_slice_figure(
            flat, highlight=highlight, origin_ecef=self.r0, origin_lat=self.lat_deg, origin_lon=self.lon_deg,
            density_geodetic=self._ionosphere.electron_density_geodetic if background else None,
            utc=self.dt, hicirf=hic, title=title, **kwargs)

    def plot_3d(self, *rays, title: Optional[str] = None, **kwargs):
        """PHaRLAP's 3-D latitude/longitude/height figure of the rays (O-mode blue dots, X-mode red dots;
        plotting.figures.rays_3d_figure). kwargs: zlim, startup, figsize, ..."""
        from plotting.figures import rays_3d_figure

        flat, _ = self._flatten_rays(rays, None)
        return rays_3d_figure(flat, title=title, **kwargs)

    def plot_scattering_function(self, ray: RayPath, *, options: Optional[ChannelOptions] = None, **kwargs):
        """HiCIRF delay-Doppler scattering function of a traced ray's leg (from channel(ray);
        plotting.figures.scattering_function_figure). kwargs: floor_db, cmap, startup, title, grid."""
        from plotting.figures import scattering_function_figure

        kwargs.setdefault("title", f"HiCIRF Scattering Function\n{ray.frequency_mhz:g} MHz {ray.mode}-mode, "
                                   f"elevation {ray.elevation_deg:g}°, bearing {ray.azimuth_deg:g}°")
        return scattering_function_figure(self.channel(ray, options=options), **kwargs)

    def plot_html(self, *rays, highlight: Optional[RayPath] = None, title: Optional[str] = None, **kwargs):
        """The interactive 3-D Plotly scene of the rays (plotting.plotly_3d.plot_ray); .write_html(path) saves it."""
        from plotting.plotly_3d import plot_ray

        flat, highlight = self._flatten_rays(rays, highlight)
        return plot_ray(flat, highlight=highlight, title=title, **kwargs)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    @property
    def ionosphere(self) -> Ionosphere:
        """The underlying E-CHAIM + IRI-2020 Ionosphere patch (geophysics/ionosphere.py)."""
        return self._ionosphere

    @property
    def magnetic_field(self) -> IgrfField:
        """The underlying IGRF field patch (geophysics/igrf.py)."""
        return self._field

    def electron_density_xyz_m(self, x, y, z):
        """This Link's background electron density [m^-3] at ECEF metres, e.g. for
        geophysics.wbmod.CrudeWBMOD(ne_column=link.electron_density_xyz_m), which integrates Ne^2 for eq. (5)."""
        return self._ionosphere.electron_density_xyz_m(x, y, z)

    def grid_info(self) -> None:
        """Print a summary of the ionosphere patch this Link built (Ionosphere.grid_info)."""
        self._ionosphere.grid_info()

    # ------------------------------------------------------------------
    # Deprecated: Monte Carlo layer (decision D6)
    # ------------------------------------------------------------------

    def scatter(
        self,
        elevation_deg: float,
        azimuth_deg: float = 0.0,
        *,
        n_rays: int = 200,
        nhops: int = 1,
        max_path_km: Optional[float] = None,
        frequency_mhz: Optional[float] = None,
        mode: Optional[Mode] = None,
        ds_step_km: float = 2.0,
        seed: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> ScatterResult:
        """
        Deprecated (decision D6): a Monte Carlo ensemble of n_rays rays with Langevin kicks from the irregularity
        spectrum (raytracing/mc_scattering.py) around the deterministic ray, which is `.reference`. Use channel()
        for the stochastic channel of a leg.

        seed : RNG seed for a reproducible ensemble; None is fresh randomness.
        verbose : print progress every 25 rays; defaults to the Link's verbose.
        """
        warnings.warn("Link.scatter() runs the deprecated Monte Carlo layer (decision D6). Use Link.channel() for "
                      "the stochastic channel of a leg.", DeprecationWarning, stacklevel=2)
        ray = self._resolve_ray(frequency_mhz, mode)
        mc = MCScatteringTracer(ray, self._resolve_spectrum(frequency_mhz), ds_step_m=ds_step_km * 1e3, seed=seed,
                                record_every=5)
        u0 = self._direction(elevation_deg, azimuth_deg)
        budget_m = self._path_budget_m(max_path_km)
        verbose = self.verbose if verbose is None else verbose

        det = mc.trace(self.r0, u0, budget_m, scatter=False, nhops=nhops)
        results = mc.ensemble(self.r0, u0, n_rays=n_rays, max_path_m=budget_m, nhops=nhops,
                              verbose_every=(25 if verbose else 0))
        summ = mc.summarize(results, self.r0)
        mode_label = _normalize_mode(self.mode if mode is None else mode)[1]
        freq = self._frequency(frequency_mhz)

        def _rp(res) -> RayPath:
            return RayPath(
                frequency_mhz=freq, mode=mode_label, elevation_deg=elevation_deg, azimuth_deg=azimuth_deg,
                landed=res.landed, escaped=res.escaped, hops_completed=res.hops_completed,
                path=res.path, arc_length=res.s_path, bounce_points=res.bounce_points,
                launch_point=res.path[0], final_point=res.r_final, final_direction=res.u_final,
                ground_range_km=res.ground_range_total_m / 1e3, arc_length_km=res.s_total / 1e3,
                phase_path_km=res.phase_path_total / 1e3, group_path_km=res.group_path_total / 1e3,
                group_delay_ms=1e3 * res.group_path_total / C, absorption_db=res.absorption_db_total,
                arrival_elevation_deg=(_arrival_elevation_deg(res.r_final, res.u_final) if res.landed else None),
                sigma_phi2=(res.sigma_phi2 if res.sigma_phi2 else None),
            )

        reference = _rp(det)
        nan = float("nan")
        if summ.get("n_landed", 0) == 0:
            return ScatterResult(
                reference=reference, rays=[], n_rays=n_rays, n_landed=0, range_mean_km=nan, range_std_km=nan,
                cross_range_std_km=nan, elevation_mean_deg=nan, elevation_std_deg=nan, group_path_mean_km=nan,
                group_delay_spread_ms=nan, absorption_mean_db=nan, absorption_std_db=nan, sigma_phi2_mean=nan,
                coherent_fraction=nan,
            )
        return ScatterResult(
            reference=reference, rays=[_rp(r) for r in results if r.landed], n_rays=n_rays,
            n_landed=summ["n_landed"], range_mean_km=summ["range_mean_km"], range_std_km=summ["range_std_km"],
            cross_range_std_km=summ["cross_std_km"], elevation_mean_deg=summ["elev_mean_deg"],
            elevation_std_deg=summ["elev_std_deg"], group_path_mean_km=summ["group_path_mean_km"],
            group_delay_spread_ms=1e3 * summ["group_path_std_km"] * 1e3 / C,
            absorption_mean_db=summ["absorption_db_mean"], absorption_std_db=summ["absorption_db_std"],
            sigma_phi2_mean=summ["sigma_phi2_mean"], coherent_fraction=summ["coherent_fraction"],
        )
