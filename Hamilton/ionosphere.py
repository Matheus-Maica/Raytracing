"""
3D sectioned WGS84-ellipsoid ionosphere, background Ne from E-CHAIM and IRI-2020.

Electron density is sourced from two climatological models on the same
grid, combined per-point:

  * E-CHAIM (Empirical Canadian High Arctic Ionospheric Model) -- via the
    ECHAIM/ package vendored in this project (ECHAIM/echaim.py, a ctypes
    binding to the CHAIN project's C library). Assimilates real Arctic
    ionosonde/GNSS data into its fit, so it's the more accurate model
    wherever it's valid: AACGM geomagnetic latitude >= 45 degrees.
  * IRI-2020 -- via `iricore`, as before. Used everywhere E-CHAIM isn't
    valid (E-CHAIM returns NaN below 45 degrees geomagnetic -- enforced
    inside its own C code, see ECHAIM/echaim.py), i.e. mid- and low-
    latitudes, and as the sole source if `use_echaim=False` or the
    ECHAIM/ package fails to load.

The combination is a hard per-point selection (E-CHAIM's value wherever
finite, else IRI's), not a smoothed blend -- E-CHAIM's own documentation
already treats its 45-50 degree band as a deliberate transition region
for exactly this kind of merge with a mid-latitude model, so the two
models' native validity domains meet without an extra blending step.
Since both models are evaluated on the same coarse angular query grid,
any residual step at the 45 degree line is smoothed out by the trilinear
upsample onto the fine ray-tracer grid below, the same way IRI's own
n_ang x n_ang -> fine-grid interpolation already smooths IRI internally.

GRID
----
The grid covers a rectangular patch in geodetic coordinates:

    h    ∈ [0, h_max]                    (height above the WGS84 ellipsoid, km)
    lat  ∈ [lat_c − dlat/2, lat_c + dlat/2]   (geodetic latitude, degrees)
    lon  ∈ [lon_c − dlon/2, lon_c + dlon/2]   (east longitude, degrees, unwrapped)

Uniform spacing dr (km) and dang_deg (degrees) throughout.
Points outside the patch return Ne = 0.

HOW IRI AND E-CHAIM ARE CALLED
-------------------------------
Both are called *once* at construction time with the same batched
(lat, lon) grid.  Per-call overhead is thus paid only for the
n_ang × n_ang angular query points rather than for every Ne(x,y,z)
lookup during ray tracing.  After construction each Ne query is a
~0.2 µs RegularGridInterpolator call plus one WGS84 inverse-transform
(geometry.xyz_m_to_latlon).

The query angular grid (n_ang × n_ang) is independent of the ray-tracer
grid (dang_deg spacing).  Both models vary slowly horizontally (horizontal
scale ~hundreds of km), so n_ang = 15 (≈ 1–2° spacing for a typical
patch) gives <1% interpolation error and builds in a few seconds.

COORDINATE CONVENTION
----------------------
    h    : height above the WGS84 ellipsoid (km), h ∈ [0, h_max]
    lat  : geodetic latitude (degrees) — the angle the ellipsoid normal
           makes with the equatorial plane, what IRI/GPS/maps mean by
           "latitude" (NOT geocentric)
    lon  : east longitude (degrees), stored in unwrapped range
           [lon_c − dlon/2, lon_c + dlon/2]

All horizontal lengths in kilometres throughout; the WGS84 forward/inverse
transforms themselves (geometry.py) work in metres and are wrapped here.

PUBLIC API
----------
    Ionosphere(center_lat, center_lon,
               dtheta_deg, dphi_deg, h_max,
               dr, dang_deg,
               dt, n_ang, use_echaim)

    .electron_density_geodetic(lat, lon, h)  → Ne  (degrees / km)
    .electron_density_xyz(x, y, z)         → Ne  (Cartesian, km)
    .electron_density_xyz_m(x, y, z)       → Ne  (Cartesian, metres)
    .__call__(x, y, z)                     → Ne  (Cartesian, km)
    .grid_info()
    .plot(alt_km, slice_lat, slice_lon)    → matplotlib Figure
"""

import numpy as np
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

import iricore
import geometry

# IRI altitude grid (fixed by IRI's native resolution)
_IRI_ALT_MIN  = 60.0    # km  — IRI D-layer base
_IRI_ALT_MAX  = 800.0   # km  — grid ceiling; Ne = 0 above this
_IRI_ALT_STEP = 5.0     # km


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Ionosphere:
    """
    E-CHAIM + IRI-2020 ionosphere on a uniform geodetic grid covering only
    a rectangular patch (in latitude × longitude) above the WGS84 ellipsoid.

    Parameters
    ----------
    Section definition
    ~~~~~~~~~~~~~~~~~~
    center_lat : float
        Geodetic latitude of the patch centre (degrees, −90 … +90).
    center_lon : float
        Geographic longitude of the patch centre (degrees, −180 … +360).
    dtheta_deg : float
        Full angular extent of the patch in latitude (degrees).
        The patch spans [lat_c − dtheta/2, lat_c + dtheta/2], clamped to
        [-90, 90]. (Name kept from the pre-ellipsoid version for call-site
        stability; it's a direct latitude extent now, not a colatitude one
        — numerically identical either way for a symmetric interval.)
    dphi_deg : float
        Full angular extent of the patch in longitude (degrees).
        Values > 360 are clamped to 360.

    Radial extent
    ~~~~~~~~~~~~~
    h_max : float
        Maximum height above the WGS84 ellipsoid (km).  Clamped to 800 km
        (IRI upper limit).  Default 800 km.

    Grid spacing (uniform throughout)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dr : float
        Height cell size (km).  Default 2 km.
    dang_deg : float
        Angular cell size for both latitude and longitude (degrees).  Default 1°.

    IRI parameters
    ~~~~~~~~~~~~~~
    dt : datetime
        UTC date and time for the IRI-2020 query.
        Determines solar activity indices, season, and time of day
        (IRI computes local solar time per longitude internally).
        Default: datetime(2000, 1, 1, 12, 0, 0).
    n_ang : int
        Number of latitude (and longitude) points in the IRI query
        grid.  IRI is called once with n_ang × n_ang batched points;
        the results are then trilinearly interpolated onto the finer
        dang_deg ray-tracer grid.
        • n_ang = 10 →  ~0.9 s,  ~2–3° angular resolution
        • n_ang = 15 →  ~2.0 s,  ~1–2° resolution  (default, recommended)
        • n_ang = 20 →  ~3.5 s,  ~1°   resolution
        IRI varies slowly horizontally so n_ang = 15 is sufficient for
        most applications.
    use_echaim : bool
        Prefer E-CHAIM over IRI wherever E-CHAIM is valid (AACGM
        geomagnetic latitude >= 45 degrees) -- see the module docstring
        for how the two are combined. Default True. Set False to force
        IRI-only behaviour (e.g. for reproducing pre-E-CHAIM results, or
        if the ECHAIM/ package isn't set up -- see ECHAIM/echaim.py).
        If True but ECHAIM/ fails to import or initialize, falls back to
        IRI-only automatically, with a one-time printed warning.
    """

    def __init__(self,
                 # --- section geometry ---
                 center_lat=0.0,
                 center_lon=0.0,
                 dtheta_deg=20.0,
                 dphi_deg=20.0,
                 # --- radial extent ---
                 h_max=800.0,
                 # --- grid spacing ---
                 dr=2.0,
                 dang_deg=1.0,
                 # --- IRI parameters ---
                 dt=None,
                 n_ang=15,
                 # --- background model selection ---
                 use_echaim=True):

        # ---- IRI datetime ----
        self.dt    = dt if dt is not None else datetime(2000, 1, 1, 12, 0, 0)
        self.n_ang = int(n_ang)
        self.use_echaim = bool(use_echaim)

        # ---- Section centre ----
        self.center_lat = float(center_lat)
        self.center_lon = float(center_lon)

        # ---- Angular half-extents (degrees) ----
        half_lat = float(dtheta_deg) / 2.0
        half_lon = min(float(dphi_deg), 360.0) / 2.0

        self.lat_min = float(np.clip(self.center_lat - half_lat, -90.0, 90.0))
        self.lat_max = float(np.clip(self.center_lat + half_lat, -90.0, 90.0))

        # lon stored as an unwrapped linear range centred on center_lon
        lon_c = self.center_lon % 360.0
        self.lon_min = lon_c - half_lon
        self.lon_max = lon_c + half_lon

        # ---- Height extent (km above the WGS84 ellipsoid) ----
        self.h_max = float(min(h_max, _IRI_ALT_MAX))
        self.h_min = 0.0

        # ---- Grid spacing ----
        self.dr       = float(dr)
        self.dang_deg = float(dang_deg)

        # ---- Build uniform axes ----
        self.h   = self._linspace_axis(self.h_min,   self.h_max,   self.dr)
        self.lat = self._linspace_axis(self.lat_min, self.lat_max, self.dang_deg)
        self.lon = self._linspace_axis(self.lon_min, self.lon_max, self.dang_deg)

        self.Nh   = len(self.h)
        self.Nlat = len(self.lat)
        self.Nlon = len(self.lon)

        # ---- Density field and interpolator ----
        self.density = self._build_density()
        self._interp = self._make_interpolator()

    # ----------------------------------------------------------------
    # Axis helper
    # ----------------------------------------------------------------

    @staticmethod
    def _linspace_axis(lo, hi, step):
        """Uniform axis lo…hi with spacing ≈ step, always including both endpoints."""
        N = max(2, int(round((hi - lo) / step)) + 1)
        return np.linspace(lo, hi, N)

    # ----------------------------------------------------------------
    # IRI density build
    # ----------------------------------------------------------------

    def _build_density(self) -> np.ndarray:
        """
        Query E-CHAIM and IRI-2020 on the same coarse (n_ang × n_ang)
        angular grid over the patch, combine them per-point (E-CHAIM
        wherever valid, else IRI -- see module docstring), then
        interpolate onto the fine (Nlat × Nlon) ray-tracer grid and the
        height axis.

        Returns
        -------
        density : ndarray, shape (Nh, Nlat, Nlon)
            Electron density in m⁻³.
        """
        n = self.n_ang

        # ---- angular query grid (degrees, native IRI convention) ----
        lat_iri = np.linspace(self.lat_min, self.lat_max, n)    # (n,) increasing
        lon_iri = np.linspace(self.lon_min, self.lon_max, n)    # (n,) increasing, unwrapped
                                                                  # (IRI accepts lon outside 0-360)

        # IRI expects flat paired (lat, lon) arrays of equal length.
        # Build all (n × n) combinations: lats_flat[i*n+j] <-> (lat_iri[i], lon_iri[j]).
        lats_flat = np.repeat(lat_iri, n)    # (n*n,)
        lons_flat = np.tile(lon_iri, n)      # (n*n,)

        alt_1d = np.arange(_IRI_ALT_MIN, self.h_max + _IRI_ALT_STEP / 2,
                            _IRI_ALT_STEP, dtype=float)
        Nalt_iri = len(alt_1d)

        print(f"  Querying IRI-2020 ({n}×{n}={n*n} pts × {Nalt_iri} alt levels) …",
              end=" ", flush=True)

        import time
        t0  = time.perf_counter()
        res = iricore.iri(
            self.dt,
            [_IRI_ALT_MIN, float(self.h_max), _IRI_ALT_STEP],
            lats_flat,
            lons_flat,
        )
        elapsed = time.perf_counter() - t0

        print(f"done ({elapsed:.1f} s)")

        # ---- Reshape IRI output ----
        # res.edens: (n*n, Nalt_iri)
        edens_iri = np.nan_to_num(
            res.edens.reshape(n, n, Nalt_iri), nan=0.0
        )
        edens_iri = np.clip(edens_iri, 0.0, None)  # (n, n, Nalt_iri)

        # ---- Combine with E-CHAIM (same grid, same altitudes) ----
        edens_bg = edens_iri
        if self.use_echaim:
            edens_echaim = self._echaim_density_grid(lats_flat, lons_flat, alt_1d, n, Nalt_iri)
            if edens_echaim is not None:
                valid = np.isfinite(edens_echaim)
                n_valid = int(valid.any(axis=2).sum())
                print(f"  E-CHAIM valid (≥45° geomagnetic) at {n_valid}/{n*n} query points")
                edens_bg = np.where(valid, edens_echaim, edens_iri)

        # ---- Interpolate the combined grid onto the ray-tracer (lat, lon, h) axes ----
        # Add zero-density levels below IRI's 60 km floor so the interpolator
        # can handle launch points near the surface.
        h_full     = np.concatenate([[0.0, _IRI_ALT_MIN - 0.1], alt_1d])
        zero_slab  = np.zeros((n, n, 2))
        edens_full = np.concatenate([zero_slab, edens_bg], axis=2)
        # shape: (n, n, Nalt_iri+2)

        # Build the intermediate interpolator on (lat_iri, lon_iri, h_full)
        # (all three axes increasing).
        interp_bg = RegularGridInterpolator(
            (lat_iri, lon_iri, h_full),
            edens_full,               # (n, n, Nalt_iri+2)
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        # ---- Sample onto the final (h, lat, lon) ray-tracer grid ----
        Hg, LATg, LONg = np.meshgrid(self.h, self.lat, self.lon, indexing="ij")
        # shape of each: (Nh, Nlat, Nlon)

        pts = np.stack([LATg.ravel(), LONg.ravel(), Hg.ravel()], axis=-1)
        density = interp_bg(pts).reshape(self.Nh, self.Nlat, self.Nlon)

        return np.clip(density, 0.0, None)

    def _echaim_density_grid(self, lats_flat, lons_flat, alt_1d, n, Nalt_iri):
        """
        E-CHAIM electron density on the same (n*n, Nalt_iri) query grid as
        the IRI call above, reshaped to (n, n, Nalt_iri). Returns None (and
        prints a one-time warning) if the ECHAIM/ package can't be loaded,
        so callers fall back to IRI-only automatically.
        """
        try:
            import ECHAIM
        except Exception as exc:
            print(f"  E-CHAIM unavailable ({exc}); falling back to IRI-only. "
                  f"Set use_echaim=False to silence this, or see ECHAIM/echaim.py.")
            self.use_echaim = False
            return None

        print(f"  Querying E-CHAIM ({n}×{n}={n*n} pts × {Nalt_iri} alt levels) …",
              end=" ", flush=True)
        import time
        t0 = time.perf_counter()
        edens = ECHAIM.density_profile(lats_flat, lons_flat, self.dt, alt_1d)
        print(f"done ({time.perf_counter() - t0:.1f} s)")

        return edens.reshape(n, n, Nalt_iri)

    # ----------------------------------------------------------------
    # Interpolator (over the final ray-tracer grid)
    # ----------------------------------------------------------------

    def _make_interpolator(self) -> RegularGridInterpolator:
        """RegularGridInterpolator over (h, lat, lon) on the ray-tracer grid."""
        return RegularGridInterpolator(
            (self.h, self.lat, self.lon),
            self.density,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def grid_info(self):
        """Print a human-readable summary of the grid."""
        arc_km = (geometry.R_MEAN / 1e3) * np.deg2rad(self.dang_deg)

        model_str = "E-CHAIM (≥45° geomag.) + IRI-2020" if self.use_echaim else "IRI-2020 only"
        print("=" * 60)
        print(f"  Model       : {model_str}  ({self.dt.strftime('%Y-%m-%d %H:%M')} UTC)")
        print(f"  Query grid  : {self.n_ang}×{self.n_ang} angular query points")
        print(f"  Centre      : lat={self.center_lat:.2f}°  lon={self.center_lon:.2f}°")
        print(f"  lat extent  : {self.lat_min:.2f}°N … {self.lat_max:.2f}°N  "
              f"(Δlat = {self.lat_max - self.lat_min:.2f}°)")
        print(f"  lon extent  : {self.lon_min:.2f}°E … {self.lon_max:.2f}°E  "
              f"(Δlon = {self.lon_max - self.lon_min:.2f}°)")
        print(f"  Altitude    : 0 … {self.h_max:.0f} km  (above WGS84 ellipsoid)")
        print(f"  Grid        : {self.Nh} × {self.Nlat} × {self.Nlon}  "
              f"= {self.Nh * self.Nlat * self.Nlon:,} voxels")
        print(f"  Spacing     : dr = {self.dr:.2f} km   "
              f"dang = {self.dang_deg:.3f}°  ({arc_km:.2f} km arc)")
        print("=" * 60)

    def plot(self, alt_km=None, slice_lat=None, slice_lon=None):
        """
        Three-panel figure: horizontal map, meridional slice, zonal slice.

        Parameters
        ----------
        alt_km    : altitude (km) for the horizontal map.  Default ~300 km.
        slice_lat : latitude  (°N) for the zonal     slice.  Default centre.
        slice_lon : longitude (°E) for the meridional slice.  Default centre.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # ---- Display axes ----
        lat_deg   = self.lat
        lon_deg   = self.lon
        alt_km_ax = self.h

        lat_min_d, lat_max_d = lat_deg.min(), lat_deg.max()
        lon_min_d, lon_max_d = lon_deg.min(), lon_deg.max()

        # ---- Snap slice positions ----
        alt_target = 300.0 if alt_km is None else float(alt_km)
        alt_target = float(np.clip(alt_target, alt_km_ax[0], alt_km_ax[-1]))
        ir = int(np.argmin(np.abs(alt_km_ax - alt_target)))
        actual_alt = alt_km_ax[ir]

        lat_target = self.center_lat if slice_lat is None else float(slice_lat)
        lat_target = float(np.clip(lat_target, lat_min_d, lat_max_d))
        ith = int(np.argmin(np.abs(lat_deg - lat_target)))
        actual_lat = lat_deg[ith]

        lon_target = self.center_lon if slice_lon is None else float(slice_lon)
        lon_target_mapped = self._map_lon(lon_target)
        lon_target_mapped = float(np.clip(lon_target_mapped, lon_deg.min(), lon_deg.max()))
        iph = int(np.argmin(np.abs(lon_deg - lon_target_mapped)))
        actual_lon = lon_deg[iph]

        # ---- Grid-line stride (~10 lines per axis) ----
        def _stride(n): return max(1, n // 10)
        sr, sth, sph = _stride(self.Nh), _stride(self.Nlat), _stride(self.Nlon)

        vmin = float(self.density.min())
        vmax = float(self.density.max())
        cmap = "plasma"
        gl_kw = dict(color="white", lw=0.5, alpha=0.55)

        # ---- Layout ----
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"IRI-2020 Ionosphere  |  centre ({self.center_lat:.1f}°N, "
            f"{self.center_lon:.1f}°E)  |  "
            f"{self.dt.strftime('%Y-%m-%d %H:%M')} UTC",
            fontsize=12, fontweight="bold", y=0.98,
        )
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0],
                      hspace=0.38, wspace=0.32)
        ax_map = fig.add_subplot(gs[0, :])
        ax_mer = fig.add_subplot(gs[1, 0])
        ax_zon = fig.add_subplot(gs[1, 1])

        # ---- Panel 1: horizontal map ----
        Ne_map = self.density[ir, :, :]        # (Nlat, Nlon)
        pm1 = ax_map.pcolormesh(lon_deg, lat_deg, Ne_map,
                                vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
        fig.colorbar(pm1, ax=ax_map, label="Ne  (m⁻³)", pad=0.02)
        for i in range(0, self.Nlat, sth): ax_map.axhline(lat_deg[i], **gl_kw)
        for j in range(0, self.Nlon, sph): ax_map.axvline(lon_deg[j], **gl_kw)
        ax_map.axhline(actual_lat, color="cyan", lw=1.4, ls="--",
                       label=f"lat slice  {actual_lat:.1f}°N")
        ax_map.axvline(actual_lon, color="lime", lw=1.4, ls="--",
                       label=f"lon slice  {actual_lon:.1f}°E")
        ax_map.plot(self.center_lon, self.center_lat,
                    "w+", ms=10, mew=2, zorder=5, label="centre")
        ax_map.set_xlim(lon_min_d, lon_max_d)
        ax_map.set_ylim(lat_min_d, lat_max_d)
        ax_map.set_xlabel("Longitude  (°E)")
        ax_map.set_ylabel("Latitude  (°N)")
        ax_map.set_title(f"Horizontal map  —  altitude = {actual_alt:.1f} km  "
                         f"(grid level {ir}/{self.Nh - 1})", fontsize=10)
        ax_map.legend(loc="upper right", fontsize=8,
                      framealpha=0.6, labelcolor="white", facecolor="#222222")

        # ---- Panel 2: meridional slice (alt vs lat) ----
        Ne_mer = self.density[:, :, iph]       # (Nh, Nlat)
        pm2 = ax_mer.pcolormesh(lat_deg, alt_km_ax, Ne_mer,
                                vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
        fig.colorbar(pm2, ax=ax_mer, label="Ne  (m⁻³)", pad=0.02)
        for i in range(0, self.Nlat, sth): ax_mer.axvline(lat_deg[i], **gl_kw)
        for k in range(0, self.Nh, sr):    ax_mer.axhline(alt_km_ax[k], **gl_kw)
        ax_mer.axhline(actual_alt, color="yellow", lw=1.4, ls="--",
                       label=f"alt = {actual_alt:.0f} km")
        ax_mer.axvline(actual_lat, color="cyan", lw=1.4, ls="--",
                       label=f"lat = {actual_lat:.1f}°N")
        ax_mer.set_xlim(lat_min_d, lat_max_d)
        ax_mer.set_ylim(alt_km_ax[0], alt_km_ax[-1])
        ax_mer.set_xlabel("Latitude  (°N)")
        ax_mer.set_ylabel("Altitude  (km)")
        ax_mer.set_title(f"Meridional slice  —  lon = {actual_lon:.1f}°E  "
                         f"(grid col {iph}/{self.Nlon - 1})", fontsize=10)
        ax_mer.legend(fontsize=8, framealpha=0.6,
                      labelcolor="white", facecolor="#222222")

        # ---- Panel 3: zonal slice (alt vs lon) ----
        Ne_zon = self.density[:, ith, :]       # (Nh, Nlon)
        pm3 = ax_zon.pcolormesh(lon_deg, alt_km_ax, Ne_zon,
                                vmin=vmin, vmax=vmax, cmap=cmap, shading="auto")
        fig.colorbar(pm3, ax=ax_zon, label="Ne  (m⁻³)", pad=0.02)
        for j in range(0, self.Nlon, sph): ax_zon.axvline(lon_deg[j], **gl_kw)
        for k in range(0, self.Nh, sr):    ax_zon.axhline(alt_km_ax[k], **gl_kw)
        ax_zon.axhline(actual_alt, color="yellow", lw=1.4, ls="--",
                       label=f"alt = {actual_alt:.0f} km")
        ax_zon.axvline(actual_lon, color="lime", lw=1.4, ls="--",
                       label=f"lon = {actual_lon:.1f}°E")
        ax_zon.set_xlim(lon_min_d, lon_max_d)
        ax_zon.set_ylim(alt_km_ax[0], alt_km_ax[-1])
        ax_zon.set_xlabel("Longitude  (°E)")
        ax_zon.set_ylabel("Altitude  (km)")
        ax_zon.set_title(f"Zonal slice  —  lat = {actual_lat:.1f}°N  "
                         f"(grid row {ith}/{self.Nlat - 1})", fontsize=10)
        ax_zon.legend(fontsize=8, framealpha=0.6,
                      labelcolor="white", facecolor="#222222")

        return fig

    # ---- Coordinate helpers ----

    @staticmethod
    def xyz_to_geodetic(x, y, z):
        """Cartesian (km) → geodetic (lat deg, lon deg, height km above WGS84)."""
        x, y, z = (np.asarray(v, float) for v in (x, y, z))
        lat, lon, h_m = geometry.xyz_m_to_latlon(np.stack([x, y, z]) * 1e3)
        return lat, lon, h_m / 1e3

    @staticmethod
    def geodetic_to_xyz(lat, lon, h_km):
        """Geodetic (lat deg, lon deg, height km above WGS84) → Cartesian (km)."""
        xyz_m = geometry.latlon_to_xyz_m(lat, lon, np.asarray(h_km, float) * 1e3)
        return xyz_m[0] / 1e3, xyz_m[1] / 1e3, xyz_m[2] / 1e3

    def _map_lon(self, lon):
        """
        Map lon (deg) into the unwrapped axis range [lon_min, lon_max].
        Handles patches straddling the 0/360° meridian transparently.
        """
        return self.lon_min + ((lon - self.lon_min) % 360.0)

    # ---- Interpolation entry points ----

    def electron_density_geodetic(self, lat, lon, h):
        """
        Electron density at geodetic (lat deg, lon deg, h km above WGS84).
        Returns 0 outside the section, below h=0, or above h_max.
        Accepts scalars or numpy arrays.
        """
        lat, lon, h = (np.asarray(v, float) for v in (lat, lon, h))
        scalar = (lat.ndim == 0)
        lat, lon, h = np.atleast_1d(lat), np.atleast_1d(lon), np.atleast_1d(h)

        lon_mapped = self._map_lon(lon)

        in_section = (
            (h   >= self.h[0])          & (h   <= self.h[-1])   &
            (lat >= self.lat[0])        & (lat <= self.lat[-1]) &
            (lon_mapped >= self.lon[0]) & (lon_mapped <= self.lon[-1])
        )

        result = np.zeros(lat.shape)
        if np.any(in_section):
            # NOTE: interpolator axes are (h, lat, lon) -- see _make_interpolator --
            # so query points must be stacked in that order, not the public
            # (lat, lon, h) argument order.
            pts = np.stack([h[in_section],
                            lat[in_section],
                            lon_mapped[in_section]], axis=-1)
            result[in_section] = self._interp(pts)

        return float(result[0]) if scalar else result

    def electron_density_xyz(self, x, y, z):
        """Electron density at Cartesian (x, y, z) in kilometres."""
        return self.electron_density_geodetic(*self.xyz_to_geodetic(x, y, z))

    def electron_density_xyz_m(self, x, y, z):
        """Electron density at Cartesian (x, y, z) in metres."""
        return self.electron_density_xyz(
            np.asarray(x) / 1e3,
            np.asarray(y) / 1e3,
            np.asarray(z) / 1e3,
        )

    def __call__(self, x, y, z):
        """Ne = iono(x, y, z)  with x, y, z in km."""
        return self.electron_density_xyz(x, y, z)


# ================================================================
# Example / smoke-test
# ================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")   # change to "Qt5Agg" / "MacOSX" as needed
    import matplotlib.pyplot as plt

    iono = Ionosphere(
        center_lat=41.9,
        center_lon=12.5,
        dtheta_deg=30.0,
        dphi_deg=40.0,
        h_max=800.0,
        dr=2.0,
        dang_deg=0.5,
        dt=datetime(2023, 6, 21, 14, 0, 0),
        n_ang=15,
    )
    iono.grid_info()

    x_c, y_c, z_c = iono.geodetic_to_xyz(iono.center_lat, iono.center_lon, 300.0)
    print(f"\nNe at section centre, 300 km : {iono(x_c, y_c, z_c):.3e} m⁻³")

    x_out, y_out, z_out = iono.geodetic_to_xyz(80.0, 120.0, 300.0)
    print(f"Ne outside section           : {iono(x_out, y_out, z_out):.3e} m⁻³  (expected 0)")

    print(f"Ne inside Earth              : "
          f"{iono.electron_density_geodetic(iono.center_lat, iono.center_lon, -100.0):.3e} m⁻³  (expected 0)")

    fig = iono.plot(alt_km=300.0)
    plt.show()
