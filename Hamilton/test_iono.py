"""
Synthetic 3D sectioned spherical ionosphere generator.

Builds a Chapman-layer-based electron density field on a uniform 3D
spherical grid centred at Earth's centre, covering only a user-specified
patch of the sphere:

    r     ∈ [R_E,  R_E + h_max]          (geocentric radius, km)
    θ     ∈ [θ_c − dθ/2,  θ_c + dθ/2]   (co-latitude, rad)
    φ     ∈ [φ_c − dφ/2,  φ_c + dφ/2]   (east longitude, rad)

The section is defined by its centre co-latitude/longitude (or,
equivalently, geographic latitude/longitude) and its full angular
extents dθ and dφ.  Outside this patch the electron density is
considered zero (i.e. we simply don't store it).

The grid is homogeneous: dr, dθ, dφ are constant throughout.

Provides trilinear interpolation (in r, θ, φ space) to evaluate the
density at arbitrary points, supplied either as spherical or Cartesian
coordinates.  Points outside the stored section return 0.

COORDINATE CONVENTION
----------------------
    r   : geocentric radius (km),  r ∈ [R_E, R_E + h_max]
    θ   : co-latitude  (rad),      θ = 0 at N pole, π at S pole
    φ   : east longitude (rad),    φ ∈ [0, 2π)

The section centre is given as geographic (latitude, longitude) in
degrees, which are converted to (θ_c, φ_c) internally.

The φ axis is stored in an *unwrapped* linear range
[φ_c − dφ/2, φ_c + dφ/2] so that sections straddling the 0 / 360°
meridian are handled transparently.

All lengths in kilometres throughout.

PUBLIC API
----------
    Ionosphere(center_lat, center_lon,
                        dtheta_deg, dphi_deg, h_max,
                        dr, dang_deg, ...)

    .electron_density_sph(r, theta, phi)   → Ne  (spherical, km / rad)
    .electron_density_xyz(x, y, z)         → Ne  (Cartesian, km)
    .electron_density_xyz_m(x, y, z)       → Ne  (Cartesian, metres)
    .__call__(x, y, z)                     → Ne  (Cartesian, km)
    .regenerate(seed=None)
    .grid_info()
    .plot(alt_km, slice_lat, slice_lon)    → matplotlib Figure
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Earth constant
# ---------------------------------------------------------------------------
R_E = 6371.0   # mean Earth radius, km


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Ionosphere:
    """
    Chapman-layer ionosphere on a uniform spherical grid covering only a
    rectangular patch (in co-latitude × longitude) of the sphere.

    Parameters
    ----------
    Section definition
    ~~~~~~~~~~~~~~~~~~
    center_lat : float
        Geographic latitude of the patch centre (degrees, −90 … +90).
    center_lon : float
        Geographic longitude of the patch centre (degrees, −180 … +360).
    dtheta_deg : float
        Full angular extent of the patch in co-latitude (degrees).
        The patch spans [θ_c − dθ/2, θ_c + dθ/2], clamped to [0, π].
    dphi_deg : float
        Full angular extent of the patch in longitude (degrees).
        Values > 360 are clamped to 360 (full longitude wrap).

    Radial extent
    ~~~~~~~~~~~~~
    h_max : float
        Maximum altitude above ground (km).  Default 800 km.

    Grid spacing (uniform throughout)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dr : float
        Radial cell size (km).  Default 2 km.
    dang_deg : float
        Angular cell size for both θ and φ (degrees).  Default 1°.
        (Use separate dr / dang_deg to control radial vs angular
        resolution independently.)

    Physics parameters
    ~~~~~~~~~~~~~~~~~~
    hour : float
        Local solar time at the Greenwich meridian (φ = 0), 0–24.
    solar_flux : float
        F10.7-like solar activity proxy (sfu).
        ~70 = solar min, ~150 = moderate, ~250 = solar max.
    turbulence_strength : float
        Fractional amplitude of small-scale density fluctuations (0–1).
    turbulence_scale : float
        Correlation length of turbulence (km).
    horizontal_strength : float
        Fractional amplitude of large-scale (gravity-wave-like) horizontal
        undulation of the ionospheric layer.
    horizontal_scale_deg : float
        Angular wavelength (degrees) of horizontal structure.
    seed : int or None
        Random seed for reproducibility.
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
                 # --- physics ---
                 hour=14.0,
                 solar_flux=180.0,
                 turbulence_strength=0.08,
                 turbulence_scale=15.0,
                 horizontal_strength=0.12,
                 horizontal_scale_deg=30.0,
                 seed=42):

        # ---- Section centre in spherical coords ----
        self.center_lat = float(center_lat)
        self.center_lon = float(center_lon)
        self.theta_c = np.deg2rad(90.0 - center_lat)       # co-latitude
        self.phi_c   = np.deg2rad(center_lon % 360.0)      # east longitude [0, 2π)

        # ---- Angular half-extents ----
        half_theta = np.deg2rad(float(dtheta_deg)) / 2.0
        half_phi   = np.deg2rad(min(float(dphi_deg), 360.0)) / 2.0

        # θ boundaries clamped to the valid co-latitude range [0, π]
        self.theta_min = float(np.clip(self.theta_c - half_theta, 0.0,    np.pi))
        self.theta_max = float(np.clip(self.theta_c + half_theta, 0.0,    np.pi))

        # φ stored as an *unwrapped* linear range centred on φ_c.
        # This transparently handles patches that straddle the 0 / 2π seam.
        # Query points are mapped into this range via modular arithmetic.
        self.phi_min = self.phi_c - half_phi   # may be negative
        self.phi_max = self.phi_c + half_phi   # may exceed 2π

        # ---- Radial extent ----
        self.h_max   = float(h_max)
        self.r_min   = R_E
        self.r_max   = R_E + h_max

        # ---- Grid spacing ----
        self.dr      = float(dr)
        self.dang    = float(np.deg2rad(dang_deg))

        # ---- Physics ----
        self.hour                = float(hour)
        self.solar_flux          = float(solar_flux)
        self.turbulence_strength = float(turbulence_strength)
        self.turbulence_scale    = max(float(turbulence_scale), 1e-6)
        self.horizontal_strength = float(horizontal_strength)
        self.horizontal_scale    = float(np.deg2rad(horizontal_scale_deg))

        self.rng = np.random.default_rng(seed)

        # ---- Build uniform axes ----
        self.r     = self._linspace_axis(self.r_min,     self.r_max,     self.dr)
        self.theta = self._linspace_axis(self.theta_min, self.theta_max, self.dang)
        self.phi   = self._linspace_axis(self.phi_min,   self.phi_max,   self.dang)

        self.Nr  = len(self.r)
        self.Nth = len(self.theta)
        self.Nph = len(self.phi)

        # ---- Density field and interpolator ----
        self.density = self._build_density()
        self._interp = self._make_interpolator()

    # ----------------------------------------------------------------
    # Axis helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _linspace_axis(lo, hi, step):
        """
        Uniform axis from lo to hi with spacing ≈ step.
        Always includes both endpoints; N = max(2, round((hi-lo)/step)+1).
        """
        N = max(2, int(round((hi - lo) / step)) + 1)
        return np.linspace(lo, hi, N)

    # ----------------------------------------------------------------
    # Physics helpers
    # ----------------------------------------------------------------

    def _diurnal_factor_field(self, PHIg):
        """
        Vectorised diurnal (local-solar-time) factor for every grid longitude.
        LST at longitude φ (rad):  LST = hour + φ_deg / 15  (mod 24).
        Returns (Nm_scale, hm_shift) arrays shaped like PHIg.
        """
        phi_deg  = np.rad2deg(PHIg)
        lst      = (self.hour + phi_deg / 15.0) % 24.0
        phase    = (lst - 12.0) / 24.0 * 2.0 * np.pi
        cos_like = np.cos(phase)

        night_floor = 0.15
        Nm_scale = night_floor + (1.0 - night_floor) * (0.5 * (1.0 + cos_like)) ** 1.5
        Nm_scale = np.clip(Nm_scale, night_floor, 1.0)

        hm_shift = -20.0 * cos_like   # layer lower at noon, higher at night
        return Nm_scale, hm_shift

    def _solar_factor(self):
        """Scalar solar-flux scaling for peak density and layer height."""
        ratio    = self.solar_flux / 150.0
        Nm_scale = ratio ** 0.7
        hm_scale = 1.0 + 0.15 * np.log(max(ratio, 1e-3))
        return float(Nm_scale), float(hm_scale)

    def _horizontal_modulation(self, THg, PHg):
        """
        Quasi-periodic wave-like modulation in the (θ, φ) plane.
        Returns a multiplicative factor centred on 1.0.
        """
        if self.horizontal_strength <= 0:
            return np.ones_like(THg)
        k0 = 2.0 * np.pi / self.horizontal_scale
        components = [
            (1.00 * k0,  0.0,         1.0),
            (0.55 * k0,  np.pi / 3,   0.6),
            (1.70 * k0, -np.pi / 5,   0.3),
        ]
        mod = np.zeros_like(THg)
        for k, angle, weight in components:
            mod += weight * np.sin(k * np.cos(angle) * THg
                                   + k * np.sin(angle) * PHg)
        mod /= sum(w for _, _, w in components)
        return 1.0 + self.horizontal_strength * mod

    def _turbulence_field(self, shape):
        """Gaussian-smoothed multiplicative noise field."""
        if self.turbulence_strength <= 0:
            return np.ones(shape)
        white    = self.rng.normal(0.0, 1.0, size=shape)
        arc_km   = R_E * self.dang    # angular arc-length per cell at Earth's surface
        sigma_vox = (
            max(self.turbulence_scale / self.dr,  0.5),
            max(self.turbulence_scale / arc_km,   0.5),
            max(self.turbulence_scale / arc_km,   0.5),
        )
        smoothed = gaussian_filter(white, sigma=sigma_vox, mode="nearest")
        smoothed /= smoothed.std() + 1e-12
        return 1.0 + self.turbulence_strength * smoothed

    # ----------------------------------------------------------------
    # Build the 3-D density array   shape: (Nr, Nth, Nph)
    # ----------------------------------------------------------------

    def _build_density(self):
        # Full 3-D meshgrid in (r, θ, φ)
        Rg, THg, PHg = np.meshgrid(self.r, self.theta, self.phi, indexing="ij")
        Hg = Rg - R_E   # altitude above ground (km)

        # ---- Chapman F2 layer ----
        Nm0 = 1.2e12    # baseline peak density  (m⁻³)
        hm0 = 300.0     # baseline peak altitude (km)
        H   = 60.0      # scale height           (km)

        solar_Nm,   solar_hm   = self._solar_factor()
        diurnal_Nm, diurnal_hm = self._diurnal_factor_field(PHg)

        Nm = Nm0 * diurnal_Nm * solar_Nm
        hm = (hm0 + diurnal_hm) * solar_hm

        zp      = (Hg - hm) / H
        chapman = Nm * np.exp(0.5 * (1.0 - zp - np.exp(-zp)))

        # ---- Secondary E-layer ----
        Nm_E    = 0.15 * Nm0 * np.clip(diurnal_Nm, 0.2, None)
        zpE     = (Hg - 110.0) / 20.0
        e_layer = Nm_E * np.exp(0.5 * (1.0 - zpE - np.exp(-zpE)))

        density = (chapman + e_layer) \
                  * self._horizontal_modulation(THg, PHg) \
                  * self._turbulence_field(Rg.shape)

        return np.clip(density, 0.0, None)

    # ----------------------------------------------------------------
    # Interpolator
    # ----------------------------------------------------------------

    def _make_interpolator(self):
        """
        RegularGridInterpolator over (r, θ, φ).
        The φ axis is the unwrapped linear range [φ_min, φ_max];
        query points must be mapped into this range before calling
        (handled by electron_density_sph).
        """
        return RegularGridInterpolator(
            (self.r, self.theta, self.phi),
            self.density,
            method="linear",
            bounds_error=False,
            fill_value=0.0,     # outside the stored section → zero
        )

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def regenerate(self, seed=None):
        """
        Rebuild the density field (new turbulence realisation or after
        changing a parameter by direct attribute assignment).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.density = self._build_density()
        self._interp = self._make_interpolator()

    def grid_info(self):
        """Print a human-readable summary of the grid."""
        lat_min = 90.0 - np.rad2deg(self.theta_max)
        lat_max = 90.0 - np.rad2deg(self.theta_min)
        lon_min = np.rad2deg(self.phi_min)
        lon_max = np.rad2deg(self.phi_max)
        arc_km  = R_E * self.dang

        print("=" * 60)
        print(f"  Centre      : lat={self.center_lat:.2f}°  "
              f"lon={self.center_lon:.2f}°")
        print(f"  θ extent    : {lat_min:.2f}°N … {lat_max:.2f}°N  "
              f"(Δlat = {lat_max - lat_min:.2f}°)")
        print(f"  φ extent    : {lon_min:.2f}°E … {lon_max:.2f}°E  "
              f"(Δlon = {lon_max - lon_min:.2f}°)")
        print(f"  Altitude    : 0 … {self.h_max:.0f} km")
        print(f"  Grid        : {self.Nr} × {self.Nth} × {self.Nph}  "
              f"= {self.Nr * self.Nth * self.Nph:,} voxels")
        print(f"  Spacing     : dr = {self.dr:.2f} km   "
              f"dang = {np.rad2deg(self.dang):.3f}°  ({arc_km:.2f} km arc)")
        print("=" * 60)

    def plot(self, alt_km=None, slice_lat=None, slice_lon=None):
        """
        Produce a three-panel figure visualising the electron density field.

        Each panel is a 2-D cross-section through the stored density array,
        cutting the grid at a chosen altitude, latitude, and longitude
        respectively.  Grid lines are overlaid to show the cell structure.

        Parameters
        ----------
        alt_km : float, optional
            Altitude (km above ground) of the horizontal map (top panel).
            Snapped to the nearest grid level.
            Default: altitude of the F2 peak (~300 km) or the closest
            available grid level.
        slice_lat : float, optional
            Geographic latitude (°N) of the zonal slice (bottom-right panel:
            altitude vs longitude).  Snapped to the nearest θ grid line.
            Default: patch centre latitude.
        slice_lon : float, optional
            Geographic longitude (°E) of the meridional slice (bottom-left
            panel: altitude vs latitude).  Snapped to the nearest φ grid
            line.  Default: patch centre longitude.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object (not shown; call fig.show() or plt.show()
            yourself, or save with fig.savefig(...)).
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        # ---- Derived display axes (geographic, degrees) ----
        lat_deg = 90.0 - np.rad2deg(self.theta)          # (Nth,)
        lon_deg = np.rad2deg(self.phi)                    # (Nph,) — unwrapped
        alt_km_ax = self.r - R_E                          # (Nr,)

        lat_min_d = lat_deg.min();  lat_max_d = lat_deg.max()
        lon_min_d = lon_deg.min();  lon_max_d = lon_deg.max()

        # ---- Snap slice positions to nearest grid indices ----
        alt_target = 300.0 if alt_km is None else float(alt_km)
        alt_target = float(np.clip(alt_target, alt_km_ax[0], alt_km_ax[-1]))
        ir = int(np.argmin(np.abs(alt_km_ax - alt_target)))
        actual_alt = alt_km_ax[ir]

        lat_target = self.center_lat if slice_lat is None else float(slice_lat)
        lat_target = float(np.clip(lat_target, lat_min_d, lat_max_d))
        ith = int(np.argmin(np.abs(lat_deg - lat_target)))
        actual_lat = lat_deg[ith]

        lon_target = self.center_lon if slice_lon is None else float(slice_lon)
        # Map lon_target into the (possibly unwrapped) axis range
        lon_target_mapped = np.rad2deg(self._map_phi(np.deg2rad(lon_target)))
        lon_target_mapped = float(np.clip(lon_target_mapped,
                                          lon_deg.min(), lon_deg.max()))
        iph = int(np.argmin(np.abs(lon_deg - lon_target_mapped)))
        actual_lon = lon_deg[iph]

        # ---- Decide grid-line stride (aim for ~10 lines per axis) ----
        def _stride(n, target=10):
            return max(1, n // target)

        sr  = _stride(self.Nr)
        sth = _stride(self.Nth)
        sph = _stride(self.Nph)

        # ---- Shared colour scale: global min/max of the density cube ----
        vmin = float(self.density.min())
        vmax = float(self.density.max())

        cmap = "plasma"

        # ================================================================
        # Layout: 2 rows × 2 cols, top row merged into a wide panel
        # ================================================================
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"Ionosphere  |  centre ({self.center_lat:.1f}°N, "
            f"{self.center_lon:.1f}°E)  |  "
            f"hour={self.hour}  F10.7={self.solar_flux}",
            fontsize=12, fontweight="bold", y=0.98,
        )

        # GridSpec: row 0 spans both columns (horizontal map),
        #           row 1 has two equal panels (meridional + zonal slice)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig,
                      height_ratios=[1.1, 1.0],
                      hspace=0.38, wspace=0.32)

        ax_map  = fig.add_subplot(gs[0, :])   # top: horizontal map
        ax_mer  = fig.add_subplot(gs[1, 0])   # bottom-left:  meridional (alt vs lat)
        ax_zon  = fig.add_subplot(gs[1, 1])   # bottom-right: zonal      (alt vs lon)

        gl_kw = dict(color="white", lw=0.5, alpha=0.55)   # grid-line style

        # ============================================================
        # Panel 1 — Horizontal map at alt = ir
        # density[ir, :, :] has axes (Nth, Nph) → (lat, lon)
        # ============================================================
        Ne_map = self.density[ir, :, :]        # shape (Nth, Nph)

        pm1 = ax_map.pcolormesh(
            lon_deg, lat_deg, Ne_map,
            vmin=vmin, vmax=vmax, cmap=cmap, shading="auto",
        )
        fig.colorbar(pm1, ax=ax_map, label="Ne  (m⁻³)", pad=0.02)

        # Grid lines
        for i in range(0, self.Nth, sth):
            ax_map.axhline(lat_deg[i],  **gl_kw)
        for j in range(0, self.Nph, sph):
            ax_map.axvline(lon_deg[j],  **gl_kw)

        # Mark the two slice positions
        ax_map.axhline(actual_lat, color="cyan",   lw=1.4, ls="--",
                       label=f"lat slice  {actual_lat:.1f}°N")
        ax_map.axvline(actual_lon, color="lime",   lw=1.4, ls="--",
                       label=f"lon slice  {actual_lon:.1f}°E")
        ax_map.plot(self.center_lon, self.center_lat,
                    "w+", ms=10, mew=2, zorder=5, label="centre")

        ax_map.set_xlim(lon_min_d, lon_max_d)
        ax_map.set_ylim(lat_min_d, lat_max_d)
        ax_map.set_xlabel("Longitude  (°E)")
        ax_map.set_ylabel("Latitude  (°N)")
        ax_map.set_title(
            f"Horizontal map  —  altitude = {actual_alt:.1f} km  "
            f"(grid level {ir}/{self.Nr - 1})",
            fontsize=10,
        )
        ax_map.legend(loc="upper right", fontsize=8,
                      framealpha=0.6, labelcolor="white",
                      facecolor="#222222")

        # ============================================================
        # Panel 2 — Meridional slice at iph  (altitude vs latitude)
        # density[:, :, iph] has axes (Nr, Nth) → (alt, lat)
        # ============================================================
        Ne_mer = self.density[:, :, iph]      # shape (Nr, Nth)

        pm2 = ax_mer.pcolormesh(
            lat_deg, alt_km_ax, Ne_mer,
            vmin=vmin, vmax=vmax, cmap=cmap, shading="auto",
        )
        fig.colorbar(pm2, ax=ax_mer, label="Ne  (m⁻³)", pad=0.02)

        # Grid lines
        for i in range(0, self.Nth, sth):
            ax_mer.axvline(lat_deg[i],  **gl_kw)
        for k in range(0, self.Nr, sr):
            ax_mer.axhline(alt_km_ax[k], **gl_kw)

        # Mark the horizontal-map altitude and the chosen lat
        ax_mer.axhline(actual_alt, color="yellow", lw=1.4, ls="--",
                       label=f"alt = {actual_alt:.0f} km")
        ax_mer.axvline(actual_lat, color="cyan",   lw=1.4, ls="--",
                       label=f"lat = {actual_lat:.1f}°N")

        ax_mer.set_xlim(lat_min_d, lat_max_d)
        ax_mer.set_ylim(alt_km_ax[0], alt_km_ax[-1])
        ax_mer.set_xlabel("Latitude  (°N)")
        ax_mer.set_ylabel("Altitude  (km)")
        ax_mer.set_title(
            f"Meridional slice  —  lon = {actual_lon:.1f}°E  "
            f"(grid col {iph}/{self.Nph - 1})",
            fontsize=10,
        )
        ax_mer.legend(fontsize=8, framealpha=0.6,
                      labelcolor="white", facecolor="#222222")

        # ============================================================
        # Panel 3 — Zonal slice at ith  (altitude vs longitude)
        # density[:, ith, :] has axes (Nr, Nph) → (alt, lon)
        # ============================================================
        Ne_zon = self.density[:, ith, :]      # shape (Nr, Nph)

        pm3 = ax_zon.pcolormesh(
            lon_deg, alt_km_ax, Ne_zon,
            vmin=vmin, vmax=vmax, cmap=cmap, shading="auto",
        )
        fig.colorbar(pm3, ax=ax_zon, label="Ne  (m⁻³)", pad=0.02)

        # Grid lines
        for j in range(0, self.Nph, sph):
            ax_zon.axvline(lon_deg[j],  **gl_kw)
        for k in range(0, self.Nr, sr):
            ax_zon.axhline(alt_km_ax[k], **gl_kw)

        # Mark the horizontal-map altitude and the chosen lon
        ax_zon.axhline(actual_alt, color="yellow", lw=1.4, ls="--",
                       label=f"alt = {actual_alt:.0f} km")
        ax_zon.axvline(actual_lon, color="lime",   lw=1.4, ls="--",
                       label=f"lon = {actual_lon:.1f}°E")

        ax_zon.set_xlim(lon_min_d, lon_max_d)
        ax_zon.set_ylim(alt_km_ax[0], alt_km_ax[-1])
        ax_zon.set_xlabel("Longitude  (°E)")
        ax_zon.set_ylabel("Altitude  (km)")
        ax_zon.set_title(
            f"Zonal slice  —  lat = {actual_lat:.1f}°N  "
            f"(grid row {ith}/{self.Nth - 1})",
            fontsize=10,
        )
        ax_zon.legend(fontsize=8, framealpha=0.6,
                      labelcolor="white", facecolor="#222222")

        return fig

    # ---- Coordinate helpers ----

    @staticmethod
    def xyz_to_sph(x, y, z):
        """Cartesian (km) → spherical (r km, θ rad, φ rad ∈ [0, 2π))."""
        x, y, z = (np.asarray(v, float) for v in (x, y, z))
        r = np.sqrt(x**2 + y**2 + z**2)
        with np.errstate(invalid="ignore", divide="ignore"):
            theta = np.where(r > 0, np.arccos(np.clip(z / r, -1.0, 1.0)), 0.0)
        phi = np.arctan2(y, x) % (2.0 * np.pi)
        return r, theta, phi

    @staticmethod
    def sph_to_xyz(r, theta, phi):
        """Spherical (r km, θ rad, φ rad) → Cartesian (km)."""
        s = np.sin(theta)
        return (r * s * np.cos(phi),
                r * s * np.sin(phi),
                r * np.cos(theta))

    def _map_phi(self, phi):
        """
        Map an arbitrary φ (rad) into the unwrapped axis range
        [phi_min, phi_max] used by the interpolator.

        Because the φ axis may be stored outside [0, 2π) (e.g. for a
        patch centred near 0° the axis runs from −Δ to +Δ), we shift
        the query point by the appropriate multiple of 2π so that it
        lands in the stored range.
        """
        # Shift phi so that it is within phi_min and phi_min + 2π,
        # then the interpolator's fill_value=0 handles out-of-patch points.
        return self.phi_min + ((phi - self.phi_min) % (2.0 * np.pi))

    # ---- Interpolation entry points ----

    def electron_density_sph(self, r, theta, phi):
        """
        Electron density at spherical coordinates (r km, θ rad, φ rad).
        Points outside the section, inside Earth, or above h_max return 0.
        Accepts scalars or numpy arrays of matching shape.
        """
        r, theta, phi = (np.asarray(v, float) for v in (r, theta, phi))
        scalar = (r.ndim == 0)
        r, theta, phi = np.atleast_1d(r), np.atleast_1d(theta), np.atleast_1d(phi)

        # Map φ into the unwrapped axis range
        phi_mapped = self._map_phi(phi)

        # Determine which points actually lie inside the stored section.
        # Points outside get zero without ever touching the interpolator.
        in_section = (
            (r     >= self.r[0])     & (r     <= self.r[-1])     &
            (theta >= self.theta[0]) & (theta <= self.theta[-1]) &
            (phi_mapped >= self.phi[0])  & (phi_mapped <= self.phi[-1])
        )

        result = np.zeros(r.shape)
        if np.any(in_section):
            pts = np.stack([r[in_section],
                            theta[in_section],
                            phi_mapped[in_section]], axis=-1)
            result[in_section] = self._interp(pts)

        return float(result[0]) if scalar else result

    def electron_density_xyz(self, x, y, z):
        """Electron density at Cartesian (x, y, z) in kilometres."""
        return self.electron_density_sph(*self.xyz_to_sph(x, y, z))

    def electron_density_xyz_m(self, x, y, z):
        """Electron density at Cartesian (x, y, z) in metres."""
        return self.electron_density_xyz(
            np.asarray(x) / 1e3,
            np.asarray(y) / 1e3,
            np.asarray(z) / 1e3,
        )

    def __call__(self, x, y, z):
        """Ne = iono(x, y, z) with x, y, z in km."""
        return self.electron_density_xyz(x, y, z)


# ================================================================
# Example / smoke-test
# ================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")   # change to "Qt5Agg" / "MacOSX" etc. as needed
    import matplotlib.pyplot as plt

    # A 30° × 40° patch centred on Rome (41.9°N, 12.5°E),
    # covering latitudes 26.9°N–56.9°N and longitudes −7.5°E–32.5°E.
    iono = Ionosphere(
        center_lat=41.9,
        center_lon=12.5,
        dtheta_deg=30.0,
        dphi_deg=40.0,
        h_max=800.0,
        dr=0.5,
        dang_deg=0.5,
        hour=14.0,
        solar_flux=180.0,
        turbulence_strength=0.08,
        turbulence_scale=15.0,
        horizontal_strength=0.12,
        seed=42,
    )
    iono.grid_info()

    # ---- Interpolation tests ----
    th_c = iono.theta_c
    ph_c = iono.phi_c
    x_c, y_c, z_c = iono.sph_to_xyz(R_E + 300.0, th_c, ph_c)
    print(f"\nNe at section centre, 300 km alt : {iono(x_c, y_c, z_c):.3e} m⁻³")

    x_out, y_out, z_out = iono.sph_to_xyz(
        R_E + 300.0, np.deg2rad(80.0), np.deg2rad(120.0))
    print(f"Ne outside section (80°N,120°E) : {iono(x_out, y_out, z_out):.3e} m⁻³  (expected 0)")

    print(f"Ne inside Earth (r=R_E-100 km)  : "
          f"{iono.electron_density_sph(R_E - 100.0, th_c, ph_c):.3e} m⁻³  (expected 0)")

    # ---- Visualise ----
    # Slices at the patch centre, horizontal map at 300 km altitude
    fig = iono.plot(alt_km=300.0)
    plt.show()