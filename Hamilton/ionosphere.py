"""
Synthetic 3D ionosphere generator.

Builds a Chapman-layer-based electron density field on a 3D grid, with
optional day/night variation, solar-activity scaling, horizontal
(large-scale) structure, and small-scale turbulence/noise. Provides
trilinear interpolation to evaluate the density at arbitrary
(non-grid) points.

COORDINATE CONVENTION
----------------------
z is "up" (height). x and y are horizontal. The grid spans:

    x in [-Nx/2, +Nx/2]
    y in [-Ny/2, +Ny/2]
    z in [0, Nz]

so the origin (0, 0, 0) sits on the ground, at the center of the
horizontal (x-y) extent -- i.e. the x-z and y-z "walls" of the box are
centered on x=0 and y=0, while z=0 is the floor, not the center.

All lengths are in the same units the caller chooses (e.g. metres or
kilometres) -- just be consistent between Nx, Ny, Nz, the grid
resolution, and any code (such as a ray tracer) that queries this
density field.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class Ionosphere:
    def __init__(self, Nx, Ny, Nz, dx=2.0, dy=2.0, dz=2.0,
                 hour=14.0, solar_flux=180.0,
                 turbulence_strength=0.08, turbulence_scale=15.0,
                 horizontal_strength=0.12, horizontal_scale=None,
                 seed=42):
        """
        Nx, Ny, Nz : number of grid points along x (horizontal),
                     y (horizontal), z (height)
        dx, dy, dz : grid spacing along each axis (same length units
                     as everything else in your simulation, e.g. km)

        hour              : local solar time, 0-24 (hour=12 -> noon,
                             peak ionization; hour=0/24 -> midnight,
                             minimum ionization)
        solar_flux        : proxy for solar activity, roughly an F10.7
                             index in solar flux units (sfu).
                             ~70  = solar minimum
                             ~150 = moderate
                             ~250 = solar maximum
                             Scales both peak density and layer height.

        turbulence_strength : fractional amplitude (0-1ish) of
                               small-scale random density fluctuations
        turbulence_scale     : characteristic correlation length (in
                                grid-units of length) of the turbulence,
                                via Gaussian smoothing of white noise

        horizontal_strength : fractional amplitude of large-scale
                               horizontal (x,y) wave-like structure
                               (gravity-wave-like undulation of the
                               layer height / density)
        horizontal_scale     : wavelength (length units) of the
                                horizontal structure; default is
                                ~1/3 of the horizontal domain size

        seed : random seed for reproducibility (None -> random)
        """
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz

        self.hour = hour
        self.solar_flux = solar_flux
        self.turbulence_strength = turbulence_strength
        self.turbulence_scale = max(turbulence_scale, 1e-6)
        self.horizontal_strength = horizontal_strength

        # Grid coordinates: x, y centered on 0; z from 0 up.
        self.x = (np.arange(Nx) - (Nx - 1) / 2.0) * dx
        self.y = (np.arange(Ny) - (Ny - 1) / 2.0) * dy
        self.z = np.arange(Nz) * dz

        if horizontal_scale is None:
            Lx = Nx * dx
            Ly = Ny * dy
            horizontal_scale = max(Lx, Ly) / 3.0
        self.horizontal_scale = horizontal_scale

        self.rng = np.random.default_rng(seed)

        self.density = self._build_density()

        # Trilinear interpolator. RegularGridInterpolator expects axes
        # in the order matching the array's axes; we store density as
        # density[ix, iy, iz] to match (x, y, z) argument order.
        self._interp = RegularGridInterpolator(
            (self.x, self.y, self.z),
            self.density,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate at edges (clamped, see below)
        )

    # ------------------------------------------------------------
    # Diurnal and solar-activity scaling
    # ------------------------------------------------------------
    def _diurnal_factor(self):
        """
        Returns (Nm_scale, hm_shift) where Nm_scale multiplies the
        peak density and hm_shift is an additive shift (length units)
        to the layer's peak height, both as functions of local hour.

        Simple model: density peaks at local noon and falls to a
        small but nonzero "nighttime floor" (the ionosphere doesn't
        vanish at night -- residual F-layer ionization persists).
        Layer height rises somewhat at night (lower density -> slower
        recombination balance shifts the peak up) and is lowest at
        noon.
        """
        # Solar zenith angle proxy: cos-like factor, 1 at noon, -1 at midnight
        phase = (self.hour - 12.0) / 24.0 * 2.0 * np.pi
        cos_like = np.cos(phase)  # +1 at noon, -1 at midnight

        night_floor = 0.15  # nighttime density relative to noon peak
        Nm_scale = night_floor + (1.0 - night_floor) * max(cos_like, -0.2) \
            if cos_like > -0.2 else night_floor
        # Smooth version (avoid the sharp kink above) using a shifted cosine:
        Nm_scale = night_floor + (1.0 - night_floor) * (0.5 * (1.0 + cos_like)) ** 1.5
        Nm_scale = float(np.clip(Nm_scale, night_floor, 1.0))

        hm_shift = -20.0 * cos_like  # layer rises at night, lowers at noon
        return Nm_scale, hm_shift

    def _solar_factor(self):
        """
        Returns (Nm_scale, hm_scale) from solar_flux (F10.7-like proxy).
        Higher solar activity -> denser, higher ionosphere.
        Normalized so solar_flux=150 gives scale factors of 1.0.
        """
        ref = 150.0
        ratio = self.solar_flux / ref
        Nm_scale = ratio ** 0.7   # peak density grows sub-linearly with flux
        hm_scale = 1.0 + 0.15 * np.log(max(ratio, 1e-3))
        return Nm_scale, hm_scale

    # ------------------------------------------------------------
    # Large-scale horizontal structure (gravity-wave-like undulation)
    # ------------------------------------------------------------
    def _horizontal_modulation(self, Xg, Yg):
        """
        Smooth, quasi-periodic horizontal variation built from a few
        superposed oblique plane waves, returned as a multiplicative
        factor centered around 1.0.
        """
        if self.horizontal_strength <= 0:
            return np.ones_like(Xg)

        k0 = 2.0 * np.pi / self.horizontal_scale
        components = [
            (1.00 * k0, 0.0, 0.0),
            (0.55 * k0, np.pi / 3, 0.6),
            (1.7 * k0, -np.pi / 5, 0.3),
        ]
        mod = np.zeros_like(Xg)
        for k, angle, weight in components:
            kx = k * np.cos(angle)
            ky = k * np.sin(angle)
            mod += weight * np.sin(kx * Xg + ky * Yg)
        mod /= sum(w for _, _, w in components)
        return 1.0 + self.horizontal_strength * mod

    # ------------------------------------------------------------
    # Small-scale turbulence (smoothed multiplicative noise)
    # ------------------------------------------------------------
    def _turbulence_field(self, shape):
        if self.turbulence_strength <= 0:
            return np.ones(shape)

        white = self.rng.normal(0.0, 1.0, size=shape)

        # Smooth with a separable Gaussian kernel sized to
        # turbulence_scale, expressed in grid points per axis.
        sigma_vox = (
            max(self.turbulence_scale / self.dx, 0.5),
            max(self.turbulence_scale / self.dy, 0.5),
            max(self.turbulence_scale / self.dz, 0.5),
        )
        try:
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(white, sigma=sigma_vox, mode="wrap")
        except ImportError:
            smoothed = white  # fallback: unsmoothed noise

        smoothed /= (smoothed.std() + 1e-12)
        return 1.0 + self.turbulence_strength * smoothed

    # ------------------------------------------------------------
    # Build the full 3D density field
    # ------------------------------------------------------------
    def _build_density(self):
        Xg, Yg, Zg = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        # --- Baseline Chapman-layer profile (height-dependent) ---
        Nm0 = 1.2e12   # baseline peak density, m^-3 (~1.2e6 cm^-3, typical F2 peak)
        hm0 = 300.0    # baseline peak height (same length units as dz)
        H = 60.0       # scale height

        diurnal_Nm, diurnal_hm = self._diurnal_factor()
        solar_Nm, solar_hm = self._solar_factor()

        Nm = Nm0 * diurnal_Nm * solar_Nm
        hm = (hm0 + diurnal_hm) * solar_hm

        zp = (Zg - hm) / H
        chapman = Nm * np.exp(0.5 * (1.0 - zp - np.exp(-zp)))

        # Small secondary E-layer bump for realism (daytime only-ish)
        Nm_E = 0.15 * Nm0 * max(diurnal_Nm, 0.2)
        hm_E = 110.0
        H_E = 20.0
        zpE = (Zg - hm_E) / H_E
        e_layer = Nm_E * np.exp(0.5 * (1.0 - zpE - np.exp(-zpE)))

        baseline = chapman + e_layer

        # --- Horizontal structure ---
        horiz = self._horizontal_modulation(Xg, Yg)

        # --- Turbulence ---
        turb = self._turbulence_field(Xg.shape)

        density = baseline * horiz * turb
        density = np.clip(density, 0.0, None)  # density can't be negative
        return density

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def regenerate(self, seed=None):
        """Rebuild the density field (e.g. after changing parameters
        or to draw a new random turbulence realization)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.density = self._build_density()
        self._interp = RegularGridInterpolator(
            (self.x, self.y, self.z),
            self.density,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def electron_density(self, xx, yy, zz):
        """
        Return the (trilinearly) interpolated electron density at
        position (x, y, z), in the same units as the grid was built
        with. Accepts scalars or numpy arrays of matching shape.

        Points outside the grid are clamped to the nearest edge of
        the domain (no extrapolation beyond the boundary values).
        """
        # Convert from meters to km.
        x = xx / 1000
        y = yy / 1000
        z = zz / 1000

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        x_c = np.clip(x, self.x[0], self.x[-1])
        y_c = np.clip(y, self.y[0], self.y[-1])
        z_c = np.clip(z, self.z[0], self.z[-1])

        pts = np.stack([x_c, y_c, z_c], axis=-1)
        result = self._interp(pts)

        if result.shape == ():
            return float(result)

        return float(result[0])

    # Convenience: allow Ionosphere instances to be called directly
    # as N(x, y, z), matching the signature expected by the ray tracer.
    def __call__(self, x, y, z):
        return self.electron_density(x, y, z)


# ----------------------------------------------------------------
# Example usage / quick visualization
# ----------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    iono = Ionosphere(
        Nx=200, Ny=200, Nz=400,
        dx=2.0, dy=2.0, dz=2.0,     # km per grid step
        hour=14.0, solar_flux=180.0,
        turbulence_strength=0.08, turbulence_scale=15.0,
        horizontal_strength=0.12,
        seed=42,
    )

    # Slice through y = 0 for an (x, z) cross-section
    iy = np.argmin(np.abs(iono.y))
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(iono.x, iono.z, iono.density[:, iy, :].T,
                   shading="auto", cmap="viridis")
    plt.colorbar(label="Electron density (m$^{-3}$)")
    plt.xlabel("x (km)")
    plt.ylabel("z (km, height)")
    plt.title(f"Synthetic ionosphere slice (y={iono.y[iy]:.0f} km), "
              f"hour={iono.hour}, F10.7={iono.solar_flux}")

    # Test interpolation at an arbitrary non-grid point
    test_pt = (37.3, -2.2, 281.7)
    Ne = iono(37.3, -2.2, 281.7)

    print(iono.density)

    plt.show()