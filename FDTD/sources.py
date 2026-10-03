import numpy as np

# Current-density sources for the 3-D staggered (Yee) grid.
#
# Grid convention (matches grid3d.Grid3D):
#   Jx/Ex[i,j,k] sits at ((i+1/2)dx, j dy, k dz)
#   Jy/Ey[i,j,k] sits at (i dx, (j+1/2)dy, k dz)
#   Jz/Ez[i,j,k] sits at (i dx, j dy, (k+1/2)dz)
#
# Every source implements currents(t) -> (Jx, Jy, Jz) full-grid arrays
# [A/m^2], evaluated at time t (the driver calls it at the half step
# t = (n+1/2) dt so the source is time-centered in the E update).
# Gaussians are only evaluated inside a bounding box around the source.

_HALF = {"x": (0.5, 0.0, 0.0), "y": (0.0, 0.5, 0.0), "z": (0.0, 0.0, 0.5)}


class SourceBase:
    def __init__(self, shape, spacing, center, extent):
        """center, extent in meters; extent = half-size of the bounding box."""
        self.shape = shape
        self.spacing = spacing
        self.center = np.asarray(center, dtype=np.float64)

        self.box = tuple(
            slice(max(int((c - extent) / d) - 1, 0),
                  min(int((c + extent) / d) + 2, N))
            for c, d, N in zip(self.center, spacing, shape)
        )
        self._J = [np.zeros(shape) for _ in range(3)]
        self._coords = {c: self._box_coords(c) for c in "xyz"}

    def _box_coords(self, component):
        off = _HALF[component]
        out = []
        for axis, (sl, d, o) in enumerate(zip(self.box, self.spacing, off)):
            idx = np.arange(sl.start, sl.stop, dtype=np.float64)
            coord = (idx + o) * d
            bshape = [1, 1, 1]
            bshape[axis] = coord.size
            out.append(coord.reshape(bshape))
        return out

    def currents(self, t):
        for J, comp in zip(self._J, "xyz"):
            J[self.box] = self._density(comp, t, *self._coords[comp])
        return self._J

    def _density(self, component, t, X, Y, Z):
        raise NotImplementedError


def _ramp(t, tau):
    return 1.0 - np.exp(-(t / tau) ** 2)


class RotatingCharge(SourceBase):
    """Gaussian ball of charge Q rigidly orbiting in the xy-plane.

    J = rho(r - r0(t)) * v(t) with r0 = center + R(cos wt, sin wt, 0):
    exact charge continuity by construction. Radiates circular polarization
    whose handedness (relative to B0 along +z) selects whistler vs L-mode
    coupling; set omega < 0 to reverse the rotation sense.
    """

    def __init__(self, shape, spacing, center, Q=5e-13, R=None, omega=2*np.pi*1e5,
                 sigma=None, ramp_periods=2.0):
        dx = spacing[0]
        self.sigma = sigma if sigma is not None else 3 * dx
        self.R = R if R is not None else 6 * dx
        self.Q = Q
        self.omega = omega
        self.tau = ramp_periods * 2 * np.pi / abs(omega)
        super().__init__(shape, spacing, center, self.R + 5 * self.sigma)
        # 3-D Gaussian normalization: integral of rho over volume = Q.
        self.norm = Q / ((2 * np.pi) ** 1.5 * self.sigma ** 3)

    def _density(self, component, t, X, Y, Z):
        w, R = self.omega, self.R
        x0 = self.center[0] + R * np.cos(w * t)
        y0 = self.center[1] + R * np.sin(w * t)
        z0 = self.center[2]
        v = {"x": -R * w * np.sin(w * t),
             "y": R * w * np.cos(w * t),
             "z": 0.0}[component]
        if v == 0.0:
            return 0.0
        rho = self.norm * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2)
                                 / (2 * self.sigma ** 2))
        return _ramp(t, self.tau) * rho * v


class HertzianDipole(SourceBase):
    """Point-like current element J = p_hat * I(t) * gaussian(r).

    waveform="cw": ramped sinusoid at `omega`.
    waveform="ricker": Gaussian-derivative pulse centered at t0 (broadband,
    good for looking at dispersion across a band in one run).
    """

    def __init__(self, shape, spacing, center, polarization=(0, 0, 1),
                 amplitude=1e-6, omega=2*np.pi*1e5, sigma=None,
                 waveform="cw", ramp_periods=2.0):
        self.sigma = sigma if sigma is not None else 3 * spacing[0]
        super().__init__(shape, spacing, center, 5 * self.sigma)
        p = np.asarray(polarization, dtype=np.float64)
        self.p = p / np.linalg.norm(p)
        self.amplitude = amplitude
        self.omega = omega
        self.waveform = waveform
        self.tau = ramp_periods * 2 * np.pi / omega
        self.norm = 1.0 / ((2 * np.pi) ** 1.5 * self.sigma ** 3)

    def _I(self, t):
        if self.waveform == "cw":
            return self.amplitude * _ramp(t, self.tau) * np.sin(self.omega * t)
        # Ricker (2nd derivative of a Gaussian), centered at t0 = 3/f.
        f = self.omega / (2 * np.pi)
        a = (np.pi * f * (t - 3.0 / f)) ** 2
        return self.amplitude * (1 - 2 * a) * np.exp(-a)

    def _density(self, component, t, X, Y, Z):
        pc = {"x": self.p[0], "y": self.p[1], "z": self.p[2]}[component]
        if pc == 0.0:
            return 0.0
        g = np.exp(-((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2
                     + (Z - self.center[2]) ** 2) / (2 * self.sigma ** 2))
        return pc * self._I(t) * self.norm * g


class MagneticLoop(SourceBase):
    """Azimuthal ring current in the xy-plane (a magnetic-dipole antenna
    along z -- the classic whistler-wave exciter when B0 || z)."""

    def __init__(self, shape, spacing, center, R=None, amplitude=1e-6,
                 omega=2*np.pi*1e5, sigma=None, ramp_periods=2.0):
        dx = spacing[0]
        self.sigma = sigma if sigma is not None else 2 * dx
        self.R = R if R is not None else 6 * dx
        super().__init__(shape, spacing, center, self.R + 5 * self.sigma)
        self.amplitude = amplitude
        self.omega = omega
        self.tau = ramp_periods * 2 * np.pi / omega

    def _density(self, component, t, X, Y, Z):
        if component == "z":
            return 0.0
        xr, yr, zr = X - self.center[0], Y - self.center[1], Z - self.center[2]
        s = np.sqrt(xr ** 2 + yr ** 2)
        g = np.exp(-((s - self.R) ** 2 + zr ** 2) / (2 * self.sigma ** 2))
        I_t = self.amplitude * _ramp(t, self.tau) * np.sin(self.omega * t)
        if component == "x":
            phi_hat = -yr / np.maximum(s, 1e-30)
        else:
            phi_hat = xr / np.maximum(s, 1e-30)
        return I_t * g * phi_hat


class CurrentSheet(SourceBase):
    """Tapered planar current sheet at x = center_x, polarized in-plane:
    launches a quasi-plane wave along +/- x. Good for scattering studies
    against density structures."""

    def __init__(self, shape, spacing, center, polarization=(0, 0, 1),
                 amplitude=1e-8, omega=2*np.pi*1e5, thickness=None,
                 width_frac=0.6, ramp_periods=2.0):
        self.thickness = thickness if thickness is not None else 2 * spacing[0]
        Ly = shape[1] * spacing[1]
        Lz = shape[2] * spacing[2]
        self.wy = width_frac * Ly / 2
        self.wz = width_frac * Lz / 2
        extent = max(5 * self.thickness, self.wy + 4 * spacing[1], self.wz + 4 * spacing[2])
        super().__init__(shape, spacing, center, extent)
        p = np.asarray(polarization, dtype=np.float64)
        p[0] = 0.0  # sheet current must be tangential
        self.p = p / np.linalg.norm(p)
        self.amplitude = amplitude
        self.omega = omega
        self.tau = ramp_periods * 2 * np.pi / omega

    def _density(self, component, t, X, Y, Z):
        pc = {"x": 0.0, "y": self.p[1], "z": self.p[2]}[component]
        if pc == 0.0:
            return 0.0
        gx = np.exp(-((X - self.center[0]) / self.thickness) ** 2 / 2)
        wy = 0.5 * (1 + np.cos(np.pi * np.clip(np.abs(Y - self.center[1]) / self.wy, 0, 1)))
        wz = 0.5 * (1 + np.cos(np.pi * np.clip(np.abs(Z - self.center[2]) / self.wz, 0, 1)))
        I_t = self.amplitude * _ramp(t, self.tau) * np.sin(self.omega * t)
        return pc * I_t * gx * wy * wz


SOURCES = {
    "rotating_charge": RotatingCharge,
    "dipole": HertzianDipole,
    "loop": MagneticLoop,
    "sheet": CurrentSheet,
}


def make_source(name, shape, spacing, center, **kwargs):
    if name not in SOURCES:
        raise KeyError(f"Unknown source '{name}'. Available: {sorted(SOURCES)}")
    return SOURCES[name](shape, spacing, center, **kwargs)
