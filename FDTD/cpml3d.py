import numpy as np
from constants import mu_0, epsilon_0

class CPML3D:
    """Convolutional PML for the 3-D Yee grid.

    Profiles are one-dimensional per axis and come in two flavors:
      - integer-node profiles ("n"): used by the E-field updates, whose
        transverse H-derivatives sit at integer positions along the
        derivative axis;
      - half-node profiles ("h"): used by the H-field updates, whose
        E-derivatives sit at half-integer positions.
    This is the staggering the legacy 2-D code ignored (it used a single
    profile for both).

    The b/c recursive-convolution coefficients follow Roden & Gedney:
        b = exp(-(sigma/kappa + alpha) dt / eps0)
        c = sigma (b - 1) / (kappa (sigma + kappa alpha))
    """

    def __init__(self, shape, dt, cell_size, thickness=10,
                 kappa_max=5.0, alpha_max=0.0, grading_order=3,
                 reflection_target=1e-8):
        # alpha_max (the CFS shift) defaults to 0: a nonzero alpha only
        # absorbs frequencies ABOVE the CFS corner omega = alpha/eps0. The
        # legacy code used alpha ~ 0.1 S/m, whose corner (~1e10 rad/s) sits
        # far above VLF/MF waves -- the PML then stretches but does not
        # absorb, trapping fields. If late-time linear growth ever shows up
        # in very long runs, set alpha_max <~ eps0 * omega_min / 3.
        self.shape = shape
        self.thickness = thickness
        self.m = grading_order

        eta0 = np.sqrt(mu_0 / epsilon_0)
        # Optimal sigma uses the natural log (the legacy code used log10).
        self.sigma_max = (-(self.m + 1) * np.log(reflection_target)
                          / (2 * eta0 * thickness * cell_size))
        self.kappa_max = kappa_max
        self.alpha_max = alpha_max

        # Per-axis 1-D profiles, reshaped for broadcasting against (Nx,Ny,Nz).
        self.b_n, self.c_n, self.kappa_n = {}, {}, {}
        self.b_h, self.c_h, self.kappa_h = {}, {}, {}

        for axis, (name, N) in enumerate(zip("xyz", shape)):
            bcast = [1, 1, 1]
            bcast[axis] = N
            for half, (bs, cs, ks) in ((False, (self.b_n, self.c_n, self.kappa_n)),
                                       (True, (self.b_h, self.c_h, self.kappa_h))):
                b, c, kappa = self._profiles_1d(N, dt, half)
                bs[name] = b.reshape(bcast)
                cs[name] = c.reshape(bcast)
                ks[name] = kappa.reshape(bcast)

    def _profiles_1d(self, N, dt, half):
        pos = np.arange(N, dtype=np.float64) + (0.5 if half else 0.0)

        # Normalized depth into the PML (0 at the interface, 1 at the wall).
        d = np.zeros(N)
        left = pos < self.thickness
        d[left] = (self.thickness - pos[left]) / self.thickness
        right = pos > (N - 1 - self.thickness)
        d[right] = (pos[right] - (N - 1 - self.thickness)) / self.thickness
        d = np.clip(d, 0.0, 1.0)

        sigma = self.sigma_max * d**self.m
        kappa = 1.0 + (self.kappa_max - 1.0) * d**self.m
        alpha = self.alpha_max * (1.0 - d)

        b = np.exp(-(sigma / kappa + alpha) * dt / epsilon_0)
        c = np.zeros(N)
        mask = sigma > 0
        c[mask] = (sigma[mask] * (b[mask] - 1.0)
                   / (kappa[mask] * (sigma[mask] + kappa[mask] * alpha[mask])))
        return b, c, kappa

    def allocate_psi(self):
        """The 12 convolution memory fields, keyed by (field, deriv_axis)."""
        psi = {}
        for name in ("Exy", "Exz", "Eyz", "Eyx", "Ezx", "Ezy",
                     "Hxy", "Hxz", "Hyz", "Hyx", "Hzx", "Hzy"):
            psi[name] = np.zeros(self.shape)
        return psi
