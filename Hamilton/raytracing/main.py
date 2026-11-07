import numpy as np

from core.constants import q_e, m_e, epsilon_0, mu_0, C
from geophysics.ionosphere import Ionosphere
from core.geometry import (
    WGS84_A, WGS84_B, R_MEAN,
    latlon_to_xyz_m, xyz_m_to_latlon, _ray_ellipsoid_intersect_m,
)

# ---------------------------------------------------------------------------
# Physical / model constants
# ---------------------------------------------------------------------------
H_GRID = 800.0         # ionosphere grid ceiling, km   (above this → Ne = 0)
MARGIN = 1.3           # multiplicative safety factor on angular section size

# Below this X, Ray._haselgrove_physical_rhs treats the medium as exact
# vacuum rather than solving the (numerically degenerate there) dispersion
# relation -- see that method's docstring. 1e-6 means the wave frequency
# is 1000x the local plasma frequency: n^2-1 differs from 0 by ~1e-6,
# undetectably small for any bending this ray tracer resolves. It also
# comfortably covers the floating-point residue ionosphere.py's linear
# interpolation produces near the IRI/E-CHAIM zero floor -- observed in
# practice jumping from ~1e-28 to ~1e-6 within a single ~2 km grid cell,
# i.e. there's normally no continuum of "genuinely small but nonzero" X
# values to weigh against this threshold in the first place.
X_VACUUM_EPS = 1e-6

# Below this |Y| (gyro-to-wave frequency ratio) the exact group index is
# 0/0 numerically; the isotropic limit n_g = 1/n is used there instead
# (docs/implementation_plan.md §3.3).
Y_ISOTROPIC_EPS = 1e-3


# ---------------------------------------------------------------------------
# Ionosphere section sizing
# ---------------------------------------------------------------------------

def build_ionosphere_for_ray(
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
    direction: np.ndarray,
    max_path_m: float,
    # --- background model parameters ---
    dt=None,
    n_ang: int = 15,
    use_echaim: bool = True,
    # --- grid resolution ---
    dr_km: float = 2.0,
    dang_deg: float = 1.0,
) -> Ionosphere:
    """
    Construct an E-CHAIM + IRI-2020 Ionosphere patch that covers the
    region the ray can reach (see ionosphere.py for how the two combine).

    Parameters
    ----------
    lat0_deg, lon0_deg : float
        Geographic start position of the ray (degrees).
    alt0_m : float
        Altitude of the ray's start position above Earth's surface (metres).
    direction : array-like, shape (3,)
        Initial ray direction in geocentric Cartesian coordinates.
        Need not be normalised — normalised internally.
    max_path_m : float
        Maximum ray path length (metres).
    dt : datetime or None
        UTC date and time passed to the background model(s).  Determines
        solar/geomagnetic indices, season, and local solar time per
        longitude.
        Default: datetime(2000, 1, 1, 12, 0, 0).
    use_echaim : bool
        Prefer E-CHAIM over IRI wherever E-CHAIM is valid (AACGM
        geomagnetic latitude >= 45 degrees); IRI everywhere else.
        Default True. See Ionosphere.__init__ (ionosphere.py) and
        ECHAIM/echaim.py.
    n_ang : int
        Number of angular query points per axis for the batch call
        (n_ang × n_ang total).  Larger values improve accuracy at the
        cost of a longer build time (~2 s at n_ang=15).
    dr_km, dang_deg : float
        Radial and angular spacing of the ray-tracer grid (km / degrees).

    Returns
    -------
    Ionosphere
        A sectioned Ionosphere instance whose angular extent covers the
        reachable region with a safety margin.
    """
    from datetime import datetime as _datetime
    if dt is None:
        dt = _datetime(2000, 1, 1, 12, 0, 0)

    d_hat = np.asarray(direction, dtype=float)
    d_hat = d_hat / np.linalg.norm(d_hat)

    P0 = latlon_to_xyz_m(lat0_deg, lon0_deg, alt0_m)
    # top-of-grid ceiling, as a similar ellipsoid offset outward by H_GRID
    # (an approximation -- a true constant-height offset of an ellipsoid
    # isn't itself an ellipsoid -- fine here since this is only an entry-
    # point/patch-sizing heuristic, not a hard physics boundary)
    A_grid = WGS84_A + H_GRID * 1e3
    B_grid = WGS84_B + H_GRID * 1e3
    outside_grid = ((P0[0] ** 2 + P0[1] ** 2) / A_grid ** 2 + P0[2] ** 2 / B_grid ** 2) > 1.0

    # ------------------------------------------------------------------
    # Determine the ionosphere entry point and available path length
    # ------------------------------------------------------------------
    if outside_grid:
        # Ray starts at or above the grid ceiling.
        # Propagate straight-line to find where it enters the ionosphere.
        t_entry = _ray_ellipsoid_intersect_m(P0, d_hat, A_grid, B_grid)

        if t_entry is None or t_entry >= max_path_m:
            # Ray never reaches the ionosphere — build a minimal token patch.
            print("[build_ionosphere] Warning: ray does not intersect the "
                  "ionosphere within max_path_m. Building minimal patch.")
            return Ionosphere(
                center_lat=lat0_deg, center_lon=lon0_deg,
                dtheta_deg=1.0, dphi_deg=1.0,
                h_max=H_GRID, dr=dr_km, dang_deg=dang_deg,
                dt=dt, n_ang=n_ang, use_echaim=use_echaim,
            )

        # Entry point on the grid ceiling
        entry_xyz = P0 + t_entry * d_hat
        path_in_iono_m = max(0.0, max_path_m - t_entry)

        # Section centred on the entry point
        entry_lat, entry_lon, _ = xyz_m_to_latlon(entry_xyz)
        center_lat = entry_lat
        center_lon = entry_lon

        print(f"[build_ionosphere] Ray enters ionosphere at "
              f"lat={center_lat:.2f}°N  lon={center_lon:.2f}°E  "
              f"(t_entry={t_entry/1e3:.1f} km)")

    else:
        # Ray starts inside the ionosphere grid.
        path_in_iono_m = max_path_m
        center_lat     = lat0_deg
        center_lon     = lon0_deg

        print(f"[build_ionosphere] Ray starts inside ionosphere at "
              f"lat={center_lat:.2f}°N  lon={center_lon:.2f}°E  "
              f"alt={alt0_m/1e3:.1f} km")

    # ------------------------------------------------------------------
    # Angular extent: the ray can deviate at most arc = path / R_mean
    # laterally, so the section half-angle = arc (conservative straight-line
    # bound).  Multiply by MARGIN for safety and clamp to [2°, 180°].
    # (R_MEAN, not the exact local WGS84 radius, is fine here -- this is a
    # conservative patch-sizing heuristic, not a precision quantity.)
    # ------------------------------------------------------------------
    arc_deg   = np.rad2deg(path_in_iono_m / R_MEAN)   # straight-line upper bound
    half_deg  = float(np.clip(arc_deg * MARGIN, 2.0, 180.0))
    full_deg  = 2.0 * half_deg

    print(f"[build_ionosphere] Section angular extent: {full_deg:.1f}° × {full_deg:.1f}°  "
          f"(path_in_iono={path_in_iono_m/1e3:.1f} km, margin={MARGIN}×)")

    return Ionosphere(
        center_lat=center_lat,
        center_lon=center_lon,
        dtheta_deg=full_deg,
        dphi_deg=full_deg,
        h_max=H_GRID,
        dr=dr_km,
        dang_deg=dang_deg,
        dt=dt,
        n_ang=n_ang,
        use_echaim=use_echaim,
    )


# ---------------------------------------------------------------------------
# Ray class — Haselgrove equations in geocentric Cartesian metres
# ---------------------------------------------------------------------------

class Ray:
    """
    Haselgrove ray physics in geocentric Cartesian metres.

    All physics methods are shape-agnostic: states of shape (6,) or (..., 6) go in,
    arrays with the same leading shape come out. Phase 1a (docs/implementation_plan.md
    §5): the same code serves one ray and a batch of rays, so batch and scalar tracing
    cannot diverge. The electron-density and field callables must accept arrays
    (Ionosphere and IgrfField do); a scalar callable is broadcast.
    """

    _M_MAG = 7.8e22                            # dipole moment magnitude, A m²
    _I_RAD = np.deg2rad(9.6)                   # tilt from geographic axis

    def __init__(self, omega: float, MODE: int, electron_density, B_field=None):
        """
        B_field : callable(x, y, z) -> (..., 3) array, metres in, Tesla out (e.g.
            igrf.IgrfField.B_vec_xyz_m). If None, the analytic tilted dipole is used:
            fast and offline, but not the real field.
        """
        self.omega = float(omega)
        self.MODE = int(MODE)
        self.electron_density = electron_density
        self._B_field = B_field

        # Dipole moment: mostly -z (south), small -x component.
        self.m = -self._M_MAG * np.array([np.sin(self._I_RAD), 0.0, np.cos(self._I_RAD)])

        self.X_factor = q_e ** 2 / (epsilon_0 * m_e * self.omega ** 2)
        self.Y_factor = q_e / (m_e * self.omega)   # |Y_vec| = Y_factor * |B|

    # ------------------------------------------------------------------
    # Local fields (shape-agnostic)
    # ------------------------------------------------------------------

    def _ne(self, x, y, z):
        """Electron density broadcast to the shape of x (scalar callables allowed)."""
        ne = np.asarray(self.electron_density(x, y, z), dtype=float)
        return ne + np.zeros(np.shape(x))

    def B_vec(self, x, y, z):
        """Magnetic field in Tesla, shape (..., 3)."""
        if self._B_field is not None:
            B = np.asarray(self._B_field(x, y, z), dtype=float)
            return np.broadcast_to(B, np.shape(x) + (3,)).copy()
        return self._dipole_B_vec(x, y, z)

    def _dipole_B_vec(self, x, y, z):
        """Tilted-dipole field (Tesla) at geocentric (x, y, z) in metres."""
        r_vec = np.stack(np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float),
                                             np.asarray(z, float)), axis=-1)
        r = np.maximum(np.linalg.norm(r_vec, axis=-1), 1.0)
        r_hat = r_vec / r[..., None]
        mdotr = r_hat @ self.m
        factor = mu_0 / (4.0 * np.pi)
        return factor * (3.0 * mdotr[..., None] * r_hat - self.m) / (r ** 3)[..., None]

    def Y_vec(self, x, y, z):
        """Dimensionless gyro-vector Y = (q/(m omega)) B, shape (..., 3); |Y| = omega_c/omega."""
        return self.Y_factor * self.B_vec(x, y, z)

    def grad_Ne(self, x, y, z, h: float = 500.0):
        """Central-difference gradient of Ne, m^-4, shape (..., 3). Step 500 m (spec §3.4)."""
        dNdx = (self._ne(x + h, y, z) - self._ne(x - h, y, z)) / (2.0 * h)
        dNdy = (self._ne(x, y + h, z) - self._ne(x, y - h, z)) / (2.0 * h)
        dNdz = (self._ne(x, y, z + h) - self._ne(x, y, z - h)) / (2.0 * h)
        return np.stack([dNdx, dNdy, dNdz], axis=-1)

    def _grad_Y_vec(self, x, y, z, h: float = 500.0):
        """(dY/dx, dY/dy, dY/dz), each shape (..., 3)."""
        dYdx = (self.Y_vec(x + h, y, z) - self.Y_vec(x - h, y, z)) / (2.0 * h)
        dYdy = (self.Y_vec(x, y + h, z) - self.Y_vec(x, y - h, z)) / (2.0 * h)
        dYdz = (self.Y_vec(x, y, z + h) - self.Y_vec(x, y, z - h)) / (2.0 * h)
        return dYdx, dYdy, dYdz

    # ------------------------------------------------------------------
    # Appleton-Hartree surface in Haselgrove's homogeneous form
    # ------------------------------------------------------------------

    def _appleton_hartree_p(self, X, Y, Ydotu, MODE, allow_complex: bool = False):
        """
        Solve a p^2 + b p + c = 0 for p = n^2 (Haselgrove's homogeneous form),
        MODE = +1 for O, -1 for X. Ydotu = Y_hat . u = sqrt(p) cos(theta).

            a = 1 - X - Y^2
            b = -2 X^2 + X Y^2 Yd^2 - X Y^2 + 4 X + 2 Y^2 - 2
            c = -X^3 + 3 X^2 - X Y^2 Yd^2 + X Y^2 - 3 X - Y^2 + 1

        Checked against the textbook limits (docs/algorithm.md §2.3). With
        allow_complex the discriminant is not clipped and a complex root is returned.
        """
        Yd2 = Ydotu ** 2
        Y2 = Y ** 2
        a = 1.0 - X - Y2
        b = -2.0 * X ** 2 + X * Y2 * Yd2 - X * Y2 + 4.0 * X + 2.0 * Y2 - 2.0
        c = -X ** 3 + 3.0 * X ** 2 - X * Y2 * Yd2 + X * Y2 - 3.0 * X - Y2 + 1.0
        disc = b ** 2 - 4.0 * a * c
        if allow_complex:
            sqrt_disc = np.emath.sqrt(disc)
        else:
            disc = np.maximum(disc, 0.0)
            sqrt_disc = np.sqrt(disc)
        p = (-b + MODE * sqrt_disc) / (2.0 * a)
        return a, b, c, p, disc

    def _dispersion_eps(self, X, Y, Ydotu, MODE):
        """
        Stable companion to _appleton_hartree_p: returns (eps, a, B_eps), eps = p - 1,
        computed without forming p. Substituting p = 1 + eps gives

            a eps^2 + X B_eps eps + X^2 (1 - X) = 0,   B_eps = 2 - 2X + Y^2 Yd^2 - Y^2,

        so eps = X (-B_eps + MODE sqrt(B_eps^2 - 4a(1-X))) / (2a) is X times an O(1)
        bracket (spec §4.2; docs/algorithm.md §2.5).
        """
        Y2 = Y ** 2
        a = 1.0 - X - Y2
        B_eps = 2.0 - 2.0 * X + Y2 * Ydotu ** 2 - Y2
        disc_ratio = np.maximum(B_eps ** 2 - 4.0 * a * (1.0 - X), 0.0)
        eps = X * (-B_eps + MODE * np.sqrt(disc_ratio)) / (2.0 * a)
        return eps, a, B_eps

    def n_angle(self, X, Y, cos_theta, MODE: int, U=1.0):
        """
        Angle-form Appleton-Hartree relation (spec eq. 4.1, Budden 1985) with collisions
        through U = 1 - i nu/omega:

            n^2 = 1 - X / (U - Y_T^2/(2(U-X)) + MODE sqrt(Y_T^4/(4(U-X)^2) + Y_L^2)),

        Y_L = Y cos(theta), Y_T = Y sin(theta). MODE=+1 is O-mode. U sits in the
        denominators next to Y, as in Budden's form. X/(1-iZ) in the lossless quadratic
        is the wrong collision model for B != 0 (docs/implementation_plan.md §3.2).
        """
        Y_L = Y * cos_theta
        Y_T2 = Y * Y * (1.0 - cos_theta * cos_theta)
        UX = U - X
        sq = np.emath.sqrt(Y_T2 * Y_T2 / (4.0 * UX * UX) + Y_L * Y_L)
        return 1.0 - X / (U - Y_T2 / (2.0 * UX) + MODE * sq)

    # ------------------------------------------------------------------
    # Haselgrove right-hand side
    # ------------------------------------------------------------------

    def _fields(self, x, y, z, grad: bool, h: float = 500.0):
        """
        Electron density and gyro-vector at (x, y, z), and, if grad, their central-difference
        gradients (step h). The centre and the six offsets are evaluated in one density call
        and one field call on the stacked stencil, not in seven calls each. Each call is
        dominated by coordinate transforms and interpolator set-up, so this is the main
        saving of Phase 1a. Returns (ne, Y_vec, grads) with grads = (grad_ne, dYdx, dYdy, dYdz)
        or None.
        """
        if not grad:
            return self._ne(x, y, z), self.Y_vec(x, y, z), None
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        zs = np.asarray(z, dtype=float)
        offsets = np.array([[0.0, 0.0, 0.0], [h, 0.0, 0.0], [-h, 0.0, 0.0],
                            [0.0, h, 0.0], [0.0, -h, 0.0], [0.0, 0.0, h], [0.0, 0.0, -h]])
        pad = (1,) * xs.ndim
        X = xs[None] + offsets[:, 0].reshape((7,) + pad)
        Y = ys[None] + offsets[:, 1].reshape((7,) + pad)
        Z = zs[None] + offsets[:, 2].reshape((7,) + pad)
        ne_all = self._ne(X, Y, Z)                              # (7, ...)
        Yv_all = self.Y_factor * self.B_vec(X, Y, Z)            # (7, ..., 3)
        grad_ne = np.stack([ne_all[1] - ne_all[2], ne_all[3] - ne_all[4], ne_all[5] - ne_all[6]],
                           axis=-1) / (2.0 * h)
        dYdx = (Yv_all[1] - Yv_all[2]) / (2.0 * h)
        dYdy = (Yv_all[3] - Yv_all[4]) / (2.0 * h)
        dYdz = (Yv_all[5] - Yv_all[6]) / (2.0 * h)
        return ne_all[0], Yv_all[0], (grad_ne, dYdx, dYdy, dYdz)

    def _haselgrove_physical_rhs(self, tau, state6, _cache=None):
        """
        Right-hand side of Haselgrove's ray equations in (x, y, z, ux, uy, uz).

        Shape-agnostic: state6 of shape (..., 6) returns (..., 6). Plasma branch:

            rdot = J u - K Y,     udot = L X_factor grad(Ne) + (K u + Msc Y)·grad(Y),

        with J, K, L, Msc the derivatives of F = a p^2 + b p + c (see _dispersion_eps).
        Vacuum branch (X <= X_VACUUM_EPS): rdot = u, udot = 0.

        Sign (P0.1, docs/implementation_plan.md §3.1): the plasma branch is multiplied
        by MODE, so the traced ray moves along the physical group velocity for both
        modes. The vacuum branch is not flipped (O and X coincide there).
        """
        state6 = np.asarray(state6, dtype=float)
        x, y, z = state6[..., 0], state6[..., 1], state6[..., 2]
        u = state6[..., 3:6]

        ne, Y_vec, grads = self._fields(x, y, z, grad=True)
        grad_ne, dYdx, dYdy, dYdz = grads
        X = ne * self.X_factor
        vac = X <= X_VACUUM_EPS
        if _cache is not None:
            _cache.update(X=X, Y_vec=Y_vec, grad_ne=grad_ne)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            Y = np.linalg.norm(Y_vec, axis=-1)
            Ysafe = np.where(Y > 0.0, Y, 1.0)
            v = np.where((Y > 0.0)[..., None], Y_vec / Ysafe[..., None], 0.0)
            Ydotu = np.sum(v * u, axis=-1)

            eps, a, B_eps = self._dispersion_eps(X, Y, Ydotu, self.MODE)
            J = 2.0 * X * B_eps + 4.0 * a * eps
            K = -2.0 * X * Y * Ydotu * eps
            L = X * (3.0 * X - 2.0) + eps * (2.0 * X - B_eps + eps)
            Msc = 2.0 * eps * (X + eps)

            KuMY = K[..., None] * u + Msc[..., None] * Y_vec
            r_term = np.stack([np.sum(KuMY * dYdx, -1), np.sum(KuMY * dYdy, -1),
                               np.sum(KuMY * dYdz, -1)], axis=-1)
            rdot = J[..., None] * u - K[..., None] * Y_vec
            udot = (L * self.X_factor)[..., None] * grad_ne + r_term
            plasma = self.MODE * np.concatenate([rdot, udot], axis=-1)

        vacuum = np.concatenate([u, np.zeros_like(u)], axis=-1)
        return np.where(vac[..., None], vacuum, plasma)

    # ------------------------------------------------------------------
    # Finite-difference Jacobian of the physical RHS (tangent map, scalar)
    # ------------------------------------------------------------------

    def _state_jacobian_fd(self, tau: float, state6, rel_step: float = 1e-6, abs_step: float = 1e-3):
        state6 = np.asarray(state6, dtype=float)
        n = state6.size
        Jm = np.zeros((n, n), dtype=float)
        for j in range(n):
            step = abs_step + rel_step * max(1.0, abs(state6[j]))
            d = np.zeros(n)
            d[j] = step
            fp = self._haselgrove_physical_rhs(tau, state6 + d)
            fm = self._haselgrove_physical_rhs(tau, state6 - d)
            Jm[:, j] = (fp - fm) / (2.0 * step)
        return Jm

    # ------------------------------------------------------------------
    # Local quantities built from the same partials as the RHS
    # ------------------------------------------------------------------

    def group_index(self, state6, _cache=None):
        """
        Exact group refractive index n_g = c/|v_g| (spec §2.7; docs/implementation_plan.md §3.3).

        n_g = |2 p G_p + Yd G_Yd + 2 X G_X + Y G_Y| / |grad_u F|, with G = a p^2 + b p + c,
        grad_u F = 2 G_p u + G_Yd v = rdot up to the MODE sign, and the closed-form
        partials of Appendix A. Vacuum: 1. Below Y_ISOTROPIC_EPS the isotropic limit 1/n.
        Shape-agnostic, (..., 6) -> (...).
        """
        state6 = np.asarray(state6, dtype=float)
        x, y, z = state6[..., 0], state6[..., 1], state6[..., 2]
        u = state6[..., 3:6]
        if _cache is not None and "X" in _cache:
            X, Y_vec = _cache["X"], _cache["Y_vec"]
        else:
            ne, Y_vec, _ = self._fields(x, y, z, grad=False)
            X = ne * self.X_factor
        Y = np.linalg.norm(Y_vec, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Ysafe = np.where(Y > 0.0, Y, 1.0)
            v = np.where((Y > 0.0)[..., None], Y_vec / Ysafe[..., None], 0.0)
            Yd = np.sum(v * u, axis=-1)
            # Partials on the dispersion root p = 1 + eps, the same root the RHS uses. At the state's
            # |u|^2 the combination G_p = 2 a p + b is a difference of O(1) terms that is O(X) on the
            # shell, so a 1e-4 off-shell error in |u|^2 is an O(1) error in G_p (group path noise).
            eps, a, B_eps = self._dispersion_eps(X, Y, Yd, self.MODE)
            p = 1.0 + eps
            Y2, Yd2 = Y * Y, Yd * Yd
            G_p = X * B_eps + 2.0 * a * eps                  # = 2 a p + b on the shell, no cancellation
            G_X = -p * p + (-4.0 * X + Y2 * Yd2 - Y2 + 4.0) * p + (-3.0 * X * X + 6.0 * X - Y2 * Yd2 + Y2 - 3.0)
            G_Y = -2.0 * Y * p * p + (2.0 * X * Y * Yd2 - 2.0 * X * Y + 4.0 * Y) * p \
                + (-2.0 * X * Y * Yd2 + 2.0 * X * Y - 2.0 * Y)
            G_Yd = 2.0 * X * Y2 * Yd * eps                   # = 2 X Y^2 Yd (p - 1)
            numerator = 2.0 * p * G_p + Yd * G_Yd + 2.0 * X * G_X + Y * G_Y
            grad_F = 2.0 * G_p[..., None] * u + G_Yd[..., None] * v
            ng = np.abs(numerator) / np.linalg.norm(grad_F, axis=-1)
            iso = 1.0 / np.linalg.norm(u, axis=-1)
        ng = np.where(Y < Y_ISOTROPIC_EPS, iso, ng)
        return np.where(X <= X_VACUUM_EPS, 1.0, ng)

    def phase_sensitivity(self, state6, rdot=None, _cache=None):
        """
        Phase sensitivity g = k0 (dn/dNe)|_{u_hat} cos(alpha_rw)  [rad m^2] (spec eq. 4.2).

        Exact identity, from F(n u_hat, Ne) = 0 at fixed u_hat, with rdot the unflipped
        gradient of F:  g = k0 MODE L X_factor / |rdot|,  L = -dF/dX. The MODE factor is
        there because cos(alpha_rw) uses the physical (sign-corrected) ray direction.
        Isotropic limit: g = -k0 X_factor/(2 n) = -r_e lambda0 / n. Verified against
        finite differences of n(Ne) for both modes (tests/test_phase0.py). Shape-agnostic.
        """
        state6 = np.asarray(state6, dtype=float)
        x, y, z = state6[..., 0], state6[..., 1], state6[..., 2]
        u = state6[..., 3:6]
        if _cache is not None and "X" in _cache:
            X, Y_vec = _cache["X"], _cache["Y_vec"]
        else:
            ne, Y_vec, _ = self._fields(x, y, z, grad=False)
            X = ne * self.X_factor
        Y = np.linalg.norm(Y_vec, axis=-1)
        k0 = self.omega / C
        n = np.linalg.norm(u, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Ysafe = np.where(Y > 0.0, Y, 1.0)
            Yd = np.sum(Y_vec / Ysafe[..., None] * u, axis=-1)
            eps, _, B_eps = self._dispersion_eps(X, Y, Yd, self.MODE)
            L = X * (3.0 * X - 2.0) + eps * (2.0 * X - B_eps + eps)
            if rdot is None:
                rdot = self._haselgrove_physical_rhs(0.0, state6)[..., :3]
            rnorm = np.linalg.norm(rdot, axis=-1)
            g_exact = k0 * self.MODE * L * self.X_factor / rnorm
        g_iso = -k0 * self.X_factor / (2.0 * n)
        g = np.where(Y < Y_ISOTROPIC_EPS, g_iso, g_exact)
        return np.where((X <= X_VACUUM_EPS) | (rnorm == 0.0), 0.0, g)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
 
    import matplotlib.pyplot as plt
    from raytracing.mc_scattering import IrregularitySpectrum, MCScatteringTracer
    from plotting.plotly_3d import plot_ray
 
    # ------------------------------------------------------------------
    # User parameters — edit these            (unchanged from your script)
    # ------------------------------------------------------------------
    from datetime import datetime
    DT           = datetime(2024, 6, 21, 14, 0, 0)  # UTC for IRI-2020
 
    FREQ_HZ      = 8e6     # carrier frequency, Hz
    MODE         = 1       # +1 O-mode, -1 X-mode
    MAX_PATH_M   = 1600e3  # per-ray path budget — enough for a full skip
 
    INIT_LAT_DEG = 57.0
    INIT_LON_DEG = 10.0
    INIT_ALT_M   = 0.0
 
    ELEVATION_DEG = 28.0   # oblique launch → clean single skip
    AZIMUTH_DEG   = 0.0

    NHOPS    = 1            # number of ground bounces to trace through (Phase 1: multi-hop support)

    N_ANG    = 75
    DR_KM    = 2.0
    DANG_DEG = 1.0
 
    # ------------------------------------------------------------------
    # NEW — irregularity / Monte Carlo parameters
    # ------------------------------------------------------------------
    XI_RMS      = 5.0e-3   # RMS fractional dNe/Ne  (1e-3 quiet … 1e-1 spread-F)
    L0_M        = 5.0e3    # outer scale of the irregularity spectrum [m]
    P_INDEX     = 4.0      # 3-D spectral index (in-situ 1-D index + 2)
    AXIAL_RATIO = 10.0     # field-aligned elongation along B
    N_RAYS      = 200      # ensemble size (statistics ~ 1/sqrt(N))
    DS_STEP_M   = 2000.0   # integrator step; resolve background scale height
    SEED        = 42       # reproducibility
 
    # ------------------------------------------------------------------
    # Launch frame and direction                (unchanged from your script)
    # ------------------------------------------------------------------
    _lat_r = np.deg2rad(INIT_LAT_DEG)
    _lon_r = np.deg2rad(INIT_LON_DEG)
 
    up_hat = np.array([
        np.cos(_lat_r) * np.cos(_lon_r),
        np.cos(_lat_r) * np.sin(_lon_r),
        np.sin(_lat_r),
    ])
    north_hat = np.array([
        -np.sin(_lat_r) * np.cos(_lon_r),
        -np.sin(_lat_r) * np.sin(_lon_r),
        np.cos(_lat_r),
    ])
    east_hat = np.array([-np.sin(_lon_r), np.cos(_lon_r), 0.0])
 
    el  = np.deg2rad(ELEVATION_DEG)
    az  = np.deg2rad(AZIMUTH_DEG)
    horiz    = np.cos(az) * north_hat + np.sin(az) * east_hat
    INIT_DIR = np.cos(el) * horiz + np.sin(el) * up_hat
    INIT_DIR /= np.linalg.norm(INIT_DIR)
 
    # ------------------------------------------------------------------
    # Ionosphere patch + Ray                    (unchanged from your script)
    # ------------------------------------------------------------------
    print("=== Building ionosphere section ===")
    iono = build_ionosphere_for_ray(
        lat0_deg=INIT_LAT_DEG, lon0_deg=INIT_LON_DEG, alt0_m=INIT_ALT_M,
        direction=INIT_DIR, max_path_m=MAX_PATH_M,
        dt=DT, n_ang=N_ANG, dr_km=DR_KM, dang_deg=DANG_DEG,
    )
    iono.grid_info()

    print("\n=== Building IGRF magnetic field over the same patch ===")
    from geophysics.igrf import build_igrf_for_ionosphere
    bfield = build_igrf_for_ionosphere(iono)

    omega = 2.0 * np.pi * FREQ_HZ
    ray = Ray(omega=omega, MODE=MODE,
              electron_density=iono.electron_density_xyz_m,
              B_field=bfield.B_vec_xyz_m)
 
    r0 = latlon_to_xyz_m(INIT_LAT_DEG, INIT_LON_DEG, INIT_ALT_M)
    u0 = INIT_DIR.copy()
 
    # ------------------------------------------------------------------
    # NEW — spectrum, tracer, deterministic reference, ensemble
    # ------------------------------------------------------------------
    print("\n=== Monte Carlo scattering setup ===")
    k0 = omega / 2.99792458e8
 
    spec = IrregularitySpectrum(
        xi_rms=XI_RMS, L0_m=L0_M,
        p_index=P_INDEX, axial_ratio=AXIAL_RATIO,
    )
    spec.tabulate(k0=k0)          # Bragg cutoff q_max = 2 k0
 
    mc = MCScatteringTracer(
        ray, spec,
        ds_step_m=DS_STEP_M,
        seed=SEED,
        record_every=5,           # keep every 5th point of each path
    )
 
    print("=== Deterministic reference ray ===")
    det = mc.trace(r0, u0, max_path_m=MAX_PATH_M, scatter=False, nhops=NHOPS)
    print(f"landed={det.landed}  escaped={det.escaped}  hops={det.hops_completed}/{NHOPS}  "
          f"arc length={det.s_total/1e3:.1f} km")
    print(f"  ground_range = {det.ground_range_total_m/1e3:.1f} km   "
          f"phase_path = {det.phase_path_total/1e3:.1f} km   "
          f"group_path = {det.group_path_total/1e3:.1f} km   "
          f"absorption = {det.absorption_db_total:.2f} dB")

    if not det.landed:
        print("Reference ray did not land — raise MAX_PATH_M, lower the "
              "elevation, or lower the frequency before running the ensemble.")
        raise SystemExit(1)

    print(f"\n=== Scattered ensemble ({N_RAYS} rays) ===")
    results = mc.ensemble(r0, u0, n_rays=N_RAYS, max_path_m=MAX_PATH_M,
                          nhops=NHOPS, verbose_every=25)
 
    summ = mc.summarize(results, r0)
 
    print("\n--- ensemble summary ---")
    print(f"  landed rays        : {summ['n_landed']} / {N_RAYS}")
    print(f"  mean ground range  : {summ['range_mean_km']:.1f} km")
    print(f"  down-range spread  : {summ['range_std_km']:.2f} km (1σ)")
    print(f"  cross-range spread : {summ['cross_std_km']:.2f} km (1σ)")
    print(f"  arrival elevation  : {summ['elev_mean_deg']:.2f}° "
          f"± {summ['elev_std_deg']:.2f}°")
    print(f"  path-length spread : {summ['s_std_km']*1e3:.0f} m (1σ)")
    print(f"  group-path spread  : {summ['group_path_std_km']*1e3:.0f} m (1σ)  "
          f"[mean {summ['group_path_mean_km']:.1f} km — the physically correct delay proxy]")
    print(f"  absorption         : {summ['absorption_db_mean']:.2f} "
          f"± {summ['absorption_db_std']:.2f} dB")
    print(f"  mean σ_φ²          : {summ['sigma_phi2_mean']:.2f} rad²")
    print(f"  coherent fraction  : {summ['coherent_fraction']:.3e}   "
          f"(⟨E⟩/E₀ = exp(−σ_φ²/2))")
 
    # ------------------------------------------------------------------
    # NEW — figure: ray fan, landing footprint, AOA, path-length spread
    # ------------------------------------------------------------------
    def _range_alt(path):
        """Great-circle ground range [km] (spherical approx, R_MEAN) and true WGS84 altitude [km]."""
        rr, aa = [], []
        for p in path:
            r = np.linalg.norm(p)
            ang = np.arccos(np.clip(np.dot(p / r, up_hat), -1.0, 1.0))
            rr.append(R_MEAN * ang / 1e3)
            _, _, alt_m = xyz_m_to_latlon(p)
            aa.append(alt_m / 1e3)
        return np.array(rr), np.array(aa)
 
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axC, axD = axes[0]
    axE, axF = axes[1]
 
    fig.suptitle(
        f"MC ray scattering — {FREQ_HZ/1e6:.0f} MHz, el {ELEVATION_DEG:.0f}°, "
        f"ξ_rms={XI_RMS:.0e}, L₀={L0_M/1e3:.0f} km, α={AXIAL_RATIO:.0f}, "
        f"N={N_RAYS}", fontweight="bold")
 
    # (1) side view: deterministic + ensemble fan
    for res in results[::2]:
        rr, aa = _range_alt(res.path)
        axC.plot(rr, aa, color="tab:orange", lw=0.5, alpha=0.25)
    rr, aa = _range_alt(det.path)
    axC.plot(rr, aa, color="tab:blue", lw=2.2, label="deterministic")
    axC.plot([], [], color="tab:orange", lw=1.0, label="scattered ensemble")
    axC.set_xlabel("ground range [km]")
    axC.set_ylabel("altitude [km]")
    axC.set_title("Skip through the irregular ionosphere")
    axC.legend(fontsize=8)
    axC.grid(alpha=0.3)
 
    # (2) landing footprint
    axD.scatter(summ["ranges_km"], summ["crosses_km"], s=12,
                color="tab:orange", alpha=0.7, edgecolors="none")
    rr_det, _ = _range_alt(det.path[-1:])
    axD.scatter([rr_det[0]], [0.0], s=140, marker="*", color="tab:blue",
                zorder=5, label="deterministic")
    axD.set_xlabel("down-range [km]")
    axD.set_ylabel("cross-range [km]")
    axD.set_title(f"Landing footprint  "
                  f"(σ∥={summ['range_std_km']:.1f} km, "
                  f"σ⊥={summ['cross_std_km']:.1f} km)")
    axD.legend(fontsize=8)
    axD.grid(alpha=0.3)
    axD.set_aspect("equal", adjustable="datalim")
 
    # (3) angle-of-arrival histogram
    axE.hist(summ["elevs_deg"], bins=25, color="tab:green", alpha=0.85)
    axE.axvline(summ["elev_mean_deg"], color="k", lw=1.0, ls="--")
    axE.set_xlabel("arrival elevation [deg]")
    axE.set_ylabel("rays")
    axE.set_title(f"Angle-of-arrival spread (σ = {summ['elev_std_deg']:.2f}°)")
    axE.grid(alpha=0.3)
 
    # (4) group-path spread (delay-spread proxy — group path, not geometric
    # length, is the physically correct one: it's the one that maps to
    # actual propagation delay via delay = group_path / c)
    g_rel_m = (summ["group_km"] - summ["group_km"].mean()) * 1e3
    axF.hist(g_rel_m, bins=25, color="tab:purple", alpha=0.85)
    axF.set_xlabel("group path − mean  [m]")
    axF.set_ylabel("rays")
    axF.set_title(f"Group-path (delay) spread (σ = {summ['group_path_std_km']*1e3:.0f} m)  ·  "
                  f"⟨σ_φ²⟩ = {summ['sigma_phi2_mean']:.1f} rad²")
    axF.grid(alpha=0.3)
 
    fig.tight_layout()
    fig.savefig("mc_ensemble.png", dpi=150, facecolor=fig.get_facecolor())
    print("\nfigure saved -> mc_ensemble.png")

    # ------------------------------------------------------------------
    # NEW — interactive true-3D ray scene (Plotly): deterministic ray over
    # a WGS84 ellipsoid patch, with a subsample of the scattered ensemble
    # and ground-bounce markers (see plotting/plotly_3d.py for why this
    # replaced the old lon/lat/alt "box" plot: it distorts a ray's true
    # 3-D shape). Saved as HTML -- open it in any browser to rotate/zoom;
    # not shown automatically (no display assumed in a headless run).
    # ------------------------------------------------------------------
    fig3d = plot_ray(
        results[::5], highlight=det,
        title=f"Ray trajectory — {FREQ_HZ/1e6:.0f} MHz, el {ELEVATION_DEG:.0f}°, "
              f"{NHOPS} hop(s)",
    )
    fig3d.write_html("ray_scene_3d.html")
    print("figure saved -> ray_scene_3d.html  (open in a browser to rotate/zoom)")

    plt.show()