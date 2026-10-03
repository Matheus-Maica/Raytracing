import numpy as np

from constants import q_e, m_e, epsilon_0, mu_0
from visualization import plot_ray
from ionosphere import Ionosphere
from geometry import (
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
    _M_MAG = 7.8e22                            # dipole moment magnitude, A m²
    _I_RAD = np.deg2rad(9.6)                   # tilt from geographic axis

    def __init__(self, omega: float, MODE: int, electron_density, B_field=None):
        """
        B_field : callable(x, y, z) -> np.ndarray or None
            Real magnetic field source, metres in / Tesla out (e.g.
            igrf.IgrfField.B_vec_xyz_m). If None (default), falls back to
            the fast analytic tilted-dipole model below -- useful for quick
            synthetic-density testing without ppigrf/pandas installed, but
            NOT the real field; pass B_field for physically meaningful runs.
        """
        self.omega            = float(omega)
        self.MODE             = int(MODE)
        self.electron_density = electron_density
        self._B_field          = B_field

        # Dipole moment: mostly -z (south), small -x component.
        # Negating [sin I, 0, cos I] gives the correct observed field polarity.
        self.m = -self._M_MAG * np.array([
            np.sin(self._I_RAD),
            0.0,
            np.cos(self._I_RAD),
        ])

        self.X_factor = q_e ** 2 / (epsilon_0 * m_e * self.omega ** 2)
        self.Y_factor = q_e / (m_e * self.omega)   # |Y_vec| = Y_factor * |B|

    # ------------------------------------------------------------------
    # Magnetic field helpers
    # ------------------------------------------------------------------

    def B_vec(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Geocentric magnetic field vector at (x, y, z) in metres, Tesla.
        Delegates to the real field (B_field=... at construction, e.g.
        IGRF) if one was supplied, else falls back to the analytic dipole.
        """
        if self._B_field is not None:
            return self._B_field(x, y, z)
        return self._dipole_B_vec(x, y, z)

    def _dipole_B_vec(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Geocentric tilted-dipole magnetic field vector at (x, y, z) in
        metres. Returns B in Tesla. Fast, analytic, offline -- but not the
        real field (no crustal/secular terms); see B_vec's docstring.
        """
        factor = mu_0 / (4.0 * np.pi)
        r_vec  = np.array([x, y, z], dtype=float)
        r      = np.linalg.norm(r_vec)
        if r < 1.0:          # guard against origin
            r = 1.0
        r_hat  = r_vec / r
        mdotr  = np.dot(self.m, r_hat)
        return factor * (3.0 * mdotr * r_hat - self.m) / r ** 3

    def Y_vec(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Dimensionless gyro-vector Y_vec = (q_e / (m_e ω)) · B  (Haselgrove notation).
        |Y_vec| = ω_c / ω.
        """
        return self.Y_factor * self.B_vec(x, y, z)

    # ------------------------------------------------------------------
    # Electron-density gradient  (finite difference, step in metres)
    # ------------------------------------------------------------------

    def grad_Ne(self, x: float, y: float, z: float, h: float = 500.0):
        """
        Central-difference gradient of Ne (m⁻³ m⁻¹).
        Step h = 500 m gives a good balance between truncation and
        floating-point error for a grid with dr ~ 2 km (= 2000 m).
        """
        Ne = self.electron_density
        dNdx = (Ne(x + h, y, z) - Ne(x - h, y, z)) / (2.0 * h)
        dNdy = (Ne(x, y + h, z) - Ne(x, y - h, z)) / (2.0 * h)
        dNdz = (Ne(x, y, z + h) - Ne(x, y, z - h)) / (2.0 * h)
        return np.array([dNdx, dNdy, dNdz])

    # ------------------------------------------------------------------
    # Y_vec gradient  (finite difference)
    # ------------------------------------------------------------------

    def _grad_Y_vec(self, x: float, y: float, z: float, h: float = 500.0):
        """
        Returns (dY/dx, dY/dy, dY/dz) where each entry is a 3-vector.
        Shape: (3, 3) — first index is spatial direction, second is Y component.
        """
        dYdx = (self.Y_vec(x + h, y, z) - self.Y_vec(x - h, y, z)) / (2.0 * h)
        dYdy = (self.Y_vec(x, y + h, z) - self.Y_vec(x, y - h, z)) / (2.0 * h)
        dYdz = (self.Y_vec(x, y, z + h) - self.Y_vec(x, y, z - h)) / (2.0 * h)
        return dYdx, dYdy, dYdz

    # ------------------------------------------------------------------
    # Appleton–Hartree refractive-index surface (shared by the lossless
    # ray-tracing RHS below and by absorption.py's complex-X reuse)
    # ------------------------------------------------------------------

    def _appleton_hartree_p(self, X, Y, Ydotu, MODE, allow_complex: bool = False):
        """
        Solve a·p² + b·p + c = 0 for p = n² (Appleton–Hartree), selecting
        the branch via MODE (+1 = O-mode, -1 = X-mode). Here Ydotu is the
        wave-normal vector u (not its unit direction) dotted with the unit
        vector along B, i.e. Ydotu = |u|·cosθ = √p·cosθ — Haselgrove's own
        homogenization of the angle θ so the whole relation is a polynomial
        in u rather than n and θ separately.

        a, b, c below are the *homogeneous* (u-vector, not angle-based)
        form of the general oblique-incidence magneto-ionic dispersion
        relation, re-derived directly from the cold-plasma dielectric
        tensor (sin²θ(Sn²−RL)(n²−P) + cos²θ·P(n²−R)(n²−L) = 0, Stix
        notation) and the substitution cosθ = Ydotu/√p above, then
        matched in sign to a = 1−X−Y². Verified against the textbook
        closed forms at both special angles (perpendicular: O-mode
        n²=1−X, X-mode n²=[(1−X)²−Y²]/(1−X−Y²); field-aligned: the R/L
        whistler modes n²=1−X/(1±Y)) and, end-to-end, against the known
        vertical-incidence reflection height X=1 on a synthetic ramp
        ionosphere. A previous version of this function used a simpler
        but incorrect a·p²+b·p+c that failed all of the above checks —
        including the trivial vacuum limit X=0 ⟹ p=1 for any Y — because
        it wasn't actually equivalent to the oblique-incidence relation
        it was meant to encode.

        allow_complex=False (default): X, Y, Ydotu are real, disc is
        clipped to >= 0 (matches the original inline lossless algebra
        exactly — used for the traced ray itself).
        allow_complex=True: X may be complex (collisionally-damped X for
        absorption); disc is not clipped and a complex sqrt is used, so
        p (and hence n) comes out complex — Im(n) is what absorption.py
        wants.
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
            disc = max(disc, 0.0)
            sqrt_disc = np.sqrt(disc)
        p = (-b + MODE * sqrt_disc) / (2.0 * a)
        return a, b, c, p, disc

    def _dispersion_eps(self, X, Y, Ydotu, MODE):
        """
        Numerically stable companion to _appleton_hartree_p: returns
        (eps, a, B_eps) where eps = p - 1, computed WITHOUT ever forming
        p = 1 + eps directly.

        Why this exists: _haselgrove_physical_rhs's J/K/L/Msc all vanish
        at X=0 (see its docstring) and are built from combinations that
        are, algebraically, proportional to (p-1). Computing p itself via
        the standard quadratic formula and then forming e.g. `2*p - 2`
        loses that (p-1) signal to rounding as soon as it drops below
        float64's ~1e-16 relative precision -- which is routine, not
        exotic: real D-region electron densities of ~1e5-1e6 m^-3 (a
        legitimate, weak but physical density, not a floor artifact)
        already put X below 1e-6, and J computed the naive way is already
        wrong in its leading digit there (verified numerically: at
        X=1e-6, naive J is off by ~8x; by X=1e-8 it's pure noise; below
        ~1e-10 it silently rounds to exactly 0, stalling the ray).

        The fix: substitute p=1+eps into a·p²+b·p+c=0 directly and
        re-simplify BEFORE solving. Every coefficient of the resulting
        quadratic in eps picks up an explicit factor of X (verified
        symbolically): a·eps² + X·B_eps·eps + X²(1-X) = 0, so its own
        discriminant factors as X²·[B_eps²-4a(1-X)], and eps itself comes
        out as X times a well-conditioned O(1) bracket -- never a
        subtraction of two near-equal large numbers. J/K/L/Msc,
        re-expressed in terms of eps the same way, are correspondingly
        clean (J=2X·B_eps+4a·eps, K=-2XY·Ydotu·eps, Msc=2eps(X+eps),
        L=X(3X-2)+eps(2X-B_eps+eps)) -- confirmed to agree with the
        original p-based formulas to 4+ significant figures everywhere
        the latter are still numerically trustworthy (X above ~1e-4),
        and to remain accurate down to X~1e-29 where the naive formula is
        pure noise.
        """
        Y2 = Y ** 2
        a = 1.0 - X - Y2
        B_eps = 2.0 - 2.0 * X + Y2 * Ydotu ** 2 - Y2
        disc_ratio = B_eps ** 2 - 4.0 * a * (1.0 - X)   # = disc/X^2
        disc_ratio = max(disc_ratio, 0.0)
        eps = X * (-B_eps + MODE * np.sqrt(disc_ratio)) / (2.0 * a)
        return eps, a, B_eps

    # ------------------------------------------------------------------
    # Haselgrove RHS — physical state (x, y, z, ux, uy, uz)
    # ------------------------------------------------------------------

    def _haselgrove_physical_rhs(self, tau: float, state6: np.ndarray) -> np.ndarray:
        """
        Right-hand side of Haselgrove's (1954) ray equations.

        State: (x, y, z, ux, uy, uz) all in metres / metres.
        Tau is the independent parameter (ray parameter, units: metres).

        J, K, L, Msc are ∂F/∂u and ∂F/∂r of F(r,u) = a·p²+b·p+c (p=|u|²,
        a/b/c from _appleton_hartree_p), giving the standard Hamiltonian
        ray equations dr/dτ=∂F/∂u, du/dτ=-∂F/∂r for the dispersion surface
        F=0. Re-derived symbolically alongside the corrected a/b/c (see
        _appleton_hartree_p) and cross-checked there against F itself:
        J = ∂F/∂p·2, K = -(1/Y)∂F/∂Ydotu (matching both its appearance in
        rdot=J·u-K·Y_vec and in r_term below), L = -∂F/∂X,
        Msc = -∂F/∂Y/Y + Ydotu·∂F/∂Ydotu/Y² — all four identities verified
        to cancel to exactly zero symbolically before being transcribed
        here.
        """
        x, y, z, ux, uy, uz = state6

        Ne    = self.electron_density(x, y, z)
        X     = Ne * self.X_factor

        u_vec = np.array([ux, uy, uz])

        if X <= X_VACUUM_EPS:
            # Negligible/no plasma: O-mode and X-mode become physically
            # identical as X->0 (both reduce to the plain vacuum wave,
            # n=1, regardless of B) -- F(r,u) has a genuine double root at
            # X=0, so J=K=L=Msc=0 exactly there (verified: F factors as
            # (1-Y²)(p-1)² at X=0, for any Y). That's not a numerical
            # glitch to patch around, it's a correct but degenerate
            # Hamiltonian at a true mode coincidence -- so rather than
            # differentiate through it, integrate the exact closed-form
            # vacuum solution directly: u constant, r advancing along u
            # at speed |u|=n=1.
            #
            # The X_VACUUM_EPS margin (rather than X<=0 exactly) is a
            # numerical necessity, not just a physics convenience: J/K/L/
            # Msc are proportional to X near this degeneracy (see
            # _dispersion_eps), so a fixed-step integrator's dtau=ds/|rdot|
            # compensation swings from a stall (rdot underflowing to
            # exactly 0) to a catastrophic overshoot (dtau exploding to
            # compensate) as X crosses machine precision -- observed in
            # practice: a real IRI/E-CHAIM profile's linear interpolation
            # produces X ~ 1e-29 immediately below the D-layer floor,
            # where |u| blew up to ~1e18 in a single step before this
            # guard was widened. X_VACUUM_EPS=1e-6 is well clear of that
            # regime and still physically negligible (see its definition).
            return np.concatenate([u_vec, np.zeros(3)])

        Y_vec = self.Y_vec(x, y, z)
        Y     = np.linalg.norm(Y_vec)
        # Guard against zero B (shouldn't happen above Earth's surface)
        v     = Y_vec / Y if Y > 0.0 else np.zeros(3)

        Ydotu = np.dot(v, u_vec)

        grad_ne       = self.grad_Ne(x, y, z)
        dYdx, dYdy, dYdz = self._grad_Y_vec(x, y, z)

        # eps = p-1, computed via the numerically stable reformulation
        # (see _dispersion_eps) rather than via _appleton_hartree_p's
        # p directly -- J/K/L/Msc below amplify whatever error is in
        # (p-1), and that error is catastrophic once X drops much below
        # 1e-6 if p itself is what gets differenced.
        eps, a, B_eps = self._dispersion_eps(X, Y, Ydotu, self.MODE)

        # Haselgrove auxiliary scalars (see docstring above)
        J = 2.0*X*B_eps + 4.0*a*eps
        K = -2.0 * X * Y * Ydotu * eps
        L = X*(3.0*X - 2.0) + eps*(2.0*X - B_eps + eps)
        Msc = 2.0 * eps * (X + eps)

        KuMY = K * u_vec + Msc * Y_vec
        r_term = np.array([
            np.dot(KuMY, dYdx),
            np.dot(KuMY, dYdy),
            np.dot(KuMY, dYdz),
        ])

        rdot = J * u_vec - K * Y_vec
        udot = (L * self.X_factor) * grad_ne + r_term

        return np.concatenate([rdot, udot])

    # ------------------------------------------------------------------
    # Finite-difference Jacobian of the physical RHS  (for tangent map)
    # ------------------------------------------------------------------

    def _state_jacobian_fd(
        self,
        tau: float,
        state6: np.ndarray,
        rel_step: float = 1e-6,
        abs_step: float = 1e-3,
    ) -> np.ndarray:
        state6 = np.asarray(state6, dtype=float)
        n  = state6.size
        Jm = np.zeros((n, n), dtype=float)
        for j in range(n):
            step = abs_step + rel_step * max(1.0, abs(state6[j]))
            d    = np.zeros(n)
            d[j] = step
            fp = self._haselgrove_physical_rhs(tau, state6 + d)
            fm = self._haselgrove_physical_rhs(tau, state6 - d)
            Jm[:, j] = (fp - fm) / (2.0 * step)
        return Jm



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
 
    import matplotlib.pyplot as plt
    from mc_scattering import IrregularitySpectrum, MCScatteringTracer
 
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
    from igrf import build_igrf_for_ionosphere
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
    # and ground-bounce markers (see visualization.py for why this
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