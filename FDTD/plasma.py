import numpy as np
from constants import epsilon_0, k_B, e_charge

def plasma_frequency_sq(n, q, m):
    """omega_p^2 [rad^2/s^2] for a species of density n, charge q, mass m."""
    return n * q**2 / (epsilon_0 * m)


def coulomb_collision_frequency(n_e, T_e_K, lnLambda=13.0):
    """Electron-ion Coulomb collision frequency [1/s] (NRL formulary).

    nu_ei = 2.91e-6 * n_e[cm^-3] * lnLambda * T_e[eV]^{-3/2}

    The legacy code fed n_e in m^-3 and T in Kelvin straight into this
    formula; the two unit errors (1e6 too large in n, ~1.2e6 too small in
    T^{3/2}) happened to roughly cancel, but both are fixed here.
    """
    n_cm3 = n_e / 1e6
    T_eV = T_e_K * k_B / e_charge
    return 2.91e-6 * n_cm3 * lnLambda / np.maximum(T_eV, 1e-30)**1.5


def cross_matrix(Omega):
    """[Omega]_x such that [Omega]_x @ v = Omega x v. Omega: (...,3)."""
    O = np.zeros(Omega.shape[:-1] + (3, 3), dtype=Omega.dtype)
    Ox, Oy, Oz = Omega[..., 0], Omega[..., 1], Omega[..., 2]
    O[..., 0, 1] = -Oz; O[..., 0, 2] = Oy
    O[..., 1, 0] = Oz;  O[..., 1, 2] = -Ox
    O[..., 2, 0] = -Oy; O[..., 2, 1] = Ox
    return O


def choose_subcycles(nu_max, Omega_mag, dt, target=0.1):
    """Number of J sub-steps M so the sub-step resolves gyration/collisions.

    Only nu and Omega set the *internal* rates of the frozen-E current ODE
    (omega_p is the coupling strength to E, which is frozen during the
    window, so it constrains the Yee dt, not M).
    """
    fastest = max(float(nu_max), float(Omega_mag))
    return max(int(np.ceil(dt * fastest / target)), 1)


def subcycle_matrices(omega_p_sq, nu, Omega, dt, M, dtype=np.float64):
    """Closed-form map for the subcycled magnetized-plasma current update.

    Continuous model (cold electron fluid, J = q n u):
        dJ/dt = eps0 * omega_p^2 * E - nu J + J x Omega
    with Omega = (q/m) B0 the SIGNED gyrovector (q < 0 for electrons -- the
    sign fixes the gyration handedness, i.e. which circular polarization is
    the whistler mode).

    Rewriting J x Omega = -[Omega]_x J gives dJ/dt = -K J + eps0 wp^2 E with
        K = nu I + [Omega]_x.
    Crank-Nicolson over a sub-step dt_c = dt/M:
        A J' = B J + eps0 wp^2 dt_c E,  A = I + (dt_c/2) K,  B = I - (dt_c/2) K
        J' = C J + D E,  C = A^-1 B,  D = A^-1 * (eps0 wp^2 dt_c)
    M sub-steps with E frozen collapse to
        J^{n+1/2} = X J^{n-1/2} + G E^n
        X = C^M                (the legacy code had C^{M-1})
        G = (sum_{k=0}^{M-1} C^k) D   (legacy summed only to M-2)
    J lives on half-integer time steps so E^n sits at the center of the
    sub-cycle window: this makes the scheme the standard time-centered
    leapfrog ADE (stable for omega_p*dt < 2), which the endpoint-J variant
    in the legacy code was not.

    X and C^k are Cayley transforms of a rotation+damping generator, so
    ||C|| <= 1 always: the sub-cycled J update itself is unconditionally
    stable no matter how large nu*dt or Omega*dt are.

    omega_p_sq, nu : (...) grids;  Omega : (3,) or (...,3);  returns X, G
    of shape (...,3,3), computed with O(log M) batched 3x3 multiplies.
    """
    shape = np.shape(omega_p_sq)
    Omega = np.asarray(Omega, dtype=dtype)
    if Omega.shape == (3,):
        Omega = np.broadcast_to(Omega, shape + (3,))

    dt_c = dt / M
    K = cross_matrix(Omega)
    idx = np.arange(3)
    K[..., idx, idx] += np.asarray(nu, dtype=dtype)[..., None]

    I = np.zeros(shape + (3, 3), dtype=dtype)
    I[..., idx, idx] = 1.0

    A = I + (dt_c / 2) * K
    Bm = I - (dt_c / 2) * K
    A_inv = np.linalg.inv(A)
    C = np.matmul(A_inv, Bm)
    D = A_inv * (epsilon_0 * np.asarray(omega_p_sq, dtype=dtype) * dt_c)[..., None, None]
    del A, Bm, K

    # Binary powering: P = C^M and S = sum_{k=0}^{M-1} C^k together.
    res_P = I.copy()
    res_S = np.zeros_like(I)
    base_P = C
    base_S = I
    m = M
    while m > 0:
        if m & 1:
            res_S = res_S + np.matmul(res_P, base_S)
            res_P = np.matmul(res_P, base_P)
        m >>= 1
        if m > 0:
            base_S = base_S + np.matmul(base_P, base_S)
            base_P = np.matmul(base_P, base_P)

    X = res_P
    G = np.matmul(res_S, D)
    return X, G


def _parallel_perp(F, b_hat):
    Fpar = np.sum(F * b_hat, axis=-1, keepdims=True) * b_hat
    return Fpar, F - Fpar


def exact_velocity_push(u, F, nu, Omega, dt):
    """Exact solution over dt of du/dt = F - nu u + u x Omega (frozen F, Omega).

    Split along/perpendicular to Omega to stay well-conditioned as nu -> 0:
      parallel:      u' = u e^{-nu dt} + F * (1 - e^{-nu dt})/nu
      perpendicular: relax toward the drift u_s = (nu F - Omega x F)/(nu^2+O^2)
                     while gyrating: u' = u_s + e^{-nu dt} R(-|O| dt) (u - u_s)
    This is the M -> infinity limit of the sub-cycled scheme, evaluated in
    closed form each step, so the nonlinear mode never needs sub-cycling and
    is unconditionally stable in nu and Omega.

    u, F: (...,3); nu: (...); Omega: (...,3) or (3,).
    """
    shape = u.shape
    Omega = np.broadcast_to(np.asarray(Omega, dtype=u.dtype), shape)
    nu = np.asarray(nu, dtype=u.dtype)[..., None]

    Omag = np.linalg.norm(Omega, axis=-1, keepdims=True)
    b_hat = Omega / np.maximum(Omag, 1e-300)

    decay = np.exp(-nu * dt)
    growth = -np.expm1(-nu * dt) / np.maximum(nu, 1e-300)
    growth = np.where(nu * dt < 1e-12, dt, growth)  # nu -> 0 limit

    u_par, u_perp = _parallel_perp(u, b_hat)
    F_par, F_perp = _parallel_perp(F, b_hat)

    u_par_new = u_par * decay + F_par * growth

    denom = nu**2 + Omag**2
    OxF = np.cross(Omega, F_perp)
    u_s = (nu * F_perp - OxF) / np.maximum(denom, 1e-300)

    # Rodrigues rotation of (u_perp - u_s) about b_hat by angle -|Omega| dt.
    v = u_perp - u_s
    theta = -Omag * dt
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    bxv = np.cross(b_hat, v)
    v_rot = v * cos_t + bxv * sin_t  # v is perp to b_hat, so no parallel term

    # Degenerate cells (nu and Omega both ~0) have no drift state; there the
    # exact solution is plain decay + accumulation, same as the parallel part.
    u_perp_new = np.where(denom > 1e-300,
                          u_s + decay * v_rot,
                          u_perp * decay + F_perp * growth)
    return u_par_new + u_perp_new
