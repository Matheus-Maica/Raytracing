"""
screens.py -- irregularity spectrum, phase screens along a ray, structure functions.

Spec §8.2 (anisotropic power-law spectrum), §9 (unfolded path, screens).
Everything is vectorised over screens and SI units are used throughout: metres,
rad, m^-3 for densities, rad m^2 for the phase sensitivity g.

    psi_p(u, p)              normalised phase structure function Psi_p(u), eq (9.4), series (9.5)
    spectrum_norm(p)         N_p of eq (8.1), normalisation (2 pi)^3 convention
    F_of_p(p)                F(p) of eq (9.3)
    screen_matrices(...)     W_i of eq (9.1) and the projections b_B on each screen
    build_screens(...)       ScreenSet: all per-screen quantities of spec §9.3
    struct_fn(s, ...)        D_i(s) of eq (9.4) for one displacement per screen

Conventions. The screen frame (t1, t2) is perpendicular to the local ray; the
unfolded leg has the array end at z = 0 and the far end at z = Z_L. w_i = 1 - z_i/Z_L
and z_eff,i = z_i w_i (spec §9.1). Spectra are normalised with
int Phi d^3q/(2 pi)^3 = sigma_N^2 (spec eq. 8.1).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import gamma, gammaln, kve

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------

def spectrum_norm(p):
    """N_p of eq. (8.1): N_p = 8 pi^{3/2} Gamma(p/2) / Gamma(p/2 - 3/2), valid for p > 3."""
    p = np.asarray(p, dtype=float)
    return 8.0 * np.pi**1.5 * gamma(p / 2.0) / gamma(p / 2.0 - 1.5)


def F_of_p(p):
    """F(p) of eq. (9.3): l_par = L0 F(p) a b / sqrt(det W).  F(4) = 2."""
    p = np.asarray(p, dtype=float)
    return 4.0 * np.sqrt(np.pi) * gamma(p / 2.0) / ((p - 2.0) * gamma(p / 2.0 - 1.5))


def psi_p(u, p):
    """
    Normalised phase structure function Psi_p(u) = 1 - rho_phi(u), eq. (9.4):

        Psi_p(u) = 1 - 2^{1-mu} u^mu K_mu(u) / Gamma(mu),   mu = (p-2)/2.

    For u < 1e-3 the small-argument series (9.5) is used, which avoids the
    cancellation in 1 - rho at small u. Broadcasts u against p (per screen).
    """
    u, p = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(p, dtype=float))
    mu = 0.5 * (p - 2.0)
    small = u < 1.0e-3

    # Exact branch, evaluated on a safe copy so that small entries do not
    # reach the Bessel function at u = 0.
    uu = np.where(small, 1.0, u)
    ln_rho = (1.0 - mu) * np.log(2.0) - gammaln(mu) + mu * np.log(uu) + np.log(kve(mu, uu)) - uu
    exact = 1.0 - np.exp(ln_rho)

    # Series, eq. (9.5), for mu != 1 and mu = 1 (p = 4).
    us = np.where(small, u, 1.0e-3)
    m1 = np.abs(mu - 1.0) < 1.0e-12
    mu_safe = np.where(m1, 1.5, mu)
    gen = (gamma(1.0 - mu_safe) / gamma(1.0 + mu_safe)) * (us / 2.0) ** (2.0 * mu_safe) \
        + us**2 / (4.0 * (mu_safe - 1.0))
    us_log = np.where(us > 0.0, us, 1.0e-300)
    log_term = 0.5 * us**2 * (np.log(2.0 / us_log) - EULER_GAMMA + 0.5)
    series = np.where(m1, log_term, gen)

    out = np.where(small, series, exact)
    return np.where(u > 0.0, out, 0.0)


# ---------------------------------------------------------------------------
# Screen geometry
# ---------------------------------------------------------------------------

def screen_matrices(t1, t2, B_hat, e_sh, a, b):
    """
    Screen geometry matrix W_i of eq. (9.1):

        W_i = a^2 b_B b_B^T + b^2 b_sh b_sh^T + b_3 b_3^T,

    with b_B = (t1.B, t2.B), b_sh = (t1.e_sh, t2.e_sh), b_3 = (t1.e_3, t2.e_3),
    e_3 = B x e_sh. Inputs are (N,3) unit vectors except a, b which are (N,).
    Returns W (N,2,2) and b_B (N,2).
    """
    e3 = np.cross(B_hat, e_sh)
    bB = np.stack([np.sum(t1 * B_hat, -1), np.sum(t2 * B_hat, -1)], -1)
    bS = np.stack([np.sum(t1 * e_sh, -1), np.sum(t2 * e_sh, -1)], -1)
    b3 = np.stack([np.sum(t1 * e3, -1), np.sum(t2 * e3, -1)], -1)
    W = ((a**2)[:, None, None] * bB[:, :, None] * bB[:, None, :]
         + (b**2)[:, None, None] * bS[:, :, None] * bS[:, None, :]
         + b3[:, :, None] * b3[:, None, :])
    return W, bB


def inverse_2x2(W):
    """Batch inverse of symmetric 2x2 matrices (N,2,2) -> (inv (N,2,2), det (N,))."""
    det = W[:, 0, 0] * W[:, 1, 1] - W[:, 0, 1] * W[:, 1, 0]
    inv = np.empty_like(W)
    inv[:, 0, 0] = W[:, 1, 1] / det
    inv[:, 1, 1] = W[:, 0, 0] / det
    inv[:, 0, 1] = -W[:, 0, 1] / det
    inv[:, 1, 0] = inv[:, 0, 1]
    return inv, det


def struct_fn(s, W_inv, sigma2, L0, p):
    """
    D_i(s) = 2 sigma_i^2 Psi_p(u), u = sqrt(s^T W_i^-1 s)/L0   (eq. 9.4).
    s: (N,2) displacement in each screen's frame. Returns (N,).
    """
    q = np.einsum("ni,nij,nj->n", s, W_inv, s)
    u = np.sqrt(np.maximum(q, 0.0)) / L0
    return 2.0 * sigma2 * psi_p(u, p)


# ---------------------------------------------------------------------------
# Screens along a leg
# ---------------------------------------------------------------------------

@dataclass
class ScreenSet:
    """
    All screens of one leg, struct-of-arrays, length N (spec §9.3, Table 11).

    Inputs (set by build_screens): z, ds, g, sigma_N, L0, p, a, b, n, t1, t2,
    B_hat, e_sh, v_d, sigma_v, Z_L.
    Derived (filled by build_screens): w, z_eff, W, W_inv, det_W, ell_par,
    sigma2, v (drift in the screen frame), k-independent.
    """

    z: np.ndarray
    ds: np.ndarray
    g: np.ndarray
    sigma_N: np.ndarray
    L0: np.ndarray
    p: np.ndarray
    a: np.ndarray
    b: np.ndarray
    n: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    B_hat: np.ndarray
    e_sh: np.ndarray
    v_d: np.ndarray
    sigma_v: np.ndarray
    Z_L: float
    w: np.ndarray = None
    z_eff: np.ndarray = None
    W: np.ndarray = None
    W_inv: np.ndarray = None
    det_W: np.ndarray = None
    ell_par: np.ndarray = None
    sigma2: np.ndarray = None
    v: np.ndarray = None

    @property
    def N(self) -> int:
        return int(self.z.size)

    def subset(self, index) -> "ScreenSet":
        """A ScreenSet with the screens selected by index (same Z_L)."""
        kw = {}
        for name in ("z", "ds", "g", "sigma_N", "L0", "p", "a", "b", "n", "t1", "t2",
                     "B_hat", "e_sh", "v_d", "sigma_v"):
            kw[name] = np.asarray(getattr(self, name))[index]
        return build_screens(Z_L=self.Z_L, **kw)


def build_screens(*, z, ds, g, sigma_N, L0, p, a, b, n, t1, t2, B_hat, e_sh, v_d,
                  sigma_v, Z_L) -> ScreenSet:
    """
    Build a ScreenSet and its derived per-screen quantities (spec §9.3).

    z      : distance from the array end (group path) [m]
    ds     : screen thickness [m]
    g      : phase sensitivity, eq. (4.2), [rad m^2] (sign irrelevant)
    sigma_N: rms electron density fluctuation [m^-3] (xi * Ne)
    L0     : outer scale [m] (correlation length convention, spec §8.2)
    p      : 3-D spectral index, 3 < p (clipped to (3.05, 5.95) with a warning)
    a, b   : axial ratios along B and along the sheet axis (>= 1)
    n      : phase refractive index of the local medium (for steep flags)
    t1, t2 : (N,3) transverse frame at each screen
    B_hat  : (N,3) unit field direction
    e_sh   : (N,3) unit sheet axis, orthogonalised against B_hat here
    v_d    : (N,3) drift velocity [m/s], component along B removed here
    sigma_v: (N,) rms turbulent velocity per transverse component [m/s]
    Z_L    : leg group-path length [m]
    """
    arr = lambda x: np.asarray(x, dtype=float)  # noqa: E731
    z, ds, g, sigma_N, L0, a, b, n = map(arr, (z, ds, g, sigma_N, L0, a, b, n))
    p = arr(p)
    t1, t2, B_hat, e_sh, v_d = map(arr, (t1, t2, B_hat, e_sh, v_d))
    sigma_v = arr(sigma_v)
    if np.any(p <= 3.0):
        raise ValueError("spectral index p must exceed 3 (spec eq. 8.1)")
    if np.any(p < 3.05) or np.any(p > 5.95):
        warnings.warn("spectral index clipped to (3.05, 5.95) (spec §8.3)")
        p = np.clip(p, 3.05, 5.95)
    if np.any(a < 1.0) or np.any(b < 1.0):
        raise ValueError("axial ratios a, b must be >= 1")
    if np.any(sigma_N < 0.0):
        raise ValueError("negative density fluctuation")

    # Frame conventions: unit field direction, sheet axis perpendicular to B.
    # The drift is used as given: the spec's Appendix B test vectors are
    # computed with v_d = (-800, 200, 0) m/s, which has a component along B. The
    # "remove the component along B" check of spec §8.3 is applied to model
    # outputs in environment.py, not to these low-level inputs.
    B_hat = B_hat / np.linalg.norm(B_hat, axis=-1, keepdims=True)
    e_sh = e_sh - np.sum(e_sh * B_hat, -1, keepdims=True) * B_hat
    e_sh = e_sh / np.linalg.norm(e_sh, axis=-1, keepdims=True)

    scr = ScreenSet(z=z, ds=ds, g=g, sigma_N=sigma_N, L0=L0, p=p, a=a, b=b, n=n,
                    t1=t1, t2=t2, B_hat=B_hat, e_sh=e_sh, v_d=v_d, sigma_v=sigma_v, Z_L=float(Z_L))
    scr.w = 1.0 - z / Z_L
    scr.z_eff = z * scr.w
    scr.W, _ = screen_matrices(t1, t2, B_hat, e_sh, a, b)
    scr.W_inv, scr.det_W = inverse_2x2(scr.W)
    # eq. (9.3): integral scale of delta Ne along the ray and the screen variance
    scr.ell_par = L0 * F_of_p(p) * a * b / np.sqrt(scr.det_W)
    scr.sigma2 = g**2 * ds * sigma_N**2 * scr.ell_par
    scr.v = np.stack([np.sum(t1 * v_d, -1), np.sum(t2 * v_d, -1)], -1)
    return scr
