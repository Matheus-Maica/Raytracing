"""
channel.py -- stochastic channel parameters of one leg (spec §10, Algorithms 4 and 5).

From the screens of a leg (screens.ScreenSet) to the parameters of the Dana-Wittwer
GPSD at the array end: l_x, l_y, principal axes, tau_o, C_xt, C_yt, <tau>, omega_coh,
alpha, S4 and the Rician amplitudes.

The formulas follow the spec. Where the spec's wording is ambiguous the choice is
noted in the code and in docs/spec_deviations.md.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from core.constants import C
from core.records import ChannelParams
from stochastic.screens import ScreenSet, psi_p, struct_fn

EPS_C = 1.0e-3                                   # eq. (10.6) rescaling tolerance
ZETA = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
WGT = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
NODES = np.array([(za, zb) for za in ZETA for zb in ZETA])   # (9, 2), eq. (10.2)
NODE_W = np.array([wa * wb for wa in WGT for wb in WGT])     # (9,)


@dataclass
class ChannelOptions:
    """Choices the spec leaves open (§10.5, §10.6, §9.1)."""

    wander: str = "full"      # "full": sigma_w^2 = sigma_phi^2 (HiCIRF); "cit": eq. (10.9) option
    T_cit: float = 0.0        # coherent integration time for wander="cit", s
    s4: str = "baseline"      # "baseline": eq. (10.10); "refined": eq. (10.11)
    n_psi: int = 18           # directions for the correlation ellipse, Algorithm 4
    n_phi: int = 1024        # TV-2 is not converged at 256 points (0.06 % change); 1024 is converged          # azimuth points for the refined S4 integral
    eps_C: float = EPS_C


# ---------------------------------------------------------------------------
# Diffuse-part threshold (10.3)
# ---------------------------------------------------------------------------

def D_star(sigma2_total: float) -> float:
    """D*(sigma^2) = -2 ln(e^-1 + e^-sigma^2 (1 - e^-1)); tends to 2 in strong scatter."""
    return float(-2.0 * np.log(np.exp(-1.0) + np.exp(-sigma2_total) * (1.0 - np.exp(-1.0))))


# ---------------------------------------------------------------------------
# Algorithm 4: correlation ellipse at the array end
# ---------------------------------------------------------------------------

def _D_array(scr: ScreenSet, rho: np.ndarray) -> float:
    """D_A(rho, 0) = sum_i D_i(w_i rho) (eq. 10.1 at zero time lag)."""
    s = scr.w[:, None] * rho[None, :]
    return float(np.sum(struct_fn(s, scr.W_inv, scr.sigma2, scr.L0, scr.p)))


def fit_ellipse(scr: ScreenSet, Dstar: float, n_psi: int = 18):
    """
    Returns (lx, ly, x_hat, y_hat, residual, r_m). x_hat is the axis with the
    smaller correlation length; the axis signs are fixed later (see channel_parameters).
    """
    psis = np.arange(n_psi) * np.pi / n_psi
    e = np.stack([np.cos(psis), np.sin(psis)], 1)
    r = np.empty(n_psi)
    for m in range(n_psi):
        def f(lnr, m=m):
            return _D_array(scr, np.exp(lnr) * e[m]) - Dstar
        r[m] = np.exp(brentq(f, np.log(1.0e-2), np.log(1.0e7), xtol=1e-12, rtol=1e-12))
    # least squares for r^-2 = M11 cos^2 + 2 M12 cos sin + M22 sin^2  (Algorithm 4, step 4)
    A = np.stack([np.cos(psis) ** 2, 2.0 * np.cos(psis) * np.sin(psis), np.sin(psis) ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, 1.0 / r**2, rcond=None)
    M = np.array([[coef[0], coef[1]], [coef[1], coef[2]]])
    lam, vec = np.linalg.eigh(M)                 # ascending eigenvalues
    lam_big, lam_small = lam[1], lam[0]          # l_x = lam_big^-1/2 <= l_y = lam_small^-1/2
    lx = lam_big ** -0.5
    ly = lam_small ** -0.5
    x_hat = vec[:, 1]
    y_hat = vec[:, 0]
    fit = A @ coef
    residual = float(np.max(np.abs(r**2 * fit - 1.0)))
    return lx, ly, x_hat, y_hat, residual, r


# ---------------------------------------------------------------------------
# Decorrelation time (10.2, 10.4 step 4)
# ---------------------------------------------------------------------------

def _D_time(scr: ScreenSet, dt: float) -> float:
    """Turbulence-averaged D_A(0, dt) with the 3x3 Gauss-Hermite rule (eq. 10.2)."""
    total = 0.0
    for zn, wgt in zip(NODES, NODE_W):
        vel = scr.v + scr.sigma_v[:, None] * zn[None, :]
        total += wgt * np.sum(struct_fn(-vel * dt, scr.W_inv, scr.sigma2, scr.L0, scr.p))
    return float(total)


def solve_tau_o(scr: ScreenSet, Dstar: float) -> float:
    """tau_o from D_A(0, tau_o) = D* (Brent in ln dt, bracket [1e-6, 1e5] s)."""
    f = lambda ld: _D_time(scr, np.exp(ld)) - Dstar  # noqa: E731
    lo, hi = np.log(1.0e-6), np.log(1.0e5)
    if f(lo) * f(hi) > 0.0:
        raise RuntimeError("no decorrelation time bracketed in [1e-6, 1e5] s")
    return float(np.exp(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)))


# ---------------------------------------------------------------------------
# Secant matrices, apparent velocity (10.4, 10.5)
# ---------------------------------------------------------------------------

def secant_matrices(scr: ScreenSet, lx: float, ly: float):
    """A_i of eq. (10.4) and the typical scales u*_i. Returns (A (N,2,2), u_star (N,))."""
    ell_bar = np.sqrt(lx * ly)
    tr_inv = np.trace(scr.W_inv, axis1=1, axis2=2)
    u_star = scr.w * ell_bar / scr.L0 * np.sqrt(0.5 * tr_inv)
    psi = psi_p(u_star, scr.p)
    # A screen at the far end of the leg (z = Z_L: w = 0, so u* = 0) has no lever arm. Every use of A_i is weighted
    # by w_i or z_eff,i = z_i w_i, both zero there, so its A_i is 0 rather than 0/0 (which made V, <tau> and
    # omega_coh NaN for every leg built by leg_screens.screens_for_leg, whose last screen sits at Z_L).
    ok = u_star > 0.0
    coef = np.where(ok, scr.sigma2 * psi / (np.where(ok, u_star, 1.0) ** 2 * scr.L0**2), 0.0)
    return coef[:, None, None] * scr.W_inv, u_star


def apparent_velocity(scr: ScreenSet, A: np.ndarray) -> np.ndarray:
    """V = M_q^-1 m_q, eq. (10.5), in the array-end transverse frame."""
    Mq = np.einsum("n,nij->ij", scr.w**2, A)
    mq = np.einsum("n,nij,nj->i", scr.w, A, scr.v)
    return np.linalg.solve(Mq, mq)


def space_time(V: np.ndarray, lx: float, ly: float, tau_o: float, x_hat, y_hat, eps_C=EPS_C):
    """C_xt, C_yt of eq. (10.6), rescaled if C^2 > 1 - eps_C. Returns (Cxt, Cyt, clipped)."""
    Cxt = float(V @ x_hat) * tau_o / lx
    Cyt = float(V @ y_hat) * tau_o / ly
    C2 = Cxt * Cxt + Cyt * Cyt
    clipped = False
    if C2 > 1.0 - eps_C:
        f = np.sqrt((1.0 - eps_C) / C2)
        Cxt *= f
        Cyt *= f
        clipped = True
    return Cxt, Cyt, clipped


# ---------------------------------------------------------------------------
# Frequency parameters (10.7, 10.8, 10.9)
# ---------------------------------------------------------------------------

def mean_excess_delay(scr: ScreenSet, A: np.ndarray, k0: float, Dstar: float) -> float:
    """<tau> = (2/D*) (1/(c k0^2)) sum_i z_eff,i tr A_i   (eq. 10.7)."""
    trA = np.trace(A, axis1=1, axis2=2)
    return float(2.0 / Dstar * np.sum(scr.z_eff * trA) / (C * k0**2))


def coherence_bandwidth(lx: float, ly: float, tau_mean: float) -> float:
    """omega_coh of eq. (10.8)."""
    leff2 = np.sqrt(2.0 * lx**4 * ly**4 / (lx**4 + ly**4))
    return float(leff2 / (2.0 * tau_mean) * (1.0 / lx**2 + 1.0 / ly**2))


def wander(scr: ScreenSet, omega0: float, sigma_phi2: float, omega_coh: float,
           options: ChannelOptions):
    """alpha = omega0 / (sigma_w omega_coh), eq. (10.9). Returns (alpha, sigma_w)."""
    if options.wander == "cit":
        if options.T_cit <= 0.0:
            raise ValueError("wander='cit' needs T_cit > 0")
        dis = scr.v * options.T_cit
        sw2 = 0.5 * float(np.sum(struct_fn(dis, scr.W_inv, scr.sigma2, scr.L0, scr.p)))
    elif options.wander == "full":
        sw2 = sigma_phi2
    else:
        raise ValueError(f"unknown wander option {options.wander!r}")
    sigma_w = np.sqrt(sw2)
    return float(omega0 / (sigma_w * omega_coh)), float(sigma_w)


# ---------------------------------------------------------------------------
# Scintillation (10.10 - 10.13)
# ---------------------------------------------------------------------------

def s4_baseline_S2(scr: ScreenSet, sigma_phi2: float, f0: float) -> float:
    """S^2_4,weak of eq. (10.10), with the sigma^2-weighted z-bar and L-bar (see spec note)."""
    wts = scr.sigma2 / np.sum(scr.sigma2)
    z_bar = float(np.sum(wts * scr.z_eff))
    L_bar = float(np.sum(wts * scr.L0))
    lam0 = C / f0
    xi_e = lam0 * z_bar / (2.0 * np.pi * L_bar**2)
    return float(2.0 * sigma_phi2 * (1.0 - np.cos(0.5 * np.arctan(4.0 * xi_e))
                                     / (1.0 + 16.0 * xi_e**2) ** 0.25))


def gp_direct(p: float, beta: float) -> float:
    """
    G_p(beta) = int_0^inf (1+x)^(-p/2) sin^2(beta x) dx   (eq. 10.12).

    Integrated directly on [0, x0], x0 = 50/beta, and with the Fourier-weighted
    tail only beyond x0, so that the small-beta cancellation of
    (1 - cos)/2 is never formed (the pitfall in spec §10.6).
    """
    f = lambda x: (1.0 + x) ** (-0.5 * p)  # noqa: E731
    x0 = 50.0 / beta
    part1, _ = quad(lambda x: f(x) * np.sin(beta * x) ** 2, 0.0, x0, limit=4000)
    tail_const = 0.5 * (1.0 + x0) ** (1.0 - 0.5 * p) / (0.5 * p - 1.0)
    tail_cos, _ = quad(f, x0, np.inf, weight="cos", wvar=2.0 * beta, limlst=200)
    return float(part1 + tail_const - 0.5 * tail_cos)


_GP_TABLES: dict = {}


def gp(p: float, beta):
    """G_p(beta) by log-log interpolation of a tabulated direct integral (cached per p)."""
    key = round(float(p), 9)
    if key not in _GP_TABLES:
        logb = np.linspace(np.log(1.0e-5), np.log(1.0e3), 220)
        vals = np.array([gp_direct(key, float(np.exp(lb))) for lb in logb])
        _GP_TABLES[key] = (logb, np.log(vals))
    logb, logg = _GP_TABLES[key]
    beta = np.asarray(beta, dtype=float)
    lb = np.log(np.maximum(beta, 1e-300))
    out = np.exp(np.interp(lb, logb, logg))
    # Power-law extension below the table: G ~ beta^(p/2 - 1); above it G -> 1/(p-2).
    below = lb < logb[0]
    if np.any(below):
        out = np.where(below, np.exp(logg[0]) * np.exp((0.5 * key - 1.0) * (lb - logb[0])), out)
    above = lb > logb[-1]
    if np.any(above):
        out = np.where(above, 1.0 / (key - 2.0), out)
    return out


def s4_refined_S2(scr: ScreenSet, k0: float, n_phi: int = 256) -> float:
    """
    Refined S^2_4,weak, eq. (10.11): sum over screens of
    (p-2) sigma_i^2 sqrt(det W_i)/pi * int_0^{2pi} G_p(beta)/q dphi,
    with q = e^T W e and beta = r_F^2 / (2 L0^2 q), r_F^2 = z_eff/k0.
    """
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    e = np.stack([np.cos(phi), np.sin(phi)], 1)
    q = np.einsum("mi,nij,mj->nm", e, scr.W, e)
    rF2 = scr.z_eff / k0
    beta = rF2[:, None] / (2.0 * scr.L0[:, None] ** 2 * q)
    G = np.empty_like(q)
    for pv in np.unique(scr.p):
        idx = scr.p == pv
        G[idx] = gp(pv, beta[idx])
    integral = (2.0 * np.pi / n_phi) * np.sum(G / q, axis=1)
    S2_i = (scr.p - 2.0) * scr.sigma2 * np.sqrt(scr.det_W) / np.pi * integral
    return float(np.sum(S2_i))


def rician(S4: float):
    """A_c, A_d of eq. (10.13): A_c^2 + 2 A_d^2 = 1 and S4 is reproduced exactly."""
    Ac = (1.0 - S4**2) ** 0.25
    Ad = np.sqrt(0.5 * (1.0 - np.sqrt(1.0 - S4**2)))
    return float(Ac), float(Ad)


# ---------------------------------------------------------------------------
# Algorithm 5: all parameters of one leg
# ---------------------------------------------------------------------------

def channel_parameters(scr: ScreenSet, f0: float, options: ChannelOptions | None = None) -> ChannelParams:
    """
    Algorithm 5. scr: screens of one leg (already built). f0: carrier [Hz].
    Returns ChannelParams with diagnostics (fit residuals, angular spreads,
    Doppler width, clipping and steep-fraction flags).
    """
    options = options or ChannelOptions()
    omega0 = 2.0 * np.pi * f0
    k0 = omega0 / C

    sigma_phi2 = float(np.sum(scr.sigma2))                                    # step 2
    Dstar = D_star(sigma_phi2)

    lx, ly, x_hat, y_hat, resid, r_m = fit_ellipse(scr, Dstar, options.n_psi)  # step 3
    tau_o = solve_tau_o(scr, Dstar)                                           # step 4

    A, u_star = secant_matrices(scr, lx, ly)                                  # step 5
    V = apparent_velocity(scr, A)
    # Axis signs (spec §10.3): V.x_hat >= 0, and det[x_hat y_hat] = +1 in (t1, t2).
    if V @ x_hat < 0.0:
        x_hat = -x_hat
    y_hat = np.array([-x_hat[1], x_hat[0]])
    Cxt, Cyt, clipped = space_time(V, lx, ly, tau_o, x_hat, y_hat, options.eps_C)

    tau_mean = mean_excess_delay(scr, A, k0, Dstar)                          # step 6
    w_coh = coherence_bandwidth(lx, ly, tau_mean)
    alpha, sigma_w = wander(scr, omega0, sigma_phi2, w_coh, options)

    S2_base = s4_baseline_S2(scr, sigma_phi2, f0)                             # step 7
    S2_ref = s4_refined_S2(scr, k0, options.n_phi)
    S2 = S2_ref if options.s4 == "refined" else S2_base
    if options.s4 not in ("baseline", "refined"):
        raise ValueError(f"unknown s4 option {options.s4!r}")
    S4 = float(np.tanh(np.sqrt(max(S2, 0.0))))
    A_c, A_d = rician(S4)

    steep = scr.n < 0.1
    steep_fraction = float(np.sum(scr.sigma2[steep]) / sigma_phi2) if sigma_phi2 > 0 else 0.0
    if steep_fraction > 0.0:
        warnings.warn(f"{steep_fraction:.1%} of the phase variance lies on steep screens (n < 0.1)")

    diagnostics = {
        "ellipse_residual": resid,
        "r_m": r_m,
        "u_star": u_star,
        "sigma_theta_x": np.sqrt(2.0) / (k0 * lx),
        "sigma_theta_y": np.sqrt(2.0) / (k0 * ly),
        "sigma_doppler_hz": 1.0 / (np.pi * tau_o),
        "omega_coh_over_2pi_hz": w_coh / (2.0 * np.pi),
        "sigma_w_rad": sigma_w,
        "C_clipped": clipped,
        "steep_fraction": steep_fraction,
        "weak_scatter": sigma_phi2 < 5.0,
        "S4_baseline": float(np.tanh(np.sqrt(max(S2_base, 0.0)))),
        "S2_weak_baseline": S2_base,
        "S4_refined": float(np.tanh(np.sqrt(max(S2_ref, 0.0)))),
        "S2_weak_refined": S2_ref,
        "s4_option": options.s4,
        "wander_option": options.wander,
    }
    return ChannelParams(
        sigma_phi2=sigma_phi2, D_star=Dstar, lx=float(lx), ly=float(ly),
        x_hat=np.asarray(x_hat, float), y_hat=np.asarray(y_hat, float),
        tau_o=float(tau_o), V_app=np.asarray(V, float), Cxt=float(Cxt), Cyt=float(Cyt),
        tau_mean=float(tau_mean), w_coh=float(w_coh), alpha=float(alpha), S4=S4,
        A_c=A_c, A_d=A_d, diagnostics=diagnostics,
    )
