"""
Reference test vectors TV-1 .. TV-4 of the spec (Appendix B) and the reference values
printed in Tables 11 and 12.

Inputs are built exactly as Appendix B.1 describes them. The expected values were
copied from the spec's tables; they are not produced by this code, so comparing
against them is a real check of the implementation. Where a printed value has only
four or five significant digits, the tolerance in the tests reflects that.
"""

from __future__ import annotations

import numpy as np

from core.constants import C, epsilon_0, m_e, q_e
from stochastic.screens import build_screens

F0 = 10.0e6                      # carrier, Hz
ZL = 2100.0e3                    # leg group path, m
OMEGA0 = 2.0 * np.pi * F0
R_E = q_e**2 / (4.0 * np.pi * epsilon_0 * m_e * C**2)   # classical electron radius
LAMBDA0 = C / F0
NC = epsilon_0 * m_e * OMEGA0**2 / q_e**2                # critical density, B.1 says 1.2404e12

_dip = np.radians(80.0)
B_HAT = np.array([0.0, np.cos(_dip), -np.sin(_dip)])     # northern hemisphere, dip 80 deg
E_SH = np.array([1.0, 0.0, 0.0])                         # magnetic east


def _frame(beta_deg: float):
    """t1 = east, t2 = t1 x s, s = (0, cos beta, sin beta) (App. B.1)."""
    b = np.radians(beta_deg)
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([0.0, -np.sin(b), np.cos(b)])
    return t1, t2


def _rows():
    """The 14 screens in table order: 0 (E, 420 km), 1..12 (F region), 13 (E, 1680 km)."""
    rows = [dict(z=420e3, beta=12.0, Ne=1.0e11, xi=0.05, L0=3e3, p=3.6, a=30.0, b=1.0,
                 ds=80e3, vd=(-400.0, 0.0, 0.0), sv=200.0)]
    for k in range(12):
        z = 725e3 + k * 650e3 / 11.0
        beta = 8.5 - k * 17.0 / 11.0
        phi = -np.pi / 2.0 + k * np.pi / 11.0
        Ne = 2.0e11 * (1.0 + np.cos(phi))
        rows.append(dict(z=z, beta=beta, Ne=Ne, xi=0.03, L0=10e3, p=4.0, a=10.0, b=3.0,
                         ds=59.1e3, vd=(-800.0, 200.0, 0.0), sv=150.0))
    rows.append(dict(z=1680e3, beta=-12.0, Ne=1.0e11, xi=0.05, L0=3e3, p=3.6, a=30.0, b=1.0,
                     ds=80e3, vd=(-400.0, 0.0, 0.0), sv=200.0))
    return rows


ROWS = _rows()


def screens(indices, xi_scale: float = 1.0, sigma_v_zero: bool = False):
    """ScreenSet for the given table indices. xi_scale multiplies xi (sigma^2 scales as xi^2)."""
    rows = [ROWS[i] for i in indices]
    z = np.array([r["z"] for r in rows])
    Ne = np.array([r["Ne"] for r in rows])
    X = Ne / NC
    n = np.sqrt(1.0 - X)
    g = R_E * LAMBDA0 / n
    sigma_N = np.array([r["xi"] for r in rows]) * xi_scale * Ne
    t1 = np.array([_frame(r["beta"])[0] for r in rows])
    t2 = np.array([_frame(r["beta"])[1] for r in rows])
    v_d = np.array([r["vd"] for r in rows], dtype=float)
    sv = np.array([r["sv"] for r in rows], dtype=float)
    if sigma_v_zero:
        sv = np.zeros_like(sv)
    return build_screens(
        z=z,
        ds=np.array([r["ds"] for r in rows]),
        g=g,
        sigma_N=sigma_N,
        L0=np.array([r["L0"] for r in rows]),
        p=np.array([r["p"] for r in rows]),
        a=np.array([r["a"] for r in rows]),
        b=np.array([r["b"] for r in rows]),
        n=n,
        t1=t1,
        t2=t2,
        B_hat=np.tile(B_HAT, (len(rows), 1)),
        e_sh=np.tile(E_SH, (len(rows), 1)),
        v_d=v_d,
        sigma_v=sv,
        Z_L=ZL,
    )


def case(name: str):
    """ScreenSet for TV-1 .. TV-4 (Appendix B.1)."""
    F_IDX = list(range(1, 13))
    if name == "TV1":
        return screens(F_IDX)
    if name == "TV2":
        return screens(list(range(14)))
    if name == "TV3":
        return screens([6], sigma_v_zero=True)           # table screen 6 alone, frozen
    if name == "TV4":
        return screens(F_IDX, xi_scale=0.01)             # sigma^2 x 1e-4 (weak scatter)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Table 11 (per screen, table order 0..13)
# ---------------------------------------------------------------------------
T11 = {
    "sigma2": [6.4294e1, 3.6219e2, 6.2975e2, 9.6211e2, 1.3133e3, 1.6145e3, 1.7934e3,
               1.8018e3, 1.6374e3, 1.3446e3, 9.9443e2, 6.5714e2, 3.8158e2, 6.9295e1],
    "ell_par_km": [4.14, 20.01, 20.03, 20.06, 20.11, 20.18, 20.26, 20.36, 20.47, 20.59,
                   20.74, 20.90, 21.08, 4.46],
    "W11": [1.000] + [9.000] * 12 + [1.000],
    "W22": [898.905, 99.932, 99.721, 99.366, 98.869, 98.231, 97.454, 96.541, 95.494,
            94.316, 93.010, 91.581, 90.032, 773.843],
    "v1": [-400.0] + [-800.0] * 12 + [-400.0],
    "v2": [0.0, -29.56, -24.22, -18.85, -13.48, -8.09, -2.70, 2.70, 8.09, 13.48, 18.85,
           24.22, 29.56, 0.0],
    "w": [0.8000, 0.6548, 0.6266, 0.5985, 0.5703, 0.5422, 0.5141, 0.4859, 0.4578, 0.4297,
          0.4015, 0.3734, 0.3452, 0.2000],
    "z_eff_km": [336.0, 474.7, 491.3, 504.6, 514.6, 521.3, 524.6, 524.6, 521.3, 514.6,
                 504.6, 491.3, 474.7, 336.0],
}

# ---------------------------------------------------------------------------
# Table 12 (Algorithm 5 outputs). Tolerances are applied in the tests.
# ---------------------------------------------------------------------------
T12 = {
    "TV1": dict(sigma_phi2=13492.1, D_star=2.0000, lx=298.21, ly=979.50, resid=1.2e-5,
                tau_o=0.18482, V=(-1573.24, -2.516), Cxt=0.97501, Cyt=4.75e-4,
                tau_mean=1.8010e-6, w_coh=4.2719e5, alpha=1.2663, alpha_cit=9.929,
                S2_base=96.33, S2_ref=62.49, S4=1.0 - 6e-9, Ac=0.01045, Ad=0.70707,
                sig_thx=1.2964, sig_thy=0.3947, sig_dop=1.7223),
    "TV2": dict(sigma_phi2=13625.7, D_star=2.0000, lx=169.19, ly=894.04, resid=0.134,
                tau_o=0.14450, V=(-1029.81, -2.510), Cxt=0.87953, Cyt=4.06e-4,
                tau_mean=3.8593e-6, w_coh=1.8966e5, alpha=2.8380, alpha_cit=20.82,
                S2_base=99.31, S2_ref=105.98, S4=1.0 - 4e-9, Ac=0.00969, Ad=0.70707,
                sig_thx=2.2851, sig_thy=0.4324, sig_dop=2.2029),
    "TV3": dict(sigma_phi2=1793.37, D_star=2.0000, lx=889.55, ly=2927.19, resid=2.7e-11,
                tau_o=0.57161, V=(-1556.21, -5.247), Cxt=0.99950, Cyt=1.02e-3,
                tau_mean=1.9539e-7, w_coh=3.9365e6, alpha=0.37691, alpha_cit=None,
                S2_base=13.385, S2_ref=8.485, S4=0.99867, Ac=0.22695, Ad=0.68866,
                sig_thx=0.4346, sig_thy=0.1321, sig_dop=0.55686),
    "TV4": dict(sigma_phi2=1.34921, D_star=1.26268, lx=70983.0, ly=232563.0, resid=1.1e-4,
                tau_o=44.361, V=(-1610.02, 1.953), Cxt=0.99950, Cyt=-3.70e-4,
                tau_mean=2.524e-11, w_coh=3.049e10, alpha=1.774e-3, alpha_cit=None,
                S2_base=9.633e-3, S2_ref=6.249e-3, S4=0.09783, Ac=0.99760, Ad=0.04898,
                sig_thx=5.45e-3, sig_thy=1.66e-3, sig_dop=7.18e-3),
}
