"""
Phase 3 gate: the spec's reference test vectors TV-1 .. TV-4 (Appendix B, Tables 11
and 12). Tolerances follow Table 12's caption where it states them; where the
printed value has four significant digits the tolerance is set by the rounding.
"""

from __future__ import annotations

import numpy as np

import tv_cases as tv
from stochastic.channel import ChannelOptions, channel_parameters

TOL_REL_STRICT = 1e-4


def _rel(a, b):
    return abs(a - b) / abs(b)


def test_table11_per_screen_quantities():
    """Per-screen values of Table 11 (TV-2 geometry: all 14 screens)."""
    scr = tv.case("TV2")
    T = tv.T11
    sig = np.asarray(scr.sigma2)
    assert np.allclose(sig, T["sigma2"], rtol=3e-4, atol=0.0), (sig, T["sigma2"])
    assert np.allclose(scr.ell_par / 1e3, T["ell_par_km"], rtol=2e-3), scr.ell_par / 1e3
    assert np.allclose(scr.W[:, 0, 0], T["W11"], atol=2e-3), scr.W[:, 0, 0]
    assert np.allclose(scr.W[:, 1, 1], T["W22"], rtol=2e-4), scr.W[:, 1, 1]
    assert np.allclose(scr.w, T["w"], atol=6e-5), scr.w
    assert np.allclose(scr.z_eff / 1e3, T["z_eff_km"], atol=0.06), scr.z_eff / 1e3
    assert np.allclose(scr.v[:, 0], T["v1"], atol=0.01)
    assert np.allclose(scr.v[:, 1], T["v2"], atol=0.02), scr.v[:, 1]


def _check(name, scr, tol_long=5e-3, tol_short=1e-2):
    t = tv.T12[name]
    p = channel_parameters(scr, tv.F0, ChannelOptions())
    d = p.diagnostics
    out = {}

    # closed forms: sigma^2 and D* are fixed by the screens (Table 12 prints 5-6 digits)
    assert _rel(p.sigma_phi2, t["sigma_phi2"]) < 2e-4, ("sigma_phi2", p.sigma_phi2, t["sigma_phi2"])
    assert abs(p.D_star - t["D_star"]) < 2e-4, ("D_star", p.D_star, t["D_star"])

    # correlation lengths and decorrelation time
    assert _rel(p.lx, t["lx"]) < tol_long, ("lx", p.lx, t["lx"])
    assert _rel(p.ly, t["ly"]) < tol_long, ("ly", p.ly, t["ly"])
    assert _rel(p.tau_o, t["tau_o"]) < tol_long, ("tau_o", p.tau_o, t["tau_o"])

    # apparent velocity and space-time coupling
    Vn = np.linalg.norm(t["V"])
    assert np.all(np.abs(p.V_app - np.asarray(t["V"])) < 0.01 * Vn), ("V", p.V_app, t["V"])
    assert abs(p.Cxt - t["Cxt"]) < 1e-2 * abs(t["Cxt"]) + 1e-6, ("Cxt", p.Cxt, t["Cxt"])
    assert abs(p.Cyt - t["Cyt"]) < 1e-2 * abs(t["Cyt"]) + 2e-6, ("Cyt", p.Cyt, t["Cyt"])

    # frequency parameters
    assert _rel(p.tau_mean, t["tau_mean"]) < 1e-2, ("tau_mean", p.tau_mean, t["tau_mean"])
    assert _rel(p.w_coh, t["w_coh"]) < 1e-2, ("w_coh", p.w_coh, t["w_coh"])
    assert _rel(p.alpha, t["alpha"]) < 1e-2, ("alpha", p.alpha, t["alpha"])

    # scintillation: baseline and refined weak-scatter S^2, S4, Rician split
    assert _rel(d["S2_weak_baseline"], t["S2_base"]) < 5e-3, ("S2 base", d["S2_weak_baseline"], t["S2_base"])
    assert _rel(d["S2_weak_refined"], t["S2_ref"]) < 2e-2, ("S2 refined", d["S2_weak_refined"], t["S2_ref"])
    assert abs(d["S4_baseline"] - t["S4"]) < 1e-3 or _rel(d["S4_baseline"], t["S4"]) < 1e-3, \
        ("S4", d["S4_baseline"], t["S4"])
    assert _rel(p.A_c, t["Ac"]) < 1e-2 or abs(p.A_c - t["Ac"]) < 1e-4, ("A_c", p.A_c, t["Ac"])
    assert abs(p.A_d - t["Ad"]) < 1e-3, ("A_d", p.A_d, t["Ad"])

    # angular spreads and Doppler width (degrees for the angles)
    assert _rel(d["sigma_theta_x"] * 180 / np.pi, t["sig_thx"]) < 1e-2, \
        ("sigma_thx", d["sigma_theta_x"] * 180 / np.pi, t["sig_thx"])
    assert _rel(d["sigma_theta_y"] * 180 / np.pi, t["sig_thy"]) < 1e-2, \
        ("sigma_thy", d["sigma_theta_y"] * 180 / np.pi, t["sig_thy"])
    assert _rel(d["sigma_doppler_hz"], t["sig_dop"]) < 1e-2, ("sig_dop", d["sigma_doppler_hz"], t["sig_dop"])
    return p


def test_TV1_F_region_only():
    _check("TV1", tv.case("TV1"))


def test_TV2_F_plus_auroral_E():
    _check("TV2", tv.case("TV2"), tol_long=2e-2)


def test_TV3_frozen_single_screen_and_clip():
    p = _check("TV3", tv.case("TV3"))
    # Single frozen screen: V = v/w exactly, and C_xt is clipped at sqrt(1 - eps_C).
    assert p.diagnostics["C_clipped"]
    assert abs(p.V_app[0] - (-800.0 / 0.51407)) < 0.5


def test_TV4_weak_scatter():
    _check("TV4", tv.case("TV4"), tol_long=2e-2)


def test_alpha_cit_option_matches_table():
    """Wander option of eq. (10.9): alpha with the CIT-limited variance, T_CIT = 4 s."""
    for name in ("TV1", "TV2"):
        t = tv.T12[name]
        p = channel_parameters(tv.case(name), tv.F0,
                               ChannelOptions(wander="cit", T_cit=4.0))
        assert _rel(p.alpha, t["alpha_cit"]) < 1e-2, (name, p.alpha, t["alpha_cit"])
