"""
Phase 5 (two-way, part 1) tests: target Doppler eq. (6.1) and its monostatic limit, range migration
(§12.3), the radar equation scalings (eq. 6.2), the component's delay, Doppler and phase (§12.3,
eq. 6.4), and the reciprocity diagnostic (eq. 12.3, §12.2).
"""

from __future__ import annotations

import numpy as np

from echoes import twoway
from core.constants import C
from propagation.legs import L_POL_DB
from core.records import LegMode

F0 = 8.0e6
LAM0 = C / F0


def _leg(label, Pg, P, doppler, absorption_dB=10.0, ground=0.0, d_eff=1000e3, mode="O", k_far=(1, 0, 0)):
    return LegMode(site="tx", mode=mode, label=label, n_hops=1, beta=0.5, alpha=0.0,
                   e_site=np.array([1.0, 0, 0]), k_far=np.array(k_far, dtype=float), grazing_far=0.3,
                   Pg=Pg, P=P, delay=Pg / C, doppler=doppler, absorption_dB=absorption_dB,
                   absorption_D_dB=float("nan"), ground_loss_dB=ground, d_eff=d_eff, caustic_flag=False)


def test_monostatic_doppler_is_minus_two_v_over_lambda():
    """Eq. (6.1) monostatic limit: f_T = -2 v . k_i / lambda0, negative for a receding target."""
    v = np.array([300.0, 0.0, 0.0])
    f_T = twoway.target_doppler_hz(v, k_i=np.array([1.0, 0, 0]), s_f=np.array([1.0, 0, 0]), lam0=LAM0)
    assert abs(f_T + 2.0 * 300.0 / LAM0) < 1e-9
    assert f_T < 0.0


def test_bistatic_doppler_uses_both_directions():
    """Bistatic: f_T = -(k_i + s_f) . v / lambda0, the average of the two monostatic-like terms."""
    v = np.array([0.0, 200.0, 0.0])
    k_i = np.array([1.0, 0.0, 0.0])
    s_f = np.array([0.0, 1.0, 0.0])
    f_T = twoway.target_doppler_hz(v, k_i, s_f, LAM0)
    assert abs(f_T + 200.0 / LAM0) < 1e-9


def test_range_migration_over_a_cit_is_six_km_one_way():
    """§12.3: 300 m/s over a 20 s CIT gives 6 km of one-way range (two-way delay drift 2 v / c)."""
    v = 300.0
    f_T = -2.0 * v / LAM0
    tau_dot = twoway.range_migration_rate(f_T, LAM0)
    assert abs(tau_dot - 2.0 * v / C) < 1e-15
    drift_one_way_m = 0.5 * C * tau_dot * 20.0
    assert abs(drift_one_way_m - 6000.0) < 1e-6


def test_radar_equation_scalings():
    """Eq. (6.2): doubling d_T costs 6 dB, doubling sigma gains 3 dB, doubling the wavelength gains 6 dB."""
    base = twoway.radar_power_dBW(40.0, 0.0, 0.0, F0, 1.0, 1000e3, 1000e3, 0.0, 0.0)
    assert abs(twoway.radar_power_dBW(40.0, 0.0, 0.0, F0, 1.0, 2000e3, 1000e3, 0.0, 0.0) - base + 6.0206) < 1e-3
    assert abs(twoway.radar_power_dBW(40.0, 0.0, 0.0, F0, 2.0, 1000e3, 1000e3, 0.0, 0.0) - base - 3.0103) < 1e-3
    assert abs(twoway.radar_power_dBW(40.0, 0.0, 0.0, F0 / 2, 1.0, 1000e3, 1000e3, 0.0, 0.0) - base - 6.0206) < 1e-3


def test_component_delay_doppler_phase_follow_the_legs():
    out = _leg("1F-low-O", Pg=1.2e6, P=1.1e6, doppler=-0.5)
    ret = _leg("1F-low-O", Pg=1.3e6, P=1.25e6, doppler=-0.7)
    v = np.array([0.0, 0.0, 100.0])
    comp = twoway.target_component(out, ret, f0=F0, v_T=v, k_i=np.array([1.0, 0, 0]),
                                   s_f=np.array([1.0, 0, 0]), sigma_T_m2=10.0, Pt_dBW=40.0)
    assert abs(comp.delay - (1.2e6 + 1.3e6) / C) < 1e-15
    f_T = twoway.target_doppler_hz(v, np.array([1.0, 0, 0]), np.array([1.0, 0, 0]), LAM0)
    assert abs(comp.doppler - (-0.5 - 0.7 + f_T)) < 1e-9
    k0 = 2.0 * np.pi / LAM0
    assert abs(comp.phase - (-k0 * (1.1e6 + 1.25e6))) < 1e-6
    assert comp.kind == "target" and comp.power > 0.0


def test_gamma_or_limits_and_classes():
    """Eq. (12.3): zero separation gives Gamma = 1 (reciprocal), large separation gives 0 (independent)."""
    assert abs(twoway.gamma_or([0.0, 0.0]) - 1.0) < 1e-15
    assert twoway.gamma_or([50.0, 50.0]) < 1e-10
    assert twoway.reciprocity_class(0.05) == "independent"
    assert twoway.reciprocity_class(0.95) == "reciprocal"
    assert twoway.reciprocity_class(0.5) == "partial"


def _straight_leg(n=40, step=5e3, y0=0.0):
    """A straight leg along x at transverse position y0 (metres), with a fixed transverse frame."""
    from core.records import RaySamples
    s = np.arange(n) * step
    r = np.stack([s, np.full(n, y0), np.zeros(n)], axis=-1)
    e = np.tile([1.0, 0.0, 0.0], (n, 1))
    t1 = np.tile([0.0, 1.0, 0.0], (n, 1))
    t2 = np.tile([0.0, 0.0, 1.0], (n, 1))
    z = np.zeros(n)
    return RaySamples(r=r, s=s, P=s.copy(), Pg=s.copy(), s_hat=e.copy(), u_hat=e.copy(), n=z + 1,
                      X=z.copy(), B=np.zeros((n, 3)), Ne=z.copy(), grad_Ne=np.zeros((n, 3)), alt=z.copy(),
                      hop=np.zeros(n, dtype=int), dA_dB=z.copy(), absorption_db=z.copy(),
                      vacuum=np.zeros(n, dtype=bool), steep=np.zeros(n, dtype=bool), t1=t1, t2=t2)


def _screens(zs, L0=1e3, sigma2=1.0, p=4.0):
    from stochastic.screens import ScreenSet
    N = zs.size
    W_inv = np.tile(np.eye(2), (N, 1, 1))
    one = np.ones(N)
    return ScreenSet(z=zs, ds=one * 1e3, g=one, sigma_N=one, L0=one * L0, p=one * p, a=one, b=one, n=one,
                     t1=np.tile([0.0, 1.0, 0.0], (N, 1)), t2=np.tile([0.0, 0.0, 1.0], (N, 1)),
                     B_hat=np.array([0.0, 0.0, 1.0]), e_sh=np.array([0.0, 1.0, 0.0]),
                     v_d=np.zeros(3), sigma_v=one, Z_L=1e5, W=np.tile(np.eye(2), (N, 1, 1)), W_inv=W_inv,
                     det_W=one, ell_par=one, sigma2=one * sigma2, v=np.zeros((N, 2)))


def test_gamma_or_is_one_for_coincident_legs_and_ignores_along_track_offsets():
    from echoes.twoway import gamma_or_legs
    out = _straight_leg()
    scr = _screens(np.array([10e3, 60e3, 120e3]))
    assert abs(gamma_or_legs(out, out, scr) - 1.0) < 1e-12
    # the same line, sampled differently: nearest points are on the line, so the offset is 0
    ret = _straight_leg(n=23, step=9e3)
    assert abs(gamma_or_legs(out, ret, scr) - 1.0) < 1e-12


def test_gamma_or_is_small_for_legs_separated_across_the_screens():
    from echoes.twoway import gamma_or_legs
    out = _straight_leg()
    far = _straight_leg(y0=50e3)
    scr = _screens(np.array([10e3, 60e3, 120e3]), sigma2=1.0)
    g = gamma_or_legs(out, far, scr)
    assert g < 0.1, g                                   # structure saturates at 2 sigma^2 per screen
    assert twoway.reciprocity_class(g) == "independent"
