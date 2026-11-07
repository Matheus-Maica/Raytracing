"""
Phase 2 tests: absorption models (spec §7). Sen-Wyller limits and the non-deviative agreement with
the complex form (§7.6 items 1 and 5), riometer consistency of eq. (7.3) (item 2), frequency scaling
(item 3), merge continuity (§7.2) and the O/X split of the Sen-Wyller branch (§7.5).
"""

from __future__ import annotations

import numpy as np

from raytracing import absorption as ab
from core.constants import C
from core.geometry import latlon_to_xyz_m
from raytracing.main import Ray
from test_records_ground import _layer

KAPPA = ab.KAPPA_PREF
W0 = 2.0 * np.pi * 8e6


def _kappa_ah(Ne, nu_eff, omega):
    """Non-deviative complex-form power coefficient [dB/m]: 4.61e-5 Ne nu/(omega^2 + nu^2)."""
    return KAPPA * Ne * nu_eff / (omega ** 2 + nu_eff ** 2)


def test_sen_wyller_limits_give_nu_eff_2p5_and_1p5():
    """Eq. (7.2): F -> (5/2) x for nu_m << omega' (nu_eff = 5/2 nu_m) and F -> (2/3)/x for the
    opposite limit (nu_eff = 3/2 nu_m). The tabulated F matches the quadrature."""
    assert abs(ab.sw_F(1e-4) / 1e-4 - 2.5) < 1e-5
    assert abs(ab.sw_F(1e4) * 1e4 - 2.0 / 3.0) < 1e-5
    for x in (0.05, 0.7, 3.0):
        assert abs(ab.sw_F(x) / ab.sw_F_exact(x) - 1.0) < 2e-4, x


def test_non_deviative_sen_wyller_matches_complex_form_to_1_percent():
    """§7.6 item 1: with nu_eff = 2.5 nu_m, the Sen-Wyller coefficient agrees with the non-deviative
    complex form for nu_m << omega' (the regime where the rule in §7.2 switches the two)."""
    omega_p = 2.0 * np.pi * 8e6
    Ne = 1.0e11
    for x in (1e-3, 1e-2, 2e-2):      # the difference grows as x^2 (2% at x = 0.05)
        nu_m = x * omega_p
        sw = ab.kappa_sen_wyller(Ne, nu_m, omega_p)
        ah = _kappa_ah(Ne, 2.5 * nu_m, omega_p)
        assert abs(sw / ah - 1.0) < 1e-2, (x, sw / ah)


def test_riometer_density_reproduces_the_vertical_absorption():
    """§7.6 item 2 with eq. (7.3): the density built from A_r and w(h) gives A_r for a vertical path at
    f_r = 30 MHz, with the non-deviative coefficient integrated over height."""
    f_r = 30e6
    omega_r = 2.0 * np.pi * f_r
    h = np.arange(50e3, 150e3, 100.0)                       # m
    dh = 100.0
    w = np.exp(-0.5 * ((h - 87e3) / 8e3) ** 2)
    w /= np.sum(w) * dh                                     # normalised: sum w dh = 1
    nu = ab._default_profile(h / 1e3) * 0.1                 # s^-1, a neutral-atmosphere-like profile
    A_r = 1.0
    Ne = ab.riometer_density(A_r, w, nu, f_r)
    A = np.sum(_kappa_ah(Ne, nu, omega_r)) * dh
    assert abs(A / A_r - 1.0) < 1e-9, A


def test_frequency_scaling_is_minus_two_when_nu_is_small():
    """§7.6 item 3: for a D-region-only profile with nu << omega, A(f) scales as f^-2."""
    h = np.arange(60e3, 110e3, 100.0)
    dh = 100.0
    Ne = 1e9 * np.exp(-0.5 * ((h - 85e3) / 6e3) ** 2)
    nu = 1e5                                                 # s^-1, far below omega at 5-30 MHz
    freqs = np.array([5e6, 10e6, 20e6, 30e6])
    A = np.array([np.sum(_kappa_ah(Ne, nu, 2 * np.pi * f)) * dh for f in freqs])
    slope = np.polyfit(np.log(freqs), np.log(A), 1)[0]
    assert abs(slope + 2.0) < 0.02, slope


def test_merge_density_is_continuous_and_blends_as_specified():
    h = np.array([90e3, 100e3, 110e3, 120e3, 130e3])
    model = np.full(h.shape, 3.0)
    background = np.full(h.shape, 7.0)
    tot = ab.merge_density(h, model, background, kind="total_below_merge", h_merge=(100e3, 120e3))
    assert tot[0] == 3.0 and tot[-1] == 7.0                  # model below h1, background above h2
    assert abs(tot[2] - 5.0) < 1e-12                         # cosine blend midpoint
    enh = ab.merge_density(h, model, background, kind="enhancement")
    assert np.all(enh == 10.0)


def test_sen_wyller_branch_absorbs_x_more_than_o_along_b():
    """§7.5: O and X are absorbed differently. Along B (cos theta = 1) omega' = omega(1 +/- Y), so
    the X mode (lower sign) has the larger coefficient at the same point below 100 km."""
    B = np.array([5e-5, 0.0, 0.0])                           # vertical at lat 0, lon 0 (ECEF x-axis)

    def B_field(x, y, z):
        return np.broadcast_to(B, np.shape(x) + (3,)).copy()

    r = latlon_to_xyz_m(0.0, 0.0, 90e3)
    u = np.array([1.0, 0.0, 0.0])
    model = ab.ProfileModel(nu_profile=lambda h: 3e6 * np.ones_like(h), nu_kind="nu_m")
    alpha = {}
    for MODE in (+1, -1):
        ray = Ray(2.0 * np.pi * 8e6, MODE, _layer(h0=90e3, peak=5e10, sigma=10e3), B_field=B_field)
        alpha[MODE] = float(ab.absorption_rate(ray, r[None], u[None], model=model)[0])
    assert alpha[+1] > 0.0 and alpha[-1] > 1.2 * alpha[+1], alpha


def test_secant_law_through_a_thin_layer():
    """§7.6 item 4: oblique absorption through a thin layer equals the vertical value times sec chi at
    the layer height. Straight lines through a 2 km layer at 90 km, isotropic medium (B = 0)."""
    from core.geometry import WGS84_A
    from core.geometry import latlon_to_xyz_m

    def Ne_thin(x, y, z):
        h = np.sqrt(x * x + y * y + z * z) - WGS84_A
        return 1e10 * np.exp(-0.5 * ((h - 90e3) / 2e3) ** 2)

    def B_zero(x, y, z):
        return np.zeros(np.shape(x) + (3,))

    ray = Ray(W0, +1, Ne_thin, B_field=B_zero)
    model = ab.ProfileModel(nu_profile=lambda h: 1e6 * np.ones_like(h), nu_kind="nu_eff")
    r_c = latlon_to_xyz_m(0.0, 0.0, 90e3)
    up = r_c / np.linalg.norm(r_c)
    north = np.array([0.0, 0.0, 1.0])
    s = np.linspace(-60e3, 60e3, 2401)
    I0 = None
    for chi_deg in (0.0, 30.0, 60.0):
        chi = np.radians(chi_deg)
        d = np.cos(chi) * up + np.sin(chi) * north
        pts = r_c[None] + s[:, None] * d[None]
        alpha = ab.absorption_rate(ray, pts, np.tile(d, (s.size, 1)), model=model, rdot=np.tile(d, (s.size, 1)))
        I = np.trapz(alpha, s)
        if I0 is None:
            I0 = I
        assert I0 > 0.0
        assert abs(I / I0 - 1.0 / np.cos(chi)) < 0.01 / np.cos(chi), (chi_deg, I / I0)


def test_attenuation_flag_trips_for_strong_collisions_only():
    """§7.3: the leg is flagged where any step has k0 |Im n| lambda0 > 0.1."""
    from test_records_ground import _layer, _launch
    from raytracing.batch import BatchTracer as BT
    r0, _, up, east = _launch(lat=57.0, lon=10.0, elev=28.0, az=0.0)
    north = np.cross(up, east)
    u0 = np.cos(np.radians(28.0)) * north + np.sin(np.radians(28.0)) * up
    ray = Ray(W0, +1, _layer(peak=1.2e12))
    strong = BT(ray, ds_m=4000.0, absorption_model=ab.ProfileModel(
        nu_profile=lambda h: 3e7 * np.ones_like(h), nu_kind="nu_eff"))
    weak = BT(ray, ds_m=4000.0, absorption_model="default")
    res_s = strong.trace(r0, u0[None], 2500e3, nhops=1, want_group=False, want_absorption=True)
    res_w = weak.trace(r0, u0[None], 2500e3, nhops=1, want_group=False, want_absorption=True)
    assert res_s.att_per_wavelength_max[0] > 0.1, res_s.att_per_wavelength_max
    assert res_w.att_per_wavelength_max[0] < 0.1, res_w.att_per_wavelength_max
