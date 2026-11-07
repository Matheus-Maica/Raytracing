"""
Phase 5e tests: receiver noise (spec §12.6). The external factor is an input (ITU-R P.372), so these tests
check the arithmetic, the galactic attenuation by the upward path and the statistics of the samples.
"""

from __future__ import annotations

import numpy as np

from receiver import noise
from raytracing.batch import BatchTracer
from core.geometry import latlon_to_xyz_m
from raytracing.main import Ray
from test_records_ground import _layer

W0 = 2.0 * np.pi * 8e6


def test_thermal_noise_power_and_factor_arithmetic():
    P = noise.noise_power_W(10.0, 3e3)
    assert abs(P / (1.380649e-23 * 290.0 * 3e3 * 10.0) - 1.0) < 1e-12
    assert abs(noise.external_noise_dB(20.0, 20.0, 0.0) - 10.0 * np.log10(200.0)) < 1e-9
    assert abs(noise.external_noise_dB(20.0, 20.0, 1e6) - 20.0) < 1e-6          # galactic term removed


def test_upward_path_absorption_is_zero_without_plasma_and_positive_with_collisions():
    r = latlon_to_xyz_m(60.0, 0.0, 0.0)
    B_zero = lambda x, y, z: np.zeros(np.shape(x) + (3,))                       # noqa: E731
    empty = Ray(W0, +1, lambda x, y, z: np.zeros(np.shape(x)), B_field=B_zero)
    assert noise.upward_path_absorption_dB(empty, r) == 0.0
    lossy = Ray(W0, +1, _layer(h0=90e3, peak=5e10, sigma=10e3), B_field=B_zero)
    from raytracing import absorption as ab
    model = ab.ProfileModel(nu_profile=lambda h: 3e7 * np.ones_like(h), nu_kind="nu_eff")
    assert noise.upward_path_absorption_dB(lossy, r, model=model) > 0.0


def test_noise_samples_have_the_requested_power_and_are_circular():
    rng = np.random.default_rng(3)
    P = 2.5e-16
    n = noise.noise_samples(4, 200000, P, rng)
    assert abs(np.mean(np.abs(n) ** 2) / P - 1.0) < 0.01
    assert abs(np.mean(n ** 2) / P) < 0.01                                      # circular: E[n^2] = 0
    assert abs(np.mean(n[0] * np.conj(n[1]))) / P < 0.01                       # independent elements


def test_galactic_noise_matches_itu_r_p372_table1_curve_e():
    # ITU-R P.372-18 Table 1: Fa = 52.0 - 23.0 log10(f_MHz); at f = 1 MHz (log10 f = 0), Fa = 52.0 exactly.
    assert abs(noise.galactic_noise_itur_p372_dB(1.0) - 52.0) < 1e-9
    assert abs(noise.galactic_noise_itur_p372_dB(10.0) - 29.0) < 1e-9            # 52.0 - 23.0*log10(10)
    assert noise.galactic_noise_itur_p372_dB(20.0) < noise.galactic_noise_itur_p372_dB(10.0)  # falls with f


def test_man_made_noise_matches_itu_r_p372_table1_and_orders_the_categories():
    # at f = 1 MHz (log10 f = 0), Fam = c exactly, for every category in Table 1.
    assert abs(noise.man_made_noise_itur_p372_dB(1.0, "city") - 76.8) < 1e-9
    assert abs(noise.man_made_noise_itur_p372_dB(1.0, "residential") - 72.5) < 1e-9
    assert abs(noise.man_made_noise_itur_p372_dB(1.0, "rural") - 67.2) < 1e-9
    assert abs(noise.man_made_noise_itur_p372_dB(1.0, "quiet_rural") - 53.6) < 1e-9
    # noisier environments must stay noisier across the band: city > residential > rural > quiet_rural
    f = 8.0
    vals = [noise.man_made_noise_itur_p372_dB(f, c) for c in ("city", "residential", "rural", "quiet_rural")]
    assert vals == sorted(vals, reverse=True)
    try:
        noise.man_made_noise_itur_p372_dB(8.0, "not_a_category")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown category should raise")


def test_atmospheric_noise_wrapper_matches_itu_p372_directly():
    from receiver import itu_p372
    FaA_direct, _, _ = itu_p372.atmospheric_noise_dB(40.0, -105.0, 12, 10.0, 0)
    FaA_wrapper = noise.atmospheric_noise_itur_p372_dB(40.0, -105.0, 12, 10.0, 0)
    assert FaA_wrapper == FaA_direct
