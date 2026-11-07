"""
Phase 5d tests: surface clutter (spec §12.4). Bragg frequency (0.323 Hz at 10 MHz), the spreading normalisation,
the line strength identity 2^6 pi k0^4 S(K_B) = 4 pi B_s G (Barrick's normalisation, user decision 2026-11-07),
the ratio of the two lines (wind direction), the
open-water fraction, the current shift, and the range-Doppler map: the lines appear at +/- f_B and the map
conserves the deposited power (power-sum check).
"""

from __future__ import annotations

import numpy as np

from echoes import clutter
from core.constants import C


def test_bragg_frequency_matches_the_spec_value_at_10_mhz():
    assert abs(clutter.bragg_frequency_hz(10e6) - 0.323) < 5e-4
    lam = C / 10e6
    assert abs(clutter.bragg_frequency_hz(10e6) - np.sqrt(9.80665 / (np.pi * lam))) < 1e-12


def test_spreading_is_normalised_and_peaks_at_the_wind():
    phi = np.linspace(-np.pi, np.pi, 200001)
    G = clutter.spreading_G(phi, 3.0)
    assert abs(np.trapz(G, phi) - 1.0) < 1e-6
    assert np.argmax(G) == np.argmin(np.abs(phi))


def test_line_strength_identity_and_dominant_line_level():
    f0 = 10e6
    k0 = 2.0 * np.pi * f0 / C
    for phi in (0.0, 0.7, 2.0):
        G = float(clutter.spreading_G(np.array([phi]), 3.0)[0])
        lhs = clutter.bragg_line_sigma0(f0, phi, 0.005, 3.0)
        assert abs(lhs - 4.0 * np.pi * 0.005 * G) < 1e-12 * max(1.0, lhs)
        assert abs(lhs - 2.0 ** 6 * np.pi * k0 ** 4 * clutter.wave_saturation_S(2.0 * k0, 0.005, G)) < 1e-18
    dominant_dB = 10.0 * np.log10(clutter.bragg_line_sigma0(f0, 0.0, 0.005, 3.0))
    assert abs(dominant_dB + 14.95) < 0.05                 # 4 pi B_s G_max(s = 3) = -14.95 dB


def test_semi_isotropic_sea_gives_barricks_minus_17_dB():
    """Barrick (1972) eq. (12:23): a fully developed sea with all waves in one half-plane, spread evenly
    (G = 1/pi there), gives sigma0 = 4 B = 0.02 = -17 dB with B = 0.005."""
    f0 = 10e6
    k0 = 2.0 * np.pi * f0 / C
    sigma0 = 2.0 ** 6 * np.pi * k0 ** 4 * clutter.wave_saturation_S(2.0 * k0, 0.005, 1.0 / np.pi)
    assert abs(10.0 * np.log10(sigma0) + 16.99) < 0.01


def test_wind_toward_the_radar_makes_the_approaching_line_dominant():
    lines = dict(clutter.bragg_lines(10e6, 0.0))
    fB = clutter.bragg_frequency_hz(10e6)
    assert lines[fB] > 100.0 * lines[-fB]                  # cardioid: the receding line is nearly absent


def test_open_water_and_ice_and_current_terms():
    assert clutter.open_water_fraction(1.0) == 0.0 and clutter.open_water_fraction(0.25) == 0.75
    assert all(s == 0.0 for _, s in clutter.bragg_lines(10e6, 0.0, c_ice=1.0))
    shift = clutter.current_doppler_shift_hz(1.0, 10e6)
    assert abs(shift - 2.0 / (C / 10e6)) < 1e-12
    lines = clutter.bragg_lines(10e6, 0.0, U_radial=1.0)
    assert abs(lines[0][0] - (clutter.bragg_frequency_hz(10e6) + shift)) < 1e-12


def test_range_doppler_map_shows_the_bragg_lines_and_conserves_power():
    f0 = 10e6
    fB = clutter.bragg_frequency_hz(f0)
    area = 2.0e6                                             # one cell (m^2), illustrative
    comps = []
    for doppler, s0 in clutter.bragg_lines(f0, np.pi / 2):     # crosswind: both lines present
        p = clutter.clutter_cell_power_W(sigma0=s0, area_m2=area, Pt_dBW=60.0, f0=f0, d_T=1.2e6, d_R=1.1e6,
                                         loss_T_dB=20.0, loss_R_dB=20.0)
        comps.append((2.0e-3, doppler, p))
    delays = np.linspace(1.5e-3, 2.5e-3, 41)
    dopp = np.linspace(-1.0, 1.0, 2001)
    P = clutter.range_doppler_map(comps, delays, dopp)
    total = P.sum()
    assert abs(total / sum(p for _, _, p in comps) - 1.0) < 1e-9        # power-sum check
    profile = P.sum(axis=0)
    peaks = np.sort(dopp[np.argsort(profile)[-2:]])
    assert abs(peaks[0] + fB) < 0.01 and abs(peaks[1] - fB) < 0.01      # the two lines at +/- f_B
    assert abs(profile[np.argmin(np.abs(dopp - fB))] / profile[np.argmin(np.abs(dopp + fB))] - 1.0) < 1e-6  # equal lines


def test_land_return_is_a_zero_doppler_line():
    s0 = clutter.land_sigma0_linear(-25.0)
    assert abs(10.0 * np.log10(s0) + 25.0) < 1e-12
    P = clutter.range_doppler_map([(2e-3, 0.0, 1e-12)], np.linspace(1.5e-3, 2.5e-3, 11), np.linspace(-1, 1, 401))
    assert np.argmax(P.sum(axis=0)) == np.argmin(np.abs(np.linspace(-1, 1, 401)))


def test_clutter_realisation_has_the_expected_delay_power_and_doppler_lines():
    """§12.4.1 step 2: the realisation's power per delay bin is the sum of the map, and its Doppler spectrum
    peaks at the Bragg lines (averaged over realisations on the FFT frequency grid)."""
    rng = np.random.default_rng(11)
    Nt, dt = 256, 0.1
    f = np.fft.fftfreq(Nt, d=dt)                               # Doppler bins on the FFT grid
    t = np.arange(Nt) * dt
    fB = clutter.bragg_frequency_hz(10e6)
    P = np.zeros((4, Nt))
    j_plus = np.argmin(np.abs(f - fB))
    j_minus = np.argmin(np.abs(f + fB))
    P[:, j_plus] = 3.0e-12
    P[:, j_minus] = 1.0e-12
    acc_power = np.zeros(4)
    spec = np.zeros(Nt)
    n_real = 200
    for _ in range(n_real):
        h = clutter.clutter_field(P, t, f, rng)               # (Nt, 4)
        acc_power += np.mean(np.abs(h) ** 2, axis=0)
        spec += np.mean(np.abs(np.fft.fft(h, axis=0)) ** 2, axis=1)
    expect = P.sum(axis=1)
    assert np.allclose(acc_power / n_real, expect, rtol=0.1), (acc_power / n_real, expect)
    top = np.sort(np.argsort(spec)[-2:])
    assert set(top) == {j_plus, j_minus}, (top, j_plus, j_minus)
