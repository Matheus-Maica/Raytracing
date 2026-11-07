"""
Phase 5a tests: the two-way cascade of §12.1. Direct convolution, delay placement, the power and
Doppler-variance additivity of independent legs (§12.2), and the deterministic phase and Doppler.
"""

from __future__ import annotations

import numpy as np

from stochastic import cascade
from core.constants import C


def _direct_conv(a, b, dtau):
    Nk, Nt = a.shape
    out = np.zeros((2 * Nk - 1, Nt), dtype=complex)
    for m in range(Nk):
        for n in range(Nk):
            out[m + n] += a[m] * b[n]
    return out * dtau


def test_fft_convolution_matches_direct_sum():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4))
    b = rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4))
    assert np.allclose(cascade.convolve_delay(a, b, 2e-6), _direct_conv(a, b, 2e-6), atol=1e-14)


def test_delta_in_both_legs_lands_at_the_sum_of_their_delays():
    Nk, Nt = 8, 3
    a = np.zeros((Nk, Nt), complex); a[2] = 1.0
    b = np.zeros((Nk, Nt), complex); b[3] = 1.0
    out = cascade.convolve_delay(a, b, 1.0)
    assert np.argmax(np.abs(out[:, 0])) == 5 and abs(abs(out[5, 0]) - 1.0) < 1e-12


def test_fractional_placement_is_a_sinc_shift():
    Nk = 64
    h = np.zeros((Nk, 1), complex)
    h[20, 0] = 1.0
    shifted = cascade.place_on_grid(h, 0.5)
    k = np.arange(Nk)
    expect = np.sinc(k - 20 + 0.5)                  # h(tau_k + shift): the delta moves to bin 19.5
    assert np.allclose(shifted[10:40, 0], expect[10:40], atol=1e-12)


def test_doppler_variances_and_mean_powers_add_for_independent_legs():
    """§12.2: for independent legs the scattering functions convolve, so the Doppler variance of the
    cascade is the sum of the two leg variances and the mean power is the product of the two powers."""
    rng = np.random.default_rng(7)
    fs, Nt = 10.0, 2 ** 17
    f = np.fft.fftfreq(Nt, d=1.0 / fs)

    def gaussian_process(sigma_hz, power):
        S = np.exp(-0.5 * (f / sigma_hz) ** 2)
        X = (rng.standard_normal(Nt) + 1j * rng.standard_normal(Nt)) * np.sqrt(S)
        x = np.fft.ifft(X)
        return x * np.sqrt(power / np.mean(np.abs(x) ** 2))

    s1, s2 = 0.5, 0.8
    h1 = gaussian_process(s1, 2.0)
    h2 = gaussian_process(s2, 3.0)
    h = cascade.convolve_delay(h1[None, :], h2[None, :], 1.0)[0]        # one bin each: the product
    assert abs(np.mean(np.abs(h)) / np.mean(np.abs(h1 * h2)) - 1.0) < 1e-12
    P = np.abs(np.fft.fft(h - np.mean(h))) ** 2
    var = np.sum(f ** 2 * P) / np.sum(P)
    assert abs(var / (s1 ** 2 + s2 ** 2) - 1.0) < 0.05, (var, s1 ** 2 + s2 ** 2)
    assert abs(np.mean(np.abs(h) ** 2) / (2.0 * 3.0) - 1.0) < 0.05


def test_deterministic_phase_and_doppler_are_applied():
    k0 = 2.0 * np.pi * 8e6 / C
    t = np.arange(4) / 10.0
    a = np.ones((1, 4), complex)
    h, off = cascade.two_way_realisation(a, a, dtau=1.0, t=t, P_out=1000.0, P_ret=500.0,
                                         Pg_out=1100.0, Pg_ret=600.0,
                                         f_out=0.5, f_ret=0.25, f_T=-1.0, k0=k0)
    assert abs(off - 1700.0 / C) < 1e-15
    expect = np.exp(-1j * k0 * 1500.0) * np.exp(2j * np.pi * (-0.25) * t)
    assert np.allclose(h[0], expect, atol=1e-12)
