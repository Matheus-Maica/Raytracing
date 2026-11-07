"""
Phase 4 tests (spec §11): GPSD normalisation, bin-integrated weights, marginals and mean
delay, the space-time correlation of realisations (eq. 11.2), the sign test (§11.7 item 5),
the Rician power and S4 bookkeeping, and element phase/delay offsets.

Monte-Carlo tests use fixed seeds; their tolerances are set by the sample sizes.
"""

from __future__ import annotations

import numpy as np

from stochastic import gpsd
import tv_cases as tv
from stochastic.channel import ChannelOptions, channel_parameters
from core.records import ChannelParams

_P = {}


def _tv_params(name: str) -> ChannelParams:
    if name not in _P:
        _P[name] = channel_parameters(tv.case(name), tv.F0, ChannelOptions())
    return _P[name]


def test_gpsd_normalisation_is_two_pi_cubed():
    """int GPSD d(omega_D) d(tau) d^2K = (2 pi)^3 (spec §11.1, property 1).

    The omega_D and tau integrals are translation invariant (their centres move with K),
    so they can be done on a fine grid at K = 0, and the K integral is then exact."""
    P = _tv_params("TV1")
    wD = np.linspace(-12.0, 12.0, 801) * np.sqrt(2.0) / P.tau_o * 4.0
    s_tau = 1.0 / (P.alpha * P.w_coh)
    tau = np.linspace(-12.0, 12.0, 801) * s_tau
    WD, TA = np.meshgrid(wD, tau, indexing="ij")
    vals = gpsd.gpsd(WD, TA, 0.0, 0.0, P)
    inner = np.trapz(np.trapz(vals, tau, axis=1), wD, axis=0)
    # K integral of exp(-(Kx^2 lx^2 + Ky^2 ly^2)/4) is 4 pi /(lx ly)
    value = inner * 4.0 * np.pi / (P.lx * P.ly)
    assert abs(value / (2.0 * np.pi) ** 3 - 1.0) < 1e-4, value / (2.0 * np.pi) ** 3


def test_bin_weights_are_normalised():
    """Q, P_j and T_k sum to one over their domains: captured fraction > 0.99 (spec §11.3)."""
    P = _tv_params("TV1")
    grid = gpsd.default_grid(P, aperture_m=300.0, Nt=128)
    assert grid.Nx >= 16 and grid.Nk >= 8
    frac = gpsd.captured_fraction(P, grid)
    assert frac > 0.99, frac


def test_doppler_and_delay_marginals_match_analytic():
    """Mean and standard deviation of the expected scattering function (eq. 11.1 marginals):
    Doppler std = sqrt(2)/(2 pi tau_o), mean delay = <tau> of ChannelParams (spec §10.5)."""
    P = _tv_params("TV1")
    grid = gpsd.default_grid(P, aperture_m=300.0, Nt=512)
    S = gpsd.scattering_function(P, grid)
    S = S / S.sum()
    f = grid.f_doppler
    fw = S.sum(axis=1)
    mean_f = float(np.sum(f * fw))
    std_f = float(np.sqrt(np.sum((f - mean_f) ** 2 * fw)))
    assert abs(mean_f) < 0.02 * std_f + 1e-3, mean_f
    expected_std = np.sqrt(2.0) / (2.0 * np.pi * P.tau_o)
    assert abs(std_f / expected_std - 1.0) < 0.02, (std_f, expected_std)
    tau = grid.tau
    mean_tau = float(np.sum(tau * S.sum(axis=0)))
    assert abs(mean_tau / P.tau_mean - 1.0) < 0.03, (mean_tau, P.tau_mean)


def test_mcf_matches_eq_11_2():
    """Empirical space-time correlation of array realisations against eq. (11.2)."""
    P = _tv_params("TV1")
    grid = gpsd.DiffuseGrid(Nx=32, Ny=32, dx=P.lx / 8.0, dy=P.ly / 8.0, Nt=64, dt=P.tau_o / 8.0,
                            Nk=8, dtau=2.0e-6, k0=4)
    rng = np.random.default_rng(11)
    Pd = 1.0
    n_real = 40
    acc = {}
    norm = 0.0
    for _ in range(n_real):
        h, _ = gpsd.realise_array(P, grid, Pd, rng)
        I = np.sum(h, axis=0)              # narrowband: sum over delay bins -> (Nx, Ny, Nt)
        pw = np.mean(np.abs(I) ** 2)
        norm += pw
        for da, db, dn in [(0, 0, 0), (4, 0, 0), (8, 0, 0), (0, 4, 0), (0, 0, 2), (4, 0, 2), (0, 4, 4)]:
            corr = np.mean(np.roll(np.roll(np.roll(I, -da, 0), -db, 1), -dn, 2) * np.conj(I))
            acc[(da, db, dn)] = acc.get((da, db, dn), 0.0) + corr.real
    norm /= n_real
    for key in acc:
        acc[key] /= (n_real * norm)
    Cxt, Cyt = P.Cxt, P.Cyt
    worst = 0.0
    for (da, db, dn), emp in acc.items():
        sx = da * grid.dx / P.lx
        sy = db * grid.dy / P.ly
        st = dn * grid.dt / P.tau_o
        theo = np.exp(-sx * sx - sy * sy - st * st + 2 * Cxt * sx * st + 2 * Cyt * sy * st)
        worst = max(worst, abs(emp - theo))
    assert worst < 0.12, f"largest MCF deviation {worst:.3f}"


def test_sign_convention_frozen_flow_moves_with_V():
    """C_xt -> 1 (frozen pattern): cross-correlation along x_hat peaks at lag dx / (V.x_hat) > 0."""
    P = _tv_params("TV3")
    assert P.Cxt > 0.99 and P.V_app @ P.x_hat > 0.0
    grid = gpsd.DiffuseGrid(Nx=64, Ny=8, dx=P.lx / 8.0, dy=P.ly / 8.0, Nt=256, dt=0.02,
                            Nk=8, dtau=5e-7, k0=4)
    rng = np.random.default_rng(5)
    h, _ = gpsd.realise_array(P, grid, 1.0, rng)
    I = np.sum(h, axis=0)                 # (Nx, Ny, Nt)
    shift = 4                             # 4 grid cells along x
    lags = np.arange(-40, 41)
    # Spec §11.7 item 5: R(lag) = E[h(x_B, t + lag) h*(x_A, t)] with x_B = x_A + shift*dx.
    # A pattern moving with +V reaches x_B later, so R peaks at lag = shift*dx/V > 0.
    cc = []
    for lag in lags:
        leading = np.roll(np.roll(I, -shift, 0), -lag, 2)     # h(x_A + shift, t + lag)
        cc.append(np.mean(leading * np.conj(I)).real)         # times h*(x_A, t)
    peak_lag = lags[int(np.argmax(cc))]
    expected = shift * grid.dx / abs(P.V_app @ P.x_hat) / grid.dt
    assert peak_lag > 0, peak_lag
    assert abs(peak_lag - expected) <= 1.5, (peak_lag, expected)


def test_realisation_power_bookkeeping():
    """Mean power of the Rician bracket is A_c^2 + 2 A_d^2 = 1 (spec §11.6)."""
    P = _tv_params("TV2")
    grid = gpsd.default_grid(P, aperture_m=0.0, Nt=128)
    rng = np.random.default_rng(2)
    n = 400
    total = 0.0
    for _ in range(n):
        h = gpsd.realise_single_element(P, grid, 2.0 * P.A_d**2, rng)
        total += np.sum(np.abs(h) ** 2) / grid.Nt
    mean_power = total / n
    assert abs(mean_power - 2.0 * P.A_d**2) < 3.0 / np.sqrt(n) * 2.0 * P.A_d**2 + 1e-3 * P.A_d, mean_power


def test_S4_of_narrowband_response_matches_input():
    """Intensity scintillation of the delay-summed response reproduces the input S4 (Rician)."""
    from dataclasses import replace
    from stochastic.channel import rician
    base = _tv_params("TV1")
    Ac, Ad = rician(0.5)
    P = replace(base, A_c=Ac, A_d=Ad, S4=0.5)
    grid = gpsd.default_grid(P, Nt=64)
    rng = np.random.default_rng(9)
    I_samples = []
    for _ in range(3000):
        h = gpsd.element_responses(P, grid, a=np.array([1.0 + 0j]), zeta=np.array([0.0]),
                                   method="single", rng=rng)
        I_samples.append(np.abs(np.sum(h[0], axis=0)) ** 2)
    I = np.concatenate(I_samples)
    S4 = np.sqrt(np.mean(I**2) / np.mean(I) ** 2 - 1.0)
    assert abs(S4 - 0.5) < 0.05, S4


def test_element_offsets_shift_coherent_peak():
    """Element delay offset: the coherent peak sits at tau = -zeta/c (spec §11.5)."""
    from dataclasses import replace
    base = _tv_params("TV1")
    P = replace(base, A_c=1.0, A_d=0.0)
    grid = gpsd.default_grid(P, Nt=16)
    zeta = np.array([0.0, 300.0])                       # metres towards the scatterer
    h = gpsd.element_responses(P, grid, a=np.array([1.0, 1.0]), zeta=zeta,
                               method="single", rng=np.random.default_rng(0))
    power = np.sum(np.abs(h) ** 2, axis=2)              # (2, Nk)
    tau_peak = grid.tau[np.argmax(power, axis=1)]
    expected = -zeta / gpsd.C_LIGHT
    assert np.all(np.abs(tau_peak - expected) <= grid.dtau), (tau_peak, expected)
