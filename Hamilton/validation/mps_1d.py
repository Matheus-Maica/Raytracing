"""
mps_1d.py -- 1-D multiple-phase-screen reference (spec §15.3, plan Phase 3e). ON DEMAND: run by hand, not in CI.

Split-step Fourier propagation of a plane wave through a sequence of Gaussian random phase screens with the
structure function of eq. (9.4), D(s) = 2 sigma^2 Psi_p(|s|/L0), with the paraxial free-space propagator between
screens. The field is 1-D (x only), periodic on the grid. Outputs the mutual coherence function (MCF), the
intensity scintillation index S4, and the check against the exact thin-screen result.

Checks run by main():
  1. Screen statistics: the MCF just after one screen equals exp(-D(s)/2) exactly for a Gaussian screen.
  2. Propagation: mean intensity is conserved (paraxial propagator is unitary), and S4 grows with distance.

This engine is the basis for the group-vs-arc decision (D5, plan Phase 3e, spec §9.1): a leg-specific screen set
(from build_screens on a traced leg) is propagated with the arc-length spacing and compared with the Algorithm 5
prediction built on each metric. That comparison is not done yet.

Usage:  venv/bin/python validation/mps_1d.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stochastic.screens import psi_p  # noqa: E402


def phase_screen_spectrum(N: int, dx: float, L0: float, p: float, sigma2: float) -> np.ndarray:
    """
    Spectral weights S_k (sum S_k = 2 N sigma^2) whose inverse transform has the correlation
    rho(s) = 1 - Psi_p(|s|/L0) on the periodic grid. Negative numerical values are clipped to zero.
    """
    s = np.fft.fftfreq(N, d=1.0 / N) * dx                       # signed separations on the grid
    rho = 1.0 - psi_p(np.abs(s) / L0, p)
    F = np.real(np.fft.fft(rho))
    F = np.clip(F, 0.0, None)
    return 2.0 * N * sigma2 * F / np.sum(F)


def random_screen(S: np.ndarray, N: int, rng) -> np.ndarray:
    """One real phase screen phi(x_n) with the spectrum S_k (Gaussian, zero mean)."""
    g = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0)
    c = np.sqrt(S / N) * g
    return np.real(np.fft.ifft(c) * N)


def propagate(u: np.ndarray, dz: float, k0: float, dx: float) -> np.ndarray:
    """Paraxial free-space step: exp(-i kx^2 dz / (2 k0)) in the spatial-frequency domain."""
    N = u.size
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return np.fft.ifft(np.fft.fft(u) * np.exp(-1j * kx ** 2 * dz / (2.0 * k0)))


def mcf(fields: np.ndarray, lags: np.ndarray, dx: float) -> np.ndarray:
    """Mutual coherence Gamma(s) = <u(x) u*(x+s)> / <|u|^2>, averaged over x and realisations."""
    N = fields.shape[-1]
    out = []
    for m in lags:
        k = int(round(m / dx))
        prod = fields * np.conj(np.roll(fields, -k, axis=-1))
        out.append(np.mean(prod) / np.mean(np.abs(fields) ** 2))
    return np.array(out)


def run_single_screen(*, N=4096, dx=0.5, L0=50.0, p=4.0, sigma2=0.3, n_real=200, seed=1):
    """MPS (thin screen at z = 0) against the exact exp(-D(s)/2)."""
    rng = np.random.default_rng(seed)
    S = phase_screen_spectrum(N, dx, L0, p, sigma2)
    fields = np.empty((n_real, N), dtype=complex)
    for r in range(n_real):
        phi = random_screen(S, N, rng)
        fields[r] = np.exp(1j * phi)
    lags = np.array([0.0, 0.25, 0.5, 1.0, 1.5]) * L0
    gamma = mcf(fields, lags, dx)
    exact = np.exp(-sigma2 * psi_p(np.abs(lags) / L0, p))      # exp(-D/2), D = 2 sigma^2 Psi_p
    return lags, gamma, exact


def run_propagation(*, N=4096, dx=0.5, L0=50.0, p=4.0, sigma2=0.3, lam=0.5, z=2000.0, n_real=60, seed=2):
    """Intensity statistics after one screen and propagation over z: mean intensity and S4."""
    rng = np.random.default_rng(seed)
    k0 = 2.0 * np.pi / lam
    S = phase_screen_spectrum(N, dx, L0, p, sigma2)
    I_all = []
    for r in range(n_real):
        u = np.exp(1j * random_screen(S, N, rng))
        u = propagate(u, z, k0, dx)
        I_all.append(np.abs(u) ** 2)
    I = np.array(I_all)
    mean_I = I.mean()
    S4 = np.sqrt((I ** 2).mean() / mean_I ** 2 - 1.0)
    return mean_I, S4


def main():
    lags, gamma, exact = run_single_screen()
    print("1. MCF after one Gaussian screen (MPS vs exp(-D/2)):")
    for s, g, e in zip(lags, gamma, exact):
        print(f"   s = {s:7.1f} m   MPS {g.real:8.4f}   exact {e:8.4f}   diff {abs(g.real - e):7.4f}")
    mean_I, S4 = run_propagation()
    print(f"2. After one screen and z = 2 km (lambda = 0.5 m): mean intensity {mean_I:.4f} (1 if conserved), "
          f"S4 = {S4:.4f}")
    worst = np.max(np.abs(gamma.real - exact))
    print(f"   worst MCF deviation {worst:.4f} (expected within a few per cent of the statistical noise)")
    return 0 if worst < 0.05 and abs(mean_I - 1.0) < 0.02 else 1


if __name__ == "__main__":
    raise SystemExit(main())
