"""
cascade.py -- two-way realisation: cascade of an outbound and a return leg (spec §12.1, §12.2).

    h_pq(t, tau) = rho_s(t) * int h_out,p(t, tau') h_ret,q(t, tau - tau') dtau'      (12.1)

The two legs convolve in delay and multiply in slow time, so for statistically independent WSSUS legs
the scattering functions convolve in delay and Doppler (12.2): Doppler variances add, mean delays add,
and mean powers multiply. The deterministic parts of §6.1 are applied here too: the carrier phase
-k0 (P_out + P_ret) and the Doppler f_out + f_ret + f_T on the slow-time axis. The absolute group
delay (P'_out + P'_ret)/c is returned as an offset: the excess delays of the legs are on their own
grids and the offset is placed by a band-limited (sinc) fractional shift, see place_on_grid().

Inputs are realisations on a common excess-delay grid (Nk bins of dtau) and a common slow-time
grid (Nt samples of 1/fs), e.g. gpsd.realise_single_element(). Output on (2 Nk - 1) delay bins.
"""

from __future__ import annotations

import numpy as np

from core.constants import C as C_LIGHT
from stochastic.gpsd import _shift_matrix


def convolve_delay(h_out: np.ndarray, h_ret: np.ndarray, dtau: float) -> np.ndarray:
    """
    Linear convolution along the delay axis (axis 0) for every slow-time sample (eq. 12.1), with the
    delay integral as a sum times dtau. h_out, h_ret: (Nk, Nt) complex on the same grid. Returns
    (2 Nk - 1, Nt).
    """
    h_out = np.asarray(h_out, dtype=complex)
    h_ret = np.asarray(h_ret, dtype=complex)
    if h_out.shape != h_ret.shape:
        raise ValueError("the two legs must share the delay and slow-time grid")
    Nk = h_out.shape[0]
    n = 2 * Nk - 1
    H = np.fft.fft(h_out, n=n, axis=0) * np.fft.fft(h_ret, n=n, axis=0)
    return np.fft.ifft(H, axis=0) * dtau


def place_on_grid(h: np.ndarray, shift_bins: float) -> np.ndarray:
    """
    Band-limited fractional shift along the delay axis: returns h(tau_k + shift_bins * dtau) on the same
    bins, by sinc interpolation (gpsd._shift_matrix). Valid for shifts well inside the grid.
    """
    Nk = h.shape[0]
    M = _shift_matrix(np.array([float(shift_bins)]), Nk)[0]
    return M @ h


def two_way_realisation(h_out: np.ndarray, h_ret: np.ndarray, *, dtau: float, t: np.ndarray,
                        P_out: float, P_ret: float, Pg_out: float, Pg_ret: float, c: float = C_LIGHT,
                        f_out: float = 0.0, f_ret: float = 0.0, f_T: float = 0.0, k0: float,
                        rho: np.ndarray | complex = 1.0):
    """
    Two-way realisation of one component (spec §12.1, §6.1). Returns (h, offset_s):
      h        (2 Nk - 1, Nt) excess-delay response including rho(t), the carrier phase
               exp(-i k0 (P_out + P_ret)) and the Doppler exp(2 pi i (f_out + f_ret + f_T) t);
      offset_s absolute group delay (Pg_out + Pg_ret) / c, to be placed with place_on_grid().
    P (phase path) and Pg (group path) are in metres.
    """
    h = convolve_delay(h_out, h_ret, dtau)
    t = np.asarray(t, dtype=float)
    rho_t = np.broadcast_to(np.asarray(rho, dtype=complex), t.shape)
    phase = np.exp(-1j * k0 * (P_out + P_ret))
    dop = np.exp(2j * np.pi * (f_out + f_ret + f_T) * t)
    return h * (rho_t * phase * dop)[None, :], (Pg_out + Pg_ret) / c
