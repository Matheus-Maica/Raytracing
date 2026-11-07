"""
gpsd.py -- Dana-Wittwer GPSD, bin-integrated expected power, channel realisations
(spec §11.1 - §11.6).

Sign and normalisation conventions (spec §2.3, eq. 2.1):

    h(x, t, tau) = sum H(K, f_D, tau) exp(-i K.x + 2 pi i f_D t)

so a diffuse component with Doppler f_D and transverse wavenumber K moves with the
pattern velocity V (spec §11.7 sign test). Realisations are synthesised with a
forward FFT over the transverse axes and an inverse FFT over Doppler (multiplied
by N_t, which undoes numpy's 1/N_t).

Delay bins are relative to the mode delay: tau_k = (k - k0) dtau, with tau = 0
on the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from core.records import ChannelParams

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# GPSD (11.1)
# ---------------------------------------------------------------------------

def gpsd(omega_D, tau, Kx, Ky, P: ChannelParams):
    """
    The Dana-Wittwer generalised power spectral density, eq. (11.1). Broadcasts its
    arguments. Integrates to (2 pi)^3 over (omega_D, tau, K_x, K_y).
    """
    lx, ly = P.lx, P.ly
    C2 = P.Cxt**2 + P.Cyt**2
    leff2 = np.sqrt(2.0 * lx**4 * ly**4 / (lx**4 + ly**4))
    pref = np.pi * P.tau_o * lx * ly * P.alpha * P.w_coh / np.sqrt(2.0 * (1.0 - C2))
    t1 = np.exp(-(P.tau_o * omega_D - P.Cxt * Kx * lx - P.Cyt * Ky * ly) ** 2 / (4.0 * (1.0 - C2)))
    t2 = np.exp(-(Kx**2 * lx**2 + Ky**2 * ly**2) / 4.0)
    t3 = np.exp(-0.5 * P.alpha**2 * (P.w_coh * tau - leff2 * (Kx**2 + Ky**2) / 4.0) ** 2)
    return pref * t1 * t2 * t3


# ---------------------------------------------------------------------------
# Sampling grids (11.2)
# ---------------------------------------------------------------------------

@dataclass
class DiffuseGrid:
    """Transverse, slow-time and delay sampling of a realisation."""

    Nx: int
    Ny: int
    dx: float
    dy: float
    Nt: int
    dt: float
    Nk: int
    dtau: float
    k0: int                     # index of tau = 0 on the delay axis

    @property
    def Kx(self) -> np.ndarray:
        return TWO_PI * np.fft.fftfreq(self.Nx, d=self.dx)

    @property
    def Ky(self) -> np.ndarray:
        return TWO_PI * np.fft.fftfreq(self.Ny, d=self.dy)

    @property
    def f_doppler(self) -> np.ndarray:
        return np.fft.fftfreq(self.Nt, d=self.dt)

    @property
    def tau(self) -> np.ndarray:
        return (np.arange(self.Nk) - self.k0) * self.dtau

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.Nt) * self.dt


def default_grid(P: ChannelParams, *, aperture_m: float = 0.0, bandwidth_hz: float = 20e3,
                 Nt: int = 256, omega0: float | None = None, alpha_tau_range=None) -> DiffuseGrid:
    """
    Sampling rules of spec §11.2:
      slow time: dt = tau_o / 4 (also satisfies 1/dt >= 2 f_ext for f_ext = 0.9/tau_o);
      transverse: dx = lx/8, dy = ly/8, footprint = aperture + 4 l margins on each side,
                  rounded up to a power of two;
      delay: tau_min = -5/(alpha w_coh), tau_max = 7 <tau> + 5/(alpha w_coh),
             dtau <= 1/(2 B), with at least 16 bins across the range.
    """
    dx, dy = P.lx / 8.0, P.ly / 8.0
    footprint_x = aperture_m + 8.0 * P.lx
    footprint_y = aperture_m + 8.0 * P.ly
    Nx = int(2 ** np.ceil(np.log2(max(footprint_x / dx, 8.0))))
    Ny = int(2 ** np.ceil(np.log2(max(footprint_y / dy, 8.0))))
    dt = P.tau_o / 4.0
    scale = 1.0 / (P.alpha * P.w_coh)
    tau_min = -5.0 * scale
    tau_max = 7.0 * P.tau_mean + 5.0 * scale
    dtau = min(1.0 / (2.0 * bandwidth_hz), (tau_max - tau_min) / 16.0)
    k0 = int(np.ceil(-tau_min / dtau))
    Nk = int(np.ceil(tau_max / dtau)) + k0 + 1
    return DiffuseGrid(Nx=Nx, Ny=Ny, dx=dx, dy=dy, Nt=Nt, dt=dt, Nk=Nk, dtau=dtau, k0=k0)


# ---------------------------------------------------------------------------
# Bin-integrated expected power (11.3)
# ---------------------------------------------------------------------------

def bin_weights(P: ChannelParams, grid: DiffuseGrid, k: int) -> np.ndarray:
    """
    E|H_mnjk|^2 / P_d of eq. (11.3) for delay bin k: array (Nx, Ny, Nt) over
    transverse wavenumber cells (m, n) and Doppler bins (j). Q, P_j and T_k are
    the bin-integrated weights of the spec; the Doppler sum over a in {-1, 0, 1}
    folds the spectrum into the Nyquist band.
    """
    lx, ly = P.lx, P.ly
    Kx = grid.Kx[:, None]
    Ky = grid.Ky[None, :]
    dKx = TWO_PI / (grid.Nx * grid.dx)
    dKy = TWO_PI / (grid.Ny * grid.dy)
    Q = (lx * ly / (4.0 * np.pi)) * np.exp(-(Kx**2 * lx**2 + Ky**2 * ly**2) / 4.0) * dKx * dKy

    C2 = P.Cxt**2 + P.Cyt**2
    s_f = np.sqrt(2.0 * (1.0 - C2)) / (TWO_PI * P.tau_o)
    fbar = (P.Cxt * Kx * lx + P.Cyt * Ky * ly) / (TWO_PI * P.tau_o)           # (Nx, Ny)
    df = 1.0 / (grid.Nt * grid.dt)
    fs = 1.0 / grid.dt
    f = grid.f_doppler[None, None, :]
    fb = fbar[:, :, None]
    Pj = np.zeros((grid.Nx, grid.Ny, grid.Nt))
    for a in (-1.0, 0.0, 1.0):
        Pj += ndtr((f + 0.5 * df - fb - a * fs) / s_f) - ndtr((f - 0.5 * df - fb - a * fs) / s_f)

    leff2 = np.sqrt(2.0 * lx**4 * ly**4 / (lx**4 + ly**4))
    K2 = Kx**2 + Ky**2
    taubar = leff2 * K2 / (4.0 * P.w_coh)
    s_tau = 1.0 / (P.alpha * P.w_coh)
    tau_k = grid.tau[k]
    Tk = ndtr((tau_k + 0.5 * grid.dtau - taubar) / s_tau) - ndtr((tau_k - 0.5 * grid.dtau - taubar) / s_tau)
    return Q[:, :, None] * Pj * Tk[:, :, None]


_S_CACHE: dict = {}


def _grid_key(P: ChannelParams, grid: DiffuseGrid):
    return (P.lx, P.ly, P.tau_o, P.Cxt, P.Cyt, P.alpha, P.w_coh,
            grid.Nx, grid.Ny, grid.dx, grid.dy, grid.Nt, grid.dt, grid.Nk, grid.dtau, grid.k0)


def single_element_weights(P: ChannelParams, grid: DiffuseGrid) -> np.ndarray:
    """
    E|H_jk|^2 / P_d summed over the transverse cells: S[k, j] of shape (Nk, Nt).

    This is eq. (11.3) with the sum over K done first, as a matrix product
    S = (Q T)^T P over the transverse cells K. Cells whose Gaussian weight Q is
    below 1e-12 of the peak are dropped (exact to that tolerance). Cached per
    (P, grid), because the single-element method calls it once per realisation.
    """
    key = _grid_key(P, grid)
    if key in _S_CACHE:
        return _S_CACHE[key]
    lx, ly = P.lx, P.ly
    KX, KY = np.meshgrid(grid.Kx, grid.Ky, indexing="ij")
    KX = KX.ravel()
    KY = KY.ravel()
    dKx = TWO_PI / (grid.Nx * grid.dx)
    dKy = TWO_PI / (grid.Ny * grid.dy)
    Q = (lx * ly / (4.0 * np.pi)) * np.exp(-(KX**2 * lx**2 + KY**2 * ly**2) / 4.0) * dKx * dKy
    keep = Q > 1.0e-12 * Q.max()
    KX, KY, Q = KX[keep], KY[keep], Q[keep]

    C2 = P.Cxt**2 + P.Cyt**2
    s_f = np.sqrt(2.0 * (1.0 - C2)) / (TWO_PI * P.tau_o)
    fbar = (P.Cxt * KX * lx + P.Cyt * KY * ly) / (TWO_PI * P.tau_o)           # (NK,)
    df = 1.0 / (grid.Nt * grid.dt)
    fs = 1.0 / grid.dt
    f = grid.f_doppler[None, :]
    fb = fbar[:, None]
    Pj = np.zeros((KX.size, grid.Nt))
    for a in (-1.0, 0.0, 1.0):
        Pj += ndtr((f + 0.5 * df - fb - a * fs) / s_f) - ndtr((f - 0.5 * df - fb - a * fs) / s_f)

    leff2 = np.sqrt(2.0 * lx**4 * ly**4 / (lx**4 + ly**4))
    taubar = leff2 * (KX**2 + KY**2) / (4.0 * P.w_coh)                       # (NK,)
    s_tau = 1.0 / (P.alpha * P.w_coh)
    tau = grid.tau[None, :]
    tb = taubar[:, None]
    Tk = ndtr((tau + 0.5 * grid.dtau - tb) / s_tau) - ndtr((tau - 0.5 * grid.dtau - tb) / s_tau)
    S = (Q[:, None] * Tk).T @ Pj                                              # (Nk, Nt)
    _S_CACHE[key] = S
    return S


def captured_fraction(P: ChannelParams, grid: DiffuseGrid) -> float:
    """Sum of E|H|^2/P_d over all bins; must exceed 0.99 (spec §11.3)."""
    return float(single_element_weights(P, grid).sum())


def scattering_function(P: ChannelParams, grid: DiffuseGrid) -> np.ndarray:
    """Expected power density S(f_j, tau_k) (Nt, Nk), not normalised (diagnostic, Fig. 4)."""
    return single_element_weights(P, grid).T.copy()


# ---------------------------------------------------------------------------
# Realisations (11.4)
# ---------------------------------------------------------------------------

def _unit_complex(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def realise_single_element(P: ChannelParams, grid: DiffuseGrid, Pd: float, rng) -> np.ndarray:
    """
    Single-element method (11.4b): one tapped delay line at the array end.
    Sum over K first: S_jk = sum_mn E|H|^2, then H_jk = sqrt(S_jk) r_jk, inverse FFT in j.
    Returns h (Nk, Nt) complex.
    """
    S = single_element_weights(P, grid)                  # (Nk, Nt), cached
    out = np.empty((grid.Nk, grid.Nt), dtype=complex)
    for k in range(grid.Nk):
        H = np.sqrt(Pd * S[k]) * _unit_complex(rng, (grid.Nt,))
        out[k] = np.fft.ifft(H) * grid.Nt
    return out


def realise_array(P: ChannelParams, grid: DiffuseGrid, Pd: float, rng,
                  positions_xy: np.ndarray | None = None):
    """
    Array method (11.4a). Returns (h_grid, h_elements):

      h_grid     (Nk, Nx, Ny, Nt): the field on the transverse grid (x_a = a dx, y_b = b dy).
      h_elements (Np, Nk, Nt):     the field at positions_xy (Np, 2), by direct sums over
                                   the same random amplitudes (None if positions_xy is None).

    Memory is Nk * Nx * Ny * Nt complex: use small grids in tests.
    """
    Kx = grid.Kx
    Ky = grid.Ky
    h_grid = np.empty((grid.Nk, grid.Nx, grid.Ny, grid.Nt), dtype=complex)
    h_el = None
    if positions_xy is not None:
        positions_xy = np.atleast_2d(np.asarray(positions_xy, dtype=float))
        h_el = np.empty((positions_xy.shape[0], grid.Nk, grid.Nt), dtype=complex)
    for k in range(grid.Nk):
        w = bin_weights(P, grid, k)
        H = np.sqrt(Pd * w) * _unit_complex(rng, w.shape)
        X = np.fft.fft2(H, axes=(0, 1))                  # sum_m,n H exp(-i (K_m x_a + K_n y_b))
        h_grid[k] = np.fft.ifft(X, axis=2) * grid.Nt     # sum_j exp(+2 pi i f_j t_n)
        if h_el is not None:
            for p, (xp, yp) in enumerate(positions_xy):
                ex = np.exp(-1j * Kx * xp)
                ey = np.exp(-1j * Ky * yp)
                Xp = np.einsum("m,n,mnj->j", ex, ey, H)
                h_el[p, k] = np.fft.ifft(Xp) * grid.Nt
    return h_grid, h_el


def _shift_matrix(shift_bins: np.ndarray, Nk: int) -> np.ndarray:
    """
    Bandlimited fractional shift along the delay axis: row p gives the weights that map
    h(tau_k) to h(tau_k + shift_p * dtau), by sinc interpolation (truncated to the grid).
    Returns (Np, Nk, Nk) matrices M_p with h_shifted = M_p @ h.
    """
    idx = np.arange(Nk)
    M = np.sinc(idx[None, None, :] - idx[None, :, None] - shift_bins[:, None, None])
    return M


def element_responses(P: ChannelParams, grid: DiffuseGrid, *, a, zeta, A_c: float | None = None,
                      f_D: float = 0.0, Pd: float | None = None, rng=None, method: str = "array",
                      positions_xy=None, pattern_gain=None):
    """
    Rician assembly of eq. (11.4) for Np elements of one leg:

        h_p(t, tau_k) = a_p exp(2 pi i f_D t) [ A_c delta(tau_k + zeta_p/c)
                                              + h_d,p(t, tau_k + zeta_p/c) ]

    with h_d having total power P_d = 2 A_d^2 (so the bracket has mean power 1).
    The delta is bandlimited on the grid, and the diffuse part is shifted by
    sinc interpolation. a: (Np,) complex deterministic gain. zeta: (Np,) [m].
    method: "array" (positions_xy required) or "single" (all elements identical).
    Returns h (Np, Nk, Nt).
    """
    a = np.asarray(a, dtype=complex)
    zeta = np.asarray(zeta, dtype=float)
    Np = a.size
    rng = rng if rng is not None else np.random.default_rng()
    Ad = P.A_d
    if Pd is None:
        Pd = 2.0 * Ad * Ad
    A_c = P.A_c if A_c is None else A_c

    if method == "array":
        if positions_xy is None:
            raise ValueError("method='array' needs positions_xy (Np, 2)")
        _, h_d = realise_array(P, grid, Pd, rng, positions_xy=positions_xy)
    elif method == "single":
        h_single = realise_single_element(P, grid, Pd, rng)
        h_d = np.broadcast_to(h_single[None], (Np, grid.Nk, grid.Nt)).copy()
    else:
        raise ValueError(f"unknown method {method!r}")

    shifts = zeta / C_LIGHT / grid.dtau                  # element shifts in delay bins
    M = _shift_matrix(shifts, grid.Nk)
    t = grid.t
    phase_t = np.exp(2j * np.pi * f_D * t)               # (Nt,)
    out = np.empty((Np, grid.Nk, grid.Nt), dtype=complex)
    for p in range(Np):
        # diffuse part at tau_k + zeta_p/c, then the coherent bandlimited delta at -zeta_p/c
        hd_shift = M[p] @ h_d[p]
        tau_eff = grid.tau + zeta[p] / C_LIGHT
        coherent = A_c * np.sinc(tau_eff / grid.dtau)    # (Nk,) bandlimited delta(tau + zeta/c)
        gain = a[p] if pattern_gain is None else a[p] * pattern_gain[p]
        out[p] = gain * (coherent[:, None] * phase_t[None, :] + hd_shift * phase_t[None, :])
    return out


C_LIGHT = 299792458.0
