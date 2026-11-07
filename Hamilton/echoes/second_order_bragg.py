"""
second_order_bragg.py -- second-order HF sea-echo Doppler cross section (HiCIRF paragraph 31: "we have
incorporated first- and second-order Bragg scatter from the sea"), from Barrick (1972), "Remote sensing of sea
state by radar", Remote Sensing of the Troposphere (V.E. Derr, ed.), ch. 12, section 12.3, eqs. (12:31)-(12:34)
(docs/1972Barrick_RST.pdf, pp. 12-26 to 12-31).

Near-grazing vertical-polarisation backscatter, radar looking along +x (incident wavenumber k0):

    sigma2(eta) = 2^6 pi k0^4 sum_{m,m'=+-1} int int |Gamma|^2 S(m k1) S(m' k2)
                  delta(eta - m sqrt(g kappa1) - m' sqrt(g kappa2)) dp dq
    k1 = (p - k0, q),  k2 = (-p - k0, -q)   (k1 + k2 = -2 k0 x: the first-order Bragg vector)
    Gamma = Gamma_EM + Gamma_H
    Gamma_EM = 1/2 (k1x k2x - 2 k1.k2) / (sqrt(k1.k2) + k0 Delta)                                 (12:32)
    Gamma_H  = -i/2 [kappa1 + kappa2 + (kappa1 kappa2 - k1.k2)
                     {1 - 2 eta^2/(eta^2 - omega_B^2)} g/(omega1 omega2)],  omega_B = sqrt(2 g k0)  (12:33)

written in the physical convention used by clutter.py's first order (S(k) is the energy of waves travelling
ALONG k, int S d^2k = h^2; sigma1(eta) = 2^6 pi k0^4 sum_m S(-2 m k0 x) delta(eta - m omega_B)). Barrick writes
the same integral with his symmetric spectrum W = 4 S_B and the frequency sign tied to sgn(kx) (his
"semi-isotropic" construction); summing over m, m' with S(m k) is that construction for a general directional
sea. Delta = 1/sqrt(eps_c) is the sea's normalised surface impedance (environment.SURFACE_CONSTANTS 'sea');
sqrt(k1.k2) takes the principal branch (imaginary for an evanescent intermediate wave).

Validation: with Barrick's own Phillips model (12:13) -- barrick_phillips_semi_isotropic() -- the first-order
level is his -17 dB (12:23), and total second-order sigma0 is compared in tests/test_second_order_bragg.py with
the values printed on his Figs. 12.18 (upwind) and 12.19 (crosswind) at 10 MHz for 15-40 kt winds.

Numerics: the delta is integrated by binning (each (p, q) cell's |Gamma|^2 S S dp dq goes into the Doppler bin
of its eta) on a uniform (p, q) grid; sigma0 (the integral over eta) needs no binning at all.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from core.constants import C, epsilon_0
from echoes.clutter import G_GRAVITY, PLACEHOLDER_BS, PLACEHOLDER_SPREAD_S, spreading_G, wave_saturation_S


# ---------------------------------------------------------------------------
# Directional wave spectra, physical convention: S(kx, ky) [m^4], int S dkx dky = h^2
# (angles in the radar frame: x along the look; wind_dir = direction the WAVES travel)
# ---------------------------------------------------------------------------

def barrick_phillips_semi_isotropic(B: float, u_mps: float, wind_dir_rad: float):
    """
    Barrick (1972) eq. (12:13): Phillips spectrum, k^-4 above the wind cutoff g/u^2, "semi-isotropic" (all
    waves in the downwind half-plane, equally in every direction there). In the physical convention that is
    (B/pi) k^-4 on the half-plane -- twice Barrick's symmetric B/(2 pi k^4) -- which gives his first-order
    sigma0 = 4 B = -17 dB for B = 0.005 (12:23).
    """
    kc = G_GRAVITY / u_mps ** 2

    def S(kx, ky):
        k = np.hypot(kx, ky)
        downwind = (kx * np.cos(wind_dir_rad) + ky * np.sin(wind_dir_rad)) > 0.0
        ok = (k > kc) & downwind
        return np.where(ok, (B / np.pi) / np.where(ok, k, 1.0) ** 4, 0.0)

    return S


def cardioid_phillips(u_mps: float, wind_dir_rad: float, B_s: float = PLACEHOLDER_BS,
                      s: float = PLACEHOLDER_SPREAD_S):
    """clutter.py's own first-order spectrum, clutter.wave_saturation_S = B_s k^-4 G(phi - phi_wind) (Barrick's
    normalisation, int G dphi = 1), with Barrick's wind cutoff g/u^2 added (the k^-4 law needs it for the 2-D
    integral). Calls wave_saturation_S itself, so first and second order cannot use different spectra."""
    kc = G_GRAVITY / u_mps ** 2

    def S(kx, ky):
        k = np.hypot(kx, ky)
        G = spreading_G(np.arctan2(ky, kx) - wind_dir_rad, s)
        ok = k > kc
        return np.where(ok, wave_saturation_S(np.where(ok, k, 1.0), B_s, G), 0.0)

    return S


def sea_impedance(f0: float, eps_r: float = 70.0, sigma_S_m: float = 5.0) -> complex:
    """
    Delta = 1/sqrt(eps_c) in the e^{-i omega t} convention of Barrick's theory, i.e. the complex conjugate of
    the value in Hamilton's own e^{+i omega t} convention (environment.relative_permittivity_complex). It must
    match the principal-branch sqrt(k1.k2) of Gamma_EM (+i for an evanescent intermediate wave); the usual
    literature value is 0.011 - 0.012i. Sea defaults from environment.SURFACE_CONSTANTS['sea'].
    """
    eps_c = eps_r - 1j * sigma_S_m / (2.0 * np.pi * f0 * epsilon_0)
    return complex(np.conj(1.0 / np.sqrt(eps_c)))


# ---------------------------------------------------------------------------
# First and second order
# ---------------------------------------------------------------------------

def first_order_sigma0(f0: float, S) -> tuple:
    """(sigma0 at eta = +omega_B (approaching), sigma0 at -omega_B (receding)) = 2^6 pi k0^4 S(-+2 k0 x)."""
    k0 = 2.0 * np.pi * f0 / C
    pref = 2.0 ** 6 * np.pi * k0 ** 4
    return float(pref * S(-2.0 * k0, 0.0)), float(pref * S(2.0 * k0, 0.0))


def _gamma2(k0, k1x, k1y, k2x, k2y, kap1, kap2, eta, m, mp, delta):
    """|Gamma_EM + Gamma_H|^2, eqs. (12:32)-(12:34)."""
    dot = k1x * k2x + k1y * k2y
    g_em = 0.5 * (k1x * k2x - 2.0 * dot) / (np.sqrt(dot.astype(complex)) + k0 * delta)
    omega_B2 = 2.0 * G_GRAVITY * k0
    eta2 = eta ** 2
    ratio = 1.0 - 2.0 * eta2 / (eta2 - omega_B2)
    g_h = -0.5j * (kap1 + kap2 + (kap1 * kap2 - dot) * ratio / (m * mp * np.sqrt(kap1 * kap2)))
    return np.abs(g_em + g_h) ** 2


def second_order(f0: float, S, *, eta_edges=None, p_max_k0: float = 5.0, dp_k0: float = 0.005,
                 delta: complex | None = None, rows_per_chunk: int = 200):
    """
    sigma2 of eq. (12:31) for a radar looking along +x. Returns (sigma0_2, eta_centres, sigma2_eta):
    sigma0_2 = int sigma2 d(eta) (the second-order normalised cross section), and, if eta_edges [rad/s] is
    given, sigma2 per unit eta on those bins (else None). p, q span [-p_max_k0, p_max_k0] k0 in steps of
    dp_k0 k0.
    """
    k0 = 2.0 * np.pi * f0 / C
    delta = sea_impedance(f0) if delta is None else delta
    axis = np.arange(-p_max_k0, p_max_k0 + 0.5 * dp_k0, dp_k0) * k0
    dA = (dp_k0 * k0) ** 2
    pref = 2.0 ** 6 * np.pi * k0 ** 4
    q = axis[None, :]
    sigma0 = 0.0
    hist = None if eta_edges is None else np.zeros(len(eta_edges) - 1)
    for i0 in range(0, axis.size, rows_per_chunk):
        p = axis[i0:i0 + rows_per_chunk, None]
        k1x, k1y = p - k0, np.broadcast_to(q, (p.size, q.size))
        k2x, k2y = -p - k0, -k1y
        kap1, kap2 = np.hypot(k1x, k1y), np.hypot(k2x, k2y)
        kap1 = np.broadcast_to(kap1, k1y.shape)
        kap2 = np.broadcast_to(kap2, k1y.shape)
        k1x = np.broadcast_to(k1x, k1y.shape)
        k2x = np.broadcast_to(k2x, k1y.shape)
        for m in (1, -1):
            s1 = S(m * k1x, m * k1y)
            if not np.any(s1):
                continue
            for mp in (1, -1):
                s2 = S(mp * k2x, mp * k2y)
                w = s1 * s2
                live = w > 0.0
                if not np.any(live):
                    continue
                kk1, kk2 = kap1[live], kap2[live]
                eta = m * np.sqrt(G_GRAVITY * kk1) + mp * np.sqrt(G_GRAVITY * kk2)
                g2 = _gamma2(k0, k1x[live], k1y[live], k2x[live], k2y[live], kk1, kk2, eta, m, mp, delta)
                contrib = pref * g2 * w[live] * dA
                sigma0 += float(np.sum(contrib))
                if hist is not None:
                    hist += np.histogram(eta, bins=eta_edges, weights=contrib)[0]
    if hist is None:
        return sigma0, None, None
    centres = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    return sigma0, centres, hist / np.diff(eta_edges)


@lru_cache(maxsize=512)
def _cached_lines(f0, u_mps, phi_rel_deg, B_s, s, n_bins, p_max_k0, dp_k0):
    k0 = 2.0 * np.pi * f0 / C
    omega_B = np.sqrt(2.0 * G_GRAVITY * k0)
    # phi_rel = 0: waves travel TOWARDS the radar (clutter.bragg_lines' convention) -> wind dir = pi
    S = cardioid_phillips(u_mps, np.pi + np.deg2rad(phi_rel_deg), B_s, s)
    edges = np.linspace(-2.5 * omega_B, 2.5 * omega_B, n_bins + 1)
    _, centres, sig = second_order(f0, S, eta_edges=edges, p_max_k0=p_max_k0, dp_k0=dp_k0)
    return tuple(centres / (2.0 * np.pi)), tuple(sig * np.diff(edges))


def second_order_lines(f0: float, phi_rel: float, u_mps: float, *, B_s: float = PLACEHOLDER_BS,
                       s: float = PLACEHOLDER_SPREAD_S, n_bins: int = 100, p_max_k0: float = 4.0,
                       dp_k0: float = 0.005):
    """
    The second-order continuum as discrete Doppler lines [(doppler_Hz, sigma0_of_that_bin), ...], for
    clutter.bragg_lines(second_order=True). Same spectrum and phi_rel convention as the first-order lines.
    Cached on (f0, u rounded to 0.5 m/s, phi_rel rounded to 5 deg) -- a scene has many cells with similar
    sea state, and each evaluation is a 2-D integral.
    """
    u_key = max(0.5, round(u_mps * 2.0) / 2.0)
    phi_key = (round(np.rad2deg(phi_rel) / 5.0) * 5.0) % 360.0
    f, s0 = _cached_lines(float(f0), u_key, phi_key, float(B_s), float(s), int(n_bins), float(p_max_k0),
                          float(dp_k0))
    return [(fd, sv) for fd, sv in zip(f, s0) if sv > 0.0]
