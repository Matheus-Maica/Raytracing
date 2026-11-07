"""
clutter.py -- surface clutter (spec §12.4): first-order Bragg lines from the sea, open-water fraction and the
ice return, land and ice-sheet returns, cell components, and the expected range-Doppler power map (step 1 of
§12.4.1). The field realisation of §12.4.1 step 2 is not implemented here.

Inputs that come from the surface model (wind direction, B_s, ice concentration, sigma0 of ice and land) are
PLACEHOLDERS (decision D9) and must be supplied by the user. f_B = sqrt(g/(pi lambda0)) for backscatter.

Wave spectrum normalisation (user decision 2026-11-07, "HiCIRF-exact"): Barrick's (1972) / Phillips', S(K, phi) =
B_s K^-4 G(phi) with int G dphi = 1, so sigma0 per line = 2^6 pi k0^4 S(2 k0) = 4 pi B_s G(phi). The engineering
spec's (B_s / 2 pi) K^-4 G (sigma0 = 2 B_s G, "order -25 dB") has 2 pi (8 dB) less energy for the same B_s; with
Barrick's form a fully developed semi-isotropic sea gives his -17 dB (Barrick 1972, eq. 12:23), and the default
cardioid (s = 3) gives -15 dB for the dominant line. second_order_bragg.py uses the same spectrum.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn

from core.constants import C
from echoes.twoway import radar_power_dBW

G_GRAVITY = 9.80665
BRAGG_SPREAD_DOPPLER_HZ = 0.015        # each line: small Gaussian width, spec §12.4 (0.01 to 0.02 Hz)
PLACEHOLDER_BS = 0.005                 # saturation-spectrum constant B_s, spec §12.4 (~0.005)
PLACEHOLDER_SPREAD_S = 3.0             # cardioid exponent s, spec §12.4 (2 to 4)


def cardioid_norm(s: float) -> float:
    """I_s = int_{-pi}^{pi} cos^{2s}(phi/2) dphi, so that G = cos^{2s}(phi/2)/I_s has int G dphi = 1."""
    return 2.0 * np.pi * gamma_fn(s + 0.5) / (np.sqrt(np.pi) * gamma_fn(s + 1.0))


def spreading_G(phi_rel: np.ndarray, s: float = PLACEHOLDER_SPREAD_S) -> np.ndarray:
    """Cardioid directional spreading about the wind direction, normalised to int G dphi = 1 (§12.4)."""
    phi = np.asarray(phi_rel, dtype=float)
    phi = (phi + np.pi) % (2.0 * np.pi) - np.pi          # wrap into (-pi, pi]
    c = np.cos(0.5 * phi)
    return np.where(c > 0.0, np.clip(c, 0.0, None) ** (2.0 * s), 0.0) / cardioid_norm(s)


def bragg_frequency_hz(f0: float) -> float:
    """First-order Bragg frequency for backscatter: f_B = sqrt(g/(pi lambda0)) (0.323 Hz at 10 MHz)."""
    lam0 = C / f0
    return float(np.sqrt(G_GRAVITY / (np.pi * lam0)))


def wave_saturation_S(K: float, Bs: float, G: float) -> float:
    """Directional wave-height spectrum S(K, phi) = B_s K^-4 G(phi - phi_w), int G dphi = 1 (Barrick 1972 /
    Phillips normalisation; see the module docstring)."""
    return Bs * K ** -4 * G


def bragg_line_sigma0(f0: float, phi_rel: float, Bs: float = PLACEHOLDER_BS,
                      s: float = PLACEHOLDER_SPREAD_S) -> float:
    """
    Cross-section of one backscatter line, eq. (12.4): 2^6 pi k0^4 S(K_B) with |K_B| = 2 k0, which equals
    4 pi B_s G(phi_rel). Returns the linear sigma0 (per unit area).
    """
    k0 = 2.0 * np.pi * f0 / C
    G = float(spreading_G(np.array([phi_rel]), s)[0])
    return 2.0 ** 6 * np.pi * k0 ** 4 * wave_saturation_S(2.0 * k0, Bs, G)


def open_water_fraction(c_ice: float) -> float:
    """Sea ice: the Bragg lines are multiplied by 1 - c_ice (§12.4). Consolidated pack ice (c = 1) removes them."""
    return float(np.clip(1.0 - c_ice, 0.0, 1.0))


def current_doppler_shift_hz(U_radial: float, f0: float) -> float:
    """Surface current shift of both lines for backscatter: K_B . U / 2 pi = 2 U_r / lambda0 (§12.4)."""
    return 2.0 * U_radial / (C / f0)


def land_sigma0_linear(sigma0_dB: float) -> float:
    """Land, tundra or ice sheet: sigma0 from the surface model, typically -30 to -20 dB (placeholder input)."""
    return 10.0 ** (sigma0_dB / 10.0)


def clutter_cell_power_W(*, sigma0: float, area_m2: float, Pt_dBW: float, f0: float, d_T: float, d_R: float,
                         loss_T_dB: float, loss_R_dB: float, Gt_dB: float = 0.0, Gr_dB: float = 0.0) -> float:
    """
    Received power of one cell component, eq. (6.2) with sigma_T -> sigma0 A_c (§12.4). Returns watts.
    """
    dBW = radar_power_dBW(Pt_dBW, Gt_dB, Gr_dB, f0, sigma0 * area_m2, d_T, d_R, loss_T_dB, loss_R_dB)
    return float(10.0 ** (dBW / 10.0))


def range_doppler_map(components, delay_edges, doppler_edges, *, spread_hz: float = BRAGG_SPREAD_DOPPLER_HZ):
    """
    Expected power density on a (delay, Doppler) grid (§12.4.1, step 1): each component is deposited at its
    delay bin with a Gaussian Doppler spread, normalised on the grid so that the deposited power is conserved.
    components: iterable of (delay_s, doppler_Hz, power_W). Returns P (Ntau, Nf) in W per bin.
    """
    tau_c = np.asarray(delay_edges, dtype=float)
    f = np.asarray(doppler_edges, dtype=float)
    dtau = np.diff(tau_c)
    tau_mid = 0.5 * (tau_c[:-1] + tau_c[1:])
    df = f[1] - f[0]
    P = np.zeros((tau_mid.size, f.size))
    for (dl, fd, pw) in components:
        k = int(np.argmin(np.abs(tau_mid - dl)))
        g = np.exp(-0.5 * ((f - fd) / spread_hz) ** 2)
        if g.sum() == 0.0:
            continue
        P[k] += pw * g / g.sum()
    return P


def bragg_lines(f0: float, phi_rel: float, *, c_ice: float = 0.0, Bs: float = PLACEHOLDER_BS,
                s: float = PLACEHOLDER_SPREAD_S, U_radial: float = 0.0, second_order: bool = False,
                wind_speed_mps: float = 10.0):
    """
    The two backscatter lines for one wind direction: approaching (+f_B) and receding (-f_B) waves, with
    sigma0 per line from eq. (12.4), the open-water fraction, and the surface current shift. The lines use
    phi_rel for the approaching waves and phi_rel + pi for the receding ones (cardioid spreading, §12.4).
    second_order=True appends the second-order continuum (second_order_bragg.py, Barrick 1972 eq. 12:31) as
    many weak lines -- same spectrum, wind cutoff g/u^2 from wind_speed_mps, same ice and current factors.
    Returns [(doppler_Hz, sigma0), ...].
    """
    fB = bragg_frequency_hz(f0)
    w = open_water_fraction(c_ice)
    shift = current_doppler_shift_hz(U_radial, f0)
    s_plus = w * bragg_line_sigma0(f0, phi_rel, Bs, s)
    s_minus = w * bragg_line_sigma0(f0, phi_rel + np.pi, Bs, s)
    lines = [(fB + shift, s_plus), (-fB + shift, s_minus)]
    if second_order and w > 0.0:
        from echoes.second_order_bragg import second_order_lines
        lines += [(fd + shift, w * sv) for fd, sv in second_order_lines(f0, phi_rel, wind_speed_mps, B_s=Bs, s=s)]
    return lines


def clutter_field(P_map, t, dopp_hz, rng) -> np.ndarray:
    """
    §12.4.1 step 2 (single element): one Gaussian realisation of the expected (delay, Doppler) power map,
    h(t_n, tau_k) = sum_j a_kj exp(2 pi i f_j t_n) with E|a_kj|^2 = P_kj. Returns h of shape (Nt, Ntau).
    Its power in delay bin k is sum_j P_kj and its Doppler content follows P_kj (the map is per Doppler bin).
    """
    P = np.asarray(P_map, dtype=float)
    f = np.asarray(dopp_hz, dtype=float)
    t = np.asarray(t, dtype=float)
    if P.shape[1] != f.size:
        raise ValueError("P_map must have one column per Doppler bin")
    a = np.sqrt(np.clip(P, 0.0, None) / 2.0) * (rng.standard_normal(P.shape) + 1j * rng.standard_normal(P.shape))
    E = np.exp(2j * np.pi * np.outer(f, t))                    # (Nf, Nt)
    return (a @ E).T                                           # (Nt, Ntau)
