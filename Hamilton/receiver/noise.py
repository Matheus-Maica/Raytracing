"""
noise.py -- receiver noise (spec §12.6, §7.5).

Baseline: independent complex white Gaussian noise per element with power k_B T0 B F_a, where F_a is the
external noise factor (atmospheric, galactic, man-made). F_a comes from ITU-R P.372 for the site, season,
time and frequency. At high latitude the galactic part is attenuated by the same D-region absorption that
riometers measure: it is reduced by the absorption of a representative upward path above the Rx (§12.6).

HiCIRF plan, Stage 6 / Decision 4 ("use ITU-R-HF as a dependency"): galactic_noise_itur_p372_dB() and
man_made_noise_itur_p372_dB() below are the real formulas from ITU-R Recommendation P.372-18 (2026),
Table 1 (downloaded and read directly: https://www.itu.int/dms_pubrec/itu-r/rec/p/
R-REC-P.372-18-202609-I!!PDF-E.pdf), not placeholders.

Atmospheric (lightning) noise -- Part 5 of the same recommendation, the ~75-page world block-data
climatology that made this intractable to transcribe from the PDF directly (it's contour maps, Figs
13a-36a, not tables) -- IS now implemented too: see itu_p372.atmospheric_noise_dB(), a direct port of
ITU-R-HF's own C reference implementation (github.com/programmin1/ITU-R-HF, P372/Src/P372/Noise.c),
validated against that C code compiled and run directly, not just read (see itu_p372.py's docstring).
PLACEHOLDER_F_ATM_DB below is no longer the only option for this term -- it remains only as a fallback
default for callers that don't pass a site/time, kept for backward compatibility.

Directional noise (lightning propagated by sky wave, Coleman) is not included; it would be a noise field on
the (u, v) grid of §12.4.1.
"""

from __future__ import annotations

import numpy as np

from core.constants import C
from core.geometry import WGS84_A

K_BOLTZMANN = 1.380649e-23
T0 = 290.0

# PLACEHOLDER atmospheric (lightning) noise factor (dB above k T0 B) -- see the module docstring for why
# this one specifically is still a placeholder. Replace with ITU-R P.372 Part 5 for the site, season and time.
PLACEHOLDER_F_ATM_DB = 20.0
PLACEHOLDER_F_GAL_DB = 20.0   # superseded by galactic_noise_itur_p372_dB(); kept for existing callers

# ITU-R P.372-18 Table 1: Fam = c - d*log10(f_MHz), valid 0.3-250 MHz for curves A-C, "below fxF2" for D.
_P372_TABLE1 = {
    "city": (76.8, 27.7),
    "residential": (72.5, 27.7),
    "rural": (67.2, 27.7),
    "quiet_rural": (53.6, 28.6),
}
_P372_GALACTIC = (52.0, 23.0)   # Table 1, curve E


def galactic_noise_itur_p372_dB(frequency_mhz: float) -> float:
    """ITU-R P.372-18 Table 1, curve E: Fa = 52.0 - 23.0 log10(f_MHz) dB above k T0 b."""
    c, d = _P372_GALACTIC
    return float(c - d * np.log10(frequency_mhz))


def atmospheric_noise_itur_p372_dB(lat_deg: float, lon_deg: float, utc_hour: float, frequency_mhz: float,
                                   month_index0: int) -> float:
    """
    ITU-R P.372's real atmospheric (lightning) noise factor [dB above k T0 b] at a site, time and
    frequency -- see itu_p372.py. month_index0: 0 = January .. 11 = December. Returns FaA only (the
    median); itu_p372.atmospheric_noise_dB() also gives the upper/lower decile deviations DuA, DlA.
    """
    from receiver.itu_p372 import atmospheric_noise_dB
    FaA, _, _ = atmospheric_noise_dB(lat_deg, lon_deg, utc_hour, frequency_mhz, month_index0)
    return FaA


def man_made_noise_itur_p372_dB(frequency_mhz: float, category: str = "residential") -> float:
    """
    ITU-R P.372-18 Table 1, curves A-D: median outdoor man-made noise factor [dB above k T0 b],
    Fam = c - d log10(f_MHz). category: "city", "residential", "rural" or "quiet_rural". Valid
    0.3-250 MHz (curves A-C); "quiet_rural" (curve D, ionosphere-reflected) is only observed below
    the ionosphere's extraordinary-wave critical frequency fxF2, which this function does not check.
    """
    if category not in _P372_TABLE1:
        raise ValueError(f"category must be one of {sorted(_P372_TABLE1)} -- got {category!r}")
    if not (0.3 <= frequency_mhz <= 250.0):
        import warnings
        warnings.warn(f"frequency_mhz={frequency_mhz} is outside ITU-R P.372-18 Table 1's validity "
                      "range (0.3-250 MHz) for man-made noise")
    c, d = _P372_TABLE1[category]
    return float(c - d * np.log10(frequency_mhz))


def noise_power_W(F_a_dB: float, bandwidth_hz: float, T_0: float = T0) -> float:
    """Per-element noise power k_B T0 B F_a [W], with F_a in dB."""
    return K_BOLTZMANN * T_0 * bandwidth_hz * 10.0 ** (F_a_dB / 10.0)


def external_noise_dB(F_atm_dB: float, F_gal_dB: float, A_up_dB: float = 0.0) -> float:
    """
    Total external noise factor [dB]: atmospheric plus galactic, the galactic part attenuated by the
    upward-path absorption A_up_dB (§12.6). Powers add linearly.
    """
    F_lin = 10.0 ** (F_atm_dB / 10.0) + 10.0 ** ((F_gal_dB - A_up_dB) / 10.0)
    return float(10.0 * np.log10(F_lin))


def upward_path_absorption_dB(ray, r_rx: np.ndarray, *, h_top_m: float = 800e3, n: int = 400,
                              model="default") -> float:
    """
    One-way absorption [dB] of the vertical path from the receiver up to h_top (the galactic-noise path of
    §12.6), integrated with the same absorption rate as the rays (absorption.absorption_rate), O or X as the
    ray's MODE. Straight vertical line; the ray direction is the vertical.
    """
    from raytracing.absorption import absorption_rate
    from raytracing.absorption import DB_PER_NEPER

    r_rx = np.asarray(r_rx, dtype=float)
    up = r_rx / np.linalg.norm(r_rx)
    h0 = np.linalg.norm(r_rx) - WGS84_A
    h_s = np.linspace(0.0, max(h_top_m - h0, 0.0), n)
    pts = r_rx[None, :] + h_s[:, None] * up[None, :]
    u = np.tile(up, (n, 1))
    alpha = absorption_rate(ray, pts, u, model=model, rdot=u)
    return float(DB_PER_NEPER * np.trapz(alpha, h_s))


def noise_samples(n_elements: int, n_samples: int, P_noise_W: float, rng) -> np.ndarray:
    """Independent complex white Gaussian noise, shape (n_elements, n_samples), E|n|^2 = P_noise_W."""
    sigma = np.sqrt(P_noise_W / 2.0)
    return sigma * (rng.standard_normal((n_elements, n_samples)) + 1j * rng.standard_normal((n_elements, n_samples)))
