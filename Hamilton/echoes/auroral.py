"""
auroral.py -- direct auroral backscatter (spec §12.5, decision for Phase 6a): the Bragg vector of the ray,
the aspect angle to the field, the aspect-sensitivity law, the power of a volume element, and the Doppler of
a volume element on the quasi-monostatic baseline. The fan-based volume integration (trace from the midpoint
of the sites, deposit into the clutter field) is not implemented here.

Bragg-scale irregularity parameters (eta_perp, the aspect sensitivity A_s, the phase velocity v_ph) come from
the irregularity model (Listing 3, bragg): they are PLACEHOLDER inputs until the user supplies them (D9).
"""

from __future__ import annotations

import numpy as np

from core.constants import C
from echoes.twoway import radar_power_dBW


def bragg_vector_backscatter(n: float, k0: float, u_hat) -> np.ndarray:
    """q_B = -2 n k0 u_hat for backscatter (§12.5)."""
    return -2.0 * n * k0 * np.asarray(u_hat, dtype=float)


def aspect_angle_rad(q_B, e_B) -> float:
    """psi_a = arcsin |q_hat_B . e_B|: zero when the Bragg vector is perpendicular to B (perfect aspect)."""
    q = np.asarray(q_B, dtype=float)
    e = np.asarray(e_B, dtype=float)
    cosv = abs(np.dot(q, e)) / (np.linalg.norm(q) * np.linalg.norm(e))
    return float(np.arcsin(np.clip(cosv, 0.0, 1.0)))


def eta_aspect(psi_rad: float, eta_perp: float, A_s_dB_per_deg: float) -> float:
    """
    Reflectivity with the aspect law (§12.5): eta = eta_perp 10^(-A_s |psi_a|/10), psi_a in degrees.
    eta_perp in m^-1 (the Born reflectivity per unit volume at perfect aspect).
    """
    psi_deg = np.degrees(abs(psi_rad))
    return float(eta_perp * 10.0 ** (-A_s_dB_per_deg * psi_deg / 10.0))


def volume_element(A_perp: float, dbeta: float, dalpha: float, ds: float) -> float:
    """dV = A_perp dbeta dalpha ds (A_perp: ray-tube cross-section per unit dbeta dalpha, §12.5)."""
    return A_perp * dbeta * dalpha * ds


def volume_power_dBW(*, Pt_dBW: float, Gt_dB: float, Gr_dB: float, f0: float, eta: float, dV: float,
                     d_T: float, d_R: float, loss_T_dB: float, loss_R_dB: float) -> float:
    """
    Received power of one volume element (§12.5): dP_r = P_t G_t G_r lambda0^2 eta dV / ((4 pi)^3 d_T^2 d_R^2
    L_T L_R). This is the radar equation with sigma_T = eta dV (eta is a cross-section per unit volume).
    """
    return radar_power_dBW(Pt_dBW, Gt_dB, Gr_dB, f0, eta * dV, d_T, d_R, loss_T_dB, loss_R_dB)


def auroral_doppler_hz(q_B, v_ph, f_iono_Hz: float) -> float:
    """Quasi-monostatic Doppler of a volume element: q_B . v_ph / 2 pi plus twice the ionospheric Doppler (§12.5)."""
    return float(np.dot(np.asarray(q_B, dtype=float), np.asarray(v_ph, dtype=float)) / (2.0 * np.pi)
                 + 2.0 * f_iono_Hz)


def wavelength_m(f0: float) -> float:
    return C / f0
