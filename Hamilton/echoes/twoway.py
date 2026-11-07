"""
twoway.py -- two-way point-target components from an outbound and a return leg (spec §6.1-6.3,
§12.1, §12.3) and the reciprocity diagnostic (§12.2, eq. 12.3).

A target component pairs an outbound leg (Tx to T) with a return leg (Rx to T), each a LegMode.
  delay    tau = (P'_out + P'_ret) / c                      (§6.1, §12.3)
  Doppler  f_D = f_D,out + f_D,ret + f_T                    (§6.1), f_T from eq. (6.1)
  phase    -k0 (P_out + P_ret)                              (eq. 6.4; ground phases not included here)
  power    radar equation (6.2) with the user's sigma_T and the flux-tube distances d_T, d_R
  range migration  tau_dot = -f_T lambda0 / c               (§12.3)

No coherence term (§6.3). The power is a deterministic received power before stochastic spreading.
Legs are not searched here: homing (homing.py) or the fans (fans.py) supply them.
"""

from __future__ import annotations

import numpy as np

from core.constants import C
from propagation.legs import L_POL_DB
from core.records import Component, LegMode


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def target_doppler_hz(v_T, k_i, s_f, lam0: float) -> float:
    """
    Eq. (6.1): f_T = (k_s - k_i) . v_T / (2 pi) with k_s = -k0 s_f (§6.2: the scattered direction
    toward the receiver is minus the return leg's direction at the scatterer). With k_i = k0 k_hat_i:

        f_T = -(k_hat_i + s_f) . v_T / lambda0,

    which reduces to -2 v_T . k_hat_i / lambda0 for a monostatic radar (s_f = k_hat_i). A receding
    target gives a negative Doppler.
    """
    return float(-np.dot(_unit(k_i) + _unit(s_f), np.asarray(v_T, dtype=float)) / lam0)


def range_migration_rate(f_T: float, lam0: float) -> float:
    """§12.3: the delay drifts at tau_dot = -f_T lambda0 / c (motion where n = 1)."""
    return -f_T * lam0 / C


def radar_power_dBW(Pt_dBW: float, Gt_dB: float, Gr_dB: float, f0: float, sigma_T_m2: float,
                    d_T: float, d_R: float, loss_T_dB: float, loss_R_dB: float) -> float:
    """
    Eq. (6.2): P_r = P_t G_t G_r lambda0^2 sigma_T / ((4 pi)^3 d_T^2 d_R^2 L_T L_R), in dBW.
    Losses are the per-leg dB totals of eq. (6.3) (absorption, ground, polarization, other).
    """
    lam = C / f0
    return (Pt_dBW + Gt_dB + Gr_dB + 20.0 * np.log10(lam) + 10.0 * np.log10(sigma_T_m2)
            - 30.0 * np.log10(4.0 * np.pi) - 20.0 * np.log10(d_T) - 20.0 * np.log10(d_R)
            - loss_T_dB - loss_R_dB)


def leg_loss_dB(leg: LegMode, L_pol_dB: float = L_POL_DB) -> float:
    """Total dB loss of a leg, eq. (6.3): absorption + ground losses + polarization."""
    return float(leg.absorption_dB + leg.ground_loss_dB + L_pol_dB)


def target_component(out: LegMode, ret: LegMode, *, f0: float, v_T, k_i, s_f, sigma_T_m2: float,
                     Pt_dBW: float, Gt_dB: float = 0.0, Gr_dB: float = 0.0,
                     L_pol_dB: float = L_POL_DB) -> Component:
    """
    One two-way point-target component (spec §12.3) from a pair of homed legs. The outbound
    incident direction k_i and the return leg direction s_f at the target come from the legs'
    k_far fields and the caller's geometry (k_i = out.k_far, s_f = ret.k_far for a monostatic
    check; in general the return leg's far-end direction).
    """
    lam0 = C / f0
    k0 = 2.0 * np.pi / lam0
    f_T = target_doppler_hz(v_T, k_i, s_f, lam0)
    d_T, d_R = out.d_eff, ret.d_eff
    dBW = radar_power_dBW(Pt_dBW, Gt_dB, Gr_dB, f0, sigma_T_m2, d_T, d_R,
                          leg_loss_dB(out, L_pol_dB), leg_loss_dB(ret, L_pol_dB))
    return Component(
        kind="target", out=out, ret=ret,
        scatterer={"sigma_T_m2": sigma_T_m2, "v_T": np.asarray(v_T, dtype=float), "f_T": f_T,
                   "range_migration_s_per_s": range_migration_rate(f_T, lam0)},
        power=float(10.0 ** (dBW / 10.0)),
        delay=float((out.Pg + ret.Pg) / C),
        doppler=float(out.doppler + ret.doppler + f_T),
        phase=float(-k0 * (out.P + ret.P)),
    )


def gamma_or(structure_values) -> float:
    """
    Eq. (12.3): Gamma_or = exp(-(1/2) sum_i D_i(Delta_i)), with D_i(Delta_i) the structure function
    of screen i evaluated at the transverse separation to the other leg. Gamma < 0.1 means
    independent legs (the HiCIRF baseline), Gamma > 0.9 reciprocal legs. Returns Gamma.
    """
    D = np.asarray(structure_values, dtype=float)
    return float(np.exp(-0.5 * np.sum(D)))


def _interp_rows(arr: np.ndarray, s_grid: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return np.stack([np.interp(s_query, s_grid, arr[:, k]) for k in range(arr.shape[1])], axis=-1)


def gamma_or_legs(out, ret, screens) -> float:
    """
    Reciprocity diagnostic of §12.2, eq. (12.3). For every screen i of the outbound leg (at arc length
    z_i from the site, the array end), the transverse separation Delta_i in that screen's frame (t1, t2
    of the outbound ray there) between the piercing point and the nearest point of the return leg; then
    Gamma_or = exp(-1/2 sum_i D_i(Delta_i)) with D_i the screen's structure function (screens.struct_fn).
    out, ret: RaySamples of the outbound and return legs (out needs the transverse frame t1, t2).
    screens: the ScreenSet of the outbound leg (build_screens).
    """
    from stochastic.screens import struct_fn
    if out.t1 is None or out.t2 is None:
        raise ValueError("the outbound leg needs its transported transverse frame (records.RaySamples.t1/t2)")
    z = np.asarray(screens.z, dtype=float)                       # group-path distance from the site (§9.3)
    s_out = np.asarray(out.Pg if out.Pg is not None else out.s, dtype=float)
    r_i = _interp_rows(out.r, s_out, z)
    t1_i = _interp_rows(out.t1, s_out, z)
    t2_i = _interp_rows(out.t2, s_out, z)
    ret_r = np.asarray(ret.r, dtype=float)
    d_all = ret_r[None, :, :] - r_i[:, None, :]
    j = np.argmin(np.linalg.norm(d_all, axis=-1), axis=1)
    d = d_all[np.arange(z.size), j]
    delta = np.stack([np.sum(d * t1_i, axis=-1), np.sum(d * t2_i, axis=-1)], axis=-1)
    D = np.array([struct_fn(delta[i:i + 1], screens.W_inv[i][None], screens.sigma2[i],
                            screens.L0[i], screens.p[i])[0] for i in range(z.size)])
    return gamma_or(D)


def reciprocity_class(gamma: float) -> str:
    """§12.2 classes: 'independent' (< 0.1), 'reciprocal' (> 0.9), otherwise 'partial' (flagged)."""
    if gamma < 0.1:
        return "independent"
    if gamma > 0.9:
        return "reciprocal"
    return "partial"
