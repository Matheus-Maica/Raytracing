"""
auroral_scene.py -- direct auroral backscatter by fan integration (spec §12.5, quasi-monostatic baseline).

A fan of rays is traced from the midpoint of the sites (on the batch path, record_states). At every accepted step of every
ray the Bragg vector, the aspect angle and the reflectivity are evaluated, and the volume element is formed from the
ray-tube cross-section of the fan:

    A_perp(s) = | d r/d beta  x  d r/d alpha |         (per unit dbeta dalpha, from the neighbours at equal arc length)
    dV        = A_perp dbeta dalpha ds
    d^2       = A_perp / cos(beta)                      (flux-tube distance, §12.5)
    power     = radar equation with sigma_T = eta dV, losses to the point on both legs (plus L_pol per leg)
    delay     = 2 P' / c                                (quasi-monostatic)
    Doppler   = q_B . v_ph / (2 pi)  (+ twice the ionospheric Doppler, zero here: placeholder)
    direction = the launch direction of the ray

The Bragg-scale inputs (eta_perp, A_s, v_ph) are USER INPUTS (inputs.BraggInputs); the defaults are placeholders.
The bistatic refinement (fans from both sites with 3-D interpolation) is not done.

HiCIRF scoping note (plan, Stage 7, Decision 2): direct backscatter from field-aligned ionization structure is
explicitly listed as future work NOT YET in HiCIRF as of Nickisch et al. (2012), paragraph 35. Nothing in
api.py/cir.py calls auroral_components() automatically -- it is opt-in by construction (only this module's own
tests call it) -- so a scene assembled without calling it already matches HiCIRF's 2012 feature set on this
point; no gating flag is needed on top of that. Call it explicitly when you want this Hamilton extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echoes import auroral
from propagation import legs
from raytracing.batch import samples_from_record
from core.constants import C
from core.inputs import BraggInputs, RadarInputs
from propagation.legs import L_POL_DB
from echoes.twoway import radar_power_dBW


@dataclass
class AuroralComponent:
    power_W: float
    delay_s: float
    doppler_Hz: float
    psi_deg: float
    eta: float
    dV: float
    position_ecef: np.ndarray
    e_launch: np.ndarray


def trace_midpoint_fan(tracer, r0, up, north, east, elev_grid, az_grid, max_path_m, *, nhops: int = 1,
                       ceiling_alt_m: float = 800e3):
    """Trace the fan from the midpoint with recording. Returns (samples[i, j], elev, az) on the grid (Nb, Na)."""
    elev_grid = np.asarray(elev_grid, dtype=float)
    az_grid = np.asarray(az_grid, dtype=float)
    E, A = np.meshgrid(elev_grid, az_grid, indexing="ij")
    dirs = np.array([legs.launch_direction(e, a, up, north, east) for e, a in zip(E.ravel(), A.ravel())])
    res = tracer.trace(r0, dirs, max_path_m, nhops=nhops, ceiling_alt_m=ceiling_alt_m, want_group=True,
                       want_absorption=True, record_states=True)
    samples = [samples_from_record(tracer, rec, transport=False) for rec in res.records]
    return samples, E, A


def _neighbour_position(sam_nb, s_query):
    return np.stack([np.interp(s_query, sam_nb.s, sam_nb.r[:, k]) for k in range(3)], axis=-1)


def auroral_components(tracer, samples, elev_grid, az_grid, *, up, north, east, f0: float, bragg: BraggInputs,
                       radar: RadarInputs, eta_threshold: float = 0.0) -> list[AuroralComponent]:
    """Volume elements of the fan with eta above eta_threshold (absolute, m^-1), as components."""
    Nb, Na = len(elev_grid), len(az_grid)
    dbeta = np.deg2rad(np.abs(np.gradient(np.asarray(elev_grid, dtype=float))).mean())
    dalpha = np.deg2rad(np.abs(np.gradient(np.asarray(az_grid, dtype=float))).mean())
    k0 = 2.0 * np.pi * f0 / C
    v_ph = np.asarray(bragg.v_ph_mps, dtype=float)
    out: list[AuroralComponent] = []
    for i in range(Nb):
        for j in range(Na):
            sam = samples[i * Na + j]
            ion = (~sam.vacuum) & (sam.Ne > 0.0)
            if not np.any(ion):
                continue
            # neighbours at the same arc length (backward difference on the grid edges)
            i2 = i + 1 if i + 1 < Nb else i - 1
            j2 = j + 1 if j + 1 < Na else j - 1
            s = sam.s
            r_b = _neighbour_position(samples[i2 * Na + j], s)
            r_a = _neighbour_position(samples[i * Na + j2], s)
            dr_b = (r_b - sam.r) / ((i2 - i) * dbeta)
            dr_a = (r_a - sam.r) / ((j2 - j) * dalpha)
            A_perp = np.linalg.norm(np.cross(dr_b, dr_a), axis=-1)
            ds = np.gradient(s)
            dV = A_perp * dbeta * dalpha * ds
            n = sam.n
            u_hat = sam.u_hat
            q = -2.0 * n[:, None] * k0 * u_hat
            q_hat = q / np.linalg.norm(q, axis=-1, keepdims=True)
            B = sam.B
            B_hat = B / np.maximum(np.linalg.norm(B, axis=-1, keepdims=True), 1e-300)
            cos_qB = np.abs(np.sum(q_hat * B_hat, axis=-1))
            psi = np.arcsin(np.clip(cos_qB, 0.0, 1.0))
            eta = bragg.eta_perp_m1 * 10.0 ** (-bragg.A_s_db_per_deg * np.degrees(psi) / 10.0)
            # ray launch elevation for d^2 = A_perp / cos(beta)
            beta = np.deg2rad(float(elev_grid[i]))
            d2 = np.maximum(A_perp, 1e-300) / np.cos(beta)
            d = np.sqrt(d2)
            loss = sam.absorption_db + L_POL_DB                       # one-way to the point, plus pol per leg
            sigma = np.maximum(eta * dV, 1e-300)
            dBW = radar_power_dBW(radar.Pt_dBW, radar.Gt_dB, radar.Gr_dB, f0, sigma, d, d, loss, loss)
            power = 10.0 ** (np.asarray(dBW) / 10.0)
            delay = 2.0 * sam.Pg / C
            doppler = q @ v_ph / (2.0 * np.pi)
            keep = ion & (eta > eta_threshold) & np.isfinite(power) & (dV > 0.0)
            e_launch = legs.launch_direction(float(elev_grid[i]), float(az_grid[j]), up, north, east)
            for k in np.flatnonzero(keep):
                out.append(AuroralComponent(power_W=float(power[k]), delay_s=float(delay[k]),
                                            doppler_Hz=float(doppler[k]), psi_deg=float(np.degrees(psi[k])),
                                            eta=float(eta[k]), dV=float(dV[k]),
                                            position_ecef=sam.r[k].copy(), e_launch=e_launch))
    return out
