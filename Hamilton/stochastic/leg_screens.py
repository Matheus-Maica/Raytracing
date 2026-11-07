"""
leg_screens.py -- screens along a traced leg (spec §9.1-9.3): the bridge from RaySamples to build_screens, so that the
reciprocity diagnostic Gamma_or (§12.2, twoway.gamma_or_legs) can be evaluated for homed legs.

The scalar xi/L0/p/a/b/v_d kwargs are PLACEHOLDERS (decision D9), sourced from inputs.IrregularityInputs (the single
place these defaults are defined -- see its docstring). Pass irregularity_model / drift_model instead (any object
satisfying environment.py's IrregularityModel / DriftModel Protocol) to query real, position- and time-varying models
per sample instead of one uniform value for the whole leg; the scalar kwargs are then ignored. The geometry (screen
positions in group path, transverse frames, B direction, phase sensitivity g, refractive index) always comes from the
traced ray.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.inputs import IrregularityInputs
from stochastic.screens import build_screens

_DEFAULTS = IrregularityInputs()
PLACEHOLDER_XI = _DEFAULTS.xi               # fractional rms density fluctuation xi = sigma_N / N_e (spec §8.4 range 1e-3 to 1e-1)
PLACEHOLDER_L0 = _DEFAULTS.L0_m             # outer scale [m]
PLACEHOLDER_P = _DEFAULTS.p                 # 3-D spectral index (spec: 3.05 < p < 5.95)
PLACEHOLDER_A = _DEFAULTS.axial_ratio_B     # axial ratio along B
PLACEHOLDER_B = _DEFAULTS.axial_ratio_sheet  # axial ratio along the sheet axis


def _unit(v: np.ndarray, fallback) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    fb = np.broadcast_to(np.asarray(fallback, dtype=float), v.shape)
    return np.where(n > 0.0, v / np.where(n > 0.0, n, 1.0), fb)


def _coarsen_by_elevation(u_hat: np.ndarray, keep: np.ndarray, step_deg: float) -> np.ndarray:
    """
    HiCIRF paragraph 17: "a screen is placed whenever the ray elevation angle has changed by 5 deg"
    (default HiCIRF value). Keeps the first and last of `keep`, and every index in between whose ray
    direction has rotated by at least step_deg from the last kept one -- an angular-change filter on
    u_hat, not literally "elevation" (which would also need to exclude azimuth rotation; the ray stays
    close to a single vertical plane for most legs, so the two are nearly the same thing in practice).
    """
    u = np.asarray(u_hat, dtype=float)[keep]
    cos_step = np.cos(np.deg2rad(step_deg))
    out = [int(keep[0])]
    last_dir = u[0]
    for i in range(1, keep.size - 1):
        if np.dot(u[i], last_dir) < cos_step:
            out.append(int(keep[i]))
            last_dir = u[i]
    out.append(int(keep[-1]))
    return np.array(sorted(set(out)), dtype=keep.dtype)


def screens_for_leg(samples, ray, *, xi: float = PLACEHOLDER_XI, L0: float = PLACEHOLDER_L0,
                    p: float = PLACEHOLDER_P, a: float = PLACEHOLDER_A, b: float = PLACEHOLDER_B,
                    v_d=None, sigma_v_frac: float = 0.2, irregularity_model=None, drift_model=None, t=None,
                    coarsen_elevation_deg: Optional[float] = None):
    """
    ScreenSet of one traced leg: one screen per ionospheric sample (vacuum samples excluded), at the group-path
    distance z from the array end (the site), with thickness equal to the spacing of the samples.
    samples: RaySamples with transverse frames t1, t2 (batch.samples_from_record(..., transport=True)).
    ray: the main.Ray of the leg (for the phase sensitivity g).

    irregularity_model, drift_model: optional environment.IrregularityModel / DriftModel instances, queried
    per sample point as model.at(r_ecef, t); when given, these replace the scalar xi/L0/p/a/b/v_d/sigma_v_frac
    kwargs entirely (which keep their old, uniform-over-the-leg behaviour when the models are not given, so
    every existing caller is unaffected).

    coarsen_elevation_deg: None (default) keeps the current one-screen-per-sample placement (finer than
    HiCIRF's own rule, and the default to leave Hamilton's own test vectors unaffected); pass 5.0 to match
    HiCIRF's own screen-placement rule exactly (plan, Stage 3, Decision 3).
    """
    keep = np.flatnonzero((~samples.vacuum) & (samples.Ne > 0.0))
    if keep.size < 2:
        raise ValueError("the leg has fewer than two ionospheric samples")
    if samples.t1 is None or samples.t2 is None:
        raise ValueError("the leg needs its transported frame (samples_from_record(..., transport=True))")
    if coarsen_elevation_deg is not None:
        keep = _coarsen_by_elevation(samples.u_hat, keep, coarsen_elevation_deg)
        if keep.size < 2:
            raise ValueError("the leg has fewer than two screens after elevation coarsening")
    z = np.asarray(samples.Pg, dtype=float)[keep]
    ds = np.gradient(z)
    ds = np.where(ds > 0.0, ds, np.median(ds[ds > 0.0]) if np.any(ds > 0.0) else 1.0)
    n = np.asarray(samples.n, dtype=float)[keep]
    u_state = np.asarray(samples.u_hat, dtype=float)[keep] * n[:, None]
    state6 = np.concatenate([np.asarray(samples.r, dtype=float)[keep], u_state], axis=1)
    g = np.abs(np.asarray(ray.phase_sensitivity(state6), dtype=float))
    B = np.asarray(samples.B, dtype=float)[keep]
    B_hat = _unit(B, [0.0, 0.0, 1.0])
    s_hat = _unit(np.asarray(samples.s_hat, dtype=float)[keep], [1.0, 0.0, 0.0])
    e_sh = _unit(np.cross(B_hat, s_hat), [0.0, 1.0, 0.0])
    N = keep.size
    r_ecef = np.asarray(samples.r, dtype=float)[keep]
    Ne = np.asarray(samples.Ne, dtype=float)[keep]

    if drift_model is not None:
        d = drift_model.at(r_ecef, t)
        v = np.broadcast_to(np.asarray(d["v_d"], dtype=float), (N, 3)).copy()
        sigma_v = np.broadcast_to(np.asarray(d["sigma_v"], dtype=float), (N,)).copy()
    else:
        v = np.zeros((N, 3)) if v_d is None else np.broadcast_to(np.asarray(v_d, dtype=float), (N, 3)).copy()
        sigma_v = sigma_v_frac * np.linalg.norm(v, axis=-1)

    if irregularity_model is not None:
        irr = irregularity_model.at(r_ecef, t)
        sigma_N = (np.broadcast_to(np.asarray(irr["sigma_N"], dtype=float), (N,)) if "sigma_N" in irr
                   else np.broadcast_to(np.asarray(irr["xi"], dtype=float), (N,)) * Ne)
        L0_arr = np.broadcast_to(np.asarray(irr["L0"], dtype=float), (N,))
        p_arr = np.broadcast_to(np.asarray(irr["p"], dtype=float), (N,))
        a_arr = np.broadcast_to(np.asarray(irr["a"], dtype=float), (N,))
        b_arr = np.broadcast_to(np.asarray(irr["b"], dtype=float), (N,))
    else:
        sigma_N = xi * Ne
        L0_arr, p_arr, a_arr, b_arr = np.full(N, L0), np.full(N, p), np.full(N, a), np.full(N, b)

    return build_screens(
        z=z, ds=ds, g=g, sigma_N=sigma_N, L0=L0_arr, p=p_arr, a=a_arr, b=b_arr, n=n,
        t1=np.asarray(samples.t1, dtype=float)[keep], t2=np.asarray(samples.t2, dtype=float)[keep],
        B_hat=B_hat, e_sh=e_sh, v_d=v, sigma_v=sigma_v, Z_L=float(samples.Pg[keep[-1]]))
