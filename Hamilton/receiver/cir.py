"""
cir.py -- assembly of the output channel (spec §12.7), the received-signal helper, and the output writer
(§14.2, npz + JSON: the HDF5 fallback of decision D10, since h5py is not installed).

    h_pq(t, tau) = sum over components of h_c(t, tau) (+ surface and auroral fields), on an absolute delay
    grid tau_k = tau_0 + k dtau and the slow-time grid t_n (eq. 12.5).

A component contributes  sqrt(P_c) exp(i phi_c) exp(2 pi i f_c t) delta(tau - tau_c(t)), where tau_c(t) is the
range-migrating delay of §12.3 and the delta is band-limited (sinc) on the grid. Noise is added to the received
signal, not to h (§12.7):  y(t) = sum_k h(t, tau_k) x(t - tau_k) dtau + n(t).

Convention (spec §2.3): e^{+i omega t}; the baseband phase of a path is -k0 P; f_D = -(1/lambda0) dP/dt.
Units: h is sqrt(W) per component (received amplitude, power |h|^2).
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess

import numpy as np

from receiver import noise
from echoes.targets import component_track
from echoes.twoway import reciprocity_class

REPO = pathlib.Path(__file__).resolve().parent.parent      # the Hamilton repository root


def _require_finite_doppler(comps) -> None:
    for c in comps:
        if not np.isfinite(c.doppler):
            raise ValueError("a component has no finite Doppler: build the legs with a Doppler accumulator "
                             "(batch.DopplerRates) before assembling the channel")


def target_impulse_response(comps, t, tau_grid) -> np.ndarray:
    """h (Nt, Ntau) from point-target components (§12.3): range migration, Doppler and carrier phase."""
    t = np.asarray(t, dtype=float)
    tau = np.asarray(tau_grid, dtype=float)
    dtau = tau[1] - tau[0]
    _require_finite_doppler(comps)
    h = np.zeros((t.size, tau.size), dtype=complex)
    for c in comps:
        tau_c, fD = component_track(c, t)                       # (Nt,) delays, Doppler [Hz]
        amp = np.sqrt(c.power) * np.exp(1j * c.phase)
        dop = np.exp(2j * np.pi * fD * t)
        band = np.sinc((tau[None, :] - tau_c[:, None]) / dtau)  # band-limited delta at the moving delay
        h += amp * dop[:, None] * band
    return h


def assemble_cir(comps, t, tau_grid, extra=None) -> np.ndarray:
    """
    Eq. (12.5): sum of the target responses plus any other field on the same grid (surface clutter,
    auroral backscatter) passed as `extra` with shape (Nt, Ntau).
    """
    h = target_impulse_response(comps, t, tau_grid)
    if extra is not None:
        extra = np.asarray(extra, dtype=complex)
        if extra.shape != h.shape:
            raise ValueError(f"extra field has shape {extra.shape}, expected {h.shape}")
        h = h + extra
    return h


def received_signal(h, t, tau_grid, x, *, noise_power_W: float = 0.0, rng=None) -> np.ndarray:
    """
    Received signal of one element (§12.7): y(t_n) = sum_k h(t_n, tau_k) x(t_n - tau_k) dtau + n(t_n).
    x: callable giving the fast-time waveform at the given times (zero outside its support).
    Fast time is taken on the slow-time grid t.
    """
    t = np.asarray(t, dtype=float)
    tau = np.asarray(tau_grid, dtype=float)
    dtau = tau[1] - tau[0]
    X = x(t[:, None] - tau[None, :])
    y = np.sum(np.asarray(h) * X, axis=1) * dtau
    if noise_power_W > 0.0:
        rng = rng if rng is not None else np.random.default_rng()
        y = y + noise.noise_samples(1, t.size, noise_power_W, rng)[0]
    return y


def fmcw_sweep(tau, bandwidth_hz: float, duration_s: float):
    """Linear FM sweep, x(s) = exp(i pi (B/T) s^2) on 0 <= s < T, zero elsewhere (fast time s)."""
    s = np.asarray(tau, dtype=float)
    beta = bandwidth_hz / duration_s
    return np.where((s >= 0.0) & (s < duration_s), np.exp(1j * np.pi * beta * s ** 2), 0.0)


def pulse_train(tau, width_s: float, period_s: float, n_pulses: int):
    """Rectangular pulse train: n_pulses pulses of width_s at spacing period_s, from s = 0."""
    s = np.asarray(tau, dtype=float)
    k = np.floor(s / period_s)
    inside = (s >= 0.0) & (k < n_pulses) & ((s - k * period_s) < width_s)
    return inside.astype(complex)


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _clean(x):
    """JSON-safe: numpy scalars to Python, NaN and inf to None, arrays to lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _leg_row(leg) -> dict:
    return dict(site=leg.site, mode=leg.mode, label=leg.label, n_hops=leg.n_hops,
                beta_deg=float(np.degrees(leg.beta)), alpha_deg=float(np.degrees(leg.alpha)),
                delay_s=leg.delay, doppler_Hz=leg.doppler, absorption_dB=leg.absorption_dB,
                absorption_D_dB=leg.absorption_D_dB, ground_loss_dB=leg.ground_loss_dB,
                d_eff_m=leg.d_eff, caustic_flag=leg.caustic_flag, xmode_validated=leg.xmode_validated,
                attenuation_flag=leg.attenuation_flag)


def write_cir(stem, *, h, t, tau_grid, comps, f0: float, models: dict | None = None,
              seeds: dict | None = None, options: dict | None = None) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Write <stem>.npz (h, t, tau, and the component columns) and <stem>.json (conventions, component and
    leg tables, environment model identifiers, seeds, options, provenance). Returns the two paths.
    """
    stem = pathlib.Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(t, dtype=float)
    tau = np.asarray(tau_grid, dtype=float)
    cols = dict(
        power_W=np.array([c.power for c in comps], dtype=float),
        delay_s=np.array([c.delay for c in comps], dtype=float),
        doppler_Hz=np.array([c.doppler for c in comps], dtype=float),
        phase_rad=np.array([c.phase for c in comps], dtype=float),
        gamma_or=np.array([np.nan if c.gamma_or is None else c.gamma_or for c in comps], dtype=float),
    )
    npz_path = stem.with_suffix(".npz")
    np.savez_compressed(npz_path, h=np.asarray(h), t=t, tau=tau, **cols)

    seen, legs_rows = set(), []
    for c in comps:
        for leg in (c.out, c.ret):
            if id(leg) not in seen:
                seen.add(id(leg))
                legs_rows.append(_leg_row(leg))
    comp_rows = []
    for c in comps:
        comp_rows.append(dict(kind=c.kind, out_label=c.out.label, ret_label=c.ret.label,
                              power_W=c.power, delay_s=c.delay, doppler_Hz=c.doppler, phase_rad=c.phase,
                              gamma_or=c.gamma_or,
                              reciprocity=None if c.gamma_or is None else reciprocity_class(c.gamma_or),
                              f_T_Hz=c.scatterer.get("f_T"),
                              range_migration_s_per_s=c.scatterer.get("range_migration_s_per_s")))
    doc = dict(
        conventions="e^{+i omega t} (spec §2.3); baseband phase -k0 P; f_D = -(1/lambda0) dP/dt",
        h_units="sqrt(W) per component (received amplitude; power = |h|^2)",
        carrier_Hz=f0,
        slow_time_step_s=float(t[1] - t[0]) if t.size > 1 else None,
        delay_origin_s=float(tau[0]),
        delay_step_s=float(tau[1] - tau[0]) if tau.size > 1 else None,
        components=comp_rows,
        legs=legs_rows,
        models=models or {},
        seeds=seeds or {},
        options=options or {},
        provenance=dict(code_version=_git_head(),
                        written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        arrays=sorted(["h", "t", "tau"] + list(cols))),
    )
    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(_clean(doc), indent=2))
    return npz_path, json_path
