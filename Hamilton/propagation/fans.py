"""
fans.py -- launch fans, landing maps, the quad inversion and mode classification (spec §5.1,
§5.2, §5.4).

A fan is a regular grid of launch directions (elevation beta, azimuth alpha) traced in one batch.
For each hop h the landing points form a map x_L,h(i, j) that is many-to-one: low and high rays,
E and F reflections and hop counts are separate sheets. Algorithm 2 finds every launch angle
whose landing point is a given ground point by inverting each quadrilateral cell bilinearly. A
cell that a caustic crosses is subdivided (five extra rays, depth <= 4) and its results flagged.

Planar coordinates are azimuthal equidistant about the site on the geocentric sphere (R_MEAN).
Heights are above the WGS84 ellipsoid. The E/F split uses a single valley altitude (placeholder
until the background profile analysis, spec §5.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.geometry import R_MEAN
from propagation.legs import VALLEY_M, layer_letter, launch_direction

SUB_DEPTH_MAX = 4          # Algorithm 2: subdivision depth
DEDUP_DEG = 0.02           # §5.4: same label and |d beta|, |d alpha| below this is the same mode
_NV = 8                    # per-hop values: x, y, P, P', A_db, Doppler, valid, apex


def site_planar(r_site, r_pts) -> np.ndarray:
    """Azimuthal equidistant (east, north) coordinates about the site [m], shape (..., 2)."""
    r_site = np.asarray(r_site, dtype=float)
    p0 = r_site / np.linalg.norm(r_site)
    east = np.cross([0.0, 0.0, 1.0], p0)
    east /= np.linalg.norm(east)
    north = np.cross(p0, east)
    p = np.asarray(r_pts, dtype=float)
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    cosg = np.clip(p @ p0, -1.0, 1.0)
    t = p - cosg[..., None] * p0
    sin_g = np.linalg.norm(t, axis=-1)
    gamma = np.arccos(cosg)
    safe = np.where(sin_g > 1e-15, sin_g, 1.0)
    scale = np.where(sin_g > 1e-15, R_MEAN * gamma / safe, R_MEAN)
    return np.stack([t @ east, t @ north], axis=-1) * scale[..., None]


@dataclass
class FanResult:
    """
    Per-ray, per-hop results of one batch trace (spec §5.1). Arrays are (N, H, ...) with N the
    number of rays in launch order. For a grid fan the order is row-major over (beta, alpha) and
    grid_shape = (N_beta, N_alpha). NaN where a hop was not reached.
    """

    elev_deg: np.ndarray            # (N,)
    az_deg: np.ndarray              # (N,)
    grid_shape: tuple | None
    hops: np.ndarray                # (N,) hops reached
    valid: np.ndarray               # (N, H) bool
    land: np.ndarray                # (N, H, 3) ECEF landing point of hop h
    planar: np.ndarray              # (N, H, 2) east/north about the site [m]
    P: np.ndarray                   # (N, H) cumulative phase path [m]
    Pg: np.ndarray                  # (N, H) cumulative group path [m]
    A_db: np.ndarray                # (N, H) cumulative one-way absorption [dB]
    doppler_hz: np.ndarray          # (N, H) Doppler of hop h [Hz] (NaN without a Doppler tracer)
    apex_m: np.ndarray              # (N, H) apex height of hop h above the ellipsoid [m]
    u_inc: np.ndarray               # (N, H, 3) wave normal of the incident ray at landing
    s_inc: np.ndarray               # (N, H, 3) physical incident direction at landing
    mode: int = 1


def fan_from_result(tracer, res, r0, elev, az, H: int, grid_shape=None) -> FanResult:
    """Pack a BatchResult (with hop_states and hop_apex) into a FanResult."""
    N = elev.size
    valid = np.zeros((N, H), dtype=bool)
    land = np.full((N, H, 3), np.nan)
    planar = np.full((N, H, 2), np.nan)
    P = np.full((N, H), np.nan)
    Pg = np.full((N, H), np.nan)
    A_db = np.full((N, H), np.nan)
    dop = np.full((N, H), np.nan)
    apex = np.full((N, H), np.nan)
    u_inc = np.full((N, H, 3), np.nan)
    s_inc = np.full((N, H, 3), np.nan)
    rows, hs, states, apexes, dops = [], [], [], [], []
    for i in range(N):
        st = res.hop_states[i]
        for h in range(min(len(st), H)):
            rows.append(i)
            hs.append(h)
            states.append(st[h])
            apexes.append(res.hop_apex[i][h])
            dops.append(res.doppler_hops_hz[i][h] if res.doppler_hops_hz else np.nan)
    hops = np.array([len(res.hop_states[i]) for i in range(N)], dtype=int)
    if rows:
        I = np.array(rows)
        K = np.array(hs)
        S = np.array(states)
        valid[I, K] = True
        land[I, K] = S[:, :3]
        u_inc[I, K] = S[:, 3:6]
        P[I, K] = S[:, 7]
        Pg[I, K] = S[:, 8]
        A_db[I, K] = 8.686 * S[:, 9]
        apex[I, K] = np.array(apexes)
        dop[I, K] = np.array(dops)
        planar[I, K] = site_planar(r0, S[:, :3])
        d = tracer.ray._haselgrove_physical_rhs(0.0, S[:, :6])[..., :3]
        s_inc[I, K] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return FanResult(elev_deg=elev, az_deg=az, grid_shape=grid_shape, hops=hops, valid=valid,
                     land=land, planar=planar, P=P, Pg=Pg, A_db=A_db, doppler_hz=dop,
                     apex_m=apex, u_inc=u_inc, s_inc=s_inc, mode=int(tracer.ray.MODE))


def trace_fan(tracer, r0, up, north, east, elev_deg, az_deg, max_path_m: float, *, nhops: int = 1,
              grid: bool = False, ceiling_alt_m: float = 800e3) -> FanResult:
    """
    Trace a launch fan in one batch. grid=True takes the outer product of elev_deg and az_deg
    (row-major), which the quad inversion needs; grid=False traces the listed directions.
    """
    e = np.atleast_1d(np.asarray(elev_deg, dtype=float))
    a = np.atleast_1d(np.asarray(az_deg, dtype=float))
    if grid:
        E, A = np.meshgrid(e, a, indexing="ij")
        shape = (e.size, a.size)
        E, A = E.ravel(), A.ravel()
    else:
        if e.size != a.size:
            raise ValueError("elev_deg and az_deg must have the same length when grid=False")
        E, A, shape = e, a, None
    dirs = np.array([launch_direction(x, y, up, north, east) for x, y in zip(E, A)])
    res = tracer.trace(r0, dirs, max_path_m, nhops=nhops, ceiling_alt_m=ceiling_alt_m,
                       want_group=True, want_absorption=True)
    return fan_from_result(tracer, res, r0, E, A, nhops, shape)


def ray_vectors(fan: FanResult) -> np.ndarray:
    """
    Flat per-ray vector, shape (N, 2 + 8H): [elev, az] then, for each hop k, the values
    x, y, P, P', A_db, Doppler, valid, apex (see _NV). Corner interpolation works on these.
    """
    N, H = fan.valid.shape
    cols = [fan.elev_deg[:, None], fan.az_deg[:, None]]
    for k in range(H):
        cols += [fan.planar[:, k, :], fan.P[:, k, None], fan.Pg[:, k, None], fan.A_db[:, k, None],
                 fan.doppler_hz[:, k, None], fan.valid[:, k, None].astype(float), fan.apex_m[:, k, None]]
    return np.concatenate(cols, axis=1)


def _col(h: int, field_idx: int) -> int:
    """Column of value field_idx (0 x, 1 y, 2 P, 3 Pg, 4 A, 5 Doppler, 6 valid, 7 apex) for hop h."""
    return 2 + _NV * h + field_idx


def _det(dxi: np.ndarray, deta: np.ndarray) -> float:
    return float(dxi[0] * deta[1] - dxi[1] * deta[0])


def _bilinear_map(x: np.ndarray, xi: float, eta: float) -> np.ndarray:
    """F(xi, eta) of Algorithm 2 for corners x = [x00, x10, x11, x01] (each (2,))."""
    x00, x10, x11, x01 = x
    return ((1 - xi) * (1 - eta) * x00 + xi * (1 - eta) * x10
            + xi * eta * x11 + (1 - xi) * eta * x01)


def _bilinear_jac(x: np.ndarray, xi: float, eta: float):
    x00, x10, x11, x01 = x
    dxi = (1 - eta) * (x10 - x00) + eta * (x11 - x01)
    deta = (1 - xi) * (x01 - x00) + xi * (x11 - x10)
    return dxi, deta


def _caustic_crosses(x: np.ndarray) -> bool:
    """det dF/d(xi, eta) changes sign over the cell (tested at the corners and the centre)."""
    dets = []
    for xi, eta in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)):
        dets.append(_det(*_bilinear_jac(x, xi, eta)))
    dets = np.array(dets)
    return bool(dets.min() < 0.0 < dets.max()) or bool(np.any(dets == 0.0))


@dataclass
class Candidate:
    """One launch angle whose hop-h landing point is the target (Algorithm 2 output, §5.4 labels)."""

    elev_deg: float
    az_deg: float
    hop: int                  # hop index (0 = one hop)
    xi: float
    eta: float
    subdivided: bool          # found in a cell refined around a caustic
    P: float
    Pg: float
    A_db: float
    doppler_hz: float
    apex_m: list              # apex height of each hop up to this one [m]
    dD_dbeta: float           # sign gives low (< 0) or high (> 0) ray (§5.4)
    label: str = ""
    layers: str = ""
    flags: list = field(default_factory=list)


def _solve_cell(corners, target, hop, depth, angle_tracer, out, H, mode, unresolved):
    """Algorithm 2 on one cell. corners: (4, V) per-ray vectors ordered 00, 10, 11, 01."""
    xcol = _col(hop, 0), _col(hop, 1)
    x = corners[:, [xcol[0], xcol[1]]]
    if not np.all(corners[:, _col(hop, 6)] > 0.5):
        return
    lo, hi = x.min(axis=0), x.max(axis=0)
    if np.any(target < lo - 1e-6) or np.any(target > hi + 1e-6):
        return
    if _caustic_crosses(x):
        if depth >= SUB_DEPTH_MAX:
            unresolved.append(depth)
            return
        b0, b1 = corners[0, 0], corners[1, 0]
        a0, a1 = corners[0, 1], corners[3, 1]
        mb, ma = 0.5 * (b0 + b1), 0.5 * (a0 + a1)
        sub = angle_tracer(np.array([mb, b1, mb, b0, mb]), np.array([a0, ma, a1, ma, ma]))
        v5 = ray_vectors(sub)
        bottom, right, top, left, centre = v5
        c00, c10, c11, c01 = corners
        for sc in (np.array([c00, bottom, centre, left]),
                   np.array([bottom, c10, right, centre]),
                   np.array([centre, right, c11, top]),
                   np.array([left, centre, top, c01])):
            _solve_cell(sc, target, hop, depth + 1, angle_tracer, out, H, mode, unresolved)
        return
    xi, eta = 0.5, 0.5
    for _ in range(5):                                  # Newton from (1/2, 1/2), 2-5 iterations
        F = _bilinear_map(x, xi, eta) - target
        dxi, deta = _bilinear_jac(x, xi, eta)
        J = np.column_stack([dxi, deta])
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return
        xi, eta = xi + step[0], eta + step[1]
        if np.hypot(*step) < 1e-12:
            break
    tol = 1e-9
    if not (-tol <= xi <= 1.0 + tol and -tol <= eta <= 1.0 + tol):
        return
    w = np.array([(1 - xi) * (1 - eta), xi * (1 - eta), xi * eta, (1 - xi) * eta])
    vals = w @ corners
    dD = (1 - eta) * (np.linalg.norm(corners[1, [_col(hop, 0), _col(hop, 1)]])
                      - np.linalg.norm(corners[0, [_col(hop, 0), _col(hop, 1)]])) \
        + eta * (np.linalg.norm(corners[2, [_col(hop, 0), _col(hop, 1)]])
                 - np.linalg.norm(corners[3, [_col(hop, 0), _col(hop, 1)]]))
    apex = [float(vals[_col(k, 7)]) for k in range(hop + 1)]
    out.append(Candidate(elev_deg=float(vals[0]), az_deg=float(vals[1]), hop=hop,
                         xi=float(xi), eta=float(eta), subdivided=depth > 0,
                         P=float(vals[_col(hop, 2)]), Pg=float(vals[_col(hop, 3)]),
                         A_db=float(vals[_col(hop, 4)]), doppler_hz=float(vals[_col(hop, 5)]),
                         apex_m=apex, dD_dbeta=float(dD)))


def invert_fan(fan: FanResult, hop: int, target_xy, *, angle_tracer=None, valley_m: float = VALLEY_M,
               H: int | None = None) -> list:
    """
    Algorithm 2: every launch angle of a grid fan whose hop-`hop` landing point is target_xy
    (east, north metres about the site). angle_tracer(elev_deg, az_deg) -> FanResult traces the
    five subdivision rays of a caustic cell; without it caustic cells are reported unresolved.
    Returns labelled candidates (§5.4), duplicates removed.
    """
    if fan.grid_shape is None:
        raise ValueError("invert_fan needs a grid fan (trace_fan(..., grid=True))")
    Nb, Na = fan.grid_shape
    H = fan.valid.shape[1] if H is None else H
    vec = ray_vectors(fan).reshape(Nb, Na, -1)
    target = np.asarray(target_xy, dtype=float)
    out, unresolved = [], []
    if angle_tracer is None:
        def angle_tracer(_e, _a):
            raise RuntimeError("caustic cell met but no angle_tracer was given")
    for i in range(Nb - 1):
        for j in range(Na - 1):
            corners = np.stack([vec[i, j], vec[i + 1, j], vec[i + 1, j + 1], vec[i, j + 1]])
            try:
                _solve_cell(corners, target, hop, 0, angle_tracer, out, H, fan.mode, unresolved)
            except RuntimeError:
                unresolved.append(0)
    kept = classify(out, mode=fan.mode, valley_m=valley_m)
    if unresolved:
        for c in kept:
            c.flags.append(f"unresolved caustic cells: {len(unresolved)}")
    return kept


def classify(cands: list, *, mode: int, valley_m: float = VALLEY_M) -> list:
    """§5.4: label = hop count, one layer letter per hop, low/high ray, O/X; then deduplicate."""
    for c in cands:
        c.layers = "".join(layer_letter(a, valley_m) for a in c.apex_m)
        c.label = f"{c.hop + 1}{c.layers}-{'low' if c.dD_dbeta < 0.0 else 'high'}-{'O' if mode > 0 else 'X'}"
    kept = []
    for c in sorted(cands, key=lambda c: (c.label, c.elev_deg, c.az_deg)):
        dup = next((k for k in kept if k.label == c.label
                    and abs(k.elev_deg - c.elev_deg) < DEDUP_DEG and abs(k.az_deg - c.az_deg) < DEDUP_DEG), None)
        if dup is None:
            kept.append(c)
    return kept
