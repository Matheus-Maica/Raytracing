"""
homing.py -- Newton homing of one mode (spec §5.3, Algorithm 3) and the reciprocity test (§5.5).

Homing solves for the launch direction (beta, alpha) whose ray lands on a target point. The
residual is the miss in the tangent plane at the target (east, north). The Jacobian is by forward
differences of the landing point (delta = 0.005 deg), traced in one batch with the centre ray.
A converged ray is corrected to first order: P and P' gain s_hat . (r_G - r_L), since the ground
is vacuum (n = 1, group path equals phase path to first order there).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from propagation.fans import layer_letter, VALLEY_M
from core.geometry import ellipsoid_outward_normal_m
from propagation.legs import launch_direction

DEG = np.pi / 180.0


def tangent_basis(r_t: np.ndarray):
    """East and north unit vectors of the tangent plane at the ground point r_t."""
    up = ellipsoid_outward_normal_m(np.asarray(r_t, dtype=float))
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return east, north


@dataclass
class HomedRay:
    """Result of Algorithm 3 for one mode (spec §5.3)."""

    elev_deg: float
    az_deg: float
    hop: int
    converged: bool
    iters: int
    residual_m: float
    r_land: np.ndarray
    s_hat: np.ndarray          # physical incident direction at the landing point
    P: float                   # phase path, corrected for the residual miss [m]
    Pg: float                  # group path, corrected [m]
    A_db: float
    layers: str
    flags: list = field(default_factory=list)


def _layers(res, i: int, valley_m: float = VALLEY_M) -> str:
    return "".join(layer_letter(a, valley_m) for a in res.hop_apex[i])


def _probe(tracer, r0, up, north, east, thetas, hop, max_path_m, ceiling_alt_m):
    dirs = np.array([launch_direction(e, a, up, north, east) for e, a in thetas])
    res = tracer.trace(r0, dirs, max_path_m, nhops=hop, ceiling_alt_m=ceiling_alt_m,
                       want_group=True, want_absorption=True)
    ok = res.landed & (res.hops == hop)
    return ok, res


def newton_home(tracer, r0, up, north, east, elev0_deg: float, az0_deg: float, r_target, *,
                hop: int = 1, eps_m: float = 50.0, max_iter: int = 10, delta_deg: float = 0.005,
                max_step_deg=(0.125, 0.25), max_path_m: float = 3000e3,
                ceiling_alt_m: float = 800e3) -> HomedRay:
    """
    Algorithm 3. hop is the hop count (1 = one bounce-free leg). Rejects the mode (converged
    False, with a flag) if the hop count or the layer sequence changes, or det J changes sign.
    """
    r_t = np.asarray(r_target, dtype=float)
    Ge, Gn = tangent_basis(r_t)
    G = np.vstack([Ge, Gn])                                    # (2, 3) tangent-plane projection
    theta = np.array([elev0_deg, az0_deg], dtype=float)

    ok, res = _probe(tracer, r0, up, north, east, [theta], hop, max_path_m, ceiling_alt_m)
    if not ok[0]:
        raise RuntimeError("start ray does not land at the requested hop")
    layers0 = _layers(res, 0)
    e = G @ (res.r_final[0] - r_t)
    det0 = None
    converged, flags, it = False, [], 0
    for it in range(1, max_iter + 1):
        if np.hypot(*e) < eps_m:
            converged = True
            break
        thetas = [theta, theta + np.array([delta_deg, 0.0]), theta + np.array([0.0, delta_deg])]
        ok3, res3 = _probe(tracer, r0, up, north, east, thetas, hop, max_path_m, ceiling_alt_m)
        if not np.all(ok3):
            flags.append("Jacobian probe missed the hop")
            break
        dL = np.column_stack([(res3.r_final[1] - res3.r_final[0]), (res3.r_final[2] - res3.r_final[0])])
        dL = dL / (delta_deg * DEG)
        J2 = G @ dL
        det = float(np.linalg.det(J2))
        if det0 is None:
            det0 = det
        if np.sign(det) != np.sign(det0):
            flags.append("det J2 changed sign (caustic crossed)")
            break
        step = -np.linalg.solve(J2, e) / DEG                  # degrees
        step = np.clip(step, -np.asarray(max_step_deg), np.asarray(max_step_deg))
        lam, accepted = 1.0, False
        for _ in range(12):
            th_new = theta + lam * step
            okn, resn = _probe(tracer, r0, up, north, east, [th_new], hop, max_path_m, ceiling_alt_m)
            if okn[0] and _layers(resn, 0) == layers0:
                e_new = G @ (resn.r_final[0] - r_t)
                if np.hypot(*e_new) < np.hypot(*e):
                    accepted = True
                    break
            lam *= 0.5
        if not accepted:
            flags.append("backtracking failed")
            break
        theta, e, res = th_new, e_new, resn
    else:
        converged = bool(np.hypot(*e) < eps_m)

    r_l = res.r_final[0]
    s_hat = tracer._direction(np.concatenate([r_l, res.u_final[0]]))
    miss_along = float(np.dot(s_hat, r_t - r_l))
    return HomedRay(elev_deg=float(theta[0]), az_deg=float(theta[1]), hop=hop, converged=converged,
                    iters=it, residual_m=float(np.hypot(*e)), r_land=r_l, s_hat=s_hat,
                    P=float(res.phase_path[0]) + miss_along, Pg=float(res.group_path[0]) + miss_along,
                    A_db=8.686 * float(res.absorption_amplitude[0]), layers=_layers(res, 0), flags=flags)


def reciprocity(tracer, r0, up, north, east, elev_deg: float, az_deg: float, *, hop: int = 1,
                max_path_m: float = 3000e3, ceiling_alt_m: float = 800e3) -> dict:
    """
    Spec §5.5 reciprocity test. Trace the leg from the site; launch from the landing point with the
    reversed incident wave normal (same mode); the reversed ray must return to the site, with
    P, P' and absorption equal to the forward values (relative differences reported).
    """
    d = launch_direction(elev_deg, az_deg, up, north, east)[None]
    fwd = tracer.trace(r0, d, max_path_m, nhops=hop, ceiling_alt_m=ceiling_alt_m,
                       want_group=True, want_absorption=True)
    if not fwd.landed[0]:
        raise RuntimeError("forward leg does not land")
    rev = tracer.trace(fwd.r_final[0], -fwd.u_final[0][None], max_path_m, nhops=hop,
                       ceiling_alt_m=ceiling_alt_m, want_group=True, want_absorption=True)

    def rel(a, b):
        return float(abs(a - b) / max(abs(b), 1e-30))

    return dict(landed=bool(rev.landed[0]),
                miss_m=float(np.linalg.norm(rev.r_final[0] - r0)),
                rel_P=rel(rev.phase_path[0], fwd.phase_path[0]),
                rel_Pg=rel(rev.group_path[0], fwd.group_path[0]),
                rel_A=rel(rev.absorption_amplitude[0], fwd.absorption_amplitude[0]),
                P=float(fwd.phase_path[0]), Pg=float(fwd.group_path[0]),
                A_db=8.686 * float(fwd.absorption_amplitude[0]))
