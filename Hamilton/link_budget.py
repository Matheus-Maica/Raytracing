"""
link_budget.py — point-source ray-tube focusing factor, combined with
absorption and (deterministic) phase-variance coherence loss into a single
relative dB "receiver statistics" report.

Scope: this module reports RELATIVE gains/losses only (focusing_gain_db,
absorption_db, coherence_loss_db, total_db) -- there is no transmitter
EIRP / antenna gain / receiver sensitivity model anywhere in this codebase,
so combining these into an absolute received power is the caller's job.

Focusing factor
----------------
main.py::Ray.ray_tube_jacobian used to compute det(Phi[:3,:3]), the
position-to-position tangent-map block -- the wrong quantity for a fixed
launch POINT with a bundle of nearby launch ANGLES (our actual geometry).
The physically relevant Jacobian is the off-diagonal block
Phi_ru = tangent_map_final[0:3, 3:6] (d(r_final)/d(u_initial)).

Given launch direction u0 (assumed |u0| ~= 1, i.e. launched from a
vacuum/neutral-atmosphere point), build (t1, t2) spanning the plane
perpendicular to u0_hat (reuses raytracer._any_perp). v1 = Phi_ru @ t1,
v2 = Phi_ru @ t2 are projected onto the plane perpendicular to the
GROUP-velocity direction at the landing point (the local rdot from
Ray._haselgrove_physical_rhs, not the wave-normal u_final -- these differ
by the ray/wave-normal splitting angle in a magnetized medium).
Area_ratio = |cross(v1_perp, v2_perp)|.

Note on the tangent map's fixed-parameter vs. fixed-crossing subtlety: the
adaptive driver stops on a ground-crossing EVENT (at ray-parameter tau*),
not at fixed tau, so naively Phi (a fixed-tau derivative) isn't quite "how
the landing point moves with launch angle" -- the standard event/Poincare-
crossing correction for this is subtracting a term of the form
outer(rdot, something)/denom from Phi_ru. That correction's OUTPUT is,
structurally, always parallel to rdot (proof: outer(rdot, g)@x = rdot *
dot(g, x) for any x) -- and the transverse-plane projection onto perp-rdot
below removes exactly that component regardless of its value. So the
correction and this projection cancel exactly (verified both analytically
and against random matrices), making the correction a no-op here and safe
to skip -- it would matter if projecting onto the wave-normal (u_final)
instead, but not against the group direction.

Free-space reference at the same accumulated arc length s_total is EXACTLY
s_total**2: in vacuum (X=Ne*X_factor=0 identically along the whole path),
Ray._haselgrove_physical_rhs's K and Msc terms vanish identically regardless
of B/Y/p, so u(tau) never changes and r(s) = r0 + s*u0_hat exactly -- this
pins the reference to s_total specifically (not chord distance, not group
path), since Phi_ru@t_i = s_total*t_i falls straight out of that exact
straight-line solution.

    focusing_gain_db = 10*log10(s_total**2 / Area_ratio)   (power convention)

positive = more concentrated than free-space spreading (e.g. ionospheric
convergence), negative = defocused.

Known limitations (not fixed here):
  * Only meaningful for reason=="GROUND" (landed) results -- None otherwise.
  * Only rigorous for nhops=1. For nhops>1 the tangent map is carried across
    intermediate ground bounces without differentiating the specular-
    reflection map itself (which has its own r-dependence via n_hat=r/|r|,
    plus its direct action on u) -- a real additional correction this module
    does not attempt.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from raytracer import RayTracer, TraceResult, _any_perp

_PHASE_VARIANCE_TO_DB = 10.0 / np.log(10.0)   # ~4.342944819, from -10*log10(exp(-x)) = (10/ln10)*x


@dataclass
class ReceiverStatistics:
    landed: bool
    escaped: bool
    ground_range_total_m: float

    focusing_gain_db: float | None     # None if not landed (no receiver / no tangent map)
    absorption_db: float               # positive-loss-magnitude convention
    coherence_loss_db: float           # positive-loss-magnitude convention
    total_db: float | None             # None if focusing_gain_db is None

    sigma_phi2: float
    coherent_fraction: float           # exp(-sigma_phi2/2), matches mc_scattering.summarize's convention

    result: TraceResult                # full underlying trace, for callers who want more


def _focusing_gain_db(tracer: RayTracer, u0: np.ndarray, result: TraceResult) -> float | None:
    if not result.landed or result.tangent_map_final is None:
        return None
    u0_hat = np.asarray(u0, dtype=float)
    u0_hat = u0_hat / np.linalg.norm(u0_hat)
    t1 = _any_perp(u0_hat)
    t2 = np.cross(u0_hat, t1)

    final_state6 = np.concatenate([result.r_final, result.u_final])
    rdot = tracer.ray._haselgrove_physical_rhs(0.0, final_state6)[:3]
    rdot_hat = rdot / np.linalg.norm(rdot)

    Phi_ru = result.tangent_map_final[0:3, 3:6]
    v1, v2 = Phi_ru @ t1, Phi_ru @ t2
    v1_perp = v1 - np.dot(v1, rdot_hat) * rdot_hat
    v2_perp = v2 - np.dot(v2, rdot_hat) * rdot_hat

    area_ratio = float(np.linalg.norm(np.cross(v1_perp, v2_perp)))
    R = result.s_total
    if area_ratio <= 0.0 or R <= 0.0:
        return None
    return 10.0 * np.log10((R * R) / area_ratio)


def compute_receiver_statistics(tracer: RayTracer, r0, u0, max_path_m: float, *,
                                 nhops: int = 1,
                                 ceiling_alt_m: float = 800e3) -> ReceiverStatistics:
    """
    Runs ONE adaptive trace (tangent map + absorption + group/phase path +
    deterministic phase-variance, all in a single call) and assembles a
    relative dB link-budget report. `tracer.spectrum` must be set (needed
    for compute_phase_variance=True) -- raises the same clear error as
    RayTracer.trace() itself if missing.
    """
    result = tracer.trace(
        r0, u0, max_path_m, nhops=nhops, method="adaptive", scatter=False,
        carry_tangent_map=True, compute_phase_variance=True,
        compute_group_path=True, compute_absorption=True,
        ceiling_alt_m=ceiling_alt_m,
    )

    focusing_gain_db = _focusing_gain_db(tracer, u0, result)
    absorption_db = result.absorption_db_total
    coherence_loss_db = _PHASE_VARIANCE_TO_DB * result.sigma_phi2
    total_db = (focusing_gain_db - absorption_db - coherence_loss_db
                if focusing_gain_db is not None else None)

    return ReceiverStatistics(
        landed=result.landed, escaped=result.escaped,
        ground_range_total_m=result.ground_range_total_m,
        focusing_gain_db=focusing_gain_db,
        absorption_db=absorption_db,
        coherence_loss_db=coherence_loss_db,
        total_db=total_db,
        sigma_phi2=result.sigma_phi2,
        coherent_fraction=float(np.exp(-result.sigma_phi2 / 2.0)),
        result=result,
    )
