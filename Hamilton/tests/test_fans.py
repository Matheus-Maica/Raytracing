"""
Phase 1i tests: launch fans, the landing map, the quad inversion (Algorithm 2) and the caustic
subdivision and classification (spec §5.1, §5.2, §5.4). Dipped field, as in test_homing.py.
"""

from __future__ import annotations

import re

import numpy as np

from propagation import fans
from propagation import legs
from raytracing.batch import BatchTracer
from raytracing.main import Ray
from test_homing import _setup, _tracer
from test_records_ground import _layer

F0 = 8.0e6
W0 = 2.0 * np.pi * F0


def _fan(tracer, r0, up, north, east, elev, az):
    return fans.trace_fan(tracer, r0, up, north, east, elev, az, 3000e3, nhops=1, grid=True)


def test_quad_inversion_recovers_a_known_launch_angle():
    r0, up, north, east, B_field = _setup()
    tr = _tracer(+1, B_field)
    fan = _fan(tr, r0, up, north, east, np.arange(22.0, 32.01, 0.25), np.arange(-0.5, 0.501, 0.25))
    known = fans.trace_fan(tr, r0, up, north, east, [27.3], [0.2], 3000e3, nhops=1, grid=False)
    assert known.valid[0, 0]
    target = known.planar[0, 0]
    cands = fans.invert_fan(fan, 0, target, angle_tracer=None)
    assert cands, "no candidate found for a landing point inside the fan"
    best = min(cands, key=lambda c: (c.elev_deg - 27.3) ** 2 + (c.az_deg - 0.2) ** 2)
    assert abs(best.elev_deg - 27.3) < 0.05 and abs(best.az_deg - 0.2) < 0.05, (best.elev_deg, best.az_deg)
    assert re.fullmatch(r"1[EF]-(low|high)-O", best.label), best.label


# A bilinear landing map with a true fold in one cell: the corner Jacobians are +1, +0.2, -0.6, so
# det dF/d(xi, eta) changes sign inside the cell (spec Algorithm 2, step 3).
FOLD_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.2], [0.0, 1.0]])   # 00, 10, 11, 01


def _fold_map(e, a):
    """Planar image of (elev, az) in [0, 1]^2 under the folded bilinear map."""
    x00, x10, x11, x01 = FOLD_CORNERS
    e = np.asarray(e, float)[..., None]
    a = np.asarray(a, float)[..., None]
    return (1 - e) * (1 - a) * x00 + e * (1 - a) * x10 + e * a * x11 + (1 - e) * a * x01


def _fold_fan(elev, az):
    e, a = np.meshgrid(np.asarray(elev, float), np.asarray(az, float), indexing="ij")
    e, a = e.ravel(), a.ravel()
    N = e.size
    z = np.zeros((N, 1))
    return fans.FanResult(elev_deg=e, az_deg=a, grid_shape=(len(elev), len(az)), hops=np.ones(N, dtype=int),
                          valid=np.ones((N, 1), dtype=bool), land=np.zeros((N, 1, 3)),
                          planar=_fold_map(e, a)[:, None, :], P=z.copy(), Pg=z.copy(), A_db=z.copy(),
                          doppler_hz=z.copy(), apex_m=np.full((N, 1), 300e3), u_inc=np.zeros((N, 1, 3)),
                          s_inc=np.zeros((N, 1, 3)), mode=1)


def _fold_ray(e, a):
    e, a = np.atleast_1d(e).ravel(), np.atleast_1d(a).ravel()
    N = e.size
    z = np.zeros((N, 1))
    return fans.FanResult(elev_deg=e, az_deg=a, grid_shape=None, hops=np.ones(N, dtype=int),
                          valid=np.ones((N, 1), dtype=bool), land=np.zeros((N, 1, 3)),
                          planar=_fold_map(e, a)[:, None, :], P=z.copy(), Pg=z.copy(), A_db=z.copy(),
                          doppler_hz=z.copy(), apex_m=np.full((N, 1), 300e3), u_inc=np.zeros((N, 1, 3)),
                          s_inc=np.zeros((N, 1, 3)), mode=1)


def test_caustic_cell_triggers_subdivision_on_a_fold():
    """Algorithm 2 on a folded cell: the cell is refined with five extra rays (depth <= 4), and the
    preimage of the target, the cell centre (0.5, 0.5), is found in the refined cells and flagged."""
    fan = _fold_fan([0.0, 1.0], [0.0, 1.0])
    target = _fold_map(0.5, 0.5)
    calls = []

    def angle_tracer(e, a):
        calls.append(len(e))
        return _fold_ray(e, a)

    cands = fans.invert_fan(fan, 0, target, angle_tracer=angle_tracer)
    assert calls and all(n == 5 for n in calls), calls
    hit = [c for c in cands if abs(c.elev_deg - 0.5) < 1e-6 and abs(c.az_deg - 0.5) < 1e-6]
    assert hit and all(c.subdivided for c in hit), [(c.elev_deg, c.az_deg, c.subdivided) for c in cands]


def test_no_subdivision_on_a_fold_free_physical_fan():
    r0, up, north, east, B_field = _setup()
    tr = _tracer(+1, B_field)
    fan = _fan(tr, r0, up, north, east, np.arange(3.0, 36.01, 0.25), np.linspace(-0.5, 0.5, 5))
    known = fans.trace_fan(tr, r0, up, north, east, [20.0], [0.1], 3000e3, nhops=1, grid=False)
    calls = []

    def angle_tracer(e, a):
        calls.append(len(e))
        return fans.trace_fan(tr, r0, up, north, east, e, a, 3000e3, nhops=1, grid=False)

    cands = fans.invert_fan(fan, 0, known.planar[0, 0], angle_tracer=angle_tracer)
    assert cands and not calls


def test_low_and_high_ray_labels_follow_the_sign_of_dD_dbeta():
    """§5.4: low rays have dD/dbeta < 0, high rays dD/dbeta > 0."""
    r0, up, north, east, B_field = _setup()
    tr = _tracer(+1, B_field)
    fan = _fan(tr, r0, up, north, east, np.arange(14.0, 36.01, 0.25), np.linspace(-0.5, 0.5, 5))
    known = fans.trace_fan(tr, r0, up, north, east, [27.0], [0.0], 3000e3, nhops=1, grid=False)
    cands = fans.invert_fan(fan, 0, known.planar[0, 0], angle_tracer=None)
    for c in cands:
        assert (c.dD_dbeta < 0.0) == c.label.endswith("-low-O"), (c.label, c.dD_dbeta)
