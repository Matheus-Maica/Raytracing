"""
Phase 6c tests: the screens of a traced leg (spec §9.1-9.3) and the reciprocity diagnostic on real legs. Identical
legs give Gamma_or = 1; a displaced return leg gives a smaller Gamma_or. Placeholder irregularity parameters.
"""

from __future__ import annotations

import numpy as np

from raytracing import batch
from stochastic import leg_screens
from propagation import legs
from echoes import twoway
from raytracing.batch import BatchTracer
from raytracing.main import Ray
from test_records_ground import _layer, _launch

W0 = 2.0 * np.pi * 8e6
B_EAST = np.array([0.0, 2e-5, 0.0])


def _record(tracer, elev):
    r0, _, up, east = _launch(lat=20.0, lon=0.0, elev=elev, az=0.0)
    north = np.cross(up, east)
    d = legs.launch_direction(elev, 0.0, up, north, east)[None]
    res = tracer.trace(r0, d, 2500e3, nhops=1, record_states=True, want_group=True, want_absorption=True)
    return batch.samples_from_record(tracer, res.records[0], transport=True), r0


def test_screens_of_a_leg_and_gamma_or_on_real_legs():
    def B_field(x, y, z):
        return np.broadcast_to(B_EAST, np.shape(x) + (3,)).copy()
    tracer = BatchTracer(Ray(W0, +1, _layer(), B_field=B_field), ds_m=4000.0)
    out, _ = _record(tracer, 28.0)
    scr = leg_screens.screens_for_leg(out, tracer.ray)
    assert scr.N > 10 and np.all(np.isfinite(scr.sigma2))
    assert np.all(np.diff(scr.z) > 0.0)
    g_same = twoway.gamma_or_legs(out, out, scr)
    assert abs(g_same - 1.0) < 1e-12
    other, _ = _record(tracer, 25.0)
    g_other = twoway.gamma_or_legs(out, other, scr)
    assert 0.0 <= g_other < 1.0 and np.isfinite(g_other)


def test_coarsen_elevation_deg_gives_fewer_screens_matching_HiCIRFs_own_rule():
    """Plan Stage 3 / Decision 3: coarsen_elevation_deg=5.0 (HiCIRF paragraph 17's own rule) must produce
    strictly fewer, but still valid, screens than the default one-per-sample placement, on the same leg."""
    tracer = BatchTracer(Ray(W0, +1, _layer()), ds_m=4000.0)
    out, _ = _record(tracer, 15.0)   # a shallow, long-path ray: plenty of elevation change to coarsen
    fine = leg_screens.screens_for_leg(out, tracer.ray)
    coarse = leg_screens.screens_for_leg(out, tracer.ray, coarsen_elevation_deg=5.0)
    assert coarse.N < fine.N
    assert coarse.N >= 2 and np.all(np.isfinite(coarse.sigma2)) and np.all(np.diff(coarse.z) > 0.0)
    # a very loose threshold keeps essentially every sample: should fall back to (near) the fine placement
    loose = leg_screens.screens_for_leg(out, tracer.ray, coarsen_elevation_deg=1e-6)
    assert loose.N == fine.N
