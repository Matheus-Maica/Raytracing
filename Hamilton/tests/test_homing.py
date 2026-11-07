"""
Phase 1i tests: Newton homing of one mode (spec §5.3, Algorithm 3) and the reciprocity test
(§5.5). The field is dipped (70 deg), so neither mode is in the degenerate B-perpendicular case.
"""

from __future__ import annotations

import numpy as np

from propagation import homing
from propagation import legs
from raytracing.batch import BatchTracer
from raytracing.main import Ray
from test_records_ground import _layer, _launch

F0 = 8.0e6
W0 = 2.0 * np.pi * F0
LAT, LON = 60.0, 0.0
DIP = np.radians(70.0)


def _setup():
    r0, _, up, east = _launch(lat=LAT, lon=LON, elev=28.0, az=0.0)
    north = np.cross(up, east)
    B = 2e-5 * (np.cos(DIP) * north - np.sin(DIP) * up)

    def B_field(x, y, z):
        return np.broadcast_to(B, np.shape(x) + (3,)).copy()

    return r0, up, north, east, B_field


def _tracer(MODE, B_field, ds=1000.0):
    return BatchTracer(Ray(W0, MODE, _layer(), B_field=B_field), ds_m=ds)


def test_newton_homing_converges_below_50m_in_10_iterations():
    r0, up, north, east, B_field = _setup()
    tr = _tracer(+1, B_field)
    target = tr.trace(r0, legs.launch_direction(28.0, 0.0, up, north, east)[None], 3000e3,
                      nhops=1).r_final[0]
    homed = homing.newton_home(tr, r0, up, north, east, 27.6, 0.25, target, hop=1)
    assert homed.converged, homed.flags
    assert homed.iters <= 10, homed.iters
    assert homed.residual_m < 50.0, homed.residual_m
    assert abs(homed.elev_deg - 28.0) < 0.05 and abs(homed.az_deg) < 0.05, (homed.elev_deg, homed.az_deg)


def test_homed_phase_path_corrected_to_target_miss():
    """P is corrected to first order by s_hat . (r_G - r_L): the homed P must not depend on the
    residual within the 50 m tolerance (checked against a very tight tolerance)."""
    r0, up, north, east, B_field = _setup()
    tr = _tracer(+1, B_field)
    target = tr.trace(r0, legs.launch_direction(28.0, 0.0, up, north, east)[None], 3000e3,
                      nhops=1).r_final[0]
    loose = homing.newton_home(tr, r0, up, north, east, 27.6, 0.25, target, hop=1, eps_m=50.0)
    tight = homing.newton_home(tr, r0, up, north, east, 27.6, 0.25, target, hop=1, eps_m=0.01)
    assert tight.residual_m < 0.01
    assert abs(loose.P - tight.P) < 50.0 * 1.2       # within the loose tolerance, after correction


def test_reciprocity_O_and_X_one_and_two_hops():
    """Spec §5.5: launching from the landing point with the reversed wave normal returns to the
    site with P, P' and absorption equal to 1e-6 (relative), for O and X, one and two hops."""
    r0, up, north, east, B_field = _setup()
    for MODE in (+1, -1):
        for hop in (1, 2):
            rr = homing.reciprocity(_tracer(MODE, B_field), r0, up, north, east, 28.0, 0.0, hop=hop)
            assert rr["landed"] and rr["miss_m"] < 1.0, (MODE, hop, rr)
            assert max(rr["rel_P"], rr["rel_Pg"], rr["rel_A"]) < 1e-6, (MODE, hop, rr)
