"""
Phase 1g tests: Doppler Method B (spec §4.6, eq. 4.8) against Method A.

Method A advances the ionosphere by +/- DT, re-homes the leg to the same ground point and
differences the phase paths: f_D = -(dP/dt)/lambda0. Method B accumulates g dNe/dt ds along the
unperturbed ray in one trace. Two prescribed ionospheres: uniform decay (model evolution only)
and a rigid northward translation (convective term -v_d . grad(Ne) only). The spec's validation
is agreement to 5% for oblique legs.

Homing here is a 1-D secant on launch elevation at fixed azimuth, enough for a planar leg; it
stands in for homing.py (Algorithm 3) until Phase 1i.
"""

from __future__ import annotations

import numpy as np

from propagation import legs
from raytracing.batch import BatchTracer, DopplerRates
from core.constants import C
from raytracing.main import Ray
from test_records_ground import _launch, _layer

F0 = 8.0e6
W0 = 2.0 * np.pi * F0
LAMBDA0 = C / F0
DT = 60.0                                  # epoch spacing [s], spec §4.6 default
TAU = 900.0                                # uniform-decay time constant [s]
ELEV = 28.0                                # launch elevation [deg]
LAT, LON = 20.0, 0.0                       # meridian plane y = 0; B along east keeps the leg planar
B_EAST = np.array([0.0, 2e-5, 0.0])
NE0 = _layer()
V_DRIFT = 300.0 * np.array([-np.sin(np.radians(LAT)), 0.0, np.cos(np.radians(LAT))])   # 300 m/s north


def _ne_decay(t):
    return lambda x, y, z: NE0(x, y, z) * np.exp(-t / TAU)


def _ne_shift(t):
    return lambda x, y, z: NE0(x - V_DRIFT[0] * t, y - V_DRIFT[1] * t, z - V_DRIFT[2] * t)


def _ray(Ne_of_t, t):
    def B_field(x, y, z):
        return np.broadcast_to(B_EAST, np.shape(x) + (3,)).copy()
    return Ray(W0, +1, Ne_of_t(t), B_field=B_field)


def _model_rate(Ne_of_t, t0):
    """Model evolution: two-epoch central difference on one grid, [Ne(t0+DT) - Ne(t0-DT)]/(2DT)."""
    def rate(r):
        x, y, z = r[..., 0], r[..., 1], r[..., 2]
        return (Ne_of_t(t0 + DT)(x, y, z) - Ne_of_t(t0 - DT)(x, y, z)) / (2.0 * DT)
    return rate


def _drift(r):
    return np.broadcast_to(V_DRIFT, r.shape).copy()


def _setup():
    r0, _, up, east = _launch(lat=LAT, lon=LON, elev=ELEV, az=0.0)
    return r0, up, np.cross(up, east), east


def _home(tracer, r0, up, north, east, target_m):
    """Secant on launch elevation (azimuth 0) until the landing ground range equals target_m."""
    def land(e_deg):
        d = legs.launch_direction(e_deg, 0.0, up, north, east)[None]
        res = tracer.trace(r0, d, 3000e3, nhops=1, want_group=False, want_absorption=False)
        assert res.landed[0], e_deg
        return res.ground_range_m[0] - target_m, res

    e0, e1 = ELEV, ELEV + 0.1
    f0, _ = land(e0)
    f1, res = land(e1)
    for _ in range(20):
        if abs(f1) < 1e-4:
            break
        e0, f0, e1 = e1, f1, e1 - f1 * (e1 - e0) / (f1 - f0)
        f1, res = land(e1)
    assert abs(f1) < 1e-3, f1
    return res


def _method_a(Ne_of_t, r0, up, north, east, target_m):
    """Method A (spec §4.6): phase paths of the re-homed leg at t = -DT and +DT, then -dP/dt / lambda0."""
    P = {}
    for sign in (+1.0, -1.0):
        tracer = BatchTracer(_ray(Ne_of_t, sign * DT), ds_m=4000.0)
        P[sign] = float(_home(tracer, r0, up, north, east, target_m).phase_path[0])
    return -(P[1.0] - P[-1.0]) / (2.0 * DT) / LAMBDA0


def _method_b(Ne_of_t, r0, up, north, east, rate=None, drift=None):
    """Method B (spec §4.8): one trace of the unperturbed leg at t = 0 with the accumulator."""
    u0 = legs.launch_direction(ELEV, 0.0, up, north, east)[None]
    tracer = BatchTracer(_ray(Ne_of_t, 0.0), ds_m=4000.0, doppler=DopplerRates(rate=rate, drift=drift))
    res = tracer.trace(r0, u0, 3000e3, nhops=1, want_group=False, want_absorption=False)
    assert res.landed[0]
    return float(res.doppler_hz[0]), float(res.ground_range_m[0])


def test_uniform_decay_method_b_matches_method_a():
    """Prescribed uniform decay dNe/dt = -Ne/TAU on an oblique leg: Method B within 5% of Method A."""
    r0, up, north, east = _setup()
    fB, target = _method_b(_ne_decay, r0, up, north, east, rate=_model_rate(_ne_decay, 0.0))
    fA = _method_a(_ne_decay, r0, up, north, east, target)
    assert abs(fA) > 0.1, fA
    assert abs(fB - fA) < 0.05 * abs(fA), (fA, fB)


def test_convective_advection_method_b_matches_method_a():
    """Rigid northward translation at 300 m/s: the convective term -v_d . grad(Ne) alone, within 5%."""
    r0, up, north, east = _setup()
    fB, target = _method_b(_ne_shift, r0, up, north, east, drift=_drift)
    fA = _method_a(_ne_shift, r0, up, north, east, target)
    assert abs(fA) > 0.1, fA
    assert abs(fB - fA) < 0.05 * abs(fA), (fA, fB)


def test_accumulator_leaves_trajectory_unchanged():
    r0, up, north, east = _setup()
    u0 = legs.launch_direction(ELEV, 0.0, up, north, east)[None]
    ray = _ray(_ne_decay, 0.0)
    plain = BatchTracer(ray, ds_m=4000.0).trace(r0, u0, 3000e3, nhops=2)
    acc = BatchTracer(ray, ds_m=4000.0, doppler=DopplerRates(rate=_model_rate(_ne_decay, 0.0))).trace(
        r0, u0, 3000e3, nhops=2)
    assert np.array_equal(plain.r_final, acc.r_final)
    assert np.array_equal(plain.phase_path, acc.phase_path)
    assert plain.doppler_hz is None and acc.doppler_hz is not None


def test_hop_dopplers_sum_to_leg_total():
    """Spec §4.6: the Doppler is stored per hop. Hop values come from differences of the cumulative
    integral at each ground contact, so for a landed leg they must sum to the leg's total."""
    r0, up, north, east = _setup()
    u0 = legs.launch_direction(ELEV, 0.0, up, north, east)[None]
    tracer = BatchTracer(_ray(_ne_decay, 0.0), ds_m=4000.0, doppler=DopplerRates(rate=_model_rate(_ne_decay, 0.0)))
    res = tracer.trace(r0, u0, 3000e3, nhops=2)
    assert res.landed[0] and res.hops[0] >= 1
    hops = np.asarray(res.doppler_hops_hz[0])
    assert hops.size == len(res.bounce_points[0]) == res.hops[0]
    assert abs(hops.sum() - res.doppler_hz[0]) < 1e-9 * max(1.0, abs(res.doppler_hz[0]))
