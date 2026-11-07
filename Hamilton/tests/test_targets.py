"""
Phase 5c tests: point targets (spec §5.2, §5.3, §5.5, §12.3). Ground target at 30 N on the meridian of
the Tx at 20 N and the Rx at 18 N (synthetic layer, B east keeps every leg in the meridian plane).
"""

from __future__ import annotations

import re

import numpy as np

from echoes import targets
from raytracing.batch import BatchTracer
from core.constants import C
from raytracing.main import Ray
from test_records_ground import _layer

W0 = 2.0 * np.pi * 8e6
LAM0 = C / 8e6
B_EAST = np.array([0.0, 2e-5, 0.0])
TX = (20.0, 0.0, 0.0)
RX = (18.0, 0.0, 0.0)
TARGET = (30.0, 0.0)
ELEV = np.arange(10.0, 26.01, 0.5)     # the Rx needs ~14 deg or less
AZ = np.linspace(-0.5, 0.5, 5)


def _tracer():
    def B_field(x, y, z):
        return np.broadcast_to(B_EAST, np.shape(x) + (3,)).copy()
    return BatchTracer(Ray(W0, +1, _layer(), B_field=B_field), ds_m=4000.0)


def _legs(tracer, site):
    return targets.home_site_to_point(tracer, "tx" if site is TX else "rx", *site, *TARGET,
                                      elev_grid=ELEV, az_grid=AZ, max_path_m=2500e3)


def test_both_sites_home_converged_one_hop_legs_to_the_target():
    tracer = _tracer()
    tx = _legs(tracer, TX)
    rx = _legs(tracer, RX)
    for group in (tx, rx):
        ok = [h for h in group if h.converged]
        assert ok, [(h.converged, h.residual_m, h.flags) for h in group]
        for h in ok:
            assert h.residual_m < 50.0
            assert re.fullmatch(r"1[EF]-(low|high)-O", h.leg.label), h.leg.label   # apex ~177 km: E by the 180 km valley
            assert h.leg.n_hops == 1 and h.leg.k_far is not None and np.linalg.norm(h.leg.k_far) > 0.99
    return tx, rx


def test_monostatic_component_obeys_eq_6_1_and_the_delay_is_twice_p_prime():
    tracer = _tracer()
    tx = _legs(tracer, TX)
    ok = [h for h in tx if h.converged]
    assert ok
    h = ok[0]
    k_i = h.leg.k_far
    v = 300.0 * k_i                                  # target moving toward the radar along the ray
    comps = targets.target_components([h], [h], f0=8e6, v_T=v, sigma_T_m2=10.0, Pt_dBW=40.0)
    assert len(comps) == 1
    comp = comps[0]
    assert abs(comp.scatterer["f_T"] + 2.0 * 300.0 / LAM0) < 1e-6
    assert abs(comp.delay - 2.0 * h.leg.Pg / C) < 1e-12
    assert comp.gamma_or is not None and abs(comp.gamma_or - 1.0) < 1e-9   # the same leg both ways: reciprocal
    tau, _ = targets.component_track(comp, np.array([0.0, 20.0]))
    assert abs((tau[1] - tau[0]) * C / 2.0 - 6000.0) < 1e-3        # 300 m/s over 20 s: 6 km one way (§12.3)
    assert abs((tau[1] - tau[0]) / 20.0 - comp.scatterer["range_migration_s_per_s"]) < 1e-18


def test_bistatic_components_add_the_two_group_delays():
    tracer = _tracer()
    tx = [h for h in _legs(tracer, TX) if h.converged]
    rx = [h for h in _legs(tracer, RX) if h.converged]
    assert tx and rx
    comps = targets.target_components(tx, rx, f0=8e6, v_T=np.array([0.0, 0.0, 50.0]),
                                      sigma_T_m2=10.0, Pt_dBW=40.0)
    assert len(comps) == len(tx) * len(rx)
    for comp, (o, r) in zip(comps, [(o, r) for o in tx for r in rx]):
        assert abs(comp.delay - (o.leg.Pg + r.leg.Pg) / C) < 1e-12
        assert comp.power > 0.0 and np.isfinite(comp.scatterer["f_T"])
        assert comp.gamma_or is None or 0.0 <= comp.gamma_or <= 1.0


def test_aircraft_target_is_reached_on_the_descending_branch_at_its_height():
    """§5.3: a target at altitude h_T is reached by the crossing event at h_T on the final hop. The homed leg's
    last sample is at h_T, and the ray is descending there (its altitude falls on the last steps)."""
    from raytracing.batch import BatchTracer as BT
    h_T = 10e3

    def B_field(x, y, z):
        return np.broadcast_to(B_EAST, np.shape(x) + (3,)).copy()

    tracer = BT(Ray(W0, +1, _layer(), B_field=B_field), ds_m=4000.0, landing_alt_m=h_T)
    homed = targets.home_site_to_point(tracer, "tx", *TX, *TARGET, target_alt_m=h_T, elev_grid=ELEV,
                                       az_grid=AZ, max_path_m=2500e3)
    ok = [h for h in homed if h.converged]
    assert ok, [(h.converged, h.residual_m, h.flags[:1]) for h in homed]
    for h in ok:
        assert abs(h.samples.alt[-1] - h_T) < 100.0, h.samples.alt[-1]
        assert h.samples.alt[-3] > h.samples.alt[-1]
        assert h.residual_m < 50.0
    try:
        targets.home_site_to_point(tracer, "tx", *TX, *TARGET, target_alt_m=0.0, elev_grid=ELEV[:3],
                                   az_grid=AZ[:1], max_path_m=2500e3)
    except ValueError:
        pass
    else:
        raise AssertionError("a tracer with a different landing height must be refused")
