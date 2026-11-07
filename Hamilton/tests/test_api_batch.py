"""
Batch-path public API (api.Link): trace() runs on batch.BatchTracer, the RayPath agrees with a direct batch run,
unsupported options raise, the phase variance comes from the leg's screens, and the link budget uses the same ray
(so absorption agrees with trace()). Small patch to keep the test quick.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

import api
from raytracing.batch import BatchTracer, samples_from_record
from core.constants import C


def _link():
    return api.Link(57.0, 10.0, frequency_mhz=8.0, dt=datetime(2024, 6, 21, 14, 0, 0), mode="O",
                    max_path_km=1500.0, max_nhops=1, n_ang=4, dang_deg=2.0, igrf_n_ang=3, igrf_n_r=3,
                    verbose=False)


def test_trace_agrees_with_a_direct_batch_run_and_is_consistent():
    link = _link()
    ray = link.trace(28.0)
    tracer = BatchTracer(link._resolve_ray(None, None), ds_m=2000.0, absorption_model=link.absorption_model)
    u0 = link._direction(28.0, 0.0)
    res = tracer.trace(link.r0, u0[None], link._path_budget_m(None), nhops=1, want_group=True,
                       want_absorption=True, record_states=True)
    assert ray.landed == bool(res.landed[0])
    assert abs(ray.arc_length_km - res.s_total[0] / 1e3) < 1e-9
    assert abs(ray.group_path_km - res.group_path[0] / 1e3) < 1e-9
    assert abs(ray.absorption_db - res.absorption_db[0]) < 1e-9
    assert ray.path.shape[0] == samples_from_record(tracer, res.records[0], transport=False).r.shape[0]
    assert ray.landed and 300.0 < ray.ground_range_km < 1500.0
    assert abs(ray.group_delay_ms - 1e3 * ray.group_path_km * 1e3 / C) < 1e-9


def test_scalar_path_options_are_gone_and_phase_variance_is_finite():
    link = _link()
    for kwargs in ({"tangent_map": True}, {"method": "adaptive"}):    # scalar-path options: not on the batch path
        try:
            link.trace(28.0, **kwargs)
        except TypeError:
            pass
        else:
            raise AssertionError(f"{kwargs} is not a trace() option any more")
    ray = link.trace(28.0, phase_variance=True)
    assert ray.sigma_phi2 is not None and np.isfinite(ray.sigma_phi2) and ray.sigma_phi2 > 0.0
    assert abs(ray.sigma_phi2 - link.channel(ray).sigma_phi2) < 1e-9 * ray.sigma_phi2


def test_link_budget_uses_the_same_ray_as_trace():
    link = _link()
    ray = link.trace(28.0)
    budget = link.link_budget(28.0)
    assert budget.landed == ray.landed
    assert abs(budget.absorption_db - ray.absorption_db) < 1e-6
    assert abs(budget.ground_range_km - ray.ground_range_km) < 5.0


def test_leg_channel_returns_the_channel_parameters_of_the_traced_leg():
    link = _link()
    cp = link.leg_channel(28.0)
    assert np.isfinite(cp.sigma_phi2) and cp.sigma_phi2 > 0.0
    assert 0.0 < cp.S4 and np.isfinite(cp.tau_o) and cp.tau_o > 0.0
    assert cp.lx <= cp.ly
    # the last screen sits at the far end of the leg (w = 0); it must not turn these into NaN
    assert np.all(np.isfinite(cp.V_app)) and np.isfinite(cp.tau_mean) and cp.tau_mean > 0.0
    assert np.isfinite(cp.w_coh) and cp.w_coh > 0.0 and np.isfinite(cp.alpha)


def test_channel_of_a_traced_ray_is_the_leg_channel_and_uses_the_ray_itself():
    link = _link()
    ray = link.trace(28.0)
    assert ray.samples is not None and ray.samples.r.shape == ray.path.shape
    a, b = link.channel(ray), link.leg_channel(28.0)
    for name in ("sigma_phi2", "S4", "tau_o", "tau_mean", "w_coh", "lx", "ly"):
        assert abs(getattr(a, name) - getattr(b, name)) <= 1e-9 * abs(getattr(a, name)), name
    # a shorter path budget gives a different leg, and channel(ray) follows that ray, not the Link's defaults
    short = link.trace(28.0, max_path_km=600.0)
    assert short.status == "incomplete"
    assert link.channel(short).sigma_phi2 < a.sigma_phi2
    no_group = link.trace(28.0, group_path=False)
    try:
        link.channel(no_group)
    except ValueError:
        pass
    else:
        raise AssertionError("the channel needs the group path")


def test_link_inputs_mode_and_identity():
    from dataclasses import replace
    base = api.IrregularityInputs(axial_ratio_sheet=2.0, drift_east_north_mps=(0.0, 50.0))
    link = api.Link(57.0, 10.0, frequency_mhz=8.0, dt=datetime(2024, 6, 21, 14, 0, 0), mode=-1,
                    max_path_km=1500.0, max_nhops=1, n_ang=4, dang_deg=2.0, igrf_n_ang=3, igrf_n_r=3,
                    verbose=False, irregularity=base, xi_rms=1e-2)
    assert link.irregularity == replace(base, xi=1e-2)                  # shortcut overrides one field only
    assert link.xi_rms == 1e-2 and link.axial_ratio == base.axial_ratio_B
    assert link.mode == "X" and "mode='X'" in repr(link)
    ray = link.trace(28.0)
    assert ray.mode == "X" and ray == ray and ray != link.trace(28.0)   # equality is identity
    assert "phase variance" not in str(ray)

