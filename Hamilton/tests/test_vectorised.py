"""
Phase 1a: the shape-agnostic (batch) physics against the golden outputs of the scalar code
as it stood after Phase 0 (tests/golden_make.py -> tests/data/golden_phase0.npz).

The vectorised code must reproduce the RHS, group index and phase sensitivity at 300
random states, and the fixed-step trace summaries, to floating-point accuracy.
"""

from __future__ import annotations

import pathlib

import numpy as np

import golden_make as gm
from raytracing.main import Ray

GOLD = pathlib.Path(__file__).resolve().parent / "data" / "golden_phase0.npz"


def _load():
    return np.load(GOLD)


def _rel(a, b):
    scale = np.maximum(np.abs(b), 1e-300)
    return np.max(np.abs(a - b) / scale)


def test_rhs_batch_matches_golden():
    g = _load()
    states = g["states"]
    for MODE, key in ((+1, "O"), (-1, "X")):
        ray = Ray(gm.W0, MODE, gm.Ne_layer)
        batch = ray._haselgrove_physical_rhs(0.0, states)           # (300, 6) in one call
        err = _rel(batch, g[f"rhs_{key}"])
        assert err < 1e-10, f"{key}: relative error {err:.2e}"


def test_rhs_scalar_and_batch_agree_exactly():
    """The same code path for one state and for a batch of states."""
    g = _load()
    states = g["states"][:20]
    for MODE in (+1, -1):
        ray = Ray(gm.W0, MODE, gm.Ne_layer)
        batch = ray._haselgrove_physical_rhs(0.0, states)
        single = np.array([ray._haselgrove_physical_rhs(0.0, s) for s in states])
        assert np.max(np.abs(batch - single)) <= 1e-12 * np.max(np.abs(batch))


def test_group_index_batch_matches_golden():
    g = _load()
    for MODE, key in ((+1, "O"), (-1, "X")):
        ray = Ray(gm.W0, MODE, gm.Ne_layer)
        states = g[f"states_{key}"]                 # on the dispersion shell of this mode
        batch = ray.group_index(states)
        err = _rel(batch, g[f"ng_{key}"])
        assert err < 1e-10, f"{key}: relative error {err:.2e}"


def test_phase_sensitivity_batch_matches_golden():
    g = _load()
    for MODE, key in ((+1, "O"), (-1, "X")):
        ray = Ray(gm.W0, MODE, gm.Ne_layer)
        states = g[f"states_{key}"]
        batch = ray.phase_sensitivity(states)
        err = _rel(batch, g[f"g_{key}"])
        assert err < 1e-10, f"{key}: relative error {err:.2e}"


def _launch():
    r0 = gm.latlon_to_xyz_m(57.0, 10.0, 0.0)
    up = r0 / np.linalg.norm(r0)
    north = np.array([-np.sin(np.radians(57)) * np.cos(np.radians(10)),
                      -np.sin(np.radians(57)) * np.sin(np.radians(10)),
                      np.cos(np.radians(57))])
    u0 = np.cos(np.radians(28.0)) * north + np.sin(np.radians(28.0)) * up
    return r0, u0


def test_batch_trace_matches_golden_traces():
    """BatchTracer (vectorised, N = 1) reproduces the scalar RayTracer.trace outputs."""
    from raytracing.batch import BatchTracer
    g = _load()
    r0, u0 = _launch()
    for MODE, key in ((+1, "O"), (-1, "X")):
        ray = Ray(gm.W0, MODE, gm.Ne_layer)
        res = BatchTracer(ray, ds_m=2000.0).trace(r0, u0[None, :], 2500e3, nhops=2, record_every=10)
        final = np.concatenate([res.r_final[0], res.u_final[0]])
        assert _rel(final, g[f"trace_{key}_final_state"]) < 1e-8, f"{key} final state"
        for name, value in (("s_total", res.s_total[0]), ("phase", res.phase_path[0]),
                            ("group", res.group_path[0]),
                            ("absorption_amp", res.absorption_amplitude[0]),
                            ("absorption_db", res.absorption_db[0]),
                            ("ground_range", res.ground_range_m[0])):
            ref = float(g[f"trace_{key}_{name}"])
            assert abs(value - ref) <= 1e-8 * max(abs(ref), 1.0), f"{key} {name}: {value} vs {ref}"
        assert int(res.landed[0]) == int(g[f"trace_{key}_landed"]), f"{key} landed"
        assert int(res.hops[0]) == int(g[f"trace_{key}_hops"]), f"{key} hops"
        path = res.paths[0][::10]
        ref_path = g[f"trace_{key}_path"]
        assert path.shape == ref_path.shape, (path.shape, ref_path.shape)
        assert np.max(np.abs(path - ref_path)) <= 1e-6 * np.max(np.abs(ref_path)), f"{key} path"


def test_batch_matches_scalar_for_many_rays():
    """A batch of rays gives the same result as tracing each ray alone."""
    from raytracing.batch import BatchTracer
    r0, u0 = _launch()
    rng = np.random.default_rng(4)
    dirs = u0 + 0.02 * rng.normal(size=(6, 3))
    ray = Ray(gm.W0, +1, gm.Ne_layer)
    bt = BatchTracer(ray, ds_m=4000.0)
    together = bt.trace(r0, dirs, 1500e3, nhops=1)
    for i in range(dirs.shape[0]):
        alone = bt.trace(r0, dirs[i:i + 1], 1500e3, nhops=1)
        assert np.allclose(together.r_final[i], alone.r_final[0], rtol=1e-12, atol=1e-6)
        assert abs(together.s_total[i] - alone.s_total[0]) <= 1e-9 * max(alone.s_total[0], 1.0)
        assert together.landed[i] == alone.landed[0]
