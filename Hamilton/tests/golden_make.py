"""
Generate the golden reference used by the vectorisation equivalence tests (Phase 1a).

Run once, from the scalar code as it stands after Phase 0:

    python tests/golden_make.py

The output tests/data/golden_phase0.npz holds RHS, group index, phase sensitivity and
absorption values at random states, and two fixed-step traces (O and X) through a
synthetic layer. The vectorised code must reproduce them (tests/test_vectorised.py).
Regenerate only if the physics is deliberately changed, and say so in the log.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from raytracing import absorption  # noqa: E402
from core.geometry import WGS84_A, latlon_to_xyz_m  # noqa: E402
from raytracing.main import Ray  # noqa: E402
from raytracing.raytracer import RayTracer  # noqa: E402

F0 = 8.0e6
W0 = 2.0 * np.pi * F0


def Ne_layer(x, y, z):
    h = np.sqrt(x * x + y * y + z * z) - WGS84_A
    return 1.2e12 * np.exp(-0.5 * ((h - 300e3) / 60e3) ** 2)


def random_states(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, 6))
    for i in range(n):
        lat = rng.uniform(-80.0, 80.0)
        lon = rng.uniform(-180.0, 180.0)
        h = rng.uniform(0.0, 500e3)
        r = latlon_to_xyz_m(lat, lon, h)
        u = rng.normal(size=3)
        u = u / np.linalg.norm(u) * rng.uniform(0.3, 1.0)
        out[i, :3] = r
        out[i, 3:] = u
    return out


def on_shell(ray, states, iters: int = 60):
    """
    Project states onto the dispersion shell of ray.MODE: keep the direction of u and choose its
    magnitude so that |u|^2 = 1 + eps(X, Y, Y_hat . u), the root the RHS and group index use.
    The group index is defined only on the shell (its partials are evaluated there).
    """
    st = np.array(states, dtype=float)
    x, y, z = st[:, 0], st[:, 1], st[:, 2]
    ne, Yv, _ = ray._fields(x, y, z, grad=False)
    X = ne * ray.X_factor
    Y = np.linalg.norm(Yv, axis=-1)
    v = Yv / Y[:, None]
    t = np.linalg.norm(st[:, 3:6], axis=-1)
    d = st[:, 3:6] / t[:, None]
    c = np.sum(v * d, axis=-1)
    for _ in range(iters):
        eps, _, _ = ray._dispersion_eps(X, Y, t * c, ray.MODE)
        t = np.sqrt(1.0 + eps)
    st[:, 3:6] = d * t[:, None]
    return st


def trace_summary(res):
    return dict(
        final_state=np.concatenate([res.r_final, res.u_final]),
        s_total=res.s_total, phase=res.phase_path_total, group=res.group_path_total,
        absorption_amp=res.absorption_amplitude_nepers_total, absorption_db=res.absorption_db_total,
        ground_range=res.ground_range_total_m, landed=int(res.landed), hops=res.hops_completed,
        path=res.path[::10], s_path=res.s_path[::10],
    )


def main():
    out = {}
    states = random_states(300)
    out["states"] = states
    for MODE in (+1, -1):
        ray = Ray(W0, MODE, Ne_layer)
        tracer = RayTracer(ray, ds_step_m=2000.0)
        key = "O" if MODE == +1 else "X"
        shell = on_shell(ray, states)
        shell = shell[np.all(np.isfinite(shell), axis=1)]     # drop evanescent states (no real root)
        out[f"states_{key}"] = shell
        rhs = np.array([ray._haselgrove_physical_rhs(0.0, s) for s in states])
        ng = np.array([ray.group_index(s) for s in shell])
        g = np.array([ray.phase_sensitivity(s) for s in shell])
        alpha = np.array([absorption.absorption_rate(ray, s[:3], s[3:]) for s in shell])
        out[f"rhs_{key}"] = rhs
        out[f"ng_{key}"] = ng
        out[f"g_{key}"] = g
        out[f"alpha_{key}"] = alpha
        r0 = latlon_to_xyz_m(57.0, 10.0, 0.0)
        up = r0 / np.linalg.norm(r0)
        north = np.array([-np.sin(np.radians(57)) * np.cos(np.radians(10)),
                          -np.sin(np.radians(57)) * np.sin(np.radians(10)),
                          np.cos(np.radians(57))])
        u0 = np.cos(np.radians(28.0)) * north + np.sin(np.radians(28.0)) * up
        res = tracer.trace(r0, u0, 2500e3, nhops=2, method="fixed", compute_group_path=True,
                           compute_absorption=True, compute_phase_variance=False)
        summ = trace_summary(res)
        for name, value in summ.items():
            out[f"trace_{key}_{name}"] = np.asarray(value, dtype=float)
        print(key, "landed", summ["landed"], "range km", round(summ["ground_range"] / 1e3, 1))
    path = HERE / "data" / "golden_phase0.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **out)
    print("wrote", path, "keys:", len(out))


if __name__ == "__main__":
    main()
