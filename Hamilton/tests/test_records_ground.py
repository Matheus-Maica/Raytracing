"""
Phase 1b-1e tests: per-step records (spec §4.1), the local helpers and branch check (§4.2),
the transported frame (§4.3, Algorithm 1) and Fresnel coefficients (§4.4, eq. 4.3-4.4).
"""

from __future__ import annotations

import numpy as np

from raytracing import batch
from raytracing import ground
from core.geometry import latlon_to_xyz_m, WGS84_A
from raytracing.main import Ray
from core.constants import C

F0 = 8.0e6
W0 = 2.0 * np.pi * F0


def _layer(h0=300e3, peak=1.2e12, sigma=60e3):
    def Ne(x, y, z):
        h = np.sqrt(x * x + y * y + z * z) - WGS84_A
        return peak * np.exp(-0.5 * ((h - h0) / sigma) ** 2)
    return Ne


def _launch(lat=57.0, lon=10.0, elev=28.0, az=0.0):
    r0 = latlon_to_xyz_m(lat, lon, 0.0)
    up = r0 / np.linalg.norm(r0)
    la, lo = np.radians(lat), np.radians(lon)
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    e, a = np.radians(elev), np.radians(az)
    d = np.cos(e) * (np.cos(a) * north + np.sin(a) * east) + np.sin(e) * up
    return r0, d / np.linalg.norm(d), up, east


def test_recording_does_not_change_trajectory():
    ray = Ray(W0, +1, _layer())
    bt = batch.BatchTracer(ray, ds_m=4000.0)
    r0, u0, _, _ = _launch()
    plain = bt.trace(r0, u0[None], 1800e3, nhops=2)
    recorded = bt.trace(r0, u0[None], 1800e3, nhops=2, record_states=True, bounce_models=True)
    assert np.array_equal(plain.r_final, recorded.r_final)
    assert np.array_equal(plain.u_final, recorded.u_final)
    assert plain.s_total[0] == recorded.s_total[0]
    assert plain.landed[0] == recorded.landed[0] and plain.hops[0] == recorded.hops[0]


def test_branch_check_and_direction_on_samples():
    """Spec §4.2 item 2: n_angle(X, Y, cos theta; mode) equals |u|^2 to 1e-4 relative, and the
    physical ray direction points along the wave normal (forward) in the plasma."""
    for MODE in (+1, -1):
        ray = Ray(W0, MODE, _layer())
        bt = batch.BatchTracer(ray, ds_m=4000.0)
        r0, u0, _, _ = _launch()
        res = bt.trace(r0, u0[None], 900e3, nhops=1, record_states=True)
        rec = res.records[0]
        sam = batch.samples_from_record(bt, rec, transport=False)
        plasma = ~sam.vacuum
        assert plasma.sum() > 50
        Y = np.linalg.norm(sam.B, axis=1) * ray.Y_factor
        cos_th = np.sum(sam.u_hat * sam.B, axis=1) / np.linalg.norm(sam.B, axis=1)
        p_angle = np.array([ray.n_angle(sam.X[i], Y[i], cos_th[i], MODE) for i in np.flatnonzero(plasma)])
        n2 = sam.n[plasma] ** 2
        assert np.max(np.abs(np.real(p_angle) / n2 - 1.0)) < 1e-4
        assert np.all(np.sum(sam.s_hat[plasma] * sam.u_hat[plasma], axis=1) > 0.0)


def test_samples_fields_consistent():
    ray = Ray(W0, +1, _layer())
    bt = batch.BatchTracer(ray, ds_m=4000.0)
    r0, u0, _, _ = _launch()
    res = bt.trace(r0, u0[None], 700e3, nhops=1, record_states=True)
    sam = batch.samples_from_record(bt, res.records[0], transport=False)
    assert np.allclose(sam.X, sam.Ne * ray.X_factor, rtol=1e-12, atol=0.0)
    assert np.allclose(np.linalg.norm(sam.s_hat, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(sam.u_hat, axis=1), 1.0, atol=1e-12)
    assert np.all(np.diff(sam.s) >= 0.0)
    assert np.all(sam.absorption_db >= -1e-12)


def test_frame_planar_ray_stays_normal_to_plane():
    """Algorithm 1: a ray in a meridian plane (isotropic medium, azimuth 0): t1 stays along east
    (normal to the plane) to 1e-9, and both vectors stay orthonormal, through a ground bounce."""
    def Ne_shell(x, y, z):
        h = np.sqrt(x * x + y * y + z * z) - WGS84_A
        return 1.0e12 * np.exp(-0.5 * ((h - 300e3) / 60e3) ** 2)
    # B exactly zero is degenerate for Haselgrove's form (rdot = 0, see algorithm.md §8.6). A field
    # along east (normal to the meridian plane y = 0) keeps the mirror symmetry, so the ray stays planar.
    B_east = np.array([0.0, 2e-5, 0.0])
    ray = Ray(W0, +1, Ne_shell, B_field=lambda x, y, z: np.broadcast_to(B_east, np.shape(x) + (3,)).copy())
    bt = batch.BatchTracer(ray, ds_m=4000.0)
    r0, u0, up, east = _launch(lat=20.0, lon=0.0, elev=30.0, az=0.0)
    res = bt.trace(r0, u0[None], 1200e3, nhops=2, record_states=True)
    rec = res.records[0]
    sam = batch.samples_from_record(bt, rec, transport=True)
    assert np.min(np.abs(np.sum(sam.t1 * east, axis=1))) > 1 - 1e-9
    assert np.max(np.abs(np.sum(sam.t1 * sam.t2, axis=1))) < 1e-10
    assert np.allclose(np.linalg.norm(sam.t1, axis=1), 1.0, atol=1e-10)
    assert np.allclose(np.linalg.norm(sam.t2, axis=1), 1.0, atol=1e-10)


def test_fresnel_limits():
    """Eq. (4.3): R_h = -R_v at normal incidence; |R| -> 1 at grazing; |R| < 1 otherwise;
    bounce loss positive."""
    Rh, Rv = ground.fresnel(np.pi / 2.0, 70.0, 5.0, F0)
    assert abs(Rh + Rv) < 1e-12
    Rh_g, Rv_g = ground.fresnel(np.radians(1e-4), 70.0, 5.0, F0)   # |eps_c| ~ 9e3: asymptote needs tiny psi
    assert abs(abs(Rh_g) - 1.0) < 1e-3 and abs(abs(Rv_g) - 1.0) < 1e-3
    Rh_m, Rv_m = ground.fresnel(np.radians(20.0), 70.0, 5.0, F0)
    assert abs(Rh_m) < 1.0 and abs(Rv_m) < 1.0
    assert ground.bounce_loss_dB(Rh_m, Rv_m) > 0.0


def test_bounce_record_has_loss_and_grazing():
    """Each ground reflection gives a Bounce record: grazing angle in (0, 90 deg), loss > 0,
    surface type set, and the record index points at the crossing sample."""
    ray = Ray(W0, +1, _layer())
    bt = batch.BatchTracer(ray, ds_m=4000.0)
    r0, u0, _, _ = _launch()
    res = bt.trace(r0, u0[None], 2500e3, nhops=2, bounce_models=True, record_states=True)
    assert res.hops[0] >= 1
    assert len(res.bounces[0]) == res.hops[0] - (1 if res.landed[0] else 0)   # final landing has no bounce record
    for b in res.bounces[0]:
        assert 0.0 < b.grazing < np.pi / 2.0
        assert b.loss_dB > 0.0
        assert b.surface in ("sea", "land")
    assert len(res.records[0]["bounce_index"]) == res.hops[0]
