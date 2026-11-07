"""
Phase 6a scene tests (spec §12.5): the fan from the midpoint, the volume elements from the ray-tube cross-section, the
linear scaling with the reflectivity, the aspect law along the field (in-plane rays have psi = 0 with B east), and the
Doppler from the placeholder phase velocity. Synthetic layer; the Bragg inputs are placeholders.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from echoes import auroral_scene as asc
from echoes import auroral
from raytracing.batch import BatchTracer
from core.geometry import latlon_to_xyz_m
from core.inputs import BraggInputs, RadarInputs
from raytracing.main import Ray
from test_records_ground import _layer

W0 = 2.0 * np.pi * 8e6
B_EAST = np.array([0.0, 2e-5, 0.0])
ELEV = np.arange(12.0, 24.01, 2.0)
AZ = np.array([-2.0, 0.0, 2.0])


def _setup():
    lat, lon = 19.0, 0.0
    r0 = latlon_to_xyz_m(lat, lon, 0.0)
    up = r0 / np.linalg.norm(r0)
    lo = np.radians(lon)
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.cross(up, east)

    def B_field(x, y, z):
        return np.broadcast_to(B_EAST, np.shape(x) + (3,)).copy()

    tracer = BatchTracer(Ray(W0, +1, _layer(), B_field=B_field), ds_m=4000.0)
    return tracer, r0, up, north, east


def test_fan_components_scale_linearly_with_the_reflectivity_and_are_finite():
    tracer, r0, up, north, east = _setup()
    samples, E, A = asc.trace_midpoint_fan(tracer, r0, up, north, east, ELEV, AZ, 2500e3)
    bragg = BraggInputs()
    kw = dict(up=up, north=north, east=east, f0=8e6, radar=RadarInputs(), eta_threshold=0.0)
    comps = asc.auroral_components(tracer, samples, ELEV, AZ, bragg=bragg, **kw)
    assert len(comps) > 50
    p = np.array([c.power_W for c in comps])
    d = np.array([c.delay_s for c in comps])
    assert np.all(np.isfinite(p)) and np.all(p > 0.0) and np.all(d > 0.0)
    comps10 = asc.auroral_components(tracer, samples, ELEV, AZ,
                                     bragg=dataclasses.replace(bragg, eta_perp_m1=10 * bragg.eta_perp_m1), **kw)
    p10 = np.array([c.power_W for c in comps10])
    assert np.allclose(p10 / p, 10.0, rtol=1e-9)


def test_in_plane_rays_are_at_perfect_aspect_and_the_doppler_follows_v_ph():
    tracer, r0, up, north, east = _setup()
    samples, E, A = asc.trace_midpoint_fan(tracer, r0, up, north, east, ELEV, AZ, 2500e3)
    kw = dict(up=up, north=north, east=east, f0=8e6, radar=RadarInputs())
    comps = asc.auroral_components(tracer, samples, ELEV, AZ, bragg=BraggInputs(), **kw)
    psi = np.array([c.psi_deg for c in comps])
    assert psi.min() < 1e-6                                       # the meridian rays lie in the B-perpendicular plane
    for c in comps:
        if c.psi_deg < 1e-6:
            assert abs(c.eta - BraggInputs().eta_perp_m1) < 1e-18 * max(1.0, c.eta)
            break
    assert all(abs(c.doppler_Hz) < 1e-12 for c in comps)          # v_ph = 0 by default
    comps_v = asc.auroral_components(tracer, samples, ELEV, AZ,
                                     bragg=BraggInputs(v_ph_mps=(0.0, 0.0, 120.0)), **kw)
    assert any(abs(c.doppler_Hz) > 1e-6 for c in comps_v)
