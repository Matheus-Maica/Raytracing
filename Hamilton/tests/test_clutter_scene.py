"""
Phase 5d scene tests (spec §12.4, §12.4.1): the Tx and Rx landing maps rasterised onto a ground grid, the clutter
components of the scene, the Bragg lines in the range-Doppler map (exit criterion of Phase 5), the power-sum check,
and the array element phases. Synthetic layer, B east, Tx (20 N), Rx (18 N). Surface and radar inputs are placeholders.
"""

from __future__ import annotations

import numpy as np

from echoes import clutter
from echoes import clutter_scene as cs
from propagation import fans
from propagation import legs
from raytracing.batch import BatchTracer
from core.constants import C
from core.geometry import latlon_to_xyz_m
from core.inputs import RadarInputs, SurfaceInputs
from raytracing.main import Ray
from test_records_ground import _layer

W0 = 2.0 * np.pi * 8e6
B_EAST = np.array([0.0, 2e-5, 0.0])
TX = (20.0, 0.0)
RX = (18.0, 0.0)
ELEV = np.arange(12.0, 24.01, 0.5)          # fans must be denser than the cells: quads ~13 km in range
AZ = np.linspace(-1.0, 1.0, 9)


def _fan(lat, lon):
    def B_field(x, y, z):
        return np.broadcast_to(B_EAST, np.shape(x) + (3,)).copy()
    tracer = BatchTracer(Ray(W0, +1, _layer(), B_field=B_field), ds_m=4000.0)
    r0, up, north, east = (latlon_to_xyz_m(lat, lon, 0.0), None, None, None)
    up = r0 / np.linalg.norm(r0)
    lo = np.radians(lon)
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.cross(up, east)
    fan = fans.trace_fan(tracer, r0, up, north, east, ELEV, AZ, 2500e3, nhops=1, grid=True)
    return fan, r0


def test_scene_components_show_the_bragg_lines_and_conserve_power():
    fan_tx, r_tx = _fan(*TX)
    fan_rx, r_rx = _fan(*RX)
    surface = SurfaceInputs(wind_direction_deg=90.0)              # crosswind: both lines present
    radar = RadarInputs()
    comps, grid = cs.scene_components(fan_tx, fan_rx, r_tx, r_rx, f0=8e6, surface=surface, radar=radar,
                                      cell_m=10e3, surface_of=lambda lat, lon: "sea")
    assert comps and grid["n_cells_hit"] >= 3
    delays = np.array([c.delay_s for c in comps])
    assert 5e-3 < delays.min() and delays.max() < 12e-3, (delays.min(), delays.max())
    tau_edges = np.linspace(delays.min() - 1e-4, delays.max() + 1e-4, 81)
    dopp = np.linspace(-1.0, 1.0, 801)
    P = clutter.range_doppler_map([(c.delay_s, c.doppler_Hz, c.power_W) for c in comps], tau_edges, dopp)
    total_in = sum(c.power_W for c in comps)
    assert abs(P.sum() / total_in - 1.0) < 1e-9                    # power-sum check
    profile = P.sum(axis=0)
    fB = clutter.bragg_frequency_hz(8e6)
    peaks = np.sort(dopp[np.argsort(profile)[-2:]])
    assert abs(peaks[0] + fB) < 0.01 and abs(peaks[1] - fB) < 0.01, peaks


def test_land_cells_give_a_zero_doppler_line():
    fan_tx, r_tx = _fan(*TX)
    fan_rx, r_rx = _fan(*RX)
    comps, _ = cs.scene_components(fan_tx, fan_rx, r_tx, r_rx, f0=8e6, surface=SurfaceInputs(),
                                   radar=RadarInputs(), cell_m=10e3, surface_of=lambda lat, lon: "land")
    assert comps and all(c.surface == "land" and c.line_hz == 0.0 for c in comps)
    assert all(abs(c.sigma0 - 10.0 ** (-25.0 / 10.0)) < 1e-12 for c in comps)


def test_element_phases_follow_the_direction_cosines():
    fan_tx, r_tx = _fan(*TX)
    fan_rx, r_rx = _fan(*RX)
    comps, _ = cs.scene_components(fan_tx, fan_rx, r_tx, r_rx, f0=8e6, surface=SurfaceInputs(wind_direction_deg=90.0),
                                   radar=RadarInputs(), cell_m=10e3, surface_of=lambda lat, lon: "sea")
    up = r_rx / np.linalg.norm(r_rx)
    lo = np.radians(RX[1])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.cross(up, east)
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 20.0]])
    H = cs.element_phases(comps[:5], pos, f0=8e6, east=east, north=north)
    assert np.allclose(np.abs(H), 1.0)
    k0 = 2.0 * np.pi * 8e6 / C
    for k, c in enumerate(comps[:5]):
        u, v = np.dot(c.e_ret, east), np.dot(c.e_ret, north)
        assert abs(np.angle(H[1, k] / H[0, k]) - np.angle(np.exp(1j * k0 * u * 10.0))) < 1e-9
