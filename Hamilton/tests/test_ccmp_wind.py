"""
HiCIRF plan, Stage 4 tests: the CCMPWind lookup (synthetic data -- no real CCMP file is available, see
ccmp_wind.py's docstring for why) and that clutter_scene.scene_components actually uses it, per cell,
instead of the single scene-wide SurfaceInputs.wind_direction_deg, when given one.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from geophysics import ccmp_wind as cw


def _synthetic_field():
    lat = np.linspace(-10.0, 30.0, 5)
    lon = np.linspace(0.0, 20.0, 5)
    t = [datetime(2024, 6, 21, 0), datetime(2024, 6, 21, 6)]
    # A simple field: pure eastward wind (10 m/s) at t[0], pure northward (10 m/s) at t[1], uniform in space.
    uwnd = np.zeros((2, 5, 5)); uwnd[0] = 10.0
    vwnd = np.zeros((2, 5, 5)); vwnd[1] = 10.0
    return cw.CCMPWind(lat_deg=lat, lon_deg=lon, time=t, uwnd=uwnd, vwnd=vwnd)


def _fans():
    from propagation import fans
    from raytracing.batch import BatchTracer
    from core.geometry import latlon_to_xyz_m
    from raytracing.main import Ray
    from test_records_ground import _layer
    W0 = 2.0 * np.pi * 8e6
    B_EAST = np.array([0.0, 2e-5, 0.0])

    def _fan(lat, lon):
        def B_field(x, y, z):
            return np.broadcast_to(B_EAST, np.shape(x) + (3,)).copy()
        tracer = BatchTracer(Ray(W0, +1, _layer(), B_field=B_field), ds_m=4000.0)
        r0 = latlon_to_xyz_m(lat, lon, 0.0)
        up = r0 / np.linalg.norm(r0)
        east = np.array([-np.sin(np.radians(lon)), np.cos(np.radians(lon)), 0.0])
        north = np.cross(up, east)
        return fans.trace_fan(tracer, r0, up, north, east, np.arange(12.0, 24.01, 0.5),
                              np.linspace(-1.0, 1.0, 9), 2500e3, nhops=1, grid=True), r0

    fan_tx, r_tx = _fan(20.0, 0.0)
    fan_rx, r_rx = _fan(18.0, 0.0)
    return fan_tx, r_tx, fan_rx, r_rx


def test_eastward_wind_gives_90_degree_direction_and_right_speed():
    field = _synthetic_field()
    out = field.at(10.0, 10.0, datetime(2024, 6, 21, 0))
    assert abs(out["wind_direction_deg"][0] - 90.0) < 1e-6   # blowing towards east
    assert abs(out["wind_speed_mps"][0] - 10.0) < 1e-6


def test_northward_wind_at_the_second_time_gives_0_degrees():
    field = _synthetic_field()
    out = field.at(10.0, 10.0, datetime(2024, 6, 21, 6))
    assert abs(out["wind_direction_deg"][0] - 0.0) < 1e-6    # blowing towards north
    assert abs(out["wind_speed_mps"][0] - 10.0) < 1e-6


def test_nearest_time_lookup_snaps_to_the_closer_grid_point():
    field = _synthetic_field()
    out = field.at(10.0, 10.0, datetime(2024, 6, 21, 0) + timedelta(hours=1))   # closer to t[0] than t[1]
    assert abs(out["wind_direction_deg"][0] - 90.0) < 1e-6


def test_longitude_wraps_both_0_360_and_signed_conventions():
    field = _synthetic_field()
    a = field.at(10.0, 10.0, datetime(2024, 6, 21, 0))
    b = field.at(10.0, 10.0 - 360.0, datetime(2024, 6, 21, 0))   # same physical point, signed convention
    assert abs(a["wind_direction_deg"][0] - b["wind_direction_deg"][0]) < 1e-6


def test_vectorised_lookup_over_multiple_points():
    field = _synthetic_field()
    out = field.at(np.array([10.0, 20.0]), np.array([5.0, 15.0]), datetime(2024, 6, 21, 0))
    assert out["wind_direction_deg"].shape == (2,) and np.allclose(out["wind_direction_deg"], 90.0)


def test_scene_components_use_wind_model_per_cell_instead_of_the_scalar_placeholder():
    from echoes import clutter_scene as cs
    from propagation import fans
    from raytracing.batch import BatchTracer
    from core.constants import C  # noqa: F401
    from core.geometry import latlon_to_xyz_m
    from core.inputs import RadarInputs, SurfaceInputs
    from raytracing.main import Ray
    from test_records_ground import _layer

    W0 = 2.0 * np.pi * 8e6
    B_EAST = np.array([0.0, 2e-5, 0.0])

    def _fan(lat, lon):
        def B_field(x, y, z):
            return np.broadcast_to(B_EAST, np.shape(x) + (3,)).copy()
        tracer = BatchTracer(Ray(W0, +1, _layer(), B_field=B_field), ds_m=4000.0)
        r0 = latlon_to_xyz_m(lat, lon, 0.0)
        up = r0 / np.linalg.norm(r0)
        east = np.array([-np.sin(np.radians(lon)), np.cos(np.radians(lon)), 0.0])
        north = np.cross(up, east)
        return fans.trace_fan(tracer, r0, up, north, east, np.arange(12.0, 24.01, 0.5),
                              np.linspace(-1.0, 1.0, 9), 2500e3, nhops=1, grid=True), r0

    fan_tx, r_tx = _fan(20.0, 0.0)
    fan_rx, r_rx = _fan(18.0, 0.0)
    radar = RadarInputs()
    # Placeholder direction (0 deg, along the radar look) gives only one Bragg line of significant strength
    # near the cardioid's peak; a crosswind (90 deg) from CCMPWind should move where the power concentrates.
    wind = cw.CCMPWind(lat_deg=np.array([-90.0, 90.0]), lon_deg=np.array([-180.0, 180.0]),
                       time=[datetime(2024, 6, 21, 0)],
                       uwnd=np.array([[[10.0, 10.0], [10.0, 10.0]]]),
                       vwnd=np.array([[[0.0, 0.0], [0.0, 0.0]]]))
    comps_default, _ = cs.scene_components(fan_tx, fan_rx, r_tx, r_rx, f0=8e6,
                                           surface=SurfaceInputs(wind_direction_deg=0.0), radar=radar,
                                           cell_m=10e3, surface_of=lambda lat, lon: "sea")
    comps_ccmp, _ = cs.scene_components(fan_tx, fan_rx, r_tx, r_rx, f0=8e6,
                                        surface=SurfaceInputs(wind_direction_deg=0.0), radar=radar,
                                        cell_m=10e3, surface_of=lambda lat, lon: "sea",
                                        wind_model=wind, t=datetime(2024, 6, 21, 0))
    assert comps_default and comps_ccmp
    sigma_default = sorted(c.sigma0 for c in comps_default)
    sigma_ccmp = sorted(c.sigma0 for c in comps_ccmp)
    assert sigma_default != sigma_ccmp   # a different wind direction must change the per-line sigma0 split


def test_ccmp_url_and_filename_follow_the_rss_layout():
    from datetime import date
    assert cw.ccmp_filename(date(2026, 9, 15)) == "CCMP_Wind_Analysis_20260915_V03.1_L4.nc"
    assert cw.ccmp_url(date(2026, 9, 15)) == \
        "https://data.remss.com/ccmp/v03.1/Y2026/M09/CCMP_Wind_Analysis_20260915_V03.1_L4.nc"


def test_real_cached_ccmp_file_if_present():
    """Runs on a real RSS file when it is in the cache (ccmp_wind.fetch_ccmp_day downloads it); without one,
    the synthetic tests above cover the logic. No network access from the test suite."""
    import os
    from datetime import date
    path = os.path.join(cw.ccmp_cache_dir(), cw.ccmp_filename(date(2026, 9, 15)))
    if not os.path.exists(path):
        return
    w = cw.load_ccmp_netcdf(path)
    assert w.uwnd.shape == (4, 720, 1440) and len(w.time) == 4
    assert w.time[0] == datetime(2026, 9, 15, 0) and w.time[-1] == datetime(2026, 9, 15, 18)
    out = w.at(np.array([55.0, -50.0]), np.array([-30.0, 60.0]), datetime(2026, 9, 15, 7))
    assert np.all(np.isfinite(out["wind_speed_mps"])) and np.all(out["wind_speed_mps"] < 40.0)


def test_ccmp_absolute_direction_is_converted_to_phi_rel_per_cell():
    """A wind blowing exactly towards the radar (cells lie north of the sites, so towards 180 deg) must give
    the same per-cell lines as the scalar phi_rel = 0 input -- the absolute-vs-relative mix-up this replaces
    would have treated CCMP's 180 deg as 'waves moving away'."""
    from echoes import clutter_scene as cs
    from core.inputs import RadarInputs, SurfaceInputs
    fan_tx, r_tx, fan_rx, r_rx = _fans()
    radar = RadarInputs()
    towards_radar = cw.CCMPWind(lat_deg=np.array([-90.0, 90.0]), lon_deg=np.array([-180.0, 180.0]),
                                time=[datetime(2024, 6, 21, 0)], uwnd=np.zeros((1, 2, 2)),
                                vwnd=np.full((1, 2, 2), -10.0))
    scalar, _ = cs.scene_components(fan_tx, fan_rx, r_tx, r_rx, f0=8e6, surface=SurfaceInputs(wind_direction_deg=0.0),
                                    radar=radar, cell_m=10e3, surface_of=lambda lat, lon: "sea")
    ccmp, _ = cs.scene_components(fan_tx, fan_rx, r_tx, r_rx, f0=8e6, surface=SurfaceInputs(wind_direction_deg=90.0),
                                  radar=radar, cell_m=10e3, surface_of=lambda lat, lon: "sea",
                                  wind_model=towards_radar, t=datetime(2024, 6, 21, 0))
    a = np.array([c.sigma0 for c in scalar])
    b = np.array([c.sigma0 for c in ccmp])
    assert a.shape == b.shape and np.allclose(a, b, rtol=0.05)
