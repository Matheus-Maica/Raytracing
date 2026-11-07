"""
Phase 1h tests: scene-covering patch, the 50 km floor with 1 km spacing below 130 km, E-CHAIM's
FIRI D-region (decision D4), and the two-epoch pair of eq. (4.8). Small patches keep E-CHAIM and
IRI queries cheap (4 x 4 angular points).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from geophysics import ionosphere as I
from core.geometry import latlon_to_xyz_m

T0 = datetime(2024, 6, 21, 12, 0, 0)
SMALL = dict(center_lat=65.0, center_lon=20.0, dtheta_deg=2.0, dphi_deg=4.0, h_max=300.0,
             dr=2.0, dang_deg=0.5, n_ang=4)


def test_covering_patch_contains_points_with_margin():
    pts = [(66.0, 10.0), (60.0, 40.0), (62.0, 25.0)]
    margin_km = 1500.0
    p = I.covering_patch(pts, margin_km)
    m_lat = margin_km / 111.2
    half_lat = p["dtheta_deg"] / 2.0
    lat = np.array([q[0] for q in pts])
    lon = np.array([q[1] for q in pts])
    assert np.all(np.abs(lat - p["center_lat"]) <= half_lat - m_lat + 1e-9)
    assert abs(np.max(np.abs(lat - p["center_lat"])) - (half_lat - m_lat)) < 1e-9   # margin is exactly m
    assert np.all(np.abs(lon - p["center_lon"]) <= p["dphi_deg"] / 2.0)


def test_covering_patch_handles_antimeridian():
    p = I.covering_patch([(0.0, 179.0), (0.0, -179.0)], 0.0)
    assert abs(p["dphi_deg"] - 2.0) < 1e-9             # the short way round, not 358 degrees
    assert abs(abs(p["center_lon"]) - 180.0) < 1e-9


def test_altitude_axis_floor_and_spacing():
    h = I.Ionosphere._altitude_axis(50.0, 800.0, 2.0)
    assert h[0] == 50.0 and abs(h[-1] - 800.0) < 1e-9
    assert np.all(np.diff(h) > 0.0)
    assert np.max(np.diff(h[h <= 130.0])) <= 1.0 + 1e-9
    assert np.max(np.diff(h[h >= 130.0])) <= 2.0 + 1e-9


def test_scene_patch_grid_covers_sites():
    p = I.covering_patch([(65.0, 20.0), (60.0, 30.0)], 300.0)
    iono = I.Ionosphere(**p, h_max=300.0, dr=2.0, dang_deg=0.5, n_ang=4, dt=T0)
    for lat, lon in ((65.0, 20.0), (60.0, 30.0)):
        assert iono.lat_min <= lat <= iono.lat_max and iono.lon_min <= lon <= iono.lon_max


def test_dregion_nonzero_and_smooth_in_50_60_km():
    """E-CHAIM's FIRI D-region (dregion=True) at 65 N: positive and smoothly rising through 52-60 km,
    and the accessor returns a profile with the shape of its height argument (broadcast fix)."""
    iono = I.Ionosphere(**SMALL, dt=T0)
    alts = np.arange(52.0, 60.0, 1.0)
    prof = iono.electron_density_geodetic(65.0, 20.0, alts)
    assert prof.shape == alts.shape
    assert np.all(prof > 0.0)
    ratio = prof[1:] / prof[:-1]
    assert np.all((ratio > 1.1) & (ratio < 2.0)), ratio


def test_dregion_off_leaves_no_d_layer_at_valid_points():
    """Control for the D4 switch: without FIRI, E-CHAIM's base profile is zero below ~70 km here."""
    iono = I.Ionosphere(**SMALL, dt=T0, echaim_dregion=False)
    prof = iono.electron_density_geodetic(65.0, 20.0, np.arange(52.0, 60.0, 1.0))
    assert np.all(prof == 0.0)


def test_epoch_pair_shares_grid_and_rate_is_central_difference():
    pair = I.EpochPair(T0, half_dt_s=60.0, **SMALL)
    assert np.array_equal(pair.minus.h, pair.plus.h)
    assert np.array_equal(pair.minus.lat, pair.plus.lat) and np.array_equal(pair.minus.lon, pair.plus.lon)
    xyz = np.array([latlon_to_xyz_m(la, lo, h) for la, lo, h in
                    ((64.5, 19.0, 100e3), (65.0, 20.0, 110e3), (65.5, 21.0, 90e3))])
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rate = pair.dne_dt_xyz_m(x, y, z)
    manual = (pair.plus.electron_density_xyz_m(x, y, z) - pair.minus.electron_density_xyz_m(x, y, z)) / 120.0
    assert np.allclose(rate, manual, rtol=1e-12, atol=0.0)
    assert np.any(rate != 0.0)
