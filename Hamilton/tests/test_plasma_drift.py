"""
Plasma drift tests: the IGRF centred-dipole frame, the full Heelis port (docs/heelis.F90.txt), our own crude
mid/low-latitude pattern, and their superposition (WBMODStyleDrift), end to end through api.Link.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

import api
from geophysics import plasma_drift as pd
from core.geometry import latlon_to_xyz_m

CTP = 30.0
THETA0_DEG = -3.80 + 8.48 * CTP ** 0.1875      # heelis_init


def test_subsolar_declination_matches_the_solstices():
    _, _, decl_june = pd.subsolar_point_deg(datetime(2024, 6, 21, 12, 0, 0))
    _, _, decl_dec = pd.subsolar_point_deg(datetime(2024, 12, 21, 12, 0, 0))
    assert abs(decl_june - 23.4) < 0.6
    assert abs(decl_dec + 23.4) < 0.6


def test_subsolar_point_is_near_local_noon_at_its_own_longitude():
    lat, lon, _ = pd.subsolar_point_deg(datetime(2024, 3, 20, 12, 0, 0))
    assert abs(lat) < 2.0
    assert abs(((lon + 180.0) % 360.0) - 180.0) < 5.0


def test_igrf_dipole_pole_is_where_igrf_puts_it():
    # IGRF-13/14 north geomagnetic (centred dipole) pole, epoch 2025: about 80.8 N, 72.8 W.
    ax = pd.dipole_axis_ecef(datetime(2025, 1, 1))
    assert abs(np.rad2deg(np.arcsin(ax[2])) - 80.8) < 0.2
    assert abs(np.rad2deg(np.arctan2(ax[1], ax[0])) + 72.8) < 0.5


def test_magnetic_east_increases_magnetic_longitude():
    t = datetime(2025, 1, 1)
    r0 = latlon_to_xyz_m(55.0, 20.0, 300e3)
    r_hat = r0 / np.linalg.norm(r0)
    east = np.cross(pd.dipole_axis_ecef(t), r_hat)
    east /= np.linalg.norm(east)
    _, lon0 = pd.geomagnetic_lat_lon_rad(r0[None, :], t)
    lat1, lon1 = pd.geomagnetic_lat_lon_rad((r0 + 2000.0 * east)[None, :], t)
    lat0, _ = pd.geomagnetic_lat_lon_rad(r0[None, :], t)
    assert (lon1[0] - lon0[0]) > 0.0 and abs(lat1[0] - lat0[0]) < 1e-6


def test_heelis_dawn_positive_dusk_negative_zero_at_noon_and_midnight():
    mlat = np.deg2rad(80.0)
    dawn, noon, dusk, midnight = (float(pd.heelis_potential_kV(mlat, np.deg2rad(d), CTP)) for d in (-90, 0, 90, 180))
    assert dawn > 0.0 and dusk < 0.0 and abs(dawn + dusk) < 1e-9
    assert abs(noon) < 1e-9 and abs(midnight) < 1e-9
    assert dawn < 0.5 * CTP + 1e-9   # never exceeds psim = ctpoten/2


def test_heelis_longitudinal_function_reduces_to_a_single_harmonic_with_default_parameters():
    """With TIE-GCM's default parameters the 7-segment function is -psim sin(alon); on the convection reversal
    boundary (x = 1, where the cubic equals phifun) the potential is that harmonic -- checked off the
    auroral circle's offset axis, at dlon = -90 and -45 deg."""
    ofs = np.deg2rad(1.0)
    th0 = np.deg2rad(THETA0_DEG)
    for d in (-90.0, -45.0):
        dl = np.deg2rad(d)
        # magnetic latitude whose auroral-circle colatitude is exactly theta0 at this dlon (solve numerically)
        lats = np.deg2rad(np.linspace(70.0, 85.0, 200001))
        colat = np.arccos(np.cos(ofs) * np.sin(lats) - np.sin(ofs) * np.cos(lats) * np.cos(dl))
        lat = lats[np.argmin(np.abs(colat - th0))]
        alon = np.arctan2(np.sin(dl) * np.cos(lat), np.sin(lat) * np.sin(ofs) + np.cos(ofs) * np.cos(lat) * np.cos(dl))
        got = float(pd.heelis_potential_kV(lat, dl, CTP))
        assert abs(got - (-0.5 * CTP * np.sin(alon))) < 2e-3 * CTP, (d, got)


def test_heelis_is_continuous_across_the_reversal_boundary_and_zero_below_30_deg():
    dl = np.deg2rad(-90.0)
    lat_b = np.deg2rad(90.0 - THETA0_DEG + 1.0)     # circle centre is 1 deg towards midnight: at dawn ~boundary
    a = float(pd.heelis_potential_kV(lat_b + 1e-5, dl, CTP))
    b = float(pd.heelis_potential_kV(lat_b - 1e-5, dl, CTP))
    assert abs(a - b) < 1e-2
    assert float(pd.heelis_potential_kV(np.deg2rad(29.9), dl, CTP)) == 0.0
    assert float(pd.heelis_potential_kV(np.deg2rad(-29.9), dl, CTP)) == 0.0


def test_polar_cap_flow_is_anti_sunward():
    """phifn2 = -phifun makes the polar-cap potential linear across the cap (phifun (1.5x - 0.5x^3)), i.e. a
    uniform dawn-to-dusk E and anti-sunward flow over the pole -- the earlier smoothstep gave ~zero there."""
    t = datetime(2024, 12, 21, 12, 0, 0)
    ax = pd.dipole_axis_ecef(t)
    r = 1.05 * 6371e3 * ax
    v = pd.HeelisDrift(ctpoten_kV=CTP).at(r[None, :], t)["v_d"][0]
    lat, lon, _ = pd.subsolar_point_deg(t)
    s = latlon_to_xyz_m(lat, lon, 0.0)
    s = s - np.dot(s, ax) * ax
    s /= np.linalg.norm(s)
    assert np.dot(v, s) / np.linalg.norm(v) < -0.99
    assert 100.0 < np.linalg.norm(v) < 1000.0


def test_heelis_drift_is_perpendicular_to_B():
    t = datetime(2024, 6, 21, 6, 0, 0)
    r = latlon_to_xyz_m(70.0, -60.0, 300e3)
    out = pd.HeelisDrift(ctpoten_kV=CTP).at(r[None, :], t)
    v = out["v_d"][0]
    B = pd.centred_dipole_B(r[None, :], t)[0]
    assert np.linalg.norm(v) > 1.0
    assert abs(np.dot(v, B)) / (np.linalg.norm(v) * np.linalg.norm(B)) < 1e-6
    assert abs(out["sigma_v"][0] - 0.2 * np.linalg.norm(v)) < 1e-9


def test_richmond_crude_pattern_is_eastward_at_night_and_off_at_high_latitude():
    t = datetime(2024, 6, 21, 0, 0, 0)            # UT 0: local midnight near 0 E
    m = pd.RichmondDrift()
    v = m.at(latlon_to_xyz_m(10.0, 0.0, 300e3)[None, :], t)["v_d"][0]
    assert 90.0 < np.linalg.norm(v) < 160.0      # HiCIRF paragraph 11: 100-150 m/s
    east = np.cross(pd.dipole_axis_ecef(t), latlon_to_xyz_m(10.0, 0.0, 300e3))
    assert np.dot(v, east) > 0.0
    assert np.linalg.norm(m.at(latlon_to_xyz_m(75.0, 0.0, 300e3)[None, :], t)["v_d"][0]) == 0.0


def test_combined_model_is_the_superposition():
    combo = pd.WBMODStyleDrift()
    t = datetime(2024, 6, 21, 0, 0, 0)
    for lat in (10.0, 50.0, 75.0):
        r = latlon_to_xyz_m(lat, 0.0, 300e3)[None, :]
        v = combo.at(r, t)["v_d"][0]
        assert np.allclose(v, combo.heelis.at(r, t)["v_d"][0] + combo.richmond.at(r, t)["v_d"][0])
    low = latlon_to_xyz_m(10.0, 0.0, 300e3)[None, :]
    assert np.linalg.norm(combo.heelis.at(low, t)["v_d"][0]) == 0.0   # Heelis is zero within 30 deg mlat


def test_link_leg_channel_runs_with_the_combined_drift_model():
    link = api.Link(68.0, 20.0, frequency_mhz=8.0, dt=datetime(2024, 6, 21, 6, 0, 0), mode="O",
                    max_path_km=1200.0, max_nhops=1, n_ang=4, dang_deg=2.0, igrf_n_ang=3, igrf_n_r=3,
                    verbose=False, drift_model=pd.WBMODStyleDrift())
    cp = link.leg_channel(24.0)
    assert np.isfinite(cp.sigma_phi2) and cp.sigma_phi2 > 0.0
    assert np.isfinite(cp.tau_o) and cp.tau_o > 0.0
