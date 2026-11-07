"""
HiCIRF plan, Stage 5 tests: the MIF radiant table (Fentzke, Janches & Sparks 2009, Table 1, transcribed),
the ecliptic/equatorial/topocentric geometry, the underdense-trail radar backscatter (eqs. 48-50 of the
arXiv:2209.07690 reading), and the end-to-end realisation/component assembly.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from echoes import meteor as mt
from core.geometry import latlon_to_xyz_m, xyz_m_to_latlon


def test_radiant_table_has_five_sources_each_a_valid_mixture():
    assert set(mt.RADIANT_SOURCES) == {"apex", "helion", "anti_helion", "north_toroidal", "south_toroidal"}
    for name, src in mt.RADIANT_SOURCES.items():
        assert abs(sum(src["A"]) - 1.0) < 1e-9, name
        assert src["s_lam"] > 0.0 and src["s_beta"] > 0.0


def test_ecliptic_to_equatorial_matches_the_known_obliquity_point():
    # At beta=0, lam=90 deg (a point on the ecliptic 90 deg from the equinox), dec must equal the
    # obliquity of the date (~23.4 deg) and RA must be 90 deg, by the standard rotation formula.
    t = datetime(2024, 6, 21, 0, 0, 0)
    ra, dec = mt.ecliptic_to_equatorial_deg(90.0, 0.0, t)
    eps_deg = np.rad2deg(mt._obliquity_rad(t))
    assert abs(dec - eps_deg) < 1e-6
    assert abs(ra - 90.0) < 1e-6
    # At the equinox point itself (lam=0, beta=0), RA=Dec=0 regardless of obliquity.
    ra0, dec0 = mt.ecliptic_to_equatorial_deg(0.0, 0.0, t)
    assert abs(ra0) < 1e-9 and abs(dec0) < 1e-9


def test_topocentric_elevation_is_90_at_the_zenith_point():
    t = datetime(2024, 3, 20, 12, 0, 0)
    lst = mt._gmst_deg(t)   # at lon=0, LST = GMST
    # a source exactly at RA=LST, Dec=lat is at the zenith (H=0, matching sin(el)=1)
    el = mt.topocentric_elevation_deg(lst, 45.0, 45.0, 0.0, t)
    assert abs(float(np.atleast_1d(el)[0]) - 90.0) < 1e-6


def test_underdense_rcs_is_positive_and_scales_as_q_squared():
    lam0 = 37.5   # 8 MHz
    r0 = mt.underdense_trail_rcs(1e12, lam0)
    r1 = mt.underdense_trail_rcs(2e12, lam0)
    assert r0 > 0.0 and np.isfinite(r0)
    assert abs(r1 / r0 - 4.0) < 1e-9   # q_e^2 scaling, eq. (50)


def test_diffusion_decay_is_one_at_formation_and_falls_with_time():
    lam0 = 37.5
    assert abs(mt.diffusion_decay_factor(0.0, 2.0, lam0) - 1.0) < 1e-12
    d1 = mt.diffusion_decay_factor(0.05, 2.0, lam0)
    d2 = mt.diffusion_decay_factor(0.5, 2.0, lam0)
    assert 0.0 < d2 < d1 < 1.0


def test_ambipolar_diffusion_increases_with_altitude():
    assert mt.ambipolar_diffusion_coefficient(110e3) > mt.ambipolar_diffusion_coefficient(90e3) > 0.0


def test_sample_meteors_gives_visible_realisations_within_the_scene():
    rng = np.random.default_rng(7)
    reals = mt.sample_meteors(20, 65.0, 20.0, 300.0, datetime(2024, 6, 21, 3, 0, 0), rng)
    assert len(reals) == 20
    for m in reals:
        lat, lon, alt = xyz_m_to_latlon(np.asarray(m.r_ecef))
        assert 79e3 < m.alt_m < 121e3
        assert abs(float(lat) - 65.0) < 3.5 and m.speed_kms > 0.0
        assert abs(np.linalg.norm(m.v_dir_ecef) - 1.0) < 1e-6
        assert m.source in mt.RADIANT_SOURCES


def test_meteor_components_are_finite_and_physically_sane():
    rng = np.random.default_rng(11)
    reals = mt.sample_meteors(15, 65.0, 20.0, 300.0, datetime(2024, 6, 21, 3, 0, 0), rng)
    r_tx = latlon_to_xyz_m(64.5, 19.0, 0.0)
    r_rx = latlon_to_xyz_m(65.5, 21.0, 0.0)
    comps = mt.meteor_components(reals, r_tx, r_rx, f0=30e6, Pt_dBW=60.0, Gt_dB=10.0, Gr_dB=10.0)
    assert len(comps) == len(reals)
    for c in comps:
        assert c.kind == "meteor"
        assert c.power > 0.0 and np.isfinite(c.power)
        assert c.delay > 0.0 and np.isfinite(c.delay)
        assert np.isfinite(c.doppler)
        assert "source" in c.scatterer and "q_e_m1" in c.scatterer


def test_source_weights_are_fentzke_janches_2008_and_sum_to_one():
    assert abs(sum(mt.SOURCE_WEIGHT.values()) - 1.0) < 1e-12
    assert mt.SOURCE_WEIGHT["apex"] == 0.33 and mt.SOURCE_WEIGHT["helion"] == 0.22
    assert mt.SOURCE_WEIGHT["north_toroidal"] == mt.SOURCE_WEIGHT["south_toroidal"] == 0.115


def test_ambipolar_diffusion_is_the_user_profile():
    # D_a(h) = 0.025 exp(0.154 h/km) cm^2/s
    for h_km in (80.0, 90.0, 100.0, 110.0):
        expect = 0.025 * np.exp(0.154 * h_km) * 1e-4
        assert abs(mt.ambipolar_diffusion_coefficient(h_km * 1e3) / expect - 1.0) < 1e-12
    assert 2.0 < mt.ambipolar_diffusion_coefficient(90e3) < 3.0    # ~2.6 m^2/s
