"""
HiCIRF plan, Stage 0 tests: eq. (5)'s fractional-density-variance formula, the placeholder WBMOD-shaped
IrregularityModel/DriftModel (environment.py Protocols), and that api.Link actually uses them end to end
when given, while still falling back to the old scalar/uniform-drift behaviour when it is not.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

import api
from geophysics import wbmod
from core.inputs import IrregularityInputs


def _link(**kwargs):
    return api.Link(57.0, 10.0, frequency_mhz=8.0, dt=datetime(2024, 6, 21, 14, 0, 0), mode="O",
                    max_path_km=1500.0, max_nhops=1, n_ang=4, dang_deg=2.0, igrf_n_ang=3, igrf_n_r=3,
                    verbose=False, **kwargs)


def test_xi_squared_eq5_is_positive_and_scales_linearly_with_CkL():
    h_m = np.linspace(80e3, 1000e3, 400)
    ne = 1e11 * np.exp(-0.5 * ((h_m - 300e3) / 60e3) ** 2) + 1e9  # a simple F-layer-like profile, never zero
    xi2_a = wbmod.xi_squared_eq5(q=2.5, CkL=1e-16, ne_profile_m3=ne, h_m=h_m)
    xi2_b = wbmod.xi_squared_eq5(q=2.5, CkL=2e-16, ne_profile_m3=ne, h_m=h_m)
    assert xi2_a > 0.0 and np.isfinite(xi2_a)
    assert abs(xi2_b / xi2_a - 2.0) < 1e-9          # eq. (5) is linear in CkL
    xi2_arr = wbmod.xi_squared_eq5(q=np.array([2.0, 2.5, 3.0]), CkL=1e-16, ne_profile_m3=ne, h_m=h_m)
    assert xi2_arr.shape == (3,) and np.all(np.isfinite(xi2_arr)) and np.all(xi2_arr > 0.0)


def test_placeholder_irregularity_fixes_L0_at_eq5_outer_scale():
    model = wbmod.PlaceholderWBMODIrregularity()
    out = model.at(np.zeros((5, 3)), None)
    assert np.all(out["L0"] == wbmod.WBMOD_LO_M) and abs(wbmod.WBMOD_LO_M - 30e3) < 1e-6
    assert out["xi"].shape == (5,) and np.all(out["xi"] == model.xi)
    assert np.all(out["a"] == model.axial_ratio_B) and np.all(out["b"] == model.axial_ratio_sheet)


def test_placeholder_drift_uses_each_points_own_local_frame():
    # Two ECEF points far apart in longitude at the same latitude: a fixed (east, north) pair must map to
    # different ECEF directions at each, unlike a single site-wide frame (the bug this replaces).
    from core.geometry import latlon_to_xyz_m
    r1 = latlon_to_xyz_m(45.0, 0.0, 300e3)
    r2 = latlon_to_xyz_m(45.0, 90.0, 300e3)
    model = wbmod.PlaceholderWBMODDrift(v_east_north_mps=(150.0, 0.0))
    out = model.at(np.stack([r1, r2]), None)
    v1, v2 = out["v_d"][0], out["v_d"][1]
    assert abs(np.linalg.norm(v1) - 150.0) < 1e-6 and abs(np.linalg.norm(v2) - 150.0) < 1e-6
    assert np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) < 0.5   # not parallel: different frames
    assert out["sigma_v"][0] == model.sigma_v_frac * 150.0


def test_link_defaults_now_match_inputs_py_single_source():
    # api.Link's own defaults (no override) must equal inputs.IrregularityInputs()'s fields -- the bug this
    # fixes: three previously-independent, silently-differing placeholder sets (api.py, inputs.py,
    # leg_screens.py).
    link = _link()
    d = IrregularityInputs()
    assert link.xi_rms == d.xi and link.L0_km == d.L0_m / 1e3 and link.p_index == d.p
    assert link.axial_ratio == d.axial_ratio_B == d.axial_ratio_sheet
    assert abs(link.L0_km - 30.0) < 1e-9   # HiCIRF eq. (5)'s fixed outer scale


def test_leg_channel_without_models_keeps_the_old_scalar_behaviour():
    link = _link()
    cp = link.leg_channel(28.0)
    assert np.isfinite(cp.sigma_phi2) and cp.sigma_phi2 > 0.0
    assert np.isfinite(cp.tau_o) and cp.tau_o > 0.0


def test_leg_channel_uses_irregularity_and_drift_models_when_given():
    link = _link(irregularity_model=wbmod.PlaceholderWBMODIrregularity(),
                drift_model=wbmod.PlaceholderWBMODDrift())
    cp = link.leg_channel(28.0)
    assert np.isfinite(cp.sigma_phi2) and cp.sigma_phi2 > 0.0
    assert np.isfinite(cp.tau_o) and cp.tau_o > 0.0 and np.isfinite(cp.S4)

    ray = link.trace(28.0, phase_variance=True)
    assert ray.sigma_phi2 is not None and np.isfinite(ray.sigma_phi2) and ray.sigma_phi2 > 0.0


def test_p_from_q_is_q_plus_two_and_clipped():
    assert abs(float(wbmod.p_from_q(2.0)) - 4.0) < 1e-12
    assert float(wbmod.p_from_q(0.5)) == 3.05 and float(wbmod.p_from_q(5.0)) == 5.95


def test_crude_wbmod_returns_numbers_in_every_regime():
    """Our own crude WBMOD (user decision): only needs to return sane numbers. Quiet mid-latitude < auroral
    oval, equatorial pre-midnight > equatorial noon; everything finite and inside the clip range."""
    from datetime import datetime
    from core.geometry import latlon_to_xyz_m
    m = wbmod.CrudeWBMOD()
    t = datetime(2024, 12, 21, 0, 0, 0)
    pts = np.array([latlon_to_xyz_m(lat, lon, 300e3) for lat, lon in
                    [(0.0, 0.0), (0.0, 180.0), (40.0, 0.0), (65.0, -100.0), (85.0, -70.0)]])
    out = m.at(pts, t)
    for key in ("xi", "L0", "p", "a", "b"):
        assert out[key].shape == (5,) and np.all(np.isfinite(out[key]))
    assert np.all((out["xi"] >= 1e-4) & (out["xi"] <= 0.3))
    assert out["xi"][0] > out["xi"][1]        # equatorial: ~midnight MLT > ~noon MLT
    assert out["xi"][3] > out["xi"][2]        # auroral oval > quiet mid-latitude
    assert np.all(out["L0"] == wbmod.WBMOD_LO_M) and np.all(out["p"] == 4.0)


def test_crude_wbmod_uses_the_link_ionosphere_and_drives_leg_channel():
    m = wbmod.CrudeWBMOD()
    link = _link(irregularity_model=m)
    m.ne_column = link.electron_density_xyz_m      # integrate Ne^2 over the Link's own ionosphere (eq. 5)
    cp = link.leg_channel(28.0)
    assert np.isfinite(cp.sigma_phi2) and cp.sigma_phi2 > 0.0 and np.isfinite(cp.S4)
