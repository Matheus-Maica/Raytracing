"""
HiCIRF plan, Stage 5 completion: meteor.meteor_components_raytraced() against the same proven
target-homing geometry test_targets.py already uses (so convergence isn't left to chance), just with the
target altitude raised from the ground into the meteor range (80-120 km) instead of a fixed aircraft
height, to confirm the real ray-traced path (not meteor_components()'s straight-line one) actually works
end to end: real homed legs, real absorption/focusing/ionospheric Doppler, a real gamma_or.
"""

from __future__ import annotations

import numpy as np

from echoes import meteor as mt
from raytracing.main import Ray
from test_records_ground import _layer

W0 = 2.0 * np.pi * 8e6
B_EAST = np.array([0.0, 2e-5, 0.0])
TX = dict(lat=20.0, lon=0.0, alt=0.0)
RX = dict(lat=18.0, lon=0.0, alt=0.0)
TARGET_LATLON = (26.0, 0.0)   # at 100 km altitude (meteor range), NOT test_targets.py's ground point (30 N):
                              # a ray landing at 30 N, 0 km crosses 100 km altitude further back along its
                              # descent, confirmed by direct search to converge from both sites at 26 N
ELEV = np.arange(5.0, 60.01, 1.0)
AZ = np.linspace(-1.0, 1.0, 5)


def _ray():
    def B_field(x, y, z):
        return np.broadcast_to(B_EAST, np.shape(x) + (3,)).copy()
    return Ray(W0, +1, _layer(), B_field=B_field)


def _one_meteor(alt_m: float) -> mt.MeteorRealization:
    from core.geometry import latlon_to_xyz_m
    r_ecef = latlon_to_xyz_m(TARGET_LATLON[0], TARGET_LATLON[1], alt_m)
    v_dir = np.array([0.0, 0.0, -1.0])   # an arbitrary unit direction, straight down, for the test
    return mt.MeteorRealization(r_ecef=r_ecef, v_dir_ecef=v_dir, speed_kms=40.0, source="apex",
                               q_e_m1=1e12, r0_m=mt.PLACEHOLDER_R0_M, alt_m=alt_m)


def test_meteor_components_raytraced_converges_and_is_physically_sane():
    ray = _ray()
    reals = [_one_meteor(100e3)]
    comps = mt.meteor_components_raytraced(reals, ray, TX, RX, f0=8e6, Pt_dBW=60.0, Gt_dB=10.0, Gr_dB=10.0,
                                           elev_grid=ELEV, az_grid=AZ, max_path_m=2500e3)
    assert comps, "expected at least one converged (Tx, Rx) leg pair to this well-tested target geometry"
    for c in comps:
        assert c.kind == "meteor"
        assert c.out is not None and c.ret is not None    # real homed legs, unlike meteor_components()
        assert c.out.n_hops == 1 and c.ret.n_hops == 1
        assert c.power > 0.0 and np.isfinite(c.power)
        assert c.delay > 0.0 and np.isfinite(c.delay)
        # c.doppler = out.doppler + ret.doppler + f_T (twoway.target_component): the two legs' own
        # ionospheric Doppler is NaN on this single-epoch synthetic layer (no Doppler accumulator), the
        # same convention test_legs.py's test_leg_budget_on_the_synthetic_layer documents -- not a bug.
        assert np.isnan(c.doppler)
        assert np.isfinite(c.scatterer["f_T"])   # the target-motion term alone is still well defined
        assert c.scatterer["alt_m"] == 100e3 and c.scatterer["source"] == "apex"
        # the homed legs' own real absorption/ground-loss must have been applied (not a geometric guess)
        assert np.isfinite(c.out.absorption_dB) and c.out.absorption_dB >= 0.0


def test_meteors_in_different_altitude_bins_each_get_their_own_tracer():
    ray = _ray()
    reals = [_one_meteor(85e3), _one_meteor(115e3)]
    comps = mt.meteor_components_raytraced(reals, ray, TX, RX, f0=8e6, Pt_dBW=60.0,
                                           elev_grid=ELEV, az_grid=AZ, max_path_m=2500e3, alt_bin_m=5e3)
    alts = {c.scatterer["alt_m"] for c in comps}
    assert alts == {85e3, 115e3}, alts   # both altitude bins must have produced real components


def test_an_unreachable_meteor_is_silently_dropped_not_raised():
    ray = _ray()
    # Far outside both sites' fan coverage and the elevation grid's reach: should converge for neither
    # site, and the function must simply omit it rather than raise.
    from core.geometry import latlon_to_xyz_m
    far = mt.MeteorRealization(r_ecef=latlon_to_xyz_m(85.0, 170.0, 100e3), v_dir_ecef=np.array([0., 0., -1.]),
                               speed_kms=40.0, source="apex", q_e_m1=1e12, r0_m=mt.PLACEHOLDER_R0_M, alt_m=100e3)
    comps = mt.meteor_components_raytraced([far], ray, TX, RX, f0=8e6, Pt_dBW=60.0,
                                           elev_grid=ELEV, az_grid=AZ, max_path_m=2500e3)
    assert comps == []
