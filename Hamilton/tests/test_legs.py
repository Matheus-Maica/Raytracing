"""
Phase 1f/1j tests: flux-tube distance (spec §4.5, eq. 4.7) against the analytic flat-ground
answer, the caustic cap, and a smoke test of the landing mini-fan on the synthetic layer.
"""

from __future__ import annotations

import numpy as np

from propagation import legs
from raytracing.batch import BatchTracer
from core.geometry import WGS84_A, latlon_to_xyz_m, xyz_m_to_latlon  # noqa: F401
from raytracing.main import Ray

F0 = 8.0e6
W0 = 2.0 * np.pi * F0


def test_flux_distance_matches_slant_range_on_flat_ground():
    """Spec §4.5 unit test: with the ionosphere off, a source at altitude H aimed at depression
    psi lands at slant range H/sin(psi); eq. (4.7) must return that to 1e-4."""
    H = 2000.0
    for psi_deg in (10.0, 25.0, 60.0):
        psi = np.radians(psi_deg)
        beta = -psi                                    # launch elevation (downwards)
        alpha = np.radians(30.0)

        def landing(b, a):
            rho = H / np.tan(-b)                       # horizontal distance to the plane z = 0
            return np.array([rho * np.cos(a), rho * np.sin(a), 0.0])

        h = 1e-7
        J_beta = (landing(beta + h, alpha) - landing(beta - h, alpha)) / (2 * h)
        J_alpha = (landing(beta, alpha + h) - landing(beta, alpha - h)) / (2 * h)
        d2, A = legs.flux_tube_distance2(J_beta, J_alpha, psi, beta)
        assert abs(np.sqrt(d2) / (H / np.sin(psi)) - 1.0) < 1e-4, (psi_deg, np.sqrt(d2), H / np.sin(psi))


def test_caustic_cap_limits_focusing():
    P = 1000e3
    G, d2_used, caustic = legs.focusing((P / 1e3) ** 2 * 1e-3, P)      # very focused
    assert caustic and abs(G - 10.0) < 1e-12
    assert abs(np.sqrt(d2_used) / (P * 10 ** (-10.0 / 20.0)) - 1.0) < 1e-12
    G2, d2b, c2 = legs.focusing(P * P, P)                               # no focusing: G = 0
    assert not c2 and abs(G2) < 1e-12


def test_mini_fan_landing_on_layer_is_sane():
    """Smoke test on the synthetic layer: the mini-fan lands, the Jacobians are finite and
    non-zero, and the focusing gain stays inside a sensible range."""
    from test_records_ground import _layer, _launch
    ray = Ray(W0, +1, _layer(peak=1.2e12))
    tracer = BatchTracer(ray, ds_m=4000.0)
    r0, u0, up, east = _launch(lat=57.0, lon=10.0, elev=28.0, az=0.0)
    north = np.cross(up, east)
    res = legs.landing_mini_fan(tracer, r0, 28.0, 0.0, up, north, east, 2500e3, nhops=1)
    assert res.landed
    assert np.all(np.isfinite(res.J_beta)) and np.linalg.norm(res.J_beta) > 1e3
    assert np.linalg.norm(res.J_alpha) > 1e3
    assert res.G_focus_db is not None and -60.0 < res.G_focus_db <= 10.0
    assert 0.0 < res.psi < np.pi / 2


def test_one_way_power_matches_friis_free_space():
    """Eq. (6.2)-(6.3) one-way with no losses and unit gains reduces to the Friis form."""
    from core.constants import C
    f0, d = 8.0e6, 1.0e6
    lam = C / f0
    expect = 10.0 * np.log10((lam / (4.0 * np.pi * d)) ** 2)
    assert abs(legs.one_way_power_dB(0.0, 0.0, 0.0, f0, d, 0.0) - expect) < 1e-9


def test_budget_has_polarization_loss_and_no_coherence_term():
    import dataclasses
    from propagation import link_budget
    names = {f.name for f in dataclasses.fields(link_budget.LinkBudgetResult)}
    assert not any("coher" in n or "phi" in n for n in names), names
    assert legs.L_POL_DB == 3.01


def test_leg_budget_on_the_synthetic_layer():
    """A landed one-hop leg: label from §5.4, delay P'/c, loss identity (A + G + L_pol), and a finite
    Doppler only when the tracer carries a Doppler accumulator (here it does not, so NaN)."""
    import re
    from propagation import link_budget
    from core.constants import C
    from test_records_ground import _layer, _launch
    ray = Ray(W0, +1, _layer(peak=1.2e12))
    tracer = BatchTracer(ray, ds_m=4000.0)
    r0, _, up, east = _launch(lat=57.0, lon=10.0, elev=28.0, az=0.0)
    north = np.cross(up, east)
    bud = link_budget.one_way_budget(tracer, r0, up, north, east, 28.0, 0.0, 2500e3, nhops=1,
                                     Pt_dBW=40.0, Gt_dB=10.0, Gr_dB=5.0)
    assert bud.landed and bud.leg is not None
    assert re.fullmatch(r"1F-(low|high)-O", bud.leg.label), bud.leg.label
    assert abs(bud.delay_s - bud.leg.Pg / C) < 1e-12 and bud.delay_s > 0.0
    assert abs(bud.total_loss_db - (bud.absorption_db + bud.ground_loss_db + 3.01)) < 1e-9
    assert np.isnan(bud.doppler_hz)
    assert bud.received_dBW is not None and bud.received_dBW < 40.0 + 10.0 + 5.0
