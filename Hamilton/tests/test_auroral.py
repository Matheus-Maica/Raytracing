"""
Phase 6a tests: direct auroral backscatter (spec §12.5): the Bragg vector, the aspect angle and law, the
volume-element power through the radar equation, and the quasi-monostatic Doppler of a volume element.
"""

from __future__ import annotations

import numpy as np

from echoes import auroral
from core.constants import C
from echoes.twoway import radar_power_dBW


def test_bragg_vector_and_aspect_angle():
    k0 = 2.0 * np.pi * 8e6 / C
    u = np.array([0.0, 0.6, 0.8])
    q = auroral.bragg_vector_backscatter(1.0, k0, u)
    assert abs(np.linalg.norm(q) - 2.0 * k0) < 1e-12 and np.allclose(q / np.linalg.norm(q), -u)
    B = np.array([0.0, 0.0, 1.0])
    perp = np.array([1.0, 0.0, 0.0])                       # Bragg vector perpendicular to B: perfect aspect
    assert auroral.aspect_angle_rad(perp, B) == 0.0
    assert abs(auroral.aspect_angle_rad(B, B) - np.pi / 2) < 1e-12
    theta = np.radians(30.0)                               # 30 deg from B: psi = 90 - 30 = 60 deg
    assert abs(np.degrees(auroral.aspect_angle_rad(np.array([np.sin(theta), 0.0, np.cos(theta)]), B)) - 60.0) < 1e-9


def test_aspect_law_drops_by_a_db_per_degree():
    assert auroral.eta_aspect(0.0, 1e-12, 1.0) == 1e-12
    assert abs(auroral.eta_aspect(np.radians(10.0), 1e-12, 1.0) - 1e-13) < 1e-25
    assert auroral.eta_aspect(np.radians(3.0), 1e-12, 0.0) == 1e-12


def test_volume_power_is_the_radar_equation_with_sigma_eta_dv():
    dV = auroral.volume_element(2.0e6, 0.01, 0.02, 5e3)
    assert abs(dV - 2.0e6 * 0.01 * 0.02 * 5e3) < 1e-6
    p = auroral.volume_power_dBW(Pt_dBW=60.0, Gt_dB=10.0, Gr_dB=5.0, f0=10e6, eta=1e-12, dV=dV,
                                 d_T=1.2e6, d_R=1.1e6, loss_T_dB=20.0, loss_R_dB=18.0)
    ref = radar_power_dBW(60.0, 10.0, 5.0, 10e6, 1e-12 * dV, 1.2e6, 1.1e6, 20.0, 18.0)
    assert abs(p - ref) < 1e-12
    p2 = auroral.volume_power_dBW(Pt_dBW=60.0, Gt_dB=10.0, Gr_dB=5.0, f0=10e6, eta=1e-12, dV=2 * dV,
                                  d_T=1.2e6, d_R=1.1e6, loss_T_dB=20.0, loss_R_dB=18.0)
    assert abs((p2 - p) - 10.0 * np.log10(2.0)) < 1e-9


def test_volume_doppler_adds_the_bragg_term_and_twice_the_ionospheric_doppler():
    k0 = 2.0 * np.pi * 10e6 / C
    q = auroral.bragg_vector_backscatter(1.0, k0, np.array([1.0, 0.0, 0.0]))
    v = np.array([0.0, 0.0, 200.0])                        # no component along q: the Bragg term vanishes
    assert abs(auroral.auroral_doppler_hz(q, v, 0.4) - 0.8) < 1e-12
    v2 = np.array([-200.0, 0.0, 0.0])
    expect = np.dot(q, v2) / (2.0 * np.pi) + 0.8
    assert abs(auroral.auroral_doppler_hz(q, v2, 0.4) - expect) < 1e-9
