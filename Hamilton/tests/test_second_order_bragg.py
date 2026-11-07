"""
Second-order sea echo (second_order_bragg.py) against Barrick (1972), "Remote sensing of sea state by radar",
ch. 12 of Remote Sensing of the Troposphere (docs/1972Barrick_RST.pdf): his first-order level (12:23), the total
second-order sigma0 printed on his Figs. 12.18 (upwind) and 12.19 (crosswind) at 10 MHz for 15-40 kt winds
with his semi-isotropic Phillips spectrum (12:13), and the sqrt(2) / 2^(3/4) singularities he describes.
"""

from __future__ import annotations

import numpy as np

from echoes import clutter
from echoes import second_order_bragg as sb
from core.constants import C

F0 = 10e6
KT = 0.514444
# Barrick (1972) Figs. 12.18 / 12.19: second-order sigma0 [dB], read off the figure labels.
BARRICK = {"upwind": {15: -32.6, 20: -29.0, 30: -24.8, 40: -20.6},
           "crosswind": {15: -29.8, 20: -27.0, 30: -23.7, 40: -20.4}}
WAVE_DIR = {"upwind": np.pi, "crosswind": np.pi / 2}   # radar looks along +x; upwind: waves come towards it


def test_first_order_level_is_barricks_minus_17_dB():
    S = sb.barrick_phillips_semi_isotropic(0.005, 20 * KT, np.pi)
    approaching, receding = sb.first_order_sigma0(F0, S)
    assert abs(10 * np.log10(approaching) + 17.0) < 0.05     # (12:23): sigma0 = 0.02 = -17 dB
    assert receding == 0.0                                  # semi-isotropic: no waves moving away


def test_second_order_sigma0_matches_barricks_figures_12_18_and_12_19():
    """Within 2 dB in all eight cases (measured: 0.04-1.5 dB; the worst is crosswind at 15 kt, where the
    sharp edge of the semi-isotropic half-plane lies exactly along the Bragg direction), and increasing with
    wind speed like his."""
    for case, table in BARRICK.items():
        got = []
        for w, ref in table.items():
            S = sb.barrick_phillips_semi_isotropic(0.005, w * KT, WAVE_DIR[case])
            s0, _, _ = sb.second_order(F0, S, dp_k0=0.005)
            got.append(10 * np.log10(s0))
            assert abs(got[-1] - ref) < 2.0, (case, w, got[-1], ref)
        assert np.all(np.diff(got) > 0.0), (case, got)


def test_binned_spectrum_integrates_to_sigma0_and_has_the_sqrt2_and_2_to_3_4_singularities():
    k0 = 2 * np.pi * F0 / C
    wB = np.sqrt(2 * 9.80665 * k0)
    S = sb.barrick_phillips_semi_isotropic(0.005, 15 * KT, np.pi)
    edges = np.linspace(1.25 * wB, 1.85 * wB, 241)
    _, c, sig = sb.second_order(F0, S, eta_edges=edges, dp_k0=0.0015, p_max_k0=3.0)
    x = c / wB
    peaks = [x[i] for i in range(2, len(sig) - 2) if sig[i] == max(sig[i - 2:i + 3]) and sig[i] > 1.5 * np.median(sig)]
    assert any(abs(p - np.sqrt(2.0)) < 0.004 for p in peaks), peaks
    assert any(abs(p - 2.0 ** 0.75) < 0.004 for p in peaks), peaks

    S40 = sb.barrick_phillips_semi_isotropic(0.005, 40 * KT, np.pi)
    edges = np.linspace(-2.5 * wB, 2.5 * wB, 501)
    s0, _, sig = sb.second_order(F0, S40, eta_edges=edges, dp_k0=0.005)
    assert abs(np.sum(sig * np.diff(edges)) / s0 - 1.0) < 1e-3


def test_sea_impedance_uses_barricks_time_convention():
    d = sb.sea_impedance(10e6)
    assert 0.005 < d.real < 0.02 and -0.02 < d.imag < -0.005   # same quadrant as the usual 0.011 - 0.012i


def test_bragg_lines_second_order_adds_a_weaker_continuum():
    phi_rel = 0.0
    first = clutter.bragg_lines(F0, phi_rel)
    both = clutter.bragg_lines(F0, phi_rel, second_order=True, wind_speed_mps=15.0, U_radial=0.5)
    assert both[:2] == clutter.bragg_lines(F0, phi_rel, U_radial=0.5)[:2]   # first-order lines unchanged
    extra = both[2:]
    assert len(extra) > 10
    fB = clutter.bragg_frequency_hz(F0)
    shift = clutter.current_doppler_shift_hz(0.5, F0)
    assert all(abs(fd - shift) <= 2.5 * fB + 1e-9 for fd, _ in extra)
    s2 = sum(sv for _, sv in extra)
    s1 = sum(sv for _, sv in first)
    assert 0.0 < s2 < s1
    ice = clutter.bragg_lines(F0, phi_rel, second_order=True, wind_speed_mps=15.0, U_radial=0.5, c_ice=0.5)
    assert abs(sum(sv for _, sv in ice[2:]) / s2 - 0.5) < 1e-9           # open-water fraction applies too


def test_scene_components_with_second_order_add_weak_components_around_the_bragg_lines():
    from echoes import clutter_scene as cs
    from core.inputs import RadarInputs, SurfaceInputs
    from test_ccmp_wind import _fans
    fan_tx, r_tx, fan_rx, r_rx = _fans()
    kw = dict(f0=8e6, surface=SurfaceInputs(wind_direction_deg=0.0, wind_speed_mps=12.0), radar=RadarInputs(),
              cell_m=10e3, surface_of=lambda lat, lon: "sea")
    first, _ = cs.scene_components(fan_tx, fan_rx, r_tx, r_rx, **kw)
    both, _ = cs.scene_components(fan_tx, fan_rx, r_tx, r_rx, second_order=True, **kw)
    assert len(both) > len(first)
    p1 = sum(c.power_W for c in first)
    p2 = sum(c.power_W for c in both) - p1
    assert 0.0 < p2 < p1
