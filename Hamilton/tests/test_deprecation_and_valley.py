"""
Phase 5 housekeeping tests: the D6 deprecation warnings (the scalar tracer and the Monte Carlo layer
still work, and say so), and the E-F valley taken from a density column (spec §5.4).
"""

from __future__ import annotations

import warnings

import numpy as np

from propagation import legs
from raytracing.main import Ray
from raytracing.raytracer import RayTracer
from test_records_ground import _layer
from raytracing.batch import BatchTracer


def test_scalar_tracer_warns_and_batch_does_not_():
    ray = Ray(2.0 * np.pi * 8e6, +1, _layer())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        RayTracer(ray)
    assert any(issubclass(w.category, DeprecationWarning) and "D6" in str(w.message) for w in rec)
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        BatchTracer(ray)
    assert not any(issubclass(w.category, DeprecationWarning) for w in rec2)


def _two_layer_column():
    h = np.arange(60e3, 400e3, 500.0)
    e_layer = 1e11 * np.exp(-0.5 * ((h - 110e3) / 8e3) ** 2)
    f_layer = 1e12 * np.exp(-0.5 * ((h - 250e3) / 50e3) ** 2)
    return h, e_layer + f_layer


def test_valley_is_the_minimum_between_the_e_and_f_peaks():
    h, ne = _two_layer_column()
    valley = legs.valley_altitude_m(ne, h)
    assert valley is not None and 110e3 < valley < 200e3     # the minimum between the peaks, ~128 km here
    between = (h > 120e3) & (h < 240e3)
    assert abs(valley - h[between][np.argmin(ne[between])]) < 1e3


def test_no_valley_without_an_e_peak_or_without_a_minimum():
    h, ne = _two_layer_column()
    assert legs.valley_altitude_m(np.exp(-0.5 * ((h - 250e3) / 50e3) ** 2), h) is None
    monotone = np.linspace(1.0, 2.0, h.size)
    assert legs.valley_altitude_m(monotone, h) is None
