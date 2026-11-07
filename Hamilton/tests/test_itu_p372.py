"""
HiCIRF plan, Stage 6 completion: itu_p372.atmospheric_noise_dB() against reference values from the real
ITU-R Study Group 3 C implementation (P372/Src/P372/Noise.c of github.com/programmin1/ITU-R-HF), compiled
directly with gcc -fcommon and run for these exact (lat, lon, hour, frequency, month) cases -- not a
structural/sanity check, a bit-for-bit (to float-print precision) reproduction check.
"""

from __future__ import annotations

from receiver import itu_p372 as p

# (lat_deg, lon_deg, utc_hour, frequency_mhz, month_index0, (FaA, DuA, DlA) from the compiled C reference)
REFERENCE_CASES = [
    (40.0, -105.0, 12, 10.0, 0, (35.528489, 7.190908, 5.960067)),
    (65.0, 20.0, 3, 8.0, 5, (38.897721, 8.160172, 8.250090)),
    (-20.0, 150.0, 18, 15.0, 0, (29.321304, 5.452639, 4.969312)),
    (0.0, 0.0, 0, 5.0, 0, (57.282947, 7.204835, 5.746907)),
    (40.0, -105.0, 12, 20.0, 0, (13.156122, 4.758147, 3.149705)),
]


def test_atmospheric_noise_matches_the_compiled_c_reference():
    for lat, lon, hour, f, month0, ref in REFERENCE_CASES:
        got = p.atmospheric_noise_dB(lat, lon, hour, f, month0)
        for g, r in zip(got, ref):
            assert abs(g - r) < 1e-4, (lat, lon, hour, f, month0, got, ref)


def test_coefficient_cache_returns_the_same_array_object():
    c1 = p._coeffs(0)
    c2 = p._coeffs(0)
    assert c1 is c2
    assert c1["fakp"].shape == (6, 16, 29)
    assert c1["fakabp"].shape == (6, 2)
    assert c1["dud"].shape == (5, 12, 5)
    assert c1["fam"].shape == (12, 14)


def test_all_twelve_months_parse_without_error():
    for m in range(12):
        c = p._coeffs(m)
        assert c["fakp"].shape == (6, 16, 29)


def test_time_block_interpolation_is_continuous_across_a_boundary():
    # At lon=-105, local time block boundaries fall at UTC hour 3/4, 7/8, 11/12, ... (lrxmt = hour-7).
    # hour=10 (lrxmt=3, slp=0.75, still block 0) and hour=11 (lrxmt=4, slp=0, now block 1) straddle one
    # such boundary -- the two-timeblock blend this function does should keep FaA from jumping sharply.
    a = p.atmospheric_noise_dB(40.0, -105.0, 10, 10.0, 0)
    b = p.atmospheric_noise_dB(40.0, -105.0, 11, 10.0, 0)
    assert abs(a[0] - b[0]) < 3.0
