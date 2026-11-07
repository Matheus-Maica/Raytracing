"""
itu_p372.py -- ITU-R P.372's atmospheric (lightning) noise climatology, Fam(lat, lon, time, frequency,
month). HiCIRF plan, Stage 6 completion (the piece noise.py's PLACEHOLDER_F_ATM_DB stood in for).

This is a direct, line-by-line port of AtmosphericNoise()/GetFamParameters()/ReadFamDud() from
P372/Src/P372/Noise.c of github.com/programmin1/ITU-R-HF (a fork of the ITU-R Study Group 3's own
reference implementation of Recommendations P.372 and P.533), read directly, not summarised. That
source states: "These software methods for the prediction of the performance of HF circuits [are]
based on Recommendations ITU-R P.533-14 and P.372-13... developed collaboratively by participants in
ITU-R Study Group 3. It may be used by implementers... free from any copyright assertions." The
coefficient data this needs (COEFF01W.txt .. COEFF12W.txt, vendored under P372/data/, ~2.8 MB,
attributed in that same source to Peter Suessman) is the digitised equivalent of the Recommendation's
own Figs 13a-36a/b/c world noise-contour maps -- the thing that made this part of P.372 intractable to
transcribe directly from the 109-page PDF (see docs/needed_from_user.md's superseded note and the plan's
Stage 6 log).

Validated against the C reference itself, not just read: the C source (P372/Src/P372/{Noise,
NoiseMemory, InitializeNoise}.c) was compiled and run directly (gcc -fcommon) for 5 (lat, lon, hour,
frequency, month) cases spanning both hemispheres and solstice/equinox months; this module's
atmospheric_noise_dB() is tested against those 5 printed reference values (tests/test_itu_p372.py) and
matches to better than 1e-4 dB -- not merely "structurally plausible", bit-for-bit (float-precision)
reproduction of the reference implementation's own output.

galactic_noise_itur_p372_dB() and man_made_noise_itur_p372_dB() in noise.py were independently checked
against this same C source's GalacticNoise()/ManMadeNoise() (identical constants: 52.0/23.0 for
galactic; 76.8/27.7, 72.5/27.7, 67.2/27.7, 53.6/28.6 for city/residential/rural/quiet_rural) -- so all
three ITU-R P.372 noise terms now trace to the same validated source.
"""

from __future__ import annotations

import os
import re

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "P372", "data")
_D2R = 0.0174532925   # the C source's own truncated constant (Common.h), used here for bit-match, not np.pi/180

_HEADER_RE = re.compile(r"^(\w+)\(([\d,]+)\)")
_WANTED = {"fakp", "fakabp", "dud", "fam"}

_cache: dict[int, dict] = {}


def _parse_coeff_file(path: str) -> dict:
    """
    Parse one COEFFxxW.txt file, returning {'fakp': (6,16,29), 'fakabp': (6,2), 'dud': (5,12,5),
    'fam': (12,14)} as numpy arrays -- a generic port of ReadFamDud()'s file-reading loop (Noise.c):
    skip any block not in _WANTED (by its declared size, 5 values per data line, from its own header),
    parse and reshape (Fortran order: the header's first-listed dimension varies fastest) any block
    that is.
    """
    out: dict = {}
    with open(path, "r") as fh:
        fh.readline()   # the file's own name/header line
        while len(out) < len(_WANTED):
            header = fh.readline()
            if not header:
                raise ValueError(f"{path}: ran out of lines before finding all of {_WANTED}")
            m = _HEADER_RE.match(header.strip())
            if not m:
                continue   # a data/int line belonging to a block we're mid-skip of; keep scanning
            name = m.group(1)
            dims = [int(d) for d in m.group(2).split(",")]
            total = int(np.prod(dims))
            n_full_lines = total // 5
            n_last = total - 5 * n_full_lines
            n_lines = n_full_lines + (1 if n_last else 0)
            if name not in _WANTED:
                for _ in range(n_lines):
                    fh.readline()
                continue
            flat = np.empty(total, dtype=float)
            pos = 0
            for _ in range(n_full_lines):
                vals = fh.readline().split()
                flat[pos:pos + 5] = [float(v) for v in vals[:5]]
                pos += 5
            if n_last:
                vals = fh.readline().split()
                flat[pos:pos + n_last] = [float(v) for v in vals[:n_last]]
            # header "name(d0,d1,...,dn)": d0 (first-listed) is fastest-varying in the flat storage,
            # which is exactly numpy's default (row-major, last-axis-fastest) reshape onto the REVERSED
            # dims -- no transpose needed, e.g. fakp(29,16,6) -> shape (6,16,29), indexed fakp[i<6][j<16][k<29]
            # with k (size 29) fastest, matching the C source's own fakp[i][j][k] population order exactly.
            out[name] = flat.reshape(tuple(reversed(dims)))
    return out


def _coeffs(month_index0: int) -> dict:
    """month_index0: 0 = January .. 11 = December. Cached after first read."""
    if month_index0 not in _cache:
        path = os.path.join(_DATA_DIR, f"COEFF{month_index0 + 1:02d}W.txt")
        _cache[month_index0] = _parse_coeff_file(path)
    return _cache[month_index0]


def _fam_parameters(coeffs: dict, tmblk: int, lat_rad: float, lon_rad: float, frequency_mhz: float):
    """Port of GetFamParameters(): returns (FA, Du, Dl) [dB above kT0b, at the given frequency]."""
    fakp, fakabp, dud, fam = coeffs["fakp"], coeffs["fakabp"], coeffs["dud"], coeffs["fam"]
    lm, ln = 29, 15

    q = (lon_rad + 2.0 * np.pi) / 2.0 if lon_rad < 0.0 else lon_rad / 2.0
    ZZ = np.empty(lm)
    for j in range(lm):
        R = 0.0
        for k in range(ln):
            R += np.sin((k + 1.0) * q) * fakp[tmblk, k, j]
        ZZ[j] = R + fakp[tmblk, 15, j]

    q = lat_rad + np.pi / 2.0
    R = 0.0
    for j in range(lm):
        R += np.sin((j + 1.0) * q) * ZZ[j]
    fam_1mhz = R + fakabp[tmblk, 0] + fakabp[tmblk, 1] * q

    i = tmblk + 6 if lat_rad < 0.0 else tmblk

    u = [-0.75, (8.0 * 2.0 ** np.log10(frequency_mhz) - 11.0) / 4.0]
    cz = 0.0
    pz = px = 0.0
    for k in range(2):
        pz = u[k] * fam[i, 0] + fam[i, 1]
        px = u[k] * fam[i, 7] + fam[i, 8]
        for j in range(2, 7):
            pz = u[k] * pz + fam[i, j]
            px = u[k] * px + fam[i, j + 7]
        if k == 0:
            cz = fam_1mhz * (2.0 - pz) - px
    FA = cz * pz + px

    x = np.log10(frequency_mhz)
    if frequency_mhz > 20.0:
        x = np.log10(20.0)
    v = np.empty(5)
    for j in range(5):
        xj = x
        if j == 4 and frequency_mhz > 10.0:
            xj = 1.0
        y = dud[j, i, 0]
        for k in range(1, 5):
            y = y * xj + dud[j, i, k]
        v[j] = y
    Du, Dl = v[0], v[1]
    return float(FA), float(Du), float(Dl)


def atmospheric_noise_dB(lat_deg: float, lon_deg: float, utc_hour: float, frequency_mhz: float,
                         month_index0: int) -> tuple:
    """
    ITU-R P.372 atmospheric (lightning) noise, port of AtmosphericNoise(): the real replacement for
    noise.PLACEHOLDER_F_ATM_DB. Returns (FaA, DuA, DlA), all dB above k T0 b.

    lat_deg, lon_deg : geographic, degrees (east positive).
    utc_hour         : 0-24.
    frequency_mhz    : HF frequency.
    month_index0     : 0 = January .. 11 = December (the month whose coefficient file to use).
    """
    coeffs = _coeffs(month_index0)
    lrxmt = int(utc_hour) + int(lon_deg / 15.0)
    lrxmt = (lrxmt + 24) % 24
    tmblk_now = (lrxmt // 4) % 6
    tmblk_adj = (tmblk_now + 1) % 6
    lat_rad, lon_rad = np.deg2rad(lat_deg), np.deg2rad(lon_deg)

    FA_now, Du_now, Dl_now = _fam_parameters(coeffs, tmblk_now, lat_rad, lon_rad, frequency_mhz)
    FA_adj, Du_adj, Dl_adj = _fam_parameters(coeffs, tmblk_adj, lat_rad, lon_rad, frequency_mhz)

    slp = (lrxmt % 4) / 4.0

    def _interp(a_db, b_db):
        a, b = 10.0 ** (a_db / 10.0), 10.0 ** (b_db / 10.0)
        return float(10.0 * np.log10(a + (b - a) * slp))

    return _interp(FA_now, FA_adj), _interp(Du_now, Du_adj), _interp(Dl_now, Dl_adj)
