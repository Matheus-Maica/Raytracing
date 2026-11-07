"""
ECHAIM/echaim.py — ctypes binding to the vendored E-CHAIM C library.

E-CHAIM (Empirical Canadian High Arctic Ionospheric Model, Themens et al.
2017/2018) is IRI's high-latitude counterpart: a climatological electron-
density model built on data assimilated specifically from the Canadian
Arctic, valid above ~45 degrees AACGM (Altitude-Adjusted Corrected
Geomagnetic, Shepherd 2014) latitude -- where IRI's mid/low-latitude
parameterization is known to be poor. See ionosphere.py for how this
plugs into Hamilton as a per-point override of IRI, not a replacement.

Why vendored C, not a Python port
----------------------------------
The C sources under lib/ (published by the CHAIN project, University of
New Brunswick) evaluate the model's spherical-cap-harmonic + Fourier
seasonal + solar-cycle basis functions directly from two bundled SQLite
databases:
    etc/COEFS_DB.db  -- the fitted model coefficients (small, static)
    etc/CHAIM_DB.db  -- historical F10.7/AE/Dst/ap/PC/IG index time series
                        (~90 MB; this is what lets a query for an arbitrary
                        PAST date work fully offline, unlike a live feed)
plus etc/AACGM_coeffs/ (AACGM-v2 + IGRF coefficients, for the geomagnetic
coordinate transform E-CHAIM's horizontal structure is expressed in) and
etc/{ECHAIM_FIRI,PRECIP}_DB.db (D-region and auroral-precipitation add-ons).
This is real physics + a real fitted model, published and version-pinned --
reimplementing it from scratch would mean silently drifting from the
published, citable model. ctypes (stdlib, no extra dependency) is enough
to call it directly.

One patch was made to the vendored source: getDir.c's getDir() -- called
internally by every public ECHAIM function -- normally reads a
configECHAIM.dat from the process's CWD, and BLOCKS ON STDIN if that file
is missing. That's fatal for a library called from Python. getDir.c now
exposes echaim_init(dir), which sets the same global directly and makes
getDir() a no-op once called (see getDir.c/.h). This module calls it once,
at import time, pointed at etc/ below.

Not wired up: updateLocalDB() (refreshes CHAIM_DB.db from
chain-project.net). It's the only vendored file needing libcurl, which
isn't installed as a -dev package here, and Hamilton doesn't auto-refresh
IRI's coefficients either (see igrf.py's docstring on the same tradeoff) --
so this runs off the bundled CHAIM_DB.db snapshot. That snapshot's index
coverage ends at whatever date it was downloaded; querying past that (e.g.
a "today" run some months after cloning this repo) still returns a value
-- the C library falls back to the all-time median/forecast per index (see
the E-CHAIM primer's error-flag table) -- just with reduced accuracy for
the storm/precipitation terms specifically. Re-download
etc/CHAIM_DB.db from https://chain-new.chain-project.net/echaim_downloads/DBFiles/CHAIM_DB.db
to refresh it manually.

Validity / NaN behaviour
-------------------------
densityProfile() (called below) returns NaN, per query point, wherever
the AACGM geomagnetic latitude (computed at 350 km, matching E-CHAIM's own
convention) is below 45 degrees -- enforced inside the C library itself
(calcMF2.c), using the real AACGM transform, not a dipole approximation.
Between 45-50 degrees E-CHAIM still returns a value but its own
documentation calls this a transition band meant specifically for
blending with a mid/low-latitude model -- exactly the IRI fallback
ionosphere.py performs. Below 45 degrees: NaN, always. This module does
not re-decide that boundary; ionosphere.py's fallback is driven directly
by whichever entries come back non-finite.

Public API
----------
    density_profile(lat_deg, lon_deg, dt, alt_km,
                     storm=False, precip=True, dregion=False) -> ndarray

    lat_deg, lon_deg : (N,) geographic degrees
    dt                : one datetime, applied to every (lat, lon) row
    alt_km            : (M,) km above the WGS84 ellipsoid, shared by every row
    returns           : (N, M) electron density, m^-3, NaN where invalid

License: E-CHAIM is GPL-3.0 (see licenses/LICENSE.txt); AACGM and the
astronomical-algorithms component carry their own included licenses.
"""

from __future__ import annotations

import ctypes
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIBPATH = _HERE / "libechaim.so"
_ETCDIR = _HERE / "etc"


def _ensure_built() -> None:
    if _LIBPATH.exists():
        return
    print("[echaim] libechaim.so not found, building it (see ECHAIM/build.sh) ...")
    subprocess.run(["bash", str(_HERE / "build.sh")], check=True, cwd=_HERE)


_ensure_built()

_lib = ctypes.CDLL(str(_LIBPATH))
_libc = ctypes.CDLL(None)   # process's own libc, for freeing ECHAIM's malloc'd output

_D_P = ctypes.POINTER(ctypes.c_double)
_DD_P = ctypes.POINTER(_D_P)

_lib.echaim_init.argtypes = [ctypes.c_char_p]
_lib.echaim_init.restype = ctypes.c_int

_lib.densityProfile.argtypes = [
    _D_P, _D_P,                       # lat, lon
    _D_P, _D_P, _D_P,                 # year, month, day
    _D_P, _D_P, _D_P,                 # hour, minute, second
    ctypes.c_int, ctypes.c_int, ctypes.c_int,   # storm, precip, dregion
    ctypes.c_int,                     # l0 (number of lat/lon/time rows)
    _D_P, ctypes.c_int,               # alt, l1 (number of altitudes)
    ctypes.c_int,                     # err (log error-code flags -- unused here)
]
_lib.densityProfile.restype = _DD_P

_libc.free.argtypes = [ctypes.c_void_p]
_libc.free.restype = None

_init_rc = _lib.echaim_init(str(_ETCDIR).encode())
if _init_rc != 0:
    raise RuntimeError(
        f"ECHAIM/echaim.py: echaim_init({_ETCDIR!r}) failed (rc={_init_rc}). "
        "Check that ECHAIM/etc/ contains AACGM_coeffs/, COEFS_DB.db, "
        "CHAIM_DB.db, ECHAIM_FIRI.db, PRECIP_DB.db."
    )


def _as_c_double_array(values) -> ctypes.Array:
    arr = np.ascontiguousarray(values, dtype=np.float64)
    return (ctypes.c_double * arr.size)(*arr)


def density_profile(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    dt: datetime,
    alt_km: np.ndarray,
    storm: bool = False,
    precip: bool = True,
    dregion: bool = False,
) -> np.ndarray:
    """
    E-CHAIM electron density, m^-3, on an (N lat/lon rows) x (M altitudes)
    grid -- one shared altitude vector applied to every (lat, lon) row,
    matching iricore.iri's batched call shape (see ionosphere.py).

    NaN wherever the query point's AACGM magnetic latitude is below 45
    degrees (outside E-CHAIM's validity domain) -- see module docstring.

    storm   : apply the NmF2 storm (AE/Dst-driven) perturbation.
    precip  : apply the auroral-precipitation E-region ionization model
              (recommended at high latitude; on by default, matching the
              vendored examples.c).
    dregion : apply E-CHAIM's own FIRI-2018 D-region addition. Off by
              default -- Hamilton's absorption.py already models the
              D-region via its own collision-frequency profile (see
              absorption.py); turn this on only if you want E-CHAIM's
              D-region density itself, not just absorption from it.
    """
    lat = np.ascontiguousarray(lat_deg, dtype=np.float64)
    lon = np.ascontiguousarray(lon_deg, dtype=np.float64)
    n = lat.size
    if lon.size != n:
        raise ValueError(f"lat_deg and lon_deg must be the same length ({n} vs {lon.size})")

    alt = np.ascontiguousarray(alt_km, dtype=np.float64)
    m = alt.size

    year = np.full(n, dt.year, dtype=np.float64)
    month = np.full(n, dt.month, dtype=np.float64)
    day = np.full(n, dt.day, dtype=np.float64)
    hour = np.full(n, dt.hour, dtype=np.float64)
    minute = np.full(n, dt.minute, dtype=np.float64)
    second = np.full(n, dt.second, dtype=np.float64)

    out_ptr = _lib.densityProfile(
        _as_c_double_array(lat), _as_c_double_array(lon),
        _as_c_double_array(year), _as_c_double_array(month), _as_c_double_array(day),
        _as_c_double_array(hour), _as_c_double_array(minute), _as_c_double_array(second),
        int(storm), int(precip), int(dregion),
        n,
        _as_c_double_array(alt), m,
        0,
    )

    try:
        result = np.empty((n, m), dtype=np.float64)
        for i in range(n):
            result[i, :] = np.ctypeslib.as_array(out_ptr[i], shape=(m,))
    finally:
        for i in range(n):
            _libc.free(out_ptr[i])
        _libc.free(out_ptr)

    return result


MIN_MLAT_VALID_DEG = 45.0    # below this AACGM latitude, densityProfile() returns NaN
MIN_MLAT_RECOMMENDED_DEG = 50.0  # E-CHAIM's own documented lower boundary; 45-50 is its
                                  # stated IRI-blending transition band (see module docstring)
