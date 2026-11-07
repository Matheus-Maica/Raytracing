"""
ccmp_wind.py -- CCMP (Cross-Calibrated Multi-Platform) ocean surface wind, for the sea-Bragg clutter of
clutter.py/clutter_scene.py (HiCIRF paragraph 24: "Bragg clutter is driven by cross-calibrated
multiplatform (CCMP) ocean surface wind models").

Data: RSS's CCMP v3.1 daily L4 files, CCMP_Wind_Analysis_YYYYMMDD_V03.1_L4.nc (~33 MB each), served without
an account at https://data.remss.com/ccmp/v03.1/YYYY/MMM/ (e.g. .../Y2026/M09/). Each holds four 6-hourly
analyses (00, 06, 12, 18 UT) of uwnd, vwnd (and ws, nobs) [m/s at 10 m] on a 0.25 deg global grid,
dims (time, latitude, longitude), longitude 0..360, time in hours since 1987-01-01, fill -9999 (NaN here --
mostly sea ice). fetch_ccmp_day() downloads a day into a local cache (HAMILTON_CACHE, default
~/.cache/hamilton/ccmp) the first time it is needed; ccmp_wind_for(t) returns the CCMPWind for t's day.
Reading needs the netCDF4 package (installed into the venv, 2026-11-07, with the user's permission).

CCMPWind is a plain in-memory lookup (nearest neighbour in space and time), constructible from arrays too
(the unit tests use synthetic arrays; tests/test_ccmp_wind.py also checks a real cached file when present).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

CCMP_BASE_URL = "https://data.remss.com/ccmp/v03.1"
_EPOCH = datetime(1970, 1, 1)


def _seconds(t: datetime) -> float:
    """Seconds since 1970 for a naive UTC datetime, without the local-timezone step of .timestamp()."""
    return (datetime(t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond) - _EPOCH).total_seconds()


@dataclass
class CCMPWind:
    """
    An in-memory CCMP-shaped wind field: uwnd/vwnd [m/s] on a (time, lat, lon) grid. Construct it directly
    from arrays or via load_ccmp_netcdf(path) / ccmp_wind_for(t).

    lat_deg   : (Nlat,) ascending or descending, degrees
    lon_deg   : (Nlon,) either 0..360 or -180..180 (wrapped automatically at lookup)
    time      : (Nt,) sequence of datetime (naive UTC, matching the rest of Hamilton)
    uwnd, vwnd: (Nt, Nlat, Nlon) eastward / northward wind components [m/s]; NaN where CCMP has no value
    """

    lat_deg: np.ndarray
    lon_deg: np.ndarray
    time: list
    uwnd: np.ndarray
    vwnd: np.ndarray

    def __post_init__(self):
        self.lat_deg = np.asarray(self.lat_deg, dtype=float)
        self.lon_deg = np.asarray(self.lon_deg, dtype=float) % 360.0
        self.uwnd = np.asarray(self.uwnd, dtype=np.float32)
        self.vwnd = np.asarray(self.vwnd, dtype=np.float32)
        order = np.argsort(self.lon_deg)
        self.lon_deg = self.lon_deg[order]
        self.uwnd = self.uwnd[:, :, order]
        self.vwnd = self.vwnd[:, :, order]
        self._t_sec = np.array([_seconds(t) for t in self.time], dtype=float)

    def _nearest(self, arr: np.ndarray, value: float) -> int:
        return int(np.argmin(np.abs(arr - value)))

    def at(self, lat_deg, lon_deg, t: datetime) -> dict:
        """
        Nearest-neighbour lookup in space and time (0.25 deg / 6 h grid). Returns wind_direction_deg (the
        direction the wind blows TOWARDS, inputs.SurfaceInputs.wind_direction_deg's convention) and
        wind_speed_mps, each shaped like lat_deg; NaN where CCMP has no value (sea ice) -- callers fall back.
        """
        lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
        lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)) % 360.0
        it = self._nearest(self._t_sec, _seconds(t)) if t is not None else 0
        ilat = np.array([self._nearest(self.lat_deg, v) for v in lat])
        ilon = np.array([self._nearest(self.lon_deg, v) for v in lon])
        u = self.uwnd[it, ilat, ilon].astype(float)
        v = self.vwnd[it, ilat, ilon].astype(float)
        direction = (np.degrees(np.arctan2(u, v))) % 360.0   # 0 = towards north, 90 = towards east
        speed = np.sqrt(u ** 2 + v ** 2)
        return dict(wind_direction_deg=direction, wind_speed_mps=speed)


def load_ccmp_netcdf(path: str) -> CCMPWind:
    """Read a CCMP v3.1 file (uwnd, vwnd, latitude, longitude, time) into a CCMPWind."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        lat = ds.variables["latitude" if "latitude" in ds.variables else "lat"][:]
        lon = ds.variables["longitude" if "longitude" in ds.variables else "lon"][:]
        time_var = ds.variables["time"]
        times = netCDF4.num2date(time_var[:], time_var.units,
                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        uwnd = np.ma.filled(ds.variables["uwnd"][:].astype(np.float32), np.nan)
        vwnd = np.ma.filled(ds.variables["vwnd"][:].astype(np.float32), np.nan)
    times = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in times]
    return CCMPWind(lat_deg=np.asarray(lat), lon_deg=np.asarray(lon), time=times, uwnd=uwnd, vwnd=vwnd)


def ccmp_cache_dir() -> str:
    return os.path.join(os.environ.get("HAMILTON_CACHE", os.path.expanduser("~/.cache/hamilton")), "ccmp")


def ccmp_filename(day: date) -> str:
    return f"CCMP_Wind_Analysis_{day:%Y%m%d}_V03.1_L4.nc"


def ccmp_url(day: date) -> str:
    return f"{CCMP_BASE_URL}/Y{day:%Y}/M{day:%m}/{ccmp_filename(day)}"


def fetch_ccmp_day(day: date, cache_dir: str | None = None) -> str:
    """Path of the day's CCMP file, downloading it into the cache first if it is not there."""
    cache_dir = cache_dir or ccmp_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, ccmp_filename(day))
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(ccmp_url(day), tmp)
        os.replace(tmp, path)
    return path


def ccmp_wind_for(t: datetime, cache_dir: str | None = None) -> CCMPWind:
    """CCMPWind for the UTC day of t (its four analyses; nearest-in-time selection happens in .at())."""
    return load_ccmp_netcdf(fetch_ccmp_day(t.date(), cache_dir))
