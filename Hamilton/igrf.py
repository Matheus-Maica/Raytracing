"""
igrf.py — real geomagnetic field (IGRF) on a coarse, pre-interpolated grid.

Mirrors ionosphere.py::Ionosphere's pattern for the same underlying reason:
the source (ppigrf.igrf) is expensive per call, the field itself is
smooth, so we pay the expensive part ONCE on a coarse grid and interpolate
cheaply afterward.

Why not call ppigrf per point (like Ionosphere calls IRI per interpolated
lookup)?
--------------------------------------------------------------------------
Ray._grad_Y_vec calls B_vec 6 times per RHS evaluation (central-difference
gradient), and a single fixed-step RK4 step calls the RHS 4 times -- tens of
thousands of B_vec calls per traced ray. ppigrf's IGRF functions re-read and
re-parse the IGRF .shc coefficient file into a fresh pandas DataFrame on
EVERY call (see ppigrf.ppigrf.read_shc). Doing that per finite-difference
point would be a severe performance regression. So we batch: one call for a
whole coarse grid, then plain interpolation.

Unlike Ionosphere's two-stage (coarse-IRI-then-upsample-to-fine-grid)
approach, there is no upsampling stage here: IGRF is a truncated
spherical-harmonic field (max degree 13) with no structure below its own
smooth length scale (~1500 km), so the coarse query grid is already dense
enough to interpolate directly.

Unlike Ionosphere (which returns Ne = 0 outside its patch, a legitimate
physical statement), a magnetic field must never silently go to zero
outside the patch -- B_vec_xyz_m clamps the query point into the grid's
range instead of extrapolating or zero-filling.

Grid: (h [km above WGS84], lat [deg, geodetic], lon [deg]) -- matches
ionosphere.py's convention. Uses ppigrf.igrf (geodetic-native: lon, lat,
height above ellipsoid, date -> Be, Bn, Bu east/north/up), the natural
pairing now that the grid itself is geodetic -- no geocentric round-trip
needed. The ENU basis trig below is the same lat/lon-only formulas as
before; only "up" stops being synonymous with position/|position| (true
only on a sphere) and becomes the real geodetic normal.

Public API
----------
    IgrfField(center_lat, center_lon, dtheta_deg, dphi_deg, h_max,
              dt, n_ang=5, n_r=5, max_degree=13)
    IgrfField.from_bounds(lat_min, lat_max, lon_min, lon_max,
                           h_min, h_max, dt, n_ang=5, n_r=5, max_degree=13)
    build_igrf_for_ionosphere(iono, n_ang=5, n_r=5, max_degree=13) -> IgrfField

    .B_vec_xyz_m(x, y, z) -> np.ndarray   # Tesla, x/y/z in metres
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import ppigrf

import geometry


class IgrfField:
    """IGRF magnetic field on a coarse geodetic grid, interpolated to arbitrary points."""

    def __init__(self, center_lat: float = 0.0, center_lon: float = 0.0,
                 dtheta_deg: float = 20.0, dphi_deg: float = 20.0,
                 h_max: float = 800.0, dt=None,
                 n_ang: int = 5, n_r: int = 5, max_degree: int = 13):
        self.center_lat = float(center_lat)
        self.center_lon = float(center_lon)

        half_lat = float(dtheta_deg) / 2.0
        half_lon = min(float(dphi_deg), 360.0) / 2.0

        lat_min = float(np.clip(center_lat - half_lat, -90.0, 90.0))
        lat_max = float(np.clip(center_lat + half_lat, -90.0, 90.0))
        lon_c = center_lon % 360.0
        lon_min = lon_c - half_lon
        lon_max = lon_c + half_lon
        h_min = 0.0
        h_max_ = float(h_max)

        self._build(lat_min, lat_max, lon_min, lon_max, h_min, h_max_,
                    dt, n_ang, n_r, max_degree)

    @classmethod
    def from_bounds(cls, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                     h_min: float, h_max: float, dt,
                     n_ang: int = 5, n_r: int = 5, max_degree: int = 13) -> "IgrfField":
        """Build directly from already-resolved (lat, lon, h) bounds (degrees / km)."""
        obj = cls.__new__(cls)
        obj.center_lat = None
        obj.center_lon = None
        obj._build(lat_min, lat_max, lon_min, lon_max, h_min, h_max,
                   dt, n_ang, n_r, max_degree)
        return obj

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build(self, lat_min, lat_max, lon_min, lon_max, h_min, h_max,
               dt, n_ang, n_r, max_degree):
        self.dt = dt
        self.lat_min, self.lat_max = float(lat_min), float(lat_max)
        self.lon_min, self.lon_max = float(lon_min), float(lon_max)
        self.h_min, self.h_max = float(h_min), float(h_max)

        self.h   = np.linspace(self.h_min,   self.h_max,   max(2, int(n_r)))
        self.lat = np.linspace(self.lat_min, self.lat_max, max(2, int(n_ang)))
        self.lon = np.linspace(self.lon_min, self.lon_max, max(2, int(n_ang)))
        self.Nh, self.Nlat, self.Nlon = len(self.h), len(self.lat), len(self.lon)

        Hg, LATg, LONg = np.meshgrid(self.h, self.lat, self.lon, indexing="ij")
        h_km  = Hg.ravel()
        lat_deg = LATg.ravel()
        lon_deg = LONg.ravel()

        Be, Bn, Bu = ppigrf.igrf(lon_deg, lat_deg, h_km, dt, max_degree=max_degree)
        # ppigrf keeps a leading date axis; we always pass a single date.
        Be = np.asarray(Be).reshape(-1)[: h_km.size] * 1e-9      # nT -> T
        Bn = np.asarray(Bn).reshape(-1)[: h_km.size] * 1e-9
        Bu = np.asarray(Bu).reshape(-1)[: h_km.size] * 1e-9

        lat_r, lon_r = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
        sinlat, coslat = np.sin(lat_r), np.cos(lat_r)
        sinlon, coslon = np.sin(lon_r), np.cos(lon_r)
        up_hat = np.stack([coslat * coslon, coslat * sinlon, sinlat], axis=-1)
        east_hat = np.stack([-sinlon, coslon, np.zeros_like(sinlon)], axis=-1)
        north_hat = np.cross(up_hat, east_hat)

        B_cart = (Bu[:, None] * up_hat + Bn[:, None] * north_hat + Be[:, None] * east_hat)
        shape = (self.Nh, self.Nlat, self.Nlon)
        Bx = B_cart[:, 0].reshape(shape)
        By = B_cart[:, 1].reshape(shape)
        Bz = B_cart[:, 2].reshape(shape)

        axes = (self.h, self.lat, self.lon)
        self._interp_x = RegularGridInterpolator(axes, Bx, method="linear", bounds_error=False, fill_value=0.0)
        self._interp_y = RegularGridInterpolator(axes, By, method="linear", bounds_error=False, fill_value=0.0)
        self._interp_z = RegularGridInterpolator(axes, Bz, method="linear", bounds_error=False, fill_value=0.0)

    # ------------------------------------------------------------------
    # Coordinate helper (local; mirrors Ionosphere's _map_lon but
    # deliberately not imported from ionosphere.py so igrf.py has no
    # dependency on iricore)
    # ------------------------------------------------------------------

    def _map_lon(self, lon: float) -> float:
        return self.lon_min + ((lon - self.lon_min) % 360.0)

    # ------------------------------------------------------------------
    # Public lookup
    # ------------------------------------------------------------------

    def B_vec_xyz_m(self, x: float, y: float, z: float) -> np.ndarray:
        """IGRF field in Tesla at geocentric Cartesian (x, y, z) in metres."""
        lat, lon, h_m = geometry.xyz_m_to_latlon(np.array([x, y, z], dtype=float))
        h_km = h_m / 1e3

        h_c = float(np.clip(h_km, self.h[0], self.h[-1]))
        lat_c = float(np.clip(lat, self.lat[0], self.lat[-1]))
        lon_c = float(np.clip(self._map_lon(lon), self.lon[0], self.lon[-1]))

        pt = np.array([[h_c, lat_c, lon_c]])
        return np.array([
            float(self._interp_x(pt)[0]),
            float(self._interp_y(pt)[0]),
            float(self._interp_z(pt)[0]),
        ])


def build_igrf_for_ionosphere(iono, n_ang: int = 5, n_r: int = 5, max_degree: int = 13) -> IgrfField:
    """
    Build an IgrfField over exactly the same patch as an already-constructed
    ionosphere.Ionosphere -- same (lat, lon, h) bounds and the same dt, so
    the B and Ne patches describe the same region and UTC snapshot with no
    duplicated patch-sizing math.
    """
    field = IgrfField.from_bounds(
        lat_min=iono.lat_min, lat_max=iono.lat_max,
        lon_min=iono.lon_min, lon_max=iono.lon_max,
        h_min=iono.h_min, h_max=iono.h_max,
        dt=iono.dt, n_ang=n_ang, n_r=n_r, max_degree=max_degree,
    )
    field.center_lat = iono.center_lat
    field.center_lon = iono.center_lon
    return field
