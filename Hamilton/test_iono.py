"""
Ionosphere built from the International Reference Ionosphere (IRI),
via the `iricore` wrapper (https://iricore.readthedocs.io).

This mirrors the structure of the earlier synthetic `Ionosphere` class:
a 3D grid with the origin at the center of the horizontal (x-z) plane,
y running "up" from the floor, and trilinear interpolation to query
arbitrary (non-grid) points.

FLAT-EARTH APPROXIMATION (degrees -> metres)
----------------------------------------------
IRI works in (latitude, longitude, altitude) on a spherical/ellipsoidal
Earth. There is no distortion-free way to map a patch of a sphere onto
a flat Cartesian plane, so this class uses a standard *local tangent
plane* (equirectangular) projection, which is the simplest mapping
that is exact along one reference parallel and reasonable for regional
(tens to a few hundred km) extents:

    phi0    = reference latitude  = center of lat_range   (radians)
    lambda0 = reference longitude = center of lon_range   (radians)

    x = R_EARTH * cos(phi0) * (lambda - lambda0)     (east-west, metres)
    z = R_EARTH * (phi - phi0)                       (north-south, metres)
    y = altitude, converted from km to metres          ("up")

where R_EARTH is Earth's mean radius (6371 km) and lambda, phi are the
longitude/latitude of each sample point, in radians.

Assumptions and resulting distortion, explicitly:
  1. Earth is treated as a sphere of constant radius R_EARTH (no
     ellipsoidal flattening). This is a ~0.3% effect on R_EARTH-scale
     distances -- usually negligible next to ionospheric variability.
  2. The east-west (x) scale factor cos(phi0) is computed ONCE at the
     center latitude and applied uniformly across the whole grid. In
     reality, meridians converge toward the poles, so a degree of
     longitude spans less real distance at the edges of the lat_range
     than at phi0. This means x is increasingly overstated as you move
     away from phi0 in latitude -- the further north/south of center,
     the more compressed the real world is relative to this flat grid.
     Over a +-5 deg latitude span this error is small (~0.4%); over
     +-20 deg it is a few percent.
  3. North-south (z) distance is exact under the spherical assumption,
     since meridian arc length per degree of latitude is constant on a
     sphere (no such convergence effect along that axis).
  4. The mapping is linear (one global scale factor per axis), so the
     resulting grid is uniformly spaced in x and z -- a true Cartesian
     grid, not a sphere-segment forced flat. This uniformity is what
     makes it usable as a drop-in for the Ray class, at the cost of the
     distortion described above.
  5. Altitude is taken as the "y" (up) coordinate directly, simply unit
     converted (km -> m). Altitude in IRI is height above the
     reference ellipsoid/sea level, not a true Cartesian height in this
     tangent-plane frame, but the difference is not modeled here.

As before, x and z are centered on 0 (the origin sits below the center
of the lat/lon box, at sea level), and y starts at 0 (the lowest
altitude sampled).
"""

import numpy as np
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

import iricore

R_EARTH = 6371000.0  # mean Earth radius, metres


class Ionosphere:
    def __init__(self, Nx, Ny, Nz,
                 lon_range=(-5.0, 5.0), lat_range=(-5.0, 5.0),
                 alt_range=(60.0, 600.0),
                 time=None, iri_version=20):
        """
        Nx, Ny, Nz : number of grid points along x (<-longitude),
                     y (<-altitude), z (<-latitude)

        lon_range  : (lon_min, lon_max) in degrees, sampled with Nx points
        lat_range  : (lat_min, lat_max) in degrees, sampled with Nz points
        alt_range  : (alt_min, alt_max) in km, sampled with Ny points

        time         : datetime for the IRI run (default: now, UTC-naive)
        iri_version  : 16 or 20, passed straight to iricore.iri()

        After construction:
            self.x : Nx points, east-west distance in METRES,
                     centered on 0 at the reference longitude
            self.y : Ny points, altitude in METRES, starting at 0
            self.z : Nz points, north-south distance in METRES,
                     centered on 0 at the reference latitude
            self.density : array of shape (Nx, Ny, Nz), electron
                            density in m^-3 from IRI's edens output

        See the module docstring for the flat-earth (local tangent
        plane) projection used to convert lat/lon/alt into x/y/z.
        """
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.alt_range = alt_range
        self.time = time if time is not None else datetime.utcnow()
        self.iri_version = iri_version

        # --- Real geographic sample points (used only to call IRI) ---
        self._lons = np.linspace(lon_range[0], lon_range[1], Nx)
        self._lats = np.linspace(lat_range[0], lat_range[1], Nz)
        alt_step = (alt_range[1] - alt_range[0]) / (Ny - 1)
        self._alts = np.linspace(alt_range[0], alt_range[1], Ny)
        # iricore wants (alt_start, alt_stop, alt_step) for altrange
        self._altrange_arg = (alt_range[0], alt_range[1], alt_step)

        # --- Reference point for the local tangent-plane projection ---
        self.lat0 = 0.5 * (lat_range[0] + lat_range[1])  # degrees
        self.lon0 = 0.5 * (lon_range[0] + lon_range[1])  # degrees
        phi0 = np.radians(self.lat0)
        self._cos_phi0 = np.cos(phi0)

        # --- Cartesian axes, in metres, via the flat-earth projection ---
        # x: east-west, scaled by cos(phi0) evaluated ONCE at the
        #    reference latitude (see distortion notes in module docstring)
        self.x = R_EARTH * self._cos_phi0 * np.radians(self._lons - self.lon0)
        # z: north-south, exact under the spherical-Earth assumption
        self.z = R_EARTH * np.radians(self._lats - self.lat0)
        # y: altitude, km -> m, starting at 0
        self.y = (self._alts - self._alts[0]) * 1000.0

        self.density = self._build_density()

        self._interp = RegularGridInterpolator(
            (self.x, self.y, self.z),
            self.density,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    # ------------------------------------------------------------
    def _build_density(self):
        """
        Query IRI once per (lat, lon) pair (iricore pairs lat/lon
        elementwise rather than giving a full outer-product grid), and
        stack the resulting altitude profiles into a full 3D array of
        shape (Nx, Ny, Nz) = (n_lon, n_alt, n_lat).
        """
        density = np.empty((self.Nx, self.Ny, self.Nz))

        for ix, lon in enumerate(self._lons):
            for iz, lat in enumerate(self._lats):
                out = iricore.iri(
                    self.time,
                    self._altrange_arg,
                    float(lat),
                    float(lon),
                    version=self.iri_version,
                )
                # out.edens has shape (Ny,) for scalar lat/lon
                density[ix, :, iz] = out.edens

        # IRI returns negative sentinel values where it can't compute
        # a density (e.g. bad inputs); clip those to zero.
        density = np.where(density < 0, 0.0, density)
        return density

    # ------------------------------------------------------------
    def regenerate(self, time=None):
        """Rebuild the density field, optionally for a new time."""
        if time is not None:
            self.time = time
        self.density = self._build_density()
        self._interp = RegularGridInterpolator(
            (self.x, self.y, self.z),
            self.density,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def electron_density(self, x, y, z):
        """
        Trilinearly interpolated electron density at (x, y, z), where
        x, y, z are in METRES in the flat-earth tangent-plane frame
        described in the module docstring (x = east-west, y = up
        altitude, z = north-south). Accepts scalars or numpy arrays.
        Out-of-range points are clamped to the grid edges.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        x_c = np.clip(x, self.x[0], self.x[-1])
        y_c = np.clip(y, self.y[0], self.y[-1])
        z_c = np.clip(z, self.z[0], self.z[-1])

        pts = np.stack([x_c, y_c, z_c], axis=-1)
        result = self._interp(pts)

        if result.shape == ():
            return float(result)
        return result

    def __call__(self, x, y, z):
        return self.electron_density(x, y, z)


# ----------------------------------------------------------------
# Example usage / quick visualization
# ----------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    iono = Ionosphere(
        Nx=30, Ny=80, Nz=20,
        lon_range=(-10.0, 10.0),
        lat_range=(30.0, 50.0),
        alt_range=(60.0, 600.0),
        time=datetime(2024, 6, 21, 12, 0, 0),
        iri_version=20,
    )

    print(f"Reference point: lat0={iono.lat0} deg, lon0={iono.lon0} deg")
    print(f"x range: {iono.x[0]/1e3:.1f} to {iono.x[-1]/1e3:.1f} km")
    print(f"y range: {iono.y[0]/1e3:.1f} to {iono.y[-1]/1e3:.1f} km")
    print(f"z range: {iono.z[0]/1e3:.1f} to {iono.z[-1]/1e3:.1f} km")

    iz = iono.Nz // 2
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(iono.x / 1e3, iono.y / 1e3, iono.density[:, :, iz].T,
                   shading="auto", cmap="viridis")
    plt.colorbar(label="Electron density (m$^{-3}$)")
    plt.xlabel("x  (east-west, km)")
    plt.ylabel("y  (altitude, km)")
    plt.title("IRI-based ionosphere slice (flat-earth projection)")

    plt.show()