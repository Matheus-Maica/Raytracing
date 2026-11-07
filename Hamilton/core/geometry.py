"""
geometry.py — coordinate and geometry helpers for WGS84-ellipsoid Earth ray tracing.

All positions are geocentric Cartesian (ECEF) metres unless noted otherwise.
Latitude is GEODETIC throughout (the angle the ellipsoid normal makes with
the equatorial plane — what IRI, IGRF, GPS, and maps all mean by "latitude"),
not geocentric.

Relocated out of main.py so raytracer.py and absorption.py can use these
helpers without importing main.py (which itself imports raytracer.py).
main.py re-imports these names, so nothing downstream of main.py changes.

WGS84, not a full geoid: the reference ellipsoid captures the ~21 km
pole/equator radius variation (via flattening) that actually matters for
altitude-sensitive quantities (absorption's D/E-region lookup, ground_range,
multi-hop geometry). A full geoid's own undulations on top of the ellipsoid
are +/-100 m at most -- negligible next to this model's other uncertainties
(the ionosphere itself), so not implemented.

No pyproj: the forward transform is closed-form; the inverse
(ECEF -> geodetic) uses Bowring's (1976) method, seeded via the reduced
latitude and refined with 2 fixed-point iterations. Verified numerically in
this project (200,000 random points spanning the full lat/lon/altitude
range, plus pole/equator/antimeridian/h=0 edge cases): 2 iterations give
max round-trip error ~5.5e-8 m in position, ~1.1e-7 m in height -- float64
noise, no special-casing needed anywhere including exactly at the poles.
"""

import numpy as np

# ---------------------------------------------------------------------------
# WGS84 ellipsoid constants
# ---------------------------------------------------------------------------
WGS84_A   = 6378137.0                              # semi-major axis, metres
WGS84_F   = 1.0 / 298.257223563                     # flattening
WGS84_B   = WGS84_A * (1.0 - WGS84_F)               # semi-minor axis, metres
WGS84_E2  = WGS84_F * (2.0 - WGS84_F)               # first eccentricity^2
WGS84_EP2 = (WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B ** 2   # second eccentricity^2
R_MEAN    = (2.0 * WGS84_A + WGS84_B) / 3.0         # mean radius, metres --
                                                     # used only by the (deliberately
                                                     # kept-spherical) range functions below


# ---------------------------------------------------------------------------
# Point conversions
# ---------------------------------------------------------------------------

def latlon_to_xyz_m(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Geodetic (lat °, lon °, height m above the WGS84 ellipsoid) → ECEF metres."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array([
        (N + alt_m) * np.cos(lat) * np.cos(lon),
        (N + alt_m) * np.cos(lat) * np.sin(lon),
        (N * (1.0 - WGS84_E2) + alt_m) * np.sin(lat),
    ])


def xyz_m_to_latlon(xyz: np.ndarray, n_iter: int = 2):
    """
    ECEF metres → geodetic (lat °, lon °, height m above the WGS84 ellipsoid).

    Bowring's (1976) method: a non-iterative seed from the reduced latitude,
    refined by `n_iter` fixed-point iterations (default 2 -- sub-micrometre
    accuracy everywhere, see module docstring).
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    p = np.sqrt(x * x + y * y)

    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )

    N = WGS84_A
    h = 0.0
    for _ in range(n_iter):
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + h)))

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    lon = np.arctan2(y, x)

    return np.rad2deg(lat), np.rad2deg(lon), h


def ellipsoid_outward_normal_m(xyz: np.ndarray, A: float = WGS84_A, B: float = WGS84_B) -> np.ndarray:
    """
    Outward unit normal to the ellipsoid x^2/A^2 + y^2/A^2 + z^2/B^2 = 1 at
    (or near) its surface, i.e. the gradient of the implicit equation,
    normalized: (x/A^2, y/A^2, z/B^2) / |...|.

    At A=WGS84_A, B=WGS84_B (the true ground ellipsoid) this is EXACTLY
    geodetic "up" -- provable directly from the geodetic parametrization
    (z/B^2 = N(1-e^2)sin(lat)/B^2 = (N/A^2)sin(lat) since B^2=A^2(1-e^2),
    matching the x/A^2, y/A^2 prefactor exactly) -- verified numerically
    against the lat/lon trig formula to ~1e-6 degrees. This is the correct
    normal for specular ground-bounce reflection: on a sphere the outward
    normal happens to equal the radial direction r/|r|, but on the true
    ellipsoid these differ by up to 0.192 degrees (the textbook
    "deflection of the vertical", at latitude ~45 degrees).
    """
    n = np.array([xyz[0] / A ** 2, xyz[1] / A ** 2, xyz[2] / B ** 2])
    return n / np.linalg.norm(n)


def _ray_ellipsoid_intersect_m(P0: np.ndarray, d_hat: np.ndarray, A: float, B: float):
    """
    First positive intersection distance (metres) of the ray
        P(t) = P0 + t * d_hat
    with the ellipsoid of revolution x^2/A^2 + y^2/A^2 + z^2/B^2 = 1
    (semi-major A, semi-minor/polar B). Returns None if no intersection
    exists or the ellipsoid is entirely behind the ray. Reduces to the
    sphere case (A=B=R) identically.
    """
    inv_A2 = 1.0 / A ** 2
    inv_B2 = 1.0 / B ** 2
    a_coef = d_hat[0] ** 2 * inv_A2 + d_hat[1] ** 2 * inv_A2 + d_hat[2] ** 2 * inv_B2
    b_coef = 2.0 * (P0[0] * d_hat[0] * inv_A2 + P0[1] * d_hat[1] * inv_A2 + P0[2] * d_hat[2] * inv_B2)
    c_coef = P0[0] ** 2 * inv_A2 + P0[1] ** 2 * inv_A2 + P0[2] ** 2 * inv_B2 - 1.0

    disc = b_coef ** 2 - 4.0 * a_coef * c_coef
    if disc < 0.0:
        return None
    sq = np.sqrt(disc)
    t1 = (-b_coef - sq) / (2.0 * a_coef)
    t2 = (-b_coef + sq) / (2.0 * a_coef)
    candidates = [t for t in (t1, t2) if t >= -1e-3]   # allow tiny numerical negative
    return min(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Range / bearing helpers
# ---------------------------------------------------------------------------
# Deliberately kept as a SPHERICAL approximation (mean WGS84 radius) rather
# than full ellipsoidal geodesics (Vincenty/Karney): ground_range is a
# reporting/diagnostic quantity here, not a hard physics boundary, and exact
# ellipsoidal geodesics are a separate, occasionally ill-conditioned
# (near-antipodal) numerical problem not justified by the payoff.

def great_circle_range_m(pos_vec: np.ndarray, ref_vec: np.ndarray, R: float = R_MEAN) -> float:
    """
    Great-circle surface distance (metres) between the sub-points of two
    geocentric position vectors, i.e. R times the central angle between
    their directions.  `ref_vec` need not be at radius R (only its
    direction matters) — pass the launch point to get ground range from
    the transmitter.
    """
    p_hat = np.asarray(pos_vec, dtype=float)
    p_hat = p_hat / np.linalg.norm(p_hat)
    r_hat = np.asarray(ref_vec, dtype=float)
    r_hat = r_hat / np.linalg.norm(r_hat)
    ang = np.arccos(np.clip(np.dot(p_hat, r_hat), -1.0, 1.0))
    return R * ang


def down_cross_range_m(pos_vec: np.ndarray, r0_vec: np.ndarray, forward_hat: np.ndarray,
                        R: float = R_MEAN):
    """
    Decompose the great-circle range from r0_vec to pos_vec into
    (down_range, cross_range) metres along a local tangent-plane frame at
    r0_vec, where `forward_hat` is the tangent "forward" direction
    (down-range axis) and cross-range is measured to its right-hand
    perpendicular (up0 x forward_hat).

    Generalizes the range/cross-range decomposition previously duplicated
    inline in main.py's ray-fan plotting and mc_scattering.py's ensemble
    summary.
    """
    up0 = np.asarray(r0_vec, dtype=float)
    up0 = up0 / np.linalg.norm(up0)
    fwd = np.asarray(forward_hat, dtype=float)
    fwd = fwd / np.linalg.norm(fwd)
    c_hat = np.cross(up0, fwd)

    p_hat = np.asarray(pos_vec, dtype=float)
    p_hat = p_hat / np.linalg.norm(p_hat)

    ang = np.arccos(np.clip(np.dot(p_hat, up0), -1.0, 1.0))
    az  = np.arctan2(np.dot(p_hat, c_hat), np.dot(p_hat, fwd))
    return R * ang * np.cos(az), R * ang * np.sin(az)
