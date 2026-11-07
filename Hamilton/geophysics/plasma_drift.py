"""
plasma_drift.py -- WBMOD-style plasma drift (HiCIRF, Nickisch et al. 2012, section 2.3: "the ionospheric plasma
drift velocity model from WBMOD, which combines the Richmond-NCAR (mid-low latitude) and Heelis-UTD (high
latitude) plasma convection models").

HeelisDrift -- a full port of flwv32()/potm() from NCAR TIE-GCM's heelis.F90 (docs/heelis.F90.txt, the file
itself): auroral-circle coordinates (circle centre 1 deg towards midnight, offc), the 7-segment half-cosine
longitudinal function over psi(1:8)/phi(1:8), the cubic inside the convection reversal boundary theta0(ctpoten),
the (sin colat)^rr1 exp(7(1 - ...)) falloff outside it, and zero equatorward of 30 deg magnetic latitude.
Reading the file settled two things about the earlier, partial port:
  (1) with TIE-GCM's default parameters (phidm0 = phidp0 = phinm0 = phinp0 = 90 deg, phid = 0, phin = 180 deg)
      the longitudinal function is exactly -psim sin(dlon): the single dawn/dusk harmonic used before;
  (2) the cubic inside theta0 mixes phifun with phifn2, the value on the OPPOSITE side of the pole, which is
      -phifun -- not +phifun as assumed before. The cubic is phifun (1.5 x - 0.5 x^3), giving the uniform
      anti-sunward flow across the polar cap. The earlier smoothstep had zero flow at the pole.
The electric field is -grad(Phi), by central differences along the local horizontal (no pole singularity),
and v = E x B / B^2.

RichmondDrift -- OUR OWN crude mid/low-latitude pattern (user decision, 2026-11-07). The real Richmond et al.
(1980) model was distributed by NSSDC only "on tape, SPAN (NSSDC ID #MI-93B)" (NSSDC/WDC-A-R&S 90-19, 1990),
and is not online anywhere we could find (SPDF's surviving model archive has no ionospheric section; CCMC's
catalogue does not list it). This one: an eastward zonal drift with a day/night reversal, night amplitude at
the 100-150 m/s HiCIRF itself quotes (paragraph 11), tapered off between 40 and 60 deg magnetic latitude.

WBMODStyleDrift -- OUR OWN crude combination rule (user decision for this item, 2026-11-07): the two are
SUPERPOSED, v = v_Heelis + v_Richmond. Each already handles its own domain (Heelis is exactly zero equatorward
of 30 deg magnetic latitude; Richmond is tapered off poleward of 40-60 deg), so no interpolation weight is
needed. TIE-GCM's own blend (colath: 15 -> 30 deg auroral-circle colatitude, docs/heelis.F90.txt) was not used:
it hands over to TIE-GCM's self-consistent dynamo potential, which carries the high-latitude forcing into the
auroral zone; a crude quiet-time pattern does not, and interpolating to it would weaken auroral convection.

Geomagnetic frame: centred dipole from IGRF's degree-1 Gauss coefficients (g10, g11, h11) for the date, from
ppigrf's own IGRF table. (Previously main.Ray's fixed 9.6 deg tilt towards 0 deg E, which put the north
geomagnetic pole near 80 N, 0 E instead of near 80.8 N, 72.7 W -- the auroral oval in the wrong place.)
Magnetic local time from the subsolar point (latitude = solar declination) in that frame. heelis.F90's own
sunloc uses a cruder solar position (no equation of time); the difference is a few degrees of dlon at most.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

import numpy as np

from core.geometry import latlon_to_xyz_m, xyz_m_to_latlon

_PI2 = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Geomagnetic frame (IGRF centred dipole) and magnetic local time
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _igrf_degree1():
    """(epoch years, g10, g11, h11) [nT] from ppigrf's IGRF coefficient table."""
    import ppigrf.ppigrf as P
    g, h = P.read_shc()
    years = np.array([d.year + (d.dayofyear - 1) / 365.25 for d in g.index])
    return years, g[(1, 0)].to_numpy(float), g[(1, 1)].to_numpy(float), h[(1, 1)].to_numpy(float)


def dipole_gauss_nT(t: datetime) -> np.ndarray:
    """(g11, h11, g10) [nT] at t, linear in time between IGRF epochs."""
    years, g10, g11, h11 = _igrf_degree1()
    y = t.year + (t.timetuple().tm_yday - 1) / 365.25
    return np.array([np.interp(y, years, g11), np.interp(y, years, h11), np.interp(y, years, g10)])


def dipole_axis_ecef(t: datetime) -> np.ndarray:
    """Unit vector to the north (boreal) geomagnetic pole, ECEF: -(g11, h11, g10)/|.|."""
    G = dipole_gauss_nT(t)
    return -G / np.linalg.norm(G)


def _mag_frame(t: datetime):
    """(axis, e1, e2): e1, e2 span the magnetic equatorial plane; magnetic longitude 0 is along e1 (the
    meridian towards the geographic north pole). Only longitude DIFFERENCES are used here."""
    axis = dipole_axis_ecef(t)
    z = np.array([0.0, 0.0, 1.0])
    e1 = z - np.dot(z, axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return axis, e1, e2


def geomagnetic_lat_lon_rad(r_ecef, t: datetime):
    """Centred-dipole magnetic latitude (signed) and longitude [rad] of each ECEF point at time t."""
    axis, e1, e2 = _mag_frame(t)
    r = np.atleast_2d(np.asarray(r_ecef, dtype=float))
    r_hat = r / np.linalg.norm(r, axis=-1, keepdims=True)
    mlat = np.arcsin(np.clip(r_hat @ axis, -1.0, 1.0))
    mlon = np.arctan2(r_hat @ e2, r_hat @ e1)
    return mlat, mlon


def subsolar_point_deg(t: datetime):
    """
    Low-precision solar position (standard NOAA/Meeus approximation, a fraction of a degree): returns
    (subsolar_lat_deg, subsolar_lon_deg, declination_deg). The subsolar latitude is the declination.
    """
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    n = (t - j2000).total_seconds() / 86400.0
    L = np.deg2rad((280.460 + 0.9856474 * n) % 360.0)
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2.0 * g)
    eps = np.deg2rad(23.439 - 0.0000004 * n)
    decl = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    eqtime_deg = ((np.rad2deg(L) - np.rad2deg(ra)) + 180.0) % 360.0 - 180.0
    ut_hours = t.hour + t.minute / 60.0 + t.second / 3600.0
    subsolar_lon = -(15.0 * ut_hours - 180.0 + eqtime_deg)
    subsolar_lon = (subsolar_lon + 180.0) % 360.0 - 180.0
    return float(np.rad2deg(decl)), float(subsolar_lon), float(np.rad2deg(decl))


def sun_magnetic_longitude_rad(t: datetime) -> float:
    lat, lon, _ = subsolar_point_deg(t)
    _, mlon = geomagnetic_lat_lon_rad(latlon_to_xyz_m(lat, lon, 0.0)[None, :], t)
    return float(mlon[0])


def magnetic_local_time_hours(r_ecef, t: datetime) -> np.ndarray:
    """MLT [h], 12 at the subsolar magnetic meridian (heelis.F90's dlon = mlon - sunlon is 0 at noon)."""
    _, mlon = geomagnetic_lat_lon_rad(r_ecef, t)
    dlon = mlon - sun_magnetic_longitude_rad(t)
    return (np.rad2deg(dlon) / 15.0 + 12.0) % 24.0


def _dlon_rad(r_ecef, t: datetime) -> np.ndarray:
    """heelis.F90's dlon: magnetic longitude minus the sun's, wrapped to (-pi, pi]."""
    _, mlon = geomagnetic_lat_lon_rad(r_ecef, t)
    return (mlon - sun_magnetic_longitude_rad(t) + np.pi) % _PI2 - np.pi


def _horizontal_basis(r: np.ndarray):
    """Unit (east, north) tangent vectors of the sphere at each point (spherical, not geodetic)."""
    r_hat = r / np.linalg.norm(r, axis=-1, keepdims=True)
    east = np.cross(np.array([0.0, 0.0, 1.0]), r_hat)
    n = np.linalg.norm(east, axis=-1, keepdims=True)
    east = np.where(n > 1e-12, east / np.where(n > 1e-12, n, 1.0), np.array([0.0, 1.0, 0.0]))
    north = np.cross(r_hat, east)
    return east, north


def _grad_E_field(potential_V, r: np.ndarray, h_m: float) -> np.ndarray:
    """E = -grad(Phi) [V/m] by central differences along the local horizontal, points kept at |r|."""
    R = np.linalg.norm(r, axis=-1, keepdims=True)
    east, north = _horizontal_basis(r)
    E = np.zeros_like(r)
    for e in (east, north):
        rp = r + h_m * e
        rm = r - h_m * e
        rp *= R / np.linalg.norm(rp, axis=-1, keepdims=True)
        rm *= R / np.linalg.norm(rm, axis=-1, keepdims=True)
        E -= ((potential_V(rp) - potential_V(rm)) / (2.0 * h_m))[:, None] * e
    return E


def centred_dipole_B(r_ecef, t: datetime) -> np.ndarray:
    """IGRF centred-dipole field [T]: B = (a/r)^3 [3 (G.r_hat) r_hat - G], G = (g11, h11, g10)."""
    a = 6371.2e3
    G = dipole_gauss_nT(t) * 1e-9
    r = np.atleast_2d(np.asarray(r_ecef, dtype=float))
    R = np.linalg.norm(r, axis=-1, keepdims=True)
    r_hat = r / R
    return (a / R) ** 3 * (3.0 * (r_hat @ G)[:, None] * r_hat - G[None, :])


# ---------------------------------------------------------------------------
# Heelis-UTD (high latitude): port of heelis.F90 flwv32()/potm()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeelisParams:
    """heelis_init() values (identical for both hemispheres in heelis.F90). offa/dskofa are set there but
    never used by flwv32's potential, so they are omitted."""

    offc_deg: float = 1.0       # offset of the auroral circle towards 0 MLT relative to the magnetic pole
    dskofc_deg: float = 0.0     # offset towards 18 MLT
    phin_deg: float = 180.0     # night convection entrance (MLT angle)
    phid_deg: float = 0.0       # dayside convection entrance
    pcen_kV: float = 0.0
    phidp0_deg: float = 90.0
    phidm0_deg: float = 90.0
    phinp0_deg: float = 90.0
    phinm0_deg: float = 90.0
    rr1: float = -2.6


def heelis_potential_kV(mlat_rad, dlon_rad, ctpoten_kV: float, p: HeelisParams = HeelisParams()):
    """
    Heelis potential [kV] at magnetic latitude mlat and dlon = mlon - sun's mlon (0 at noon), a line-by-line
    port of flwv32() (and potm()'s zero equatorward of 30 deg). psim = +ctpoten/2 (dawn), psie = -ctpoten/2
    (dusk); theta0 = -3.80 + 8.48 ctpoten^0.1875 deg (heelis_init).
    """
    mlat, dlon = np.broadcast_arrays(np.asarray(mlat_rad, dtype=float), np.asarray(dlon_rad, dtype=float))
    ctp = float(ctpoten_kV)
    psim, psie, pcen = 0.5 * ctp, -0.5 * ctp, p.pcen_kV
    theta0 = np.deg2rad(-3.80 + 8.48 * ctp ** 0.1875)
    pih, eps = 0.5 * np.pi, 1.0e-6
    d2r = np.pi / 180.0

    phin, phid = p.phin_deg * d2r, p.phid_deg * d2r
    if phin < phid:
        phin += _PI2
    phdpmx = 0.5 * min(np.pi, phin - phid)
    phnpmx = 0.5 * min(np.pi, phid - phin + _PI2)
    phnmmx, phdmmx = phdpmx, phnpmx

    offc, dskofc = p.offc_deg * d2r, p.dskofc_deg * d2r
    ofdc = np.hypot(offc, dskofc)
    cosofc, sinofc = np.cos(ofdc), np.sin(ofdc)
    aslonc = np.arcsin(dskofc / ofdc) if ofdc > 0.0 else 0.0

    sinth0 = np.sin(theta0)
    sinthr1 = np.sin(theta0 + 0.1972)   # "average AMIE results show r1 = -2.6 for 11.3 deg beyond theta0"
    psi = [psie, psie, psim, psim, psie, psie, psim, psim]

    # Transform to auroral circle coordinates.
    sinlat, coslat = np.sin(np.abs(mlat)), np.cos(mlat)
    sinlon, coslon = np.sin(dlon + aslonc), np.cos(dlon + aslonc)
    colat = np.arccos(np.clip(cosofc * sinlat - sinofc * coslat * coslon, -1.0, 1.0))
    alon = np.mod(np.arctan2(sinlon * coslat, sinlat * sinofc + cosofc * coslat * coslon) - aslonc
                  + 3.0 * np.pi, _PI2) - np.pi

    # Boundaries for the longitudinal function.
    wk1 = ((colat - theta0) / theta0) ** 2
    phi4 = phid + eps - np.minimum(p.phidm0_deg * d2r + wk1 * (pih - p.phidm0_deg * d2r), phdmmx)
    phi5 = phid - eps + np.minimum(p.phidp0_deg * d2r + wk1 * (pih - p.phidp0_deg * d2r), phdpmx)
    phi6 = phin + eps - np.minimum(p.phinm0_deg * d2r + wk1 * (pih - p.phinm0_deg * d2r), phnmmx)
    phi7 = phin - eps + np.minimum(p.phinp0_deg * d2r + wk1 * (pih - p.phinp0_deg * d2r), phnpmx)
    phi = [phi5 - _PI2, phi6 - _PI2, phi7 - _PI2, phi4, phi5, phi6, phi7, phi4 + _PI2]

    phirc = 0.0   # ring-current rotation
    wk2 = np.mod(alon + phirc + 2.0 * _PI2 + np.pi, _PI2) - np.pi
    wk3 = np.mod(alon + phirc + 3.0 * _PI2, _PI2) - np.pi

    def longitudinal(wk):
        out = np.zeros_like(wk)
        for n in range(7):
            a, b = phi[n], phi[n + 1]
            inside = np.where((wk - a) * (wk - b) < 0.0, 2.0, 0.0)   # 1 - sign(1, product)
            out += 0.25 * (psi[n] + psi[n + 1] + (psi[n] - psi[n + 1]) * np.cos(np.pi * (wk - a) / (b - a))) * inside
        return out

    phifun = longitudinal(wk2)
    phifn2 = longitudinal(wk3)

    x = colat / theta0
    p_in = ((2.0 * (pcen - phifun) + (phifun - phifn2) * 0.75) * x ** 3
            + (1.5 * (phifun + phifn2) - 3.0 * pcen) * x ** 2 + 0.75 * (phifun - phifn2) * x + pcen)
    s = np.sin(colat)
    p_out = (phifun * (np.maximum(s, sinth0) / sinth0) ** p.rr1
             * np.exp(7.0 * (1.0 - np.maximum(s, sinthr1) / sinthr1)))
    pot = np.where(colat < theta0, p_in, p_out)
    return np.where(np.abs(mlat) > np.pi / 6.0, pot, 0.0)   # potm(): zero within 30 deg of the equator


@dataclass
class HeelisDrift:
    """
    environment.DriftModel: high-latitude E x B convection from the Heelis potential above. ctpoten_kV is the
    cross-polar-cap potential (TIE-GCM reads it from input files or a namelist constant; WBMOD derives it from
    Kp internally, not public) -- 30 kV is quiet-to-moderate. B_field: callable(x, y, z) -> Tesla (e.g.
    igrf.IgrfField.B_vec_xyz_m); None uses the IGRF centred dipole for the date.
    """

    ctpoten_kV: float = 30.0
    sigma_v_frac: float = 0.2
    B_field: object = None
    h_grad_m: float = 10e3
    params: HeelisParams = field(default_factory=HeelisParams)

    def potential_V(self, r_ecef, t: datetime) -> np.ndarray:
        mlat, _ = geomagnetic_lat_lon_rad(r_ecef, t)
        return 1e3 * heelis_potential_kV(mlat, _dlon_rad(r_ecef, t), self.ctpoten_kV, self.params)

    def at(self, r_ecef, t) -> dict:
        r = np.atleast_2d(np.asarray(r_ecef, dtype=float))
        E = _grad_E_field(lambda rr: self.potential_V(rr, t), r, self.h_grad_m)
        B = (np.asarray(self.B_field(r[:, 0], r[:, 1], r[:, 2]), dtype=float) if self.B_field is not None
             else centred_dipole_B(r, t))
        v_d = np.cross(E, B) / np.maximum(np.sum(B * B, axis=-1), 1e-30)[:, None]
        return dict(v_d=v_d, sigma_v=self.sigma_v_frac * np.linalg.norm(v_d, axis=-1))


# ---------------------------------------------------------------------------
# Mid/low latitude: our own crude pattern (see module docstring)
# ---------------------------------------------------------------------------

@dataclass
class RichmondDrift:
    """
    environment.DriftModel, OUR OWN crude stand-in for Richmond-NCAR (not findable; see module docstring):
    magnetic-eastward zonal drift, night_mps at magnetic midnight, day_mps at noon, cosine in MLT between, at
    full strength equatorward of taper_deg[0] magnetic latitude and zero poleward of taper_deg[1].
    """

    night_mps: float = 125.0        # midpoint of HiCIRF's own 100-150 m/s nighttime equatorial eastward drift
    day_mps: float = -20.0          # weaker and westward by day
    sigma_v_frac: float = 0.2
    taper_deg: tuple = (40.0, 60.0)

    def at(self, r_ecef, t) -> dict:
        r = np.atleast_2d(np.asarray(r_ecef, dtype=float))
        mlat, _ = geomagnetic_lat_lon_rad(r, t)
        mlt = magnetic_local_time_hours(r, t)
        day_weight = 0.5 * (1.0 - np.cos(2.0 * np.pi * mlt / 24.0))   # 0 at midnight, 1 at noon
        speed = self.day_mps * day_weight + self.night_mps * (1.0 - day_weight)
        lo, hi = self.taper_deg
        u = np.clip((np.rad2deg(np.abs(mlat)) - lo) / (hi - lo), 0.0, 1.0)
        speed = speed * 0.5 * (1.0 + np.cos(np.pi * u))

        r_hat = r / np.linalg.norm(r, axis=-1, keepdims=True)
        east = np.cross(dipole_axis_ecef(t), r_hat)          # magnetic east
        n = np.linalg.norm(east, axis=-1, keepdims=True)
        east = east / np.where(n > 1e-12, n, 1.0)
        v_d = speed[:, None] * east
        return dict(v_d=v_d, sigma_v=self.sigma_v_frac * np.abs(speed))


@dataclass
class WBMODStyleDrift:
    """environment.DriftModel: v = v_Heelis + v_Richmond (OUR OWN crude rule; see module docstring)."""

    heelis: HeelisDrift = field(default_factory=HeelisDrift)
    richmond: RichmondDrift = field(default_factory=RichmondDrift)

    def at(self, r_ecef, t) -> dict:
        r = np.atleast_2d(np.asarray(r_ecef, dtype=float))
        h = self.heelis.at(r, t)
        lo = self.richmond.at(r, t)
        return dict(v_d=h["v_d"] + lo["v_d"], sigma_v=np.hypot(h["sigma_v"], lo["sigma_v"]))
