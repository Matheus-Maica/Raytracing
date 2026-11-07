"""
meteor.py -- sporadic meteor clutter (HiCIRF paragraph 34; plan Stage 5): "a meteor clutter model for
HiCIRF has recently been developed... using the Meteor Input Function model (MIF)... meteor mass, speed,
and direction as a function of geographic location and local time of day."

Two physics pieces, from two different sources, both read directly (not from secondary summaries):

1. The MIF source/radiant model: Table 1 of Fentzke, Janches & Sparks (2009, J. Atmos. Sol. Terr. Phys.
   71, 653-661), read as a page image. Five sporadic radiant sources (apex, helion, anti-helion, north and
   south toroidal -- apex already combines the north+south apex sub-sources by symmetry), each a Gaussian
   patch in sun-centred ecliptic coordinates with its own geocentric-speed distribution (a two-Gaussian
   mixture for apex, single Gaussian for the rest). RADIANT_SOURCES below IS this table, transcribed
   directly -- real, cited data, not a placeholder. SOURCE_WEIGHT (each source's share of the global flux
   at 1 AU) is from the companion paper, Fentzke and Janches (2008, J. Geophys. Res. 113, A03304), Table 3
   and paragraph 27 ("33% of the counts for a given mass bin are assigned to the Apex source, 22% to the
   Helion, 22% to the Anti-Helion, 11.5% to the North Toroidal, and 11.5% to the South Toroidal") -- the
   INPUT weights, before atmospheric/radar visibility. Visibility is applied here by sample_meteors()'s
   own horizon test, playing the role of their model's geometry stage.

2. Underdense-trail radar backscatter: eqs. (37), (48)-(50) of a 2022 paper using classical meteor-radar
   theory (McKinley, 1961) for an unrelated purpose (dark-matter constraints from meteor radars,
   arXiv:2209.07690), read directly as page images because they state the classical formulas cleanly with
   every symbol defined. underdense_trail_rcs() and diffusion_decay_factor() below are eqs. (50) and (49)
   transcribed exactly (eq. 50 generalised back to an explicit range R0, undoing that paper's own
   substitution R0 = h/sin(theta) for its vertical-looking radars, since Hamilton's geometry is bistatic).
   Deliberately NOT implemented (requirement dropped by the user, 2026-11-07): the ablation/ionisation
   model (Whipple 1950, 1951; McKinley, 1961; Bronshten, 1983) that would convert a meteoroid's mass, speed
   and height into its electron line density q_e and initial trail radius r0. q_e is drawn log-uniformly
   over the underdense range and r0 is one fixed value, by design. The ambipolar diffusion coefficient
   D_a(h) = 0.025 exp(0.154 h/km) cm^2/s is the user-supplied profile (2026-11-07).

Geometry: a meteor trail's visibility (whether its radiant is above the local horizon at a given site,
time and source) reuses the same low-precision ecliptic/obliquity machinery as
plasma_drift.subsolar_point_deg, extended with a standard ecliptic-to-equatorial and
equatorial-to-topocentric (via Greenwich Mean Sidereal Time) conversion -- all standard, textbook
low-precision formulas, not from either source paper.

Scene assembly (meteor_components): uses straight-line (great-circle/chord) ranges from the sites to each
trail, NOT ray-traced ionospheric paths -- unlike clutter_scene.py/auroral_scene.py, which find real
traced-ray landing points first. Meteor trails sit in the 70-130 km range (Fentzke et al.'s own stated
typical detection range), close to where Hamilton's own E-region-reflected rays already pass, so fan-based
ray intersection (the same pattern auroral_scene.py uses) is the natural next refinement, not done here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from echoes import targets
from raytracing.batch import BatchTracer
from core.constants import C
from core.geometry import latlon_to_xyz_m, xyz_m_to_latlon
from geophysics.plasma_drift import subsolar_point_deg
from core.records import Component
from echoes.twoway import radar_power_dBW

R_ELECTRON_M = 2.8179403262e-15          # classical electron radius [m]
SIGMA_E_M2 = 4.0 * np.pi * R_ELECTRON_M ** 2   # single-electron Thomson backscatter cross section [m^2]

# Table 1 of Fentzke, Janches & Sparks (2009): real, transcribed data (see module docstring).
# v/sigma in km/s; lam (sun-centred ecliptic longitude offset), beta and their widths in degrees.
RADIANT_SOURCES = {
    "apex":           dict(A=(0.8, 0.2), v=(55.0, 17.3), sigma=(2.5, 2.60),
                           lam=270.0, beta=0.0, s_lam=19.0, s_beta=32.0),
    "helion":         dict(A=(0.0, 1.0), v=(0.0, 30.0), sigma=(0.0, 2.57),
                           lam=350.0, beta=0.0, s_lam=8.0, s_beta=20.0),
    "anti_helion":    dict(A=(0.0, 1.0), v=(0.0, 30.0), sigma=(0.0, 2.57),
                           lam=190.0, beta=0.0, s_lam=8.0, s_beta=20.0),
    "north_toroidal": dict(A=(0.0, 1.0), v=(0.0, 30.0), sigma=(0.0, 2.57),
                           lam=270.0, beta=60.0, s_lam=17.0, s_beta=17.0),
    "south_toroidal": dict(A=(0.0, 1.0), v=(0.0, 30.0), sigma=(0.0, 2.57),
                           lam=270.0, beta=-60.0, s_lam=17.0, s_beta=17.0),
}

# Global flux share of each source at 1 AU: Fentzke and Janches (2008), Table 3 / paragraph 27.
SOURCE_WEIGHT = {"apex": 0.33, "helion": 0.22, "anti_helion": 0.22,
                 "north_toroidal": 0.115, "south_toroidal": 0.115}

# By design (ablation model dropped): the underdense/overdense boundary q_tr ~ 1e14 m^-1 (Kaiser & Closs
# 1952-era convention) brackets the log-uniform q_e draw; r0 is one fixed initial trail radius. The names
# keep their PLACEHOLDER_ prefix because existing code and tests import them under these names.
PLACEHOLDER_QTR_M1 = 1.0e14
PLACEHOLDER_R0_M = 0.5

# Ambipolar diffusion coefficient, D_a(h) = 0.025 exp(0.154 h/km) cm^2/s (user-supplied, 2026-11-07).
DA_COEFF_CM2S = 0.025
DA_RATE_PER_KM = 0.154


# ---------------------------------------------------------------------------
# Radiant geometry (ecliptic -> equatorial -> topocentric)
# ---------------------------------------------------------------------------

def _obliquity_rad(t: datetime) -> float:
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    n = (t - j2000).total_seconds() / 86400.0
    return np.deg2rad(23.439 - 0.0000004 * n)


def _gmst_deg(t: datetime) -> float:
    """Greenwich Mean Sidereal Time [deg], standard low-precision formula (not from either source paper)."""
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    d = (t - j2000).total_seconds() / 86400.0
    return (280.46061837 + 360.98564736629 * d) % 360.0


def ecliptic_to_equatorial_deg(lam_deg, beta_deg, t: datetime):
    """Geocentric ecliptic (lam, beta) [deg] -> equatorial (RA, Dec) [deg], standard rotation by the
    obliquity of the date."""
    eps = _obliquity_rad(t)
    lam, beta = np.deg2rad(lam_deg), np.deg2rad(beta_deg)
    sin_dec = np.sin(beta) * np.cos(eps) + np.cos(beta) * np.sin(eps) * np.sin(lam)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    ra = np.arctan2(np.sin(lam) * np.cos(eps) - np.tan(beta) * np.sin(eps), np.cos(lam))
    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def topocentric_elevation_deg(ra_deg, dec_deg, lat_deg, lon_deg, t: datetime):
    """Equatorial (RA, Dec) at site (lat, lon) and time t -> elevation above the local horizon [deg]."""
    lst = (_gmst_deg(t) + np.asarray(lon_deg, dtype=float)) % 360.0
    H = np.deg2rad((lst - np.asarray(ra_deg, dtype=float)) % 360.0)
    lat, dec = np.deg2rad(np.asarray(lat_deg, dtype=float)), np.deg2rad(np.asarray(dec_deg, dtype=float))
    sin_el = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H)
    return np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0)))


def radiant_elevation_deg(source: str, lam_offset_deg, beta_deg, lat_deg, lon_deg, t: datetime):
    """A source's radiant direction (lam_offset from the sun, beta) -> its elevation [deg] at a site/time.
    The radiant is where meteors APPEAR to come FROM; a source below the horizon cannot deliver meteors
    there (the geometric visibility constraint -- see the module docstring re the atmospheric/instrumental
    refinement this doesn't include)."""
    _, lon_sun_geo, _ = subsolar_point_deg(t)
    # geocentric ecliptic longitude of the sun ~ its subsolar geographic longitude plus GMST-based offset
    # cancels out in the (lam_sun + offset) - RA - LST chain; simpler and equally standard: use the solar
    # ecliptic longitude lam_sun directly from the same low-precision solar position used for lam_sun_geo.
    from geophysics.plasma_drift import subsolar_point_deg as _ssp  # noqa: F401  (reused for clarity, not re-imported)
    n = (t - datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / 86400.0
    L = np.deg2rad((280.460 + 0.9856474 * n) % 360.0)
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)
    lam_sun = np.rad2deg(L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2.0 * g)) % 360.0
    lam = (lam_sun + lam_offset_deg) % 360.0
    ra, dec = ecliptic_to_equatorial_deg(lam, beta_deg, t)
    return topocentric_elevation_deg(ra, dec, lat_deg, lon_deg, t)


# ---------------------------------------------------------------------------
# Underdense-trail radar backscatter (eqs. 48-50, read directly -- see module docstring)
# ---------------------------------------------------------------------------

def underdense_trail_rcs(q_e_m1, wavelength_m: float, r0_m: float = PLACEHOLDER_R0_M,
                         sigma_e_m2: float = SIGMA_E_M2) -> float:
    """
    Eq. (50) (generalised back to an explicit range R0; see module docstring), with the R0 factor left
    out of the RETURNED cross section and instead folded into the radar equation by the caller via
    twoway.radar_power_dBW's own d_T/d_R -- so this returns lambda*sigma_e*q_e^2*exp(-8 pi^2 r0^2/lambda^2)
    / 2 [m], the part of eq. (50) that is independent of range, to be multiplied by an effective range
    exactly once (not twice, as a naive sigma0-style area would) when assembling the power budget. Valid
    for underdense trails only (q_e <= q_tr ~ 1e14 m^-1, PLACEHOLDER_QTR_M1).
    """
    q_e = np.asarray(q_e_m1, dtype=float)
    return 0.5 * wavelength_m * sigma_e_m2 * q_e ** 2 * np.exp(-8.0 * np.pi ** 2 * r0_m ** 2 / wavelength_m ** 2)


def diffusion_decay_factor(t_since_formation_s, D_a_m2s: float, wavelength_m: float) -> float:
    """Eq. (49): the underdense echo's power decay from ambipolar diffusion, exp(-32 pi^2 D_a t / lambda^2)."""
    t = np.asarray(t_since_formation_s, dtype=float)
    return np.exp(-32.0 * np.pi ** 2 * D_a_m2s * t / wavelength_m ** 2)


def ambipolar_diffusion_coefficient(h_m) -> np.ndarray:
    """D_a(h) = 0.025 exp(0.154 h/km) cm^2/s, returned in m^2/s (user-supplied profile, 2026-11-07);
    about 2.6 m^2/s at 90 km and 12 m^2/s at 100 km."""
    h_km = np.asarray(h_m, dtype=float) / 1e3
    return DA_COEFF_CM2S * np.exp(DA_RATE_PER_KM * h_km) * 1e-4


# ---------------------------------------------------------------------------
# Realisations and scene assembly
# ---------------------------------------------------------------------------

@dataclass
class MeteorRealization:
    r_ecef: np.ndarray          # trail midpoint, ECEF [m]
    v_dir_ecef: np.ndarray      # unit entry-velocity direction, ECEF (meteor's direction of travel)
    speed_kms: float
    source: str
    q_e_m1: float
    r0_m: float
    alt_m: float


def sample_meteors(n: int, lat0_deg: float, lon0_deg: float, radius_km: float, t: datetime, rng,
                   alt_range_m=(80e3, 120e3), max_tries_factor: int = 20) -> list:
    """
    Draw n meteor realisations with radiants above the local horizon, uniformly over a disk of the given
    radius around (lat0, lon0) and over the altitude range (Fentzke et al.'s own stated typical 70-130 km
    detection range, narrowed slightly here; most sensitive region near 90-100 km per their Fig. 2).
    """
    names = list(RADIANT_SOURCES)
    weights = np.array([SOURCE_WEIGHT[n_] for n_ in names])
    weights /= weights.sum()
    out: list[MeteorRealization] = []
    tries = 0
    while len(out) < n and tries < n * max_tries_factor:
        tries += 1
        name = names[rng.choice(len(names), p=weights)]
        src = RADIANT_SOURCES[name]
        lam_off = src["lam"] + rng.normal(0.0, src["s_lam"])
        beta = src["beta"] + rng.normal(0.0, src["s_beta"])
        el = float(np.atleast_1d(radiant_elevation_deg(name, lam_off, beta, lat0_deg, lon0_deg, t))[0])
        if el <= 0.0:
            continue   # the geometric horizon constraint -- see module docstring
        comp = rng.choice(2, p=[src["A"][0], src["A"][1]]) if src["A"][0] > 0.0 else 1
        speed = max(0.1, rng.normal(src["v"][comp], src["sigma"][comp]))

        r_km = radius_km * np.sqrt(rng.uniform())
        ang = rng.uniform(0.0, 2.0 * np.pi)
        dlat = (r_km * np.cos(ang)) / 111.32
        dlon = (r_km * np.sin(ang)) / (111.32 * max(np.cos(np.deg2rad(lat0_deg)), 1e-3))
        lat, lon = lat0_deg + dlat, lon0_deg + dlon
        alt = rng.uniform(*alt_range_m)
        r_ecef = latlon_to_xyz_m(lat, lon, alt)

        # entry direction: the radiant's own topocentric (az, el) at this realisation's location, reversed
        # (a meteor travels FROM its radiant direction TOWARDS the observer).
        ra, dec = ecliptic_to_equatorial_deg(lam_off, beta, t)
        lst = (_gmst_deg(t) + lon) % 360.0
        H = np.deg2rad((lst - ra) % 360.0)
        lat_r, dec_r = np.deg2rad(lat), np.deg2rad(dec)
        az = np.arctan2(-np.sin(H), np.cos(lat_r) * np.tan(dec_r) - np.sin(lat_r) * np.cos(H))
        el_r = np.arcsin(np.clip(np.sin(lat_r) * np.sin(dec_r) + np.cos(lat_r) * np.cos(dec_r) * np.cos(H),
                                 -1.0, 1.0))
        up = r_ecef / np.linalg.norm(r_ecef)
        north = np.array([-np.sin(lat_r) * np.cos(np.radians(lon)), -np.sin(lat_r) * np.sin(np.radians(lon)),
                          np.cos(lat_r)])
        east = np.cross(up, north) / np.linalg.norm(np.cross(up, north))
        north = np.cross(east, up)
        from_radiant = np.cos(el_r) * (np.cos(az) * north + np.sin(az) * east) + np.sin(el_r) * up
        v_dir = -from_radiant   # travelling towards Earth, i.e. opposite the radiant direction

        q_e = 10.0 ** rng.uniform(10.0, np.log10(PLACEHOLDER_QTR_M1))   # PLACEHOLDER (see module docstring)
        out.append(MeteorRealization(r_ecef=r_ecef, v_dir_ecef=v_dir, speed_kms=float(speed), source=name,
                                     q_e_m1=float(q_e), r0_m=PLACEHOLDER_R0_M, alt_m=float(alt)))
    return out


def meteor_components(realizations: list, r_tx: np.ndarray, r_rx: np.ndarray, *, f0: float,
                      Pt_dBW: float, Gt_dB: float = 0.0, Gr_dB: float = 0.0,
                      loss_T_dB: float = 0.0, loss_R_dB: float = 0.0) -> list:
    """
    One Component per meteor realisation, straight-line (not ray-traced) Tx-trail-Rx geometry -- see
    module docstring. Doppler is the bistatic projection of the entry velocity (negative = approaching,
    matching twoway.target_doppler_hz's convention); delay is (d_T + d_R)/c.
    """
    lam0 = C / f0
    comps = []
    for m in realizations:
        d_T_vec = m.r_ecef - r_tx
        d_R_vec = m.r_ecef - r_rx
        d_T, d_R = float(np.linalg.norm(d_T_vec)), float(np.linalg.norm(d_R_vec))
        k_i, s_f = d_T_vec / d_T, -d_R_vec / d_R   # incident (Tx->trail) and return (Rx->trail) directions
        v_vec = m.speed_kms * 1e3 * m.v_dir_ecef
        f_D = float(-np.dot(k_i + s_f, v_vec) / lam0)

        sigma_half = underdense_trail_rcs(m.q_e_m1, lam0, m.r0_m)   # eq. (50)'s range-independent half
        dBW = radar_power_dBW(Pt_dBW, Gt_dB, Gr_dB, f0, sigma_half * d_T, d_T, d_R, loss_T_dB, loss_R_dB)
        power_W = float(10.0 ** (dBW / 10.0))   # R0 of eq. (50) taken as d_T (quasi-monostatic on the outbound leg)

        comps.append(Component(
            kind="meteor", out=None, ret=None,
            scatterer=dict(source=m.source, q_e_m1=m.q_e_m1, alt_m=m.alt_m, speed_kms=m.speed_kms,
                           r_ecef=m.r_ecef),
            power=power_W, delay=(d_T + d_R) / C, doppler=f_D, phase=float(-2.0 * np.pi / lam0 * (d_T + d_R)),
        ))
    return comps


# ---------------------------------------------------------------------------
# Ray-traced scene assembly (real legs, not straight lines)
# ---------------------------------------------------------------------------

def meteor_components_raytraced(realizations: list, ray, site_tx: dict, site_rx: dict, *, f0: float,
                                Pt_dBW: float, Gt_dB: float = 0.0, Gr_dB: float = 0.0,
                                max_hops: int = 1, alt_bin_m: float = 5e3, max_path_m: float = 1500e3,
                                ds_m: float = 2000.0, elev_grid=None, az_grid=None) -> list:
    """
    Real-ray-traced version of meteor_components(): each trail is homed to from both sites with
    targets.home_site_to_point (Algorithm 2 fan inversion + Algorithm 3 Newton refinement), the same
    machinery targets.py uses for aircraft at a fixed altitude -- so this gets the actual ionospheric
    Doppler, absorption and flux-tube focusing of the homed legs, not a geometric straight line.

    BatchTracer.landing_alt_m is one fixed value per tracer instance (the same constraint targets.py's
    own aircraft-target support has), but meteor trails are drawn over a continuous 80-120 km range
    (sample_meteors). To avoid building one tracer per meteor, trails are grouped into altitude bins of
    width alt_bin_m (default 5 km) and one tracer is built per bin actually populated.

    site_tx, site_rx: dict(lat=.., lon=.., alt=0.0) -- the two sites' geodetic positions.
    Meteors whose Tx or Rx leg does not converge (no reflected/near-direct path reaches that trail at
    that altitude from that site) are silently dropped, exactly as targets.py does for aircraft.

    Returns a list of records.Component(kind="meteor", ...), now with out/ret set to the real homed
    LegMode of each leg (unlike meteor_components(), which leaves them None) and a real gamma_or
    reciprocity diagnostic (twoway.gamma_or_legs via targets.target_components) when computable.
    """
    if elev_grid is None:
        elev_grid = np.arange(5.0, 60.01, 1.0)
    if az_grid is None:
        az_grid = np.linspace(-3.0, 3.0, 7)

    by_bin: dict[float, list] = {}
    for m in realizations:
        b = round(m.alt_m / alt_bin_m) * alt_bin_m
        by_bin.setdefault(b, []).append(m)

    comps: list = []
    for alt_bin, group in by_bin.items():
        tracer = BatchTracer(ray, ds_m=ds_m, landing_alt_m=alt_bin)
        for m in group:
            lat, lon, _ = xyz_m_to_latlon(np.asarray(m.r_ecef))
            lat, lon = float(lat), float(lon)
            tx_legs = targets.home_site_to_point(
                tracer, "tx", site_tx["lat"], site_tx["lon"], site_tx.get("alt", 0.0), lat, lon,
                max_hops=max_hops, elev_grid=elev_grid, az_grid=az_grid, max_path_m=max_path_m,
                target_alt_m=alt_bin)
            if not any(leg.converged for leg in tx_legs):
                continue
            rx_legs = targets.home_site_to_point(
                tracer, "rx", site_rx["lat"], site_rx["lon"], site_rx.get("alt", 0.0), lat, lon,
                max_hops=max_hops, elev_grid=elev_grid, az_grid=az_grid, max_path_m=max_path_m,
                target_alt_m=alt_bin)
            if not any(leg.converged for leg in rx_legs):
                continue

            lam0 = C / f0
            out_leg = next(leg for leg in tx_legs if leg.converged)
            R0 = out_leg.leg.d_eff   # eq. (50)'s range, the quasi-monostatic convention (see module docstring)
            sigma_T_m2 = underdense_trail_rcs(m.q_e_m1, lam0, m.r0_m) * R0
            v_T = tuple(m.speed_kms * 1e3 * m.v_dir_ecef)

            new = targets.target_components(tx_legs, rx_legs, f0=f0, v_T=v_T, sigma_T_m2=sigma_T_m2,
                                            Pt_dBW=Pt_dBW, Gt_dB=Gt_dB, Gr_dB=Gr_dB)
            for c in new:
                c.kind = "meteor"
                c.scatterer.update(source=m.source, q_e_m1=m.q_e_m1, alt_m=m.alt_m,
                                   speed_kms=m.speed_kms, r_ecef=m.r_ecef)
            comps.extend(new)
    return comps
