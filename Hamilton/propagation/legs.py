"""
legs.py -- deterministic leg parameters and the power budget (spec §4.5, §6).

Landing Jacobian and flux-tube distance (§4.5, decision D2):

    J_beta = d r_L / d beta,   J_alpha = d r_L / d alpha     (landing point per launch angle)
    A      = | J_beta x J_alpha |                              [m^2 / rad^2]
    d^2    = A sin(psi) / cos(beta)                           eq. (4.7)

with psi the grazing angle at the landing point and beta the launch elevation. The
focusing gain is G_f = 10 log10(P'^2 / d^2), with P' the group path (decision D2: P', as
the spec says). Caustic cap: G_f <= G_f,max (default 10 dB), else d^2 is raised to
P'^2 10^(-G_f,max/10) and caustic_flag is set (spec §4.5).

Jacobians come from central differences of landing points of a five-ray mini-fan,
which is valid for multi-hop legs too (spec pitfall: the tangent map is not used across
bounces). The landing point is the ground crossing that ends the leg's last hop.

Leg records (§6.1-6.2): leg_mode() turns a landing result into a LegMode (delay P'/c, phase
-k0 P, Doppler from eq. 4.8, directions, absorption, ground loss, d_eff, caustic flag).
Power (§6.3-6.4): one_way_power_dB() is eq. (6.2)-(6.3) for a one-way link. There is no
coherence term (spec §6.3 pitfall).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from raytracing.batch import BatchTracer
from core.constants import C
from core.geometry import ellipsoid_outward_normal_m
from core.records import LegMode

DEG = np.pi / 180.0
G_FOCUS_MAX_DB = 10.0          # caustic cap, spec §4.5
L_POL_DB = 3.01                # §6.4 baseline: half the power into each characteristic mode
VALLEY_M = 180e3               # E-F valley altitude for the placeholder E/F split (§5.4) [m]


def launch_direction(elev_deg: float, az_deg: float, up: np.ndarray, north: np.ndarray,
                     east: np.ndarray) -> np.ndarray:
    """Unit launch direction k(beta, alpha) of spec §2.2 (same convention as Link._direction)."""
    e, a = elev_deg * DEG, az_deg * DEG
    horiz = np.cos(a) * north + np.sin(a) * east
    d = np.cos(e) * horiz + np.sin(e) * up
    return d / np.linalg.norm(d)


def valley_altitude_m(ne_column, h_m, *, e_range=(80e3, 150e3), f_range=(150e3, 500e3)):
    """
    E-F valley of one density column (spec §5.4): the altitude of the minimum between the E peak
    (searched in e_range) and the F peak (searched in f_range). Returns None if the column has no E
    peak, no F peak, or no local minimum between them (then the caller keeps the default).
    """
    ne = np.asarray(ne_column, dtype=float)
    h = np.asarray(h_m, dtype=float)
    interior = np.zeros(ne.size, dtype=bool)
    interior[1:-1] = (ne[1:-1] >= ne[:-2]) & (ne[1:-1] >= ne[2:])     # local maxima (a peak, not a boundary)
    ie = np.flatnonzero(interior & (h >= e_range[0]) & (h <= e_range[1]))
    jf = np.flatnonzero(interior & (h >= f_range[0]) & (h <= f_range[1]))
    if ie.size == 0 or jf.size == 0:
        return None
    iE = ie[np.argmax(ne[ie])]
    iF = jf[np.argmax(ne[jf])]
    if iF <= iE + 1:
        return None
    seg = np.arange(iE + 1, iF)
    iv = seg[np.argmin(ne[seg])]
    if not (ne[iv] < ne[iE] and ne[iv] < ne[iF]):
        return None
    return float(h[iv])


def layer_letter(apex_m: float, valley_m: float = VALLEY_M) -> str:
    """Reflecting layer of one hop from its apex height (§5.4): E below the valley, else F."""
    return "E" if apex_m < valley_m else "F"


def flux_tube_distance2(J_beta: np.ndarray, J_alpha: np.ndarray, psi: float, beta: float) -> tuple:
    """
    Eq. (4.7): d^2 = A sin(psi)/cos(beta), A = |J_beta x J_alpha|. Returns (d2, A).
    J vectors in metres per radian; beta in radians (the launch elevation).
    """
    A = float(np.linalg.norm(np.cross(J_beta, J_alpha)))
    return A * np.sin(psi) / np.cos(beta), A


def focusing(d2: float, P_group: float, *, g_max_db: float = G_FOCUS_MAX_DB):
    """
    G_f = 10 log10(P'^2/d^2) with the caustic cap. Returns (G_f, d2_used, caustic_flag).
    """
    if d2 <= 0.0 or P_group <= 0.0:
        return None, d2, False
    G = 10.0 * np.log10(P_group * P_group / d2)
    if G > g_max_db:
        d2_cap = P_group * P_group * 10.0 ** (-g_max_db / 10.0)
        return g_max_db, d2_cap, True
    return float(G), d2, False


def one_way_power_dB(Pt_dBW: float, Gt_dB: float, Gr_dB: float, f0: float, d_eff: float,
                     loss_dB: float) -> float:
    """
    Eq. (6.2)-(6.3), one-way link: P_r = P_t G_t G_r lambda0^2 / ((4 pi)^2 d^2 L), with
    10 log10 L = loss_dB (absorption + ground + polarization + other). Returns dBW.
    """
    lam = C / f0
    return (Pt_dBW + Gt_dB + Gr_dB + 20.0 * np.log10(lam) - 20.0 * np.log10(4.0 * np.pi)
            - 20.0 * np.log10(d_eff) - loss_dB)


@dataclass
class LandingResult:
    """Landing geometry and leg quantities of one ray and its neighbours (mini-fan)."""

    r_land: np.ndarray        # (3,) landing point of the central ray
    J_beta: np.ndarray        # (3,) m/rad
    J_alpha: np.ndarray       # (3,) m/rad
    psi: float                # grazing angle at landing [rad]
    beta: float               # launch elevation [rad]
    alpha: float              # launch azimuth [rad]
    P_group: float            # group path of the central ray [m]
    landed: bool
    d_eff: float              # flux-tube distance [m] (after the caustic cap)
    G_focus_db: float | None
    caustic: bool
    P_phase: float = float("nan")      # phase path of the central ray [m]
    A_db: float = float("nan")         # one-way absorption of the central ray [dB]
    A_D_db: float = float("nan")       # part of it from heights below 120 km [dB]
    ground_loss_db: float = 0.0        # sum of Bounce loss_dB along the leg [dB]
    doppler_hz: float = float("nan")   # ionospheric Doppler of the leg, eq. (4.8) [Hz]
    s_hat: np.ndarray | None = None    # physical incident direction at the landing point
    e_site: np.ndarray | None = None   # launch direction at the site (§6.2)
    n_hops: int = 1
    label: str = ""                    # e.g. 1F-low-O (§5.4 labels, from the mini-fan)
    att_max: float = float("nan")     # max over steps of k0 |Im n| lambda0 (spec §7.3)


def landing_mini_fan(tracer: BatchTracer, r0: np.ndarray, elev_deg: float, az_deg: float,
                     up: np.ndarray, north: np.ndarray, east: np.ndarray, max_path_m: float, *,
                     nhops: int = 1, delta_deg: float = 0.01, ceiling_alt_m: float = 800e3,
                     valley_m: float = VALLEY_M) -> LandingResult:
    """
    Trace the central ray and four neighbours (elevation and azimuth +/- delta_deg), take
    the landing points of the last hop, and form J_beta, J_alpha by central differences.
    A central ray that does not land returns landed=False with NaN geometry.
    """
    d = np.deg2rad(delta_deg)
    dirs = np.stack([
        launch_direction(elev_deg, az_deg, up, north, east),
        launch_direction(elev_deg + delta_deg, az_deg, up, north, east),
        launch_direction(elev_deg - delta_deg, az_deg, up, north, east),
        launch_direction(elev_deg, az_deg + delta_deg, up, north, east),
        launch_direction(elev_deg, az_deg - delta_deg, up, north, east),
    ])
    res = tracer.trace(r0, dirs, max_path_m, nhops=nhops, ceiling_alt_m=ceiling_alt_m,
                       want_group=True, want_absorption=True, bounce_models=True)
    beta, alpha = np.deg2rad(elev_deg), np.deg2rad(az_deg)
    if not res.landed[0] or res.hops[0] != nhops:
        nan3 = np.full(3, np.nan)
        return LandingResult(r_land=nan3, J_beta=nan3, J_alpha=nan3, psi=float("nan"), beta=beta,
                             alpha=alpha, P_group=float("nan"), landed=False, d_eff=float("nan"),
                             G_focus_db=None, caustic=False, n_hops=nhops)
    # last ground contact of each ray (the landing point of the final hop)
    last = [b[-1] if len(b) else np.full(3, np.nan) for b in res.bounce_points]
    r_c, r_bp, r_bm, r_ap, r_am = (np.asarray(x, float) for x in last)
    J_beta = (r_bp - r_bm) / (2.0 * d)
    J_alpha = (r_ap - r_am) / (2.0 * d)
    # grazing angle of the central ray at the landing point: the incident physical direction
    s_in = tracer._direction(np.concatenate([res.r_final[0], res.u_final[0]]))
    n_land = ellipsoid_outward_normal_m(r_c)
    psi = float(np.arcsin(np.clip(-np.dot(s_in, n_land), -1.0, 1.0)))
    d2, _ = flux_tube_distance2(J_beta, J_alpha, psi, beta)
    G, d2_used, caustic = focusing(d2, float(res.group_path[0]))
    # layer sequence and low/high sign from the apex heights and the elevation pair (§5.4)
    layers = "".join(layer_letter(a, valley_m) for a in res.hop_apex[0])
    dD = np.linalg.norm(r_bp - r0) - np.linalg.norm(r_bm - r0)     # sign of dD/dbeta (low < 0)
    mode = "O" if tracer.ray.MODE > 0 else "X"
    label = f"{nhops}{layers}-{'low' if dD < 0 else 'high'}-{mode}"
    bounce_loss = float(sum(b.loss_dB for b in res.bounces[0]))
    doppler = float(res.doppler_hz[0]) if res.doppler_hz is not None else float("nan")
    return LandingResult(r_land=r_c, J_beta=J_beta, J_alpha=J_alpha, psi=psi, beta=beta, alpha=alpha,
                         P_group=float(res.group_path[0]), landed=True,
                         d_eff=float(np.sqrt(d2_used)), G_focus_db=G, caustic=caustic,
                         P_phase=float(res.phase_path[0]), A_db=float(res.absorption_db[0]),
                         A_D_db=float(res.absorption_D_db[0]),
                         att_max=float(res.att_per_wavelength_max[0]) if res.att_per_wavelength_max is not None else float("nan"),
                         ground_loss_db=bounce_loss, doppler_hz=doppler, s_hat=s_in,
                         e_site=launch_direction(elev_deg, az_deg, up, north, east), n_hops=nhops,
                         label=label)


def leg_mode(land: LandingResult, tracer: BatchTracer, *, site: str) -> LegMode:
    """
    LegMode of one leg (spec Listing 1, §6.1-6.2) from its landing result. absorption_D_dB is NaN
    until the split of the absorption below 120 km is available (Phase 2, §7).
    """
    mode = "O" if tracer.ray.MODE > 0 else "X"
    return LegMode(site=site, mode=mode, label=land.label, n_hops=land.n_hops,
                   beta=land.beta, alpha=land.alpha, e_site=land.e_site, k_far=land.s_hat,
                   grazing_far=land.psi, Pg=land.P_group, P=land.P_phase,
                   delay=land.P_group / C, doppler=land.doppler_hz, absorption_dB=land.A_db,
                   absorption_D_dB=land.A_D_db, ground_loss_dB=land.ground_loss_db,
                   d_eff=land.d_eff, caustic_flag=land.caustic,
                   xmode_validated=False if mode == "X" else None,
                   attenuation_flag=bool(land.att_max > 0.1))
