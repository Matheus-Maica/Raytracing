"""
targets.py -- point targets: homing of the outbound and return legs and two-way components
(spec §5.2, §5.3, §5.5, §12.3).

Each site (Tx, Rx) traces a launch fan. Algorithm 2 (fans.invert_fan) finds every launch angle whose hop-h
landing point is the target; Algorithm 3 (homing.newton_home) refines each to the target to 50 m. The leg
record is built at the homed angles (legs.landing_mini_fan, legs.leg_mode) with the corrected P and P'.
Every (outbound, return) pair of homed legs gives one two-way component (twoway.target_component, §12.3).

Ground targets only: a target at altitude h_T (aircraft, §5.3) needs a crossing event at h_T in the tracer,
not done yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from raytracing import batch
from propagation import fans
from propagation import homing
from stochastic import leg_screens
from propagation import legs
from echoes import twoway
from core.constants import C
from core.geometry import latlon_to_xyz_m
from core.records import LegMode

DEFAULT_ELEV_GRID = np.arange(10.0, 40.01, 0.5)
DEFAULT_AZ_GRID = np.linspace(-1.0, 1.0, 5)


@dataclass
class HomedLeg:
    """One homed leg of a site to the target point, for one mode and hop count."""

    leg: LegMode
    site: str
    hop: int                       # hop count
    converged: bool
    residual_m: float
    flags: list = field(default_factory=list)
    samples: object = None         # RaySamples of the homed leg (for Gamma_or)
    screens: object = None         # ScreenSet of the homed leg (leg_screens.screens_for_leg), or None


def _site_frame(lat_deg: float, lon_deg: float, alt_m: float):
    r0 = latlon_to_xyz_m(lat_deg, lon_deg, alt_m)
    up = r0 / np.linalg.norm(r0)
    lon = np.radians(lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(up, east)
    return r0, up, north, east


def home_site_to_point(tracer, site: str, site_lat: float, site_lon: float, site_alt: float,
                       target_lat: float, target_lon: float, *, max_hops: int = 1,
                       elev_grid=DEFAULT_ELEV_GRID, az_grid=DEFAULT_AZ_GRID,
                       max_path_m: float = 3000e3, ceiling_alt_m: float = 800e3,
                       valley_m: float = legs.VALLEY_M, eps_m: float = 50.0,
                       target_alt_m: float = 0.0) -> list[HomedLeg]:
    """
    All homed legs from one site to a ground target, for hop counts 1..max_hops. Returns one HomedLeg per
    converged candidate (duplicates after homing removed). Candidates that do not converge are kept with
    converged=False and their flags, so the caller can see them.
    """
    r0, up, north, east = _site_frame(site_lat, site_lon, site_alt)
    if abs(tracer.landing_alt_m - target_alt_m) > 1e-9:
        raise ValueError("the tracer's landing_alt_m must equal target_alt_m (build BatchTracer(..., landing_alt_m=h))")
    target = latlon_to_xyz_m(target_lat, target_lon, target_alt_m)
    fan = fans.trace_fan(tracer, r0, up, north, east, elev_grid, az_grid, max_path_m, nhops=max_hops,
                         grid=True, ceiling_alt_m=ceiling_alt_m)

    def subdivision_rays(e, a):
        return fans.trace_fan(tracer, r0, up, north, east, e, a, max_path_m, nhops=max_hops,
                              grid=False, ceiling_alt_m=ceiling_alt_m)

    target_xy = fans.site_planar(r0, target)
    out: list[HomedLeg] = []
    for hop in range(max_hops):
        cands = fans.invert_fan(fan, hop, target_xy, angle_tracer=subdivision_rays, valley_m=valley_m)
        if not cands:
            out.append(HomedLeg(leg=None, site=site, hop=hop + 1, converged=False, residual_m=float("nan"),
                                flags=["no launch angle of the fan reaches the target: widen elev_grid"]))
            continue
        seen = []
        for c in cands:
            homed = homing.newton_home(tracer, r0, up, north, east, c.elev_deg, c.az_deg, target,
                                       hop=hop + 1, eps_m=eps_m, max_path_m=max_path_m,
                                       ceiling_alt_m=ceiling_alt_m)
            if any(abs(homed.elev_deg - e) < fans.DEDUP_DEG and abs(homed.az_deg - a) < fans.DEDUP_DEG
                   for e, a in seen):
                continue
            seen.append((homed.elev_deg, homed.az_deg))
            if not homed.converged:
                out.append(HomedLeg(leg=None, site=site, hop=hop + 1, converged=False,
                                    residual_m=homed.residual_m, flags=list(homed.flags)))
                continue
            land = legs.landing_mini_fan(tracer, r0, homed.elev_deg, homed.az_deg, up, north, east,
                                         max_path_m, nhops=hop + 1, ceiling_alt_m=ceiling_alt_m,
                                         valley_m=valley_m)
            leg = legs.leg_mode(land, tracer, site=site)
            leg.Pg = homed.Pg                               # corrected for the residual miss (Algorithm 3)
            leg.P = homed.P
            leg.delay = homed.Pg / C
            d_hat = legs.launch_direction(homed.elev_deg, homed.az_deg, up, north, east)
            rec = tracer.trace(r0, d_hat[None], max_path_m, nhops=hop + 1, ceiling_alt_m=ceiling_alt_m,
                               want_group=True, want_absorption=True, record_states=True)
            samples = batch.samples_from_record(tracer, rec.records[0], transport=True)
            try:
                screens = leg_screens.screens_for_leg(samples, tracer.ray)
            except ValueError as exc:
                screens = None
                homed.flags.append(f"no screens for Gamma_or: {exc}")
            leg.samples = samples
            out.append(HomedLeg(leg=leg, site=site, hop=hop + 1, converged=True,
                                residual_m=homed.residual_m, flags=list(homed.flags),
                                samples=samples, screens=screens))
    return out


def target_components(tx_legs: list[HomedLeg], rx_legs: list[HomedLeg], *, f0: float, v_T,
                      sigma_T_m2: float, Pt_dBW: float, Gt_dB: float = 0.0, Gr_dB: float = 0.0):
    """
    One two-way component per (outbound, return) pair of converged legs with the same carrier (§5.5, §12.3).
    The incident direction is the outbound leg's direction at the target, and the scattered direction toward
    the receiver is minus the return leg's direction there, so f_T follows eq. (6.1) in both the monostatic
    and the bistatic case.
    """
    comps = []
    for o in tx_legs:
        if not o.converged:
            continue
        for r in rx_legs:
            if not r.converged:
                continue
            comp = twoway.target_component(o.leg, r.leg, f0=f0, v_T=v_T, k_i=o.leg.k_far,
                                           s_f=r.leg.k_far, sigma_T_m2=sigma_T_m2, Pt_dBW=Pt_dBW,
                                           Gt_dB=Gt_dB, Gr_dB=Gr_dB)
            if o.screens is not None and o.samples is not None and r.samples is not None:
                comp.gamma_or = twoway.gamma_or_legs(o.samples, r.samples, o.screens)   # eq. (12.3)
            comps.append(comp)
    return comps


def component_track(comp, t) -> tuple[np.ndarray, float]:
    """
    §12.3 range migration over a CIT: the delay drifts linearly, tau(t) = tau_0 + tau_dot t, and the Doppler
    of the component is constant in this model. Returns (delay [s] at times t, Doppler [Hz]).
    """
    t = np.asarray(t, dtype=float)
    tau_dot = comp.scatterer["range_migration_s_per_s"]
    return comp.delay + tau_dot * t, comp.doppler
