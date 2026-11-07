"""
link_budget.py -- one-way power budget of a leg (spec §6.2-§6.4, eq. 6.2-6.3).

    P_r = P_t G_t G_r lambda0^2 / ((4 pi)^2 d_eff^2 L),   10 log10 L = A_abs + sum L_g + L_pol + L_other

d_eff is the caustic-capped flux-tube distance of the leg (eq. 4.7, §4.5). The focusing gain
G_f = 10 log10(P'^2 / d_eff^2) is reported for information: it is already inside d_eff. L_pol is
3.01 dB per leg (§6.4). There is no coherence term (spec §6.3 pitfall); the coherent/diffuse split
belongs to the Rician model (§10.6).

Scope: one leg, one mode, from the landing mini-fan of legs.py (decision D2). Gains Gt, Gr are
supplied by the caller (beam patterns; element patterns come with §11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from propagation import legs
from raytracing.batch import BatchTracer
from core.geometry import great_circle_range_m
from core.records import LegMode


@dataclass
class LinkBudgetResult:
    """One-way budget of one leg. Values that need a landed ray are NaN/None otherwise."""

    landed: bool
    leg: Optional[LegMode]
    ground_range_m: float
    delay_s: float                       # P' / c
    doppler_hz: float                    # eq. (4.8), Method B
    focusing_gain_db: Optional[float]    # 10 log10(P'^2/d_eff^2) before the cap, dB (information)
    caustic: bool
    d_eff_m: float
    absorption_db: float
    ground_loss_db: float
    polarization_loss_db: float
    other_loss_db: float
    total_loss_db: float
    received_dBW: Optional[float]        # None unless Pt_dBW is given


def one_way_budget(tracer: BatchTracer, r0: np.ndarray, up: np.ndarray, north: np.ndarray,
                   east: np.ndarray, elev_deg: float, az_deg: float, max_path_m: float, *,
                   nhops: int = 1, site: str = "tx", Pt_dBW: float | None = None,
                   Gt_dB: float = 0.0, Gr_dB: float = 0.0, L_pol_dB: float = legs.L_POL_DB,
                   L_other_dB: float = 0.0, valley_m: float = legs.VALLEY_M) -> LinkBudgetResult:
    """
    Budget of the leg launched at (elev_deg, az_deg) and landing after nhops hops. Losses
    (dB): absorption of the central ray, ground loss of every bounce, polarization, other.
    """
    land = legs.landing_mini_fan(tracer, r0, elev_deg, az_deg, up, north, east, max_path_m, nhops=nhops,
                                 valley_m=valley_m)
    if not land.landed:
        nan = float("nan")
        return LinkBudgetResult(landed=False, leg=None, ground_range_m=nan, delay_s=nan, doppler_hz=nan,
                                focusing_gain_db=None, caustic=False, d_eff_m=nan, absorption_db=nan,
                                ground_loss_db=nan, polarization_loss_db=L_pol_dB,
                                other_loss_db=L_other_dB, total_loss_db=nan, received_dBW=None)
    loss = land.A_db + land.ground_loss_db + L_pol_dB + L_other_dB
    f0 = tracer.ray.omega / (2.0 * np.pi)
    recv = None if Pt_dBW is None else legs.one_way_power_dB(Pt_dBW, Gt_dB, Gr_dB, f0, land.d_eff, loss)
    return LinkBudgetResult(
        landed=True,
        leg=legs.leg_mode(land, tracer, site=site),
        ground_range_m=great_circle_range_m(land.r_land, r0),
        delay_s=land.P_group / legs.C,
        doppler_hz=land.doppler_hz,
        focusing_gain_db=land.G_focus_db,
        caustic=land.caustic,
        d_eff_m=land.d_eff,
        absorption_db=land.A_db,
        ground_loss_db=land.ground_loss_db,
        polarization_loss_db=L_pol_dB,
        other_loss_db=L_other_dB,
        total_loss_db=loss,
        received_dBW=recv,
    )
