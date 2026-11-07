"""
records.py -- data records passed between stages (spec Listing 1, §3.3).

Units are SI throughout: metres, seconds, radians, Hz, m^-3, T.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RaySamples:
    """
    One traced leg, N samples, struct of arrays (spec §3.3, Table 4). Samples are
    taken at every accepted step of the fixed-step driver (spec §4.1); the first
    sample is the launch point and a ground reflection appears as two samples at the
    same position (before and after the mirror).
    """

    r: np.ndarray            # (N, 3) ECEF position [m]
    s: np.ndarray            # (N,) arc length from the launch [m]
    P: np.ndarray            # (N,) phase path [m]
    Pg: np.ndarray           # (N,) group path [m]
    s_hat: np.ndarray        # (N, 3) ray (group) direction, physical sign
    u_hat: np.ndarray        # (N, 3) wave-normal direction
    n: np.ndarray            # (N,) phase refractive index |u|
    X: np.ndarray            # (N,) plasma parameter
    B: np.ndarray            # (N, 3) magnetic field [T]
    Ne: np.ndarray           # (N,) electron density [m^-3]
    grad_Ne: np.ndarray      # (N, 3) [m^-4]
    alt: np.ndarray          # (N,) WGS84 altitude [m]
    hop: np.ndarray          # (N,) hop index, 0-based
    dA_dB: np.ndarray        # (N,) absorption on the step ending at sample j [dB]
    absorption_db: np.ndarray  # (N,) cumulative absorption [dB]
    vacuum: np.ndarray       # (N,) bool: vacuum branch used
    steep: np.ndarray        # (N,) bool: n below n_min (phase-screen description invalid)
    t1: Optional[np.ndarray] = None   # (N, 3) transported transverse frame (§4.3)
    t2: Optional[np.ndarray] = None   # (N, 3)
    bounce_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


@dataclass
class Bounce:
    """One specular ground reflection (spec Listing 1, §4.4)."""

    j: int                   # sample index of the ground contact (the crossing state)
    r: np.ndarray            # ECEF [m]
    normal: np.ndarray       # outward ellipsoid normal
    grazing: float           # grazing angle psi [rad]
    surface: str             # 'sea' | 'sea_ice' | 'land' | 'ice_sheet' | 'fresh_water'
    R_h: complex
    R_v: complex
    loss_dB: float           # L_g of eq. (4.4)


@dataclass
class ChannelParams:
    """
    Stochastic parameters of one leg at its array end (spec Listing 1, §10.7).

    Lengths in m, times in s, angular frequencies in rad/s. x_hat and y_hat are
    the principal axes of the diffuse correlation in the array-end transverse
    frame (t1, t2), with lx <= ly. V_app is the apparent pattern velocity in
    the same frame.
    """

    sigma_phi2: float            # phase variance of the leg, rad^2
    D_star: float                # diffuse-part threshold, eq. (10.3)
    lx: float
    ly: float
    x_hat: np.ndarray            # (2,)
    y_hat: np.ndarray            # (2,)
    tau_o: float                 # decorrelation time, s
    V_app: np.ndarray            # (2,) m/s
    Cxt: float
    Cyt: float
    tau_mean: float              # mean excess delay <tau>, s
    w_coh: float                 # coherence bandwidth, rad/s
    alpha: float                 # wander parameter
    S4: float                    # intensity scintillation index used
    A_c: float                   # Rician coherent amplitude
    A_d: float                   # Rician diffuse quadrature std
    diagnostics: dict = field(default_factory=dict)

    @property
    def rician_K_db(self) -> float:
        """Rician K factor, coherent over diffuse power: 10 log10(A_c^2 / (2 A_d^2)) [dB]."""
        diffuse = 2.0 * self.A_d ** 2
        return float(10.0 * np.log10(self.A_c ** 2 / diffuse)) if diffuse > 0.0 else float("inf")

    def __str__(self) -> str:
        V = float(np.linalg.norm(self.V_app)) if self.V_app is not None else float("nan")
        return (
            "ChannelParams (HiCIRF channel of one leg, at its array end)\n"
            f"  phase variance : {self.sigma_phi2:.3g} rad^2\n"
            f"  S4             : {self.S4:.3f}   Rician A_c {self.A_c:.3f}, A_d {self.A_d:.3f} "
            f"(K {self.rician_K_db:.1f} dB)\n"
            f"  decorrelation  : tau_o {self.tau_o:.3g} s, apparent velocity {V:.3g} m/s\n"
            f"  coherence      : lx {self.lx:.4g} m, ly {self.ly:.4g} m\n"
            f"  delay          : <tau> {self.tau_mean:.3g} s, coherence bandwidth "
            f"{self.w_coh / (2.0 * np.pi):.3g} Hz"
        )


@dataclass
class LegMode:
    """
    One mode of one leg, from a site to a scatterer (spec Listing 1, §6).
    Deterministic fields are filled by legs.py; channel parameters by channel.py.
    """

    site: str                        # 'tx' | 'rx'
    mode: str                        # 'O' | 'X'
    label: str                       # e.g. '1F-low-O', '2EF-high-X'
    n_hops: int
    beta: float                      # launch elevation at the site [rad]
    alpha: float                     # launch azimuth at the site [rad]
    e_site: np.ndarray               # (3,) unit launch direction (ECEF)
    k_far: np.ndarray                # (3,) physical propagation direction at the far end
    grazing_far: float               # grazing angle at the scatterer [rad]
    Pg: float                        # group path of the leg [m]
    P: float                         # phase path of the leg [m]
    delay: float                     # Pg / c [s]
    doppler: float                   # -(1/2pi) * integral(g dNe/dt ds) [Hz]
    absorption_dB: float             # total absorption of the mode [dB]
    absorption_D_dB: float           # part of it below 120 km [dB]
    ground_loss_dB: float
    d_eff: float                     # flux-tube effective distance [m] (spec §4.5)
    caustic_flag: bool
    samples: Optional[RaySamples] = None
    bounces: list = field(default_factory=list)
    channel: Optional[ChannelParams] = None
    # X-mode legs are internally validated only until an external reference (PHaRLAP) is available
    # (decision D1, docs/spec_deviations.md item 19). None for O-mode legs.
    xmode_validated: Optional[bool] = None
    # True where any step has k0 |Im n| lambda0 > 0.1: the first-order absorption is not valid (spec §7.3)
    attenuation_flag: Optional[bool] = None


@dataclass
class Component:
    """One contribution to h_pq(t, tau) (spec Listing 1, §12)."""

    kind: str                        # 'target' | 'surface' | 'auroral' | 'meteor'
    out: LegMode
    ret: LegMode
    scatterer: dict                  # position, RCS or sigma0 * area, Doppler spectrum spec
    power: float                     # received power [W] before stochastic spreading
    delay: float                     # s
    doppler: float                   # Hz
    phase: float                     # rad
    gamma_or: Optional[float] = None  # reciprocity diagnostic, eq. (12.3); None if not computed
