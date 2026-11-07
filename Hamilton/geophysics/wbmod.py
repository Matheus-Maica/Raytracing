"""
wbmod.py -- HiCIRF's WBMOD-sourced irregularity climatology and plasma drift (Nickisch et al. 2012,
"HiCIRF: A high-fidelity HF channel simulation", Radio Sci. 47, RS0L11), as far as it can be built
without WBMOD itself.

WBMOD's own Fortran source is distributed under U.S. export-control restriction (confirmed via NASA
CCMC's WBMOD model page and NWRA's own listing; no public download exists, and the CCMC "Instant Run"
web form is interactive-only, undocumented as an API, and not archived, so it cannot feed a per-ray
climatology lookup). See the plan at .claude/plans/valiant-questing-robin.md, Decision 1: until the
real model is available (path (a): request the source from NWRA/Secan directly), this module
implements everything that IS public -- the paper's own eq. (5) formula, exactly, and the
environment.IrregularityModel / DriftModel Protocol shapes a real WBMOD wrapper will need -- so that
plugging in genuine WBMOD output later is a one-function swap, not a redesign (path (c)).

WBMOD_LO_M, the fixed outer scale of eq. (5), is HiCIRF's own convention (not a WBMOD output, not a
free placeholder): the paper states "the irregularity outer scale size is taken to be Lo = 30 km."

CrudeWBMOD (2026-11-07) is OUR OWN, deliberately very crude stand-in for WBMOD, by the user's decision
(WBMOD is not obtainable; this "does not need to be realistic at all, it just needs to return numbers" and
will be replaced). It produces WBMOD's two outputs -- the 1-D spectral index q and log10 CkL -- from a few
hand-picked Gaussian bumps in magnetic latitude and MLT (auroral oval, polar cap, equatorial post-sunset),
then turns them into xi with eq. (5) and into Hamilton's 3-D index with p_from_q() (p = q + 2: for a
power-law spectrum the 1-D in-situ index is the 3-D index minus 2). It is an environment.IrregularityModel,
opt-in via api.Link(irregularity_model=CrudeWBMOD(...)).

PlaceholderWBMODIrregularity / PlaceholderWBMODDrift are the earlier, even simpler constant stand-ins (kept
for existing callers), conforming to the same Protocols.

PlaceholderWBMODIrregularity.axial_ratio_B/axial_ratio_sheet: WBMOD's own irregularity model does use two
axial ratios (confirmed via secondary literature, not Secan's own papers directly -- those were not
reachable): "along-field: across-field" ratios of roughly 10-50 and a second, smaller axial ratio of
roughly 1-4. This maps onto Hamilton's existing a (along B) / b (along the sheet axis) two-ratio
convention reasonably well, though the two conventions' exact frames are not independently confirmed to
line up. Secondary sources also disagree with each other on both numbers (and on the spectral index:
figures of 4, "p", and 1.5 all appear for different WBMOD eras/regimes) -- treat the defaults below as
representative, not authoritative, until Secan's own papers or the real model are available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as _gamma

from core.geometry import xyz_m_to_latlon

WBMOD_LO_M = 30_000.0   # HiCIRF eq. (5): the WBMOD/PDM outer scale is fixed at 30 km, not a free input.


def xi_squared_eq5(q, CkL, ne_profile_m3, h_m, *, Lo_m: float = WBMOD_LO_M):
    """
    HiCIRF eq. (5): the fractional electron-density variance xi^2 = sigma_Ne^2 / <Ne^2>, from WBMOD's
    one-dimensional spectral index q and irregularity strength parameter CkL, with the outer scale
    fixed at Lo_m:

        xi^2 = (2 pi / 1000 m)^(q+2) * Gamma(q/2 - 1/2) * Lo^(q-1) * CkL
               / [8 pi^(3/2) Gamma(q/2 + 1) * integral_0^1000km(Ne^2 dh)]

    q, CkL         : WBMOD's returned one-dimensional spectral index and irregularity strength (source:
                     WBMOD; see this module's docstring for why it is not wired up yet).
    ne_profile_m3  : electron density [m^-3] on a height grid spanning the integration range.
    h_m            : the matching height grid [m]; the paper integrates Ne^2 over the first 1000 km of
                     altitude (the profile does not need to extend further; it is not truncated here).

    Transcribed directly from the published equation (OCR of the PDF's math is lossy); treat this as
    unverified against a reference WBMOD case until real (q, CkL) numbers are available to check it
    against the paper's own worked examples.
    """
    ne2_integral = np.trapz(np.asarray(ne_profile_m3, dtype=float) ** 2, np.asarray(h_m, dtype=float))
    return xi_squared_eq5_from_column(q, CkL, ne2_integral, Lo_m=Lo_m)


def xi_squared_eq5_from_column(q, CkL, ne2_column_m5, *, Lo_m: float = WBMOD_LO_M):
    """Eq. (5) with the Ne^2 column integral [m^-5] already done (broadcasts over all three arguments)."""
    q = np.asarray(q, dtype=float)
    ckl = np.asarray(CkL, dtype=float)
    numerator = (2.0 * np.pi / 1000.0) ** (q + 2.0) * _gamma(q / 2.0 - 0.5) * (Lo_m ** (q - 1.0)) * ckl
    denominator = 8.0 * np.pi ** 1.5 * _gamma(q / 2.0 + 1.0) * np.asarray(ne2_column_m5, dtype=float)
    return numerator / denominator


def _enu_basis_vectorised(r_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(up, north, east) unit ECEF vectors at each of N points, vectorised (api._enu_frame is scalar-only).
    Used so a drift given as a local (east, north) pair is converted per sample point, not pinned to one
    site-wide frame (the earlier single-frame conversion is a decent approximation over one short leg, but
    is wrong in direction once a leg's footprint moves far enough from the transmitter for "east" and
    "north" to rotate noticeably)."""
    r = np.atleast_2d(np.asarray(r_ecef, dtype=float))
    lat_deg, lon_deg, _ = xyz_m_to_latlon(r.T)
    lat_r, lon_r = np.deg2rad(np.asarray(lat_deg, dtype=float)), np.deg2rad(np.asarray(lon_deg, dtype=float))
    up = np.stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)], axis=-1)
    north = np.stack([-np.sin(lat_r) * np.cos(lon_r), -np.sin(lat_r) * np.sin(lon_r), np.cos(lat_r)], axis=-1)
    east = np.stack([-np.sin(lon_r), np.cos(lon_r), np.zeros_like(lon_r)], axis=-1)
    return up, north, east


@dataclass
class PlaceholderWBMODIrregularity:
    """
    environment.IrregularityModel stand-in for WBMOD. Fixes L0 at HiCIRF's eq. (5) outer scale
    (WBMOD_LO_M = 30 km); xi, p and the axial ratios remain explicit placeholders (decision D9 style)
    until eq. (5) is fed real (q, CkL) -- see this module's docstring.
    """

    xi: float = 5e-3
    p: float = 4.0                      # one secondary source for WBMOD gives the same value (3-D index = 4)
    axial_ratio_B: float = 10.0         # along-field: secondary sources cite a 10-50 range; 10 is their low end
    axial_ratio_sheet: float = 2.0      # across-field: secondary sources cite a 1-4 range (not independently
                                         # confirmed against Secan's own papers -- see this module's docstring)
    L0_m: float = WBMOD_LO_M

    def at(self, r_ecef, t) -> dict:
        n = np.atleast_2d(r_ecef).shape[0]
        return dict(xi=np.full(n, self.xi), L0=np.full(n, self.L0_m), p=np.full(n, self.p),
                    a=np.full(n, self.axial_ratio_B), b=np.full(n, self.axial_ratio_sheet))


@dataclass
class PlaceholderWBMODDrift:
    """
    environment.DriftModel stand-in for WBMOD's plasma-drift model (Richmond-NCAR + Heelis-UTD; see the
    plan's Stage 1). Same placeholder magnitude as the previous site-wide-constant drift
    (inputs.IrregularityInputs.drift_east_north_mps), but converted to ECEF through each sample point's
    OWN local (east, north) frame rather than one frame fixed at the transmitter site.
    """

    v_east_north_mps: tuple = (150.0, 0.0)
    sigma_v_frac: float = 0.2

    def at(self, r_ecef, t) -> dict:
        _, north, east = _enu_basis_vectorised(r_ecef)
        ve, vn = self.v_east_north_mps
        v_d = ve * east + vn * north
        sigma_v = self.sigma_v_frac * np.linalg.norm(v_d, axis=-1)
        return dict(v_d=v_d, sigma_v=sigma_v)


# ---------------------------------------------------------------------------
# Our own very crude WBMOD (user decision, 2026-11-07) -- see the module docstring
# ---------------------------------------------------------------------------

NOMINAL_NE2_COLUMN_M5 = 2.0e29   # integral Ne^2 dh for a ~1e12 m^-3, ~200 km thick F layer [m^-5]


def p_from_q(q):
    """Hamilton's 3-D spectral index from WBMOD's 1-D index: p = q + 2 (for a power-law spectrum, integrating
    over two transverse wavenumbers lowers the index by 2), clipped to Hamilton's (3.05, 5.95)."""
    return np.clip(np.asarray(q, dtype=float) + 2.0, 3.05, 5.95)


def _bump(x, centre, width):
    return np.exp(-((x - centre) / width) ** 2)


@dataclass
class CrudeWBMOD:
    """
    environment.IrregularityModel, OUR OWN crude stand-in for WBMOD (not realistic by design; to be replaced).

    log10 CkL = base + solar term + auroral-oval bump + polar-cap step + equatorial post-sunset bump, in
    centred-dipole magnetic latitude and MLT (plasma_drift's IGRF frame); q is a constant. xi follows from
    eq. (5) with the vertical integral of Ne^2 over the first 1000 km above each point -- from ne_column
    (callable(x, y, z) -> Ne [m^-3], e.g. api.Link.electron_density_xyz_m) or, if None, a nominal F-layer
    value. The results are clipped to xi in [1e-4, 0.3].
    """

    ssn: float = 50.0                    # sunspot number
    kp: float = 3.0
    q: float = 2.0                       # WBMOD-style 1-D spectral index (-> p = 4)
    ne_column: object = None
    axial_ratio_sheet: float = 2.0

    def log10_ckl(self, mlat_deg, mlt_h):
        a_lat = np.abs(np.asarray(mlat_deg, dtype=float))
        oval = 67.0 - 1.5 * (self.kp - 3.0)                     # oval moves equatorward with Kp
        auroral = _bump(a_lat, oval, 5.0)
        cap = 0.5 * (1.0 + np.tanh((a_lat - oval - 5.0) / 2.0))
        dmlt = (np.asarray(mlt_h, dtype=float) - 22.0 + 12.0) % 24.0 - 12.0
        equatorial = _bump(np.asarray(mlat_deg, dtype=float), 0.0, 12.0) * _bump(dmlt, 0.0, 2.5)
        val = (30.5 + 0.5 * min(self.ssn, 200.0) / 100.0
               + (2.0 + 0.2 * (self.kp - 3.0)) * auroral + (1.5 + 0.2 * (self.kp - 3.0)) * cap
               + 2.5 * equatorial)
        return np.clip(val, 29.0, 34.5)

    def _ne2_column(self, r: np.ndarray) -> np.ndarray:
        if self.ne_column is None:
            return np.full(r.shape[0], NOMINAL_NE2_COLUMN_M5)
        _, _, alt = xyz_m_to_latlon(r.T)
        r_hat = r / np.linalg.norm(r, axis=-1, keepdims=True)
        R0 = np.linalg.norm(r, axis=-1) - np.asarray(alt, dtype=float)
        h = np.linspace(60e3, 1000e3, 95)
        pts = r_hat[:, None, :] * (R0[:, None] + h[None, :])[..., None]
        ne = np.asarray(self.ne_column(pts[..., 0], pts[..., 1], pts[..., 2]), dtype=float).reshape(pts.shape[:2])
        return np.maximum(np.trapz(ne ** 2, h, axis=1), 1e20)

    def at(self, r_ecef, t) -> dict:
        from geophysics.plasma_drift import geomagnetic_lat_lon_rad, magnetic_local_time_hours
        r = np.atleast_2d(np.asarray(r_ecef, dtype=float))
        n = r.shape[0]
        mlat, _ = geomagnetic_lat_lon_rad(r, t)
        mlat_deg = np.rad2deg(mlat)
        ckl = 10.0 ** self.log10_ckl(mlat_deg, magnetic_local_time_hours(r, t))
        q = np.full(n, self.q)
        xi = np.clip(np.sqrt(xi_squared_eq5_from_column(q, ckl, self._ne2_column(r))), 1e-4, 0.3)
        u = np.clip((np.abs(mlat_deg) - 20.0) / 20.0, 0.0, 1.0)
        a = 30.0 - 20.0 * 0.5 * (1.0 - np.cos(np.pi * u))      # 30 near the equator -> 10 above 40 deg mlat
        return dict(xi=xi, L0=np.full(n, WBMOD_LO_M), p=p_from_q(q), a=a,
                    b=np.full(n, self.axial_ratio_sheet), q=q, log10_CkL=np.log10(ckl))
