"""
environment.py -- model interfaces and their output checks (spec §4.4, §7.2, §8.3, §8.5).

The spec leaves the contents of the absorption, irregularity, drift and surface models
to their providers. This module fixes the interfaces, provides simple default providers
(clearly labelled as placeholders, decision D9), and applies the checks the spec requires
of model outputs. Checks on the low-level screen inputs are in screens.py; the drift
projection of §8.3 is applied here, to model outputs only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

import numpy as np

from core.constants import epsilon_0

try:
    from global_land_mask import globe as _land_globe
    _HAS_LAND_MASK = True
except ImportError:            # pragma: no cover - optional dependency
    _HAS_LAND_MASK = False


# ---------------------------------------------------------------------------
# Protocols (spec Listings 2 and 3, §4.4)
# ---------------------------------------------------------------------------

class AbsorptionModel(Protocol):
    density_kind: Literal["enhancement", "total_below_merge"]
    nu_kind: Literal["nu_m", "nu_eff"]
    h_merge: tuple

    def electron_density(self, r_ecef, t) -> np.ndarray: ...
    def collision_frequency(self, r_ecef, t) -> np.ndarray: ...
    def riometer_A30(self, lat, lon, t) -> Optional[np.ndarray]: ...


class IrregularityModel(Protocol):
    def at(self, r_ecef: np.ndarray, t) -> dict: ...


class DriftModel(Protocol):
    def at(self, r_ecef: np.ndarray, t) -> dict: ...


class SurfaceModel(Protocol):
    def at(self, lat, lon, t) -> dict: ...


# ---------------------------------------------------------------------------
# Output checks
# ---------------------------------------------------------------------------

def check_drift(v_d: np.ndarray, B_hat: np.ndarray, *, warn_fraction: float = 0.05):
    """
    Spec §8.3 / §8.5: the E x B drift is perpendicular to B. Remove any component along
    B and warn if it was larger than warn_fraction of |v_d|. Returns the projected drift.
    """
    v = np.asarray(v_d, dtype=float)
    b = np.asarray(B_hat, dtype=float)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    along = np.sum(v * b, axis=-1, keepdims=True)
    mag = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(np.abs(along) > warn_fraction * np.maximum(mag, 1e-300)):
        warnings.warn("drift has a component along B; removed (spec §8.3)")
    return v - along * b


def check_irregularity(out: dict) -> dict:
    """
    Spec §8.3 checks on an IrregularityModel output: clip p to (3.05, 5.95) with a warning,
    enforce a, b >= 1, reorthogonalise e_sh against B, reject negative xi or sigma_N.
    """
    out = dict(out)
    if "xi" not in out and "sigma_N" not in out:
        raise ValueError("irregularity output needs 'xi' or 'sigma_N'")
    if "xi" in out and np.any(np.asarray(out["xi"]) < 0):
        raise ValueError("negative fractional density fluctuation")
    p = np.asarray(out["p"], dtype=float)
    if np.any(p < 3.05) or np.any(p > 5.95):
        warnings.warn("spectral index clipped to (3.05, 5.95) (spec §8.3)")
        out["p"] = np.clip(p, 3.05, 5.95)
    out["a"] = np.maximum(np.asarray(out["a"], dtype=float), 1.0)
    out["b"] = np.maximum(np.asarray(out["b"], dtype=float), 1.0)
    if "e_sh" in out and "B_hat" in out:
        B = np.asarray(out["B_hat"], dtype=float)
        B = B / np.linalg.norm(B, axis=-1, keepdims=True)
        e = np.asarray(out["e_sh"], dtype=float)
        e = e - np.sum(e * B, axis=-1, keepdims=True) * B
        out["e_sh"] = e / np.linalg.norm(e, axis=-1, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# Default providers (placeholders, D9)
# ---------------------------------------------------------------------------

@dataclass
class ConstantIrregularity:
    """Uniform irregularity model (placeholder; the spec's reference values use this form)."""

    xi: float = 5e-3
    L0: float = 10e3
    p: float = 4.0
    a: float = 10.0
    b: float = 3.0
    bragg: Optional[dict] = None

    def at(self, r_ecef, t) -> dict:
        n = np.atleast_2d(r_ecef).shape[0]
        return dict(xi=np.full(n, self.xi), L0=np.full(n, self.L0), p=np.full(n, self.p),
                    a=np.full(n, self.a), b=np.full(n, self.b), bragg=self.bragg)


@dataclass
class ConstantDrift:
    """Uniform drift and turbulence (placeholder)."""

    v_d: tuple = (-800.0, 200.0, 0.0)    # ENU east, north, up [m/s]: the spec's TV drift
    sigma_v: float = 150.0

    def at(self, r_ecef, t) -> dict:
        n = np.atleast_2d(r_ecef).shape[0]
        return dict(v_d=np.tile(np.asarray(self.v_d, float), (n, 1)), sigma_v=np.full(n, self.sigma_v))


# Typical electrical constants (ITU-R P.527 order of magnitude; placeholders, D9).
SURFACE_CONSTANTS = {
    "sea": (70.0, 5.0),            # eps_r, sigma [S/m] (spec §4.4)
    "land": (15.0, 5e-3),
    "sea_ice": (3.2, 1e-3),
    "ice_sheet": (3.0, 1e-5),
    "fresh_water": (80.0, 1e-2),
}


@dataclass
class SeaLandSurface:
    """
    Surface type from the land mask (sea vs land) and constants from SURFACE_CONSTANTS.
    Sea ice and ice sheet need user data (not in any file here); they are never chosen
    automatically. Returns dicts with keys surface, eps_r, sigma_g (per point).
    """

    ice_fraction: float = 0.0

    def at(self, lat, lon, t) -> dict:
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        if _HAS_LAND_MASK:
            land = _land_globe.is_land(lat, ((lon + 180.0) % 360.0) - 180.0)
        else:
            land = np.zeros(lat.shape, dtype=bool)
        surface = np.where(land, "land", "sea")
        eps = np.where(land, SURFACE_CONSTANTS["land"][0], SURFACE_CONSTANTS["sea"][0])
        sig = np.where(land, SURFACE_CONSTANTS["land"][1], SURFACE_CONSTANTS["sea"][1])
        return dict(surface=surface, eps_r=eps, sigma_g=sig, c_ice=np.full(lat.shape, self.ice_fraction))


def relative_permittivity_complex(eps_r, sigma_g, f0):
    """eps_c = eps_r - i sigma/(omega eps0): e^{+i omega t} convention (spec §4.4)."""
    omega = 2.0 * np.pi * f0
    return np.asarray(eps_r, dtype=complex) - 1j * np.asarray(sigma_g, dtype=float) / (omega * epsilon_0)
