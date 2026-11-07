"""
ground.py -- ground reflection: Fresnel coefficients, bounce losses and bounce records
(spec §4.4, eqs. 4.3 and 4.4; Listing 1 Bounce).

Specular reflection itself happens in batch.py (about the WGS84 outward normal, not the
radial direction); this module only computes what a reflection costs.
"""

from __future__ import annotations

import numpy as np

from core.constants import C
from geophysics.environment import relative_permittivity_complex
from core.records import Bounce


def grazing_angle(s_in: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Grazing angle psi between the incident ray and the local tangent plane (spec §4.4):
    sin(psi) = -s_in . n for an incident ray (s_in . n < 0). Shape (...,).
    """
    s = np.asarray(s_in, dtype=float)
    n = np.asarray(normal, dtype=float)
    return np.arcsin(np.clip(-np.sum(s * n, axis=-1), -1.0, 1.0))


def fresnel(psi, eps_r, sigma_g, f0: float):
    """
    Fresnel reflection coefficients, eq. (4.3), with the principal square root (Re >= 0).

        R_h = (sin psi - s) / (sin psi + s),      R_v = (eps_c sin psi - s) / (eps_c sin psi + s),
        s = sqrt(eps_c - cos^2 psi),              eps_c = eps_r - i sigma/(omega eps0).

    Returns (R_h, R_v) complex arrays. Checks (tests): R_h = -R_v at normal incidence,
    |R| -> 1 at grazing incidence.
    """
    psi = np.asarray(psi, dtype=float)
    eps_c = relative_permittivity_complex(eps_r, sigma_g, f0)
    sp = np.sin(psi)
    cp2 = np.cos(psi) ** 2
    s = np.sqrt(eps_c - cp2)
    R_h = (sp - s) / (sp + s)
    R_v = (eps_c * sp - s) / (eps_c * sp + s)
    return R_h, R_v


def bounce_loss_dB(R_h, R_v):
    """L_g = -10 log10( (|R_h|^2 + |R_v|^2)/2 ) [dB], eq. (4.4): equal mix of polarisations."""
    power = 0.5 * (np.abs(R_h) ** 2 + np.abs(R_v) ** 2)
    return -10.0 * np.log10(power)


def roughness_power_factor(sigma_h, psi, f0: float):
    """Coherent-power factor exp(-(2 k0 sigma_h sin psi)^2) for a rough surface (optional, §4.4)."""
    k0 = 2.0 * np.pi * f0 / C
    return np.exp(-(2.0 * k0 * np.asarray(sigma_h, dtype=float) * np.sin(psi)) ** 2)


def make_bounce(j: int, r: np.ndarray, s_in: np.ndarray, normal: np.ndarray, *,
                eps_r: float, sigma_g: float, surface: str, f0: float) -> Bounce:
    """A Bounce record for one specular reflection with the given surface constants."""
    psi = float(grazing_angle(s_in, normal))
    R_h, R_v = fresnel(psi, eps_r, sigma_g, f0)
    return Bounce(j=int(j), r=np.asarray(r, dtype=float).copy(), normal=np.asarray(normal, dtype=float).copy(),
                  grazing=psi, surface=surface, R_h=complex(R_h), R_v=complex(R_v),
                  loss_dB=float(bounce_loss_dB(R_h, R_v)))
