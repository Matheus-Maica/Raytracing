"""
arrays.py -- element geometry, element phase and delay offsets (spec §11.5), patterns.

Element positions are given relative to the array phase centre in a local
Cartesian frame (for example ENU at the site). For a leg with launch or look
direction e_site and the array-end transverse frame (t1, t2, x_hat, y_hat):

    zeta_p = e_site . r_p                 longitudinal offset [m]
    xi_p   = (t1 . r_p, t2 . r_p)         transverse coordinates [m]
    (x_p, y_p) = (x_hat . xi_p, y_hat . xi_p)   principal-axis coordinates [m]

Element factor exp(+i k0 zeta_p) and delay offset -zeta_p / c (spec §11.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.constants import C


@dataclass
class Array:
    """Element positions (N,3) in metres relative to the phase centre (local ENU)."""

    positions: np.ndarray

    @classmethod
    def from_enu(cls, positions) -> "Array":
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        if pos.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        return cls(pos - pos.mean(axis=0))   # phase centre at the centroid

    @classmethod
    def linear(cls, n: int, spacing_m: float, boresight_az_deg: float = 0.0) -> "Array":
        """
        Uniform linear array along the direction perpendicular to boresight.
        Boresight azimuth is measured clockwise from north (as in Link._direction).
        """
        k = np.arange(n) - 0.5 * (n - 1)
        az = np.radians(boresight_az_deg)
        # Broadside is the boresight; the line is perpendicular to it in the horizontal plane.
        east_line = np.cos(az)
        north_line = -np.sin(az)
        pos = np.zeros((n, 3))
        pos[:, 0] = k * spacing_m * east_line
        pos[:, 1] = k * spacing_m * north_line
        return cls(pos)

    @property
    def n_elements(self) -> int:
        return int(self.positions.shape[0])

    def offsets(self, e_site, t1, t2, x_hat, y_hat, f0: float):
        """
        Returns dict with zeta (N,), delay_offset_s (N,), phase factor (N,) complex,
        x, y (N,) principal-axis coordinates. f0 in Hz.
        """
        e = np.asarray(e_site, dtype=float)
        e = e / np.linalg.norm(e)
        zeta = self.positions @ e
        xi = np.stack([self.positions @ np.asarray(t1, float), self.positions @ np.asarray(t2, float)], 1)
        x = xi @ np.asarray(x_hat, float)
        y = xi @ np.asarray(y_hat, float)
        k0 = 2.0 * np.pi * f0 / C
        return dict(zeta=zeta, delay_offset_s=-zeta / C,
                    phase=np.exp(1j * k0 * zeta), x=x, y=y)


def isotropic_pattern(beta_deg, alpha_deg):
    """Element pattern G(beta, alpha) = 1 (placeholder; supply a real pattern for a design)."""
    return np.ones(np.broadcast(np.asarray(beta_deg), np.asarray(alpha_deg)).shape)
