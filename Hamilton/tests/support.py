"""
Shared helpers for the Hamilton test suite.

Plain numpy/scipy, no pytest required. Every test module can be run with
`python tests/runner.py` or, if pytest is installed, with `python -m pytest tests`.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.constants import C, epsilon_0, m_e, q_e  # noqa: E402
from raytracing.main import Ray  # noqa: E402

F0_HZ = 8.0e6                 # reference carrier used throughout the Phase 0 checks
W0 = 2.0 * np.pi * F0_HZ
B0 = 5.0e-5                   # Tesla, along +z (uniform field)
B_HAT = np.array([0.0, 0.0, 1.0])
R_TEST = np.array([6.9e6, 0.0, 0.0])   # any point: the field is uniform


def x_factor(omega: float = W0) -> float:
    """X = Ne * x_factor(omega), x_factor = q^2 / (eps0 m omega^2)."""
    return q_e**2 / (epsilon_0 * m_e * omega**2)


def uniform_ray(X: float, MODE: int, B_scale: float = 1.0, omega: float = W0):
    """A Ray with uniform Ne (giving X) and uniform B (scaled from B0)."""
    Ne0 = X / x_factor(omega)
    B = B0 * B_scale * B_HAT
    ray = Ray(omega, MODE, lambda x, y, z: Ne0, B_field=lambda x, y, z: B)
    return ray, Ne0, B


def root_p(ray: Ray, X: float, Y: float, cos_theta: float, MODE: int) -> float:
    """Self-consistent root p = n^2 of the Appleton-Hartree relation at fixed theta."""
    p = 1.0
    for _ in range(5000):
        _, _, _, pn, _ = ray._appleton_hartree_p(X, Y, np.sqrt(p) * cos_theta, MODE)
        if not np.isfinite(pn):
            return float("nan")
        if abs(pn - p) < 1e-14:
            return float(pn)
        p = 0.5 * (p + pn)
    return float(p)


def wave_normal(ray: Ray, X: float, theta_deg: float, MODE: int, B_scale: float = 1.0):
    """Wave-normal vector u = n u_hat at angle theta from B (B along z). None if no root."""
    th = np.radians(theta_deg)
    Y = q_e * B0 * B_scale / (m_e * W0)
    p = root_p(ray, X, Y, np.cos(th), MODE)
    if not np.isfinite(p) or p <= 0.0:
        return None
    return np.sqrt(p) * np.array([np.sin(th), 0.0, np.cos(th)])


def omega_of_k(kvec: np.ndarray, Ne0: float, B: np.ndarray, MODE: int, ray: Ray) -> float:
    """
    Dispersion relation omega(k) for one mode: the omega at which |u|^2 = p_MODE(omega),
    with u = c k / omega. Independent of Haselgrove's polynomial normalisation.
    """
    B_mag = np.linalg.norm(B)
    B_hat = B / B_mag

    def h(w):
        X = Ne0 * q_e**2 / (epsilon_0 * m_e * w**2)
        Yw = q_e * B_mag / (m_e * w)
        u = C * kvec / w
        Yd = B_hat @ u
        _, _, _, pm, _ = ray._appleton_hartree_p(X, Yw, Yd, MODE)
        return (u @ u) - pm

    return brentq(h, 0.7 * W0, 1.3 * W0, xtol=1e-6)


def group_velocity_fd(kvec: np.ndarray, Ne0: float, B: np.ndarray, MODE: int, ray: Ray) -> np.ndarray:
    """d(omega)/dk by central differences of omega_of_k (physical group velocity direction)."""
    dk = 1e-6 * np.linalg.norm(kvec)
    g = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = dk
        g[i] = (omega_of_k(kvec + e, Ne0, B, MODE, ray)
                - omega_of_k(kvec - e, Ne0, B, MODE, ray)) / (2.0 * dk)
    return g


def physical_rdot(ray: Ray, state6: np.ndarray) -> np.ndarray:
    """The traced ray direction (sign-corrected RHS), position part."""
    return ray._haselgrove_physical_rhs(0.0, state6)[:3]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
