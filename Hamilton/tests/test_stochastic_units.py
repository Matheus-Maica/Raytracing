"""
Phase 3 unit tests with analytic or independent numerical answers (spec Table 9, Phase 3 rows):
spectrum normalisation, the Psi_p series, the screen integral scale, rods' determinant,
the G_p asymptotes, and the Rician split.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import kv, gamma

from stochastic import channel
from stochastic.screens import F_of_p, psi_p, spectrum_norm, struct_fn, inverse_2x2, screen_matrices


def test_spectrum_normalisation_eq_8_1():
    """int Phi d^3q/(2pi)^3 = sigma^2 for eq (8.1), a = b = 1, L0 = 1, for several p."""
    for p in (3.6, 4.0, 4.5, 5.0):
        Np = float(spectrum_norm(p))
        # spherical integral of (1 + q^2)^(-p/2), q = |q| in units of 1/L0
        radial, _ = quad(lambda q: 4.0 * np.pi * q * q * (1.0 + q * q) ** (-0.5 * p), 0.0, np.inf,
                         limit=500)
        value = Np * radial / (2.0 * np.pi) ** 3
        assert abs(value - 1.0) < 1e-7, f"p={p}: normalisation {value}"


def test_psi_series_matches_exact_below_1e_3():
    """Psi_p series (9.5) against the Bessel-function formula (9.4) at u = 9.9e-4."""
    u = 9.9e-4
    for p in (3.6, 4.0, 4.5):
        mu = 0.5 * (p - 2.0)
        exact = 1.0 - 2.0 ** (1.0 - mu) / gamma(mu) * u**mu * kv(mu, u)
        series = float(psi_p(u, p))
        assert abs(series / exact - 1.0) < 1e-5, f"p={p}: series {series} exact {exact}"


def test_psi_limits():
    """Psi -> 1 at large u (decorrelated), Psi ~ u^2 (ln(2/u) - gamma + 1/2)/2 for p = 4."""
    assert abs(float(psi_p(50.0, 4.0)) - 1.0) < 1e-9
    u = 2.0e-3
    approx = 0.5 * u * u * (np.log(2.0 / u) - np.euler_gamma + 0.5)
    assert abs(float(psi_p(u, 4.0)) / approx - 1.0) < 2e-2   # next-order corrections


def test_scale_integral_closed_form_equals_phi_integral():
    """
    l_par = L0 F(p) a b / sqrt(det W) (eq. 9.3) must equal the phase-variance integral
    sigma^2 = g^2 ds sigma_N^2 l_par, evaluated numerically over azimuth:
    int d^2k Phi(k)/(2pi)^2 with Phi = g^2 ds sigma_N^2 a b N_p L0^3 (1 + L0^2 k^T W k)^(-p/2).
    """
    rng = np.random.default_rng(3)
    for _ in range(4):
        p = rng.uniform(3.2, 5.5)
        a, b = rng.uniform(1.0, 20.0), rng.uniform(1.0, 5.0)
        L0 = 7.0
        # random positive-definite 2x2 W
        A = rng.normal(size=(2, 2))
        W = A @ A.T + 0.3 * np.eye(2)
        detW = np.linalg.det(W)
        Np = float(spectrum_norm(p))
        # radial integral in each direction: int_0^inf k (1 + L0^2 k^2 q)^(-p/2) dk = 1/((p-2) L0^2 q)
        def angular(phi):
            e = np.array([np.cos(phi), np.sin(phi)])
            q = e @ W @ e
            return 1.0 / ((p - 2.0) * L0**2 * q)
        ang, _ = quad(angular, 0.0, 2.0 * np.pi, limit=200)
        numeric = a * b * Np * L0**3 * ang / (2.0 * np.pi) ** 2      # per unit g^2 ds sigma_N^2
        closed = L0 * float(F_of_p(p)) * a * b / np.sqrt(detW)
        assert abs(closed / numeric - 1.0) < 1e-6, (p, closed, numeric)


def test_rod_determinant_identity():
    """Rods (b = 1): det W = cos^2(theta) + a^2 sin^2(theta), theta = angle between ray and B (§9.3)."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        s = rng.normal(size=3)
        s /= np.linalg.norm(s)
        B = rng.normal(size=3)
        B /= np.linalg.norm(B)
        e_sh = np.cross(s, B)              # perpendicular to B and to the ray
        if np.linalg.norm(e_sh) < 1e-3:
            continue
        e_sh = np.cross(B, np.cross(B, s))  # any unit vector perpendicular to B
        e_sh /= np.linalg.norm(e_sh)
        t1 = np.cross(s, B)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(s, t1)
        t2 /= np.linalg.norm(t2)
        a = rng.uniform(1.0, 30.0)
        W, _ = screen_matrices(t1[None], t2[None], B[None], e_sh[None], np.array([a]), np.array([1.0]))
        det = np.linalg.det(W[0])
        cos_t = float(B @ s)
        assert abs(det - (cos_t**2 + a**2 * (1.0 - cos_t**2))) < 1e-10, (det, cos_t, a)


def test_struct_function_saturates_at_two_sigma_squared():
    """D_i(s) -> 2 sigma^2 for large separations and vanishes at zero."""
    W_inv = np.eye(2)[None]
    sig2 = np.array([3.0])
    L0 = np.array([5.0]); p = np.array([4.0])
    assert abs(struct_fn(np.array([[1e6, 0.0]]), W_inv, sig2, L0, p)[0] - 6.0) < 1e-9
    assert struct_fn(np.array([[0.0, 0.0]]), W_inv, sig2, L0, p)[0] == 0.0


def test_G_p_asymptotes():
    """G_p(beta) -> 1/(p-2) for large beta; G_4(beta) -> (pi/2) beta for small beta (c_4 = pi/2)."""
    for p in (3.6, 4.0, 4.5):
        big = float(channel.gp(p, 1.0e2))
        assert abs(big * (p - 2.0) - 1.0) < 1e-3, (p, big)
    # The next term is O(beta) relative (about 0.85% at beta = 1e-3), so test at 1e-5.
    small = float(channel.gp(4.0, 1.0e-5))
    assert abs(small / (0.5 * np.pi * 1.0e-5) - 1.0) < 1e-3, small


def test_rician_split_reproduces_S4():
    """A_c^2 + 2 A_d^2 = 1, and (P_d^2 + 2 P_c P_d)/(P_c + P_d)^2 = S4^2 for the Rician form."""
    for S4 in (0.1, 0.5, 0.9, 0.999):
        Ac, Ad = channel.rician(S4)
        assert abs(Ac**2 + 2.0 * Ad**2 - 1.0) < 1e-12
        Pc, Pd = Ac**2, 2.0 * Ad**2
        assert abs((Pd**2 + 2.0 * Pc * Pd) / (Pc + Pd) ** 2 - S4**2) < 1e-10


def test_inverse_2x2():
    W = np.array([[[2.0, 0.3], [0.3, 5.0]]])
    inv, det = inverse_2x2(W)
    assert np.allclose(inv[0] @ W[0], np.eye(2), atol=1e-12)
    assert abs(det[0] - (10.0 - 0.09)) < 1e-12


def test_D_star_limits():
    """D*(sigma^2) -> 2 in strong scatter; ~ 1.264 sigma^2 in weak scatter (spec §10.2)."""
    assert abs(channel.D_star(50.0) - 2.0) < 1e-12
    s2 = 1e-4
    assert abs(channel.D_star(s2) / (1.264 * s2) - 1.0) < 1e-3
