"""
Phase 0 tests (docs/implementation_plan.md §3): X-mode direction (P0.1), absorption
amplitude and dB factor (P0.2), exact group index (P0.3), phase sensitivity (P0.4),
and the spec's absorption table (§7.1) as a consistency check.

Each test compares against an independent quantity: a finite difference of the
dispersion relation omega(k), a closed form, or the spec's own numbers.
"""

from __future__ import annotations

import numpy as np

from support import (B0, C, F0_HZ, R_TEST, W0, cosine, group_velocity_fd, omega_of_k,
                     physical_rdot, root_p, uniform_ray, wave_normal, x_factor)
from core.constants import epsilon_0, m_e, q_e
from raytracing import absorption

# (X, theta in degrees, MODE): points where both roots exist (checked when written)
POINTS = [(0.3, 60.0, +1), (0.9, 60.0, +1), (0.5, 60.0, -1), (0.7, 20.0, -1), (0.95, 85.0, +1)]


# ---------------------------------------------------------------------------
# P0.1: X-mode direction
# ---------------------------------------------------------------------------

def test_ray_direction_follows_physical_group_velocity():
    """cos(traced ray direction, d(omega)/dk from the dispersion relation) == +1, both modes."""
    for MODE in (+1, -1):
        for X in (0.3, 0.5):
            for theta in (30.0, 60.0, 89.0):
                ray, Ne0, B = uniform_ray(X, MODE)
                u = wave_normal(ray, X, theta, MODE)
                assert u is not None, f"no root X={X} theta={theta} MODE={MODE}"
                rdot = physical_rdot(ray, np.concatenate([R_TEST, u]))
                vg = group_velocity_fd(W0 * u / C, Ne0, B, MODE, ray)
                c = cosine(rdot, vg)
                assert c > 0.9999, f"X={X} theta={theta} MODE={MODE}: cos={c:.6f}"


def test_direction_forward_along_wave_normal():
    """Physical group velocity has positive projection on the wave normal (energy flux
    argument: S.k = (1/2) omega mu0 |H|^2 > 0). Checked for both modes."""
    for MODE in (+1, -1):
        ray, Ne0, B = uniform_ray(0.5, MODE)
        u = wave_normal(ray, 0.5, 60.0, MODE)
        rdot = physical_rdot(ray, np.concatenate([R_TEST, u]))
        assert rdot @ u > 0.0, f"MODE={MODE}: ray runs against the wave normal"


def test_vacuum_threshold_direction_continuity():
    """For X just above X_VACUUM_EPS the X-mode direction must still be the vacuum direction."""
    for MODE in (+1, -1):
        ray_lo, _, _ = uniform_ray(1.0e-6, MODE)
        ray_hi, _, _ = uniform_ray(1.0e-5, MODE)
        u_lo = wave_normal(ray_lo, 1.0e-6, 60.0, MODE)
        u_hi = wave_normal(ray_hi, 1.0e-5, 60.0, MODE)
        r_lo = physical_rdot(ray_lo, np.concatenate([R_TEST, u_lo]))
        r_hi = physical_rdot(ray_hi, np.concatenate([R_TEST, u_hi]))
        assert cosine(r_lo, u_lo) > 0.999
        assert cosine(r_hi, u_hi) > 0.999, f"MODE={MODE}: direction jump at the vacuum threshold"


def test_modes_coincide_in_isotropic_limit():
    """As B -> 0 the O and X directions must coincide (both forward)."""
    X, theta = 0.3, 60.0
    ray_o, _, _ = uniform_ray(X, +1, B_scale=1e-3)
    ray_x, _, _ = uniform_ray(X, -1, B_scale=1e-3)
    u_o = wave_normal(ray_o, X, theta, +1, B_scale=1e-3)
    u_x = wave_normal(ray_x, X, theta, -1, B_scale=1e-3)
    r_o = physical_rdot(ray_o, np.concatenate([R_TEST, u_o]))
    r_x = physical_rdot(ray_x, np.concatenate([R_TEST, u_x]))
    assert cosine(r_o, r_x) > 0.999


# ---------------------------------------------------------------------------
# P0.3: exact group index
# ---------------------------------------------------------------------------

def test_group_index_matches_dispersion_derivative():
    """n_g = c/|v_g| from d(omega)/dk (finite differences of the dispersion root) equals
    Ray.group_index to 1e-6 relative, at five points (both modes)."""
    for X, theta, MODE in POINTS:
        ray, Ne0, B = uniform_ray(X, MODE)
        u = wave_normal(ray, X, theta, MODE)
        if u is None:
            continue
        ng_code = ray.group_index(np.concatenate([R_TEST, u]))
        vg = group_velocity_fd(W0 * u / C, Ne0, B, MODE, ray)
        ng_fd = C / np.linalg.norm(vg)
        rel = abs(ng_code / ng_fd - 1.0)
        assert rel < 1e-6, f"X={X} theta={theta} MODE={MODE}: code {ng_code:.8f} fd {ng_fd:.8f}"


def test_group_index_isotropic_limit():
    """Near B=0 the exact expression tends to 1/n. At perpendicular propagation the O/X
    split is O(Y^2), so the correction is O(Y^2); at oblique angles the split is
    O(Y cos theta), so the correction is O(Y)."""
    for MODE in (+1, -1):
        ray, _, _ = uniform_ray(0.3, MODE, B_scale=0.01)      # Y ~ 1.75e-3, exact branch
        u = wave_normal(ray, 0.3, 90.0, MODE, B_scale=0.01)
        n = np.linalg.norm(u)
        ng = ray.group_index(np.concatenate([R_TEST, u]))
        assert abs(ng * n - 1.0) < 1e-5, f"MODE={MODE} theta=90: n_g*n = {ng * n:.8f}"
        u60 = wave_normal(ray, 0.3, 60.0, MODE, B_scale=0.01)
        ng60 = ray.group_index(np.concatenate([R_TEST, u60]))
        Y = q_e * B0 * 0.01 / (m_e * W0)
        assert abs(ng60 * np.linalg.norm(u60) - 1.0) < 0.2 * Y, f"MODE={MODE} theta=60"
        # Below the threshold the exact formula is replaced by the limit 1/n itself.
        ray0, _, _ = uniform_ray(0.3, MODE, B_scale=1e-4)
        u0 = wave_normal(ray0, 0.3, 60.0, MODE, B_scale=1e-4)
        assert abs(ray0.group_index(np.concatenate([R_TEST, u0])) * np.linalg.norm(u0) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# P0.4: phase sensitivity
# ---------------------------------------------------------------------------

def _g_finite_difference(X: float, theta: float, MODE: int) -> float:
    """g = k0 (dn/dNe at fixed u_hat) cos(alpha_rw), by finite differences of the root."""
    ray, Ne0, _ = uniform_ray(X, MODE)
    th = np.radians(theta)
    cth = np.cos(th)
    Y = q_e * B0 / (m_e * W0)
    u = wave_normal(ray, X, theta, MODE)
    uhat = u / np.linalg.norm(u)
    rdot = physical_rdot(ray, np.concatenate([R_TEST, u]))
    cos_alpha = float(uhat @ (rdot / np.linalg.norm(rdot)))

    def n_of(Ne):
        return np.sqrt(root_p(ray, Ne * x_factor(), Y, cth, MODE))

    d = 1e-4 * Ne0
    dn = (n_of(Ne0 + d) - n_of(Ne0 - d)) / (2.0 * d)
    return (W0 / C) * dn * cos_alpha


def test_phase_sensitivity_matches_finite_difference():
    """g from the RHS internals (k0 MODE L X_f/|rdot|) equals the finite-difference value
    for both modes."""
    for X, theta, MODE in [(0.3, 60.0, +1), (0.5, 60.0, -1), (0.5, 30.0, -1), (0.7, 20.0, -1)]:
        ray, _, _ = uniform_ray(X, MODE)
        u = wave_normal(ray, X, theta, MODE)
        g_code = ray.phase_sensitivity(np.concatenate([R_TEST, u]))
        g_fd = _g_finite_difference(X, theta, MODE)
        assert abs(g_code / g_fd - 1.0) < 1e-5, f"X={X} theta={theta} MODE={MODE}: {g_code} vs {g_fd}"


def test_phase_sensitivity_unmagnetised_limit():
    """B -> 0: g -> -r_e lambda0 / n (spec eq. 4.2), for both modes."""
    r_e = q_e**2 / (4.0 * np.pi * epsilon_0 * m_e * C**2)
    lam0 = C / F0_HZ
    for MODE in (+1, -1):
        ray, _, _ = uniform_ray(0.3, MODE, B_scale=1e-4)
        u = wave_normal(ray, 0.3, 60.0, MODE, B_scale=1e-4)
        g = ray.phase_sensitivity(np.concatenate([R_TEST, u]))
        n = np.linalg.norm(u)
        assert abs(g / (-r_e * lam0 / n) - 1.0) < 1e-9


def test_phase_sensitivity_branches_agree_at_threshold():
    """The isotropic formula and the exact identity agree for small Y: difference O(Y^2) at
    perpendicular propagation, O(Y cos theta) at oblique angles."""
    k0 = W0 / C
    for MODE in (+1, -1):
        ray_e, _, _ = uniform_ray(0.3, MODE, B_scale=0.01)    # Y ~ 1.75e-3, exact branch
        u = wave_normal(ray_e, 0.3, 90.0, MODE, B_scale=0.01)
        g_exact = ray_e.phase_sensitivity(np.concatenate([R_TEST, u]))
        g_iso = -k0 * x_factor() / (2.0 * np.linalg.norm(u))
        assert abs(g_exact / g_iso - 1.0) < 1e-5, f"MODE={MODE} theta=90: {g_exact} vs {g_iso}"


def test_angle_form_matches_haselgrove_root():
    """Lossless angle form (spec eq. 4.1) equals the Haselgrove root p = n^2 for both modes
    and several angles: this fixes the MODE <-> sign mapping (spec §4.2 item 2)."""
    from support import root_p
    for X in (0.2, 0.6):
        for theta in (10.0, 45.0, 80.0):
            for MODE in (+1, -1):
                ray, _, _ = uniform_ray(X, MODE)
                Y = q_e * B0 / (m_e * W0)
                cth = np.cos(np.radians(theta))
                p_h = root_p(ray, X, Y, cth, MODE)
                p_a = ray.n_angle(X, Y, cth, MODE, U=1.0)
                assert abs(p_a - p_h) < 1e-10, f"X={X} th={theta} MODE={MODE}: {p_a} vs {p_h}"


# ---------------------------------------------------------------------------
# P0.2: absorption amplitude convention and dB factor
# ---------------------------------------------------------------------------

def test_absorption_small_z_limit():
    """Isotropic, small X and Z: 2 * alpha_amp = Ne q^2 nu / (eps0 m omega^2 c) (power, Np/m)."""
    X, nu = 1.0e-3, 1.0e5
    ray, Ne0, _ = uniform_ray(X, +1, B_scale=1e-4)
    u = wave_normal(ray, X, 60.0, +1, B_scale=1e-4)
    alpha_amp = absorption.absorption_rate(ray, R_TEST, u, model=lambda alt_km: nu)
    power = 2.0 * alpha_amp
    analytic = Ne0 * q_e**2 * nu / (epsilon_0 * m_e * W0**2 * C)
    assert abs(power / analytic - 1.0) < 1e-3, f"{power} vs {analytic}"


def test_absorption_quasi_longitudinal_eq_7_1():
    """Parallel propagation (theta=0): power dB/m = 8.686 * alpha_amp must match spec eq. (7.1),
    kappa = 4.61e-5 Ne nu / (omega +- omega_L)^2, for one of the two R/L modes."""
    X, nu = 1.0e-2, 1.0e5
    Y = q_e * B0 / (m_e * W0)
    for MODE in (+1, -1):
        ray, Ne0, _ = uniform_ray(X, MODE)
        u = np.array([0.0, 0.0, root_p(ray, X, Y, 1.0, MODE) ** 0.5])
        alpha_amp = absorption.absorption_rate(ray, R_TEST, u, model=lambda alt_km: nu)
        kappa_code = 8.686 * alpha_amp
        candidates = [4.61e-5 * Ne0 * nu / (W0 * (1.0 + s * Y)) ** 2 for s in (+1, -1)]
        best = min(abs(kappa_code / k - 1.0) for k in candidates)
        assert best < 0.01, f"MODE={MODE}: code {kappa_code:.4e} candidates {candidates}"


def test_dB_factor_is_power_convention():
    """8.686 * amplitude nepers = 4.343 * power nepers (dB convention used everywhere)."""
    A = 0.37
    assert abs(8.686 * A - 4.343 * 2.0 * A) < 1e-3 * 8.686 * A


# ---------------------------------------------------------------------------
# §7.1 table (spec consistency): the numbers follow from eq. (7.1) with the
# secant law and O/X-averaged riometer normalisation.
# ---------------------------------------------------------------------------

def test_spec_absorption_table_71_is_reproduced():
    f_L_vert, f_L_obl, A30 = 1.5, 0.4, 1.0            # MHz, MHz, dB
    R_E, h = 6371.0, 90.0
    E = np.radians(10.0)
    sec_chi = 1.0 / np.sqrt(1.0 - (R_E * np.cos(E) / (R_E + h)) ** 2)
    assert abs(sec_chi - 4.19) < 0.01
    # vertical 30 MHz, O/X average -> normalisation constant C (dB MHz^2)
    avg = 0.5 * (1.0 / (30.0 + f_L_vert) ** 2 + 1.0 / (30.0 - f_L_vert) ** 2)
    C_norm = A30 / avg
    table = {  # carrier MHz: (O, X) dB per crossing, spec §7.1
        8: (53.0, 65.0), 10: (35.0, 41.0), 15: (16.0, 18.0), 20: (9.0, 10.0)}
    for f, (o_ref, x_ref) in table.items():
        o = C_norm * sec_chi / (f + f_L_obl) ** 2
        x = C_norm * sec_chi / (f - f_L_obl) ** 2
        assert abs(o - o_ref) < 1.0, f"O at {f} MHz: {o:.1f} vs {o_ref}"
        assert abs(x - x_ref) < 1.0, f"X at {f} MHz: {x:.1f} vs {x_ref}"


# ---------------------------------------------------------------------------
# End-to-end smoke test: O and X traces through the same dense layer.
# ---------------------------------------------------------------------------

def test_xmode_and_omode_traces_behave_alike_through_a_layer():
    """Both modes land at comparable ranges through the same dense layer (before the P0.1
    fix X-mode gave 821 km against 1730 km for O-mode)."""
    from raytracing.raytracer import RayTracer
    from raytracing.main import Ray
    from core.geometry import WGS84_A, latlon_to_xyz_m

    def Ne_layer(x, y, z):
        h = np.sqrt(x * x + y * y + z * z) - WGS84_A
        return 1.2e12 * np.exp(-0.5 * ((h - 300e3) / 60e3) ** 2)

    r0 = latlon_to_xyz_m(57.0, 10.0, 0.0)
    up = r0 / np.linalg.norm(r0)
    north = np.array([-np.sin(np.radians(57)) * np.cos(np.radians(10)),
                      -np.sin(np.radians(57)) * np.sin(np.radians(10)),
                      np.cos(np.radians(57))])
    u0 = np.cos(np.radians(28.0)) * north + np.sin(np.radians(28.0)) * up
    ranges = {}
    for MODE in (+1, -1):
        ray = Ray(W0, MODE, Ne_layer)
        tr = RayTracer(ray, ds_step_m=2000.0, record_every=10)
        res = tr.trace(r0, u0, 2500e3, nhops=2, method="fixed", compute_group_path=False,
                       compute_absorption=True, compute_phase_variance=False)
        assert res.landed, f"MODE={MODE} did not land"
        ranges[MODE] = res.ground_range_total_m / 1e3
    assert abs(ranges[-1] - ranges[+1]) / ranges[+1] < 0.10, ranges
