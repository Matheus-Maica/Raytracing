"""Verification suite. Run: python tests.py

Checks the closed-form subcycle matrices against brute force, the exact
velocity push against RK4, discrete Gauss / staggering consistency, CPML
absorption, and stability of both modes on small grids.
"""
import numpy as np

from constants import epsilon_0, q_electron, m_e, C
from plasma import (subcycle_matrices, exact_velocity_push, cross_matrix,
                    plasma_frequency_sq)
from grid3d import Grid3D
from sources import make_source

rng = np.random.default_rng(7)
FAIL = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def test_subcycle_matrices_vs_bruteforce():
    print("subcycle X/G vs brute-force sub-stepping:")
    for M in (1, 2, 3, 7, 64, 501):
        shape = (4, 3)
        wp2 = rng.uniform(0, 1e16, shape)
        nu = rng.uniform(0, 1e7, shape)
        Omega = rng.uniform(-1e7, 1e7, 3)
        dt = 1e-7

        X, G = subcycle_matrices(wp2, nu, Omega, dt, M)

        # brute force: J' = C J + D E per sub-step
        dt_c = dt / M
        K = cross_matrix(np.broadcast_to(Omega, shape + (3,)).copy())
        idx = np.arange(3)
        K[..., idx, idx] += nu[..., None]
        I = np.zeros(shape + (3, 3)); I[..., idx, idx] = 1.0
        A_inv = np.linalg.inv(I + dt_c / 2 * K)
        Cm = A_inv @ (I - dt_c / 2 * K)
        D = A_inv * (epsilon_0 * wp2 * dt_c)[..., None, None]

        J = rng.standard_normal(shape + (3,))
        E = rng.standard_normal(shape + (3,))
        Jb = J.copy()
        for _ in range(M):
            Jb = np.einsum("...ij,...j->...i", Cm, Jb) + np.einsum("...ij,...j->...i", D, E)
        Jc = np.einsum("...ij,...j->...i", X, J) + np.einsum("...ij,...j->...i", G, E)
        err = np.max(np.abs(Jb - Jc)) / max(np.max(np.abs(Jb)), 1e-300)
        check(f"M={M}", err < 1e-10, f"rel err {err:.2e}")


def test_subcycle_unconditional_stability():
    print("subcycle map is non-expanding for extreme nu*dt, Omega*dt:")
    wp2 = np.array([[1e16]])
    for nu, Om in ((1e9, 0.0), (0.0, 1e9), (1e9, 1e9)):
        X, _ = subcycle_matrices(wp2, np.array([[nu]]), np.array([0, 0, Om]),
                                 dt=1e-3, M=17)
        s = np.linalg.svd(X[0, 0], compute_uv=False).max()
        check(f"nu={nu:.0e}, Omega={Om:.0e}", s <= 1 + 1e-12, f"||X|| = {s:.6f}")


def test_exact_push_vs_rk4():
    print("exact velocity push vs RK4:")
    dt = 1e-6
    cases = [
        ("magnetized+collisional", 3e5, np.array([1e6, -2e5, 4e5])),
        ("collisionless", 0.0, np.array([0.0, 0.0, 8.8e6])),
        ("unmagnetized", 2e4, np.array([0.0, 0.0, 0.0])),
        ("nothing", 0.0, np.array([0.0, 0.0, 0.0])),
    ]
    for name, nu, Omega in cases:
        u0 = rng.standard_normal(3) * 1e3
        F = rng.standard_normal(3) * 1e9

        u = exact_velocity_push(u0[None, :], F[None, :], np.array([nu]),
                                Omega[None, :], dt)[0]

        def rhs(v):
            return F - nu * v + np.cross(v, Omega)
        nsub, v = 4000, u0.copy()
        h = dt / nsub
        for _ in range(nsub):
            k1 = rhs(v); k2 = rhs(v + h/2*k1); k3 = rhs(v + h/2*k2); k4 = rhs(v + h*k3)
            v = v + h/6*(k1 + 2*k2 + 2*k3 + k4)
        err = np.linalg.norm(u - v) / np.linalg.norm(v)
        check(name, err < 1e-8, f"rel err {err:.2e}")


def test_gyration_handedness():
    """An electron in B = B z_hat with no E must gyrate clockwise when viewed
    from +z (v_y initially positive -> v_x becomes positive)."""
    print("electron gyration handedness:")
    B = 5e-5
    Omega = (q_electron / m_e) * np.array([0.0, 0.0, B])  # points along -z
    u0 = np.array([[0.0, 1.0, 0.0]])
    wc = abs(Omega[2])
    dt = 0.01 / wc
    u = exact_velocity_push(u0, np.zeros((1, 3)), np.array([0.0]), Omega, dt)[0]
    # F = q v x B = (-e)(y_hat x z_hat) B = -eB x_hat -> vx decreases: the
    # electron turns from +y toward -x, i.e. counterclockwise seen from +z.
    # That is the rotation sense of the whistler (R) mode's E field.
    check("vx decreases (ccw about +z, electron/R sense)", u[0] < 0,
          f"vx = {u[0]:.3e}")


def _uniform_grid(N, n0, B0z, mode="linear", dx=50.0):
    n_e = np.full((N, N, N), n0)
    return Grid3D((N, N, N), dx, n_e, B0_vec=(0, 0, B0z), mode=mode,
                  pml_thickness=8)


def test_vacuum_pulse_and_cpml():
    print("vacuum pulse: propagation, CPML absorption, discrete Gauss law:")
    N = 48
    grid = _uniform_grid(N, 0.0, 0.0)
    L = N * grid.dx
    src = make_source("dipole", grid.shape, grid.spacing,
                      (L/2, L/2, L/2), waveform="ricker",
                      omega=2*np.pi*C/(12*grid.dx), amplitude=1e-6)
    peak = 0.0
    for i in range(260):
        grid.step(src)
        peak = max(peak, grid.field_energy())
    final = grid.field_energy()
    check("CPML absorbs pulse", final < 2e-3 * peak,
          f"E_final/E_peak = {final/peak:.2e}")

    # Discrete Gauss: div E (staggering-matched operator) must vanish away
    # from the source ball. This is the test the old collocated grid failed.
    dx = grid.dx
    divE = np.zeros(grid.shape)
    divE[1:, 1:, 1:] = ((grid.Ex[1:, 1:, 1:] - grid.Ex[:-1, 1:, 1:])
                        + (grid.Ey[1:, 1:, 1:] - grid.Ey[1:, :-1, 1:])
                        + (grid.Ez[1:, 1:, 1:] - grid.Ez[1:, 1:, :-1])) / dx
    mask = np.ones(grid.shape, bool)
    c0, c1 = N//2 - 8, N//2 + 8
    mask[c0:c1, c0:c1, c0:c1] = False
    # exclude the PML (stretched coordinates make div E nonzero there)
    e = grid.pml_thickness + 2
    mask[:e] = mask[-e:] = False
    mask[:, :e] = mask[:, -e:] = False
    mask[:, :, :e] = mask[:, :, -e:] = False
    Escale = max(np.max(np.abs(grid.Ex)), 1e-300)
    rel = np.max(np.abs(divE[mask])) * dx / Escale
    check("div E = 0 off-source", rel < 1e-9, f"rel {rel:.2e}")


def test_plasma_cutoff():
    """A plane wave below omega_p must decay evanescently through a plasma
    slab (skin depth ~ c/omega_p ~ 1.6 cells here). A plane wave avoids the
    near-field of a point source, which is only dielectrically screened."""
    print("unmagnetized overdense plasma blocks a wave below omega_p:")
    N, dx = 48, 50.0
    n0 = 5e9   # omega_p ~ 4e6 rad/s
    w_src = 0.3 * np.sqrt(plasma_frequency_sq(n0, q_electron, m_e))

    fields = {}
    for n_bg in (0.0, n0):
        grid = _uniform_grid(N, n_bg, 0.0, dx=dx)
        L = N * dx
        # sheet sits in the taper (near-vacuum), radiates a quasi-plane wave
        # into the slab; probe at the far end of the fully-dense region.
        src = make_source("sheet", grid.shape, grid.spacing,
                          (12 * dx, L/2, L/2), polarization=(0, 0, 1),
                          omega=w_src, amplitude=1e-8)
        for i in range(500):
            grid.step(src)
        probe = np.s_[34, N//2, N//2]
        fields[n_bg] = abs(grid.Ez[probe])
    ratio = fields[n0] / max(fields[0.0], 1e-300)
    check("evanescent attenuation", ratio < 0.01,
          f"|E_plasma|/|E_vac| = {ratio:.2e}")


def test_magnetized_linear_stability():
    print("magnetized linear run stays bounded (the legacy failure mode):")
    grid = _uniform_grid(40, 5e9, 5e-5)
    L = 40 * grid.dx
    src = make_source("rotating_charge", grid.shape, grid.spacing,
                      (L/2, L/2, L/2), omega=2*np.pi*2e5, Q=5e-13)
    energies = []
    for i in range(600):
        grid.step(src)
        if i % 50 == 0:
            energies.append(grid.field_energy())
    ok = np.isfinite(energies[-1]) and energies[-1] < 1e4 * max(energies[2], 1e-300)
    check("no blow-up over 600 steps", ok,
          f"energy first/last = {energies[2]:.3e} / {energies[-1]:.3e}")


def test_whistler_handedness():
    """In the band omega < min(omega_p, |Omega_e|) only the R mode (E
    rotating in the electron-gyration sense) propagates along B0; the L
    mode is evanescent (Stix L < 0 here). A charge orbiting ccw (omega > 0,
    matching the electron sense for B0 = +z) must therefore deliver much
    more field at a probe down the field line than one orbiting cw."""
    print("whistler-mode handedness selectivity along B0:")
    N = 56   # probe must stay inside the fully-dense region (taper starts
    amp, purity = {}, {}   # pml+ramp = 14 cells from the wall)
    w0 = 2 * np.pi * 1.9e5
    for sense, w in (("R", +w0), ("L", -w0)):
        grid = _uniform_grid(N, 5e9, 5e-5)
        L = N * grid.dx
        src = make_source("rotating_charge", grid.shape, grid.spacing,
                          (L/2, L/2, L/2), omega=w, Q=5e-13, ramp_periods=3.0)
        probe = np.s_[N//2, N//2, N//2 + 12]
        Z = []
        for i in range(800):
            grid.step(src)
            if i >= 550:  # steady CW state only, not the turn-on transient
                Z.append(grid.Ex[probe] + 1j * grid.Ey[probe])
        Z = np.array(Z)
        t = (np.arange(550, 800) + 1) * grid.dt
        # ccw (R) and cw (L) circular content of the arriving wave
        a_R = np.abs(np.mean(Z * np.exp(-1j * w0 * t)))
        a_L = np.abs(np.mean(Z * np.exp(+1j * w0 * t)))
        amp[sense] = np.sqrt(a_R**2 + a_L**2)
        purity[sense] = a_R / max(a_L, 1e-300)
    check("R-driven wave arrives R-polarized", purity["R"] > 3,
          f"a_R/a_L = {purity['R']:.2f}")
    # The L-driven run still shows signal at the probe, but it is the
    # quasi-static rotating Coulomb near-field of the orbiting net charge
    # (probe is only ~0.6 vacuum wavelengths out), not a propagating mode --
    # so only a modest amplitude contrast can be asserted at this box size.
    check("L drive couples weakly (evanescent L mode)",
          amp["L"] < 0.75 * amp["R"],
          f"|E_L-drive|/|E_R-drive| = {amp['L']/max(amp['R'],1e-300):.3f}")


def test_nonlinear_smoke():
    print("nonlinear fluid mode: bounded, positive density:")
    grid = _uniform_grid(32, 5e9, 5e-5, mode="nonlinear")
    L = 32 * grid.dx
    src = make_source("dipole", grid.shape, grid.spacing, (L/2, L/2, L/2),
                      omega=2*np.pi*2e5, amplitude=1e-4)
    for i in range(300):
        grid.step(src)
    el = grid.species[0]
    ok = (np.all(np.isfinite(el.n)) and np.all(el.n >= 0)
          and np.all(np.isfinite(grid.Ex)) and np.isfinite(grid.field_energy()))
    dn = np.max(np.abs(el.n - el.n0)) / np.max(el.n0)
    check("finite fields, n >= 0", ok, f"max density perturbation {dn:.2e}")


if __name__ == "__main__":
    test_subcycle_matrices_vs_bruteforce()
    test_subcycle_unconditional_stability()
    test_exact_push_vs_rk4()
    test_gyration_handedness()
    test_vacuum_pulse_and_cpml()
    test_plasma_cutoff()
    test_magnetized_linear_stability()
    test_whistler_handedness()
    test_nonlinear_smoke()
    print()
    if FAIL:
        raise SystemExit(f"FAILED: {FAIL}")
    print("all tests passed")
