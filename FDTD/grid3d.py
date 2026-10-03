import numpy as np

from constants import C, mu_0, epsilon_0, k_B
from cpml3d import CPML3D
from plasma import (plasma_frequency_sq, coulomb_collision_frequency,
                    choose_subcycles, subcycle_matrices, exact_velocity_push)
from plasma_density import pml_taper

# Staggering convention (index [i,j,k], spacing d):
#   Ex,Jx: ((i+1/2)dx, j dy, k dz)      Hx: (i dx, (j+1/2)dy, (k+1/2)dz)
#   Ey,Jy: (i dx, (j+1/2)dy, k dz)      Hy: ((i+1/2)dx, j dy, (k+1/2)dz)
#   Ez,Jz: (i dx, j dy, (k+1/2)dz)      Hz: ((i+1/2)dx, (j+1/2)dy, k dz)
# Scalars (n_e, nu, fluid u) live on integer nodes (i,j,k).
#
# Time staggering: E at integer steps n, H and J at half steps n+1/2.
# One step():  H^{n-1/2} -> H^{n+1/2} (uses E^n)
#              J^{n-1/2} -> J^{n+1/2} (uses E^n, frozen over the window)
#              E^n       -> E^{n+1}   (uses H^{n+1/2}, J^{n+1/2}, J_src^{n+1/2})


def avg_fwd(A, axis):
    """Average A[i] and A[i+1] along axis -> value at i+1/2 (node -> edge)."""
    out = A.copy()
    lo = [slice(None)] * A.ndim
    hi = [slice(None)] * A.ndim
    lo[axis] = slice(0, -1)
    hi[axis] = slice(1, None)
    out[tuple(lo)] = 0.5 * (A[tuple(lo)] + A[tuple(hi)])
    return out


def avg_bwd(A, axis):
    """Average A[i-1] and A[i] along axis -> value at i-1/2 offset removed
    (edge -> node)."""
    out = A.copy()
    lo = [slice(None)] * A.ndim
    hi = [slice(None)] * A.ndim
    lo[axis] = slice(0, -1)
    hi[axis] = slice(1, None)
    out[tuple(hi)] = 0.5 * (A[tuple(lo)] + A[tuple(hi)])
    return out


def move_edge(A, from_axis, to_axis):
    """Interpolate an edge-family field to another edge family (4-point)."""
    return avg_bwd(avg_fwd(A, to_axis), from_axis)


class FluidSpecies:
    """One charged fluid species on integer nodes (nonlinear mode).

    Full (nonlinear) cold/warm fluid:
        dn/dt = -div(n u)
        du/dt = (q/m)(E + u x B) - nu u - (u.grad)u - (kT/m) grad(n)/n
    The E, B and nu parts are integrated exactly per step (rotation +
    drift-relaxation closed form); advection and pressure enter as a frozen
    acceleration over the step. J = q n u is deposited on the E edges.
    """

    def __init__(self, name, q, m, n0, T, spacing, taper, nu_lnLambda=13.0):
        self.name, self.q, self.m, self.T = name, q, m, T
        self.spacing = spacing
        self.taper = taper
        self.n0 = n0.copy()
        self.n = n0.copy()
        self.u = np.zeros(n0.shape + (3,))
        self.lnLambda = nu_lnLambda
        self.n_floor = 1e-3 * np.max(n0)
        self.clamp_count = 0

        # The imposed background profile n0 (taper, blobs, turbulence) is not
        # an equilibrium of these fluid equations: its pressure gradient
        # would immediately drive diffusion currents (worst at the PML taper,
        # where n falls orders of magnitude over a few cells) and ring up an
        # ambipolar sheath that buries the wave fields. Physically the
        # profile is maintained by forces outside the model (ambipolar
        # balance with the ions, production/loss), so we bake in the static
        # force that holds n0 in equilibrium and let only the *perturbation*
        # pressure act. Nonlinearity in the perturbations is fully retained.
        vth2 = k_B * self.T / self.m
        self.a_eq = [vth2 * g for g in np.gradient(
            np.log(np.maximum(n0, self.n_floor)), *spacing)]

    def push(self, E_nodes, B_nodes, dt):
        dx, dy, dz = self.spacing
        qm = self.q / self.m

        nu = coulomb_collision_frequency(self.n, self.T, self.lnLambda)

        # Slow forces, frozen over the step: convection + isothermal pressure.
        a_slow = np.zeros_like(self.u)
        gx = np.gradient(self.u[..., 0], dx, dy, dz)
        gy = np.gradient(self.u[..., 1], dx, dy, dz)
        gz = np.gradient(self.u[..., 2], dx, dy, dz)
        for c, g in enumerate((gx, gy, gz)):
            a_slow[..., c] = -(self.u[..., 0] * g[0] + self.u[..., 1] * g[1]
                               + self.u[..., 2] * g[2])
        vth2 = k_B * self.T / self.m
        n_safe = np.maximum(self.n, self.n_floor)
        gn = np.gradient(np.log(n_safe), dx, dy, dz)
        for c in range(3):
            # perturbation pressure only: a_eq cancels the background's
            # pressure gradient so the initial state is force-free
            a_slow[..., c] += self.a_eq[c] - vth2 * gn[c]

        F = qm * E_nodes + a_slow
        Omega = qm * B_nodes
        self.u = exact_velocity_push(self.u, F, nu, Omega, dt)

        # Guard: keep the fluid non-relativistic and quiet inside the PML.
        speed = np.linalg.norm(self.u, axis=-1, keepdims=True)
        cap = 0.3 * C
        over = speed > cap
        if np.any(over):
            self.clamp_count += int(np.count_nonzero(over))
            self.u = np.where(over, self.u * (cap / speed), self.u)
        self.u *= self.taper[..., None]

    def advect_density(self, dt):
        # NOTE: do NOT re-apply the PML taper to n here -- n0 already carries
        # it, and multiplying every step makes n decay geometrically in the
        # taper zone with u = 0, wrecking the a_eq force balance and driving
        # spurious sheath currents at the plasma edge. The fluid is kept
        # quiet near the PML by tapering u (in push), not n.
        dx, dy, dz = self.spacing
        fx = self.n * self.u[..., 0]
        fy = self.n * self.u[..., 1]
        fz = self.n * self.u[..., 2]
        div = (np.gradient(fx, dx, axis=0) + np.gradient(fy, dy, axis=1)
               + np.gradient(fz, dz, axis=2))
        n_new = np.clip(self.n - dt * div, 0.0, None)
        n_mid = 0.5 * (self.n + n_new)
        self.n = n_new
        return n_mid

    def deposit_current(self, n_mid, Jx, Jy, Jz):
        Jx += self.q * avg_fwd(n_mid * self.u[..., 0], 0)
        Jy += self.q * avg_fwd(n_mid * self.u[..., 1], 1)
        Jz += self.q * avg_fwd(n_mid * self.u[..., 2], 2)


class Grid3D:
    def __init__(self, shape, dx, n_e, B0_vec=(0.0, 0.0, 5e-5), T_e=1000.0,
                 lnLambda=13.0, mode="linear", courant=0.99,
                 omega_p_dt_limit=0.5, subcycle_target=0.1,
                 pml_thickness=10, mobile_ions=False,
                 ion_mass=None, ion_charge=None, T_i=1000.0):
        from constants import q_electron, m_e, q_ion, m_ion

        self.shape = tuple(shape)
        self.spacing = (dx, dx, dx)
        self.dx = dx
        self.mode = mode
        self.B0 = np.asarray(B0_vec, dtype=np.float64)
        self.pml_thickness = pml_thickness

        # --- time step ---
        # The leapfrog ADE scheme's exact stability condition combines the
        # curl (CFL) and plasma-oscillation terms:
        #     3 (c dt / dx)^2 + (omega_p_max dt / 2)^2 <= 1
        # (the plasma term eats into the Courant budget -- running at
        # courant ~ 1 with omega_p*dt ~ 0.4 on top, as the legacy code
        # effectively did, is unstable at the shortest grid wavelengths).
        # On top of that, omega_p_dt_limit caps omega_p*dt for accuracy.
        wp_sq = plasma_frequency_sq(n_e, q_electron, m_e)
        wp_max = np.sqrt(np.max(wp_sq))
        dt_stab = courant / np.sqrt(3.0 * (C / dx) ** 2 + (wp_max / 2) ** 2)
        dt_wp = omega_p_dt_limit / wp_max if wp_max > 0 else np.inf
        self.dt = min(dt_stab, dt_wp)
        self.dt_limited_by = ("CFL+plasma stability" if dt_stab <= dt_wp
                              else "omega_p accuracy limit")

        # --- medium (nodes), tapered to vacuum before the PML ---
        self.taper = pml_taper(self.shape, pml_thickness)
        self.n_e = n_e * self.taper
        self.T_e = T_e
        self.wp_sq = plasma_frequency_sq(self.n_e, q_electron, m_e)
        self.nu = coulomb_collision_frequency(self.n_e, T_e, lnLambda)

        self.Omega_e = (q_electron / m_e) * self.B0  # signed electron gyrovector
        self.Omega_mag = np.linalg.norm(self.Omega_e)

        # --- fields ---
        self.Ex = np.zeros(self.shape); self.Ey = np.zeros(self.shape); self.Ez = np.zeros(self.shape)
        self.Hx = np.zeros(self.shape); self.Hy = np.zeros(self.shape); self.Hz = np.zeros(self.shape)
        self.Jx = np.zeros(self.shape); self.Jy = np.zeros(self.shape); self.Jz = np.zeros(self.shape)

        self.cpml = CPML3D(self.shape, self.dt, dx, thickness=pml_thickness)
        self.psi = self.cpml.allocate_psi()

        if mode == "linear":
            self.M = choose_subcycles(np.max(self.nu), self.Omega_mag,
                                      self.dt, subcycle_target)
            X, G = subcycle_matrices(self.wp_sq, self.nu, self.Omega_e,
                                     self.dt, self.M)
            # Coefficient rows, interpolated from nodes to each edge family.
            self.XR, self.GR = {}, {}
            for axis, comp in enumerate("xyz"):
                self.XR[comp] = [avg_fwd(np.ascontiguousarray(X[..., axis, c]), axis)
                                 for c in range(3)]
                self.GR[comp] = [avg_fwd(np.ascontiguousarray(G[..., axis, c]), axis)
                                 for c in range(3)]
            del X, G
            self.species = []
        elif mode == "nonlinear":
            self.M = 1
            self.species = [FluidSpecies("electrons", q_electron, m_e,
                                         self.n_e, T_e, self.spacing,
                                         self.taper, lnLambda)]
            if mobile_ions:
                self.species.append(FluidSpecies(
                    "ions", ion_charge or q_ion, ion_mass or m_ion,
                    self.n_e.copy(), T_i, self.spacing, self.taper, lnLambda))
        else:
            raise ValueError(f"mode must be 'linear' or 'nonlinear', got {mode!r}")

        self.step_count = 0

    # ------------------------------------------------------------------ H
    def update_H(self):
        dt, (dx, dy, dz) = self.dt, self.spacing
        b, c, kap = self.cpml.b_h, self.cpml.c_h, self.cpml.kappa_h
        E_x, E_y, E_z = self.Ex, self.Ey, self.Ez
        psi = self.psi

        # Hx at (i, j+1/2, k+1/2): dHx/dt = (dEy/dz - dEz/dy)/mu0
        R = np.s_[:, :-1, :-1]
        dEz_dy = (E_z[:, 1:, :-1] - E_z[:, :-1, :-1]) / dy
        dEy_dz = (E_y[:, :-1, 1:] - E_y[:, :-1, :-1]) / dz
        psi["Hxy"][R] = b["y"][:, :-1, :] * psi["Hxy"][R] + c["y"][:, :-1, :] * dEz_dy
        psi["Hxz"][R] = b["z"][:, :, :-1] * psi["Hxz"][R] + c["z"][:, :, :-1] * dEy_dz
        self.Hx[R] += (dt / mu_0) * (dEy_dz / kap["z"][:, :, :-1] + psi["Hxz"][R]
                                     - dEz_dy / kap["y"][:, :-1, :] - psi["Hxy"][R])

        # Hy at (i+1/2, j, k+1/2): dHy/dt = (dEz/dx - dEx/dz)/mu0
        R = np.s_[:-1, :, :-1]
        dEx_dz = (E_x[:-1, :, 1:] - E_x[:-1, :, :-1]) / dz
        dEz_dx = (E_z[1:, :, :-1] - E_z[:-1, :, :-1]) / dx
        psi["Hyz"][R] = b["z"][:, :, :-1] * psi["Hyz"][R] + c["z"][:, :, :-1] * dEx_dz
        psi["Hyx"][R] = b["x"][:-1, :, :] * psi["Hyx"][R] + c["x"][:-1, :, :] * dEz_dx
        self.Hy[R] += (dt / mu_0) * (dEz_dx / kap["x"][:-1, :, :] + psi["Hyx"][R]
                                     - dEx_dz / kap["z"][:, :, :-1] - psi["Hyz"][R])

        # Hz at (i+1/2, j+1/2, k): dHz/dt = (dEx/dy - dEy/dx)/mu0
        R = np.s_[:-1, :-1, :]
        dEy_dx = (E_y[1:, :-1, :] - E_y[:-1, :-1, :]) / dx
        dEx_dy = (E_x[:-1, 1:, :] - E_x[:-1, :-1, :]) / dy
        psi["Hzx"][R] = b["x"][:-1, :, :] * psi["Hzx"][R] + c["x"][:-1, :, :] * dEy_dx
        psi["Hzy"][R] = b["y"][:, :-1, :] * psi["Hzy"][R] + c["y"][:, :-1, :] * dEx_dy
        self.Hz[R] += (dt / mu_0) * (dEx_dy / kap["y"][:, :-1, :] + psi["Hzy"][R]
                                     - dEy_dx / kap["x"][:-1, :, :] - psi["Hzx"][R])

    # ------------------------------------------------------------------ J
    def update_J_linear(self):
        """J^{n-1/2} -> J^{n+1/2} with E^n frozen at the window center.

        Each component is updated at its own edge; the other two J and E
        components are 4-point interpolated to that edge first.
        """
        Jx, Jy, Jz = self.Jx, self.Jy, self.Jz
        Ex, Ey, Ez = self.Ex, self.Ey, self.Ez

        Jy_x = move_edge(Jy, 1, 0); Jz_x = move_edge(Jz, 2, 0)
        Ey_x = move_edge(Ey, 1, 0); Ez_x = move_edge(Ez, 2, 0)
        Jx_y = move_edge(Jx, 0, 1); Jz_y = move_edge(Jz, 2, 1)
        Ex_y = move_edge(Ex, 0, 1); Ez_y = move_edge(Ez, 2, 1)
        Jx_z = move_edge(Jx, 0, 2); Jy_z = move_edge(Jy, 1, 2)
        Ex_z = move_edge(Ex, 0, 2); Ey_z = move_edge(Ey, 1, 2)

        XR, GR = self.XR, self.GR
        Jx_new = (XR["x"][0] * Jx + XR["x"][1] * Jy_x + XR["x"][2] * Jz_x
                  + GR["x"][0] * Ex + GR["x"][1] * Ey_x + GR["x"][2] * Ez_x)
        Jy_new = (XR["y"][0] * Jx_y + XR["y"][1] * Jy + XR["y"][2] * Jz_y
                  + GR["y"][0] * Ex_y + GR["y"][1] * Ey + GR["y"][2] * Ez_y)
        Jz_new = (XR["z"][0] * Jx_z + XR["z"][1] * Jy_z + XR["z"][2] * Jz
                  + GR["z"][0] * Ex_z + GR["z"][1] * Ey_z + GR["z"][2] * Ez)

        I3 = np.s_[1:-1, 1:-1, 1:-1]
        Jx[I3] = Jx_new[I3]; Jy[I3] = Jy_new[I3]; Jz[I3] = Jz_new[I3]

    def update_J_fluid(self):
        E_nodes = np.stack([avg_bwd(self.Ex, 0), avg_bwd(self.Ey, 1),
                            avg_bwd(self.Ez, 2)], axis=-1)
        B_nodes = self.B0 + mu_0 * np.stack(
            [avg_bwd(avg_bwd(self.Hx, 1), 2),
             avg_bwd(avg_bwd(self.Hy, 0), 2),
             avg_bwd(avg_bwd(self.Hz, 0), 1)], axis=-1)

        self.Jx.fill(0.0); self.Jy.fill(0.0); self.Jz.fill(0.0)
        for sp in self.species:
            sp.push(E_nodes, B_nodes, self.dt)
            n_mid = sp.advect_density(self.dt)
            sp.deposit_current(n_mid, self.Jx, self.Jy, self.Jz)
        # keep the electron density visible for diagnostics/plots
        self.n_e = self.species[0].n

    # ------------------------------------------------------------------ E
    def update_E(self, J_src):
        dt, (dx, dy, dz) = self.dt, self.spacing
        b, c, kap = self.cpml.b_n, self.cpml.c_n, self.cpml.kappa_n
        Hx, Hy, Hz = self.Hx, self.Hy, self.Hz
        psi = self.psi
        Jsx, Jsy, Jsz = J_src

        # Ex at (i+1/2, j, k): dEx/dt = (dHz/dy - dHy/dz - Jx)/eps0
        R = np.s_[:, 1:, 1:]
        dHz_dy = (Hz[:, 1:, 1:] - Hz[:, :-1, 1:]) / dy
        dHy_dz = (Hy[:, 1:, 1:] - Hy[:, 1:, :-1]) / dz
        psi["Exy"][R] = b["y"][:, 1:, :] * psi["Exy"][R] + c["y"][:, 1:, :] * dHz_dy
        psi["Exz"][R] = b["z"][:, :, 1:] * psi["Exz"][R] + c["z"][:, :, 1:] * dHy_dz
        self.Ex[R] += (dt / epsilon_0) * (
            dHz_dy / kap["y"][:, 1:, :] + psi["Exy"][R]
            - dHy_dz / kap["z"][:, :, 1:] - psi["Exz"][R]
            - self.Jx[R] - Jsx[R])

        # Ey at (i, j+1/2, k): dEy/dt = (dHx/dz - dHz/dx - Jy)/eps0
        R = np.s_[1:, :, 1:]
        dHx_dz = (Hx[1:, :, 1:] - Hx[1:, :, :-1]) / dz
        dHz_dx = (Hz[1:, :, 1:] - Hz[:-1, :, 1:]) / dx
        psi["Eyz"][R] = b["z"][:, :, 1:] * psi["Eyz"][R] + c["z"][:, :, 1:] * dHx_dz
        psi["Eyx"][R] = b["x"][1:, :, :] * psi["Eyx"][R] + c["x"][1:, :, :] * dHz_dx
        self.Ey[R] += (dt / epsilon_0) * (
            dHx_dz / kap["z"][:, :, 1:] + psi["Eyz"][R]
            - dHz_dx / kap["x"][1:, :, :] - psi["Eyx"][R]
            - self.Jy[R] - Jsy[R])

        # Ez at (i, j, k+1/2): dEz/dt = (dHy/dx - dHx/dy - Jz)/eps0
        R = np.s_[1:, 1:, :]
        dHy_dx = (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) / dx
        dHx_dy = (Hx[1:, 1:, :] - Hx[1:, :-1, :]) / dy
        psi["Ezx"][R] = b["x"][1:, :, :] * psi["Ezx"][R] + c["x"][1:, :, :] * dHy_dx
        psi["Ezy"][R] = b["y"][:, 1:, :] * psi["Ezy"][R] + c["y"][:, 1:, :] * dHx_dy
        self.Ez[R] += (dt / epsilon_0) * (
            dHy_dx / kap["x"][1:, :, :] + psi["Ezx"][R]
            - dHx_dy / kap["y"][:, 1:, :] - psi["Ezy"][R]
            - self.Jz[R] - Jsz[R])

    # ------------------------------------------------------------------ step
    def step(self, source=None):
        self.update_H()
        if self.mode == "linear":
            self.update_J_linear()
        else:
            self.update_J_fluid()

        t_half = (self.step_count + 0.5) * self.dt
        if source is not None:
            J_src = source.currents(t_half)
        else:
            z = np.broadcast_to(0.0, self.shape)
            J_src = (z, z, z)
        self.update_E(J_src)
        self.step_count += 1

    # ------------------------------------------------------------- diagnostics
    def field_energy(self):
        cell = self.dx ** 3
        We = 0.5 * epsilon_0 * (self.Ex**2 + self.Ey**2 + self.Ez**2).sum() * cell
        Wm = 0.5 * mu_0 * (self.Hx**2 + self.Hy**2 + self.Hz**2).sum() * cell
        return We + Wm

    def E_magnitude(self):
        # interpolate all components to nodes before combining
        Exn = avg_bwd(self.Ex, 0); Eyn = avg_bwd(self.Ey, 1); Ezn = avg_bwd(self.Ez, 2)
        return np.sqrt(Exn**2 + Eyn**2 + Ezn**2)

    def summary(self):
        wp = np.sqrt(np.max(self.wp_sq))
        lines = [
            f"grid {self.shape}, dx = {self.dx:.3g} m "
            f"(domain {tuple(round(N*self.dx, 1) for N in self.shape)} m)",
            f"dt = {self.dt:.4g} s (limited by {self.dt_limited_by})",
            f"mode = {self.mode}, sub-cycles M = {self.M}",
            f"max omega_p = {wp:.4g} rad/s -> omega_p*dt = {wp*self.dt:.3f} "
            f"(leapfrog ADE needs < 2, accuracy wants < ~0.5)",
            f"|Omega_e| = {self.Omega_mag:.4g} rad/s -> Omega*dt = "
            f"{self.Omega_mag*self.dt:.3f} (resolved by sub-cycling: "
            f"Omega*dt_c = {self.Omega_mag*self.dt/self.M:.3f})",
            f"max nu = {np.max(self.nu):.4g} 1/s -> nu*dt_c = "
            f"{np.max(self.nu)*self.dt/self.M:.2e}",
        ]
        return "\n".join(lines)
