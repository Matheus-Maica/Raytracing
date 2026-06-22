import numpy as np
from scipy.integrate import solve_ivp
from constants import q_e, m_e, B_0, epsilon_0
from visualization import plot_ray
from ionosphere import Ionosphere


class Ray:
    def __init__(self, omega, MODE, electron_density):
        self.omega = omega
        self.MODE = MODE
        self.electron_density = electron_density

        # Base physical constants used in the Haselgrove equations.
        self.X_factor = q_e**2 / (epsilon_0 * m_e * self.omega**2)
        self.U_factor = q_e**2 / (epsilon_0 * m_e * self.omega**2)
        self.Y = q_e * B_0 / (self.omega * m_e)

    def grad_electron_density(self, x, y, z, h=1.0):
        dNdx = (self.electron_density(x + h, y, z) - self.electron_density(x - h, y, z)) / (2 * h)
        dNdy = (self.electron_density(x, y + h, z) - self.electron_density(x, y - h, z)) / (2 * h)
        dNdz = (self.electron_density(x, y, z + h) - self.electron_density(x, y, z - h)) / (2 * h)
        return dNdx, dNdy, dNdz

    def _haselgrove_physical_rhs(self, tau, state6):
        x, y, z, ux, uy, uz = state6

        dNdx, dNdy, dNdz = self.grad_electron_density(x, y, z)

        N = self.electron_density(x, y, z)
        X = N * self.X_factor
        Y = self.Y

        a = 1.0 - X - Y**2
        b = Y * (1.0 + uy**2)
        c = -uy**2

        disc = b**2 - 4.0 * a * c
        disc = np.maximum(disc, 0.0)
        p = (-b + self.MODE * np.sqrt(disc)) / (2.0 * a)

        J = 2.0 * (2.0 * a * p + Y * (1.0 - c))
        K = -2.0 * X * uy * (p * Y - 1.0)
        L = Y * (1.0 - Y**2) * p**2 - 2.0 * a * p - Y

        rdot = J * np.array([ux, uy, uz], dtype=float) - K * np.array([0.0, Y, 0.0], dtype=float)
        udot = (L * self.U_factor) * np.array([dNdx, dNdy, dNdz], dtype=float)

        return np.concatenate([rdot, udot])

    def _state_jacobian_fd(self, tau, state6, rel_step=1e-6, abs_step=1e-8):
        state6 = np.asarray(state6, dtype=float)
        n = state6.size
        J = np.zeros((n, n), dtype=float)

        f0 = self._haselgrove_physical_rhs(tau, state6)
        for j in range(n):
            step = abs_step + rel_step * max(1.0, abs(state6[j]))
            d = np.zeros(n, dtype=float)
            d[j] = step

            fp = self._haselgrove_physical_rhs(tau, state6 + d)
            fm = self._haselgrove_physical_rhs(tau, state6 - d)
            J[:, j] = (fp - fm) / (2.0 * step)

        return J

    def haselgrove_rhs(self, tau, state):
        state = np.asarray(state, dtype=float)
        physical = state[:6]
        s = state[6]
        phi = state[7:].reshape(6, 6)

        f = self._haselgrove_physical_rhs(tau, physical)
        M = self._state_jacobian_fd(tau, physical)
        dphi = M @ phi

        dxdt, dydt, dzdt, duxdt, duydt, duzdt = f
        dsdt = np.linalg.norm(f[:3])

        return np.concatenate([
            np.array([dxdt, dydt, dzdt, duxdt, duydt, duzdt, dsdt], dtype=float),
            dphi.reshape(-1),
        ])

    def solve_haselgrove(self, r0, u0, max_length):
        """Integrate the ray and its dynamic-ray-tracing tangent map."""
        state0 = np.concatenate([
            np.asarray(r0, dtype=float),
            np.asarray(u0, dtype=float),
            np.array([0.0], dtype=float),
            np.eye(6, dtype=float).reshape(-1),
        ])

        event = self.stop_at_length(max_length)

        return solve_ivp(
            self.haselgrove_rhs,
            (0.0, 1e9),
            state0,
            method="RK45",
            max_step=250.0,
            events=event,
            dense_output=True,
        )

    def stop_at_length(self, target_length):
        def event(tau, state):
            s = state[6]
            return s - target_length

        event.terminal = True
        event.direction = 1
        return event

    def extract_phi(self, solution):
        """Return Phi(t) for every stored solver step as an array of shape (n, 6, 6)."""
        phis = solution.y[7:, :].T.reshape(-1, 6, 6)
        return phis

    def ray_tube_jacobian(self, solution):
        """A simple ray-tube expansion factor from the position block det(d x / d x0)."""
        phis = self.extract_phi(solution)
        return np.array([np.linalg.det(phi[:3, :3]) for phi in phis])


iono = Ionosphere(Nx=200, Ny=200, Nz=400, dx=2.0, dy=2.0, dz=2.0)

ray = Ray(
    omega=2 * np.pi * 10 * 1e6,
    MODE=1,
    electron_density=iono.electron_density,
)

r0 = np.array([0.0, 0.0, 0.0], dtype=float)
u0 = np.array([0.0, 1.0, 1.0], dtype=float) / np.sqrt(2.0)

trajectory = ray.solve_haselgrove(r0, u0, 800e3)

# Optional diagnostic: the ray-tube Jacobian used in amplitude transport.
J_tube = ray.ray_tube_jacobian(trajectory)

# Prevent division by zero near caustics
eps = 1e-12
J_safe = np.maximum(np.abs(J_tube), eps)

amplitudes = np.sqrt(J_safe[0] / J_safe)

amplitudes /= amplitudes.max()

plot_ray(
    trajectory.y[:3, :].T,
    amplitudes=amplitudes,
)