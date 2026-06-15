import numpy as np
from scipy.integrate import solve_ivp
from constants import q_e, m_e, B_0, epsilon_0
from visualization import plot_ray


class Ray:
    def __init__(self, omega, MODE, electron_density):
        self.omega = omega
        self.MODE = MODE
        self.electron_density = electron_density

    def grad_electron_density(self, x, y, z, h=100.0):
        dNdx = (self.electron_density(x + h, y, z) - self.electron_density(x - h, y, z)) / (2 * h)
        dNdy = (self.electron_density(x, y + h, z) - self.electron_density(x, y - h, z)) / (2 * h)
        dNdz = (self.electron_density(x, y, z + h) - self.electron_density(x, y, z - h)) / (2 * h)
        
        return dNdx, dNdy, dNdz

    def haselgrove_rhs(self, tau, state):
        x, y, z, ux, uy, uz = state

        dNdx, dNdy, dNdz = self.grad_electron_density(x, y, z)

        X = self.electron_density(x, y, z) * q_e**2 / (epsilon_0 * m_e * self.omega**2)
        Y = q_e * B_0 / (self.omega * m_e)

        cosu = uz

        a = 1.0 - X - Y**2
        b = Y * (1.0 + cosu**2)
        c = -cosu**2

        p = (-b + self.MODE * np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        J = 2 * (2*a*p + Y*(1 - c))
        K = -2 * X * cosu * (p * Y - 1.0)
        L = Y * (1.0 - Y**2) * p**2 - 2 * a * p - Y

        dxdt = J * ux
        dydt = J * uy
        dzdt = J * uz - K * Y

        duxdt = L * q_e**2 / (epsilon_0 * m_e * self.omega) * dNdx
        duydt = L * q_e**2 / (epsilon_0 * m_e * self.omega) * dNdy
        duzdt = L * q_e**2 / (epsilon_0 * m_e * self.omega) * dNdz

        return [dxdt, dydt, dzdt, duxdt, duydt, duzdt]
    
    def solve_haselgrove(self, r0, u0):
        x0, y0, z0 = r0
        ux0, uy0, uz0 = u0

        state0 = [x0, y0, z0, ux0, uy0, uz0]
        tau_span = (0.0, 10000.0)

        return solve_ivp(
            self.haselgrove_rhs,
            tau_span,
            state0,
            method="RK45",
            max_step=1.0,
            dense_output=True,
        )
    
def electron_density(x, y, z):
    Nm = 1.0e11
    ym = 300 * 1000
    H = 50.0 * 1000
    yp = (y - ym) / H
    return Nm * np.exp(0.5 * (1.0 - yp - np.exp(-yp)))

omega = 2 * np.pi * 10 * 1e+6

ray = Ray(omega, MODE=1, electron_density=electron_density)

r0 = np.array([0.0, 0.0, 0.0])
u0 = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)

trajectory = ray.solve_haselgrove(r0, u0)

plot_ray(trajectory.y[:3, :].T)