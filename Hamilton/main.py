import numpy as np
from scipy.integrate import solve_ivp
from constants import q_e, m_e, B_0, epsilon_0
import matplotlib.pyplot as plt

class Ray:
    def __init__(self, electron_density, omega):
        self.omega_p = lambda r : np.sqrt(electron_density(r) * q_e**2 / (epsilon_0 * m_e))
        self.omega = omega

        self.X = lambda r : (self.omega_p(r) / self.omega)**2
        self.Y = q_e * B_0 / (self.omega * m_e)

    def q(self, r, u):
        alpha = 1 - self.X(r) - self.Y**2
        beta = (self.Y**2 + (self.Y*u[2])**2) / 2
        gamma = -(self.Y*u[2])**2

        return -beta/alpha + np.sqrt((beta/alpha)**2 - gamma/alpha) - 1

    def J(self, r, u):
        return 2 * (2 * (1 - self.X(r) - self.Y**2)*(self.q(r, u) + 1) + self.Y**2 + (self.Y*u[2])**2)
    
    def K(self, r, u):
        return -2 * self.q(r, u) * self.X(r) * self.Y * u[2]
    
    def L(self, r, u):
        return (1 - self.Y**2) * (self.q(r, u) + 1)**2 - 2*(1 - self.X(r) - self.Y**2)*(self.q(r, u) + 1) - self.Y**2

    def solve_system(
        self,
        t_span,
        r0,
        u0,
        rtol=1e-8,
        atol=1e-10
    ):
        def grad_omega_p(omega_p, r0):
            h = max(1e-6, 1e-6*np.linalg.norm(r0))

            grad = np.zeros(3)
            for i in range(3):
                dr = np.zeros(3)
                dr[i] = h

                grad[i] = (omega_p(r0 + dr) - omega_p(r0 - dr)) / (2 * h)

            return grad

        def rhs(t, y):
            r = y[:3]
            u = y[3:]

            omega_p = self.omega_p(r)
            grad_wp = grad_omega_p(self.omega_p, r)

            drdt = self.J(r, u) * u - self.K(r, u) * np.array([0.0, 0.0, self.Y])

            dudt = 2.0 * self.L(r, u) * (omega_p / self.omega) * grad_wp

            return np.concatenate([drdt, dudt])

        y0 = np.concatenate([r0, u0])

        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method="RK45",
            t_eval = np.linspace(0, 10, 100000),
            rtol=rtol,
            atol=atol
        )

        return sol
    

def plot_ray(positions):
    positions = np.asarray(positions)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        linewidth=2
    )

    # Mark start and end points
    ax.scatter(*positions[0], s=50, label="Start")
    ax.scatter(*positions[-1], s=50, label="End")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title("Ray Trajectory")
    ax.legend()

    plt.tight_layout()
    plt.show()

def electron_density(r):
    x, y, z = r

    # Prevent negative altitudes
    h = max(y, 0.0)

    # -------------------------
    # Vertical profile
    # -------------------------

    peak_altitude = 300e3      # 300 km
    scale_height = 80e3        # 80 km
    peak_density = 1e12        # m^-3

    vertical = peak_density * np.exp(
        -((h - peak_altitude) / scale_height)**2
    )

    # -------------------------
    # Horizontal structure
    # -------------------------

    horizontal = (
        1
        + 0.20*np.sin(2*np.pi*x/(500e3))
        + 0.15*np.cos(2*np.pi*z/(700e3))
    )

    # -------------------------
    # Localized density enhancement
    # -------------------------

    blob = 1 + 0.5*np.exp(
        -((x - 200e3)**2 + (z + 100e3)**2)/(100e3)**2
    )

    return vertical * horizontal * blob

ray = Ray(electron_density, 2*np.pi*5e6)

u0 = np.array([0,1,1], dtype=float)
u0 /= np.linalg.norm(u0)

tracer = ray.solve_system(t_span=(0, 1e5), r0=np.array([1, 0, 0]), u0=u0)

plot_ray(tracer.y[:3, :].T)