import numpy as np
import matplotlib.pyplot as plt

from animator import FieldAnimator
from CPML import CPML
from constants import C, mu_0, epsilon_0, B_0, m_e, q_e
from plasma_density import ionosphere_electron_density

cyclotron_freq = q_e * B_0 / m_e # Cyclotron frequency in rad/s

class Source:
    def __init__(self, Q, R, omega, pos):
        self.Q = Q
        self.R = R
        self.omega = omega
        self.pos = pos


class Grid:
    def __init__(self, width, height, dt=1e-9):
        self.width = width
        self.height = height

        self.dt = dt # Time step in seconds
        self.dx = 1.01 * np.sqrt(2) * dt * C # Spatial step in meters (can be adjusted based on the desired resolution)
        self.dy = 1.01 * np.sqrt(2) * dt * C # Spatial step in meters (can be adjusted based on the desired resolution)

        # Stability condition for FDTD: dt < 1 / (C * sqrt(1/dx^2 + 1/dy^2))
        if self.dt >= 1 / (C * np.sqrt(1 / self.dx**2 + 1 / self.dy**2)):
            raise ValueError("Time step dt is too large for stability. Reduce dt or increase spatial resolution.")

        # Initialize electric and magnetic field grids.
        self.Ex = np.zeros((height, width)) # Electric field in x-direction
        self.Ey = np.zeros((height, width)) # Electric field in y-direction

        self.Hz = np.zeros((height, width)) # Magnetic field in z-direction

        # Initialize current density grids
        self.Jx = np.zeros((height, width)) # Current density in x-direction
        self.Jy = np.zeros((height, width)) # Current density in y-direction

        # Additional plasma parameters
        self.n_e = ionosphere_electron_density(height, width, size=height*width*self.dx*self.dy, seed=42) # Width and height of grid, physical area, and The answer to the Ultimate Question of Life, The Universe, and Everything.
        self.nu = np.zeros((height, width)) # Collision frequency

        # CPML parameters
        cpml = CPML(grid_height=height, grid_width=width, cell_size=self.dx)

        self.b_x, self.c_x, self.kappa_x, self.b_y, self.c_y, self.kappa_y, self.psi_Hz_x, self.psi_Hz_y, self.psi_Ex_y, self.psi_Ey_x = cpml.compute_cpml_parameters(self.dt)

        self.X, self.G = self.compute_plasma_parameters(M=100, T=1000, lnLambda=10)

    def compute_plasma_parameters(self, M=20, T=1000, lnLambda=10):
        # Compute plasma frequency and collision frequency based on electron density
        self.omega_p = self.n_e * q_e**2 / (epsilon_0 * m_e) # Plasma frequency in rad/s
        self.nu = 2.9e-6 * self.n_e * lnLambda / (T**1.5) # Collision frequency in Hz, where T is the electron temperature in K

        dt_c = self.dt / M # Sub-time step for plasma updates

        # Next, compute G and X matrices, which are used in the update equations for the current field.
        # For an explanation of these matrices, see 10.1109/TAP.2018.2847601
        A = np.zeros(self.nu.shape + (2, 2), dtype=self.nu.dtype)

        A[..., 0, 0] = 1 + self.nu * dt_c / 2
        A[..., 0, 1] = -(cyclotron_freq * dt_c / 2)
        A[..., 1, 0] = cyclotron_freq * dt_c / 2
        A[..., 1, 1] = 1 + self.nu * dt_c / 2

        B = np.zeros(self.nu.shape + (2, 2), dtype=self.nu.dtype)

        B[..., 0, 0] = 1 - self.nu * dt_c / 2
        B[..., 0, 1] = cyclotron_freq * dt_c / 2
        B[..., 1, 0] = -cyclotron_freq * dt_c / 2
        B[..., 1, 1] = 1 - self.nu * dt_c / 2

        A_inv = np.linalg.inv(A)
        C = A_inv @ B

        I = np.eye(2)
        I = I.reshape(1, 1, 2, 2)  # broadcast over grid

        C_power = C.copy()
        X = None

        sum_Ck = np.zeros_like(C)

        # k = 1 to M-2
        for k in range(1, M):
            if k == M - 1:
                X = C_power.copy()   # C^(M-1)
            elif k <= M - 2:
                sum_Ck += C_power

            C_power = C_power @ C

        omega_p_squared_dt_c = epsilon_0 * self.omega_p * dt_c
        G = (I + sum_Ck) @ (A_inv * omega_p_squared_dt_c[..., None, None])

        return X, G
    
    def inject_source(self, i, source):
        x_center = source.pos[0] * self.dx
        y_center = source.pos[1] * self.dy

        q = source.Q
        R = source.R
        omega = source.omega
        t = i * self.dt
        sigma = 5 * self.dx

        def Jx_density(ii, j):
            Jx_source = -q * R * omega / (2 * np.pi * sigma**2) * np.sin(omega * t) * np.exp(-((ii*self.dx - x_center - R*np.cos(omega * t))**2 + (j*self.dy - y_center - R*np.sin(omega * t))**2) / (2 * sigma**2))

            return Jx_source
        
        def Jy_density(ii, j):
            Jy_source = q * R * omega / (2 * np.pi * sigma**2) * np.cos(omega * t) * np.exp(-((ii*self.dx - x_center - R*np.cos(omega * t))**2 + (j*self.dy - y_center - R*np.sin(omega * t))**2) / (2 * sigma**2))

            return Jy_source

        Jx_source = (1 - np.exp(-t*omega/(2*np.pi * 10))) * np.fromfunction(Jx_density, (self.height, self.width), dtype=np.float64)
        Jy_source = (1 - np.exp(-t*omega/(2*np.pi * 10))) * np.fromfunction(Jy_density, (self.height, self.width), dtype=np.float64)

        return Jx_source, Jy_source

    def update_J(self, i, source):
        interior = np.s_[1:-1, 1:-1]

        def avg4_Y(A):
            A_new = A.copy()

            # interior region only (avoid boundaries completely)
            A_new[interior] = (
                A[interior] +
                A[1:-1, 2:] +
                A[0:-2, 1:-1] +
                A[0:-2, 2:]
            ) / 4.0

            return A_new

        def avg4_X(A):
            A_new = A.copy()

            # interior region (exclude top, bottom, left, right edges)
            A_new[interior] = (
                A[interior] +   # A[i,j]
                A[1:-1, 0:-2] +   # A[i-1,j]
                A[2:, 0:-2] +   # A[i-1,j+1]
                A[2:, 1:-1]     # A[i,j+1]
            ) / 4.0

            return A_new
        
        Jx_avg = avg4_X(self.Jx)
        Jy_avg = avg4_Y(self.Jy)
        Ex_avg = avg4_X(self.Ex)
        Ey_avg = avg4_Y(self.Ey)

        X00 = self.X[..., 0, 0]
        X01 = self.X[..., 0, 1]
        X10 = self.X[..., 1, 0]
        X11 = self.X[..., 1, 1]

        G00 = self.G[..., 0, 0]
        G01 = self.G[..., 0, 1]
        G10 = self.G[..., 1, 0]
        G11 = self.G[..., 1, 1]

        # --- Jx update ---
        Jx_new = (
            G00 * self.Ex +
            G01 * Ey_avg +
            X00 * self.Jx +
            X01 * Jy_avg
        )

        # --- Jy update ---
        Jy_new = (
            G10 * Ex_avg +
            G11 * self.Ey +
            X10 * Jx_avg +
            X11 * self.Jy
        )

        self.Jx[interior] = Jx_new[interior]
        self.Jy[interior] = Jy_new[interior]
        
        Jx_source, Jy_source = self.inject_source(i, source)

        source_region = 5 * source.R
        row_start = int(max(0, source.pos[0] - source_region))
        row_end = int(source.pos[0] + source_region + 1)
        col_start = int(max(0, source.pos[1] - source_region))
        col_end = int(source.pos[1] + source_region + 1)

        self.Jx[row_start:row_end, col_start:col_end] += Jx_source[row_start:row_end, col_start:col_end]
        self.Jy[row_start:row_end, col_start:col_end] += Jy_source[row_start:row_end, col_start:col_end]

    def update_Hz(self):
        interior = np.s_[1:-1, 1:-1]

        self.Hz[interior] += (
            self.dt / mu_0
            * (
                (self.Ex[2:, 1:-1] - self.Ex[interior])
                / (self.kappa_y[interior] * self.dy)
                + self.psi_Hz_y[interior]
            )
            - self.dt / mu_0
            * (
                (self.Ey[1:-1, 2:] - self.Ey[interior])
                / (self.kappa_x[interior] * self.dx)
                + self.psi_Hz_x[interior]
            )
        )

    def update_E(self):
        interior = np.s_[1:-1, 1:-1]

        self.Ex[interior] += (
            -self.dt / epsilon_0 * (self.Jx[interior])
            + self.dt / epsilon_0
            * (
                (self.Hz[interior] - self.Hz[0:-2, 1:-1])
                / (self.kappa_y[interior] * self.dy)
                + self.psi_Ex_y[interior]
            )
        )

        self.Ey[interior] += (
            -self.dt / epsilon_0 * (self.Jy[interior])
            - self.dt / epsilon_0
            * (
                (self.Hz[interior] - self.Hz[1:-1, 0:-2])
                / (self.kappa_x[interior] * self.dx)
                + self.psi_Ey_x[interior]
            )
        )

    def update_CPML_memory_H(self):
        interior = np.s_[1:-1, 1:-1]
        self.psi_Hz_x[interior] = self.b_x[interior] * self.psi_Hz_x[interior] + self.c_x[interior] * (self.Ey[1:-1, 2:] - self.Ey[interior]) / self.dx
        self.psi_Hz_y[interior] = self.b_y[interior] * self.psi_Hz_y[interior] + self.c_y[interior] * (self.Ex[2:, 1:-1] - self.Ex[interior]) / self.dy

    def update_CPML_memory_E(self):
        interior = np.s_[1:-1, 1:-1]
        self.psi_Ex_y[interior] = self.b_y[interior] * self.psi_Ex_y[interior] + self.c_y[interior] * (self.Hz[interior] - self.Hz[0:-2, 1:-1]) / self.dy
        self.psi_Ey_x[interior] = self.b_x[interior] * self.psi_Ey_x[interior] + self.c_x[interior] * (self.Hz[interior] - self.Hz[1:-1, 0:-2]) / self.dx

WIDTH = 300
HEIGHT = 300


SOURCE_FREQUENCY = 1e9
SOURCE_CHARGE = 1e-12
SOURCE_DIPOLE_RADIUS = 5

grid = Grid(width=WIDTH, height=HEIGHT, dt=1e-11)
source = Source(Q=SOURCE_CHARGE, R=SOURCE_DIPOLE_RADIUS*grid.dx, omega=2*np.pi*SOURCE_FREQUENCY, pos=(WIDTH//2, HEIGHT//2))

print(f"Grid's physical size: {grid.dx * grid.width}m by {grid.dy * grid.height}m")
print(f"Cell size: dx={grid.dx}, dy={grid.dy}")
print(f"Resulting free space wavelength: {C / SOURCE_FREQUENCY}")
print(f"Max frequency that can exist vs. source frequency: {1 / (2 * grid.dt)} vs {SOURCE_FREQUENCY}")

print(f"COLLISION STABILITY CONDITION: NU * DT = {np.max(grid.nu) * grid.dt} -> SHOULD BE LESS THAN 0.1")
print(f"PLASMA STABILITY: {np.sqrt(1 - (np.max(np.sqrt(grid.omega_p)) * grid.dt / 2)**2)} > 0.9")


animator = FieldAnimator()

for i in range(400000):
    grid.update_CPML_memory_H()
    grid.update_Hz()
    grid.update_CPML_memory_E()
    grid.update_J(i, source=source)
    grid.update_E()

    if i % 10 == 0:
        plt.clf()

        # dFx_dy, dFx_dx = np.gradient(grid.Ex, grid.dy, grid.dx, edge_order=2)
        # dFy_dy, dFy_dx = np.gradient(grid.Ey, grid.dy, grid.dx, edge_order=2)

        # divergence = dFx_dx + dFy_dy

        # field = divergence
        field = np.sqrt(grid.Ex**2 + grid.Ey**2)

        plt.imshow(
            field.T,
            origin='lower',
            cmap='RdBu',
            vmin=-0.1,
            vmax=0.1
        )

        plt.title(f"Step {i}")
        plt.pause(0.001)

    # animator.add_frame(grid.Ex, grid.Ey)

# animator.animate(interval=1)