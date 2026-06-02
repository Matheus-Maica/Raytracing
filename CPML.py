import numpy as np
from constants import mu_0, epsilon_0

# A CPML (Convolutional Perfectly Matched Layer) implementation for absorbing boundaries in FDTD simulations.
class CPML:
    def __init__(
        self,
        grid_width,
        grid_height,
        thickness=10,
        alpha_max=0.1,
        kappa_max=5.0,
        grading_order=3,
        reflection_target=1e-8,
        cell_size=1.0
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.thickness = thickness
        self.grading_order = grading_order

        self.alpha_max = alpha_max
        self.kappa_max = kappa_max

        self.sigma_max = self.estimate_sigma_max(
            cell_size=cell_size,
            R=reflection_target
        )

        # CPML coefficient arrays

        self.b_x = np.ones((grid_height, grid_width))
        self.b_y = np.ones((grid_height, grid_width))

        self.c_x = np.zeros((grid_height, grid_width))
        self.c_y = np.zeros((grid_height, grid_width))

        self.kappa_x = np.ones((grid_height, grid_width))
        self.kappa_y = np.ones((grid_height, grid_width))

        # Auxiliary fields

        self.psi_Hz_x = np.zeros((grid_height, grid_width))
        self.psi_Hz_y = np.zeros((grid_height, grid_width))

        self.psi_Ex_y = np.zeros((grid_height, grid_width))
        self.psi_Ey_x = np.zeros((grid_height, grid_width))

    def estimate_sigma_max(self, cell_size, R=1e-8):
        eta0 = np.sqrt(mu_0 / epsilon_0)

        return (
            -(self.grading_order + 1)
            * np.log10(R)
            / (2 * eta0 * self.thickness * cell_size)
        )

    def compute_cpml_parameters(self, dt):

        sigma_x = np.zeros((self.grid_height, self.grid_width))
        sigma_y = np.zeros((self.grid_height, self.grid_width))

        alpha_x = np.zeros((self.grid_height, self.grid_width))
        alpha_y = np.zeros((self.grid_height, self.grid_width))

        #
        # X-direction profile
        #

        for i in range(self.thickness):

            d = (self.thickness - i) / self.thickness

            sigma = self.sigma_max * d**self.grading_order

            alpha = self.alpha_max * (1.0 - d)

            kappa = 1.0 + (self.kappa_max - 1.0) * d**self.grading_order

            #
            # Left boundary
            #

            sigma_x[:, i] = sigma
            alpha_x[:, i] = alpha
            self.kappa_x[:, i] = kappa

            #
            # Right boundary
            #

            sigma_x[:, -(i + 1)] = sigma
            alpha_x[:, -(i + 1)] = alpha
            self.kappa_x[:, -(i + 1)] = kappa

        #
        # Y-direction profile
        #

        for j in range(self.thickness):

            d = (self.thickness - j) / self.thickness

            sigma = self.sigma_max * d**self.grading_order

            alpha = self.alpha_max * (1.0 - d)

            kappa = 1.0 + (self.kappa_max - 1.0) * d**self.grading_order

            #
            # Top boundary
            #

            sigma_y[j, :] = sigma
            alpha_y[j, :] = alpha
            self.kappa_y[j, :] = kappa

            #
            # Bottom boundary
            #

            sigma_y[-(j + 1), :] = sigma
            alpha_y[-(j + 1), :] = alpha
            self.kappa_y[-(j + 1), :] = kappa

        #
        # Build b and c coefficients
        #

        mask_x = np.zeros(self.b_x.shape, dtype=bool)
        mask_x[:, :self.thickness] = True
        mask_x[:, -self.thickness:] = True

        self.b_x[mask_x] = np.exp(
            -(sigma_x[mask_x] / self.kappa_x[mask_x] + alpha_x[mask_x]) * dt / epsilon_0
        )

        mask_y = np.ones(self.b_y.shape, dtype=bool)
        mask_y[self.thickness:-self.thickness, :] = False  # Set middle rows to False

        self.b_y[mask_y] = np.exp(
            -(sigma_y[mask_y] / self.kappa_y[mask_y] + alpha_y[mask_y]) * dt / epsilon_0
        )

        mask_x = sigma_x > 0
        mask_y = sigma_y > 0

        self.c_x[mask_x] = (
            sigma_x[mask_x]
            / (
                self.kappa_x[mask_x]
                * (
                    sigma_x[mask_x]
                    + self.kappa_x[mask_x]
                    * alpha_x[mask_x]
                )
            )
            * (self.b_x[mask_x] - 1.0)
        )

        self.c_y[mask_y] = (
            sigma_y[mask_y]
            / (
                self.kappa_y[mask_y]
                * (
                    sigma_y[mask_y]
                    + self.kappa_y[mask_y]
                    * alpha_y[mask_y]
                )
            )
            * (self.b_y[mask_y] - 1.0)
        )

        return self.b_x, self.c_x, self.kappa_x, self.b_y, self.c_y, self.kappa_y, self.psi_Hz_x, self.psi_Hz_y, self.psi_Ex_y, self.psi_Ey_x