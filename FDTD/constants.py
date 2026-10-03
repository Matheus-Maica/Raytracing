import numpy as np

# Physical constants (SI units throughout).

C = 299_792_458.0                  # Speed of light [m/s]
mu_0 = 4 * np.pi * 1e-7            # Vacuum permeability [H/m]
epsilon_0 = 1 / (mu_0 * C**2)      # Vacuum permittivity [F/m]
k_B = 1.380649e-23                 # Boltzmann constant [J/K]

e_charge = 1.602176634e-19         # Elementary charge (positive) [C]
m_e = 9.10938356e-31               # Electron mass [kg]
m_p = 1.67262192e-27               # Proton mass [kg]

# Species charges are SIGNED. The electron's charge is negative; this sign is
# what sets the gyration handedness (and therefore which circular polarization
# propagates as the whistler mode), so it must not be dropped.
q_electron = -e_charge             # Electron charge [C]
q_ion = +e_charge                  # Singly-charged ion [C]

m_ion = 16 * m_p                   # O+ ion mass (dominant F-region ion) [kg]

# Earth-like background magnetic field magnitude [T]. The *direction* is a
# configurable vector on the Grid; only the magnitude lives here.
B_0 = 5e-5

# Kept for backward compatibility with the legacy 2-D code (positive
# elementary charge under its old name).
q_e = e_charge
