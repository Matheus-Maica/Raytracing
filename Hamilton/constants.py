import numpy as np
# Define physical constants.

C = 299_792_458 # Speed of light in m/s
mu_0 = 4 * np.pi * 1e-7 # Permeability of free space in H/m
epsilon_0 = 1 / (mu_0 * C**2) # Permittivity of free space in F/m
B_0 = 5e-5 # Earth's magnetic field strength in Tesla
m_e = 9.10938356e-31 # Electron mass in kg
q_e = 1.602176634e-19 # Electron charge in C