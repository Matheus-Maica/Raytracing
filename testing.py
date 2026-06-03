import numpy as np

def current_density(x_grid, y_grid, q, R, omega, t):
    # particle position
    x0 = R * np.cos(omega * t)
    y0 = R * np.sin(omega * t)

    # particle velocity
    vx = -R * omega * np.sin(omega * t)
    vy =  R * omega * np.cos(omega * t)

    # initialize fields
    Jx = np.zeros_like(x_grid)
    Jy = np.zeros_like(y_grid)

    # find nearest grid point (delta approximation)
    idx = np.argmin((x_grid - x0)**2 + (y_grid - y0)**2)

    # convert flat index to 2D indices
    i, j = np.unravel_index(idx, x_grid.shape)

    # deposit current
    Jx[i, j] = q * vx
    Jy[i, j] = q * vy

    return Jx, Jy




Jx, Jy = current_density(300, 300, 1.0, 5.0, 2*np.pi*1e6, 0.5e-6)