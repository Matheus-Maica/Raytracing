import numpy as np

def ionosphere_electron_density(height, width, size, seed=None):
    """Legacy 2-D synthetic ionosphere (kept for the 2-D code in legacy_2d/)."""
    if seed is not None:
        np.random.seed(seed)

    L = np.sqrt(size)
    x = np.linspace(0, L, width)
    y = np.linspace(0, L, height)
    X, Y = np.meshgrid(x, y, indexing="ij")

    n0 = 8e11
    grad = 1 + 0.15 * (X / L) - 0.10 * (Y / L)
    background = n0 * grad

    noise = np.random.randn(width, height)
    kx = np.fft.fftfreq(width)[:, None]
    ky = np.fft.fftfreq(height)[None, :]
    k2 = kx**2 + ky**2
    corr_length = 0.08 * L
    k0 = 1.0 / corr_length
    filter_kernel = np.exp(-k2 * (k0**2))
    turbulence = np.fft.ifft2(np.fft.fft2(noise) * filter_kernel).real
    turbulence = turbulence / np.std(turbulence)
    turbulence_field = 0.2 * n0 * turbulence

    blobs = np.zeros_like(background)
    n_blobs = np.random.randint(3, 8)
    for _ in range(n_blobs):
        x0 = np.random.uniform(0, L)
        y0 = np.random.uniform(0, L)
        amp = np.random.uniform(-0.4, 5) * n0
        sigma = np.random.uniform(0.03 * L, 0.12 * L)
        blobs += amp * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    return np.clip(background + turbulence_field + blobs, 1e9, None)


def ionosphere_electron_density_3d(shape, extent, n0=8e11, seed=None,
                                   turbulence_amp=0.2, blob_range=(3, 8),
                                   blob_amp=(-0.4, 3.0), grad=(0.15, -0.10, 0.0)):
    """Synthetic 3-D ionospheric electron density [m^-3].

    shape  : (Nx, Ny, Nz) grid points
    extent : (Lx, Ly, Lz) physical size in meters
    n0     : background density
    Components: a gentle large-scale linear gradient, spectrally-correlated
    turbulence (Gaussian-filtered white noise), and a few Gaussian blobs /
    depletions mimicking plasma irregularities.
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = shape
    Lx, Ly, Lz = extent

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    background = n0 * (1 + grad[0] * X / Lx + grad[1] * Y / Ly + grad[2] * Z / Lz)

    # Correlated turbulence: filter white noise with a Gaussian in k-space.
    noise = rng.standard_normal(shape)
    kx = np.fft.fftfreq(Nx, d=Lx / Nx)[:, None, None]
    ky = np.fft.fftfreq(Ny, d=Ly / Ny)[None, :, None]
    kz = np.fft.fftfreq(Nz, d=Lz / Nz)[None, None, :]
    k2 = kx**2 + ky**2 + kz**2
    corr_length = 0.08 * min(Lx, Ly, Lz)
    filter_kernel = np.exp(-k2 * corr_length**2)
    turb = np.fft.ifftn(np.fft.fftn(noise) * filter_kernel).real
    std = np.std(turb)
    if std > 0:
        turb = turb / std
    turbulence_field = turbulence_amp * n0 * turb

    blobs = np.zeros(shape)
    for _ in range(rng.integers(*blob_range)):
        c = np.array([rng.uniform(0, Lx), rng.uniform(0, Ly), rng.uniform(0, Lz)])
        amp = rng.uniform(*blob_amp) * n0
        sig = rng.uniform(0.05, 0.15) * min(Lx, Ly, Lz)
        r2 = (X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2
        blobs += amp * np.exp(-r2 / (2 * sig**2))

    return np.clip(background + turbulence_field + blobs, 0.0, None)


def blob_structures_3d(shape, extent, n_bg, n_blob_peak, n_blobs=4,
                       region=(0.40, 0.68), sigma_frac=(0.05, 0.09),
                       seed=None):
    """Localized overdense blobs in a transparent background: the scattering
    scenario. Background n_bg is chosen underdense for the wave (transparent)
    while blob cores exceed the critical density (opaque), so each blob
    carries a critical layer where omega = omega_p.

    Blobs are placed inside `region` (fractions of the x-extent) and away
    from the y/z walls, leaving a clean incident-wave region upstream and a
    transmission region downstream. Returns (n_e, blob_centers).
    """
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = shape
    Lx, Ly, Lz = extent

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    n_e = np.full(shape, float(n_bg))
    centers = []
    for _ in range(n_blobs):
        c = np.array([rng.uniform(region[0] * Lx, region[1] * Lx),
                      rng.uniform(0.30 * Ly, 0.70 * Ly),
                      rng.uniform(0.30 * Lz, 0.70 * Lz)])
        sig = rng.uniform(*sigma_frac) * min(Lx, Ly, Lz)
        amp = n_blob_peak * rng.uniform(0.6, 1.0)
        r2 = (X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2
        n_e += amp * np.exp(-r2 / (2 * sig**2))
        centers.append((c, sig, amp))
    return n_e, centers


def pml_taper(shape, pml_thickness, ramp=6):
    """Smooth mask that takes the plasma density to zero before the PML.

    A PML tuned for vacuum does not absorb correctly inside a magnetized
    plasma; ramping n_e down over `ramp` cells just inside the PML keeps the
    absorbing layer in vacuum and avoids artificial interface reflections
    being trapped in the PML.
    """
    mask = np.ones(shape)
    edge = pml_thickness + ramp
    for axis, N in enumerate(shape):
        pos = np.arange(N, dtype=np.float64)
        prof = np.ones(N)
        left = pos < edge
        prof[left] = 0.5 * (1 - np.cos(np.pi * np.clip((pos[left] - pml_thickness) / ramp, 0, 1)))
        right = pos > N - 1 - edge
        prof[right] = 0.5 * (1 - np.cos(np.pi * np.clip(((N - 1 - pos[right]) - pml_thickness) / ramp, 0, 1)))
        shape_b = [1, 1, 1]
        shape_b[axis] = N
        mask = mask * prof.reshape(shape_b)
    return mask
