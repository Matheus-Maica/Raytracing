import numpy as np
import matplotlib.pyplot as plt

def ionosphere_electron_density(height, width, size, seed=None):
    n_e = np.zeros((height, width))

    return n_e

    if seed is not None:
        np.random.seed(seed)

    # Physical domain (assume square patch)
    L = np.sqrt(size)
    x = np.linspace(0, L, width)
    y = np.linspace(0, L, height)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # --- 1. Background ionospheric density (F-region-ish) ---
    n0 = 8e15  # typical mid-F region electron density [m^-3]

    # gentle large-scale gradient (solar zenith / geomagnetic effects mimic)
    grad = 1 + 0.15 * (X / L) - 0.10 * (Y / L)

    background = n0 * grad

    # --- 2. Correlated turbulence (smoothed random field) ---
    noise = np.random.randn(width, height)

    kx = np.fft.fftfreq(width)[:, None]
    ky = np.fft.fftfreq(height)[None, :]

    k2 = kx**2 + ky**2

    # correlation length ~ fraction of domain
    corr_length = 0.08 * L
    k0 = 1.0 / corr_length

    filter_kernel = np.exp(-k2 * (k0**2))
    noise_k = np.fft.fft2(noise)
    turbulence = np.fft.ifft2(noise_k * filter_kernel).real

    turbulence = turbulence / np.std(turbulence)

    # amplitude ~ 10–25% fluctuations
    turbulence_field = 0.2 * n0 * turbulence

    # --- 3. Plasma blobs / irregularities ---
    blobs = np.zeros_like(background)
    n_blobs = np.random.randint(3, 8)

    for _ in range(n_blobs):
        x0 = np.random.uniform(0, L)
        y0 = np.random.uniform(0, L)

        amp = np.random.uniform(-0.3, 0.5) * n0
        sigma = np.random.uniform(0.03*L, 0.12*L)

        blobs += amp * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    # --- 4. Combine and enforce positivity ---
    n_e = background + turbulence_field + blobs

    # ensure physical positivity
    n_e = np.clip(n_e, 1e9, None)

    return n_e

if __name__ == "__main__":
    matrix = ionosphere_electron_density(300, 300, size=1e6, seed=42)

    plt.matshow(matrix, cmap='viridis')

    # Add a colorbar legend
    plt.colorbar()

    # Display the plot
    plt.show()