import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_ray(positions, amplitudes=None, amplitude_floor=0.05):
    """
    Parameters
    ----------
    positions : (N, 3) ndarray
        positions[:,0] = x
        positions[:,1] = y
        positions[:,2] = altitude (z)

    amplitudes : (N,) ndarray or None
        Amplitude values along the ray. If provided, they are mapped to
        segment opacity. If None, the ray is drawn fully opaque.

    amplitude_floor : float
        Minimum opacity after normalization. Set to 0.0 if you want
        the weakest segments to become fully transparent.
    """

    positions = np.asarray(positions)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N,3)")

    if amplitudes is not None:
        amplitudes = np.asarray(amplitudes)
        if amplitudes.ndim != 1:
            raise ValueError("amplitudes must have shape (N,)")
        if len(amplitudes) != len(positions):
            raise ValueError("amplitudes must have the same length as positions")

    # Remove underground points
    mask = positions[:, 2] >= 0
    positions = positions[mask]
    if amplitudes is not None:
        amplitudes = amplitudes[mask]

    if len(positions) < 2:
        raise ValueError("Not enough points above the ground.")

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Normalize amplitudes to [0, 1]
    if amplitudes is None:
        amp_norm = np.ones(len(positions))
    else:
        amp_min = np.min(amplitudes)
        amp_max = np.max(amplitudes)
        if np.isclose(amp_max, amp_min):
            amp_norm = np.ones(len(amplitudes))
        else:
            amp_norm = (amplitudes - amp_min) / (amp_max - amp_min)
        amp_norm = amplitude_floor + (1.0 - amplitude_floor) * amp_norm
        amp_norm = np.clip(amp_norm, 0.0, 1.0)

    # One opacity value per segment
    seg_alpha = 0.5 * (amp_norm[:-1] + amp_norm[1:])

    # Figure
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Ground plane (z = 0)
    margin = 0.05
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0

    xmin -= margin * dx
    xmax += margin * dx
    ymin -= margin * dy
    ymax += margin * dy

    Xg, Yg = np.meshgrid(
        np.linspace(xmin, xmax, 2),
        np.linspace(ymin, ymax, 2),
    )
    Zg = np.zeros_like(Xg)

    ax.plot_surface(
        Xg,
        Yg,
        Zg,
        alpha=0.35,
        color="forestgreen",
        linewidth=0,
        shade=True,
    )

    # Ray segments
    pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = Line3DCollection(
        segments,
        cmap="plasma",
        linewidth=3,
    )

    # Color according to altitude
    lc.set_array(z[:-1])

    # Apply opacity from amplitude while preserving the colormap coloring
    rgba = lc.cmap(lc.norm(z[:-1]))
    rgba[:, 3] = seg_alpha
    lc.set_color(rgba)

    ax.add_collection3d(lc)

    # Start / end markers
    ax.scatter(
        x[0],
        y[0],
        z[0],
        color="limegreen",
        s=80,
        label="Launch",
        edgecolor="black",
        zorder=10,
    )

    ax.scatter(
        x[-1],
        y[-1],
        z[-1],
        color="crimson",
        s=80,
        label="End",
        edgecolor="black",
        zorder=10,
    )

    # Limits
    zmax = np.max(z)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(0, max(zmax * 1.05, 1.0))

    xr = xmax - xmin
    yr = ymax - ymin
    zr = max(zmax, 1.0)
    ax.set_box_aspect((xr, yr, zr))

    # Labels
    ax.set_xlabel("East-West (m)", labelpad=12)
    ax.set_ylabel("North-South (m)", labelpad=12)
    ax.set_zlabel("Altitude (m)", labelpad=12)
    ax.set_title("Ray Trajectory with Amplitude Opacity")

    # Viewing angle
    ax.view_init(elev=25, azim=-60)

    # Cleaner axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(
        lc,
        ax=ax,
        shrink=0.75,
        pad=0.08,
    )
    cbar.set_label("Altitude (m)")

    ax.legend()
    plt.tight_layout()
    plt.show()
