import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_ray(positions):
    """
    Parameters
    ----------
    positions : (N,3) ndarray
        positions[:,0] = x
        positions[:,1] = altitude (y)
        positions[:,2] = z
    """

    positions = np.asarray(positions)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N,3)")

    # ------------------------------------------------------------
    # Remove underground points
    # ------------------------------------------------------------

    positions = positions[positions[:, 1] >= 0]

    if len(positions) < 2:
        raise ValueError("Not enough points above the ground.")

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # ------------------------------------------------------------
    # Ground plane
    # ------------------------------------------------------------

    margin = 0.05

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    dx = xmax - xmin
    dz = zmax - zmin

    if dx == 0:
        dx = 1

    if dz == 0:
        dz = 1

    xmin -= margin * dx
    xmax += margin * dx
    zmin -= margin * dz
    zmax += margin * dz

    Xg, Zg = np.meshgrid(
        np.linspace(xmin, xmax, 2),
        np.linspace(zmin, zmax, 2),
    )

    Yg = np.zeros_like(Xg)

    ax.plot_surface(
        Xg,
        Yg,
        Zg,
        alpha=0.35,
        color="forestgreen",
        linewidth=0,
        shade=True,
    )

    # ------------------------------------------------------------
    # Colored ray
    # ------------------------------------------------------------

    pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = Line3DCollection(
        segments,
        cmap="plasma",
        linewidth=3,
    )

    lc.set_array(y[:-1])

    ax.add_collection3d(lc)

    # ------------------------------------------------------------
    # Start / end markers
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------

    ymax = np.max(y)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_zlim(zmin, zmax)

    # Equal aspect ratio

    xr = xmax - xmin
    yr = ymax
    zr = zmax - zmin

    ax.set_box_aspect((xr, yr, zr))

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    ax.set_xlabel("East-West (m)", labelpad=12)
    ax.set_ylabel("Altitude (m)", labelpad=12)
    ax.set_zlabel("North-South (m)", labelpad=12)

    ax.set_title("Ray Trajectory")

    # ------------------------------------------------------------
    # Nice viewing angle
    # ------------------------------------------------------------

    ax.view_init(elev=25, azim=-60)

    # ------------------------------------------------------------
    # Cleaner axes
    # ------------------------------------------------------------

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(alpha=0.3)

    # ------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------

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