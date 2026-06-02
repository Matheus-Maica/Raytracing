import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class FieldAnimator:
    def __init__(self):
        self.Ex_frames = []
        self.Ey_frames = []

    def add_frame(self, Ex, Ey):
        self.Ex_frames.append(Ex.copy())
        self.Ey_frames.append(Ey.copy())

    def animate(self, interval=30, stride=4, cmap="viridis"):

        Ex0 = self.Ex_frames[0]
        Ey0 = self.Ey_frames[0]

        ny, nx = Ex0.shape

        x = np.arange(nx)
        y = np.arange(ny)

        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots(figsize=(8, 6))

        mag0 = np.sqrt(Ex0**2 + Ey0**2)

        # Avoid divide-by-zero
        mag_safe = np.where(mag0 > 0, mag0, 1.0)

        U0 = Ex0 / mag_safe
        V0 = Ey0 / mag_safe

        q = ax.quiver(
            X[::stride, ::stride],
            Y[::stride, ::stride],
            U0[::stride, ::stride],
            V0[::stride, ::stride],
            mag0[::stride, ::stride],
            cmap=cmap,
            pivot="middle",
            angles="xy",
            scale_units="xy",
            scale=1.5,
        )

        cbar = plt.colorbar(q, ax=ax)
        cbar.set_label("|E|")

        ax.set_aspect("equal")

        def update(frame):

            Ex = self.Ex_frames[frame]
            Ey = self.Ey_frames[frame]

            mag = np.sqrt(Ex**2 + Ey**2)

            mag_safe = np.where(mag > 0, mag, 1.0)

            U = 5 * Ex / mag_safe
            V = 5 * Ey / mag_safe

            q.set_UVC(
                U[::stride, ::stride],
                V[::stride, ::stride],
                mag[::stride, ::stride]
            )

            ax.set_title(f"Timestep {frame}")

            return (q,)

        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.Ex_frames),
            interval=interval,
            blit=False,
        )

        plt.show()

        return anim