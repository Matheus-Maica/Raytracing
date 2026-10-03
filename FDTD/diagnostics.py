import time
import numpy as np
import matplotlib.pyplot as plt


def take_slice(A, axis, index):
    sl = [slice(None)] * 3
    sl[axis] = index
    return A[tuple(sl)].T  # transpose so the first remaining axis is horizontal


class SliceViewer:
    """Live view of a 2-D slice through the 3-D volume: plasma density as
    the background (viridis) with |E| overlaid (RdBu, alpha-blended) --
    the same presentation the legacy 2-D code used."""

    def __init__(self, grid, slice_axis=2, slice_index=None, vmax=None,
                 pause=0.001):
        self.grid = grid
        self.axis = slice_axis
        self.index = (grid.shape[slice_axis] // 2
                      if slice_index is None else slice_index)
        self.vmax = vmax
        self.pause = pause
        self._running_vmax = 0.0
        plt.ion()
        self.fig = plt.figure(figsize=(7, 6))

    def show(self, step):
        n_slice = take_slice(self.grid.n_e, self.axis, self.index)
        E_slice = take_slice(self.grid.E_magnitude(), self.axis, self.index)

        if self.vmax is None:
            peak = np.percentile(E_slice, 99.5)
            self._running_vmax = max(0.95 * self._running_vmax, peak, 1e-30)
            vmax = self._running_vmax
        else:
            vmax = self.vmax

        plt.figure(self.fig.number)
        plt.clf()
        plt.imshow(n_slice, origin="lower", cmap="viridis")
        plt.imshow(E_slice, origin="lower", cmap="RdBu", alpha=0.6,
                   vmin=-vmax, vmax=vmax)
        plt.colorbar(label="|E| [V/m]")
        axes = "xyz".replace("xyz"[self.axis], "")
        plt.xlabel(axes[0]); plt.ylabel(axes[1])
        plt.title(f"step {step}  (slice {'xyz'[self.axis]}="
                  f"{self.index})  t = {step * self.grid.dt:.3e} s")
        plt.pause(self.pause)


class GifRecorder:
    """Collects density/|E| slices during the run and compiles them into an
    animated GIF afterwards. Deferring the rendering lets every frame share
    one global color scale (no flicker) and keeps the run headless.

    Same presentation as SliceViewer: n_e background, |E| overlay.
    """

    def __init__(self, grid, slice_axis=2, slice_index=None, vmax=None):
        self.grid = grid
        self.axis = slice_axis
        self.index = (grid.shape[slice_axis] // 2
                      if slice_index is None else slice_index)
        self.vmax = vmax
        self.frames = []          # (step, n_slice, E_slice)

    def capture(self, step):
        self.frames.append((
            step,
            take_slice(self.grid.n_e, self.axis, self.index).copy(),
            take_slice(self.grid.E_magnitude(), self.axis, self.index).copy(),
        ))

    def save(self, path, fps=12):
        if not self.frames:
            print("GifRecorder: no frames captured, nothing to save")
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PIL import Image

        vmax = self.vmax
        if vmax is None:
            vmax = max(np.percentile(np.abs(E), 99.5)
                       for _, _, E in self.frames)
            vmax = max(vmax, 1e-30)

        n_min = min(n.min() for _, n, _ in self.frames)
        n_max = max(n.max() for _, n, _ in self.frames)

        fig = Figure(figsize=(7, 6), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        _, n0, E0 = self.frames[0]
        im_n = ax.imshow(n0, origin="lower", cmap="viridis",
                         vmin=n_min, vmax=n_max)
        im_E = ax.imshow(E0, origin="lower", cmap="RdBu", alpha=0.6,
                         vmin=-vmax, vmax=vmax)
        fig.colorbar(im_E, ax=ax, label="|E| [V/m]")
        axes = "xyz".replace("xyz"[self.axis], "")
        ax.set_xlabel(axes[0]); ax.set_ylabel(axes[1])

        images = []
        t0 = time.time()
        for k, (step, n_slice, E_slice) in enumerate(self.frames):
            im_n.set_data(n_slice)
            im_E.set_data(E_slice)
            ax.set_title(f"step {step}  (slice {'xyz'[self.axis]}="
                         f"{self.index})  t = {step * self.grid.dt:.3e} s")
            canvas.draw()
            rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()
            images.append(Image.fromarray(rgb))
            if (k + 1) % 50 == 0 or k + 1 == len(self.frames):
                print(f"  rendering frames {k + 1}/{len(self.frames)} "
                      f"({time.time() - t0:.1f}s)", flush=True)

        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=int(round(1000 / fps)), loop=0)
        print(f"wrote {path} ({len(images)} frames, "
              f"{images[0].size[0]}x{images[0].size[1]}, {fps} fps)")


class ProgressReporter:
    """Console progress: percent done, sim time, field energy, wall-clock
    elapsed and ETA."""

    def __init__(self, grid, total_steps, every=None):
        self.grid = grid
        self.total = total_steps
        self.every = every if every is not None else max(1, total_steps // 20)
        self.t0 = time.time()

    def report(self, i):
        done = i + 1
        if done % self.every and done != self.total:
            return
        elapsed = time.time() - self.t0
        rate = done / max(elapsed, 1e-9)
        eta = (self.total - done) / max(rate, 1e-9)
        print(f"step {done:>6d}/{self.total} ({100 * done / self.total:3.0f}%)  "
              f"t = {done * self.grid.dt:.3e} s  "
              f"W = {self.grid.field_energy():.3e} J  "
              f"[{elapsed:6.1f}s elapsed, ETA {eta:6.1f}s]", flush=True)


class ScatteringReport:
    """End-of-run analysis of a plane wave hitting plasma structures.

    Accumulates cheap statistics while the simulation runs (probe time
    series every step; time-averaged E^2 over the steady-state tail) and
    renders a six-panel summary figure at the end:

      1. snapshot of Ez on the mid-slice, with the critical-density contour
         of the structures overlaid (diffraction / shadowing pattern),
      2. time-averaged intensity <Ez^2> in dB (standing waves upstream,
         shadows and focusing behind the blobs),
      3. the nonlinear density imprint (n_final - n0)/n0,
      4. field spectra at the probes, normalized to the drive frequency
         (harmonics at 2f0, 3f0 are the fingerprint of the nonlinearity),
      5. axial profile of <Ez^2> vs x against the structure density profile,
      6. run metrics (transmitted fraction, peak density perturbation,
         harmonic ratio, ...).
    """

    def __init__(self, grid, omega, total_steps, slice_axis=2,
                 tail_frac=0.30, probe_x_frac=(0.33, 0.53, 0.78)):
        from constants import epsilon_0, m_e, e_charge
        self.grid = grid
        self.omega = omega
        self.total = total_steps
        self.axis = slice_axis
        self.index = grid.shape[slice_axis] // 2
        self.tail_start = int((1 - tail_frac) * total_steps)
        self.n_crit = epsilon_0 * m_e * omega**2 / e_charge**2

        self.E2_sum = np.zeros(grid.shape)
        self.E2_count = 0

        Nx, Ny, Nz = grid.shape
        names = ("incident side", "structure region", "transmitted side")
        self.probes = {name: (int(f * Nx), Ny // 2, Nz // 2)
                       for name, f in zip(names, probe_x_frac)}
        self.series = {name: [] for name in self.probes}

        self.n0 = (grid.species[0].n0.copy() if grid.species
                   else grid.n_e.copy())

    def record(self, i):
        Ez = self.grid.Ez
        for name, idx in self.probes.items():
            self.series[name].append(Ez[idx])
        if i >= self.tail_start:
            self.E2_sum += Ez**2
            self.E2_count += 1

    # -------------------------------------------------------------- figure
    def save(self, path):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        grid = self.grid
        E2avg = self.E2_sum / max(self.E2_count, 1)
        n_now = grid.species[0].n if grid.species else grid.n_e

        Ez_sl = take_slice(grid.Ez, self.axis, self.index)
        E2_sl = take_slice(E2avg, self.axis, self.index)
        n0_sl = take_slice(self.n0, self.axis, self.index)
        # floor the denominator at a few % of the peak so the near-vacuum
        # taper region does not blow up the relative imprint
        dn_ref = np.maximum(self.n0, 0.02 * self.n0.max())
        dn_sl = take_slice((n_now - self.n0) / dn_ref, self.axis, self.index)

        fig = Figure(figsize=(16, 9), dpi=130)
        FigureCanvasAgg(fig)
        axs = fig.subplots(2, 3)
        fig.suptitle(
            f"plane-wave scattering off plasma structures -- "
            f"step {self.total}, f0 = {self.omega/2/np.pi:.3g} Hz, "
            f"n_crit = {self.n_crit:.3g} m$^{{-3}}$", fontsize=12)

        def add_crit_contour(ax):
            if n0_sl.max() > self.n_crit:
                ax.contour(n0_sl, levels=[self.n_crit], colors="k",
                           linewidths=1.2)
            ax.contour(n0_sl, levels=[0.5 * self.n_crit], colors="k",
                       linewidths=0.6, linestyles="dashed")

        # 1: field snapshot
        ax = axs[0, 0]
        v = max(np.percentile(np.abs(Ez_sl), 99.5), 1e-30)
        im = ax.imshow(Ez_sl, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
        add_crit_contour(ax)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title("Ez snapshot [V/m]  (solid: $n=n_{crit}$)")

        # 2: time-averaged intensity, dB
        ax = axs[0, 1]
        ref = max(E2_sl.max(), 1e-300)
        dB = 10 * np.log10(np.maximum(E2_sl / ref, 1e-8))
        im = ax.imshow(dB, origin="lower", cmap="inferno", vmin=-40, vmax=0)
        add_crit_contour(ax)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(r"$\langle E_z^2\rangle$ over the tail window [dB]")

        # 3: nonlinear density imprint
        ax = axs[0, 2]
        v = max(np.percentile(np.abs(dn_sl), 99.9), 1e-30)
        im = ax.imshow(dn_sl, origin="lower", cmap="PuOr_r", vmin=-v, vmax=v)
        add_crit_contour(ax)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(r"density imprint $(n - n_0)/n_0$")

        # 4: probe spectra
        ax = axs[1, 0]
        dt = grid.dt
        f0 = self.omega / (2 * np.pi)
        harmonics = {}
        for name, s in self.series.items():
            s = np.asarray(s)[self.tail_start:]
            if len(s) < 16:
                continue
            w = np.hanning(len(s))
            A = np.abs(np.fft.rfft(s * w))
            fr = np.fft.rfftfreq(len(s), d=dt) / f0
            A = A / max(A.max(), 1e-300)
            ax.semilogy(fr, A, label=name, lw=1.0)
            harmonics[name] = (fr, A)
        for h in (1, 2, 3):
            ax.axvline(h, color="gray", lw=0.5, ls=":")
        ax.set_xlim(0, 4)
        ax.set_ylim(1e-6, 1.5)
        ax.set_xlabel("f / f0"); ax.set_ylabel("|E(f)| (norm.)")
        ax.legend(fontsize=8)
        ax.set_title("probe spectra (harmonics = nonlinearity)")

        # 5: axial profile vs structure density
        ax = axs[1, 1]
        Ny, Nz = grid.shape[1], grid.shape[2]
        cy = slice(Ny // 3, 2 * Ny // 3)
        cz = slice(Nz // 3, 2 * Nz // 3)
        prof = E2avg[:, cy, cz].mean(axis=(1, 2))
        x_km = np.arange(grid.shape[0]) * grid.dx / 1e3
        ax.plot(x_km, prof / max(prof.max(), 1e-300), "C0",
                label=r"$\langle E_z^2\rangle$ (norm.)")
        ax.set_xlabel("x [km]"); ax.set_ylabel(r"$\langle E_z^2\rangle$")
        ax2 = ax.twinx()
        n_prof = self.n0.max(axis=(1, 2))
        ax2.semilogy(x_km, n_prof, "C1", lw=1.0, label="max n0(x)")
        ax2.axhline(self.n_crit, color="C1", ls="--", lw=0.8)
        ax2.set_ylabel("n0 [m$^{-3}$]  (dashed: n_crit)")
        for p, idx in self.probes.items():
            ax.axvline(idx[0] * grid.dx / 1e3, color="gray", lw=0.5, ls=":")
        ax.set_title("axial intensity vs structures")

        # 6: metrics
        ax = axs[1, 2]
        ax.axis("off")
        inc = np.asarray(self.series["incident side"])[self.tail_start:]
        tra = np.asarray(self.series["transmitted side"])[self.tail_start:]
        E_inc = np.sqrt(2 * np.mean(inc**2)) if len(inc) else 0.0
        T = (np.mean(tra**2) / max(np.mean(inc**2), 1e-300)) if len(inc) else 0.0
        dn_max = np.max(np.abs(n_now - self.n0)) / max(self.n0.max(), 1e-300)
        h_ratio = float("nan")
        if "transmitted side" in harmonics:
            fr, A = harmonics["transmitted side"]
            fund = A[np.argmin(np.abs(fr - 1))]
            sec = A[(fr > 1.6) & (fr < 2.4)].max() if np.any((fr > 1.6) & (fr < 2.4)) else 0.0
            h_ratio = sec / max(fund, 1e-300)
        clamp = grid.species[0].clamp_count if grid.species else 0
        tail_periods = (self.total - self.tail_start) * grid.dt * f0
        caveat = ("" if tail_periods >= 8 else
                  "  (SHORT window - spectra unreliable)")
        text = "\n".join([
            "run metrics (steady-state tail)",
            "-" * 34,
            f"E amplitude, incident side : {E_inc:.3g} V/m",
            f"<E^2> transmitted/incident : {T:.3g}",
            "  (upstream includes the standing wave;",
            "   <1 means shadowing/reflection)",
            f"peak |n-n0|/max(n0)        : {dn_max:.3g}",
            f"2nd/1st harmonic (transm.) : {h_ratio:.3g}",
            f"velocity clamps            : {clamp}",
            f"mode: {grid.mode},  dt = {grid.dt:.3g} s",
            f"tail window: steps {self.tail_start}..{self.total}",
            f"  = {tail_periods:.1f} wave periods{caveat}",
        ])
        ax.text(0.02, 0.95, text, family="monospace", fontsize=10,
                va="top", transform=ax.transAxes)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path)
        print(f"wrote {path}")


def print_report(grid, source, source_omega=None):
    from constants import C
    print(grid.summary())
    if source_omega is not None:
        wp = np.sqrt(np.max(grid.wp_sq))
        print(f"source omega = {source_omega:.4g} rad/s "
              f"(f = {source_omega/2/np.pi:.4g} Hz)")
        print(f"  vs plasma: omega/omega_p = {source_omega/wp:.3f}, "
              f"vs gyro: omega/|Omega_e| = {source_omega/max(grid.Omega_mag,1e-30):.3f}")
        lam0 = 2 * np.pi * C / source_omega
        print(f"  vacuum wavelength = {lam0:.4g} m -> {lam0/grid.dx:.1f} cells")
    est = 0.0
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Jx", "Jy", "Jz"):
        est += getattr(grid, name).nbytes
    for p in grid.psi.values():
        est += p.nbytes
    if grid.mode == "linear":
        est += sum(a.nbytes for rows in grid.XR.values() for a in rows)
        est += sum(a.nbytes for rows in grid.GR.values() for a in rows)
    print(f"resident field memory ~ {est/1e9:.2f} GB")
