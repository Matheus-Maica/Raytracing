"""3-D ADE-FDTD simulation of EM waves in a magnetized, collisional plasma.

Run:  python main.py [--preset demo_whistler] [--source rotating_charge]
                     [--steps N] [--mode linear|nonlinear] [--gif out.gif]
                     [--no-plot]

The run goes to completion while printing progress reports; a slice of the
volume is captured every `plot_every` steps and compiled into an animated
GIF at the end (--no-plot skips the GIF). Presets bundle a grid size, cell
size, density scale and source frequency that are physically consistent
with each other; individual pieces can be overridden on the command line.
Sources: rotating_charge, dipole, loop, sheet (see sources.py).
"""
import argparse
import numpy as np

from grid3d import Grid3D
from plasma_density import ionosphere_electron_density_3d, blob_structures_3d
from sources import make_source, SOURCES
from diagnostics import (GifRecorder, ProgressReporter, ScatteringReport,
                         print_report)

PRESETS = {
    # Whistler-regime demo sized to run interactively: density scaled down so
    # omega_p*dt stays resolved at a dt that still crosses the box quickly.
    # omega_ce (8.8e6 rad/s) > omega_pe (4e6) > omega_src (1.3e6): the source
    # sits below both, i.e. in the whistler band.
    "demo_whistler": dict(
        N=128, dx=50.0, n0=5e9, f_src=2.0e5, B0=(0.0, 0.0, 5e-5),
        mode="linear", source="rotating_charge", steps=3000,
        plot_every=10,
    ),
    # Physically-scaled mid-latitude F region. omega_p ~ 1e8 rad/s forces
    # dt ~ 4e-9 s, so real propagation distances need many steps -- this is
    # the intrinsic stiffness of ionospheric FDTD, not a code limit.
    "ionosphere": dict(
        N=128, dx=400.0, n0=8e11, f_src=1.0e4, B0=(0.0, 0.0, 5e-5),
        mode="linear", source="rotating_charge", steps=20000,
        plot_every=50,
    ),
    # Nonlinear fluid response: strong dipole drive in a lighter plasma so
    # quiver velocities become significant and density perturbations form.
    "nonlinear_demo": dict(
        N=96, dx=50.0, n0=5e9, f_src=2.0e5, B0=(0.0, 0.0, 5e-5),
        mode="nonlinear", source="dipole", steps=2000,
        plot_every=10, source_kwargs=dict(amplitude=5e-4),
    ),
    # Plane wave vs nonlinear structures: a sheet launches a z-polarized
    # (O-mode, E || B0) quasi-plane wave along x through a transparent
    # background (omega > omega_p_bg) into overdense blobs (cores above the
    # critical density n_crit ~ 1.1e10 m^-3) -> reflection, diffraction and
    # shadowing at each critical layer, with the fluid responding
    # self-consistently. Sheet amplitude gives E ~ 2 V/m, i.e. ponderomotive
    # potential ~ several % of k_B T_e: visible density imprint + harmonics.
    # Ends with a ScatteringReport figure in addition to the GIF.
    "scattering": dict(
        N=128, dx=13.0, n0=2e9, f_src=9.55e5, B0=(0.0, 0.0, 5e-5),
        mode="nonlinear", source="sheet", steps=3000, plot_every=15,
        density="blobs", n_blob=4e10, report=True,
        source_kwargs=dict(amplitude=1.6e-4, polarization=(0, 0, 1)),
    ),
}


def build(cfg):
    N = cfg["N"]
    shape = (N, N, N)
    extent = tuple(N * cfg["dx"] for _ in range(3))

    if cfg.get("density") == "blobs":
        n_e, _ = blob_structures_3d(shape, extent, n_bg=cfg["n0"],
                                    n_blob_peak=cfg["n_blob"], seed=42)
    else:
        n_e = ionosphere_electron_density_3d(shape, extent, n0=cfg["n0"],
                                             seed=42)

    grid = Grid3D(shape, cfg["dx"], n_e, B0_vec=cfg["B0"], mode=cfg["mode"],
                  mobile_ions=cfg.get("mobile_ions", False))

    omega = 2 * np.pi * cfg["f_src"]
    center = tuple(0.5 * L for L in extent)
    kwargs = dict(cfg.get("source_kwargs", {}))
    kwargs.setdefault("omega", omega)
    if cfg["source"] == "sheet":
        center = (0.25 * extent[0], 0.5 * extent[1], 0.5 * extent[2])
    source = make_source(cfg["source"], shape, grid.spacing, center, **kwargs)
    return grid, source, omega


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="demo_whistler", choices=sorted(PRESETS))
    ap.add_argument("--source", default=None, choices=sorted(SOURCES))
    ap.add_argument("--mode", default=None, choices=["linear", "nonlinear"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--no-plot", action="store_true",
                    help="skip frame capture and the GIF")
    ap.add_argument("--gif", default=None, metavar="PATH",
                    help="output GIF path (default <preset>_<source>.gif)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--ions", action="store_true",
                    help="mobile-ion fluid too (nonlinear mode only)")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    for key in ("source", "mode", "steps", "N"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.ions:
        cfg["mobile_ions"] = True

    grid, source, omega = build(cfg)
    print_report(grid, source, source_omega=omega)

    recorder = None if args.no_plot else GifRecorder(grid, slice_axis=2)
    progress = ProgressReporter(grid, cfg["steps"])
    report = (ScatteringReport(grid, omega, cfg["steps"])
              if cfg.get("report") else None)

    for i in range(cfg["steps"]):
        grid.step(source)
        if recorder is not None and i % cfg["plot_every"] == 0:
            recorder.capture(i)
        if report is not None:
            report.record(i)
        progress.report(i)

    if recorder is not None:
        path = args.gif or f"{args.preset}_{cfg['source']}.gif"
        recorder.save(path, fps=args.fps)
    if report is not None:
        report.save(f"{args.preset}_report.png")


if __name__ == "__main__":
    main()
