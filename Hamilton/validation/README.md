# validation — on-demand reference computations

Independent calculations used to check or decide something in the main code. They are run by hand, not by the
test suite.

| File | What it does |
|---|---|
| `mps_1d.py` | 1-D multiple-phase-screen reference (spec §15.3, plan Phase 3e): split-step Fourier propagation of a plane wave through Gaussian random phase screens with the structure function of `stochastic.screens.psi_p`. Checks the mutual coherence after one screen against exp(-D/2), that mean intensity is conserved, and that S4 grows with distance. It was the basis for the group-path vs. arc-length screen placement decision (D5). |

Run from the project root: `venv/bin/python validation/mps_1d.py` (the script adds the root to `sys.path`).
