# tests — the test suite

Plain functions named `test_*` in modules `test_*.py`; a test passes if it returns without raising. No pytest
needed (the files also run under pytest if it is installed).

```bash
venv/bin/python tests/runner.py                 # everything (about 4 minutes)
venv/bin/python tests/runner.py clutter         # modules or tests whose name contains "clutter"
venv/bin/python tests/runner.py -k xmode        # tests whose function name contains "xmode"
```

The runner puts the project root and this folder on `sys.path`, so tests import the packages directly
(`from echoes import clutter`). The files stay flat here (the runner does not recurse); the table below maps them
to the folders they cover.

## Infrastructure

| File | What it is |
|---|---|
| `runner.py` | The runner above. |
| `support.py` | Shared helpers: uniform-medium `Ray`s, X factors, reference constants for the Phase 0 checks. |
| `tv_cases.py` | The spec's reference test vectors TV-1 .. TV-4 (Appendix B) and the values of its Tables 11 and 12. |
| `golden_make.py` | Regenerates `data/golden_phase0.npz` from the scalar tracer. Run by hand only when the reference must change. |
| `data/golden_phase0.npz` | Golden outputs the batch tracer must reproduce (`test_vectorised.py`). |

## Tests by folder

| Folder | Test modules |
|---|---|
| `raytracing/` | `test_phase0.py` (X-mode, absorption factor, group index, phase sensitivity), `test_vectorised.py` (batch vs. golden scalar), `test_absorption.py`, `test_records_ground.py` (records, transported frame, Fresnel), `test_deprecation_and_valley.py` |
| `geophysics/` | `test_ionosphere.py` (patch, D region, epoch pair), `test_wbmod.py`, `test_plasma_drift.py`, `test_ccmp_wind.py` (includes a check against a real cached CCMP file when present) |
| `propagation/` | `test_legs.py`, `test_fans.py`, `test_homing.py`, `test_doppler.py` (Doppler Method B vs. Method A) |
| `stochastic/` | `test_stochastic_units.py`, `test_tv_gate.py` (spec test vectors), `test_gpsd.py`, `test_cascade.py`, `test_leg_screens.py` |
| `echoes/` | `test_twoway.py`, `test_targets.py`, `test_clutter.py`, `test_clutter_scene.py`, `test_second_order_bragg.py` (against Barrick 1972), `test_auroral.py`, `test_auroral_scene.py`, `test_meteor.py`, `test_meteor_raytraced.py` |
| `receiver/` | `test_noise.py`, `test_itu_p372.py` (against the ITU C reference), `test_cir.py` |
| `plotting/` | `test_plotting.py` (the `plot_ray_iono_slice` port, MuellerMap, slice grid, 3-D clipping, the `Link` figures) |
| `api.py` | `test_api_batch.py` |

Tests that trace rays build real E-CHAIM/IRI patches or uniform synthetic media; the slow ones (fans, homing,
targets, meteors) take tens of seconds each.
