# Hamilton

HF ray tracing through a 3-D magnetised ionosphere (Haselgrove's equations on a WGS84 Earth), plus a HiCIRF-style
stochastic channel simulator built on it: point targets, sea and land clutter, auroral and meteor echoes, receiver
noise, and the assembled channel impulse response (CIR).

```python
from api import Link                      # the public entry point (see example.py)
link = Link(lat_deg=57.0, lon_deg=10.0, frequency_mhz=8.0, dt=datetime(2024, 6, 21, 14), mode="O")
ray = link.trace(elevation_deg=28.0)      # RayPath; link.sweep_elevation(...) for a fan
print(link.channel(ray))                  # HiCIRF stochastic channel of the leg
link.plot(ray).savefig("ray.png")         # PHaRLAP-style figure with the HiCIRF channel
```

The whole public API is `api.Link` and the results it returns; see the docstring of `api.py` and
`docs/algorithm.md` §5.2.

## Layout

| Path | What is there |
|---|---|
| `api.py` | **Public interface**: `Link` (trace, sweep, budget, HiCIRF channel, figures) and its results `RayPath`, `LinkBudget`, `ChannelParams`. Wires everything below together. |
| `example.py` | A small example of `api.Link`: one ray, its budget, its HiCIRF channel, and a PHaRLAP-style figure (`example_ray.png`). |
| [`core/`](core/README.md) | Constants, WGS84 geometry, the records passed between stages, user inputs. |
| [`geophysics/`](geophysics/README.md) | The medium: E-CHAIM/IRI ionosphere, IGRF field, irregularity climatology (WBMOD stand-in), plasma drift, CCMP sea wind; model interfaces. Vendored `ECHAIM/`. |
| [`raytracing/`](raytracing/README.md) | The engine: `Ray` (Appleton-Hartree physics), scalar `RayTracer`, vectorised `BatchTracer`, absorption, ground bounces, Monte Carlo scattering. |
| [`propagation/`](propagation/README.md) | Deterministic legs: launch fans and landing maps, Newton homing, flux-tube focusing, the one-way power budget. |
| [`stochastic/`](stochastic/README.md) | HiCIRF stochastic channel: phase screens, channel parameters, the Dana-Wittwer GPSD and realisations, two-way cascade. |
| [`echoes/`](echoes/README.md) | What comes back: point targets, sea/land clutter (first and second order), auroral backscatter, meteors. |
| [`receiver/`](receiver/README.md) | The receiving end: array element offsets, ITU-R P.372 noise, CIR assembly, output files. Vendored `P372/data/`. |
| [`plotting/`](plotting/README.md) | Figures in PHaRLAP's MATLAB look: the range-height ionospheric slice (port of `plot_ray_iono_slice.m`), the 3-D ray plot, and the HiCIRF channel (phase screens, channel parameters, scattering function). The old Plotly scene. |
| [`legacy/`](legacy/README.md) | Unused code kept for reference. |
| [`tests/`](tests/README.md) | The test suite (`python tests/runner.py`). |
| [`validation/`](validation/README.md) | On-demand reference computations (not run by the suite). |
| [`docs/`](docs/README.md) | Algorithm description, implementation plan and status, spec deviations, papers. |

## How the folders depend on each other

```
core  <-  geophysics  <-  raytracing  <-  propagation  <-  echoes  <-  receiver
  ^                                                          |
  +------------------------  stochastic  <-------------------+
api.py uses core, geophysics, raytracing, propagation, stochastic, plotting
plotting uses core and stochastic
```

Arrows point from user to dependency; nothing points backwards (a few imports are deferred into functions, noted in
each README). Every folder is a Python package; imports are absolute from the project root, e.g.
`from echoes import clutter`, `from raytracing.main import Ray`.

## Running things

All commands from the project root, with the project's virtual environment:

```bash
venv/bin/python tests/runner.py                 # whole suite; `tests/runner.py clutter` for a subset
venv/bin/python example.py                      # example, writes example_ray.png
venv/bin/python -m raytracing.main              # the legacy demo in raytracing/main.py
venv/bin/python -m geophysics.ionosphere        # the ionosphere smoke test
venv/bin/python validation/mps_1d.py            # multiple-phase-screen reference
```

A module inside a folder must be run with `-m package.module` (not `python package/module.py`), so that the project
root is on the import path.

The detailed physics is in [docs/algorithm.md](docs/algorithm.md); what still needs data from the user is in
[docs/needed_from_user.md](docs/needed_from_user.md).
