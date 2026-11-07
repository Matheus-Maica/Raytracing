# raytracing — the ray-tracing engine

Haselgrove's ray equations in a cold magnetoplasma (Appleton-Hartree), integrated in geocentric Cartesian metres over
a WGS84 Earth, with multi-hop ground reflection, path integrals (phase path, group path, absorption, phase
variance) and an optional tangent map for focusing. Depends on `core/` and on `geophysics/` (density and field);
the demo in `main.py` also uses `plotting/`.
The physics is written up in `docs/algorithm.md` §2-4.

| File | What it does |
|---|---|
| `main.py` | `Ray`: the physical model (dispersion relation and the right-hand side of the ray equations). |
| `raytracer.py` | `RayTracer`: the scalar engine, one ray at a time; `TraceResult`. |
| `batch.py` | `BatchTracer`: the same algorithm vectorised over many rays; records, Doppler, `RaySamples`. |
| `absorption.py` | Collision-frequency profiles and the absorption rate along a ray. |
| `ground.py` | What a ground reflection costs: Fresnel coefficients, bounce loss, roughness. |
| `mc_scattering.py` | Monte Carlo scattering by small-scale irregularities (`MCScatteringTracer`). |

How they fit: a `Ray` holds the medium (density callable, field callable, frequency, O/X mode) and evaluates the
physics. `RayTracer` and `BatchTracer` integrate it; `absorption.py` supplies an integrand, `ground.py` the cost of
each bounce. `api.Link` builds the `Ray` and picks the tracer.

## main.py

The file name is historical; it holds the physics core.

- `Ray(omega, MODE, electron_density, B_field=None)`: `MODE` +1/-1 picks O/X. `B_field` is a callable
  (e.g. `geophysics.igrf.IgrfField.B_vec_xyz_m`); `None` uses an analytic tilted dipole (fast, offline, not the
  real field). Main methods: `Y_vec`, `grad_Ne` (500 m central differences), `_appleton_hartree_p` (Haselgrove's
  homogeneous quadratic in n^2) and its numerically stable companion `_dispersion_eps`, `n_angle` (angle form with
  collisions), `_haselgrove_physical_rhs` (the 6-D RHS in (r, u)), `group_index`, `phase_sensitivity`.
  Below `X_VACUUM_EPS` the medium is treated as exact vacuum.
- `build_ionosphere_for_ray(...)`: sizes and builds the `Ionosphere` patch a trace needs.
- `python -m raytracing.main` runs the old demo (deterministic + Monte Carlo rays, writes `ray_scene_3d.html` with
  `plotting/plotly_3d.py`).

## raytracer.py

`RayTracer(ray, ...)` integrates an extended state: r, u (6), arc length, phase path, group path, absorption
[Np], phase variance, and optionally the 6x6 tangent map (36 more). One RHS (`_extended_rhs`) is shared by two
segment drivers: fixed-step RK4 with step rejection, and adaptive `solve_ivp`. `trace()` runs the hop loop: integrate
to the ground, locate the crossing by bisection, reflect specularly about the WGS84 normal, repeat up to `nhops`;
stop on escape or budget. `_propagate_tangent_map` gives d(final)/d(initial) for ray-tube focusing.
`TraceResult` is the per-ray output bundle. `compute_doppler` is a stub (Doppler lives in `batch.py`).

## batch.py

`BatchTracer` applies the fixed-step algorithm of `RayTracer` to N rays at once, with per-ray masks for step
rejection, ground crossing, ceiling and budget. Equivalence with the scalar path is pinned by golden outputs
(`tests/test_vectorised.py`). Extras the scalar path does not have:

- `record_states=True`: every accepted step is kept; `samples_from_record()` turns one ray's record into a
  `core.records.RaySamples`.
- `DopplerRates`: the Doppler accumulator (spec eq. 4.8) from dNe/dt of a `geophysics.ionosphere.EpochPair`.
- `landing_alt_m`: stop at a given altitude instead of the ground (targets at altitude, meteor trails).
- `transport_frame()`: Algorithm 1, the transverse frame carried along a leg.

`BatchResult` holds the per-ray arrays (landing point, direction, paths, hops, bounces, apex heights...).

## absorption.py

Absorption is a post-hoc integral along the loss-free ray. Two collision conventions: nu_eff (complex
Appleton-Hartree, `absorption_rate`) and nu_m (Sen-Wyller, `kappa_sen_wyller` with the tabulated `sw_F`), the latter
below `SW_ALT_KM`. `ProfileModel` implements `geophysics.environment.AbsorptionModel` from a collision-frequency
profile (`collision_frequency_profile`; the default profile is a placeholder). `merge_density` and
`riometer_density` build the density used for absorption, optionally from a riometer reading (spec eq. 7.3).

## ground.py

`grazing_angle`, `fresnel` (eq. 4.3, both polarisations), `bounce_loss_dB` (eq. 4.4, equal mix),
`roughness_power_factor`, and `make_bounce` (a `core.records.Bounce`). The reflection itself happens in the tracers.

## mc_scattering.py

`IrregularitySpectrum` (anisotropic power-law spectrum, tabulated transverse integrals vs. angle to B) gives the
variance of random angular kicks; `MCScatteringTracer` is a thin subclass of `RayTracer` that applies them
(a Langevin version of the ray equations). `.ensemble()` traces many realisations; `.summarize()` gives landing
statistics. Used by `api.Link.scatter`.

Plots of rays are in `plotting/` (PHaRLAP-style figures; the former `visualization.py` is `plotting/plotly_3d.py`).
