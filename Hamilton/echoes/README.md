# echoes — what comes back to the radar

Everything that scatters power from the outbound leg into the return leg: point targets, the sea and land surface,
field-aligned auroral irregularities and meteor trails. Each produces two-way components (power, delay, Doppler,
phase) for the channel assembly in `receiver/cir.py`. Depends on `core/`, `geophysics/`, `raytracing/`,
`propagation/` and `stochastic/`.

| File | What it does |
|---|---|
| `twoway.py` | Shared two-way physics: the radar equation, target Doppler, range migration, reciprocity Gamma_or. |
| `targets.py` | Point targets: home both sites onto the target, pair the legs into components. |
| `clutter.py` | Sea and land clutter per cell: first-order Bragg lines, sea ice, currents, range-Doppler map. |
| `second_order_bragg.py` | Second-order sea echo (Barrick 1972), as a continuum of weak Doppler lines. |
| `clutter_scene.py` | Clutter over a ground grid: rasterises the Tx and Rx landing maps onto cells. |
| `auroral.py` | Direct auroral backscatter per volume element: Bragg vector, aspect angle, power, Doppler. |
| `auroral_scene.py` | Auroral backscatter integrated over a fan traced from the sites' midpoint. |
| `meteor.py` | Sporadic meteor clutter: radiant sources, trail sampling, underdense-trail RCS. |

## twoway.py

`radar_power_dBW` (eq. 6.2, bistatic radar equation with flux-tube distances), `leg_loss_dB` (eq. 6.3),
`target_doppler_hz` (eq. 6.1), `range_migration_rate` (§12.3), `target_component` (one `core.records.Component` from
an outbound and a return `LegMode`), and the reciprocity diagnostic `gamma_or` / `gamma_or_legs` /
`reciprocity_class` (eq. 12.3: independent, partial or reciprocal legs).

## targets.py

`home_site_to_point(...)`: all homed legs from one site to a target for hop counts 1..max (fan -> Algorithm 2 ->
Algorithm 3 -> `LegMode`), returned as `HomedLeg`s; `target_alt_m` > 0 homes to a point at altitude (the tracer must be built with the same
`landing_alt_m`).
`target_components(...)` pairs every outbound with every return leg; `component_track` gives the delay and Doppler
drift over a coherent integration time.

## clutter.py

Per-cell physics, spec §12.4. Sea: two first-order Bragg lines at +/- f_B = sqrt(g/(pi lambda0)), each with
sigma0 = 2^6 pi k0^4 S(2 k0) = 4 pi B_s G(phi). The wave spectrum is S = B_s K^-4 G(phi) with cardioid spreading
`spreading_G` (integral 1) — Barrick's normalisation (user decision 2026-11-07, `docs/spec_deviations.md` item 46).
Lines are scaled by the open-water fraction (1 - ice concentration) and shifted by the radial surface current.
`bragg_lines(..., second_order=True)` appends the second-order continuum. Land and ice sheet: a zero-Doppler line
with a user sigma0. `range_doppler_map` deposits components on a (delay, Doppler) grid (power conserved);
`clutter_field` draws one Gaussian realisation of such a map. Inputs (wind, B_s, s, ice) are placeholders from
`core.inputs.SurfaceInputs`.

## second_order_bragg.py

Barrick (1972) eqs. 12:31-12:34 (`docs/1972Barrick_RST.pdf`): a 2-D integral over wave pairs with the
electromagnetic and hydrodynamic coupling coefficients, binned in Doppler. `second_order(f0, S, ...)` returns the
total second-order sigma0 and its Doppler spectrum; `second_order_lines(...)` caches it as discrete lines for
`clutter.bragg_lines`. `cardioid_phillips` is the same spectrum as the first order (it calls
`clutter.wave_saturation_S`); `barrick_phillips_semi_isotropic` is Barrick's own, used for validation against his
figures (`tests/test_second_order_bragg.py`). `sea_impedance` is in Barrick's e^{-i omega t} convention.

## clutter_scene.py

`scene_components(fan_tx, fan_rx, ...)`: both sites' landing maps are rasterised (`rasterise_fan`, bilinear
inversion per quadrilateral) onto a common ground grid about the midpoint; each sea or land cell pairs the outbound
and return legs that land in it into `ClutterComponent`s with `clutter.bragg_lines` or the land line. Optional
`wind_model` (e.g. `geophysics.ccmp_wind.CCMPWind`) gives the wind per cell; its compass direction is converted to
the look-relative angle with `approach_azimuth_deg`. `element_phases` gives the array phases of each component;
`default_cell_m` / `range_resolution_m` size the cells.

## auroral.py and auroral_scene.py

`auroral.py`: per volume element, the Bragg vector q_B = -2 n k0 u, the aspect angle to B, the aspect-sensitivity
law, dV, the received power (§12.5) and the Doppler (phase velocity plus twice the ionospheric Doppler).
`auroral_scene.py`: `trace_midpoint_fan` traces a recorded fan from the sites' midpoint (quasi-monostatic baseline);
`auroral_components` forms volume elements from the ray-tube cross-section and keeps those above a reflectivity
threshold. Beyond HiCIRF (2012): opt-in, nothing calls it automatically. Bragg-scale inputs are placeholders
(`core.inputs.BraggInputs`).

## meteor.py

HiCIRF paragraph 34 (Meteor Input Function). Radiant sources from Fentzke, Janches & Sparks (2009) Table 1, source
weights from Fentzke & Janches (2008) Table 3; radiant visibility via ecliptic -> equatorial -> topocentric
conversion. `sample_meteors` draws `MeteorRealization`s; trail echoes use the underdense RCS
(`underdense_trail_rcs`) with diffusion decay (`diffusion_decay_factor`, D_a(h) = 0.025 exp(0.154 h/km) cm^2/s).
Two assemblies: `meteor_components` (straight-line geometry, fast) and `meteor_components_raytraced` (homes real
legs to each trail, grouped in 5 km altitude bins). Line density and initial radius are placeholders by design.
