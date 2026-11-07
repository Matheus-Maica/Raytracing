# stochastic — the HiCIRF stochastic channel

Small-scale irregularities along a leg turn a deterministic ray into a fading, spread channel. This folder goes from
the irregularity description along a traced leg to the statistics of the channel at the array end, and from those
to random channel realisations (Nickisch et al. 2012; spec §8-12). Depends only on `core/`: it works on
`core.records.RaySamples`, whatever produced them.

| File | What it does |
|---|---|
| `screens.py` | Irregularity spectrum and phase screens along an unfolded leg; structure functions. |
| `leg_screens.py` | Bridge from a traced leg (`RaySamples`) to its `ScreenSet`, querying the irregularity/drift models. |
| `channel.py` | Channel parameters of a leg (Algorithms 4-5): coherence lengths, tau_o, delay spread, S4, Rician split. |
| `gpsd.py` | Dana-Wittwer generalised power spectral density, bin-integrated weights, realisations. |
| `cascade.py` | Two-way realisation: outbound and return legs convolved in delay, multiplied in slow time. |

Flow: `leg_screens.screens_for_leg(samples, ...)` -> `ScreenSet` -> `channel.channel_parameters(screens, f0)` ->
`core.records.ChannelParams` -> `gpsd.element_responses` / `realise_*` -> `cascade.two_way_realisation`.
`api.Link.channel(ray)` runs the first three steps for a traced ray (`leg_channel(elevation, ...)` traces it first).

## screens.py

Spec §8.2 (anisotropic power-law spectrum with outer scale L0, index p, axial ratios a along B and b along the sheet
axis) and §9 (screens). Vectorised over screens.

- `spectrum_norm(p)`, `F_of_p(p)`, `psi_p(u, p)`: normalisation (eq. 8.1), eq. 9.3, and the normalised phase
  structure function (eq. 9.4, series 9.5).
- `screen_matrices(...)`: the geometry matrix W_i of each screen (eq. 9.1).
- `build_screens(...)` -> `ScreenSet`: every per-screen quantity of spec §9.3 (position z along the unfolded leg,
  thickness, phase sensitivity g, sigma_N, L0, p, axial ratios, frames, drift v_d and its spread, W_i, sigma^2...);
  `ScreenSet.subset(idx)` selects screens.
- `struct_fn(s, ...)`: D_i(s) for one displacement per screen.

## leg_screens.py

`screens_for_leg(samples, ...)`: one screen per ionospheric sample (vacuum excluded), at the group-path position.
Irregularity and drift come either from scalar placeholders (`core.inputs.IrregularityInputs`) or, when given, from
`irregularity_model` / `drift_model` objects (`geophysics.environment` Protocols) queried per sample (ECEF drift
vectors). The samples must carry the transported frame (`samples_from_record(..., transport=True)`).
`coarsen_elevation_deg=5.0` places screens only every 5 deg of ray direction change (HiCIRF paragraph 17).

## channel.py

`channel_parameters(screens, f0, options)` (Algorithm 5) -> `ChannelParams`:

- coherence lengths l_x, l_y and principal axes from an ellipse fit of D_A(rho) = D* (`fit_ellipse`);
- decorrelation time tau_o (`solve_tau_o`, Gauss-Hermite turbulence average), apparent velocity and space-time
  coupling C_xt, C_yt (eqs. 10.4-10.6);
- mean excess delay, coherence bandwidth and wander (eqs. 10.7-10.9);
- S4: the weak-scatter baseline (eq. 10.10) or the refined form (eq. 10.11, `gp` tables);
- `rician`: A_c, A_d so that A_c^2 + 2 A_d^2 = 1 and S4 is reproduced (eq. 10.13).

`ChannelOptions` holds the choices the spec leaves open.

## gpsd.py

- `gpsd(...)`: the GPSD of eq. 11.1. Conventions: h = sum H exp(-i K.x + 2 pi i f_D t).
- `DiffuseGrid` / `default_grid`: transverse, Doppler and delay sampling (§11.2).
- `bin_weights`, `single_element_weights`: expected power per (K, f, tau) bin, integrated over the bin in closed form
  (Gaussian CDFs) rather than point-sampled; `captured_fraction` must exceed 0.99.
- `realise_single_element`, `realise_array`, `element_responses`: random realisations (eq. 11.4), Rician assembly
  for the elements of one leg. `scattering_function` is a diagnostic.

## cascade.py

`two_way_realisation(...)` (eq. 12.1): convolves outbound and return realisations along delay
(`convolve_delay`), applies the carrier phase -k0 (P_out + P_ret) and the summed Doppler, and places the result on
the absolute delay grid with a band-limited fractional shift (`place_on_grid`).
