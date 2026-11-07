# receiver — the receiving end and the output

The last stage: where the components meet the antenna array, the external noise, and the assembled channel
impulse response written to disk. Depends on `core/`, `echoes/` (targets, two-way) and, inside one function,
`raytracing/` (absorption of the galactic-noise path).

| File | What it does |
|---|---|
| `arrays.py` | Array element geometry: phase and delay offsets of each element for a given direction. |
| `noise.py` | External noise factor (ITU-R P.372: atmospheric, galactic, man-made) and noise samples. |
| `itu_p372.py` | ITU-R P.372 atmospheric (lightning) noise climatology, ported from the ITU's C reference. |
| `P372/data/` | The 12 monthly coefficient files `itu_p372.py` reads (`COEFF01W.txt` .. `COEFF12W.txt`, ~2.8 MB). |
| `cir.py` | Channel impulse response assembly, the received-signal helper, and the npz + JSON writer. |

## arrays.py

`Array(positions)`: element positions relative to the phase centre (local ENU). For a direction and the leg's
transverse frame it gives each element's longitudinal offset (element factor exp(+i k0 zeta_p), delay offset
-zeta_p/c) and principal-axis coordinates (spec §11.5). `isotropic_pattern` is a placeholder element pattern.
Not used by the rest of the code yet.

## noise.py

Baseline: independent complex white Gaussian noise per element, power k_B T0 B F_a (`noise_power_W`,
`noise_samples`). F_a from ITU-R P.372: `atmospheric_noise_itur_p372_dB` (via `itu_p372.py`),
`galactic_noise_itur_p372_dB` and `man_made_noise_itur_p372_dB` (P.372-18 Table 1). `external_noise_dB` adds
atmospheric and galactic, with the galactic part reduced by the D-region absorption of a vertical path above the
receiver (`upward_path_absorption_dB`, spec §12.6).

## itu_p372.py

`atmospheric_noise_dB(lat_deg, lon_deg, utc_hour, frequency_mhz, month_index0)`: a line-by-line port of
`AtmosphericNoise()` / `GetFamParameters()` / `ReadFamDud()` from the ITU-R Study Group 3 C reference
(github.com/programmin1/ITU-R-HF, `Noise.c`), validated against the compiled C to better than 1e-4 dB. Reads the coefficients from `P372/data/`
(path relative to this file).

## cir.py

- `target_impulse_response(...)`: h(t, tau) of point-target components with range migration, Doppler and carrier
  phase; band-limited delta on the delay grid.
- `assemble_cir(...)`: eq. 12.5, the sum of target responses and any other field on the same grid (clutter, aurora).
- `received_signal(...)`: y = h * x + n for one element; waveforms `fmcw_sweep` and `pulse_train`.
- `write_cir(stem, ...)`: `<stem>.npz` (h, t, tau, component columns) and `<stem>.json` (conventions, component
  table, provenance including the git commit). npz + JSON instead of HDF5 because h5py is not installed (D10).
