# 3-D ADE-FDTD in magnetized, collisional plasma

Solves Maxwell's equations on a proper 3-D Yee grid coupled to a cold
magnetized electron plasma via the Auxiliary Differential Equation method
with J sub-cycling (after Pokhrel/Simpson-style schemes, generalized to a
full 3x3 gyro-matrix with the signed electron gyrovector), CPML absorbing
boundaries, selectable sources, and an optional nonlinear fluid mode.

```
python main.py                                # whistler demo -> GIF at end
python main.py --preset nonlinear_demo        # nonlinear electron fluid
python main.py --preset scattering            # plane wave vs blobs + report
python main.py --source loop --steps 5000     # pick a source
python main.py --preset ionosphere --no-plot  # physically-scaled F region
python tests.py                               # verification suite
```

Runs go to completion with console progress reports; a slice of the volume
is captured periodically and compiled into an animated GIF at the end.

The `scattering` preset is the flagship scenario: a sheet source launches a
z-polarized (O-mode, E || B0) quasi-plane wave along x through a
transparent background (`omega > omega_p,bg`) into overdense blob
structures whose cores exceed the critical density, so each blob carries a
critical layer -- reflection, diffraction and shadowing -- while the
nonlinear fluid responds self-consistently (the sheet amplitude puts the
ponderomotive potential at a few % of k_B T_e). Besides the GIF it ends
with a `ScatteringReport` figure (`scattering_report.png`): field snapshot
with critical-density contours, time-averaged intensity in dB, the density
imprint (n - n0)/n0, probe spectra (harmonics at 2f0/3f0 are the
nonlinearity fingerprint), an axial transmission profile against the
structure density, and summary metrics.

## Files

| file | contents |
|---|---|
| `constants.py` | SI constants; **signed** species charges |
| `plasma.py` | collision frequency, sub-cycle X/G matrices, exact velocity push |
| `grid3d.py` | Yee fields, curl+CPML updates, linear J update, fluid species |
| `cpml3d.py` | CPML profiles (integer + half-node staggered) |
| `sources.py` | rotating_charge, dipole, loop, sheet + registry |
| `plasma_density.py` | 3-D synthetic ionosphere, blob structures, PML taper |
| `diagnostics.py` | GIF recorder, progress reports, scattering report figure |
| `tests.py` | verification suite (math + physics) |
| `legacy_2d/` | snapshot of the original 2-D implementation |

## Scheme

Yee staggering, with J components on the E edges and scalars on nodes.
Leapfrog in time: `E` at integer steps, `H` and `J` at half steps.
Per step: update `H` (from `E^n`), update `J` (window centered on `E^n`),
update `E` (from `H^{n+1/2}`, `J^{n+1/2}`, source at `t^{n+1/2}`).

**Linear mode** — the current obeys
`dJ/dt = eps0 wp^2 E - nu J + J x Omega`, `Omega = (q_e/m_e) B0` (signed!).
Crank-Nicolson over `dt_c = dt/M` gives a per-sub-step map `C = A^-1 B`;
M sub-steps with `E` frozen collapse to `J' = X J + G E` with `X = C^M`,
`G = (sum_{k<M} C^k) A^-1 eps0 wp^2 dt_c`, computed for the whole grid in
`O(log M)` batched 3x3 multiplies and interpolated to each edge family.
`C` is a Cayley transform, so `||C|| <= 1`: the sub-cycled J update is
unconditionally stable in `nu dt` and `Omega dt`. M is chosen from
`max(nu, |Omega|)` only — `omega_p` constrains the Yee `dt` instead:

```
3 (c dt/dx)^2 + (wp_max dt/2)^2 <= 1      (stability)
wp_max dt <~ 0.5                          (accuracy, configurable)
```

**Nonlinear mode** — full electron (optionally ion) fluid per species:
`dn/dt = -div(n u)`,
`du/dt = (q/m)(E + u x B) - nu u - (u.grad)u - (kT/m) grad n / n`,
`J = q n u`. The E/B/nu part is integrated *exactly* each step (Rodrigues
rotation about the local total B plus drift relaxation — the M->infinity
limit of the sub-cycling), so no sub-cycling is needed; advection and
pressure are explicit. This captures harmonic generation, ponderomotive
density modification, and wave interaction with self-consistently evolving
structures. `--ions` adds a mobile O+ fluid (needed for slow-timescale
cavitation; electrons-only violates quasi-neutrality on ion timescales).

## Bugs fixed relative to the 2-D code

1. **No spatial staggering** — fields were collocated, so the discrete
   div-curl identities failed and a Poisson "Gauss-law cleanup" ran every
   step (and compared `div E` to `rho`, missing the `1/eps0`). With a real
   Yee grid, `div E = 0` off-source holds to ~1e-12 with no cleanup at all
   (see `tests.py`), and the 100-iteration Jacobi solve is gone.
2. **Endpoint J in Ampere's law** — J was co-timed with E and its endpoint
   value fed back, which is unstable for moderate `wp dt`. J now lives on
   half steps with the sub-cycle window centered on `E^n` (standard
   time-centered leapfrog ADE).
3. **Off-by-one in X and G** — the loop built `X = C^{M-1}` and summed
   `C^k` only to `M-2`; correct closed form is `C^M` and `sum to M-1`
   (verified against brute-force sub-stepping to machine precision).
   `M = 1` also crashed (`X` stayed `None`).
4. **Unsigned electron charge in the gyrofrequency** — `q_e B/m_e` with
   positive `q_e` gives the wrong gyration handedness, which inverts which
   circular polarization is the whistler mode. The gyrovector is now
   signed; the handedness test drives the plasma with R- and L-rotating
   sources and checks that the field arriving down B0 is R-polarized.
5. **CFL budget ignored the plasma term** — `dx = 1.01 sqrt(2) c dt` put
   the curl at ~the full Courant limit, and the plasma term on top made the
   shortest grid modes unstable (the chronic blow-ups). The combined
   condition above now sets `dt`.
6. **CPML alpha far above the wave band** — with `alpha ~ 0.1 S/m`, the
   CFS corner sits at `alpha/eps0 ~ 1e10 rad/s`; VLF/MF waves fall below it
   and the PML stretches without absorbing (~-7 dB, fields trapped in the
   layer). Default is now `alpha = 0` (measured -160 dB energy residual);
   `sigma_max` also used `log10` where the natural log belongs, and E/H
   now see integer/half-node profiles respectively.
7. **Collision frequency units** — `2.9e-6 n lnLambda T^-1.5` takes n in
   cm^-3 and T in eV; the code fed m^-3 and Kelvin (two ~1e6 errors that
   happened to nearly cancel). Now converted explicitly.
8. **Plasma extended into the PML** — the density now tapers to vacuum
   just inside the layer, so the PML absorbs in the medium it was derived
   for.

## Notes and limitations

- Real F-region densities put `wp ~ 1e8 rad/s`, forcing `dt ~ 4e-9 s`:
  that is intrinsic stiffness of explicit ionospheric FDTD, not a code
  limit. The `demo_whistler` preset scales `n0` down so `wc > wp > w_src`
  and runs interactively; `ionosphere` is physically scaled and slow.
- Nonlinear mode: B in the push is taken at `t^{n+1/2}` (half-step off
  center in the nonlinear correction only); ion collision rates reuse the
  electron-ion formula; advection is centered (add resolution rather than
  relying on it for shock-like features). Fluid speed is clamped at 0.3c
  (`species.clamp_count` reports if that ever engages).
- Performance: ~1 s/step linear at 128^3 (~1.6 GB), ~1.3 s/step nonlinear
  at 96^3, single-threaded numpy. Memory scales as N^3.
