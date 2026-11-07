# Implementation status

Companion to [implementation_plan.md](implementation_plan.md). Update this file at the end of each work block: it is what a fresh session should read first. Deviations from the spec are logged in [spec_deviations.md](spec_deviations.md) (decision D10).

## Decisions (user, 2026-10-03)

| Decision | Status | Where it is applied |
|---|---|---|
| D1 X-mode fix (multiply plasma-branch RHS by MODE) | Approved, applied | `main.py` `_haselgrove_physical_rhs`; `algorithm.md` §2.4, §8.0 |
| D1 external reference before publishing X-mode results | **Open.** No PHaRLAP offline. X-mode legs carry `LegMode.xmode_validated = False` | `records.py`, `legs.py` |
| D2 focusing reference P′ with the caustic cap | Approved, applied in Phase 1f | `legs.py` (`flux_tube_distance2`, `focusing`) |
| D3 numpy vectorisation first, numba only if the gate fails | Approved. Gate met (4000 rays × 2000 km in 62 s); numba not needed | Phase 1a |
| D4 D-region from E-CHAIM FIRI (`dregion=True`), auroral/PCA as enhancement on top | Approved, applied in Phase 1h (source) and Phase 2 (merge functions) | `ionosphere.py`, `absorption.py` `merge_density` |
| D5 group path distance metric | Recommendation taken (group path). **Phase 3e (1-D MPS check) not done** | — |
| D6 deprecate the Monte Carlo layer | Recommendation taken: deprecate in Phase 5, delete after 5g. **Not done** (`mc_scattering.py`, `raytracer.py` still present) | — |
| D7 baseline commit, `.gitignore` | **Not approved. Not done.** No commits made | — |
| D8 pytest into venv | **Not approved. Not done.** Tests run with `tests/runner.py` | — |
| D9 TV-1 parameters as reference default; other defaults labelled placeholder | Recommendation taken. Collision profile is a labelled placeholder | `absorption.py` |
| D10 spec deviations log | Applied | `docs/spec_deviations.md` |

## Phase progress

| Phase | Content | State |
|---|---|---|
| 0 | X-mode sign; absorption (Budden angle form, amplitude coefficient, cos α); exact group index; phase sensitivity | **Done**: `tests/test_phase0.py` 15/15 |
| 0 | P0.6 `algorithm.md` update (§2.4 sign, §2.8 absorption, §2.11 budget, §8.0–8.6) | **Done** |
| 1a | Vectorised shape-agnostic `Ray`, `batch.py` BatchTracer, goldens | **Done**: `tests/test_vectorised.py` 6/6 |
| 1b | Per-step records (`records.py`, Listing 1) | **Done**: `tests/test_records_ground.py` |
| 1c | Local helpers: n_angle, branch check, dn/dN by identity, cos α | **Done** (within `main.py`, `batch.py`) |
| 1d | Transported frame (Algorithm 1) | **Done**: `batch.transport_frame`, tests in `test_records_ground.py` |
| 1e | Ground: Fresnel, bounce records, roughness factor | **Done**: `ground.py`, `environment.py` |
| 1f | Landing Jacobian, flux-tube distance (eq. 4.7), caustic cap, P′ | **Done**: `tests/test_legs.py` 6/6 |
| 1g | Doppler Method B (eq. 4.8): accumulator in state slot 10, two epochs, −v_d·∇N_e; per-hop values | **Done**: `tests/test_doppler.py` 4/4 (Method A agrees to 0.05 % and 0.00 %) |
| 1h | Scene-covering patch, 50 km floor, 1 km below 130 km, E-CHAIM FIRI D layer, two epochs | **Done**: `tests/test_ionosphere.py` 7/7 |
| 1i | Fans, landing maps, Algorithm 2 (quad inversion, subdivision), Algorithm 3 (Newton), classification, reciprocity | **Done**: `tests/test_homing.py` 3/3, `tests/test_fans.py` 4/4 |
| 1j | Leg records (LegMode), one-way budget (eq. 6.2–6.4), L_pol 3.01 dB, no coherence term; API `Link.link_budget` updated | **Done**: `tests/test_legs.py` 6/6; end-to-end smoke test on the real patch |
| 2 | Absorption models: Sen–Wyller (eq. 7.2), nu_m/nu_eff rule, merge (§7.2), riometer density (eq. 7.3), per-mode integrand, below-120 km split, §7.3 attenuation flag, §7.6 item 4 (secant law) | **Done**: `tests/test_absorption.py` 8/8. Protocol conformance: `absorption.ProfileModel`. Not wired: the §7.2 merge in the public API (needs the auroral/PCA enhancement input) |
| 3 | Irregularity, screens, channel parameters (TV-1…TV-4 gate) | **Done**: `tests/test_tv_gate.py` 6/6, `tests/test_stochastic_units.py` 10/10 |
| 3e | MPS reference (§15.3, on demand, not CI) | **Engine done; D5 not decided.** `validation/mps_1d.py` reproduces exp(−D/2) for one screen to 1.2×10⁻³ and conserves intensity. The free-space unfolded model (§9.1) cannot decide group vs arc by itself: both enter the same propagator. A refracting-slab MPS (local index n(z) between screens) is needed for that comparison; not built |
| 4 | GPSD, realisations, arrays, Rician assembly | **Done**: `tests/test_gpsd.py` 8/8 |
| 5 | Two-way, targets, clutter, noise, CIR, API (Listing 4) | **Partly done.** Done: `cascade.py` (§12.1 realisation, Doppler and power additivity tested: 5/5); `twoway.py` (eq. 6.1 Doppler, eq. 6.2 radar equation, components, Γ_or eq. 12.3: 8/8); `targets.py` (ground targets: fan inversion Algorithm 2 + Newton homing Algorithm 3, one component per out/ret pair, Γ_or per component, range migration: 3/3); `clutter.py` (Bragg frequency 0.323 Hz at 10 MHz, cardioid spreading normalised, line strength 2^6 π k0⁴ S(K_B) = 4 π B_s G (Barrick's normalisation since 2026-11-07, deviation 46), sea-ice and current terms, expected range–Doppler map with power-sum check, single-element Gaussian realisation: 8/8); `noise.py` (eq. §12.6 with F_a placeholder, galactic attenuation by the upward path: 3/3); `cir.py` (eq. 12.5 assembly, received-signal helper for FMCW and pulse trains, npz + JSON writer: 4/4). **Also done this run:** targets at altitude (§5.3, `BatchTracer(landing_alt_m=h_T)`, descending crossing on the final hop, 4/4 in `test_targets.py`); `clutter_scene.py` (fan landing maps rasterised onto the midpoint grid, cell typing from the land mask, sea lines or land line, flux-tube distances from the planar Jacobian, element phases from the return launch directions; 3/3). **Not done:** the Gaussian (u, v) grid realisation (§12.4.1 step 2 for arrays: explicit components are used instead), the angular spread of the return leg, Listing 4 API, deletion of the Monte Carlo layer (D6, after 5g) |
| 6 | Auroral backscatter (§12.5), CIT wander (§10.9), partial reciprocity (§12.2), tangent-map screen geometry (§9.1, optional) | **Partly done.** 6a: `auroral.py` (structure, 4/4) and `auroral_scene.py` (fan from the midpoint on the batch path, volume elements from neighbour rays at equal arc length, power through the radar equation, quasi-monostatic delay and Doppler; 2/2). The bistatic refinement is not done. 6b: in Phase 3 (`channel.wander`, eq. 10.9, Table 12). 6c: Γ_or per target component (`leg_screens.py`, 1/1). 6d: not started (optional). 6e: not needed (gate met, D3) |
| 7 | Validation against data | Blocked: needs data from the user |

## Validation status

- X-mode: checked against ∂ω/∂k from the dispersion relation and the energy-flux argument; the end-to-end reciprocity test passes. **No external reference**: X-mode numbers are internally validated only.
- Reciprocity (spec §5.5): O and X, one and two hops, planar and dipped fields. Phase, group path and absorption agree to ≤ 2 × 10⁻⁸ (required 10⁻⁶). `tests/test_homing.py`.
- Stochastic core: TV-1…TV-4 reproduced to printed precision, except TV-2 refined S4² (0.7 %, inside the 2 % tolerance; cause not identified).
- Doppler: Method B agrees with Method A (re-homed phase-path difference) to 0.05 % (uniform decay) and 0.00 % (advection). The control without the advection term is −100 %.
- Absorption: the Sen–Wyller limits (5/2 and 2/3 of ν_m) and the non-deviative agreement with the complex form to 1 % (ν_m/ω′ ≤ 0.02; the difference grows as x², 2 % at 0.05) are tested.

## Known open points

- **X-mode external reference** (D1): not available offline. Mark X-mode outputs as internally validated only.
- **Placeholder defaults** (D9): the collision-frequency profile, the E/F valley (180 km) and the surface constants are labelled placeholders. Absolute absorption numbers inherit this.
- **API on one path (resolved this run)**: `Link.trace`, `link_budget`, `sweep_*` and `plot` run on the batch path. The example's ray agrees between RayPath and LinkBudget (absorption 17.52 dB, ground range 940.0 km, delay 3.622 ms). The budget's focusing is +3.64 dB: the central ray is now the same ray as trace (the earlier +2.11 dB came from a different launch frame). `scatter()` still runs the deprecated Monte Carlo tracer and warns; `Link.leg_channel()` is the batch replacement.
- **User inputs** (`inputs.py`, decision of this run): every value that needs your data is a field of one of six dataclasses (IrregularityInputs, BraggInputs, SurfaceInputs, NoiseInputs, TargetInputs, RadarInputs). Defaults are placeholders; `inputs.placeholders()` lists them. Nothing reads them from module state.
- **Placeholders introduced in this block** (D9): clutter wind direction, B_s = 0.005, cardioid s = 3, ice return and land σ0; external noise F_a (20 dB, both terms); Γ_or irregularity parameters (ξ = 1e-2, L0 = 1 km, p = 4, zero drift); auroral η⊥, A_s and v_ph. All of them change the numbers, none of them the structure.
- **TV-2 refined S² (0.64 % after convergence)**: the φ quadrature was not converged at 256 points (106.725 at 256, 106.662 at 1024 and 4096); the default is now 1024. The G_p table agrees with direct quadrature to 8×10⁻⁶ over the TV-2 β range (1×10⁻⁴ to 0.09), so the table is not the cause. The residual is still unexplained; it is inside the 2 % tolerance.
- **Not started, needs your input**: the Listing 4 API (decision on the API direction first); deletion of the Monte Carlo layer (confirmation first); the §7.2 merge in the public API (needs an auroral/PCA enhancement model); the clutter field on the (u, v) grid (§12.4.1 step 2 for arrays); the target at altitude (§5.3); the fan-based auroral integration (§12.5).
- **D5 (group vs arc distance)** is still open: the MPS engine is validated for one screen, and the curved-leg comparison that decides the metric has not been run.
- **Phase 7** needs data from the user (riometers, SuperDARN, OTHR records, GNSS σ_φ and S4).
- **API**: `Link` is single-epoch, so the Doppler is NaN there. The batch path has the two-epoch Doppler (`ionosphere.EpochPair`). The Listing 4 API (Scenario, Environment, ChannelSimulator, SimOptions, FanSpec, Site, Array, ElementPattern, Waveform) is not written.
- **Fold in the test ionosphere**: the single-layer test profile has no interior skip fold in a one-hop landing map (ground range is monotone over the landed rays). The subdivision logic is tested on an explicit folded bilinear map instead.
- **Group index off the shell**: `Ray.group_index` is defined on the dispersion shell; off-shell states give values that depend on the convention (goldens are now generated on the shell per mode).
- **Goldens regenerated** (`tests/golden_make.py`): group-index and phase-sensitivity goldens use on-shell states; the trace goldens come from the corrected physics. Reason: the old group index was noisy for X-mode (§3.3 of the plan, Phase 1j debugging).

## Log

- 2026-10-03: Phase 0 done. Found and fixed: collision term wrong for B != 0 (X/(1−iZ) instead of the Budden U form). Angle form now in `Ray.n_angle`.
- 2026-10-03: Phase 3 done. TV-1..TV-4 reproduced, except TV-2 refined S4² (0.7 %, cause not identified). Spec inconsistency: §8.3 vs Tables 11–12 on the drift projection; the projection is a model-output check.
- 2026-10-03: Phase 4 done. Realisation sign convention verified by the frozen-flow test (spec §11.7 item 5).
- 2026-10-03: Phase 1a done. Vectorised `Ray` and `BatchTracer`. 4000 rays in 62 s.
- 2026-10-03: Phase 1b–1f done. Records (Listing 1), transported frame, Fresnel and bounce records, landing Jacobian and P′ focusing (D2), caustic cap.
- 2026-10-03: Phase 1g done. Doppler Method B: accumulator in state slot 10, per-hop values through `hop_states`, `EpochPair` for two epochs.
- 2026-10-03: Phase 1h done. `covering_patch`, 50 km floor, 1 km below 130 km (`_altitude_axis`), E-CHAIM `dregion=True` (decision D4). FIRI D layer is zero at 50–51 km and rises from 52 km at 65° N (E-CHAIM's own bound). `electron_density_geodetic` now broadcasts scalar lat/lon with array heights.
- 2026-10-03: Phase 1i done. Batch states carry `hop_states` and `hop_apex` (height above the ellipsoid) at every ground contact. `fans.py` (landing maps, Algorithm 2 with subdivision, classification) and `homing.py` (Algorithm 3, reciprocity).
- 2026-10-03: Reciprocity failure traced to two causes, both fixed. (a) The group index partials were evaluated at the state's |u|². On the shell 2ap + b = X·B_ε + 2aε, which has no cancellation, so the partials are now evaluated on the root p = 1 + ε (the one the RHS uses). The noise was 1 % in n_g and 10⁻⁵ in P′. (b) The default collision profile was cut at 130 km, a step in the absorption integrand. It is now continuous above 130 km.
- 2026-10-03: Phase 1j done. `legs.landing_mini_fan` returns phase, absorption, per-bounce ground loss, Doppler and a §5.4 label; `link_budget.one_way_budget` implements eq. (6.2)–(6.3) with L_pol 3.01 dB and no coherence term; `api.Link.link_budget` uses it (the report's ray path still comes from `Link.trace`). Smoke test on the real patch: landed at 938 km, delay 3.62 ms, focusing +2.11 dB, absorption −17.4 dB, polarization −3.01 dB.
- 2026-10-03: Phase 2 done. Sen–Wyller F(x) (quadrature, log-log table), nu_m / nu_eff rule with ν_eff = 2.5 ν_m above 100 km, O/X through ω′ = ω(1 ± Y|cos θ|), `merge_density`, `riometer_density` (eq. 7.3), `ProfileModel` (Listing 2 protocol), below-120 km absorption in state slot 11 (`LegMode.absorption_D_dB`).
- 2026-10-03: Phase 5 (part): `twoway.py` (target components, radar equation, Γ_or). Noise, clutter, CIR and the Listing 4 API are not started.
- Deviations logged: records from the fixed-step driver only; HDF5 output falls back to npz + JSON (h5py not installed); the single-epoch API has no Doppler; the E/F split uses one valley altitude.

- 2026-10-03 (block 5/6): attenuation flag (§7.3) on the batch path; E–F valley from the density column; D6 DeprecationWarning on RayTracer and MCScatteringTracer; cascade (§12.1); Γ_or (§12.2) on traced legs and in the target pipeline; targets (§5.5, §12.3); noise (§12.6); clutter (§12.4) with the single-element field; CIR and output (§12.7, §14.2); auroral structure (§12.5); MPS engine (§15.3); TV-2 quadrature (n_phi 1024; residual 0.64 % open).
- 2026-10-31 / 2026-11-07: HiCIRF-matching work (plan `.claude/plans/valiant-questing-robin.md`; open items in `docs/needed_from_user.md`; deviations 40-49). Unified irregularity defaults (L0 = 30 km); per-sample irregularity/drift models through `Link(irregularity_model=..., drift_model=...)`; `wbmod.py` (eq. 5, our own crude WBMOD), `plasma_drift.py` (full Heelis port, crude mid/low-latitude, IGRF dipole frame), 5-deg screen coarsening, `itu_p372.py` (validated P.372 atmospheric noise), `ccmp_wind.py` (real CCMP, cached), `second_order_bragg.py` (validated against Barrick 1972), `meteor.py` (MIF meteor clutter, straight-line and ray-traced). netCDF4 installed (user permission 2026-11-07). Sea spectrum switched to Barrick's normalisation (user decision 2026-11-07, deviation 46): first-order sea lines +8 dB, second order +16 dB. Tests: 190/190.
- 2026-11-07: Code reorganised into folders (packages imported from the project root): `core/`, `geophysics/` (with the
  vendored `ECHAIM/`), `raytracing/`, `propagation/`, `stochastic/`, `echoes/`, `receiver/` (with `P372/data/`), `legacy/`;
  `api.py` and `example.py` stay at the root. File names unchanged; every import rewritten (`from echoes import clutter`).
  Each folder has a `README.md`; the root `README.md` maps them. Tests: 190/190.
- 2026-11-07: Plotting in PHaRLAP's look (new `plotting/`): `plot_ray_iono_slice` ported line by line from PHaRLAP's
  MATLAB routine, the 3-D `plot3` figure of its `raytrace_3d` examples, this group's conventions (MuellerMap, bold
  text, white fan with a red highlighted ray), and the HiCIRF channel of the highlighted ray (phase screens, channel
  parameters, delay-Doppler scattering function). `Link.plot` / `plot_3d` / `plot_scattering_function` / `plot_html`;
  `raytracing/visualization.py` became `plotting/plotly_3d.py`. Fixed on the way: `stochastic.channel.secant_matrices`
  gave 0/0 for the last screen of every leg built by `screens_for_leg` (it sits at Z_L, w = 0), making V_app, <tau>
  and omega_coh NaN; now 0 there (its exact contribution, every use is weighted by w or z_eff). Seen, not fixed:
  E-CHAIM segfaults while building patches wider than about 60 deg (same with the code before today's changes).
- 2026-11-07: Public API review (`api.py`). `RayPath.samples` keeps the traced leg, so the new `Link.channel(ray)`
  and the HiCIRF part of `Link.plot` use the very ray that was traced (before, `plot` re-traced it with the Link's
  default budget and step); `leg_channel(elevation, ...)` is `channel(trace(...))`; `plot_scattering_function(ray)`.
  `Link(irregularity=IrregularityInputs(...))`, with `xi_rms`/`L0_km`/`p_index`/`axial_ratio` as field overrides;
  mode stored as "O"/"X". Removed: `trace(method=, tangent_map=)` (could only raise), `RayPath.tangent_map` (always
  None), the dead `_build_ray_path`, duplicate imports. Result types compare by identity; `ChannelParams` got a
  readable `str()` and `rician_K_db`; `api` re-exports `ChannelParams`, `ChannelOptions`, `IrregularityInputs`.
