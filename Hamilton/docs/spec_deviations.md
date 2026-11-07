# Deviations from the engineering specification

Decision D10 (user-approved recommendation): every place where the implementation departs from, or
goes beyond, `HiCIRF_HighLatitude_OTHR_Engineering_Spec` is logged here, with the reason and the
code that carries it. Companion files: [implementation_plan.md](implementation_plan.md) and
[implementation_status.md](implementation_status.md).

## Numerical and physical corrections (the spec is not followed literally)

| # | Spec location | What the implementation does | Why | Where |
|---|---|---|---|---|
| 1 | §8.3 (remove the drift component along B) vs Tables 11–12 | `build_screens` does not project the drift. `environment.check_drift` warns when the drift has a B-component above 5 % | Table 11/12 use the unprojected drift; the two statements conflict. The projection is a model-output check, not a screen change | `screens.py`, `environment.py` |
| 2 | §4.1 / §4.2 (absorption and group index) | Absorption uses Budden's angle form with U = 1 − iν/ω; the group index uses the exact expression with the partials evaluated on the dispersion root p = 1 + ε | The earlier forms were wrong for B ≠ 0 (factor 1 ± Y). On-shell evaluation removes a cancellation that made the X-mode group path noisy (reciprocity 1e-5 before, 1e-9 after) | `absorption.py`, `main.py` `group_index` |
| 3 | §7.1 (illustrative default ν) | Default collision profile 1e7 exp(−(h−70)/6) s⁻¹ is a placeholder (D9). It is continued above 130 km instead of being cut | The cut was a step in the absorption integrand, which broke the §5.5 reciprocity test. The value above 130 km is ≈ 4e2 s⁻¹ | `absorption.py` |
| 4 | §4.5 (focusing reference) | Flux-tube focusing uses the group path P′ (decision D2) with the caustic cap at 10 dB | As the spec says; the mini-fan Jacobian is by central differences of five rays (δ = 0.01°) | `legs.py` |
| 5 | §6.3 (power budget) | No coherence term. The previous `link_budget` coherence loss is removed | Spec pitfall: scattering redistributes power and does not remove it; the split is the Rician model (§10.6) | `link_budget.py`, `api.py` |
| 6 | §5.1 (planar coordinates "azimuthal equidistant") | Azimuthal equidistant about the site on the geocentric sphere, R_MEAN | Planar map for the quad inversion; the ellipsoid error is about 0.3 % of the distance | `fans.py` `site_planar` |
| 7 | §5.4 (E / F / Es layer classification) | One valley altitude (180 km) separates E and F. No sporadic-E layer | No profile analysis yet; the model has no Es layer | `legs.py` `layer_letter`, `VALLEY_M` |
| 8 | §4.8 (fixed or adaptive driver, records) | Records and the per-hop states come from the fixed-step driver only. Adaptive dense-output sampling is not implemented | Batch fixed-step RK4 with step rejection is what the vectorised path uses; the adaptive path stays for tangent maps only | `batch.py` |
| 9 | §4.7 (patch and D-region) | E-CHAIM v3.4 with `dregion=True` gives the D layer. At 65° N the FIRI D layer is zero at 50–51 km and rises from 52 km | E-CHAIM's own lower bound. Decision D4 | `ionosphere.py` |
| 10 | §4.6 (Doppler, Method A) | Method A is implemented in the tests only, with a 1-D secant re-home on elevation (stand-in for homing.py until its Newton form is used everywhere) | Method A is expensive; the batch path uses Method B | `tests/test_doppler.py` |
| 11 | §3.4 / B = 0 | B = 0 exactly is degenerate for Haselgrove's form (ṙ ≡ 0, double root). Tests use a small non-zero B (east or dipped) | Limitation of the form, not a numerical issue | `tests/test_phase0.py`, `tests/test_records_ground.py` |

## Scope limits (not yet implemented, or implemented in part)

| # | Spec location | State | Plan |
|---|---|---|---|
| 12 | §10.6 / §12.2 (reciprocity, two-way, targets, clutter, noise, CIR, API Listing 4) | Not started | Phase 5 |
| 13 | §12.5 auroral backscatter, §12.2 partial reciprocity | Not started | Phase 6 |
| 14 | §4.6 Doppler in the public API | The API `Link` is single-epoch, so `doppler_hz` is NaN there. `EpochPair` (two snapshots, identical grids) and the batch accumulator work | Phase 5 API |
| 15 | §7.3 absorption leg flag (k0 |Im n| λ0 > 0.1) | Not implemented | Phase 2 follow-up |
| 16 | §7.2 E-CHAIM merge | `merge_density` and `ProfileModel` are implemented. The API still traces with E-CHAIM alone | Phase 5 API |
| 17 | §5.5 two-way matching | Reciprocity test implemented and passing (1e-6 for O and X, one and two hops). Two-way matching of Tx and Rx legs not yet | Phase 5 |
| 18 | §12.6 HDF5 output | h5py is not installed. Output falls back to npz + JSON | No new dependency without approval |
| 19 | §7 X-mode validation | No external X-mode reference is available offline (PHaRLAP). X-mode outputs carry `xmode_validated = False` (see the status doc) | Obtain a reference before publishing X-mode numbers |
| 20 | TV-2 refined S4² | Reproduced to 0.7 %, inside the spec's 2 % tolerance. The cause is not identified | Open, logged |

## Added in the Phase 5–6 block (2026-10-03)

| # | Spec location | What the implementation does | Why | Where |
|---|---|---|---|---|
| 21 | §12.6 (F_a from ITU-R P.372) | External noise factor is a PLACEHOLDER (20 dB atmospheric and 20 dB galactic). The galactic part is attenuated by the upward-path absorption, as the spec says | P.372 data is user data (plan Phase 5e) and is not available offline | `noise.py` |
| 22 | §12.4 (sea, ice, land inputs) | Wind direction, B_s = 0.005, cardioid s = 3, ice return and land σ0 are PLACEHOLDERS. Spectrum normalisation: see 46 (Barrick's since 2026-11-07; the dominant line at s = 3 is −15 dB, not the spec's "order −25 dB") | The surface model is user data (CCMP and similar) | `clutter.py` |
| 23 | §12.4.1 (field-based generation) | Step 1 (expected power map) and step 2 for a single element are done. The (u, v) array dimension, the rasterisation of landing-map cells and the common-reflectivity correlation are not | Needs the array geometry and the cell rasterisation; scoped to the next block | `clutter.py` |
| 24 | §12.2 (Γ_or) | Screens of a traced leg use PLACEHOLDER irregularity parameters (ξ = 1e-2, L0 = 1 km, p = 4, a = b = 1, zero drift). Γ_or values in the target pipeline depend on them | The irregularity and drift models are user data | `leg_screens.py` |
| 25 | §5.3 (target at altitude) | Ground targets only. The crossing event at h_T is not in the tracer | Aircraft targets need the altitude event in the tracer; not yet | `targets.py` |
| 26 | §12.5 (auroral backscatter) | Structure only: Bragg vector, aspect law, volume-element power and Doppler. The fan-based volume integration from the midpoint of the sites is not implemented. η⊥, A_s and v_ph are PLACEHOLDERS | The Bragg-scale irregularity model (Listing 3) is user data | `auroral.py` |
| 27 | §10.9 (CIT-limited wander) | Implemented in Phase 3 (`channel.wander`, eq. 10.9), checked against Table 12 | — | `channel.py`, `tests/test_tv_gate.py` |
| 28 | §15.3 (MPS reference) | Plane-wave 1-D engine, on demand (not CI). Validated for one screen (MCF vs exp(−D/2)) and for intensity conservation. The group-vs-arc decision (D5) is not yet made: it needs the curved-leg comparison | Decision D5 needs a leg-specific screen set and the curved geometry | `validation/mps_1d.py` |
| 29 | §9.3 / TV-2 quadrature | The azimuth quadrature of the refined S² is 1024 points (was 256, which was not converged for TV-2). Residual vs the table: 0.64 % (converged), unexplained; G_p table is not the cause | Convergence, found in this block | `channel.py` `ChannelOptions.n_phi` |
| 30 | §5.4 (E/F valley) | The valley can be taken from the density column (local maxima and the minimum between them). The public API still uses the 180 km placeholder | Wiring into the API waits for the API decision | `legs.py` `valley_altitude_m` |
| 31 | Decision D6 | `RayTracer` and `MCScatteringTracer` emit a DeprecationWarning and keep working. Deletion waits for confirmation (after 5g) | Approved as deprecate-now, delete-later | `raytracer.py`, `mc_scattering.py` |
| 32 | §12.7 / §14.2 (output) | CIR written as npz + JSON (the HDF5 fallback of item 18), with conventions, component and leg tables, reciprocity class, models, seeds, options and provenance | h5py is not installed | `cir.py` |

## Added in the batch-path run (2026-10-03, second run)

| # | Spec location | What the implementation does | Why | Where |
|---|---|---|---|---|
| 33 | Decision D6 and the API | `Link.trace`, `link_budget`, `sweep_*` run on the batch path. `tangent_map` and `method` are not `trace()` options (the scalar `RayTracer` keeps them). `phase_variance` takes sigma_phi2 from the leg's phase screens with the user irregularity inputs. `scatter()` runs the deprecated Monte Carlo tracer and warns. `Link.channel(ray)` (or `leg_channel(elevation, ...)`) is the batch replacement (channel parameters of one traced leg, from the ray's own samples) | "Move everything to the batch path" (user instruction) | `api.py` |
| 34 | §8.5 (drift) | The default drift is a PLACEHOLDER 150 m/s eastward, in the local east and north frame of the site (a TV-1 table drift is per screen and cannot be reused). With zero drift and zero turbulence the decorrelation time is undefined (no temporal structure), so a placeholder drift is needed for channel parameters | Drift model is user data | `inputs.py` `IrregularityInputs.drift_east_north_mps`, `api.py` `_drift_ecef` |
| 35 | §5.3 (target at altitude) | The aircraft landing is a descending crossing of the ellipsoid offset by h_T, on the final hop only. Intermediate hops reflect off the true ground. Homing uses the same tracer (`landing_alt_m` must equal the target height; the pipeline refuses otherwise) | As the spec describes it | `batch.py` `_segment`, `targets.py` |
| 36 | §12.4 (cell rasterisation) | Each fan quadrilateral is inverted bilinearly for each ground cell whose centre it contains. The flux-tube distance comes from the planar Jacobian in the midpoint frame (no correction for the frame distortion over the scene). The fans must be denser than the cells: a cell larger than a quadrilateral gets no contribution | Algorithm 2 per cell, scoped to the scene | `clutter_scene.py` |
| 37 | §12.4.1 (array field) | Explicit components with element phases exp(i k0 (u x + v y)), with one arrival direction per component (the return launch direction). The Gaussian (u, v) grid realisation and the return leg's angular spread are not built | Explicit components are the spec's own option for small scenes | `clutter_scene.py` `element_phases` |
| 38 | §12.5 (auroral fan) | Quasi-monostatic baseline: fan from the midpoint, volume from neighbour rays at equal arc length, delay 2 P'/c, Doppler q_B . v_ph/2 pi with the ionospheric Doppler term zero. The bistatic refinement is not built. η⊥, A_s and v_ph are user inputs (placeholders) | As the spec's baseline | `auroral_scene.py` |
| 39 | §9.1 and §15.3 (D5, distance metric) | Not decided. The free-space unfolded MPS uses the same distances as the analytic formulas, so it cannot choose between group path and arc length. A refracting-slab MPS is needed | Modelling question, not a numerical one | `validation/mps_1d.py` (engine only) |

## Added in the HiCIRF-matching work (2026-10-31 to 2026-11-07; plan `.claude/plans/valiant-questing-robin.md`)

| # | Spec / paper location | What the implementation does | Why | Where |
|---|---|---|---|---|
| 40 | §8.4 / HiCIRF eq. (5) | `inputs.IrregularityInputs` is the single source of the irregularity defaults (api.py and leg_screens.py had their own, differing sets). L0 = 30 km, HiCIRF's fixed outer scale | Three silently different default sets; HiCIRF fixes Lo | `inputs.py`, `api.py`, `leg_screens.py` |
| 41 | §8.5 / HiCIRF §2.3 | Drift models, opt-in via `Link(drift_model=...)`: `HeelisDrift` (full port of TIE-GCM heelis.F90 flwv32/potm), `RichmondDrift` (our own crude pattern: the real model is not online), `WBMODStyleDrift` (superposition, our own rule). Geomagnetic frame: IGRF centred dipole for the date | WBMOD's drift is Richmond-NCAR + Heelis-UTD; user decisions 2026-11-07 | `plasma_drift.py` |
| 42 | HiCIRF eq. (5) | `wbmod.CrudeWBMOD`: our own crude climatology of q and CkL (WBMOD is export-controlled), xi from eq. (5) over the Link's Ne^2 column, p = q + 2 | User decision 2026-11-07: only needs to return numbers | `wbmod.py` |
| 43 | HiCIRF par. 17 | Screens every 5 deg of ray-direction change, opt-in (`coarsen_elevation_deg`); default stays one per sample | Decision 3 | `leg_screens.py`, `api.py` |
| 44 | §12.4 / HiCIRF par. 24 | CCMP v3.1 wind per sea cell (`wind_model`), downloaded and cached on demand; absolute compass direction converted per cell to the look-relative phi_rel; NaN (sea ice) falls back to the scalar inputs | Real wind field as HiCIRF uses | `ccmp_wind.py`, `clutter_scene.py` |
| 45 | §12.4 / HiCIRF par. 31 | Second-order sea echo, Barrick (1972) eqs. (12:31)-(12:34), opt-in (`second_order=True`). Delta = 1/sqrt(eps_c) in the e^{-i omega t} convention. Validated against Barrick's Figs. 12.18-12.19 to 0.04-1.5 dB | HiCIRF has first and second order | `second_order_bragg.py`, `clutter.py` |
| 46 | §12.4 | Sea spectrum S = B_s k^-4 G (int G dφ = 1), Barrick's (1972) / Phillips' normalisation, instead of the spec's (B_s/2 pi) k^-4 G: 2 pi (8 dB) more energy for the same B_s. First-order line σ0 = 4 pi B_s G (−15 dB dominant line at s = 3; semi-isotropic sea −17 dB, Barrick eq. 12:23); second order rises 16 dB, i.e. 8 dB relative to first | User decision 2026-11-07 ("HiCIRF-exact"): HiCIRF uses Barrick's theory | `clutter.py`, `second_order_bragg.py` |
| 47 | §12.6 / HiCIRF | ITU-R P.372 atmospheric noise: port of ITU-R SG3's own C reference (Noise.c), coefficients vendored under `receiver/P372/data/`, validated against the compiled C to < 1e-4 dB. Galactic and man-made from P.372-18 Table 1 | Replaces the flat 20 dB placeholder | `itu_p372.py`, `noise.py` |
| 48 | HiCIRF par. 34 | Meteor clutter: radiants from Fentzke et al. (2009) Table 1, source weights from Fentzke and Janches (2008) Table 3, underdense-trail RCS (McKinley theory, arXiv:2209.07690 eqs. 48-50), D_a(h) user-supplied. q_e log-uniform and r0 fixed by design (ablation model dropped). Straight-line assembly and a ray-traced one (homing per trail, 5 km altitude bins) | — | `meteor.py` |
| 49 | HiCIRF par. 35 | Direct auroral backscatter goes beyond HiCIRF (2012); it is opt-in by construction (only called explicitly) | Decision 2 | `auroral_scene.py` |
