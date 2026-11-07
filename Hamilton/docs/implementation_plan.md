# Hamilton → high-latitude HF channel simulator: implementation plan

Written for: the engineer implementing the spec (you, and future coding sessions). Read it with [algorithm.md](algorithm.md), which describes the code as it is today.

- **Spec:** *High-Latitude HF Channel Simulation for Bistatic OTH Radar, Engineering Specification v1.0* (26 Sep 2026). "§" means a section of the spec unless marked otherwise.
- **Status:** plan only. No code was changed to write it. Checks were run from scratch scripts outside the repo (see Appendix B).
- **Date:** 2026-10-03.

---

## 0. Summary

The spec replaces Hamilton's Monte Carlo layer with a stochastic channel model (phase screens → GPSD → channel realisations). It also adds the deterministic pieces an OTH radar needs: launch fans, mode finding, absorption, Doppler, and two-way assembly. Most of its equations hold up under checking (§1). Three things must be dealt with before any of it can be trusted:

1. **X-mode rays are traced backwards. This is blocking.** The ray direction in `Ray._haselgrove_physical_rhs` is opposite to the physical group velocity for MODE = −1. Every X-mode result produced so far is affected. The spec's branch test (§4.2 item 2) checks |u| only, so it cannot catch this. Fix in Phase 0 (§3.1).
2. **Two corrections the spec depends on are missing.** Absorption in dB is 2× too large, and the cos α_rw factor is absent (§3.2). The group path is wrong by up to 26% in the regimes that matter (§3.3). Both are fixed in Phase 0. The phase sensitivity that the spec needs for Doppler can be computed exactly from quantities the RHS already has, with no finite differences (§3.4).
3. **Speed.** The current scalar path costs about 5 s per ray. The spec needs about 2×10⁴ rays per fan. Vectorising the RHS is the critical path (§5).

Work splits into two tracks that can run in parallel:

- **Track A, deterministic:** Phase 0 → Phase 1 → Phase 2.
- **Track B, stochastic math:** Phase 3 core, then Phase 4. It is checkable against the spec's synthetic test vectors TV-1…TV-4 with no ray tracing involved.

The tracks merge in Phase 5. §10 lists the decisions I need from you. The most important are approving the X-mode fix (D1) and the baseline commit (D7).

---

## 1. What I checked in the spec

| Item | How it was checked | Result |
|---|---|---|
| Angle-form Appleton–Hartree (4.1) | Compared with Budden's form, both limits (θ = 0, θ = 90°) | Matches |
| Spectrum normalisation N_p (8.1) | Analytic: ∫d³s (1+s²)^(−p/2) = π^(3/2) Γ((p−3)/2)/Γ(p/2) | Matches |
| Screen variance and ℓ∥ (9.3), F(p) | Analytic (2π)^−2 screen integral | Matches |
| Ψ_p series (9.5), including μ = 1 | Re-derived from the K_μ expansion | Matches (signs checked) |
| G_p asymptotes (10.12) | Analytic | Matches; c₄ = π/2 |
| TV-1 per-screen values (Table 11) | Screen 0: σ² from g, Δs, σ_N, ℓ∥ gives 64.4 (table 64.29). Screen 1: W₂₂ = 99.932 (matches). Σσ_i² over screens 1–12 = 13 492.15 (table 13 492.1) | Matches |
| ω_coh and α for TV-1 (10.8, 10.9) | Hand calculation | ω_coh 4.269×10⁵ vs 4.272×10⁵ (0.1%); α 1.266 (matches) |
| Absorption table (§7.1) | Eq. (7.1), secant law, O/X-averaged riometer normalisation | Reproduces 53/35/16/9 dB (O) and 65/41/18/10 dB (X) |
| Eq. (7.1) prefactor 4.61×10⁻⁵ dB m² s⁻¹ | Power-dB convention | Matches |
| Phase sensitivity g → −r_eλ₀/n (4.2) | r_eλ₀ = 8.45×10⁻¹⁴ m² at 10 MHz | Matches |
| Flux-tube distance (4.7) | Flat ground, source at height H: d = H/sin ψ | Matches analytically |
| Fresnel (4.3) and roughness factor | Standard forms | Match |
| Target Doppler (6.1) | Monostatic limit −2 v·k̂/λ | Matches |
| Doppler Method B (4.8) | Equals the Hamiltonian identity in §3.4 | Matches |

**Points the plan has to handle that are not in the spec, or conflict with it or the code:**

- §4.2 item 2 (the branch check) tests |u| only. It cannot detect the X-mode direction error (§3.1).
- The spec says the RHS is "correct and not changed". It is correct for O-mode. It is not correct for X-mode (§3.1).
- §4.5 uses P′ (group path) as the free-space reference. The code uses arc length s. This shifts `focusing_gain_db` by 20 log₁₀(P′/s) (decision D2).
- §2.3 uses e^{+iωt}, with n = μ − iχ and χ > 0. The comment in `absorption.py` (lines 120–126) uses the other convention. The sign of Im n does not change |χ|, but the labels must be made consistent (§3.5).
- §9.1 measures screen distances in group path, so the group index (§3.3) feeds the screen geometry too.

---

## 2. Gap against the spec

| Spec module | What exists | Gap | Phase |
|---|---|---|---|
| `main.py` `Ray` | Haselgrove RHS, scalar only; ε-form; vacuum branch | X-mode direction sign; no local-state accessors; scalar-only | 0, 1a |
| `raytracer.py` | Fixed RK4 with step rejection; adaptive RK45 with events; tangent-map pass; hop loop | No per-step records, no transverse frame, no Fresnel, no landing Jacobian, no Doppler; group index approximate; absorption 2× in dB | 0, 1b–1g |
| `ionosphere.py` | E-CHAIM + IRI-2020 on one geodetic patch around one site; IRI 60 km floor, 5 km IRI steps; one epoch | No D-region below 60 km; no scene-covering patch; no second epoch | 1h, 2 |
| `igrf.py` | Coarse grid, linear interpolation | `B_vec_xyz_m` is scalar-only (`float()` at lines 159–168) | 1a |
| `absorption.py` | Generic collision profile; returns 2k₀ Im n (power) | Wrong dB factor; no cos α; no O/X split in output; no AbsorptionModel | 0, 2 |
| `mc_scattering.py` | Monte Carlo kicks; spectrum with q₀ = 2π/L₀ (line 136) | To be replaced (spec marks it D) | 3, 5 |
| `link_budget.py` | Focusing from tangent map; coherence loss `exp(−σ²/2)` | Replace with flux-tube radar equation; remove coherence term (§6.3 P1) | 1j |
| `api.py` | `Link` (trace, scatter, link_budget, sweeps) | No Scenario / Environment / ChannelSimulator; `scatter()` is MC | 5 |
| (none) | — | `records.py`, `fans.py`, `homing.py`, `legs.py`, `screens.py`, `channel.py`, `gpsd.py`, `arrays.py`, `twoway.py`, `clutter.py`, `noise.py`, `cir.py`, `environment.py` | 1–5 |
| Tests | None. No `tests/`. pytest is not installed in `venv/` (Python 3.12.3) | Everything | 0 |
| Version control | The core files are untracked in the parent repo `/home/matheus/Desktop/School/Raytracing`. No `.gitignore`. | Each fix needs a reviewable diff | 0 (P0.5) |

---

## 3. Defects and corrections (Phase 0)

### 3.1 X-mode rays run backwards (blocking)

**Evidence.** I checked this two independent ways.

- *Physical group velocity from the dispersion relation.* I took ω(k) from the MODE root of the Appleton–Hartree relation (brentq on |u|² − p_MODE(ω), 8 MHz, B = 5×10⁻⁵ T) and differentiated it by finite differences to get ∂ω/∂k. At 12 points (X ∈ {0.3, 0.5}, θ ∈ {30°, 60°, 89°}, B ×1 and ×10⁻³):
  - MODE = +1: cos(ṙ_code, ∂ω/∂k) = +1.0000 at every point.
  - MODE = −1: cos(ṙ_code, ∂ω/∂k) = −1.0000 at every point.
  - The physical group velocity has cos(v_g, k̂) between +0.997 and +1.000 for both modes. This must be so: Faraday's law gives S·k̂ = ½ωμ₀|H|² > 0 for any plane wave with real k, so energy flows forward along the wave normal in a lossless medium.
- *End to end.* A dense synthetic layer (N_e peak 1.2×10¹² m⁻³ at 300 km, σ = 60 km), 57°N 10°E, 28° launch, 8 MHz, nhops = 2, tilted-dipole B, fixed RK4 with 2 km steps:

  | | landed | apex | ground range | arc length |
  |---|---|---|---|---|
  | O-mode (unchanged) | yes | 206 km | 1730 km | 1956 km |
  | X-mode, current code | yes | 200 km | **821 km** | 934 km |
  | X-mode, with the sign fix (§3.1 below) | yes | 204 km | **1679 km** | 1902 km |

  In an earlier single-layer test with a weaker layer, the current X-mode ray never landed: it ran 3001 km of path and ended 35 km from its launch point.

- *Continuity.* The vacuum branch gives ṙ = +u for both modes. For MODE = −1, cos(ṙ, u) is +1 at X ≤ 10⁻⁶ and −1 at X = 1.0000001×10⁻⁶ (and stays −1 up to X = 10⁻⁴). For MODE = +1 it stays +1 throughout. So for X-mode the ray direction jumps by 180° at the vacuum threshold.

**Cause.** `_haselgrove_physical_rhs` returns ṙ = ∂F/∂u (main.py:450). At the X-root, ∂F/∂p = MODE·√D < 0, so the traced ray moves along −v_g. The ray equations are fine. The sign of the root is not.

**Fix.** Multiply the RHS by MODE in the **plasma branch only**. Do not apply it to the vacuum branch (main.py:418), where ṙ = u for both modes.

**Tests to add (Phase 0):**
- T0.1: The FD group-velocity direction test above, at 12 or more random points, in CI.
- T0.2: Direction continuity across X_VACUUM_EPS for both modes.
- T0.3: O and X directions coincide as B → 0.
- T0.4: End-to-end X-mode regression, landing within 5% of O-mode at the synthetic geometry above (qualitative until we have an external reference; see D1).

**Consequences.** All X-mode results change, including the phase path (which also flips sign). `algorithm.md` §8 needs a new entry. I have not edited it, because you changed the file after I wrote it.

### 3.2 Absorption: dB factor and cos α

- `absorption.py:130` returns 2k₀·Im n. That is the **power** coefficient. `raytracer.py:690` converts with 8.686, which is the **amplitude**-nepers-to-dB factor. Power dB = 4.343 × (power nepers) = 8.686 × (amplitude nepers). So every `absorption_db` is 2× too large. The spec's own §7.3 formula, ΔA = 8.686·k₀|Im n|·cos α·Δs dB, agrees with this.
- The spec also has cos α_rw. The amplitude attenuation along the ray is k₀ Im(n) · (û·ŝ), because the eikonal gives Im(∇S)·ŝ. The code omits this factor. It is negligible in the D region, where û ≈ ŝ, but not negligible for oblique paths through strongly magnetised regions.
- **Fix.** Accumulate amplitude nepers A = ∫ k₀|Im n| cos α_rw ds, and report absorption_dB = 8.686·A. Rename the state field: `absorption_nepers_total` is a power exponent today, and after the fix it is an amplitude exponent.
- **Tests:** small-Z limit against the analytic power coefficient Ne·e²ν/(ε₀ m_e ω² c); the §7.1 table; §7.6 item 1 (agreement with eq. 7.1 to 1%).

### 3.3 Group index

- `raytracer.py:178–202` approximates n_g = n + ω ∂n/∂ω by re-solving the dispersion relation at ω ± δω with Y·u held fixed. The group velocity is ∂ω/∂k at fixed k. At fixed k, Y·u must scale as 1/ω too.
- **Exact form.** With G(p, X, Y, Y_d) = a p² + b p + c (the Haselgrove polynomial), the magnitude is

  μ′ = |2p·G_p + Y_d·G_{Y_d} + 2X·G_X + Y·G_Y| / |∇_u F|,  with ∇_u F = 2G_p u + G_{Y_d} v = ṙ (up to the sign in §3.1).

  The closed-form partials are in Appendix A. They agree with finite differences to 10⁻⁹ over 2000 random points. The formula agrees with a finite-difference derivative of the dispersion root at fixed k to 10⁻⁸ at the test points.
- **Against the current code** (8 MHz, B = 5×10⁻⁵ T along z; magnitudes):

  | X | θ | mode | code vs exact |
  |---|---|---|---|
  | 0.30 | 60° | O | −0.4% |
  | 0.95 | 85° | O | −0.3% |
  | 0.90 | 60° | O | −10.4% |
  | 0.50 | 60° | X | +3.3% |
  | 0.70 | 20° | X | +26% |

- **Limitation.** As Y → 0 the ratio is 0/0, because both the numerator and |∇F| vanish. In a test with B × 10⁻⁶ the exact expression returned 3.5×10⁵ instead of 1/n. Use the angle form (4.1) with a finite-difference ∂n/∂X below a threshold |Y| ≈ 10⁻³ (to be tuned).
- **Impact.** P′ feeds the delays (τ_g = P′/c), the caustic cap (§4.5), and the screen distances z_i and w_i (§9.1). Fix this before Phase 1f.

### 3.4 Phase sensitivity without finite differences

- The spec's g = k₀ n_N cos α_rw (4.2) equals −k₀ (∂F/∂N_e)/|∇_uF|. This follows by differentiating F(n û, N_e) = 0 at fixed û, using ∇F·û = |∇F| cos α. Hamilton already has the pieces: L = −∂F/∂X (main.py:440), X_factor, and |ṙ|. So g = k₀ L X_factor / |ṙ|.
- **Verified.** Against a finite difference of n(N_e) at fixed û, the agreement is 5×10⁻⁷ or better at the five test points (O and X). That was before the §3.1 sign change, so the sign must be re-checked afterwards.
- **Use it** for the Doppler integrand (4.8) and the phase variance.
- **Limit test:** as B → 0, g → −r_eλ₀/n for both modes.

### 3.5 Sign and naming conventions

- Adopt the spec's convention (§2.3): e^{+iωt}, n = μ − iχ with χ > 0. Document it in `absorption.py` and in `algorithm.md` §2.8, which currently says Im n > 0 is the "e^{−i(kz−ωt)}" branch.
- Rename the absorption fields (see §3.2).

### 3.6 Other removals

- `link_budget.py:134`: remove the coherence loss (spec §6.3, pitfall P1).
- `mc_scattering.py:136`: q₀ = 2π/L₀ treats L₀ as a wavelength. The spec's (8.1) treats it as a correlation length. Migrate the values when the spectrum is replaced (Phase 3a).
- `ionosphere.py:88`: the 60 km IRI floor and 5 km IRI steps. See Phase 1h.

### 3.7 Repository baseline (P0.5)

- Commit the current state in the parent repo before any change, so each fix is a reviewable diff.
- `.gitignore`: `venv/`, `__pycache__/`, `*.o`. Decide about `ECHAIM/etc/` (about 115 MB) and the built `libechaim.so` (D7).
- Add pytest to `venv/`. Create `tests/`, with `python -m pytest tests` as the single command.

---

## 4. Target architecture

Module map: spec Table 3, with notes.

| Module | Status | Responsibility | Notes |
|---|---|---|---|
| `main.py` (`Ray`) | M | RHS (vectorised), ε-form, vacuum branch, angle-form helper, local-state accessor | RHS physics unchanged apart from §3.1. Keep scalar wrappers for the tests. |
| `raytracer.py` | M | Drivers, `RayRecorder` (per step), transverse frame, bounce records, landing Jacobian, Doppler integrand | Batch mode (N, 6). Recording must not change the trajectory. |
| `ionosphere.py` | M | Scene-covering patch, D-region merge, 50 km floor, two epochs | See §3.6 and Phase 1h. |
| `igrf.py` | M | Vectorised `B_vec_xyz_m`, unit field, magnetic east, AACGM accessors | Accessors for drift/irregularity models. |
| `absorption.py` | M | AbsorptionModel protocol, Sen–Wyller table, amplitude nepers, riometer fallback | §7. |
| `link_budget.py` | M → replaced | Flux-tube radar equation (6.2–6.4) | Coherence term removed. |
| `mc_scattering.py` | D | Kept as historical reference until Phase 5, then deleted | §10 D6. |
| `environment.py` | N | Protocols: AbsorptionModel, IrregularityModel, DriftModel, SurfaceModel; output validation | Listings 2 and 3. |
| `records.py` | N | Dataclasses from Listing 1 | |
| `fans.py` | N | Launch fans, landing maps per hop, quad spatial index | §5.1. |
| `homing.py` | N | Quad inversion, Newton refinement, classification | §5.2–5.4. |
| `legs.py` | N | LegMode assembly, deterministic parameters, power pieces | §6. |
| `screens.py` | N | Screens from ray samples (per sample, optional coarsening) | §9. |
| `channel.py` | N | ChannelParams (Algorithms 4 and 5) | §10. |
| `gpsd.py` | N | GPSD, bin integration, realisations | §11. |
| `arrays.py` | N | Element geometry and offsets | §11.5. |
| `twoway.py` | N | Cascade, reciprocity diagnostic | §12. |
| `clutter.py` | N | Sea/ice/land clutter, Bragg lines, field generation, auroral clutter | §12.4–12.5. |
| `noise.py` | N | External noise | §12.6. |
| `cir.py` | N | h_pq(t, τ) assembly, HDF5 I/O | §12.7, §14.2. |
| `api.py` | M | Keep `Link`; add Scenario, Environment, ChannelSimulator | §14.1, §7 below. |
| `visualization.py` | M | Scattering functions, range–Doppler maps, screens along rays | |
| `tests/` | N | Analytic, golden, equivalence, statistical | §8 below. |

**Data flow** (spec Figure 1, with the caching boundary):

```
Environment (E-CHAIM+IRI, IGRF, absorption, irregularity, drift, surface)   [cached per epoch/carrier]
  → Fans (per site, per mode, per hop) → landing maps                       [cached]
  → Homing (targets, clutter cells) → LegMode records                       [cached]
  → Legs: samples → screens → ChannelParams                                  [cached]
  → Components (out-mode, scatterer, ret-mode)                               [per realisation]
  → GPSD realisations → h_pq(t, τ)                                            [per realisation]
```

**Compatibility.**

- `Link` stays as the single-site deterministic façade (trace, link_budget, sweeps).
- `scatter()` and `ScatterResult` are deprecated in Phase 5 and removed with `mc_scattering.py`.
- `LinkBudget` loses `coherence_loss_db`. `total_db` becomes focusing minus absorption (6.3).
- `RayPath` gains fields from Listing 1. These are additive.

---

## 5. Performance (gate for Phase 1a)

**Measured** (6×6 patch, real E-CHAIM + IRI, 8 MHz):

- Ne, one scalar call: **110 µs**.
- Ne, array of 4000 points: **1.6 ms** (0.41 µs per point, about 270× cheaper per point).
- One scalar RHS evaluation with real Ne and dipole B: **0.85 ms**. About 90% of that is the seven Ne calls (value plus central-difference gradient).
- A current scalar trace: about 5 s (from earlier sessions; roughly 2400 RHS calls).

**Estimate.** A vectorised batch should cost roughly 10–40 ms per ray-trace, depending on how the IGRF and absorption calls scale. So a 2×10⁴-ray fan takes roughly 3–13 minutes. That is at the edge of the spec's "minutes, cached" budget (Table 8). These are estimates. The Phase 1a benchmark replaces them with measurements.

**Gate.** A 4000-ray, 2000-km, single-hop batch in ≤ 2 minutes. If it misses the gate, the next options are, in order:

1. Analytic gradients on the interpolant (saves six of the seven Ne calls).
2. An ECEF Cartesian grid, so there is no Bowring call per query.
3. numba (a new dependency).
4. C (the toolchain already exists for E-CHAIM).

**What must become array-valued:** `Ray` (RHS, `grad_Ne`, `_grad_Y_vec`, `_group_index`, `absorption_rate`), `IgrfField.B_vec_xyz_m` (scalar-only today), the vacuum guard and the ε-form (`np.where`, not `if`), and the recording path. `Ionosphere.electron_density_xyz_m` already accepts arrays (checked above).

---

## 6. Phases

Sizes are rough: **S** is about one focused session, **M** a few, **L** many. There are no tests yet, so these are guesses.

### Phase 0. Blocking fixes and baseline (S)

Tasks: P0.1 X-mode sign (§3.1). P0.2 absorption (§3.2). P0.3 exact group index with angle-form fallback (§3.3). P0.4 phase sensitivity from internals (§3.4). P0.5 baseline commit, `.gitignore`, pytest, `tests/` (§3.7). P0.6 update `algorithm.md` §2.4, §2.8 and §8 once the fixes land.

Tests: T0.1 FD group-velocity direction (12 points). T0.2 direction continuity at the vacuum threshold. T0.3 O ≡ X as B → 0. T0.4 group index against the FD of the dispersion root (5 points, 10⁻⁶). T0.5 absorption small-Z limit (analytic). T0.6 §7.1 table reproduced for O and X. T0.7 phase sensitivity against FD (10⁻⁶) and the B → 0 limit. T0.8 smoke test: O and X traces run, and `example.py` runs.

**Exit:** all T0 tests pass; decision D1 is recorded.

### Phase 1. Deterministic layer (L; critical path)

- **1a. Vectorised RHS and batch tracer** (§4.8). Equivalence with the scalar path to 10⁻¹⁰. Benchmark against the gate in §5. **Do this first.**
- **1b. Per-step records** (§4.1, Listing 1; `records.py`). Fixed-step: every accepted step. Adaptive: dense output at Δs ≤ 2 km, and ≤ 1 km below 130 km. Test: recording does not change the trajectory (bitwise).
- **1c. Local helpers** (§4.2). Angle-form n (Budden), a branch check that covers both |u| and the direction (new), dn/dN_e by identity (§3.4), cos α from the physical direction, the n_min guard and the steep flag.
- **1d. Transverse frame** (§4.3, Algorithm 1). Planar-ray test (t̂₁ stays normal to the plane); orthonormality to 10⁻¹²; mirror at each bounce.
- **1e. Ground and surface** (§4.4). SurfaceModel protocol. Sea/land from `global-land-mask` (already a dependency). Ice and ice-sheet values come from user data. Fresnel (4.3), optional roughness, Bounce records, `ground_loss_dB`. Tests: R_h = −R_v at normal incidence; |R| → 1 at grazing; a minimum in |R_v| near the pseudo-Brewster angle.
- **1f. Landing Jacobian, flux-tube distance, caustic cap** (§4.5). Ground-projected Jacobian (4.6); d² = A sin ψ/cos β (4.7); caustic cap with P′. Test: free-space d = H/sin ψ to 10⁻⁴. Decide D2 first.
- **1g. Doppler, Method B** (§4.6). Accumulate g·∂_tN_e ds along each ray, with ∂_tN_e = [N_e(t+Δt) − N_e(t−Δt)]/(2Δt) − v_d·∇N_e. Needs two epochs (1h). Test: uniform-decay test against Method A to 5%.
- **1h. Patch and epochs** (§4.7, `ionosphere.py`). Scene-covering patch (bounding box of both sites and the scene, plus margin). Altitude floor 50 km with ≤ 1 km spacing below 130 km; this needs decision D4. Two epochs on identical grids. Tests: the grid covers both sites; E-CHAIM's FIRI D-region gives non-zero, smoothly varying N_e in the 50–60 km band.
- **1i. Fans, landing maps, homing** (§5; `fans.py`, `homing.py`). Algorithms 2 and 3, classification (§5.4), reciprocity (§6 box). Tests: homing converges to < 50 m in ≤ 10 iterations on a test fan; reciprocity to 10⁻⁶ for O and X at one and two hops; subdivision triggers near a known skip caustic.
- **1j. Legs and power budget** (§6; `legs.py`, `link_budget.py`). LegMode assembly; delay, phase and Doppler; eq. (6.1); eq. (6.2)–(6.4); L_pol = 3.01 dB; no coherence term.

**Exit:** the mode table for the reference scenario (`example.py`) agrees between the fan route and the homing route. Reciprocity holds. Power-budget unit tests pass.

### Phase 2. Absorption (M)

- 2a. AbsorptionModel protocol (Listing 2) with `density_kind` and `nu_kind`.
- 2b. Merge with E-CHAIM: enhancement or total, with a cosine blend (§7.2). Needs D4.
- 2c. Sen–Wyller table for eq. (7.2): tabulate ⟨·⟩_u against ν_m/ω′.
- 2d. Per-step ΔA for each mode (§7.3), per-hop partial sums, `absorption_D_dB`.
- 2e. Riometer fallback (§7.4).

Tests: §7.6 items 1–5 and the §7.1 table.

### Phase 3. Irregularity, drift, screens, channel parameters (L; Track B, can start after Phase 0)

- 3a. Spectrum (8.1), with L₀ as a correlation length. Normalisation test for p ∈ {3.6, 4, 4.5, 5}. Checks: clip p to (3.05, 5.95), a, b ≥ 1, Gram–Schmidt on e_sh.
- 3b. Screens (9.1–9.3). W_i, σ_i², ℓ∥, D_i through a tabulated Ψ_p(ln u) with the small-u series, and the secant matrices A_i.
- 3c. Channel parameters (10.1–10.7; Algorithms 4 and 5). D⋆, the correlation ellipse, the τ_o root, V, C_xt and C_yt, ⟨τ⟩, ω_coh, α. S4 baseline (10.10) and refined (10.11) with tabulated G_p. The Rician split.
- 3d. Test vectors TV-1…TV-4 (Appendix B), the first hard gate. Tolerances from Table 12. Store the vectors as JSON test data, generated from the spec's tables and not from our code.
- 3e. MPS reference (§15.3): a 1-D split-step script, run on demand and not in CI. Use it to decide the distance metric (D5).

**Exit:** TV-1…TV-4 within tolerance; the closed-form tests of Table 9 for this part.

Phase 3 needs no ray tracing, because the test vectors are synthetic. It can run in parallel with Phase 1.

### Phase 4. One-way channel (M–L; Track B)

- 4a. GPSD (11.1) and the bin-integrated form (11.3).
- 4b. Realisations: single-element (11.4b) first, then the array method (11.4a), then sum-of-plane-waves (11.4c).
- 4c. Element mapping (11.5) with fractional delays (`arrays.py`).
- 4d. Rician assembly (11.6).
- 4e. Self-tests (11.7), including the sign test.

**Exit:** the §11.7 self-tests pass. GPSD normalisation is (2π)³. Figures 4 and 5 are reproduced qualitatively.

### Phase 5. Two-way radar (L)

- 5a. Cascade and convolution (12.1, `twoway.py`).
- 5b. Reciprocity diagnostic Γ_or (12.2).
- 5c. Targets (12.3): Doppler from (6.1), range migration.
- 5d. Clutter (12.4): Bragg lines, open-water fraction for sea ice, ice sheet and land, field-based generation (12.4.1).
- 5e. Noise (12.6). ITU-R P.372 F_a is user data.
- 5f. Assembly (12.7, `cir.py`) and HDF5 output (14.2).
- 5g. API: Scenario, Environment, ChannelSimulator (14.1).

**Exit:** range–Doppler maps show the Bragg lines. Power-sum checks pass.

### Phase 6. High-latitude extensions (M)

- 6a. Direct auroral backscatter (12.5).
- 6b. CIT-limited wander option (10.9).
- 6c. Partial reciprocity (12.2), flagged.
- 6d. Tangent-map screen geometry (9.1 refinement). Optional.
- 6e. numba decision and tuning, if the §5 gate needs it.

### Phase 7. Validation and calibration (open; needs data)

Doppler and range data from SuperDARN, riometers, GNSS σ_φ and S4 (for ξ calibration, §8.4), and OTHR data if available. No code plan until the data are in hand.

---

## 7. Public API (§14.1)

- Keep `Link` for single-site deterministic work. It delegates to the new modules.
- Add `Scenario`, `Environment`, `ChannelSimulator`, `SimOptions` and `FanSpec` (Listing 4) in Phase 5.
- Units stay as they are at the API boundary: km, MHz, degrees, dB, ms. Everything inside is SI. ECEF positions stay in metres.
- `scatter()` and `ScatterResult` get a deprecation warning in Phase 5, and are removed with `mc_scattering.py`.

---

## 8. Verification

- **Layout:** `tests/analytic/` (Table 9), `tests/golden/` (TV-1…TV-4 and the O/X regressions), `tests/equivalence/` (vector against scalar), `tests/stat/` (§11.7, with fixed seeds and a bounded runtime).
- **Exit criteria are tests, not prose.** Each phase's exit is a list of test IDs.
- **Runtime:** the CI suite should finish in under 5 minutes. Heavy tests (MPS, full fans) are marked and run on demand.
- **Environment:** tests use `n_ang ≤ 10`. Earlier sessions saw hangs at `n_ang = 15` under memory pressure.
- **Golden values** are taken from the spec's tables, not from our code, so they test the code against the spec.
- **Spec drift:** record every deviation from the spec in `docs/spec_deviations.md` (D10).

---

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| X-mode fix changes every X-mode result | Old X-mode numbers are wrong | Say so in the changelog. Re-run any saved X-mode outputs. |
| No external X-mode reference yet | The fix is validated internally only | D1: obtain a reference (PHaRLAP output or a published case) before publishing X results. |
| Performance gate missed | Fans take hours | §5 ladder: analytic gradients, ECEF grid, numba, C. |
| Fold and caustic handling in quad inversion | Missed or duplicated modes | Adaptive subdivision (Algorithm 2), tests on a known skip. |
| D-region validity (50–70 km) | Absorption numbers uncertain | E-CHAIM FIRI (v3.4.0 has it) with a documented floor. Treat absolute dB as provisional. |
| Stochastic layer may disagree with MPS by 10–20% | Acceptable per the spec, but must be shown | Phase 3e, on-demand script. |
| Default irregularity, drift, surface and absorption models are placeholders | Absolute numbers are provisional | D9: label every default as a placeholder. |
| Memory for N_x × N_y × N_t per delay bin | Out of memory in big arrays | Single-element method is the default (11.4b). |
| Environment instability (earlier hangs under memory pressure) | Flaky runs | `n_ang ≤ 10` in tests; monitor memory. |
| The spec's wrong statements carried into code | Errors that look like physics | §10 D10: spec v1.1 and a deviations log. |

---

## 10. Decisions

Recommendations are in bold. I have not acted on any of these.

- **D1. X-mode fix.** Multiply the plasma-branch RHS by MODE (§3.1). **Approve.** Validate against an external reference before publishing X-mode results.
- **D2. Focusing reference.** P′ (the spec) or arc length s (the code). **P′, as the spec says.** Document the change. It shifts `focusing_gain_db` by 20 log₁₀(P′/s). I have not measured that shift yet.
- **D3. Vectorisation route.** **numpy first** (no new dependency). numba only if the §5 gate fails.
- **D4. D-region source and merge convention.** **E-CHAIM's FIRI D-region** (`dregion=True`, which v3.4.0 provides) for the climatological D layer. Auroral and polar-cap absorption as an enhancement on top, per the spec's §7.2 convention, so nothing is counted twice. Keep the IRI 60 km floor for everything above it.
- **D5. Distance metric.** **Group path**, as the spec's baseline. Compare with arc length using the MPS script (3e).
- **D6. Monte Carlo layer.** **Deprecate in Phase 5, delete after 5g.**
- **D7. Baseline commit and `.gitignore`.** **Yes, commit before Phase 0 code changes.** Keep `ECHAIM/etc/` in the repo, because offline runs need it. Document the fetch step if you would rather not.
- **D8. pytest.** **Install it into `venv/`.**
- **D9. Default models.** **Ship the TV-1 parameter set as the reference default, and label every other default as a placeholder.**
- **D10. Spec deviations.** **Keep `docs/spec_deviations.md`.** Ask the spec's author for v1.1, covering §4.2 item 2 and the statement that the RHS is correct.

---

## 11. Next steps

1. You decide D1, D7 and D8. D2 if you have a view.
2. On approval, I do P0.5 (baseline), then P0.1–P0.4, each with its tests, and report the results.
3. Then Phase 1a (vectorisation) and Phase 3a–3d in parallel.

---

## Appendix A. Closed-form partials for the group index

With a = 1 − X − Y², b = −2X² + XY²Y_d² − XY² + 4X + 2Y² − 2, c = −X³ + 3X² − XY²Y_d² + XY² − 3X − Y² + 1, and G = a p² + b p + c:

- G_p = 2a p + b
- G_X = −p² + (−4X + Y²Y_d² − Y² + 4) p + (−3X² + 6X − Y²Y_d² + Y² − 3)
- G_Y = −2Y p² + (2XYY_d² − 2XY + 4Y) p + (−2XYY_d² + 2XY − 2Y)
- G_{Y_d} = 2XY²Y_d (p − 1)
- ∇_u G = 2G_p u + G_{Y_d} v, with v = Y/|Y| and Y_d = v·u

The first three are verified against finite differences in `scratchpad/partials_check.py` (10⁻⁹ relative over 2000 random points).

## Appendix B. Scratch checks behind this plan

These live in the session scratchpad, outside the repo. Phase 0 turns them into tests.

- `spec_checks.py`: phase-sensitivity identity; exact versus code group index; isotropic limit; interpolator timings.
- `xmode_check.py`, `xmode_patch_test.py`: finite-difference group velocity from ω(k); direction sign for O and X; the sign fix as a monkeypatch; end-to-end landing for O and X.
- `e2e_xmode.py`: the single-layer X-mode trace.
- `vac_continuity.py`: ray direction either side of the vacuum threshold, both modes.
- `partials_check.py`: closed-form partials against finite differences.

Measured inputs for the spec's test vectors: Table 11 values, and the values I reproduced by hand from its formulas (§1).
