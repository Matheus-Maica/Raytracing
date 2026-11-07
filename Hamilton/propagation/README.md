# propagation — deterministic legs

A *leg* is one-way propagation from a site (Tx or Rx) to a point on the ground or at altitude, for one mode (hop
count, reflecting layer, low/high ray, O/X). This folder finds the legs and computes their deterministic
parameters: launch angles, group and phase path, landing geometry, focusing and the one-way power budget. Built on
`raytracing.batch.BatchTracer`; depends on `core/` and `raytracing/`.

| File | What it does |
|---|---|
| `legs.py` | Leg parameters: launch direction, flux-tube distance and focusing, E/F layer letter, `LegMode` assembly. |
| `fans.py` | Launch fans and landing maps; Algorithm 2 (which launch angles land on a given point); mode labels. |
| `homing.py` | Algorithm 3: Newton homing of one mode onto a target point; the reciprocity test. |
| `link_budget.py` | One-way power budget of a leg (spec §6.2-6.4). |

Typical flow (used by `echoes.targets`): `fans.trace_fan` -> `fans.invert_fan` (candidates per hop) ->
`homing.newton_home` (refine each to 50 m) -> `legs.landing_mini_fan` + `legs.leg_mode` (the `LegMode` record).

## legs.py

- `launch_direction(beta, alpha)`: unit launch vector (elevation, azimuth clockwise from north).
- `landing_mini_fan(...)`: traces the central ray and four neighbours (elevation/azimuth +/- delta) to get the
  landing Jacobian; returns a `LandingResult`.
- `flux_tube_distance2` (eq. 4.7, d^2 = A sin(psi)/cos(beta)) and `focusing` (G_f = 10 log10(P'^2/d^2), capped at
  `G_FOCUS_MAX_DB` at caustics).
- `valley_altitude_m` / `layer_letter`: E or F reflection from the apex height relative to the E-F valley.
- `one_way_power_dB` (eqs. 6.2-6.3) and `leg_mode(...)`, which packs everything into a `core.records.LegMode`
  (polarisation loss `L_POL_DB` = 3.01 dB per leg).

## fans.py

- `trace_fan(...)`: a regular (elevation, azimuth) grid traced in one batch -> `FanResult` (per ray, per hop:
  landing point, P, P', absorption, Doppler, apex, validity).
- `site_planar`: azimuthal-equidistant (east, north) coordinates about a site, the plane the maps live in.
- `invert_fan(...)`: Algorithm 2. Every quadrilateral cell of the landing map is inverted bilinearly; cells crossed
  by a caustic are subdivided (up to `SUB_DEPTH_MAX` levels) and flagged. Returns `Candidate`s.
- `classify(...)`: spec §5.4 labels (hop count, layer per hop, low/high ray, O/X) and de-duplication.

## homing.py

- `newton_home(...)`: solves for the launch direction whose ray lands on the target; residual is the miss in the
  target's tangent plane, Jacobian by forward differences traced in the same batch. A converged ray gets a
  first-order correction of P and P' for the remaining miss. Returns a `HomedRay` (or rejects the mode).
- `reciprocity(...)`: spec §5.5 check: trace back from the landing point and compare.
- `tangent_basis(r)`: east/north unit vectors at a ground point.

## link_budget.py

`one_way_budget(...)` -> `LinkBudgetResult`: P_r = P_t G_t G_r lambda0^2 / ((4 pi)^2 d_eff^2 L) with
10 log10 L = absorption + ground losses + polarisation. The focusing gain is reported but already inside d_eff; there
is no coherence term (the coherent/diffuse split belongs to the Rician model in `stochastic/`). Used by
`api.Link.link_budget`.
