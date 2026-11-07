# core — shared foundations

Small modules with no dependencies inside the project (numpy only). Everything else imports them; they import
nothing from the other folders.

| File | What it does |
|---|---|
| `constants.py` | Physical constants: `C`, `mu_0`, `epsilon_0`, `m_e`, `q_e`, and `B_0` (a nominal 5e-5 T field strength). |
| `geometry.py` | WGS84 coordinate transforms and ellipsoid geometry. |
| `records.py` | The data records passed between pipeline stages (spec Listing 1). |
| `inputs.py` | The user inputs of the simulator, as dataclasses with placeholder defaults. |

## geometry.py

All positions are geocentric Cartesian (ECEF) metres; latitude is **geodetic**. Constants `WGS84_A`, `WGS84_F`,
`WGS84_B`, `WGS84_E2`, `WGS84_EP2`, `R_MEAN`.

- `latlon_to_xyz_m(lat, lon, h)` / `xyz_m_to_latlon(x, y, z)`: geodetic <-> ECEF (Bowring's 1976 method for the
  inverse).
- `ellipsoid_outward_normal_m(r)`: unit normal of the ellipsoid (used for specular ground reflection).
- `_ray_ellipsoid_intersect_m(r0, u)`: first positive intersection of a straight ray with the ellipsoid.
- `great_circle_range_m`, `down_cross_range_m`: surface ranges between sub-points.

It was pulled out of `raytracing/main.py` so the tracer and absorption code can use it without a circular import.

## records.py

Plain dataclasses, SI units throughout (m, s, rad, Hz, m^-3, T):

- `RaySamples`: one traced leg as a struct of arrays: position `r`, arc length `s`, phase and group path `P`/`Pg`,
  ray and wave-normal directions, refractive index, X, B, Ne and its gradient, altitude, hop index, absorption, the
  transported transverse frame (`t1`, `t2`) and flags. Built by `raytracing.batch.samples_from_record`.
- `Bounce`: one specular ground reflection (point, normal, grazing angle, surface type, Fresnel R_h/R_v, loss).
- `ChannelParams`: the stochastic parameters of a leg at its array end (output of `stochastic.channel`, returned by
  `api.Link.channel`); `rician_K_db` and a readable `str()`.
- `LegMode`: one mode of one leg from a site to a scatterer: launch angles, phase and group path, delay, Doppler,
  absorption and ground losses, the caustic-capped distance `d_eff`, its samples, bounces and `ChannelParams`.
- `Component`: one contribution to the channel h_pq(t, tau) from an outbound and a return `LegMode`: power, delay,
  Doppler, phase and the reciprocity factor `gamma_or`; `kind` says target, clutter, aurora or meteor.

## inputs.py

Every quantity the user must supply, grouped by spec section:

- `IrregularityInputs`: xi, L0 = 30 km, p, axial ratios, drift velocity and its spread.
- `BraggInputs`: auroral Bragg-scale irregularities.
- `SurfaceInputs`: wind direction and speed, B_s, spreading exponent, sea-ice concentration, ice and land sigma0,
  surface current.
- `NoiseInputs`, `TargetInputs`, `RadarInputs`.

Defaults are **placeholders** (decision D9): they let the code run, they do not make the numbers right.
`placeholders()` lists every input with its current value.

The objects are passed explicitly to the functions that need them (`api.Link`, `echoes.targets`,
`echoes.clutter_scene`, `echoes.auroral_scene`, `receiver.noise`, `stochastic.leg_screens`); nothing reads them from
module state. `inputs.IrregularityInputs` is the single source of the irregularity defaults.
