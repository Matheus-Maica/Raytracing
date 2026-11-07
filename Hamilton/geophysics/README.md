# geophysics — models of the medium

What the world looks like to the radar: the background ionosphere, the magnetic field, the small-scale
irregularities and their drift, and the sea surface wind. Depends only on `core/`. The models that are crude
stand-ins (WBMOD, Richmond-NCAR) are labelled as such and plug in through the interfaces of `environment.py`, so
they can be swapped without touching the callers.

| File | What it does |
|---|---|
| `environment.py` | Model interfaces (Protocols), the checks the spec applies to model outputs, default providers, surface constants. |
| `ionosphere.py` | Background electron density: E-CHAIM + IRI-2020 on a 3-D WGS84 grid patch, interpolated. |
| `ECHAIM/` | Vendored E-CHAIM C library (CHAIN project, GPL-3.0) and its `ctypes` binding. |
| `igrf.py` | IGRF magnetic field on a coarse grid over the same patch, interpolated. |
| `wbmod.py` | Irregularity climatology: HiCIRF eq. (5) and `CrudeWBMOD`, our own stand-in for WBMOD. |
| `plasma_drift.py` | WBMOD-style plasma drift: Heelis convection (full port) + a crude mid/low-latitude pattern. |
| `ccmp_wind.py` | CCMP v3.1 ocean surface wind: download, cache, nearest-neighbour lookup. |

## environment.py

- Protocols: `AbsorptionModel` (density, collision frequency, riometer), `IrregularityModel` and `DriftModel`
  (`.at(r_ecef, t) -> dict`), `SurfaceModel`.
- Output checks: `check_irregularity` (clips p to (3.05, 5.95), enforces axial ratios >= 1, re-orthogonalises the
  sheet axis against B, rejects negative xi), `check_drift` (removes the component of the drift along B, spec §8.3).
- Default providers: `ConstantIrregularity`, `ConstantDrift`, `SeaLandSurface` (sea/land from the land mask, constants
  from `SURFACE_CONSTANTS`).
- `relative_permittivity_complex(eps_r, sigma, f)`: eps_r - i sigma/(omega eps0), Hamilton's e^{+i omega t} convention.

## ionosphere.py

`Ionosphere` builds the density on a uniform geodetic grid covering only a patch (centre, angular extents, ceiling
`h_max`), 1 km spacing below 130 km. E-CHAIM is used where it is valid (AACGM latitude >= 45 deg), IRI-2020 (via
`iricore`) elsewhere, blended per point; E-CHAIM's FIRI D region is optional. The expensive model calls happen once
at construction; lookups (`electron_density_xyz_m`, `electron_density_geodetic`) are trilinear interpolation.
`covering_patch()` sizes a patch around a set of sites; `EpochPair` holds two snapshots for dNe/dt (the Doppler
accumulator). Running `python -m geophysics.ionosphere` builds a patch and plots it.

## ECHAIM/

Vendored package, imported as `from geophysics import ECHAIM`. `echaim.py` loads `libechaim.so` (building it with
`build.sh` if missing) and points the library at `etc/` (coefficient and index databases, ~115 MB). Paths are relative
to the package, so the folder can be moved as a whole. One patch to the vendored C: `echaim_init(dir)` replaces the
library's interactive directory lookup. Licences in `ECHAIM/licenses/`.

## igrf.py

`IgrfField` evaluates `ppigrf` once on a coarse (lat, lon, h) grid and interpolates afterwards, because the ray
equations need the field (and its gradient) millions of times. `build_igrf_for_ionosphere(iono)` builds it over the
same patch as an `Ionosphere`.

## wbmod.py

WBMOD itself is export-controlled and not available, so:
- `xi_squared_eq5(q, CkL, ...)` / `xi_squared_eq5_from_column`: HiCIRF eq. (5), the fractional density variance from
  WBMOD's q and CkL with the fixed outer scale `WBMOD_LO_M` = 30 km.
- `p_from_q(q)`: Hamilton's 3-D spectral index from WBMOD's 1-D one, p = q + 2 (our own rule).
- `CrudeWBMOD`: an `IrregularityModel` that only has to return numbers (user decision): log10 CkL from a few bumps in
  magnetic latitude and MLT (auroral oval, polar cap, equatorial post-sunset), q = 2, xi from eq. (5) over the Ne^2
  column of the caller's own ionosphere (`ne_column=link.electron_density_xyz_m`).
- `PlaceholderWBMODIrregularity`, `PlaceholderWBMODDrift`: the earlier, simpler stand-ins (kept).

## plasma_drift.py

- Geomagnetic frame: IGRF centred dipole for the date (`dipole_axis_ecef`, `geomagnetic_lat_lon_rad`,
  `magnetic_local_time_hours`, `subsolar_point_deg`, `centred_dipole_B`).
- `heelis_potential_kV` + `HeelisParams`: line-by-line port of `flwv32()`/`potm()` from NCAR TIE-GCM's `heelis.F90`
  (`docs/heelis.F90.txt`). `HeelisDrift` turns it into E = -grad(Phi) and v = E x B / B^2; its input is the
  cross-polar-cap potential `ctpoten_kV`.
- `RichmondDrift`: our own crude stand-in for Richmond-NCAR (not available anywhere): eastward at night, westward by
  day, tapered off towards high latitude.
- `WBMODStyleDrift`: the two superposed (our own rule). All three satisfy `environment.DriftModel`.

## ccmp_wind.py

- `CCMPWind`: wind on CCMP's (time, lat, lon) grid; `.at(lat, lon, t)` returns the direction the wind blows towards
  (compass degrees) and the speed, nearest neighbour in space and time, NaN over sea ice.
- `ccmp_wind_for(t)`: downloads the day's file from `https://data.remss.com/ccmp/v03.1/` (no account needed) into
  `~/.cache/hamilton/ccmp` (override with `HAMILTON_CACHE`) and loads it (`netCDF4`).
- Used by `echoes.clutter_scene.scene_components(wind_model=...)`.

## Using a model

```python
from geophysics.wbmod import CrudeWBMOD
from geophysics.plasma_drift import WBMODStyleDrift
link = Link(..., irregularity_model=CrudeWBMOD(), drift_model=WBMODStyleDrift())
```
Any object with the right `.at(r_ecef, t)` method works; see `environment.py` for the keys each must return.
