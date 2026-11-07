# What this needs from you

Standing workflow (your instruction, 2026-10-31): whenever something the HiCIRF plan needs is outside what I can
fetch or verify myself, it goes here (what it is, why, what stands in for it, where to get the real thing), and
the code keeps a clearly labelled placeholder instead of blocking. Plan: `.claude/plans/valiant-questing-robin.md`.

## Open

### A. Optional upgrade path for Richmond-NCAR (no action needed)

The real model is not online (see item 3 below), but the original paper -- Richmond et al. (1980), J. Geophys.
Res. 85, 4658, doi:10.1029/JA085iA09p04658 -- is on the same Wiley/AGU platform your Fentzke (2008) PDF came from
(UNB access). If its coefficient table is in the paper, that would replace our crude pattern. Unverified whether
the table is printed there.

## Resolved (2026-11-07)

1. **WBMOD** -- not obtainable. Replaced by our own deliberately crude `wbmod.CrudeWBMOD` (your decision): log10
   CkL from a few bumps in magnetic latitude/MLT, q = 2, xi from eq. (5) over the Link's own Ne^2 column
   (`ne_column=link.electron_density_xyz_m`), L0 = 30 km. Opt-in via `api.Link(irregularity_model=...)`.
2. **CCMP wind** -- RSS serves the daily v3.1 files without an account (https://data.remss.com/ccmp/v03.1/,
   `CCMP_Wind_Analysis_YYYYMMDD_V03.1_L4.nc`, ~33 MB, four 6-hourly analyses). `ccmp_wind.ccmp_wind_for(t)` downloads
   the day into `~/.cache/hamilton/ccmp` (override: `HAMILTON_CACHE`) and reads it (netCDF4, installed). Checked
   against a real file (2026-09-15). Also fixed: the clutter scene was passing CCMP's absolute compass direction
   where `clutter.bragg_lines` expects a direction relative to the radar look; it is now converted per cell.
3. **Richmond-NCAR** -- searched: NSSDC's 1990 catalogue (NSSDC/WDC-A-R&S 90-19) lists it as "ISR Ion Drift Model
   1980", NSSDC ID MI-93B, distributed "on tape, SPAN" only; SPDF's surviving model archive has no ionospheric
   section; CCMC does not list it. Our own crude `plasma_drift.RichmondDrift` stays (your decision). See A above.
4. **Second-order Bragg** -- your Barrick (1972) chapter is sufficient: section 12.3, eqs. (12:31)-(12:34), both
   coupling coefficients. Implemented in `second_order_bragg.py`; validated against his Figs. 12.18-12.19 (0.04-1.5
   dB over 15-40 kt, upwind and crosswind, 10 MHz), with his -17 dB first order and the sqrt(2) and 2^(3/4)
   singularities where he puts them. Opt-in: `bragg_lines(second_order=True)`, `scene_components(second_order=True)`.
5. **Heelis MLT arrays** -- your `docs/heelis.F90.txt`: `plasma_drift.HeelisDrift` is now a full port of flwv32().
   The arrays reduce, for TIE-GCM's defaults, to the single harmonic already used; but the polar-cap cubic was wrong
   before (zero flow over the pole) and is now right (uniform anti-sunward flow). Its one input, the cross-polar-cap
   potential `ctpoten_kV` (default 30 kV), stays a user input -- WBMOD derives it from Kp internally, not public.
6. **Meteor source weights** -- Fentzke and Janches (2008) Table 3: Apex 33%, Helion 22%, Anti-Helion 22%, North and
   South Toroidal 11.5% each (1 AU input).
7. **Meteoroid ablation/ionisation model** -- dropped (your decision). q_e stays log-uniform, r0 fixed, by design.
8. **D_a(h)** -- your profile, D_a = 0.025 exp(0.154 h/km) cm^2/s.
9. **Drift blend** -- our own crude rule (your decision): Heelis + Richmond superposed; each handles its own domain.
10. **q -> p** -- our own rule (your decision): p = q + 2 (1-D in-situ vs 3-D index of a power-law spectrum).
11. **Sea-wave spectrum normalisation** -- Barrick's (your decision, "HiCIRF-exact"): `clutter.wave_saturation_S` is
    now B_s k^-4 G with int G dphi = 1 (was the spec's (B_s / 2 pi) k^-4 G). First-order sea lines are 8 dB stronger
    (dominant line at s = 3: -15 dB; a semi-isotropic sea gives Barrick's -17 dB), second order 16 dB stronger,
    so 8 dB closer to first order (about -7.5 dB below it at 15 m/s, s = 3). Deviation 46.
