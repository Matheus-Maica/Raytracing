"""
inputs.py -- the user inputs of the simulator (instruction of the next run: these stay user inputs).

Every quantity here comes from data or a model that the user supplies. The default values are PLACEHOLDERS
(decision D9): they let the code run and the tests pass, and they change the numbers, not the structure. Replace
them before reading any absolute number. Each field names its spec section and the source the user should use.

The objects are passed explicitly to the functions that need them (api.Link, targets, clutter_scene,
auroral_scene, noise, leg_screens). Nothing reads them from module state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class IrregularityInputs:
    """
    Ionospheric irregularities and their phase screens (spec §8.2 to §8.5, §9.3).

    This dataclass is the single source of truth for these defaults: api.Link takes one (irregularity=...;
    its xi_rms/L0_km/p_index/axial_ratio kwargs override single fields) and leg_screens.py's PLACEHOLDER_*
    constants resolve to these same values (previously three independently-hardcoded, silently differing sets of numbers -- see the
    plan's Stage 0). L0_m is the one field that is NOT a free placeholder: HiCIRF (Nickisch et al. 2012)
    eq. (5) fixes the WBMOD/PDM outer scale at 30 km (wbmod.WBMOD_LO_M).
    """

    xi: float = 5.0e-3                  # fractional rms density fluctuation sigma_N / N_e  (source: campaign data or model)
    L0_m: float = 30.0e3                # outer scale [m]: fixed by HiCIRF eq. (5) at 30 km, not a free placeholder
    p: float = 4.0                      # 3-D spectral index, 3.05 < p < 5.95                (source: model)
    axial_ratio_B: float = 10.0         # a: axial ratio along B                             (source: model)
    axial_ratio_sheet: float = 10.0     # b: axial ratio along the sheet axis                (source: model)
    drift_east_north_mps: tuple = (150.0, 0.0)  # v_d (east, north) [m/s], placeholder E x B drift (source: E x B model or SuperDARN map)
    sigma_v_frac: float = 0.2           # sigma_v / |v_d|, spec §8.5                         (source: model)


@dataclass(frozen=True)
class BraggInputs:
    """Bragg-scale irregularities for auroral backscatter (spec §12.5, Listing 3 'bragg')."""

    eta_perp_m1: float = 1.0e-12        # reflectivity at perfect aspect [m^-1]               (source: irregularity model)
    A_s_db_per_deg: float = 1.0         # aspect sensitivity [dB/deg]                         (source: irregularity model)
    v_ph_mps: tuple = (0.0, 0.0, 0.0)   # phase velocity along q_B [m/s]: E x B (F), Farley-Buneman (E)  (source: model)


@dataclass(frozen=True)
class SurfaceInputs:
    """Surface clutter: sea, sea ice, land and ice sheet (spec §12.4)."""

    wind_direction_deg: float = 0.0     # wind direction relative to the radar look [deg]; 0 = waves travel straight
                                        # towards the radar (clutter.bragg_lines' phi_rel)  (source: CCMP, ccmp_wind.py)
    wind_speed_mps: float = 10.0        # 10 m wind speed: second-order Bragg's wind cutoff g/u^2 (source: CCMP)
    B_s: float = 0.005                  # saturation-spectrum constant                       (source: sea-state model)
    spread_s: float = 3.0               # cardioid exponent s, 2 to 4                        (source: sea-state model)
    ice_concentration: float = 0.0      # c_ice, open-water fraction is 1 - c_ice            (source: ice charts)
    sigma0_ice_dB: float = -30.0        # ice return sigma0 [dB]                              (source: surface model)
    sigma0_land_dB: float = -25.0       # land, tundra, ice sheet sigma0 [dB], typically -30..-20 (source: land-cover model)
    surface_current_radial_mps: float = 0.0   # radial surface current U_r [m/s]               (source: ocean model)


@dataclass(frozen=True)
class NoiseInputs:
    """External noise factor and temperature (spec §12.6)."""

    F_atm_dB: float = 20.0              # atmospheric noise factor, ITU-R P.372 for site, season, time, frequency
    F_gal_dB: float = 20.0              # galactic noise factor, ITU-R P.372 (attenuated by the upward path, §12.6)
    T0_K: float = 290.0                 # reference temperature


@dataclass(frozen=True)
class TargetInputs:
    """Point targets (spec §12.3, §5.3)."""

    rcs_m2: float = 10.0                # radar cross-section sigma_T [m^2]                   (source: user; aircraft and ships are aspect dependent)
    v_T_mps: tuple = (0.0, 0.0, 0.0)    # target velocity, ECEF [m/s]                         (source: user)
    altitude_m: float = 0.0             # h_T; aircraft use the crossing event at this height (§5.3)


@dataclass(frozen=True)
class RadarInputs:
    """System parameters of the radar (used for cell sizes, the link and the noise bandwidth)."""

    bandwidth_hz: float = 20.0e3        # receive bandwidth; range resolution c / (2 B)      (source: waveform)
    beamwidth_deg: float = 3.0          # receive beamwidth [deg]                             (source: array)
    Pt_dBW: float = 60.0                # transmit power [dBW]                                (source: transmitter)
    Gt_dB: float = 0.0                  # transmit gain [dB]                                  (source: antenna)
    Gr_dB: float = 0.0                  # receive gain [dB]                                   (source: antenna)


def placeholders() -> dict:
    """Every input with its current value, grouped by class: what still needs the user's data."""
    out = {}
    for cls in (IrregularityInputs, BraggInputs, SurfaceInputs, NoiseInputs, TargetInputs, RadarInputs):
        out[cls.__name__] = {f.name: getattr(cls(), f.name) for f in fields(cls)}
    return out
