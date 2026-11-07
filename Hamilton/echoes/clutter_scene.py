"""
clutter_scene.py -- surface clutter on a ground grid (spec §12.4 cell rasterisation, §12.4.1 array field).

The landing maps of the two sites (a launch fan from the Tx and one from the Rx, both on the batch path) are
rasterised onto a common ground grid about the midpoint of the sites. Each quadrilateral of a fan is inverted
bilinearly (the map of Algorithm 2) for every grid cell whose centre it contains. A ground cell therefore receives
the outbound legs and the return legs that land in it, each with the group path P', the absorption, the Doppler, and
the flux-tube distance d^2 = A_perp sin(psi) / cos(beta) (eq. 4.7, from the planar Jacobian of the map).

Every (outbound, return) pair in a cell gives one ClutterComponent:
    power  : eq. (6.2) with sigma_T -> sigma0 * A_cell, sigma0 per Bragg line (sea) or the land value (land)
    delay  : (P'_out + P'_ret) / c
    Doppler: f_out + f_ret + the surface line (Bragg lines at +/- f_B with the current shift, or 0 for land)
The surface type of a cell comes from the land mask (environment.SeaLandSurface); sea ice is a user input.

Array part (§12.4.1, explicit components): element phases exp(i k0 (u x_q + v y_q)) with the direction cosines
(u, v) of the return leg's launch direction in the Rx array plane (east, north axes at the Rx site).

Surface and radar values are USER INPUTS (inputs.SurfaceInputs, inputs.RadarInputs); the defaults are placeholders.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from echoes import clutter
from propagation import fans
from propagation import legs
from core.constants import C
from geophysics.environment import SeaLandSurface
from core.geometry import ellipsoid_outward_normal_m
from core.inputs import RadarInputs, SurfaceInputs
from propagation.legs import L_POL_DB

LAND_LINE_HZ = 0.0          # land and ice sheet: narrow line at zero Doppler (spec §12.4)


@dataclass
class ClutterComponent:
    """One clutter component of an out/return pair in one ground cell (not a Component: no LegMode here)."""

    cell: tuple             # (ix, iy)
    surface: str            # 'sea' or 'land'
    line_hz: float          # Bragg or land line offset, Hz (before the leg Dopplers)
    sigma0: float           # linear sigma0 of the line
    power_W: float
    delay_s: float
    doppler_Hz: float
    d_T: float              # flux-tube distance of the outbound leg at the cell [m]
    d_R: float              # flux-tube distance of the return leg at the cell [m]
    e_ret: np.ndarray       # launch direction of the return leg at the Rx (for the array phases)


def range_resolution_m(bandwidth_hz: float) -> float:
    """Range resolution c / (2 B) in group range."""
    return C / (2.0 * bandwidth_hz)


def default_cell_m(radar: RadarInputs, range_m: float) -> float:
    """Cell size: at most half the range resolution and at most the receive beamwidth at the range (§12.4)."""
    return min(0.5 * range_resolution_m(radar.bandwidth_hz), range_m * np.deg2rad(radar.beamwidth_deg))


def mid_direction(r_tx: np.ndarray, r_rx: np.ndarray) -> np.ndarray:
    """Unit direction of the midpoint of the two sites: the centre of the ground grid."""
    m = r_tx / np.linalg.norm(r_tx) + r_rx / np.linalg.norm(r_rx)
    return m / np.linalg.norm(m)


def planar_to_direction(r_mid: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Inverse of fans.site_planar about r_mid: planar (east, north) metres -> unit direction (ECEF)."""
    p0 = r_mid / np.linalg.norm(r_mid)
    east = np.cross([0.0, 0.0, 1.0], p0)
    east /= np.linalg.norm(east)
    north = np.cross(p0, east)
    xy = np.asarray(xy, dtype=float)
    dist = np.linalg.norm(xy)
    if dist == 0.0:
        return p0
    gamma = dist / R_MEAN_M
    tdir = (xy[0] * east + xy[1] * north) / dist
    return np.cos(gamma) * p0 + np.sin(gamma) * tdir


R_MEAN_M = 6371008.8          # mean radius used by fans.site_planar (geometry.R_MEAN)


def approach_azimuth_deg(d_cell: np.ndarray, r_mid: np.ndarray) -> float:
    """Compass azimuth [deg, clockwise from north] at the cell of the great circle towards the sites' midpoint:
    the direction approaching waves travel (clutter.bragg_lines' phi_rel = 0)."""
    c = np.asarray(d_cell, dtype=float) / np.linalg.norm(d_cell)
    m = np.asarray(r_mid, dtype=float) / np.linalg.norm(r_mid)
    t = m - np.dot(m, c) * c
    east = np.cross([0.0, 0.0, 1.0], c)
    east /= np.linalg.norm(east)
    north = np.cross(c, east)
    return float(np.degrees(np.arctan2(np.dot(t, east), np.dot(t, north))))


def _bilinear_weights(xi: float, eta: float) -> np.ndarray:
    return np.array([(1 - xi) * (1 - eta), xi * (1 - eta), xi * eta, (1 - xi) * eta])


def rasterise_fan(fan, hop: int, r_mid: np.ndarray, planar: np.ndarray, *, origin: np.ndarray, cell_m: float,
                  nx: int, ny: int, e_launch: np.ndarray) -> list[dict]:
    """
    Rasterise one fan's landing map at one hop onto the ground cells. planar: (N, 2) landing points about r_mid
    (fans.site_planar(r_mid, fan.land[:, hop])). e_launch: (N, 3) launch directions at the site.
    Returns a list of contributions: dict(cell, d2, Pg, A_db, doppler, e_launch).
    """
    Nb, Na = fan.grid_shape
    valid = fan.valid[:, hop].reshape(Nb, Na)
    P = planar.reshape(Nb, Na, 2)
    Pg = fan.Pg[:, hop].reshape(Nb, Na)
    A_db = fan.A_db[:, hop].reshape(Nb, Na)
    dop = np.nan_to_num(fan.doppler_hz[:, hop], nan=0.0).reshape(Nb, Na)
    land = fan.land[:, hop].reshape(Nb, Na, 3)
    s_inc = fan.s_inc[:, hop].reshape(Nb, Na, 3)
    el = np.deg2rad(fan.elev_deg.reshape(Nb, Na)[:, 0])
    eL = e_launch.reshape(Nb, Na, 3)
    out = []
    for i in range(Nb - 1):
        for j in range(Na - 1):
            if not (valid[i, j] and valid[i + 1, j] and valid[i + 1, j + 1] and valid[i, j + 1]):
                continue
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]              # 00, 10, 11, 01
            x = np.array([P[c] for c in corners])
            lo, hi = x.min(axis=0), x.max(axis=0)
            ix0 = max(int(np.floor((lo[0] - origin[0]) / cell_m)), 0)
            ix1 = min(int(np.floor((hi[0] - origin[0]) / cell_m)), nx - 1)
            iy0 = max(int(np.floor((lo[1] - origin[1]) / cell_m)), 0)
            iy1 = min(int(np.floor((hi[1] - origin[1]) / cell_m)), ny - 1)
            if ix1 < ix0 or iy1 < iy0:
                continue
            dbeta = abs(el[min(i + 1, Nb - 1)] - el[i]) or abs(el[i] - el[i - 1])
            dalpha = np.deg2rad(abs(fan.az_deg.reshape(Nb, Na)[0, min(j + 1, Na - 1)] - fan.az_deg.reshape(Nb, Na)[0, j])) or 1e-9
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    c = origin + (np.array([ix, iy]) + 0.5) * cell_m
                    xi, eta, ok = 0.5, 0.5, False
                    for _ in range(12):
                        F = fans._bilinear_map(x, xi, eta) - c
                        dxi, deta = fans._bilinear_jac(x, xi, eta)
                        J = np.column_stack([dxi, deta])
                        if abs(np.linalg.det(J)) < 1e-12:
                            break
                        step = np.linalg.solve(J, -F)
                        xi, eta = xi + step[0], eta + step[1]
                        if np.hypot(*step) < 1e-10:
                            ok = True
                            break
                    if not ok or not (-1e-9 <= xi <= 1 + 1e-9 and -1e-9 <= eta <= 1 + 1e-9):
                        continue
                    w = _bilinear_weights(xi, eta)
                    pick = lambda arr: w @ np.array([arr[k, l] for k, l in corners])     # noqa: E731
                    Pg_c, A_c, dop_c = pick(Pg), pick(A_db), pick(dop)
                    el_c = w @ np.array([el[k] for k, _ in corners])
                    pt = w @ np.array([land[k, l] for k, l in corners])
                    s_c = w @ np.array([s_inc[k, l] for k, l in corners])
                    s_c = s_c / np.linalg.norm(s_c)
                    sin_psi = float(np.clip(-np.dot(s_c, ellipsoid_outward_normal_m(pt)), 1e-6, 1.0))
                    dxi, deta = fans._bilinear_jac(x, xi, eta)
                    A_perp = abs(dxi[0] * deta[1] - dxi[1] * deta[0]) / (dbeta * dalpha)
                    e_c = w @ np.array([eL[k, l] for k, l in corners])
                    out.append(dict(cell=(ix, iy), d2=float(A_perp * sin_psi / np.cos(el_c)), Pg=float(Pg_c),
                                    A_db=float(A_c), doppler=float(dop_c),
                                    e_launch=e_c / np.linalg.norm(e_c)))
    return out


def _site_frame(r_site: np.ndarray):
    up = r_site / np.linalg.norm(r_site)
    lon = np.arctan2(r_site[1], r_site[0])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(up, east)
    return up, north, east


def scene_components(fan_tx, fan_rx, r_tx: np.ndarray, r_rx: np.ndarray, *, f0: float,
                     surface: SurfaceInputs, radar: RadarInputs, cell_m: float, hop: int = 0,
                     max_pairs: int = 200000, surface_of=None, wind_model=None, t=None,
                     second_order: bool = False):
    """
    Clutter components of one scene from the Tx and Rx fans at hop `hop` (hop 0 = one hop). Returns
    (components, grid) with grid = dict(origin, nx, ny, cell_m, r_mid, n_cells_hit).
    The fans must be traced with fans.trace_fan(..., grid=True) from the two sites on the batch path.

    wind_model: optional ccmp_wind.CCMPWind (or anything with the same .at(lat, lon, t) -> dict interface,
    returning wind_direction_deg -- the absolute compass direction the wind blows TOWARDS -- and
    wind_speed_mps), queried per sea cell instead of the scene-wide surface.wind_direction_deg/wind_speed_mps.
    Its absolute direction is converted per cell to bragg_lines' phi_rel (relative to waves approaching the
    radar, i.e. travelling from the cell towards the sites' midpoint); NaN (CCMP has no value, e.g. sea ice)
    falls back to the scalar inputs. t: the epoch to query it at. second_order: add the second-order continuum.
    """
    r_mid = mid_direction(r_tx, r_rx)
    pts_tx = fans.site_planar(r_mid, fan_tx.land[:, hop])
    pts_rx = fans.site_planar(r_mid, fan_rx.land[:, hop])
    allp = np.vstack([pts_tx[fan_tx.valid[:, hop]], pts_rx[fan_rx.valid[:, hop]]])
    if allp.size == 0:
        raise ValueError("no landing point of either fan at this hop")
    origin = np.floor(allp.min(axis=0) / cell_m) * cell_m
    nx = int(np.ceil((allp[:, 0].max() - origin[0]) / cell_m)) + 1
    ny = int(np.ceil((allp[:, 1].max() - origin[1]) / cell_m)) + 1
    up_t, north_t, east_t = _site_frame(r_tx)
    up_r, north_r, east_r = _site_frame(r_rx)
    eL_tx = np.array([legs.launch_direction(e, a, up_t, north_t, east_t)
                      for e, a in zip(fan_tx.elev_deg, fan_tx.az_deg)])
    eL_rx = np.array([legs.launch_direction(e, a, up_r, north_r, east_r)
                      for e, a in zip(fan_rx.elev_deg, fan_rx.az_deg)])
    out_list = rasterise_fan(fan_tx, hop, r_mid, pts_tx, origin=origin, cell_m=cell_m, nx=nx, ny=ny,
                             e_launch=eL_tx)
    ret_list = rasterise_fan(fan_rx, hop, r_mid, pts_rx, origin=origin, cell_m=cell_m, nx=nx, ny=ny,
                             e_launch=eL_rx)
    by_out, by_ret = defaultdict(list), defaultdict(list)
    for c in out_list:
        by_out[c["cell"]].append(c)
    for c in ret_list:
        by_ret[c["cell"]].append(c)
    surf_model = SeaLandSurface()
    comps: list[ClutterComponent] = []
    cells_hit = 0
    for cell in sorted(set(by_out) & set(by_ret)):
        cells_hit += 1
        centre = origin + (np.array(cell) + 0.5) * cell_m
        d_ctr = planar_to_direction(r_mid, centre)
        lat = float(np.degrees(np.arcsin(np.clip(d_ctr[2], -1.0, 1.0))))
        lon = float(np.degrees(np.arctan2(d_ctr[1], d_ctr[0])))
        if surface_of is not None:
            surf = surface_of(lat, lon)
        else:
            surf = str(np.atleast_1d(surf_model.at(lat, lon, None)["surface"])[0])
        if surf == "sea":
            phi_rel_deg, speed = surface.wind_direction_deg, surface.wind_speed_mps
            if wind_model is not None:
                w = wind_model.at(lat, lon, t)
                wd = float(np.atleast_1d(w["wind_direction_deg"])[0])
                ws = float(np.atleast_1d(w.get("wind_speed_mps", np.nan))[0])
                if np.isfinite(wd):
                    phi_rel_deg = wd - approach_azimuth_deg(d_ctr, r_mid)
                if np.isfinite(ws):
                    speed = ws
            lines = clutter.bragg_lines(f0, np.deg2rad(phi_rel_deg), c_ice=surface.ice_concentration,
                                        Bs=surface.B_s, s=surface.spread_s,
                                        U_radial=surface.surface_current_radial_mps,
                                        second_order=second_order, wind_speed_mps=speed)
        else:
            lines = [(LAND_LINE_HZ, clutter.land_sigma0_linear(surface.sigma0_land_dB))]
        A_cell = cell_m * cell_m
        for o in by_out[cell]:
            for r in by_ret[cell]:
                for line_hz, s0 in lines:
                    if s0 <= 0.0:
                        continue
                    p = clutter.clutter_cell_power_W(
                        sigma0=s0, area_m2=A_cell, Pt_dBW=radar.Pt_dBW, f0=f0,
                        d_T=float(np.sqrt(o["d2"])), d_R=float(np.sqrt(r["d2"])),
                        loss_T_dB=o["A_db"] + L_POL_DB, loss_R_dB=r["A_db"] + L_POL_DB,
                        Gt_dB=radar.Gt_dB, Gr_dB=radar.Gr_dB)
                    comps.append(ClutterComponent(
                        cell=cell, surface=surf, line_hz=float(line_hz), sigma0=float(s0), power_W=p,
                        delay_s=(o["Pg"] + r["Pg"]) / C, doppler_Hz=o["doppler"] + r["doppler"] + float(line_hz),
                        d_T=float(np.sqrt(o["d2"])), d_R=float(np.sqrt(r["d2"])), e_ret=r["e_launch"]))
                    if len(comps) > max_pairs:
                        raise RuntimeError("too many clutter components: use a coarser cell or a smaller scene")
    grid = dict(origin=origin, nx=nx, ny=ny, cell_m=cell_m, r_mid=r_mid, n_cells_hit=cells_hit)
    return comps, grid


def element_phases(components, positions_xy, *, f0: float, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """
    Element phases of the components (§12.4.1, §11.5): exp(i k0 (u x_q + v y_q)) with (u, v) = (e . east, e . north)
    of each component's return launch direction at the Rx. positions_xy: (Np, 2) element coordinates [m].
    Returns H (Np, Ncomp).
    """
    k0 = 2.0 * np.pi * f0 / C
    pos = np.atleast_2d(np.asarray(positions_xy, dtype=float))
    H = np.empty((pos.shape[0], len(components)), dtype=complex)
    for k, c in enumerate(components):
        u, v = float(np.dot(c.e_ret, east)), float(np.dot(c.e_ret, north))
        H[:, k] = np.exp(1j * k0 * (u * pos[:, 0] + v * pos[:, 1]))
    return H
