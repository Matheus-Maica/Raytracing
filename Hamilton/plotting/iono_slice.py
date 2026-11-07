"""
iono_slice.py -- PHaRLAP's plot_ray_iono_slice in Python, plus what Hamilton needs to feed it.

plot_ray_iono_slice() is a line-by-line port of PHaRLAP's plot_ray_iono_slice.m (V1.2, M. A. Cervera,
pharlap/src/matlab/plot_ray_iono_slice.m): an ionospheric slice drawn in an arc to keep the curved-Earth geometry,
the rays over it, hand-drawn ground-range and altitude ticks, and a plasma-frequency colourbar underneath. The
geometry (Earth radius 6371 km, the tick-step rules, tick lengths, label positions, the 0.1 km pchip resampling of
the rays, the masking outside the image) and the figure layout (width 95% of a 1920 px screen, axes 98% of it,
100 px reserved below for the colourbar, figure height = top + 15 px) follow the MATLAB code; font sizes are its Linux values (14 / 16).

Inputs are PHaRLAP's: a plasma-frequency grid (height x ground range, MHz) and rays with `ground_range` (or
`gndrng`) and `height` in km. iono_slice_grid() samples a Hamilton density along a great circle into such a grid
(gen_iono_grid_2d's role), and rays_from_paths() turns Hamilton ray paths into such rays.

Differences from the MATLAB routine, all deliberate:
* ax=...: draws into an existing axes (MATLAB always resizes the current figure; that breaks subplots). Calling it
  again on the same axes only adds rays, which is how PHaRLAP scripts draw a white fan and then a red ray.
* plot keyword arguments are matplotlib's (color=..., linewidth=...), not MATLAB's property/value pairs.
* A constant grid (e.g. zeros, for rays without an ionosphere) gets colour limits [c, c + 1], so it takes the
  bottom colour of the map.
* The 'Altitude (km)' label keeps PHaRLAP's offset unless the (bold) tick labels would overlap it; then it is
  moved out just enough.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from core.geometry import R_MEAN, xyz_m_to_latlon

RAD_EARTH_KM = 6371.0                 # plot_ray_iono_slice's rad_earth
FONTSIZE1, FONTSIZE2 = 14, 16         # its Linux font sizes (ticks, axis labels)
PF_FACTOR = 80.6164e-6                # PHaRLAP: Ne [cm^-3] = f_p^2 [MHz^2] / 80.6164e-6
SLICE_TAG = "_pharlap_slice"          # attribute set on an axes that already holds a slice


def _colon_len(start: float, stop: float, step: float) -> int:
    """length(start : step : stop) in MATLAB."""
    return int(np.floor((stop - start) / step + 1e-9)) + 1


def _num2str(v: float) -> str:
    v = float(v)
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.4g}"


def _ray_fields(r):
    """(ground_range, height) of one PHaRLAP-style ray: dict or object, 'ground_range' preferred over 'gndrng'."""
    get = (lambda k: r.get(k)) if isinstance(r, dict) else (lambda k: getattr(r, k, None))
    gr = get("ground_range")
    if gr is None:
        gr = get("gndrng")
    ht = get("height")
    if gr is None or ht is None:
        raise ValueError("input ray is not correct type")
    gr, ht = np.asarray(gr, dtype=float).ravel(), np.asarray(ht, dtype=float).ravel()
    if gr.size != ht.size:
        raise ValueError("ray height and ground range vectors have diffent lengths")
    return gr, ht


def range_tick_step(max_range: float) -> float:
    """Ground-range tick step: the acceptable step nearest max_range / 8 (first one on a tie, like MATLAB's min)."""
    acceptable = np.array([100, 150, 200, 250, 500, 1000], dtype=float)
    return float(acceptable[np.argmin(np.abs(acceptable - max_range / 8.0))])


def height_ticks(start_height: float, end_height: float, max_range: float) -> tuple[int, float]:
    """(num_ticks, tick_stepsize) of the altitude axis, exactly as plot_ray_iono_slice.m computes them."""
    num_ticks = int(np.fix(75.0 * (end_height - start_height) / max_range))
    num_ticks = max(2, min(9, num_ticks))
    acceptable = np.array([50, 100, 200, 250, 300, 400, 500, 600, 1000], dtype=float)
    step = (end_height - start_height) / (num_ticks - 1)
    pp = int(np.argmin(np.abs(acceptable - step)))
    step = acceptable[pp]
    if (num_ticks - 1) * step < end_height:
        if (num_ticks - 1) * step < end_height - step:
            if pp < len(acceptable) - 1:
                step = acceptable[pp + 1]
        else:
            num_ticks += 1
    while (num_ticks - 1) * step > end_height:
        num_ticks -= 1
    return num_ticks, float(step)


def _label_clearance_km(ax, tick_labels, tick_len: float) -> float:
    """The smallest offset of the 'Altitude (km)' label from the slice edge that clears the altitude tick labels:
    PHaRLAP's fixed offset assumes MATLAB's narrower default (non-bold) digits, so the label is pushed out only
    when it would overlap. 0 when the layout cannot be measured (no renderer)."""
    try:
        ax.apply_aspect()
        renderer = ax.figure.canvas.get_renderer()
        km_per_px = 1.0 / abs(ax.transData.transform((1.0, 0.0))[0] - ax.transData.transform((0.0, 0.0))[0])
        width_px = max(lbl.get_window_extent(renderer).width for lbl in tick_labels)
        label_px = FONTSIZE2 * ax.figure.dpi / 72.0
    except Exception:                      # noqa: BLE001 -- keep PHaRLAP's offset
        return 0.0
    return 1.5 * tick_len + (width_px + 0.7 * label_px) * km_per_px


def slice_xy(ground_range_km, height_km, max_range: float, start_range: float = 0.0):
    """Curved-Earth image coordinates (km): theta = (ground range - start - max_range/2) / R_E."""
    theta = (np.asarray(ground_range_km, dtype=float) - start_range - max_range / 2.0) / RAD_EARTH_KM
    r = RAD_EARTH_KM + np.asarray(height_km, dtype=float)
    return r * np.sin(theta), r * np.cos(theta)


def plot_ray_iono_slice(iono_grid, start_range, end_range, range_inc, start_height, end_height, height_inc, ray,
                        *, ax=None, fig_width_px: int = 1824, dpi: int = 100, cmap=None, clim=None,
                        colorbar_label: str = "Plasma Frequency (MHz) ", **plot_kwargs):
    """
    Port of PHaRLAP's plot_ray_iono_slice.m. Plots the ionospheric slice iono_grid (heights x ground ranges,
    usually plasma frequency in MHz) in an arc and over-plots the rays.

    ray: a list of rays, each a dict or object with `ground_range` (or `gndrng`) and `height` [km]; [] or None for
    no rays. plot_kwargs go to every ray's line (e.g. color=(1, 1, 0.99), linewidth=2).
    ax: draw into this axes instead of building PHaRLAP's own figure. If it already holds a slice, only the rays
    are added.

    Returns (ax, ray_handles), as the MATLAB function returns (axis_handle, ray_handle).
    """
    import matplotlib.pyplot as plt
    from plotting.style import get_cmap

    iono_grid = np.asarray(iono_grid, dtype=float)
    if _colon_len(start_height, end_height, height_inc) != iono_grid.shape[0]:
        raise ValueError("start_height, end_height and height_inc inconsistent with iono_grid")
    if _colon_len(start_range, end_range, range_inc) != iono_grid.shape[1]:
        raise ValueError("start_range, end_range and range_inc inconsistent with iono_grid")
    rays = [] if ray is None else (list(ray) if isinstance(ray, (list, tuple)) else [ray])
    fields = [_ray_fields(r) for r in rays]

    max_range = end_range - start_range
    R = RAD_EARTH_KM

    # curved Earth geometry of the grid
    r = R + start_height + height_inc * np.arange(iono_grid.shape[0])[:, None]
    gnd_range = range_inc * np.arange(iono_grid.shape[1])[None, :]
    theta = (gnd_range - max_range / 2.0) / R
    iono_X, iono_Y = r * np.sin(theta), r * np.cos(theta)
    min_X, max_X, min_Y, max_Y = iono_X.min(), iono_X.max(), iono_Y.min(), iono_Y.max()

    fresh = ax is None or not getattr(ax, SLICE_TAG, False)
    if ax is None:
        # PHaRLAP's layout: axes 98% wide at 1% from the left, height from the data aspect, 100 px below for the
        # colourbar, 15 px above.
        W = float(fig_width_px)
        width_px = 0.98 * W
        pix_ratio = width_px / (max_X - min_X)
        height_px = pix_ratio * (max_Y - min_Y)
        H = 100.0 + height_px + 15.0
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = fig.add_axes((0.01, 100.0 / H, 0.98, height_px / H))
        cax = fig.add_axes((0.2, 65.0 / H, 0.6, 20.0 / H))
    else:
        fig = ax.figure
        cax = None

    handles = []
    if fresh:
        cm = get_cmap(cmap)
        if clim is None:
            lo, hi = float(np.nanmin(iono_grid)), float(np.nanmax(iono_grid))
            clim = (lo, hi) if hi > lo else (lo, lo + 1.0)
        # MATLAB's pcolor with flat shading colours each cell by its lower-left vertex
        mesh = ax.pcolormesh(iono_X, iono_Y, iono_grid[:-1, :-1], shading="flat", cmap=cm,
                             vmin=clim[0], vmax=clim[1], rasterized=True)
        ax.set_aspect("equal", adjustable="box", anchor="C")
        hspace, vspace = (max_X - min_X) / 40.0, (max_Y - min_Y) / 25.0
        ax.set_xlim(min_X - hspace, max_X)
        ax.set_ylim(min_Y - vspace, max_Y)
        ax.set_axis_off()

        # ground-range ticks
        step = range_tick_step(max_range)
        tick_gndrng = np.arange(int(np.fix(max_range / step)) + 1) * step
        tick_theta = (tick_gndrng - max_range / 2.0) / R
        tick_len = (max_range / 30000.0) * 200.0
        tick_r = R + start_height
        for gr, th in zip(tick_gndrng, tick_theta):
            ax.plot([tick_r * np.sin(th), (tick_r - tick_len) * np.sin(th)],
                    [tick_r * np.cos(th), (tick_r - tick_len) * np.cos(th)], color="k", linewidth=2)
            ax.text((tick_r - 2.0 * tick_len) * np.sin(th), (tick_r - 2.0 * tick_len) * np.cos(th),
                    _num2str(gr + start_range), ha="center", va="center", fontsize=FONTSIZE1)
        ax.text(0.0, R - tick_len * 4.0, "Ground Range (km)", ha="center", va="center", fontsize=FONTSIZE2)

        # altitude ticks (from 0 km at the left edge of the slice)
        num_ticks, hstep = height_ticks(start_height, end_height, max_range)
        th = (0.0 - max_range / 2.0) / R
        tick_len = max_range / 150.0
        height_labels = []
        for idx in range(num_ticks):
            x1, y1 = (R + idx * hstep) * np.sin(th), (R + idx * hstep) * np.cos(th)
            x2, y2 = x1 - tick_len * np.cos(abs(th)), y1 - tick_len * np.sin(abs(th))
            ax.plot([x1, x2], [y1, y2], color="k", linewidth=2)
            height_labels.append(ax.text(x2 - tick_len / 2.0, y2, _num2str(hstep * idx), ha="right", va="center",
                                         fontsize=FONTSIZE1))
        text_rot = -th * 180.0 / np.pi + 90.0
        pos_adjust = tick_len * ((end_height - start_height) / 400.0 + 5.0)
        pos_adjust = max(pos_adjust, _label_clearance_km(ax, height_labels, tick_len))
        ax.text((R + (end_height - start_height) / 2.0) * np.sin(th) - pos_adjust * np.cos(abs(th)),
                (R + (end_height - start_height) / 2.0) * np.cos(th) - pos_adjust * np.sin(abs(th)),
                "Altitude (km)", rotation=text_rot, rotation_mode="anchor", ha="center", va="center",
                fontsize=FONTSIZE2)

        # colourbar
        if cax is not None:
            cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        else:
            cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", location="bottom", shrink=0.6,
                              pad=0.02, aspect=40)
        cb.ax.tick_params(labelsize=FONTSIZE1, direction="in")
        cb.ax.xaxis.set_ticks_position("bottom")
        cb.set_label(colorbar_label, fontsize=FONTSIZE2)
        ax.pharlap_colorbar = cb
        setattr(ax, SLICE_TAG, True)
        ax.pharlap_geometry = dict(start_range=start_range, end_range=end_range, start_height=start_height,
                                   end_height=end_height, max_range=max_range)

    # the rays: resampled every 0.1 km of ground range (pchip), masked outside the image
    for gr, ht in fields:
        if gr.size == 0:
            continue
        keep = np.concatenate([[True], np.diff(gr) > 0.0])
        gr_k, ht_k = gr[keep], ht[keep]
        if gr_k.size >= 2:
            ray_gndrng = np.arange(gr_k[0], gr_k[-1] + 1e-9, 0.1)
            ray_height = PchipInterpolator(gr_k, ht_k)(ray_gndrng)
        else:
            ray_gndrng, ray_height = gr.copy(), ht.copy()
        ray_gndrng = np.where((ray_gndrng < start_range) | (ray_gndrng > end_range), np.nan, ray_gndrng)
        ray_height = np.where((ray_height < start_height) | (ray_height > end_height), np.nan, ray_height)
        X, Y = slice_xy(ray_gndrng, ray_height, max_range, start_range)
        (h,) = ax.plot(X, Y, **plot_kwargs)
        handles.append(h)
    return ax, handles


# ---------------------------------------------------------------------------
# Feeding it from Hamilton
# ---------------------------------------------------------------------------

def destination_latlon(origin_lat: float, origin_lon: float, bearing_deg: float, range_km):
    """Great-circle destination (deg) at range_km along bearing_deg (clockwise from north), on a sphere of
    radius R_MEAN -- the same sphere rays_from_paths() measures ground range on."""
    d = np.asarray(range_km, dtype=float) / (R_MEAN / 1e3)
    p1, l1, b = np.deg2rad(origin_lat), np.deg2rad(origin_lon), np.deg2rad(bearing_deg)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(b))
    l2 = l1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.rad2deg(p2), (np.rad2deg(l2) + 180.0) % 360.0 - 180.0


def iono_slice_grid(density_geodetic, origin_lat: float, origin_lon: float, bearing_deg: float, start_range: float,
                    end_range: float, range_inc: float, start_height: float, end_height: float,
                    height_inc: float) -> np.ndarray:
    """
    Plasma-frequency slice [MHz], shape (heights, ground ranges), for plot_ray_iono_slice: the electron density
    density_geodetic(lat_deg, lon_deg, h_km) -> m^-3 (e.g. geophysics.ionosphere.Ionosphere.electron_density_geodetic)
    sampled on the great circle from the origin along bearing_deg. f_p = sqrt(80.6164e-6 Ne[cm^-3]) as in PHaRLAP.
    """
    ranges = start_range + range_inc * np.arange(_colon_len(start_range, end_range, range_inc))
    heights = start_height + height_inc * np.arange(_colon_len(start_height, end_height, height_inc))
    lat, lon = destination_latlon(origin_lat, origin_lon, bearing_deg, ranges)
    H, LAT = np.meshgrid(heights, lat, indexing="ij")
    LON = np.broadcast_to(lon[None, :], H.shape)
    ne = np.asarray(density_geodetic(LAT.ravel(), LON.ravel(), H.ravel()), dtype=float).reshape(H.shape)
    return np.sqrt(PF_FACTOR * np.clip(ne, 0.0, None) * 1e-6)


def ground_range_height_km(path_ecef_m, origin_ecef_m):
    """(ground range, height) [km] of each point of a path, as PHaRLAP reports them: the great-circle distance from
    the origin to the point's sub-point (sphere of radius R_MEAN) and the height above the WGS84 ellipsoid."""
    path = np.asarray(path_ecef_m, dtype=float).reshape(-1, 3)
    o = np.asarray(origin_ecef_m, dtype=float)
    cosang = (path @ o) / (np.linalg.norm(path, axis=1) * np.linalg.norm(o))
    gr = R_MEAN * np.arccos(np.clip(cosang, -1.0, 1.0)) / 1e3
    _, _, h = xyz_m_to_latlon(path.T)
    return gr, np.asarray(h, dtype=float) / 1e3


def rays_from_paths(rays, origin_ecef_m=None) -> list[dict]:
    """PHaRLAP-style rays ({'ground_range', 'height'} in km) from objects with `.path` (ECEF m, e.g. api.RayPath)
    or bare (N, 3) arrays. Ground range is measured from origin_ecef_m, else from each ray's `launch_point`, else
    from its first point."""
    out = []
    for r in rays:
        path = np.asarray(getattr(r, "path", r), dtype=float).reshape(-1, 3)
        origin = origin_ecef_m
        if origin is None:
            origin = getattr(r, "launch_point", None)
        if origin is None:
            origin = path[0]
        gr, h = ground_range_height_km(path, origin)
        out.append({"ground_range": gr, "height": h})
    return out
