"""
figures.py -- the composed figures behind api.Link.plot / plot_3d / plot_scattering_function and RayPath.plot.

ray_slice_figure() is PHaRLAP's range-height figure as this group's scripts draw it (pharlap/src/matlab/examples/
ray_test_3d.m, "Plot Set 2"): plot_ray_iono_slice with the ionosphere along the ray bearing, the ray fan in thin
white, the highlighted ray in thick red labelled with its Tx elevation, and a monospaced UTC / frequency / bearing
note in the top-left corner. Without a highlighted ray the fan is drawn as in PHaRLAP's ray_test1.m (near-white,
line width 2). HiCIRF adds, for the highlighted ray, its phase screens (markers, area ~ the phase variance each
adds) and a box with its channel parameters in the top-right corner.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from plotting import hicirf as _hic
from plotting.iono_slice import (destination_latlon, iono_slice_grid, plot_ray_iono_slice, rays_from_paths,
                                 slice_xy)
from plotting.rays_3d import plot_rays_3d
from plotting.style import matlab_style

FAN_HIGHLIGHT_STYLE = dict(color=(1.0, 1.0, 1.0), linewidth=0.4)       # Plot Set 2: white fan
HIGHLIGHT_STYLE = dict(color=(1.0, 0.0, 0.0), linewidth=2.5)           # Plot Set 2: red eigenray
FAN_STYLE = dict(color=(1.0, 1.0, 0.99), linewidth=2.0)                # ray_test1.m


TITLE_MARGIN_PX = 40.0          # PHaRLAP leaves 15 px above the slice; a title needs more


def _add_top_margin(fig, px: float) -> None:
    """Make the figure px taller at the top, keeping every axes at the same pixel position."""
    H0 = fig.get_figheight() * fig.dpi
    boxes = [(a, a.get_position(original=True)) for a in fig.axes]
    fig.set_figheight((H0 + px) / fig.dpi)
    s = H0 / (H0 + px)
    for a, b in boxes:
        a.set_position((b.x0, b.y0 * s, b.width, b.height * s))


def _nice_extent(rays):
    """Default slice extents from the rays: end range rounded up (to 100 km below 1000 km, else 500 km), end height
    400 km unless a landed ray's apex needs more (rounded up to 100 km, at most 800 km)."""
    max_gr = max([float(np.nanmax(r["ground_range"])) for r in rays if len(r["ground_range"])] + [0.0])
    step = 100.0 if max_gr < 1000.0 else 500.0
    end_range = max(500.0, step * np.ceil((max_gr + 1e-6) / step))
    apex = [float(np.nanmax(r["height"])) for r in rays if r.get("landed") and len(r["height"])]
    end_height = 400.0
    if apex and max(apex) + 50.0 > end_height:
        end_height = min(800.0, 100.0 * np.ceil((max(apex) + 50.0) / 100.0))
    range_inc = next(i for i in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0) if end_range / i <= 1500.0)
    return end_range, range_inc, end_height


def _utc_str(dt) -> str:
    return f"{dt:%Y%m%d|%H%M%S}" if isinstance(dt, datetime) else str(dt)


def ray_slice_figure(rays, *, highlight=None, density_geodetic=None, origin_lat=None, origin_lon=None,
                     origin_ecef=None, bearing_deg=None, utc=None, start_range: float = 0.0, end_range=None,
                     range_inc=None, start_height: float = 0.0, end_height=None, height_inc: float = 2.0,
                     hicirf=None, title=None, startup: bool = True, cmap=None, fig_width_px: int = 1600,
                     note: bool = True):
    """
    PHaRLAP-style range-height figure of ray paths (objects with `.path` in ECEF metres, e.g. api.RayPath).

    highlight: the ray to draw in red (one of `rays` or another); None draws a plain PHaRLAP fan.
    density_geodetic(lat, lon, h_km) -> Ne [m^-3]: the background slice along bearing_deg from (origin_lat,
    origin_lon); None leaves it empty (zeros), as PHaRLAP does for rays without a grid.
    origin_ecef: where ground range is measured from (default: each ray's launch point).
    hicirf: dict with 'samples', 'screens', 'channel' (and optionally 'absorption_db') for the highlighted leg, or
    {'error': message}; draws the screens and the channel box.
    Extents default from the rays (see _nice_extent). Returns the matplotlib Figure.
    """
    rays = list(rays) if isinstance(rays, (list, tuple)) else [rays]
    fan = [r for r in rays if r is not highlight]
    all_rays = fan + ([highlight] if highlight is not None else [])
    slice_rays = rays_from_paths(all_rays, origin_ecef)
    for d, r in zip(slice_rays, all_rays):
        d["landed"] = bool(getattr(r, "landed", True))
    auto_end, auto_inc, auto_height = _nice_extent(slice_rays)
    end_range = auto_end if end_range is None else float(end_range)
    range_inc = auto_inc if range_inc is None else float(range_inc)
    end_height = auto_height if end_height is None else float(end_height)
    ref = highlight if highlight is not None else (rays[0] if rays else None)
    if bearing_deg is None and ref is not None:
        bearing_deg = float(getattr(ref, "azimuth_deg", 0.0))

    n_r = int(np.floor((end_range - start_range) / range_inc + 1e-9)) + 1
    n_h = int(np.floor((end_height - start_height) / height_inc + 1e-9)) + 1
    if density_geodetic is not None and origin_lat is not None and origin_lon is not None:
        grid = iono_slice_grid(density_geodetic, origin_lat, origin_lon, bearing_deg or 0.0, start_range, end_range,
                               range_inc, start_height, end_height, height_inc)
    else:
        grid = np.zeros((n_h, n_r))

    with matlab_style(startup):
        fan_style = FAN_HIGHLIGHT_STYLE if highlight is not None else FAN_STYLE
        ax, _ = plot_ray_iono_slice(grid, start_range, end_range, range_inc, start_height, end_height, height_inc,
                                    slice_rays[:len(fan)], cmap=cmap, fig_width_px=fig_width_px, **fan_style)
        fig = ax.figure
        if highlight is not None:
            hl = slice_rays[-1]
            plot_ray_iono_slice(grid, start_range, end_range, range_inc, start_height, end_height, height_inc,
                                [hl], ax=ax, **HIGHLIGHT_STYLE)
            elev = getattr(highlight, "elevation_deg", None)
            if elev is not None and len(hl["ground_range"]):
                x, y = slice_xy(hl["ground_range"][0] + 30.0, max(hl["height"][0], start_height) + 20.0,
                                end_range - start_range, start_range)
                ax.text(x, y, f"Tx Elevation:\n{elev:.2f}°", color="r", fontsize=8, fontweight="bold",
                        va="bottom")
        if title:
            _add_top_margin(fig, TITLE_MARGIN_PX)
            fig.suptitle(title, fontsize=13, fontweight="bold",
                         y=1.0 - 0.5 * TITLE_MARGIN_PX / (fig.get_figheight() * fig.dpi), va="center")
        # The note and the HiCIRF box go in PHaRLAP's 100 px band under the slice, either side of the colourbar
        # (which spans 20-80% of the width), centred 50 px up, so that they never cover the slice.
        band_mid = 50.0 / (fig.get_figheight() * fig.dpi)
        if note and ref is not None:
            lines = []
            if utc is not None:
                lines.append(f"UTC: {_utc_str(utc)}")
            if getattr(ref, "frequency_mhz", None) is not None:
                lines.append(f"Frequency: {ref.frequency_mhz:g} MHz ({getattr(ref, 'mode', '?')}-mode)")
            if bearing_deg is not None:
                lines.append(f"Bearing: {bearing_deg:.4f}°")
            if lines:
                _hic.draw_text_box(fig, lines, x=0.01, y=band_mid, va="center", box=False)
        if hicirf is not None and highlight is not None:
            if "error" in hicirf:
                _hic.draw_text_box(fig, [f"n/a: {hicirf['error']}"], x=0.99, y=band_mid, ha="right", va="center",
                                   title="HiCIRF, red ray")
            else:
                origin = origin_ecef if origin_ecef is not None else getattr(highlight, "launch_point", None)
                gr, h, s2 = _hic.screen_track(hicirf["samples"], hicirf["screens"], origin)
                _hic.draw_screens(ax, gr, h, s2)
                lines = _hic.channel_lines(hicirf["channel"], absorption_db=hicirf.get("absorption_db"))
                _hic.draw_text_box(fig, lines, x=0.99, y=band_mid, ha="right", va="center",
                                   title=f"HiCIRF, red ray ({len(s2)} screens ○)")
    return fig


def rays_3d_figure(rays, *, startup: bool = True, title=None, **kwargs):
    """PHaRLAP's 3-D lat/lon/height figure (plotting.rays_3d.plot_rays_3d). Returns the Figure."""
    with matlab_style(startup):
        ax = plot_rays_3d(rays, **kwargs)
        if title:
            ax.set_title(title)
    return ax.figure


def scattering_function_figure(channel, *, startup: bool = True, title=None, **kwargs):
    """The HiCIRF delay-Doppler scattering function of one leg (plotting.hicirf.plot_scattering_function)."""
    with matlab_style(startup):
        ax = _hic.plot_scattering_function(channel, title=title, **kwargs)
    return ax.figure


__all__ = ["ray_slice_figure", "rays_3d_figure", "scattering_function_figure", "destination_latlon"]
