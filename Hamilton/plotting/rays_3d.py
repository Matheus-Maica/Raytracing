"""
rays_3d.py -- PHaRLAP's 3-D ray figure in Python: plot3(lat, mod(lon, 360), height) as in the raytrace_3d examples
(pharlap/src/matlab/examples/ray_test_3d_sp.m, figure 1): O-mode rays as blue dots ('.b', markersize 5), X-mode as
red dots ('.r'), field-free rays as a green line ('g'), Zlim [0 500], grid on, axis labels 'latitude (deg)',
'longitude (deg)', 'Height (km)', legend 'O Mode', 'X Mode', 'No Mag-field', and MATLAB's default 3-D view
(azimuth -37.5 deg, elevation 30 deg).
"""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import MaxNLocator

from core.geometry import xyz_m_to_latlon
from plotting.style import MATLAB_LETTER

# MATLAB view(az, el) -> matplotlib view_init(elev, azim): MATLAB measures az from the -y axis, matplotlib from +x.
MATLAB_VIEW_AZ, MATLAB_VIEW_EL = -37.5, 30.0
_STYLE = {"O": dict(marker=".", linestyle="none", color=MATLAB_LETTER["b"], markersize=5, label="O Mode"),
          "X": dict(marker=".", linestyle="none", color=MATLAB_LETTER["r"], markersize=5, label="X Mode"),
          "N": dict(marker=None, linestyle="-", color=MATLAB_LETTER["g"], label="No Mag-field")}


def lat_lon_height(ray):
    """(lat deg, lon deg in [0, 360), height km) along a ray: an object with `.path` (ECEF m), an (N, 3) array, or a
    PHaRLAP-style dict/object with `lat`, `lon`, `height`."""
    get = (lambda k: ray.get(k)) if isinstance(ray, dict) else (lambda k: getattr(ray, k, None))
    if get("lat") is not None and get("lon") is not None and get("height") is not None:
        lat, lon, h = (np.asarray(get(k), dtype=float) for k in ("lat", "lon", "height"))
    else:
        path = np.asarray(getattr(ray, "path", ray), dtype=float).reshape(-1, 3)
        lat, lon, h_m = xyz_m_to_latlon(path.T)
        h = np.asarray(h_m) / 1e3
    return np.asarray(lat), np.mod(np.asarray(lon), 360.0), np.asarray(h)


def plot_rays_3d(rays, *, modes=None, ax=None, zlim=(0.0, 500.0), legend: bool = True, figsize=(8.4, 6.3)):
    """
    PHaRLAP's 3-D ray figure. rays: ray-likes (see lat_lon_height); modes: one of 'O', 'X', 'N' (no field) per ray,
    default each ray's `.mode` attribute or 'O'. zlim=None lets the height axis follow the data (PHaRLAP fixes
    [0 500]). Returns the 3-D axes.
    """
    import matplotlib.pyplot as plt

    rays = list(rays) if isinstance(rays, (list, tuple)) else [rays]
    if modes is None:
        modes = [str(getattr(r, "mode", "O") or "O").upper()[:1] for r in rays]
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
    labelled = set()
    for r, m in zip(rays, modes):
        lat, lon, h = lat_lon_height(r)
        if zlim is not None:                          # MATLAB clips at Zlim; matplotlib's 3-D axes do not
            h = np.where((h < zlim[0]) | (h > zlim[1]), np.nan, h)
        style = dict(_STYLE.get(m, _STYLE["O"]))
        if style["label"] in labelled:
            style["label"] = "_nolegend_"
        labelled.add(style["label"])
        ax.plot(lat, lon, h, **style)
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.grid(True)
    ax.set_xlabel("latitude (deg)")
    ax.set_ylabel("longitude (deg)")
    ax.set_zlabel("Height (km)")
    ax.view_init(elev=MATLAB_VIEW_EL, azim=MATLAB_VIEW_AZ - 90.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):          # MATLAB's sparser automatic ticks
        axis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 12
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.15, 0.15, 0.15, 1.0))
    if legend and labelled:
        ax.legend(loc="upper right")
    return ax
