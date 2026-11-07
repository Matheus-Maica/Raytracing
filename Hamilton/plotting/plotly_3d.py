"""
plotly_3d.py — interactive 3-D ray-trajectory plot (Plotly) in true
ECEF space over a WGS84 ellipsoid ground patch.

The old default plot (raytracing/visualization.py until 2026-11-07), kept as an
extra: the default figures are now PHaRLAP-style (plotting/figures.py).
Rotatable/zoomable HTML output. Positions are plotted in actual geocentric Cartesian (ECEF)
coordinates, in km, with `scene.aspectmode="data"` (true, undistorted
proportions) -- NOT (longitude, latitude, altitude) treated as orthogonal
axes, which is a real distortion (a straight line in true 3-D space does
not, in general, look straight in a lon/lat/alt "box": equal steps along a
great circle don't correspond to equal (delta-lat, delta-lon) steps except
along special paths). A translucent local patch of the WGS84 ellipsoid
(geometry.WGS84_A/WGS84_B) is drawn as the ground reference surface, with
a lat/lon graticule via Surface contours.

Usage
-----
    from plotting.plotly_3d import plot_ray
    fig = plot_ray(det)                       # a single TraceResult/RayResult
    fig = plot_ray(results, highlight=det)     # an ensemble + a highlighted ray
    fig.write_html("ray_scene_3d.html")        # open in any browser, rotate/zoom
    # fig.show()                               # also works if you have a
                                                # configured renderer (e.g. browser)

Any object exposing a `.path` attribute ((N,3) ndarray, geocentric metres)
works -- raytracer.TraceResult and mc_scattering.RayResult both qualify,
duck-typed (no import of either, to keep this module a generic display
layer). `.bounce_points`, `.landed`, `.escaped` are used if present, for
bounce markers and end-point styling. A bare (N,3) ndarray also works.
"""

import numpy as np
import plotly.graph_objects as go

from core.geometry import xyz_m_to_latlon, latlon_to_xyz_m

try:
    from global_land_mask import globe as _land_globe
    _HAS_LAND_MASK = True
except ImportError:
    _HAS_LAND_MASK = False

_OCEAN_COLOR = "#0b3d63"
_LAND_COLOR = "#3f6b3f"
_FLAT_GROUND_COLOR = "#3d6a96"   # fallback when global-land-mask isn't installed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ray_path_and_meta(ray):
    """Duck-type a ray-like object (or bare (N,3) array) into (path_m, bounce_points_m, landed, escaped)."""
    if hasattr(ray, "path"):
        path = np.asarray(ray.path, dtype=float)
        bounce = getattr(ray, "bounce_points", None)
        bounce = np.asarray(bounce, dtype=float) if bounce is not None and len(bounce) else None
        landed = bool(getattr(ray, "landed", False))
        escaped = bool(getattr(ray, "escaped", False))
        return path, bounce, landed, escaped
    path = np.asarray(ray, dtype=float)
    return path, None, False, False


def _altitude_km(positions_m: np.ndarray) -> np.ndarray:
    """Geocentric metres (N,3) -> WGS84 altitude (N,), km."""
    _, _, alt_m = xyz_m_to_latlon(positions_m.T)
    return alt_m / 1e3


def _ellipsoid_patch_km(lat_min, lat_max, lon_min, lon_max, n=120):
    """
    Translucent local WGS84 ellipsoid ground patch (alt=0), in km, for
    go.Surface, plus the (lat, lon) grids used to build it (so callers can
    sample land/ocean or other lat/lon-indexed data at matching resolution).
    n defaults higher than the old flat-color patch (40) needed, since a
    recognizable coastline needs more samples than a plain curved sheet did.
    """
    lat_grid = np.linspace(lat_min, lat_max, n)
    lon_grid = np.linspace(lon_min, lon_max, n)
    LATg, LONg = np.meshgrid(lat_grid, lon_grid, indexing="ij")
    xyz_m = latlon_to_xyz_m(LATg, LONg, 0.0)   # (3, n, n), broadcasts over the grids
    return xyz_m[0] / 1e3, xyz_m[1] / 1e3, xyz_m[2] / 1e3, LATg, LONg


def _land_ocean_surfacecolor(LATg: np.ndarray, LONg: np.ndarray):
    """
    (n,n) lat/lon grids -> (n,n) float array (0.0 ocean, 1.0 land) plus a
    matching 2-colour discrete Plotly colorscale, for go.Surface's
    surfacecolor/colorscale. Wraps longitude into [-180,180] first, since
    the ellipsoid patch's lon axis is an unwrapped range that can extend
    past 180 deg or below -180 deg for patches crossing the antimeridian
    (global_land_mask requires lon in [-180, 180]).
    """
    lon_wrapped = ((LONg + 180.0) % 360.0) - 180.0
    is_land = _land_globe.is_land(LATg, lon_wrapped)
    surfacecolor = is_land.astype(float)
    colorscale = [
        [0.0, _OCEAN_COLOR], [0.499, _OCEAN_COLOR],
        [0.5, _LAND_COLOR], [1.0, _LAND_COLOR],
    ]
    return surfacecolor, colorscale


def _concat_with_gaps(arrays):
    """Concatenate a list of (Ni,3) arrays into one (M,3) array with a NaN
    row between each, so they render as separate Plotly line segments
    within a SINGLE trace (much cheaper than one trace per ray for large
    ensembles)."""
    if not arrays:
        return np.zeros((0, 3))
    gap = np.full((1, 3), np.nan)
    parts = []
    for a in arrays:
        parts.append(a)
        parts.append(gap)
    return np.concatenate(parts[:-1], axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_ray(
    rays,
    highlight=None,
    cmap_ray: str = "Plasma",
    linewidth: float = 6.0,
    ensemble_linewidth: float = 2.0,
    ensemble_alpha: float = 0.25,
    ensemble_stride: int = 1,
    show_ellipsoid: bool = True,
    show_ground_projection: bool = True,
    show_bounce_points: bool = True,
    title: str | None = None,
) -> go.Figure:
    """
    Interactive 3-D ray trajectory plot in true ECEF space (km) over a
    local WGS84 ellipsoid ground patch.

    Parameters
    ----------
    rays : ray-like, or (N,3) ndarray, or a list of either
        One ray, or an ensemble (e.g. a list of RayResult from
        MCScatteringTracer.ensemble()). Each entry needs either a `.path`
        attribute (metres) or to itself be an (N,3) metres array.
    highlight : ray-like or (N,3) ndarray, or None
        The primary ray to draw opaque, thick, and colour-mapped by
        altitude, with launch/end/bounce markers. If None and `rays` is a
        single ray (not a list), that ray is used. If None and `rays` is a
        list, the first element is used. Ensemble members (every entry in
        `rays` other than the one used as `highlight`) are drawn thin,
        translucent, and flat-coloured (orange), matching the convention
        already used in main.py's 2-D ensemble figure.
    ensemble_stride : int
        Plot every Nth ensemble member (speed/clarity for large ensembles).
    show_ellipsoid : bool
        Draw a translucent local WGS84 ellipsoid patch (with a lat/lon
        graticule) as ground context, sized to the highlighted ray's
        physical extent (km) with margin -- NOT its raw lat/lon spread,
        which degenerates into an invisible sliver for a ray that happens
        to travel due north/south or east/west.
    show_ground_projection : bool
        Draw a faint projection of the highlighted ray onto the local
        ellipsoid surface (radial rescale onto the local mean ellipsoid
        radius -- a visual reading aid, not a physics quantity).
    show_bounce_points : bool
        Mark the highlighted ray's ground-bounce points (if it has
        `.bounce_points`) with distinct markers.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Not shown/saved automatically -- call fig.write_html(...) or
        fig.show() yourself.
    """
    # ------------------------------------------------------------------ #
    # 1. Normalize inputs: figure out the ensemble list and the highlight
    # ------------------------------------------------------------------ #
    is_list = isinstance(rays, (list, tuple))
    ray_list = list(rays) if is_list else [rays]

    if highlight is None:
        highlight = ray_list[0]
        ensemble_list = ray_list[1:] if is_list else []
    else:
        ensemble_list = [r for r in ray_list if r is not highlight] if is_list else []

    hi_path_m, hi_bounce_m, hi_landed, hi_escaped = _ray_path_and_meta(highlight)
    if hi_path_m.ndim != 2 or hi_path_m.shape[1] != 3 or hi_path_m.shape[0] < 2:
        raise ValueError("highlighted ray must have a (N>=2, 3) path")

    hi_xyz_km = hi_path_m / 1e3
    hi_alt_km = _altitude_km(hi_path_m)

    hi_lat, hi_lon, _ = xyz_m_to_latlon(hi_path_m.T)

    traces = []

    # ------------------------------------------------------------------ #
    # 2. Local WGS84 ellipsoid ground patch, sized by physical extent (km)
    # ------------------------------------------------------------------ #
    if show_ellipsoid:
        mean_lat_rad = np.deg2rad(hi_lat.mean())
        r_mean_km = np.linalg.norm(latlon_to_xyz_m(hi_lat.mean(), hi_lon.mean(), 0.0)) / 1e3

        lat_span_km = (hi_lat.max() - hi_lat.min()) * np.pi / 180.0 * r_mean_km
        lon_span_km = (hi_lon.max() - hi_lon.min()) * np.pi / 180.0 * r_mean_km * max(np.cos(mean_lat_rad), 1e-3)
        target_km = max(lat_span_km, lon_span_km, 300.0) * 1.3

        lat_half_deg = np.rad2deg(target_km / r_mean_km) / 2.0
        lon_half_deg = np.rad2deg(target_km / (r_mean_km * max(np.cos(mean_lat_rad), 1e-3))) / 2.0
        lat_c, lon_c = hi_lat.mean(), hi_lon.mean()

        Xg, Yg, Zg, LATg, LONg = _ellipsoid_patch_km(
            lat_c - lat_half_deg, lat_c + lat_half_deg,
            lon_c - lon_half_deg, lon_c + lon_half_deg,
        )
        if _HAS_LAND_MASK:
            surfacecolor, colorscale = _land_ocean_surfacecolor(LATg, LONg)
            surface_kwargs = dict(surfacecolor=surfacecolor, colorscale=colorscale,
                                   cmin=0.0, cmax=1.0)
        else:
            # global-land-mask not installed -- fall back to the plain
            # translucent sheet (same as before this feature existed)
            surface_kwargs = dict(colorscale=[[0, _FLAT_GROUND_COLOR], [1, _FLAT_GROUND_COLOR]])
        traces.append(go.Surface(
            x=Xg, y=Yg, z=Zg, showscale=False,
            opacity=0.75, name="Ground (WGS84)", hoverinfo="skip",
            contours=dict(
                x=dict(show=True, color="#7fb3e0", width=1),
                y=dict(show=True, color="#7fb3e0", width=1),
            ),
            **surface_kwargs,
        ))

    # ------------------------------------------------------------------ #
    # 3. Ground projection of the highlighted ray
    # ------------------------------------------------------------------ #
    if show_ground_projection:
        r_ground = np.linalg.norm(latlon_to_xyz_m(hi_lat.mean(), hi_lon.mean(), 0.0))
        r_actual = np.linalg.norm(hi_path_m, axis=1)
        proj_km = hi_path_m * (r_ground / r_actual)[:, None] / 1e3
        traces.append(go.Scatter3d(
            x=proj_km[:, 0], y=proj_km[:, 1], z=proj_km[:, 2],
            mode="lines", line=dict(color="rgba(140,140,140,0.35)", width=3),
            name="Ground projection", hoverinfo="skip", showlegend=False,
        ))

    # ------------------------------------------------------------------ #
    # 4. Ensemble members (thin, translucent, flat orange, one trace)
    # ------------------------------------------------------------------ #
    if ensemble_list:
        ens_paths_km = []
        for ray in ensemble_list[::max(1, ensemble_stride)]:
            path_m, _, _, _ = _ray_path_and_meta(ray)
            if path_m.ndim == 2 and path_m.shape[0] >= 2:
                ens_paths_km.append(path_m / 1e3)
        ens_xyz = _concat_with_gaps(ens_paths_km)
        traces.append(go.Scatter3d(
            x=ens_xyz[:, 0], y=ens_xyz[:, 1], z=ens_xyz[:, 2],
            mode="lines",
            line=dict(color=f"rgba(255,140,0,{ensemble_alpha})", width=ensemble_linewidth),
            name=f"Ensemble ({len(ens_paths_km)} rays)", hoverinfo="skip",
        ))

    # ------------------------------------------------------------------ #
    # 5. Highlighted ray, colour-mapped by altitude
    # ------------------------------------------------------------------ #
    traces.append(go.Scatter3d(
        x=hi_xyz_km[:, 0], y=hi_xyz_km[:, 1], z=hi_xyz_km[:, 2],
        mode="lines",
        line=dict(
            color=hi_alt_km, colorscale=cmap_ray, width=linewidth,
            cmin=float(hi_alt_km.min()), cmax=float(hi_alt_km.max()),
            showscale=True, colorbar=dict(title="Altitude (km)", x=1.02),
        ),
        name="Ray", hovertemplate="alt %{customdata:.1f} km<extra></extra>",
        customdata=hi_alt_km,
    ))

    # ------------------------------------------------------------------ #
    # 6. Launch / end / bounce markers (highlighted ray only)
    # ------------------------------------------------------------------ #
    traces.append(go.Scatter3d(
        x=[hi_xyz_km[0, 0]], y=[hi_xyz_km[0, 1]], z=[hi_xyz_km[0, 2]],
        mode="markers", marker=dict(size=6, color="#00ff88", line=dict(color="white", width=1)),
        name="Launch",
    ))

    if hi_escaped:
        end_color, end_label = "#33ccff", "Escaped"
    elif hi_landed:
        end_color, end_label = "#ff3355", "Landed"
    else:
        end_color, end_label = "#aaaaaa", "End (budget/stall)"
    traces.append(go.Scatter3d(
        x=[hi_xyz_km[-1, 0]], y=[hi_xyz_km[-1, 1]], z=[hi_xyz_km[-1, 2]],
        mode="markers", marker=dict(size=6, color=end_color, line=dict(color="white", width=1)),
        name=end_label,
    ))

    if show_bounce_points and hi_bounce_m is not None:
        # if the ray landed, the LAST bounce point is the same point as the
        # "Landed" marker above -- drop it here so the diamond (drawn after,
        # on top) doesn't visually hide the landed marker underneath it.
        bounce_to_show = hi_bounce_m[:-1] if hi_landed else hi_bounce_m
        if len(bounce_to_show) > 0:
            bounce_km = bounce_to_show / 1e3
            traces.append(go.Scatter3d(
                x=bounce_km[:, 0], y=bounce_km[:, 1], z=bounce_km[:, 2],
                mode="markers",
                marker=dict(size=6, color="#ffdd33", symbol="diamond", line=dict(color="#553c00", width=1)),
                name="Ground bounce",
            ))

    # ------------------------------------------------------------------ #
    # 7. Layout: dark theme, true (undistorted) aspect ratio
    # ------------------------------------------------------------------ #
    if title is None:
        lat0_s = f"{abs(hi_lat[0]):.2f}°{'N' if hi_lat[0] >= 0 else 'S'}"
        lon0_s = f"{abs(hi_lon[0]):.2f}°{'E' if hi_lon[0] >= 0 else 'W'}"
        ens_s = f"  (+{len(ensemble_list)} ensemble)" if ensemble_list else ""
        title = f"Ray trajectory — launch: {lat0_s}, {lon0_s}{ens_s}"

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d0d",
        title=dict(text=title, font=dict(color="#dddddd", size=16)),
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(20,20,20,0.6)", bordercolor="#444444", borderwidth=1,
                    font=dict(color="#cccccc")),
        scene=dict(
            aspectmode="data",   # true, undistorted 3-D proportions
            xaxis=dict(title="ECEF X (km)", backgroundcolor="#141414",
                       gridcolor="#333333", color="#aaaaaa"),
            yaxis=dict(title="ECEF Y (km)", backgroundcolor="#141414",
                       gridcolor="#333333", color="#aaaaaa"),
            zaxis=dict(title="ECEF Z (km)", backgroundcolor="#141414",
                       gridcolor="#333333", color="#aaaaaa"),
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig
