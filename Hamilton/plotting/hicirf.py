"""
hicirf.py -- HiCIRF information drawn on PHaRLAP-style figures: where the phase screens of a leg are and how much
each contributes, a box with the leg's channel parameters, and the delay-Doppler scattering function.

Inputs are Hamilton's own objects, duck-typed: core.records.RaySamples (the traced leg), stochastic.screens.ScreenSet
(its screens; `z` is the group-path position, `sigma2` the phase variance each screen adds) and
core.records.ChannelParams (the leg's channel, stochastic.channel.channel_parameters).
"""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import MaxNLocator

from plotting.iono_slice import ground_range_height_km, slice_xy

MONO = ["Liberation Mono", "DejaVu Sans Mono"]      # MATLAB's 'Monospaced'


def screen_track(samples, screens, origin_ecef_m):
    """(ground range km, height km, sigma2 rad^2) of every screen of a leg. Screens sit at samples of the leg; each
    is found from its group-path position z (samples.Pg is increasing along the leg)."""
    Pg = np.asarray(samples.Pg, dtype=float)
    z = np.asarray(screens.z, dtype=float)
    idx = np.clip(np.searchsorted(Pg, z), 0, Pg.size - 1)
    prev = np.clip(idx - 1, 0, Pg.size - 1)
    idx = np.where(np.abs(Pg[prev] - z) < np.abs(Pg[idx] - z), prev, idx)
    gr, h = ground_range_height_km(np.asarray(samples.r, dtype=float)[idx], origin_ecef_m)
    return gr, h, np.asarray(screens.sigma2, dtype=float)


def draw_screens(ax, gr_km, h_km, sigma2, *, max_markers: int = 60, color="k", facecolor="none"):
    """Phase screens on a slice made by plot_ray_iono_slice: one marker per screen, or per group of neighbouring
    screens when there are more than max_markers (the group's sigma2 summed). Hollow circles, area growing
    with sigma2, so the ray stays visible."""
    geo = getattr(ax, "pharlap_geometry", None)
    if geo is None:
        raise ValueError("draw the slice first (plot_ray_iono_slice)")
    gr, h, s2 = (np.asarray(v, dtype=float) for v in (gr_km, h_km, sigma2))
    if gr.size == 0:
        return None
    k = int(np.ceil(gr.size / max_markers))
    if k > 1:
        starts = np.arange(0, gr.size, k)
        gr = np.add.reduceat(gr, starts) / np.diff(np.append(starts, gr.size))
        h = np.add.reduceat(h, starts) / np.diff(np.append(starts, h.size))
        s2 = np.add.reduceat(s2, starts)
    inside = ((gr >= geo["start_range"]) & (gr <= geo["end_range"]) & (h >= geo["start_height"])
              & (h <= geo["end_height"]))
    X, Y = slice_xy(gr[inside], h[inside], geo["max_range"], geo["start_range"])
    peak = s2.max() if s2.size and s2.max() > 0 else 1.0
    sizes = 6.0 + 54.0 * (s2[inside] / peak)
    return ax.scatter(X, Y, s=sizes, facecolors=facecolor, edgecolors=color, linewidths=0.8, zorder=5)


def channel_lines(P, *, absorption_db=None) -> list[str]:
    """The channel parameters of a leg as compact fixed-width lines, two quantities per line."""
    f_coh = P.w_coh / (2.0 * np.pi)
    V = float(np.linalg.norm(P.V_app)) if P.V_app is not None else np.nan
    rows = [(f"σφ² {P.sigma_phi2:.3g} rad²", f"S4 {P.S4:.3f}"),
            (f"K {P.rician_K_db:.1f} dB", f"τo {_si(P.tau_o, 's')}"),
            (f"<τ> {_si(P.tau_mean, 's')}", f"f_coh {_si(f_coh, 'Hz')}"),
            (f"lx {_si(P.lx, 'm')}", f"ly {_si(P.ly, 'm')}"),
            (f"|V| {V:.3g} m/s", f"Abs {absorption_db:.2f} dB" if absorption_db is not None else "")]
    width = max(len(r[0]) for r in rows)
    return [f"{a.ljust(width)}  {b}".rstrip() for a, b in rows]


def _si(v: float, unit: str) -> str:
    """3 significant digits with an SI prefix (T, G, M, k, -, m, µ, n, p)."""
    v = float(v)
    if not np.isfinite(v) or v == 0.0:
        return f"{v:.3g} {unit}"
    for scale, prefix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"),
                          (1e-9, "n"), (1e-12, "p")):
        if abs(v) >= scale:
            return f"{v / scale:.3g} {prefix}{unit}"
    return f"{v:.3g} {unit}"


def draw_text_box(fig_or_ax, lines, *, x: float, y: float, ha: str = "left", va: str = "top", color="k",
                  fontsize: float = 8, title=None, box: bool = True):
    """A MATLAB-style annotation: monospaced, lines left-aligned, in normalised units of the figure or axes given,
    optionally in a white box."""
    text = "\n".join(([title] if title else []) + list(lines))
    transform = fig_or_ax.transAxes if hasattr(fig_or_ax, "transAxes") else fig_or_ax.transFigure
    bbox = dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor=(0.15, 0.15, 0.15), linewidth=0.5) \
        if box else None
    return fig_or_ax.text(x, y, text, transform=transform, ha=ha, va=va, multialignment="left",
                          fontsize=fontsize, family=MONO, color=color, bbox=bbox)


def plot_scattering_function(P, grid=None, *, ax=None, floor_db: float = -40.0, cmap=None, title=None):
    """
    The expected delay-Doppler power S(f_D, tau) of a leg (spec §11, HiCIRF's scattering function) as a MATLAB
    pcolor: Doppler [Hz] on x, delay relative to the mode [us] on y, dB relative to the peak (clipped at floor_db),
    with a colourbar. grid defaults to stochastic.gpsd.default_grid(P). Zooms to where the power is above the floor.
    """
    import matplotlib.pyplot as plt
    from plotting.style import get_cmap
    from stochastic import gpsd

    grid = gpsd.default_grid(P) if grid is None else grid
    S = gpsd.scattering_function(P, grid)                    # (Nt, Nk)
    f = np.fft.fftshift(grid.f_doppler)
    S = np.fft.fftshift(S, axes=0)
    tau_us = grid.tau * 1e6
    S_db = 10.0 * np.log10(np.maximum(S / S.max(), 10.0 ** (floor_db / 10.0)))
    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 6.3))
    mesh = ax.pcolormesh(f, tau_us, S_db.T, shading="nearest", cmap=get_cmap(cmap), vmin=floor_db, vmax=0.0)
    above = S_db > floor_db + 1e-9
    if np.any(above):
        fi, ti = np.nonzero(above)
        df = f[1] - f[0]
        dtau = tau_us[1] - tau_us[0] if tau_us.size > 1 else 1.0
        ax.set_xlim(f[fi.min()] - 2 * df, f[fi.max()] + 2 * df)
        ax.set_ylim(tau_us[ti.min()] - dtau, tau_us[ti.max()] + dtau)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Doppler (Hz)")
    ax.set_ylabel("Delay (µs)")
    cb = ax.figure.colorbar(mesh, ax=ax)
    cb.set_label("Relative Power (dB)")
    cb.ax.tick_params(direction="in")
    ax.set_title(title or "HiCIRF Scattering Function")
    ax.grid(False)
    return ax
