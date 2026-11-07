"""
style.py -- MATLAB look for matplotlib figures, so Hamilton's plots read like PHaRLAP's.

Two flavours:

* startup=True (default): the conventions of the startup.m used with PHaRLAP in this group (as documented in
  pharlap/src/matlab/examples/generate_plots.m): MuellerMap colormap, all text 13 pt bold, default line width 2,
  marker size 10, grid on.
* startup=False: MATLAB's factory defaults: parula colormap, 10 pt normal text, line width 0.5, marker size 6, no grid.

Both: Arial (MATLAB's Helvetica on Linux), ticks pointing in and mirrored on all four sides (MATLAB's box on),
MATLAB's default line colour order, square legends, white figures, 150 dpi when saving (print -r150).
"""

from __future__ import annotations

import contextlib

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

def _mueller_map() -> ListedColormap:
    """MuellerMap exactly as built in startup.m / generate_plots.m: dark blue -> cyan -> yellow -> red,
    interp1(nodes = [1 85 171 256], colours, 1:256, 'linear')."""
    nodes = np.array([1.0, 85.0, 171.0, 256.0])
    colours = np.array([[0.0, 0.0, 0.5], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    idx = np.arange(1, 257, dtype=float)
    table = np.stack([np.interp(idx, nodes, colours[:, c]) for c in range(3)], axis=1)
    return ListedColormap(table, name="MuellerMap")


def _parula_like() -> LinearSegmentedColormap:
    """An approximation of MATLAB's default colormap (parula), from nine anchor colours spaced evenly along it.
    Close enough to read like a stock MATLAB figure; not MathWorks' table."""
    anchors = [(0.2422, 0.1504, 0.6603), (0.2797, 0.3010, 0.9425), (0.1855, 0.5013, 0.9775),
               (0.0904, 0.6400, 0.8728), (0.0718, 0.7432, 0.7194), (0.3110, 0.7980, 0.5050),
               (0.7004, 0.7781, 0.2367), (0.9932, 0.7425, 0.2390), (0.9769, 0.9839, 0.0805)]
    return LinearSegmentedColormap.from_list("parula_like", anchors, N=256)


MUELLER_MAP = _mueller_map()
PARULA = _parula_like()

# MATLAB's default axes ColorOrder (the 'lines' colours).
MATLAB_COLOR_ORDER = [(0.0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.9290, 0.6940, 0.1250),
                      (0.4940, 0.1840, 0.5560), (0.4660, 0.6740, 0.1880), (0.3010, 0.7450, 0.9330),
                      (0.6350, 0.0780, 0.1840)]

# MATLAB's single-letter colours ('b', 'r', 'g', ...) are the pure RGB primaries; matplotlib's 'g' is not.
MATLAB_LETTER = {"b": (0.0, 0.0, 1.0), "r": (1.0, 0.0, 0.0), "g": (0.0, 1.0, 0.0), "k": (0.0, 0.0, 0.0),
                 "w": (1.0, 1.0, 1.0), "c": (0.0, 1.0, 1.0), "m": (1.0, 0.0, 1.0), "y": (1.0, 1.0, 0.0)}

_CMAPS = {"muellermap": MUELLER_MAP, "mueller": MUELLER_MAP, "parula": PARULA}


def get_cmap(cmap=None, *, startup: bool = True):
    """A colormap by name ('MuellerMap', 'parula', or any matplotlib name) or object; None gives the default of
    the chosen flavour (MuellerMap with startup conventions, parula without)."""
    if cmap is None:
        return MUELLER_MAP if startup else PARULA
    if isinstance(cmap, str):
        return _CMAPS.get(cmap.lower()) or mpl.colormaps[cmap]
    return cmap


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

def matlab_rc(startup: bool = True) -> dict:
    """rcParams for a MATLAB-looking figure (see the module docstring)."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 13 if startup else 10,
        "font.weight": "bold" if startup else "normal",
        "axes.labelweight": "bold" if startup else "normal",
        "axes.titleweight": "bold" if startup else "normal",
        "axes.titlesize": 13 if startup else 11,
        "axes.labelsize": 13 if startup else 11,
        "axes.linewidth": 0.5,
        "axes.edgecolor": (0.15, 0.15, 0.15),
        "axes.grid": bool(startup),
        "axes.prop_cycle": mpl.cycler(color=MATLAB_COLOR_ORDER),
        "axes.facecolor": "white",
        "grid.color": (0.15, 0.15, 0.15),
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0 if startup else 0.5,
        "lines.markersize": 10 if startup else 6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.color": (0.15, 0.15, 0.15),
        "ytick.color": (0.15, 0.15, 0.15),
        "legend.fancybox": False,
        "legend.edgecolor": (0.15, 0.15, 0.15),
        "legend.framealpha": 1.0,
        "legend.shadow": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 150,
        "image.cmap": "viridis",           # set per figure from get_cmap(); rcParams only accept registered names
        "mathtext.default": "regular",
    }


@contextlib.contextmanager
def matlab_style(startup: bool = True):
    """Context manager: everything drawn inside looks like a MATLAB figure."""
    with mpl.rc_context(matlab_rc(startup)):
        yield
