"""
PHaRLAP-style plotting (plotting/): the port of plot_ray_iono_slice.m (its tick rules, curved-Earth geometry, input
checks and figure layout), the MuellerMap of startup.m, the slice grid from a density, the HiCIRF screen positions,
the 3-D figure's MATLAB clipping, and the composed figures through api.Link on a small patch.
"""

from __future__ import annotations

from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plotting import iono_slice as isl  # noqa: E402
from plotting.style import MUELLER_MAP  # noqa: E402


def test_tick_rules_follow_plot_ray_iono_slice():
    assert isl.range_tick_step(3000.0) == 250.0          # 375 is equally far from 250 and 500: MATLAB takes the first
    assert isl.range_tick_step(7000.0) == 1000.0
    assert isl.range_tick_step(1000.0) == 100.0
    assert isl.height_ticks(0.0, 400.0, 3000.0) == (9, 50.0)
    assert isl.height_ticks(60.0, 400.0, 3000.0) == (9, 50.0)           # 8 ticks, then one more to reach 400
    n, step = isl.height_ticks(0.0, 400.0, 7000.0)
    assert (n - 1) * step <= 400.0 and n >= 2


def test_curved_earth_geometry_and_layout():
    x, y = isl.slice_xy([1500.0, 0.0], [300.0, 0.0], 3000.0)
    assert abs(x[0]) < 1e-9 and abs(y[0] - (6371.0 + 300.0)) < 1e-9          # mid-range is straight up
    th = -1500.0 / 6371.0
    assert abs(x[1] - 6371.0 * np.sin(th)) < 1e-9 and abs(y[1] - 6371.0 * np.cos(th)) < 1e-9
    grid = np.zeros((int(400 / 4) + 1, int(3000 / 10) + 1))
    ray = {"ground_range": np.linspace(0.0, 2000.0, 50), "height": 300.0 * np.sin(np.linspace(0.0, np.pi, 50))}
    ax, handles = isl.plot_ray_iono_slice(grid, 0.0, 3000.0, 10.0, 0.0, 400.0, 4.0, [ray], color="w")
    fig = ax.figure
    H = fig.get_figheight() * fig.dpi
    assert abs(fig.get_figwidth() * fig.dpi - 1824.0) < 1e-6
    assert abs(ax.get_position().y0 * H - 100.0) < 1.0                       # 100 px reserved for the colourbar
    assert len(handles) == 1
    X, Y = handles[0].get_data()
    assert np.isclose(np.nanmax(np.hypot(X, Y)), 6371.0 + 300.0, atol=0.5)  # pchip keeps the apex height
    ax2, more = isl.plot_ray_iono_slice(grid, 0.0, 3000.0, 10.0, 0.0, 400.0, 4.0, [ray], ax=ax, color="r")
    assert ax2 is ax and len(more) == 1 and len(ax.collections) == 1         # second call adds only the ray
    plt.close(fig)
    try:
        isl.plot_ray_iono_slice(grid[:-1], 0.0, 3000.0, 10.0, 0.0, 400.0, 4.0, [])
    except ValueError:
        pass
    else:
        raise AssertionError("a grid inconsistent with the heights must raise, as in PHaRLAP")


def test_mueller_map_matches_startup_m():
    c = MUELLER_MAP(np.array([0, 84, 170, 255]))[:, :3]
    assert np.allclose(c, [[0, 0, 0.5], [0, 1, 1], [1, 1, 0], [1, 0, 0]])
    assert MUELLER_MAP.N == 256


def test_slice_grid_is_the_plasma_frequency_along_the_bearing():
    ne = lambda lat, lon, h: 1e12 * np.exp(-((np.asarray(h) - 300.0) / 50.0) ** 2)   # noqa: E731
    pf = isl.iono_slice_grid(ne, 60.0, 10.0, 45.0, 0.0, 1000.0, 100.0, 100.0, 500.0, 100.0)
    assert pf.shape == (5, 11)
    assert np.allclose(pf[2], np.sqrt(80.6164e-6 * 1e12 * 1e-6))                 # 8.98 MHz at the peak
    lat, lon = isl.destination_latlon(60.0, 10.0, 0.0, 111.195)                  # ~1 deg north
    assert abs(lat - 61.0) < 1e-3 and abs(lon - 10.0) < 1e-9


def test_ground_range_of_a_path_is_the_great_circle_distance():
    from core.geometry import R_MEAN, latlon_to_xyz_m
    o = latlon_to_xyz_m(60.0, 10.0, 0.0)
    p = np.stack([latlon_to_xyz_m(60.0 + d, 10.0, 200e3) for d in (0.0, 1.0, 5.0)])
    gr, h = isl.ground_range_height_km(p, o)
    assert np.allclose(h, 200.0, atol=1e-6)
    assert abs(gr[0]) < 1.0 and 1.0 < gr[1] / (R_MEAN / 1e3 * np.deg2rad(1.0)) < 1.01


def test_3d_figure_clips_at_zlim_like_matlab():
    from plotting.figures import rays_3d_figure
    from core.geometry import latlon_to_xyz_m
    path = np.stack([latlon_to_xyz_m(60.0 + 0.01 * i, 10.0, 1e3 * (10.0 * i + 5.0)) for i in range(80)])
    fig = rays_3d_figure([path], modes=["X"])
    ax = fig.axes[0]
    z = np.asarray(ax.lines[0].get_data_3d()[2], dtype=float)
    assert np.nanmax(z) <= 500.0 and np.isnan(z).sum() == 30             # 5..495 km kept, 505..795 km masked
    assert ax.get_legend().get_texts()[0].get_text() == "X Mode"
    plt.close(fig)


def test_link_figures_with_the_hicirf_channel():
    import api
    link = api.Link(57.0, 10.0, frequency_mhz=8.0, dt=datetime(2024, 6, 21, 14, 0, 0), mode="O",
                    max_path_km=1500.0, max_nhops=1, n_ang=4, dang_deg=2.0, igrf_n_ang=3, igrf_n_r=3, verbose=False)
    ray = link.trace(28.0)
    fan = [link.trace(e) for e in (15.0, 40.0)]
    fig = link.plot(fan, highlight=ray)
    ax = fig.axes[0]
    texts = " ".join(t.get_text() for t in fig.texts)
    assert "HiCIRF, red ray" in texts and "S4" in texts and "nan" not in texts and "n/a" not in texts
    assert "UTC: 20240621|140000" in texts
    assert len(ax.collections) == 2                                    # the slice and the screen markers
    assert ax.collections[0].get_array().max() > 1.0                  # a real plasma-frequency background
    plt.close(fig)
    fig = ray.plot()
    assert fig.axes[0].collections[0].get_array().max() == 0.0        # no ionosphere: PHaRLAP's empty grid
    plt.close(fig)
    fig = link.plot_scattering_function(ray)
    assert fig.axes[0].get_xlabel() == "Doppler (Hz)"
    plt.close(fig)
