# plotting — PHaRLAP-style figures with the HiCIRF channel

Hamilton's figures, drawn to look like PHaRLAP's MATLAB output (`pharlap/src/matlab/`), with the HiCIRF stochastic
channel added. matplotlib throughout; the old interactive Plotly scene is kept as an extra. Depends on `core/` and,
for the scattering function, `stochastic/`. `api.py` is the usual way in:

```python
fig = link.plot(fan, highlight=ray)            # range-height slice + HiCIRF channel of the red ray
fig.savefig("rays.png")                        # 150 dpi, as PHaRLAP scripts print -r150
link.plot_3d(rays).savefig("rays_3d.png")      # PHaRLAP's 3-D lat/lon/height figure
link.plot_scattering_function(ray)             # HiCIRF delay-Doppler scattering function of the ray's leg
ray.plot()                                     # one ray, no ionosphere (PHaRLAP's empty-grid plot)
link.plot_html(rays).write_html("rays.html")   # the old interactive 3-D scene
```

| File | What it does |
|---|---|
| `iono_slice.py` | `plot_ray_iono_slice`: line-by-line port of PHaRLAP's `plot_ray_iono_slice.m`; the slice grid and ray conversions. |
| `rays_3d.py` | PHaRLAP's 3-D ray figure (`plot3` of lat, lon, height as in the `raytrace_3d` examples). |
| `hicirf.py` | HiCIRF overlays: phase-screen markers, channel-parameter box, scattering function. |
| `figures.py` | The composed figures behind `api.Link.plot`, `plot_3d`, `plot_scattering_function` and `RayPath.plot`. |
| `style.py` | MATLAB look: MuellerMap and parula colormaps, rcParams, `matlab_style()` context. |
| `plotly_3d.py` | The former `raytracing/visualization.py`: interactive Plotly 3-D scene in ECEF over the ellipsoid. |

## The look

`style.py` reproduces the MATLAB environment the PHaRLAP scripts run in. By default (`startup=True`) that is this
group's `startup.m` conventions, as documented in `pharlap/src/matlab/examples/generate_plots.m`: the MuellerMap
colormap (dark blue, cyan, yellow, red: built exactly as there), 13 pt bold text, line width 2, marker size 10, grid
on. `startup=False` gives MATLAB's factory defaults (parula, 10 pt normal text). Both use Arial, ticks pointing in on
all four sides, MATLAB's colour order and square legends. Every figure function takes `startup=` and `cmap=`
(`'MuellerMap'`, `'parula'`, or any matplotlib name).

## iono_slice.py

`plot_ray_iono_slice(iono_grid, start_range, end_range, range_inc, start_height, end_height, height_inc, ray, ...)`
takes PHaRLAP's arguments (a plasma-frequency grid, heights x ground ranges, in MHz; rays with `ground_range` and
`height` in km) and returns `(ax, ray_handles)`. Kept from the MATLAB code: Earth radius 6371 km in the image, the
ground-range tick step (nearest of 100/150/200/250/500/1000 km to a range/8, first on a tie), the altitude tick rule,
tick lengths, label positions and the tilted 'Altitude (km)' label, flat-shaded pcolor, the 0.1 km pchip resampling
of the rays and their masking outside the image, and the figure layout (95% of a 1920 px screen wide, axes 98% of it,
100 px under the slice for the horizontal 'Plasma Frequency (MHz)' colourbar, 15 px above). Deliberate differences:
`ax=` draws into an existing axes (calling it again only adds rays), plot keywords are matplotlib's, a constant grid
takes the bottom colour, and the altitude label is pushed out only if bold tick labels would overlap it.

Helpers: `iono_slice_grid(density_geodetic, origin_lat, origin_lon, bearing_deg, ...)` samples a density
(e.g. `Ionosphere.electron_density_geodetic`) along a great circle into PHaRLAP's grid (f_p = sqrt(80.6164e-6 Ne[cm^-3])
MHz); `rays_from_paths(rays, origin)` gives PHaRLAP rays from ECEF paths (ground range = great-circle distance of the
sub-point from the origin, height above WGS84); `slice_xy` maps (ground range, height) into the image.

## rays_3d.py

`plot_rays_3d(rays, modes=None, zlim=(0, 500))`: O-mode rays as blue dots, X-mode as red dots, field-free as a green
line, legend 'O Mode' / 'X Mode' / 'No Mag-field', axis labels and grid as in `ray_test_3d_sp.m`, MATLAB's default
3-D view (azimuth -37.5, elevation 30). Heights outside `zlim` are masked, as MATLAB clips them.

## hicirf.py

- `screen_track(samples, screens, origin)`: ground range, height and phase variance sigma_i^2 of every phase screen of
  a leg (`stochastic.leg_screens.screens_for_leg`).
- `draw_screens(ax, ...)`: hollow circles on the ray, area growing with sigma_i^2; neighbouring screens are grouped
  (sigma^2 summed) above 60 markers.
- `channel_lines(P)`: the leg's `ChannelParams` as text: sigma_phi^2, S4, Rician K = A_c^2 / (2 A_d^2), tau_o,
  mean excess delay <tau>, coherence bandwidth omega_coh / 2 pi, l_x, l_y, apparent velocity |V|, absorption.
- `plot_scattering_function(P, grid=None, floor_db=-40)`: S(f_D, tau) from `stochastic.gpsd`, Doppler on x, delay
  (relative to the mode) in µs on y, dB relative to the peak, zoomed to where it is above the floor.

## figures.py

`ray_slice_figure(rays, highlight=None, density_geodetic=None, ...)` builds the range-height figure the way this
group's PHaRLAP scripts do (`examples/ray_test_3d.m`, Plot Set 2): the fan in thin white, the highlighted ray in thick
red labelled with its Tx elevation, a monospaced UTC / frequency / bearing note. Without a highlight the rays are
drawn as in `ray_test1.m` (near-white, width 2). The note and the HiCIRF box sit in the band under the slice, either
side of the colourbar, so they never cover it. Extents default from the rays (end range rounded up, 400 km of
altitude unless a landed ray reflects higher). `rays_3d_figure` and `scattering_function_figure` wrap the other two
plots in the MATLAB style.

`api.Link.plot` passes the Link's ionosphere and, for the highlighted ray, its phase screens and channel, built from
the ray's own samples (`RayPath.samples`, the same as `Link.channel(ray)`). They use the Link's irregularity inputs
or models, so the HiCIRF numbers are only as real as those inputs (placeholders by default).
