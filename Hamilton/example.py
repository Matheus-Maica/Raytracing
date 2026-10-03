from datetime import datetime
from api import Link

link = Link(
    lat_deg=57.0, lon_deg=10.0,
    frequency_mhz=8.0,
    dt=datetime(2024, 6, 21, 14, 0, 0),
    mode="O",
)

ray = link.trace(elevation_deg=28.0)
print()
print(ray)

budget = link.link_budget(elevation_deg=28.0)
print()
print(budget)

fig = ray.plot(title="8 MHz O-mode skip from southern Sweden")
fig.write_html("example_ray.html")
print("\nSaved interactive plot to example_ray.html")