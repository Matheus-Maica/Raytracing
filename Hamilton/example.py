from datetime import datetime

import numpy as np

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

print()
print(link.channel(ray))              # HiCIRF stochastic channel of the same leg

# PHaRLAP-style range-height figure: a fan of rays in white, this ray in red with its HiCIRF channel
fan = link.sweep_elevation(np.arange(5.0, 50.0, 4.0))
fig = link.plot(fan, highlight=ray, title="8 MHz O-mode skip from southern Sweden")
fig.savefig("example_ray.png")
print("\nSaved example_ray.png")
