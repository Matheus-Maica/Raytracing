# legacy — unused code kept for reference

Nothing in the project imports this folder. It can be deleted once nobody needs it.

| File | What it is |
|---|---|
| `test_iono.py` | An early synthetic ionosphere: Chapman layers on a uniform spherical (r, theta, phi) grid over a patch, with trilinear interpolation. Superseded by `geophysics/ionosphere.py` (E-CHAIM + IRI-2020 on a WGS84 grid). Despite the name it is not a test; it has its own `__main__` demo (`python -m legacy.test_iono`). |
