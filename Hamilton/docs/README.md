# docs — documentation and reference material

| File | What it is |
|---|---|
| `algorithm.md` | How the ray tracer works: physics, environment models, numerical methods, software architecture, validation, known limitations. Start here. |
| `implementation_plan.md` | The phased implementation plan against the engineering spec, with the decisions D1-D10. |
| `implementation_status.md` | What is done per phase, with test counts, and a dated log of later work. |
| `spec_deviations.md` | Every place the code departs from the engineering spec or HiCIRF, and why (numbered). |
| `needed_from_user.md` | What the code still needs from the user (data, decisions), what stands in for it, and what was resolved. |
| `heelis.F90.txt` | NCAR TIE-GCM's `heelis.F90`, the source ported in `geophysics/plasma_drift.py`. |
| `1972Barrick_RST.pdf` | Barrick (1972), "Remote sensing of sea state by radar", ch. 12 of *Remote Sensing of the Troposphere* (V. E. Derr, ed.): the second-order sea echo of `echoes/second_order_bragg.py`. Local copy only (copyright; `.gitignore`). |
| `Journal of Geophysical Research ... Fentzke ... sporadic.pdf` | Fentzke & Janches (2008), J. Geophys. Res. 113, A03304, doi:10.1029/2007JA012531: the meteor source weights of `echoes/meteor.py`. Local copy only (copyright; `.gitignore`). |

Module-level documentation lives next to the code: each code folder has its own `README.md` (see the root
`README.md` for the map).
