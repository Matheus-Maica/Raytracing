"""
absorption.py — non-deviative HF absorption via the imaginary part of the
Appleton-Hartree refractive index.

Approach
--------
The ray itself is always traced loss-free (the real-X Appleton-Hartree root
Ray._haselgrove_physical_rhs already solves). Absorption is a separate,
post-hoc integral along that already-computed real ray: at each point we
re-solve the *same* quadratic root with a collisionally-damped, complex X,

    X_complex = X / (1 - i * Z),      Z = nu_eff(h) / omega

(Budden 1961; standard magneto-ionic collisional damping term), and take
Im(n) from the resulting complex p = n^2. This matches how PHaRLAP's
non-deviative ("George"-style) absorption option works: the trajectory is
unaffected by loss, and absorption is accumulated along it afterward.

Reusing Ray._appleton_hartree_p(..., allow_complex=True) for this means the
lossless ray-tracing algebra (a, b, c, the O/X mode branch selection) is
never duplicated — only X becomes complex.

Collision-frequency profile
----------------------------
IRI-2020 does not provide electron-neutral collision frequency, and per
project decision we are not pulling in a neutral-atmosphere model (e.g.
MSIS) to derive it from first principles. `collision_frequency_profile`
below is therefore a simple, height-only empirical exponential profile of
the classic textbook D/E-region form

    nu_eff(h) = nu_0 * exp(-h / H_scale)      (h in km)

with nu_0 and H_scale chosen to be the right order of magnitude and general
shape for the quiet-daytime D/E region (nu_eff ~ 1e7 /s around 70 km,
dropping to ~1e5-1e6 /s by 90-100 km, negligible above ~110 km) as commonly
illustrated in ionospheric radio texts (e.g. Davies, "Ionospheric Radio",
1990). Treat these specific coefficients as an illustrative placeholder,
not a citation-grade fit — swap in published coefficients (via `model=`)
before using absorption_db_total for anything you intend to publish.

Public API
----------
    collision_frequency_profile(alt_km, model="default") -> nu_eff [s^-1]
    absorption_rate(ray, r_vec, u_vec, model="default")   -> alpha [nepers/m], power convention
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Collision-frequency profiles
# ---------------------------------------------------------------------------

def _default_profile(alt_km: float) -> float:
    nu_0     = 1.0e7    # s^-1, reference collision frequency at h_ref
    h_ref    = 70.0     # km
    H_scale  = 6.0       # km, e-folding height
    if alt_km <= 0.0 or alt_km > 130.0:
        return 0.0
    return nu_0 * np.exp(-(alt_km - h_ref) / H_scale)


_PROFILES = {
    "default": _default_profile,
}


def collision_frequency_profile(alt_km: float, model: str = "default") -> float:
    """Effective electron-neutral collision frequency [s^-1] at a given altitude [km]."""
    if callable(model):
        return float(model(alt_km))
    try:
        fn = _PROFILES[model]
    except KeyError:
        raise ValueError(
            f"unknown collision-frequency model {model!r}; "
            f"available: {list(_PROFILES)}, or pass a callable(alt_km) -> nu_eff"
        )
    return float(fn(alt_km))


# ---------------------------------------------------------------------------
# Absorption rate
# ---------------------------------------------------------------------------

def absorption_rate(ray, r_vec: np.ndarray, u_vec: np.ndarray, model: str = "default") -> float:
    """
    Power-absorption coefficient [nepers/m] at (r_vec, u_vec) for the given
    Ray (uses ray.electron_density, ray.X_factor, ray.Y_vec, ray.omega,
    ray.MODE, ray._appleton_hartree_p). Integrating this along ds gives
    `A`, with one-way power loss in dB = 8.686 * A (the standard
    ionospheric-absorption reporting convention).

    Returns 0.0 outside the ionosphere (Ne <= 0).
    """
    from geometry import xyz_m_to_latlon
    from constants import C

    x, y, z = r_vec
    Ne = ray.electron_density(x, y, z)
    if Ne <= 0.0:
        return 0.0

    X = Ne * ray.X_factor
    Y_vec = ray.Y_vec(x, y, z)
    Y = np.linalg.norm(Y_vec)
    v = Y_vec / Y if Y > 0.0 else np.zeros(3)
    Ydotu = np.dot(v, np.asarray(u_vec, dtype=float))

    _, _, alt_m = xyz_m_to_latlon(np.array([x, y, z], dtype=float))
    nu_eff = collision_frequency_profile(alt_m / 1e3, model=model)
    if nu_eff <= 0.0:
        return 0.0

    Z = nu_eff / ray.omega
    X_complex = X / (1.0 - 1j * Z)

    _, _, _, p, _ = ray._appleton_hartree_p(X_complex, Y, Ydotu, ray.MODE, allow_complex=True)
    # -sqrt(p), not sqrt(p): np.emath.sqrt returns the principal branch
    # (Im >= 0 for the arguments it sees here), but a damped wave needs
    # Im(n) > 0 for OUR e^{-i(kz-wt)}, k=k0*n convention -- so the
    # physically-damped root is the OTHER branch. Verified against the
    # corrected _appleton_hartree_p (see main.py) across a wide sweep of
    # X, Y, Ydotu, collision frequency, and both MODEs: Im(sqrt(p)) comes
    # out strictly negative throughout, Im(-sqrt(p)) strictly positive.
    n_complex = -np.emath.sqrt(p)

    k0 = ray.omega / C
    return float(2.0 * k0 * np.imag(n_complex))   # power convention: P(s) = P0 * exp(-integral)
