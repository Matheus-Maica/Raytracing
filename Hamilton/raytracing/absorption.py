"""
absorption.py -- HF absorption along a loss-free ray (spec §7; P0.2, Phase 2).

The ray is traced loss-free (main.py). Absorption is a post-hoc integral along it. Two integrands,
one per collision convention (spec §7.2):

  * nu_eff (complex Appleton-Hartree, eq. 4.1, U = 1 - i nu_eff/omega, Budden 1961):
        alpha_amp = k0 |Im n| cos(alpha_rw)                                  [Np/m]

  * nu_m (Sen-Wyller, eq. 7.2), used below 100 km where nu_m is not small against omega:
        kappa_dB = 4.61e-5 Ne F(nu_m/omega') / omega'                        [dB/m, power]
        alpha_amp = kappa_dB cos(alpha_rw) / 8.686
    with omega' = omega (1 +/- Y |cos theta|), upper sign O, lower X (spec §7.2), and
        F(x) = < x u / (1 + x^2 u^2) >_u,   <f>_u = int f(u) u^{3/2} e^{-u} du / Gamma(5/2).
    Above 100 km the complex form is used with nu_eff = 2.5 nu_m (spec §7.2 rule).

One-way power loss in dB is 8.686 times the integral of alpha_amp ds (spec §7.3).
Conventions (spec §2.3): e^{+i omega t}, n = mu - i chi, chi > 0.

Collision frequency. IRI-2020 has no electron-neutral collision frequency, and no neutral
atmosphere is included. The default profile is an illustrative exponential placeholder (D9),
not a published fit; replace it with a ProfileModel (or a callable) built from a neutral model.

Densities (spec §7.2). The density used for absorption is the merged density used for tracing:
merge_density() builds it from the E-CHAIM background and a D-region model. riometer_density()
builds an equivalent density from a riometer map (eq. 7.3).

Public API
----------
    collision_frequency_profile(alt_km, model="default") -> nu [1/s], shape of alt_km
    absorption_rate(ray, r_vec, u_vec, model="default", rdot=None) -> alpha_amp [Np/m]
    sw_F(x) -> F(x) of eq. (7.2), tabulated;  kappa_sen_wyller(...), merge_density(...),
    riometer_density(...), ProfileModel
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma, roots_genlaguerre

from core.constants import C

KAPPA_PREF = 4.61e-5           # dB m^2 s^-1, eq. (7.1)/(7.2)
DB_PER_NEPER = 8.686
SW_ALT_KM = 100.0              # below this, nu_m (Sen-Wyller) is used for nu_kind == "nu_m"


# ---------------------------------------------------------------------------
# Collision-frequency profiles
# ---------------------------------------------------------------------------

def _default_profile(alt_km):
    """
    Placeholder D/E-region profile: nu = 1e7 exp(-(h - 70 km)/6 km) for h > 0. It is continued
    above 130 km rather than cut: a step in the absorption integrand makes the RK4 sum depend on
    the direction of travel, which breaks the reciprocity test (spec §5.5).
    """
    nu_0 = 1.0e7      # s^-1 at h_ref
    h_ref = 70.0      # km
    H_scale = 6.0     # km, e-folding height
    alt = np.asarray(alt_km, dtype=float)
    nu = nu_0 * np.exp(-(alt - h_ref) / H_scale)
    return np.where(alt > 0.0, nu, 0.0)


_PROFILES = {
    "default": _default_profile,
}


@dataclass
class ProfileModel:
    """
    AbsorptionModel of spec Listing 2 for profile-based collision frequencies.

    nu_profile   : callable(alt_km) -> nu [1/s]; nu_m if nu_kind == "nu_m", nu_eff otherwise.
    nu_kind      : "nu_m" (Sen-Wyller below 100 km, nu_eff = 2.5 nu_m above) or "nu_eff".
    density_kind : "enhancement" (dNe on top of E-CHAIM) or "total_below_merge" (model below h1,
                   E-CHAIM above h2, cosine blend between). Used by merge_density().
    h_merge      : (h1, h2) blend interval [m].
    """

    nu_profile: object
    nu_kind: str = "nu_eff"
    density_kind: str = "total_below_merge"
    h_merge: tuple = (100e3, 120e3)

    def __call__(self, alt_km):
        return self.nu_profile(alt_km)

    def collision_frequency(self, r_ecef, t=None):
        """nu [1/s] at ECEF points, in this model's nu_kind convention (Listing 2)."""
        from core.geometry import xyz_m_to_latlon

        r = np.asarray(r_ecef, dtype=float)
        _, _, alt_m = xyz_m_to_latlon(np.moveaxis(r, -1, 0))
        return collision_frequency_profile(np.asarray(alt_m) / 1e3, model=self)

    def electron_density(self, r_ecef, t=None):
        """Not provided by a collision model: the density is the E-CHAIM background merged by merge_density()."""
        raise NotImplementedError("use merge_density() with the E-CHAIM background and the D-region model")

    def riometer_A30(self, lat, lon, t=None):
        """No riometer map in a profile model (use riometer_density() with a map to build a density)."""
        return None


def collision_frequency_profile(alt_km, model="default"):
    """Collision frequency [1/s] at altitude alt_km [km], in the convention of the model's nu_kind."""
    if callable(model):
        try:
            return np.asarray(model(np.asarray(alt_km, dtype=float)), dtype=float)
        except (TypeError, ValueError):          # scalar-only callable: map over elements
            return np.vectorize(lambda h: float(model(float(h))))(np.asarray(alt_km, dtype=float))
    try:
        fn = _PROFILES[model]
    except KeyError:
        raise ValueError(
            f"unknown collision-frequency model {model!r}; "
            f"available: {list(_PROFILES)}, or pass a callable(alt_km) -> nu"
        )
    return fn(alt_km)


# ---------------------------------------------------------------------------
# Sen-Wyller average, eq. (7.2)
# ---------------------------------------------------------------------------

def sw_F_exact(x):
    """
    F(x) = <x u / (1 + x^2 u^2)>_u by adaptive quadrature, x = nu_m / omega'. The integrand has a
    scale 1/x in u, so the integral is split there (fixed-node rules miss the large-x limit).
    """
    from scipy.integrate import quad

    def one(xx):
        f = lambda u: u ** 1.5 * np.exp(-u) * xx * u / (1.0 + (xx * u) ** 2)   # noqa: E731
        pts = [min(1.0 / xx, 50.0)]
        a, _ = quad(f, 0.0, pts[0], limit=400, epsabs=0.0, epsrel=1e-12)
        b, _ = quad(f, pts[0], np.inf, limit=400, epsabs=0.0, epsrel=1e-12)
        return (a + b) / _GAMMA_52

    xs = np.asarray(x, dtype=float)
    return np.vectorize(one, otypes=[float])(xs)


_GL_N = 96
_GL_U, _GL_W = roots_genlaguerre(_GL_N, 1.5)     # nodes and weights for u^{3/2} e^{-u} (unused by sw_F)
_GAMMA_52 = gamma(2.5)


_TAB_LOGX = np.linspace(-7.0, 5.0, 481)
_TAB_LOGF = np.log(sw_F_exact(10.0 ** _TAB_LOGX))


def sw_F(x):
    """
    Tabulated F(x) of eq. (7.2) (log-log interpolation). Asymptotes: F -> (5/2) x for x -> 0 (the
    Appleton-Hartree nu_eff = 2.5 nu_m limit) and F -> (2/3)/x for x -> infinity (nu_eff = 1.5 nu_m).
    """
    x = np.asarray(x, dtype=float)
    lx = np.log10(np.maximum(x, 1e-300))
    f = np.exp(np.interp(lx, _TAB_LOGX, _TAB_LOGF))
    f = np.where(lx < _TAB_LOGX[0], 2.5 * x, f)
    f = np.where(lx > _TAB_LOGX[-1], (2.0 / 3.0) / np.maximum(x, 1e-300), f)
    return f


def kappa_sen_wyller(Ne, nu_m, omega_prime):
    """Power absorption coefficient [dB/m] of eq. (7.2): 4.61e-5 Ne F(nu_m/omega') / omega'."""
    return KAPPA_PREF * np.asarray(Ne, dtype=float) * sw_F(nu_m / omega_prime) / omega_prime


# ---------------------------------------------------------------------------
# Absorption rate
# ---------------------------------------------------------------------------

def absorption_rate(ray, r_vec, u_vec, model="default", rdot=None, _cache=None, return_att=False):
    """
    Amplitude attenuation coefficient [Np/m] along the ray (see the module docstring).
    return_att=True also returns att = k0 |Im n| lambda0 = lambda0 alpha/cos(alpha_rw), the attenuation
    per wavelength of §7.3 (the leg is flagged where it exceeds 0.1).

    r_vec, u_vec: (..., 3) position [m] and wave normal. rdot: physical ray direction
    (sign-corrected) if the caller has it, otherwise computed from the RHS. Returns an
    array of the leading shape (0-d for one point). Zero outside the ionosphere (Ne <= 0)
    and where the collision frequency is zero. Below SW_ALT_KM the nu_m convention uses
    Sen-Wyller (eq. 7.2), with omega' = omega (1 + MODE Y |cos theta|).
    """
    from core.geometry import xyz_m_to_latlon

    r = np.asarray(r_vec, dtype=float)
    u = np.asarray(u_vec, dtype=float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]

    if _cache is not None and "X" in _cache:            # centre values from the same RHS call
        X, Y_vec = _cache["X"], _cache["Y_vec"]
        Ne = X / ray.X_factor
    else:
        Ne = ray._ne(x, y, z)
        X = Ne * ray.X_factor
        Y_vec = ray.Y_vec(x, y, z)
    Y = np.linalg.norm(Y_vec, axis=-1)
    un = np.linalg.norm(u, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Ysafe = np.where(Y > 0.0, Y, 1.0)
        unsafe = np.where(un > 0.0, un, 1.0)
        cos_theta = np.sum(Y_vec / Ysafe[..., None] * u, axis=-1) / unsafe

        _, _, alt_m = xyz_m_to_latlon(np.moveaxis(r, -1, 0))
        alt_km = np.asarray(alt_m) / 1e3
        nu_raw = np.broadcast_to(np.asarray(collision_frequency_profile(alt_km, model=model), dtype=float),
                                 X.shape)
        nu_kind = getattr(model, "nu_kind", "nu_eff")
        nu_eff = 2.5 * nu_raw if nu_kind == "nu_m" else nu_raw

        Z = nu_eff / ray.omega
        U = 1.0 - 1j * Z
        p = ray.n_angle(X, Y, cos_theta, ray.MODE, U=U)
        chi = np.abs(np.imag(np.emath.sqrt(p)))      # |Im n|: independent of the root choice

        if rdot is None:
            rdot = ray._haselgrove_physical_rhs(0.0, np.concatenate([r, u], axis=-1))[..., :3]
        rdot = np.asarray(rdot, dtype=float)
        rnorm = np.linalg.norm(rdot, axis=-1)
        cos_alpha = np.sum(u * rdot, axis=-1) / (unsafe * np.where(rnorm > 0.0, rnorm, 1.0))

        k0 = ray.omega / C
        out = k0 * chi * cos_alpha
        if nu_kind == "nu_m":
            omega_p = ray.omega * (1.0 + ray.MODE * Y * np.abs(cos_theta))     # omega' = omega +/- omega_L
            kappa = kappa_sen_wyller(Ne, nu_raw, omega_p)                       # dB/m
            alpha_sw = kappa * cos_alpha / DB_PER_NEPER
            out = np.where(alt_km < SW_ALT_KM, alpha_sw, out)
    valid = (Ne > 0.0) & (nu_raw > 0.0) & (un > 0.0) & (rnorm > 0.0)
    alpha = np.where(valid, out, 0.0)
    if not return_att:
        return alpha
    lam0 = 2.0 * np.pi * C / ray.omega
    with np.errstate(divide="ignore", invalid="ignore"):
        att = np.where(valid & (cos_alpha > 0.0), lam0 * out / np.where(cos_alpha > 0.0, cos_alpha, 1.0), 0.0)
    return alpha, att


# ---------------------------------------------------------------------------
# Densities: merge with E-CHAIM (§7.2) and the riometer fallback (§7.4)
# ---------------------------------------------------------------------------

def merge_density(h_m, ne_model, ne_background, *, kind: str, h_merge=(100e3, 120e3)):
    """
    Density used for tracing and absorption (spec §7.2).

    kind == "enhancement": ne = ne_background + ne_model (the model is a dNe on top of E-CHAIM,
        which must not already contain the same D region; E-CHAIM v3.4 with dregion=True does).
    kind == "total_below_merge": ne_model below h1, ne_background above h2, and a cosine blend
        w(h) = (1 - cos(pi (h - h1)/(h2 - h1)))/2 in between.
    """
    h = np.asarray(h_m, dtype=float)
    a = np.asarray(ne_model, dtype=float)
    b = np.asarray(ne_background, dtype=float)
    if kind == "enhancement":
        return b + a
    if kind != "total_below_merge":
        raise ValueError(f"unknown density_kind {kind!r}")
    h1, h2 = h_merge
    w = np.clip((h - h1) / (h2 - h1), 0.0, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * w))
    return (1.0 - w) * a + w * b


def riometer_density(A_r_dB, w_h, nu_h, f_r: float):
    """
    Eq. (7.3): equivalent density from a vertical riometer absorption A_r [dB] at f_r with a
    normalised height profile w(h) (w dh = 1) and collision frequency nu(h):
        Ne(h) = A_r w(h) (omega_r^2 + nu^2) / (4.61e-5 nu),   omega_r = 2 pi f_r.
    Reproduces A_r exactly for a vertical path (omega_L << omega_r).
    """
    omega_r = 2.0 * np.pi * f_r
    nu = np.asarray(nu_h, dtype=float)
    return np.asarray(A_r_dB, dtype=float) * np.asarray(w_h, dtype=float) * (omega_r ** 2 + nu ** 2) \
        / (KAPPA_PREF * nu)
