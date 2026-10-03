"""
mc_scattering.py
================
Monte Carlo ray scattering module for Haselgrove-type ionospheric ray tracers.

Adds the statistical effect of small-scale ionospheric density irregularities
(delta Ne / Ne = xi, power-law spectrum, field-aligned anisotropy) to a
deterministic ray tracer, by turning Hamilton's ray equations into a Langevin
equation: deterministic Haselgrove drift + Gaussian angular kicks whose
variance is derived from the irregularity spectrum (no free parameters).

Theory summary
--------------
The medium fluctuation enters through the squared refractive index,
        delta(n^2) = -X * xi(r),        X = fp^2/f^2,
which keeps the stochastic force finite through reflection heights
(never formulate kicks through delta n = delta(n^2)/2n, which diverges
as n -> 0).

Geometric optics gives the transverse ray-slope perturbation
        d(theta_i)/ds = g * d(xi)/dq_i,      g = X / (2 n^2),
and, in the Markov (diffusion) limit, independent Gaussian direction
increments with covariance rate  (exact result for a homogeneous random
field with 3-D spectrum Phi_xi, normalised so that
Integral Phi_xi d^3q = <xi^2>):

    d<theta_i theta_j>/ds = g^2 * 2*pi * Int d^2q_perp
                              q_i q_j Phi_xi(q_perp, q_par = 0)      (K1)

where q_par is the wavevector component ALONG the ray: only medium
Fourier components transverse to the propagation direction produce
cumulative deflection (the small-angle Bragg selection rule).
The phase variance accumulates at the rate (same derivation, no
gradient weighting; phase = k0 * int(delta n) ds):

    d<sigma_phi^2>/ds = (k0 X / (2 n))^2 * 2*pi * Int d^2q_perp
                              Phi_xi(q_perp, q_par = 0)              (K2)

(K1) is dominated by the SMALLEST scales present (for spectral index
p = 4 the integrand is ~ 1/q: log-divergent, cut off at the Bragg limit
q_max = 2 k0 -- scales below lambda/2 cannot scatter propagating waves);
(K2) is dominated by the LARGEST scales (outer scale L0).

Irregularity spectrum (anisotropic power law, axis along local B):

    Phi_xi(q) = C / (q0^2 + q_perpB^2 + alpha^2 q_parB^2)^(p/2)      (S1)

with q0 = 2*pi/L0, axial ratio alpha >= 1 (field-aligned elongation),
and C fixed by the unit-variance normalisation (computed numerically).

Because the covariance (K1) depends on position/direction only through
the prefactor g^2 <xi^2> and the angle psi between the ray and B, the
angular integrals are TABULATED once over psi in [0, 90 deg] at start-up
and looked up at run time (fast).

Elastic scattering: static irregularities cannot change omega, so each
kick is a pure rotation of the wave-normal vector u (|u| preserved;
exact on-shell for isotropic plasma, O(Y * dtheta) error -- negligible --
for the magnetised case).

What this module computes (ensemble observables):
  * landing-point scatter (down-range / cross-range spread of the skip)
  * arrival-angle (elevation/azimuth) spread
  * path-length (proxy delay) spread
  * accumulated phase variance sigma_phi^2 per ray -> coherence loss
    <E>/E0 = exp(-sigma_phi^2/2), coherence bandwidth estimates
What it cannot compute (by construction -- phases randomised):
  interference fringes, caustic amplitudes, deterministic fields.

Validation included (run this file directly):
  A. Gold-standard: explicit 3-D turbulence realisation, deterministic
     rays traced through it, <theta^2>(s) growth compared against the
     tabulated diffusion prediction (K1) -- tests spectrum normalisation,
     tabulation, and the Markov limit end to end.
  B. Full demo on a mock Chapman-layer ionosphere + dipole B with the
     user's Haselgrove RHS (verbatim copy), producing an ensemble figure.

Usage with your tracer (main.py):
---------------------------------
    from mc_scattering import IrregularitySpectrum, MCScatteringTracer

    spec = IrregularitySpectrum(xi_rms=5e-3, L0_m=5e3, p_index=4.0,
                                axial_ratio=10.0)
    spec.tabulate(k0=ray.omega / 2.998e8)          # Bragg cutoff 2 k0

    mc = MCScatteringTracer(ray, spec, ds_step_m=1000.0, seed=1)
    results = mc.ensemble(r0, u0, n_rays=500, max_path_m=MAX_PATH_M)
    summary = mc.summarize(results, r0)

`ray` is your Ray instance (only `_haselgrove_physical_rhs`,
`electron_density`, `X_factor`, `Y_vec`, `omega` are used).

References: Booker (1959) JGR; Tatarskii (1961); Ishimaru (1978) Vol 1
ch. 6-8; Kontar et al. (2019) ApJ 884:122 (Langevin MC, public code);
Zabotin et al. (2004) Radio Sci. 39 RS3002 (HF multiple scattering RTE).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from raytracer import RayTracer, TraceResult
from geometry import down_cross_range_m, ellipsoid_outward_normal_m


# ===========================================================================
#  Irregularity spectrum + tabulated kick statistics
# ===========================================================================

class IrregularitySpectrum:
    """
    Anisotropic power-law spectrum of fractional density fluctuations,
    Eq. (S1), normalised to unit variance (multiply rates by xi_rms^2).

    Parameters
    ----------
    xi_rms : float
        RMS fractional electron-density fluctuation <xi^2>^(1/2)
        (typical mid-latitude 1e-3 ... 1e-2; strong spread-F up to 1e-1).
    L0_m : float
        Outer scale [m] (q0 = 2*pi/L0). Typical 5-25 km.
    p_index : float
        3-D spectral index p (in-situ 1-D index + 2). Default 4.
    axial_ratio : float
        Field-aligned elongation alpha >= 1. 1 = isotropic.
    q_max : float or None
        High-wavenumber cutoff [rad/m]. If None, set by .tabulate(k0)
        to the Bragg limit 2*k0. An inner-scale cutoff can be imposed
        by passing q_max = 2*pi/l_inner explicitly.
    """

    def __init__(self, xi_rms: float, L0_m: float, p_index: float = 4.0,
                 axial_ratio: float = 1.0, q_max: float | None = None):
        self.xi_rms = float(xi_rms)
        self.q0 = 2.0 * np.pi / float(L0_m)
        self.p = float(p_index)
        self.alpha = float(axial_ratio)
        self.q_max = q_max

        self._C_unit = None          # unit-variance normalisation constant
        self._psi_grid = None        # tabulation grid
        self._I11 = None             # transverse integrals vs psi
        self._I22 = None
        self._Iph = None

    # -- spectrum kernel (unnormalised) -----------------------------------
    def _kernel(self, q_perpB2, q_parB2):
        return (self.q0 ** 2 + q_perpB2 + self.alpha ** 2 * q_parB2) ** (-self.p / 2.0)

    # -- unit-variance normalisation --------------------------------------
    def _compute_normalisation(self):
        """
        C_unit such that  Integral C_unit * kernel d^3q = 1.
        Stretch q_parB' = alpha*q_parB -> isotropic integral / alpha.
        """
        q = np.logspace(np.log10(self.q0 * 1e-3),
                        np.log10(self.q0 * 1e6), 4000)
        integrand = 4.0 * np.pi * q ** 2 * (self.q0 ** 2 + q ** 2) ** (-self.p / 2.0)
        N_iso = np.trapz(integrand, q)
        self._C_unit = self.alpha / N_iso

    # -- tabulation of the transverse-plane integrals ----------------------
    def tabulate(self, k0: float, n_psi: int = 61, n_q: int = 400,
                 n_phi: int = 128):
        """
        Tabulate, over psi = angle(ray direction, B) in [0, pi/2]:

          I11(psi), I22(psi) = 2*pi * Int d^2q  q_i^2 * C*kernel(q; psi)
          Iph(psi)           = 2*pi * Int d^2q        C*kernel(q; psi)

        in the canonical transverse frame:
          e1 = unit projection of B onto the plane perpendicular to the ray
          e2 = ray_dir x e1.
        For q = q1 e1 + q2 e2 (plane q_par_ray = 0):
          q_parB  = q1 * sin(psi)          (component along B)
          q_perpB^2 = q^2 - q_parB^2.
        High-q cutoff: q_max (Bragg 2*k0 by default).
        """
        if self._C_unit is None:
            self._compute_normalisation()
        if self.q_max is None:
            self.q_max = 2.0 * k0

        psi = np.linspace(0.0, np.pi / 2.0, n_psi)
        qr = np.logspace(np.log10(self.q0 * 1e-3),
                         np.log10(self.q_max), n_q)          # radial grid
        ph = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

        QR, PH = np.meshgrid(qr, ph, indexing="ij")           # (n_q, n_phi)
        Q1 = QR * np.cos(PH)
        Q2 = QR * np.sin(PH)
        dphi = ph[1] - ph[0]

        I11 = np.empty(n_psi)
        I22 = np.empty(n_psi)
        Iph = np.empty(n_psi)

        for i, ps in enumerate(psi):
            sinps = np.sin(ps)
            q_parB2 = (Q1 * sinps) ** 2
            q_perpB2 = QR ** 2 - q_parB2
            F = self._C_unit * self._kernel(q_perpB2, q_parB2)
            # 2*pi * Int q_i q_j F d^2q ;  d^2q = q dq dphi
            w11 = np.trapz(Q1 ** 2 * F * QR, qr, axis=0).sum() * dphi
            w22 = np.trapz(Q2 ** 2 * F * QR, qr, axis=0).sum() * dphi
            wph = np.trapz(F * QR, qr, axis=0).sum() * dphi
            I11[i] = 2.0 * np.pi * w11
            I22[i] = 2.0 * np.pi * w22
            Iph[i] = 2.0 * np.pi * wph

        self._psi_grid = psi
        self._I11, self._I22, self._Iph = I11, I22, Iph
        return self

    # -- run-time lookup ----------------------------------------------------
    def transverse_integrals(self, psi: float):
        """Interpolated (I11, I22, Iph) at angle psi [rad] in [0, pi/2]."""
        p = abs(psi)
        if p > np.pi / 2.0:
            p = np.pi - p
        i11 = np.interp(p, self._psi_grid, self._I11)
        i22 = np.interp(p, self._psi_grid, self._I22)
        iph = np.interp(p, self._psi_grid, self._Iph)
        return i11, i22, iph


# ===========================================================================
#  Per-ray record
# ===========================================================================

@dataclass
class RayResult:
    landed: bool = False
    escaped: bool = False
    r_final: np.ndarray = None
    u_final: np.ndarray = None
    s_total: float = 0.0            # arc length [m]
    sigma_phi2: float = 0.0         # accumulated phase variance [rad^2]
    n_kicks: int = 0
    theta2_accum: float = 0.0       # accumulated kick variance (diagnostic)
    path: np.ndarray = None         # decimated (n,3) positions [m]
    s_path: np.ndarray = None
    # -- additive fields (Phase 1: PHaRLAP-style per-ray outputs) --
    phase_path_total: float = 0.0
    group_path_total: float = 0.0
    absorption_nepers_total: float = 0.0
    absorption_db_total: float = 0.0
    ground_range_total_m: float = 0.0
    doppler_shift_hz: float = None
    hops_completed: int = 0
    bounce_points: np.ndarray = None


def _to_legacy_ray_result(full: TraceResult) -> RayResult:
    """Copy the fields of a raytracer.TraceResult into a legacy RayResult."""
    return RayResult(
        landed=full.landed, escaped=full.escaped,
        r_final=full.r_final, u_final=full.u_final,
        s_total=full.s_total, sigma_phi2=full.sigma_phi2,
        n_kicks=full.n_kicks, theta2_accum=full.theta2_accum,
        path=full.path, s_path=full.s_path,
        phase_path_total=full.phase_path_total,
        group_path_total=full.group_path_total,
        absorption_nepers_total=full.absorption_nepers_total,
        absorption_db_total=full.absorption_db_total,
        ground_range_total_m=full.ground_range_total_m,
        doppler_shift_hz=full.doppler_shift_hz,
        hops_completed=full.hops_completed,
        bounce_points=full.bounce_points,
    )


# ===========================================================================
#  Monte Carlo tracer
# ===========================================================================

class MCScatteringTracer(RayTracer):
    """
    Fixed-step RK4 integrator for the user's Haselgrove RHS with Gaussian
    scattering kicks applied each step (Langevin / diffusion limit).

    This is now a thin backward-compatible wrapper around raytracer.RayTracer
    (single shared engine also used by the deterministic/tangent-map path):
    trace() forces nhops=1, method="fixed" and returns a legacy RayResult.
    Use RayTracer directly (see raytracer.py) for multi-hop (nhops>1) or
    tangent-map (method="adaptive") tracing.

    Notes
    -----
    * Integrates only the 10-dim extended state (r, u, s, phase_path,
      group_path, absorption): the tangent map is NOT propagated (12 extra
      RHS evaluations/step -- prohibitive for ensembles). Use
      RayTracer(..., ).trace(..., carry_tangent_map=True, method="adaptive")
      per-ray if amplitude along scattered rays is needed.
    * Step size ds_step_m should resolve the background scale height
      (<= a few km). Kicks are applied every step with variance
      rate * ds -- the exact diffusion-limit process.
    """

    def __init__(self, ray, spectrum: IrregularitySpectrum,
                 ds_step_m: float = 1000.0,
                 seed: int | None = None,
                 n2_floor: float = 1e-2,
                 record_every: int = 10):
        super().__init__(ray, spectrum=spectrum, ds_step_m=ds_step_m, seed=seed,
                          n2_floor=n2_floor, record_every=record_every)

    # -- single ray ----------------------------------------------------------
    def trace(self, r0: np.ndarray, u0: np.ndarray, max_path_m: float,
              scatter: bool = True,
              ceiling_alt_m: float = 800e3,
              nhops: int = 1) -> RayResult:
        full = super().trace(r0, u0, max_path_m, nhops=nhops, method="fixed",
                              scatter=scatter, compute_phase_variance=True,
                              ceiling_alt_m=ceiling_alt_m)
        return _to_legacy_ray_result(full)

    # -- ensemble ------------------------------------------------------------
    def ensemble(self, r0, u0, n_rays: int, max_path_m: float,
                 nhops: int = 1, verbose_every: int = 50):
        out = []
        for i in range(n_rays):
            out.append(self.trace(r0, u0, max_path_m, scatter=True, nhops=nhops))
            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"    ray {i+1}/{n_rays}", flush=True)
        return out

    # -- summary statistics ----------------------------------------------------
    @staticmethod
    def summarize(results, r0):
        """Landing statistics relative to the launch point r0."""
        landed = [r for r in results if r.landed]
        if not landed:
            return {"n_landed": 0}

        r0 = np.asarray(r0, dtype=float)
        up0 = r0 / np.linalg.norm(r0)
        R = np.linalg.norm(r0)

        pos = np.array([r.r_final for r in landed])
        # define local down-range axis from the mean landing direction
        mean_dir = np.mean(pos, axis=0)
        t_hat = mean_dir - np.dot(mean_dir, up0) * up0
        t_hat = t_hat / np.linalg.norm(t_hat)

        rng_m, cross_m = [], []
        for p in pos:
            dr, cr = down_cross_range_m(p, r0, t_hat, R=R)
            rng_m.append(dr)
            cross_m.append(cr)
        rng_m = np.array(rng_m)
        cross_m = np.array(cross_m)

        # arrival elevation (true WGS84 local vertical, not geocentric radial --
        # these differ by up to 0.192 deg, the "deflection of the vertical")
        elev = []
        for r in landed:
            up = ellipsoid_outward_normal_m(r.r_final)
            u_hat = r.u_final / np.linalg.norm(r.u_final)
            elev.append(np.degrees(np.arcsin(np.clip(-np.dot(u_hat, up), -1, 1))))
        s_tot = np.array([r.s_total for r in landed])
        group_tot = np.array([r.group_path_total for r in landed])
        abs_db = np.array([r.absorption_db_total for r in landed])
        sp2 = np.array([r.sigma_phi2 for r in landed])

        return {
            "n_landed": len(landed),
            "range_mean_km": np.mean(rng_m) / 1e3,
            "range_std_km": np.std(rng_m) / 1e3,
            "cross_std_km": np.std(cross_m) / 1e3,
            "elev_mean_deg": np.mean(elev),
            "elev_std_deg": np.std(elev),
            "s_std_km": np.std(s_tot) / 1e3,
            "group_path_mean_km": np.mean(group_tot) / 1e3,
            "group_path_std_km": np.std(group_tot) / 1e3,
            "absorption_db_mean": np.mean(abs_db),
            "absorption_db_std": np.std(abs_db),
            "sigma_phi2_mean": np.mean(sp2),
            "coherent_fraction": np.exp(-np.mean(sp2) / 2.0),
            "ranges_km": rng_m / 1e3,
            "crosses_km": cross_m / 1e3,
            "elevs_deg": np.array(elev),
            "s_km": s_tot / 1e3,
            "group_km": group_tot / 1e3,
        }