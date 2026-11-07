"""
raytracer.py — consolidated Haselgrove ray-tracing engine.

Replaces the two previously-independent, hand-duplicated integrators:
  * main.py::Ray.solve_haselgrove          (adaptive solve_ivp, tangent map)
  * mc_scattering.py::MCScatteringTracer   (fixed-step RK4, stochastic kicks)

with one physics RHS (`RayTracer._extended_rhs`) and one hop-aware outer
loop (`RayTracer.trace`), fed by two thin segment drivers that share that
RHS. This is what guarantees the deterministic and statistical tracers
can't silently diverge as features are added: a change to Appleton-Hartree,
absorption, or phase/group-path formulas is automatically seen by both.

Extended state layout (metres / metres, tau = ray parameter)
--------------------------------------------------------------
    state[0:6]   x, y, z, ux, uy, uz     physical (Ray._haselgrove_physical_rhs)
    state[6]     s          geometric arc length      ds/dtau       = |rdot|
    state[7]     P_phase    phase path                dP_phase/dtau = u . rdot        (eikonal identity)
    state[8]     P_group    group path                dP_group/dtau = mu_group * |rdot|
    state[9]     A          absorption [nepers]        dA/dtau       = absorption_rate(...) * |rdot|
    state[10]    sigma_phi2 phase variance [rad^2]     dsigma/dtau   = rate_phase(...) * |rdot|,
                                                        rate_phase from _kick_rates -- deterministic,
                                                        no random kicks needed to accumulate this
    state[11:47] Phi (6x6, flat)  tangent map, present only when carry_tangent_map=True

`ground_range` is NOT a state row (it's a closed-form function of the final
position, see geometry.great_circle_range_m) since it doesn't need
integrating.

Doppler shift is deliberately NOT computed here yet: TraceResult.doppler_
shift_hz is always None. The physically correct definition needs two ray
traces at different IRI epochs with phase_path finite-differenced between
them (a moving/evolving medium), which is a different call shape than a
single static trace — see the module-level `compute_doppler` stub below
for the intended Phase 1.5 extension. A single-snapshot proxy
(d(group_path)/d(frequency)) exists but is *not* physically Doppler shift
(it diverges near ray cutoffs), so it is intentionally not offered here
under that name.
"""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp

from core.constants import q_e, m_e, epsilon_0, C
from core.geometry import great_circle_range_m, ellipsoid_outward_normal_m, WGS84_A, WGS84_B
from raytracing.absorption import absorption_rate as _absorption_rate


def _ellipsoid_quadric(r_vec: np.ndarray, A: float, B: float) -> float:
    """x^2/A^2 + y^2/A^2 + z^2/B^2 - 1: negative inside, positive outside, zero on the surface."""
    x, y, z = r_vec[0], r_vec[1], r_vec[2]
    return x * x / (A * A) + y * y / (A * A) + z * z / (B * B) - 1.0


# ---------------------------------------------------------------------------
# Extended-state layout offsets.
# ---------------------------------------------------------------------------
# Base rows [0:_PHI_OFFSET] are ALWAYS present; want_group/want_absorption/
# want_phase_variance flags in _extended_rhs gate VALUE (real term vs 0.0),
# not presence, so segment drivers / the hop loop / the ground-bounce
# transition can index by fixed position regardless of which optional terms
# a given trace() call requested. The tangent map (want_tangent) is the one
# exception: _extended_rhs never carries it (see its docstring) --
# _segment_adaptive appends it to final_state itself, via a separate pass
# (_propagate_tangent_map), only when requested.
_S_IDX = 6
_PHASE_PATH_IDX = 7
_GROUP_PATH_IDX = 8
_ABSORPTION_IDX = 9
_SIGMA_PHI2_IDX = 10
_PHI_OFFSET = 11   # flat 6x6 tangent map starts here, only if carry_tangent_map=True
                   # (state length 11, or 11+36=47 with the tangent map)


# ===========================================================================
#  Results
# ===========================================================================

@dataclass
class SegmentResult:
    """One hop's (or hop-fragment's) worth of integration."""
    final_state: np.ndarray
    path: np.ndarray            # (n, 3) decimated positions, metres
    s_path: np.ndarray          # (n,) arc length at each decimated sample
    reason: str                 # "GROUND" | "CEILING" | "LENGTH_BUDGET" | "STALL"
    n_kicks: int = 0
    theta2_accum: float = 0.0


@dataclass
class TraceResult:
    """Full multi-hop trace result — the PHaRLAP-style per-ray output bundle."""
    landed: bool = False
    escaped: bool = False
    safety_cap_hit: bool = False
    hops_completed: int = 0

    r_final: np.ndarray = None
    u_final: np.ndarray = None

    s_total: float = 0.0                    # geometric path length, m
    phase_path_total: float = 0.0           # m (electrical length / k0)
    group_path_total: float = 0.0           # m
    absorption_amplitude_nepers_total: float = 0.0   # amplitude exponent A = integral(k0|Im n|cos(alpha) ds)
    absorption_db_total: float = 0.0        # power, dB (one-way) = 8.686 * A
    ground_range_total_m: float = 0.0
    doppler_shift_hz: float = None          # always None in Phase 1 — see module docstring

    sigma_phi2: float = 0.0                 # scattering phase-variance (Langevin ensemble only)
    n_kicks: int = 0
    theta2_accum: float = 0.0

    path: np.ndarray = None                 # (n, 3) decimated positions across all hops, metres
    s_path: np.ndarray = None               # (n,) arc length at each decimated sample
    bounce_points: np.ndarray = None        # (hops_completed, 3) ground-bounce positions
    tangent_map_final: np.ndarray = None    # (6, 6), only if carry_tangent_map=True


def _any_perp(v: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to unit v."""
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e = np.cross(v, a)
    return e / np.linalg.norm(e)


# ===========================================================================
#  Engine
# ===========================================================================

class RayTracer:
    """
    Consolidated ray-tracing engine for a given Ray (physical model) and,
    optionally, an IrregularitySpectrum (for stochastic scattering kicks).

    Parameters
    ----------
    ray : main.Ray
    spectrum : mc_scattering.IrregularitySpectrum or None
        Required only if you intend to call .trace(..., scatter=True).
    ds_step_m : float
        Fixed step size used by method="fixed" (the Langevin-ensemble
        driver; kick variance scaling assumes a small, roughly uniform
        step). Also used as the "arc length since last bounce" guard
        scale for both drivers' ground-crossing detection.
    seed : int or None
        RNG seed for stochastic kicks.
    n2_floor : float
        Floor on |u|^2 used in the kick-rate slope-response factor.
    record_every : int
        Decimation factor for stored path points (method="fixed" only;
        method="adaptive" stores solve_ivp's own accepted steps).
    absorption_model : str or callable
        Passed through to absorption.collision_frequency_profile.
    """

    def __init__(self, ray, spectrum=None, ds_step_m: float = 1000.0,
                 seed: int | None = None, n2_floor: float = 1e-2,
                 record_every: int = 10, absorption_model: str = "default"):
        warnings.warn("RayTracer (raytracer.py) is deprecated (decision D6): use batch.BatchTracer, which "
                      "carries the same physics. It is kept until Phase 5g, then removed.",
                      DeprecationWarning, stacklevel=2)
        self.ray = ray
        self.spectrum = spectrum
        self.ds = float(ds_step_m)
        self.rng = np.random.default_rng(seed)
        self.n2_floor = float(n2_floor)
        self.record_every = int(record_every)
        self.absorption_model = absorption_model

        self.k0 = ray.omega / C
        if spectrum is not None and spectrum._I11 is None:
            spectrum.tabulate(k0=self.k0)

    # ------------------------------------------------------------------
    # Group refractive index (for group_path) — reuses Ray._appleton_hartree_p
    # ------------------------------------------------------------------

    def _group_index(self, phys: np.ndarray) -> float:
        """
        Exact group index n_g = c/|v_g| (P0.3): see Ray.group_index. The
        earlier finite-difference version held Y.u fixed while varying omega,
        which is not the group velocity (up to 26% wrong in magnetised
        oblique regions; docs/implementation_plan.md §3.3).
        """
        return self.ray.group_index(phys)

    # ------------------------------------------------------------------
    # Extended RHS — the single physics function both drivers share
    # ------------------------------------------------------------------

    def _extended_rhs(self, tau: float, state: np.ndarray, *,
                       want_group: bool, want_absorption: bool,
                       want_phase_variance: bool) -> np.ndarray:
        """
        The base (11-dim) extended state only -- never carries the tangent
        map. See _propagate_tangent_map for why: dPhi/dtau=Jm@Phi driven by
        the SAME tau used here blows Phi up by an astronomical, physically
        meaningless factor in weakly-ionized regions, where dtau per unit
        of real distance is enormous (see _dispersion_eps in main.py).
        _segment_adaptive integrates this base state only, then propagates
        Phi across the finished segment as a separate pass parametrized by
        arc length instead, which has no such degeneracy.
        """
        phys = state[:6]
        f = self.ray._haselgrove_physical_rhs(tau, phys)
        rdot = f[:3]
        u_vec = phys[3:6]

        dsdt = float(np.linalg.norm(rdot))
        dphase = float(np.dot(u_vec, rdot))
        dgroup = self._group_index(phys) * dsdt if want_group else 0.0
        dabs = (_absorption_rate(self.ray, phys[:3], u_vec, model=self.absorption_model,
                                 rdot=rdot) * dsdt
                if want_absorption else 0.0)
        if want_phase_variance:
            _, _, _, _, rate_ph = self._kick_rates(phys[:3], u_vec)
            dsigma = rate_ph * dsdt
        else:
            dsigma = 0.0

        return np.concatenate([f, [dsdt, dphase, dgroup, dabs, dsigma]])

    # ------------------------------------------------------------------
    # Stochastic kick rates (moved verbatim from mc_scattering.py)
    # ------------------------------------------------------------------

    def _kick_rates(self, r_vec: np.ndarray, u_vec: np.ndarray):
        """
        Returns (rate11, rate22, e1, e2, rate_phase) at (r, u): angular-
        variance rates [rad^2/m] along the canonical transverse axes, the
        axes themselves, and the phase-variance rate [rad^2/m].
        """
        x, y, z = r_vec
        Ne = self.ray.electron_density(x, y, z)
        if Ne <= 0.0:
            return 0.0, 0.0, None, None, 0.0
        X = Ne * self.ray.X_factor
        if X <= 0.0:
            return 0.0, 0.0, None, None, 0.0

        n2 = max(float(np.dot(u_vec, u_vec)), self.n2_floor)

        u_hat = u_vec / np.sqrt(n2)
        B = self.ray.Y_vec(x, y, z)
        Bn = np.linalg.norm(B)
        if Bn > 0.0:
            b_hat = B / Bn
            cospsi = abs(float(np.dot(u_hat, b_hat)))
            cospsi = min(cospsi, 1.0)
            psi = np.arccos(cospsi)
            e1 = b_hat - np.dot(b_hat, u_hat) * u_hat
            n1 = np.linalg.norm(e1)
            if n1 < 1e-9:
                e1 = _any_perp(u_hat)
            else:
                e1 = e1 / n1
        else:
            psi = 0.0
            e1 = _any_perp(u_hat)
        e2 = np.cross(u_hat, e1)

        I11, I22, Iph = self.spectrum.transverse_integrals(psi)
        xi2 = self.spectrum.xi_rms ** 2

        g = X / (2.0 * n2)
        rate11 = g * g * xi2 * I11
        rate22 = g * g * xi2 * I22
        rate_ph = (self.k0 * X) ** 2 / (4.0 * n2) * xi2 * Iph

        return rate11, rate22, e1, e2, rate_ph

    # ------------------------------------------------------------------
    # Ground-crossing localization
    # ------------------------------------------------------------------

    @staticmethod
    def _locate_ground_crossing(state_before: np.ndarray, state_after: np.ndarray) -> np.ndarray:
        """
        Bisect the straight-line interpolant between two accepted states
        for the fraction lambda where the WGS84 ellipsoid quadric crosses
        zero, and linearly interpolate the WHOLE state (accumulators
        included) there. Good enough given a short (~ds_step_m) step.
        """
        def f_excess(lam):
            r = state_before[:3] + lam * (state_after[:3] - state_before[:3])
            return _ellipsoid_quadric(r, WGS84_A, WGS84_B)

        lo, hi = 0.0, 1.0
        f_lo = f_excess(lo)
        if f_lo <= 0.0:
            return state_before.copy()
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if f_excess(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        return state_before + lam * (state_after - state_before)

    # ------------------------------------------------------------------
    # Segment driver 1: fixed-step RK4 (Langevin ensemble)
    # ------------------------------------------------------------------

    def _segment_fixed_step_rk4(self, state0: np.ndarray, s_budget: float, *,
                                 scatter: bool, want_group: bool, want_absorption: bool,
                                 want_phase_variance: bool,
                                 ceiling_alt_m: float,
                                 launched_r: float) -> SegmentResult:
        state = state0.copy()
        s0 = state[6]
        A_ceil, B_ceil = WGS84_A + ceiling_alt_m, WGS84_B + ceiling_alt_m

        pts = [state[:3].copy()]
        ss = [state[6]]
        step_i = 0
        n_kicks = 0
        theta2_accum = 0.0
        reason = "LENGTH_BUDGET"

        rhs_kwargs = dict(want_group=want_group, want_absorption=want_absorption,
                           want_phase_variance=want_phase_variance)

        while state[6] - s0 < s_budget:
            f1 = self._extended_rhs(0.0, state, **rhs_kwargs)
            v1 = np.linalg.norm(f1[:3])
            if v1 < 1e-12:
                reason = "STALL"
                break
            dtau = self.ds / v1

            # dtau above assumes |rdot| stays close to v1 (its value at the
            # START of the step) for the whole step -- normally fine, but
            # ds/dtau is NOT O(1) throughout (see main.py::Ray._dispersion_
            # eps): it can be tiny in a weakly-ionized layer and grow by
            # many orders of magnitude within a few hundred metres of arc
            # length (observed directly: v1 ~ 1e-6 just above the vacuum
            # threshold, THEN grows sharply -- a step sized for that tiny
            # v1 overshoots by 100,000+ km before this check existed).
            # Retry with a halved dtau whenever the resulting step turns
            # out far larger than intended -- the only reliable sign
            # v1 badly underestimated the true local rate anywhere within
            # the step just taken.
            for _ in range(60):
                f2 = self._extended_rhs(0.0, state + 0.5 * dtau * f1, **rhs_kwargs)
                f3 = self._extended_rhs(0.0, state + 0.5 * dtau * f2, **rhs_kwargs)
                f4 = self._extended_rhs(0.0, state + dtau * f3, **rhs_kwargs)
                new_state = state + (dtau / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
                ds_actual = np.linalg.norm(new_state[:3] - state[:3])
                if ds_actual <= 5.0 * self.ds:
                    break
                dtau *= 0.5
            else:
                reason = "STALL"
                break

            if scatter:
                r_vec = new_state[:3]
                u_vec = new_state[3:6]
                r11, r22, e1, e2, _ = self._kick_rates(r_vec, u_vec)
                if r11 > 0.0 or r22 > 0.0:
                    d1 = self.rng.normal(0.0, np.sqrt(max(r11, 0.0) * ds_actual))
                    d2 = self.rng.normal(0.0, np.sqrt(max(r22, 0.0) * ds_actual))
                    umag = np.linalg.norm(u_vec)
                    u_hat = u_vec / umag
                    u_hat = u_hat + d1 * e1 + d2 * e2
                    new_state[3:6] = umag * u_hat / np.linalg.norm(u_hat)
                    n_kicks += 1
                    theta2_accum += d1 * d1 + d2 * d2

            step_i += 1
            state_prev = state
            state = new_state

            if step_i % self.record_every == 0:
                pts.append(state[:3].copy())
                ss.append(state[6])

            if (_ellipsoid_quadric(state[:3], WGS84_A, WGS84_B) <= 0.0
                    and (state[6] - s0) > 2.0 * self.ds):
                state = self._locate_ground_crossing(state_prev, state)
                reason = "GROUND"
                break
            rn = np.linalg.norm(state[:3])
            if _ellipsoid_quadric(state[:3], A_ceil, B_ceil) > 0.0 and rn > launched_r:
                reason = "CEILING"
                break

        pts.append(state[:3].copy())
        ss.append(state[6])
        return SegmentResult(final_state=state, path=np.array(pts), s_path=np.array(ss),
                              reason=reason, n_kicks=n_kicks, theta2_accum=theta2_accum)

    # ------------------------------------------------------------------
    # Segment driver 2: adaptive solve_ivp (tangent map / high-accuracy)
    # ------------------------------------------------------------------

    def _segment_adaptive(self, state0: np.ndarray, s_budget: float, *,
                           want_group: bool, want_absorption: bool,
                           want_phase_variance: bool, want_tangent: bool,
                           ceiling_alt_m: float, launched_r: float,
                           max_step: float = np.inf,
                           rtol: float = 1e-6, atol: float = 1e-9) -> SegmentResult:
        s0 = state0[6]
        # Base (11-dim) state only -- see _extended_rhs and _propagate_
        # tangent_map for why the tangent map, when wanted, is computed as
        # a separate pass afterward instead of alongside this integration.
        base_state0 = state0[:_PHI_OFFSET] if want_tangent else state0
        rhs_kwargs = dict(want_group=want_group, want_absorption=want_absorption,
                           want_phase_variance=want_phase_variance)

        def rhs(tau, state):
            return self._extended_rhs(tau, state, **rhs_kwargs)

        guard_s = 2.0 * self.ds
        A_ceil, B_ceil = WGS84_A + ceiling_alt_m, WGS84_B + ceiling_alt_m

        def ev_ground(tau, state):
            if state[6] - s0 < guard_s:
                return 1.0
            return _ellipsoid_quadric(state[:3], WGS84_A, WGS84_B)
        ev_ground.terminal = True
        ev_ground.direction = -1.0

        def ev_ceiling(tau, state):
            return _ellipsoid_quadric(state[:3], A_ceil, B_ceil)
        ev_ceiling.terminal = True
        ev_ceiling.direction = 1.0

        def ev_length(tau, state):
            return state[6] - (s0 + s_budget)
        ev_length.terminal = True
        ev_length.direction = 1.0

        # max_step defaults to scipy's own np.inf (no artificial cap): tau
        # is Haselgrove's ray parameter, not arc length, and ds/dtau is
        # NOT O(1) in general -- it legitimately shrinks toward 0 in
        # weakly-ionized regions (see main.py::Ray._dispersion_eps). A
        # fixed tau-space cap sized for "a few km of arc length" under
        # ds/dtau~1 becomes, in such a region, a cap on ARC-LENGTH
        # progress per step of a small fraction of a millimetre --
        # solve_ivp then needs millions of steps to cross a routine
        # weakly-ionized layer (observed directly: capped at 5000, it
        # advanced 21 m of arc length in 21,000+ RHS evaluations crossing
        # the D-layer base). RK45's own error control (rtol/atol) already
        # sizes each step for accuracy without this cap; pass an explicit
        # max_step= only if you specifically need guaranteed-fine dense
        # output resolution and know ds/dtau stays of order 1 throughout.
        #
        # rtol/atol default tighter than scipy's own (1e-3/1e-6): verified
        # directly that the default is nowhere near converged for a real
        # reflection -- it put a ray's apex at 210 km and its landing range
        # 25% short of the value both 1e-6/1e-9 and 1e-9/1e-12 agree on
        # (166 km apex, matching the fixed-step driver's own step-halving-
        # verified converged answer to within 0.1%). The dynamic range this
        # RHS spans (X from ~1e-29 to ~1 across one trace) is exactly the
        # regime where scipy's default, tuned for smoother problems, under-
        # resolves the one region -- the reflection turning point -- that
        # actually determines where the ray comes down.
        sol = solve_ivp(rhs, (0.0, 1e12), base_state0, method="RK45", max_step=max_step,
                         rtol=rtol, atol=atol,
                         events=[ev_ground, ev_ceiling, ev_length], dense_output=True)

        final_state = sol.y[:, -1]
        if len(sol.t_events[0]) > 0:
            reason = "GROUND"
        elif len(sol.t_events[1]) > 0:
            reason = "CEILING"
        elif len(sol.t_events[2]) > 0:
            reason = "LENGTH_BUDGET"
        else:
            reason = "STALL"

        path = sol.y[:3, :].T
        s_path = sol.y[6, :]

        if want_tangent:
            phi0 = state0[_PHI_OFFSET:].reshape(6, 6)
            phi_final = self._propagate_tangent_map(sol, tau_final=sol.t[-1], s_start=s0, phi0=phi0)
            final_state = np.concatenate([final_state, phi_final.reshape(-1)])

        return SegmentResult(final_state=final_state, path=path, s_path=s_path, reason=reason)

    # ------------------------------------------------------------------
    # Tangent map: separate pass over an already-traced segment
    # ------------------------------------------------------------------

    def _propagate_tangent_map(self, sol, tau_final: float, s_start: float, phi0: np.ndarray) -> np.ndarray:
        """
        Propagate the 6x6 tangent map Phi (d(final r,u)/d(initial r,u))
        across a segment already traced by _segment_adaptive, returning
        the updated Phi.

        Why a separate pass, parametrized by arc length s instead of
        Haselgrove's ray parameter tau: dPhi/dtau = Jm(r,u)@Phi is exact,
        but tau is not arc length, and ds/dtau shrinks legitimately toward
        0 in weakly-ionized regions (see main.py::Ray._dispersion_eps).
        Crossing such a region can take an enormous span of tau to cover
        even a short real distance -- and integrating a LINEAR ODE like
        Phi's over an enormous independent-variable span blows it up by
        exp(Jm * that huge span), even when Jm itself is small and
        perfectly well-behaved. Observed directly: solve_ivp stalling
        completely (tens of thousands of RHS evaluations, no progress)
        69 km into an otherwise unremarkable trace, with |Phi| already at
        ~1e5 from a Jacobian entry no larger than ~4e-6.

        Arc length doesn't have this problem: ds/ds = 1 always, by
        definition, so there's no analogous stretching to blow anything
        up. Reparametrizing state-as-a-function-of-tau into state-as-a-
        function-of-s via ds/dtau = |rdot| gives, by the chain rule,

            dPhi_s/ds = [Jm/v - outer(F, grad_v)/v**2] @ Phi_s,   v = |rdot|

        where grad_v = d|rdot|/d(r,u) = (rdot/v) @ Jm[:3,:] -- obtained
        algebraically from the SAME Jm and rdot already computed for the
        physical RHS, no extra finite-differencing needed. This is
        integrated as its own solve_ivp call, over s directly (bounded,
        known in advance from the already-completed base integration --
        no events needed), looking up the physical state at each s via
        the base integration's own dense output. Being a second, fully
        decoupled pass also means a numerical difficulty in the tangent
        map can never again corrupt the physical trajectory itself, the
        way it did when both were integrated together.
        """
        tau0 = 0.0
        s_end = float(sol.sol(tau_final)[6])
        s_span = s_end - s_start
        if s_span <= 0.0:
            return phi0.copy()

        def rhs_s(s, y):
            tau_lookup = min(max(y[0], tau0), tau_final)
            phys = sol.sol(tau_lookup)[:6]
            f = self.ray._haselgrove_physical_rhs(0.0, phys)
            rdot = f[:3]
            v = float(np.linalg.norm(rdot))
            v = max(v, 1e-300)
            Jm = self.ray._state_jacobian_fd(0.0, phys)
            grad_v = (rdot / v) @ Jm[:3, :]              # (6,)
            A = Jm / v - np.outer(f, grad_v) / (v * v)   # (6,6)
            phi = y[1:].reshape(6, 6)
            dphi = (A @ phi).reshape(-1)
            return np.concatenate([[1.0 / v], dphi])

        y0 = np.concatenate([[tau0], phi0.reshape(-1)])
        res = solve_ivp(rhs_s, (0.0, s_span), y0, method="RK45", rtol=1e-6, atol=1e-9)
        return res.y[1:, -1].reshape(6, 6)

    # ------------------------------------------------------------------
    # Outer hop loop
    # ------------------------------------------------------------------

    def trace(self, r0: np.ndarray, u0: np.ndarray, max_path_m: float, *,
              nhops: int = 1, method: str = "fixed", scatter: bool = False,
              carry_tangent_map: bool = False,
              compute_group_path: bool = True, compute_absorption: bool = True,
              compute_phase_variance: bool = False,
              max_total_path_m: float | None = None,
              ceiling_alt_m: float = 800e3) -> TraceResult:
        """
        Trace a ray for up to `nhops` ground bounces (specular reflection
        about the local WGS84 ellipsoid outward normal). "Ground" is
        always the true WGS84 reference ellipsoid (geometry.WGS84_A/
        WGS84_B) -- there is no configurable ground radius, since every
        call site historically passed the same literal Earth-radius
        default anyway.

        `max_path_m` is the budget PER HOP. `max_total_path_m` is an
        overall safety cap across all hops combined (guards against a
        grazing ray ducting/bouncing indefinitely); defaults to
        `max_path_m * nhops`, so nhops=1 reproduces the single-hop budget
        exactly as before.

        method="fixed": fixed-step RK4, supports scatter=True (Langevin
        ensemble kicks from self.spectrum). Does not support the tangent
        map.
        method="adaptive": solve_ivp RK45 with ground/ceiling/length
        events; required for carry_tangent_map=True. Does not support
        scatter=True.

        compute_phase_variance: accumulate sigma_phi2 (deterministic phase
        VARIANCE rate from the same spectrum used for scatter kicks) along
        whatever path is actually taken -- independent of scatter, and
        available on BOTH drivers (unlike scatter, which is fixed-only).
        Requires a spectrum (pass spectrum= to RayTracer()).
        """
        if scatter and method != "fixed":
            raise ValueError("scatter=True requires method='fixed'.")
        if scatter and self.spectrum is None:
            raise ValueError("scatter=True requires a spectrum (pass spectrum= to RayTracer()).")
        if compute_phase_variance and self.spectrum is None:
            raise ValueError("compute_phase_variance=True requires a spectrum "
                              "(pass spectrum= to RayTracer()).")
        if carry_tangent_map and method != "adaptive":
            raise ValueError("carry_tangent_map=True requires method='adaptive'.")

        r0 = np.asarray(r0, dtype=float)
        u0 = np.asarray(u0, dtype=float)
        launched_r = np.linalg.norm(r0)
        max_total_path_m = max_total_path_m if max_total_path_m is not None else max_path_m * nhops

        tail = [np.eye(6, dtype=float).reshape(-1)] if carry_tangent_map else []
        state = np.concatenate([r0, u0, [0.0, 0.0, 0.0, 0.0, 0.0]] + tail)

        all_pts = [state[:3].copy()]
        all_s = [0.0]
        bounce_points = []
        n_kicks_total = 0
        theta2_accum_total = 0.0

        result = TraceResult()
        hop = 0

        while True:
            remaining_total = max_total_path_m - state[6]
            if remaining_total <= 1e-6:
                result.safety_cap_hit = True
                break
            s_budget = min(max_path_m, remaining_total)

            if method == "fixed":
                seg = self._segment_fixed_step_rk4(
                    state, s_budget, scatter=scatter,
                    want_group=compute_group_path, want_absorption=compute_absorption,
                    want_phase_variance=compute_phase_variance,
                    ceiling_alt_m=ceiling_alt_m,
                    launched_r=launched_r,
                )
            elif method == "adaptive":
                seg = self._segment_adaptive(
                    state, s_budget,
                    want_group=compute_group_path, want_absorption=compute_absorption,
                    want_phase_variance=compute_phase_variance,
                    want_tangent=carry_tangent_map,
                    ceiling_alt_m=ceiling_alt_m,
                    launched_r=launched_r,
                )
            else:
                raise ValueError(f"unknown method {method!r}")

            all_pts.extend(list(seg.path[1:]))
            all_s.extend(list(seg.s_path[1:]))
            n_kicks_total += seg.n_kicks
            theta2_accum_total += seg.theta2_accum
            state = seg.final_state

            if seg.reason == "CEILING":
                result.escaped = True
                break
            if seg.reason == "STALL":
                break
            if seg.reason == "GROUND":
                hop += 1
                bounce_points.append(state[:3].copy())
                if hop >= nhops:
                    result.landed = True
                    break
                r_vec = state[:3]
                n_hat = ellipsoid_outward_normal_m(r_vec)
                u_vec = state[3:6]
                state = state.copy()
                state[3:6] = u_vec - 2.0 * np.dot(u_vec, n_hat) * n_hat
                continue
            if seg.reason == "LENGTH_BUDGET":
                break   # per-hop budget exhausted without landing or escaping

        result.hops_completed = hop
        result.r_final = state[:3]
        result.u_final = state[3:6]
        result.s_total = float(state[6])
        result.phase_path_total = float(state[7])
        result.group_path_total = float(state[8])
        result.absorption_amplitude_nepers_total = float(state[9])
        result.absorption_db_total = 8.686 * float(state[9])
        result.ground_range_total_m = great_circle_range_m(result.r_final, r0)
        result.doppler_shift_hz = None
        result.sigma_phi2 = float(state[_SIGMA_PHI2_IDX])
        result.n_kicks = n_kicks_total
        result.theta2_accum = theta2_accum_total
        result.path = np.array(all_pts)
        result.s_path = np.array(all_s)
        result.bounce_points = np.array(bounce_points) if bounce_points else np.zeros((0, 3))
        if carry_tangent_map:
            result.tangent_map_final = state[_PHI_OFFSET:].reshape(6, 6)
        return result


def compute_doppler(tracer: RayTracer, r0, u0, max_path_m, dt1, dt2, **trace_kwargs):
    """
    Phase-1.5 stub for the physically-correct Doppler definition:
    f_D = -(f0/c) * d(phase_path)/dt, estimated by tracing at two IRI
    epochs (dt1, dt2) through ionospheres built at those epochs and
    finite-differencing phase_path_total.

    NOT implemented yet: requires building two Ionosphere/Ray instances
    (one per epoch) and re-running .trace() through each, which this
    function's caller must arrange (build_ionosphere_for_ray takes `dt`).
    Left unimplemented deliberately per Phase 1 scope; wire this up once
    Phase 1's engine is validated.
    """
    raise NotImplementedError(
        "compute_doppler is a Phase 1.5 stub — see module docstring. "
        "TraceResult.doppler_shift_hz is None in Phase 1."
    )
