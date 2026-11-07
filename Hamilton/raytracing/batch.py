"""
batch.py -- vectorised fixed-step ray tracing for many rays at once (Phase 1a; spec §4.1, §4.8).

The algorithm is that of raytracer.RayTracer._segment_fixed_step_rk4 and RayTracer.trace
(scalar path), applied to a batch of rays with per-ray masks for step rejection, ground
crossing, ceiling and budget, and for the hop loop. Equivalence with the scalar path is
tested against golden outputs in tests/test_vectorised.py.

Records (Phase 1b). With record_states=True every accepted step is kept (state at each
step, hop index), the crossing state replaces the step that passed the ground, and the
reflected state follows as the next sample. samples_from_record() turns a record into a
RaySamples (spec Listing 1) and transport_frame() applies Algorithm 1 (spec §4.3).

Scope: deterministic tracing. Phase variance and Monte-Carlo kicks stay in the scalar
path (raytracer.py), which is deprecated with mc_scattering.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from raytracing.absorption import absorption_rate
from core.constants import C
from core.geometry import R_MEAN, WGS84_A, WGS84_B, xyz_m_to_latlon
from raytracing.ground import make_bounce
from core.records import RaySamples

LENGTH, GROUND, CEILING, STALL = 0, 1, 2, 3
REASON_NAMES = {LENGTH: "LENGTH_BUDGET", GROUND: "GROUND", CEILING: "CEILING", STALL: "STALL"}
EXTENDED = 12          # x y z ux uy uz s P_phase P_group A_amp Doppler-integral A_amp(h < 120 km)
X_VACUUM_EPS = 1e-6    # must match main.X_VACUUM_EPS


def _quadric(r, A, B):
    """x^2/A^2 + y^2/A^2 + z^2/B^2 - 1 (negative inside, positive outside)."""
    return (r[..., 0] ** 2 + r[..., 1] ** 2) / (A * A) + r[..., 2] ** 2 / (B * B) - 1.0


def _outward_normal(r):
    """Outward unit normal of the WGS84 ellipsoid at (or near) r, shape (..., 3)."""
    n = np.stack([r[..., 0] / WGS84_A ** 2, r[..., 1] / WGS84_A ** 2, r[..., 2] / WGS84_B ** 2], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _alt_m(r):
    """Height above the WGS84 ellipsoid along the geocentric direction of r [m] (apex tracking)."""
    rn = np.linalg.norm(r, axis=-1)
    u = r / rn[..., None]
    rho = 1.0 / np.sqrt((u[..., 0] ** 2 + u[..., 1] ** 2) / WGS84_A ** 2 + u[..., 2] ** 2 / WGS84_B ** 2)
    return rn - rho


def _great_circle_m(r, r0):
    """Spherical great-circle distance (mean radius), as in geometry.great_circle_range_m."""
    p = r / np.linalg.norm(r, axis=-1, keepdims=True)
    q = r0 / np.linalg.norm(r0)
    return R_MEAN * np.arccos(np.clip(p @ q, -1.0, 1.0))


@dataclass
class BatchResult:
    """Per-ray results of BatchTracer.trace (arrays of length N)."""

    landed: np.ndarray
    escaped: np.ndarray
    safety_cap_hit: np.ndarray
    hops: np.ndarray
    reason: np.ndarray                 # last segment's reason code, see REASON_NAMES
    r_final: np.ndarray                # (N, 3)
    u_final: np.ndarray                # (N, 3)
    s_total: np.ndarray                # arc length, m
    phase_path: np.ndarray             # m
    group_path: np.ndarray             # m
    absorption_amplitude: np.ndarray   # amplitude nepers (one-way)
    absorption_db: np.ndarray          # dB (one-way) = 8.686 * amplitude nepers
    ground_range_m: np.ndarray
    bounce_points: list = field(default_factory=list)   # per ray: (k, 3) arrays
    paths: list = field(default_factory=list)           # per ray: (n, 3) arrays (if record_every)
    records: list = field(default_factory=list)         # per ray: dict(states, hop, bounce_index, normals, bounces)
    bounces: list = field(default_factory=list)         # per ray: list of Bounce (if bounce_models)

    doppler_integral: np.ndarray | None = None   # rad/s: integral of g dNe/dt ds (spec eq. 4.8)
    doppler_hz: np.ndarray | None = None         # -doppler_integral / 2 pi, Hz (spec §4.6)
    doppler_hops_hz: list = field(default_factory=list)   # per ray: Hz of each hop, aligned with bounce_points
    hop_states: list = field(default_factory=list)        # per ray: (hops, 11) full state at each ground contact
    hop_apex: list = field(default_factory=list)          # per ray: (hops,) apex height of each hop [m]
    absorption_D_db: np.ndarray | None = None             # one-way dB from heights below 120 km
    att_per_wavelength_max: np.ndarray | None = None      # max over steps of k0|Im n| lambda0 (spec §7.3)

    @property
    def N(self) -> int:
        return int(self.landed.size)


@dataclass
class DopplerRates:
    """
    Inputs of the Doppler accumulator (spec §4.6, Method B, eq. 4.8).

    rate(r_ecef) -> dNe/dt at fixed r from the model evolution (m^-3 s^-1). For a two-epoch
    model it is the central difference of two snapshots on one grid, [Ne(t+dt)-Ne(t-dt)]/(2dt).
    drift(r_ecef) -> v_d (m/s, ECEF). It adds the convective term -v_d . grad(Ne), with
    grad(Ne) taken from the RHS. Either may be None.
    """

    rate: Callable | None = None
    drift: Callable | None = None


class BatchTracer:
    """Fixed-step RK4 with step rejection, for a batch of rays through one Ray."""

    def __init__(self, ray, *, ds_m: float = 2000.0, absorption_model="default",
                 surface=None, f0: float | None = None, doppler: DopplerRates | None = None,
                 landing_alt_m: float = 0.0):
        """
        surface: callable(lat, lon, t) -> dict(surface, eps_r, sigma_g), used for bounce records
        (default environment.SeaLandSurface). f0: carrier [Hz] for the Fresnel coefficients.
        doppler: DopplerRates switches on the Doppler accumulator (state slot 10).
        """
        from geophysics.environment import SeaLandSurface
        self.ray = ray
        self.ds = float(ds_m)
        self.absorption_model = absorption_model
        self.surface = surface if surface is not None else SeaLandSurface()
        self.f0 = float(f0) if f0 is not None else ray.omega / (2.0 * np.pi)
        self.doppler = doppler
        self.landing_alt_m = float(landing_alt_m)   # aircraft target height h_T (spec §5.3); 0 = ground

    # ------------------------------------------------------------------
    # Extended right-hand side (batch): base RHS + group path + absorption
    # ------------------------------------------------------------------

    def _rhs(self, s, want_group: bool, want_absorption: bool):
        phys = s[..., :6]
        cache = {}                       # centre values shared by the three calls below
        f = self.ray._haselgrove_physical_rhs(0.0, phys, _cache=cache)
        rdot = f[..., :3]
        u = phys[..., 3:6]
        dsdt = np.linalg.norm(rdot, axis=-1)
        dphase = np.sum(u * rdot, axis=-1)
        if want_group:
            dgroup = self.ray.group_index(phys, _cache=cache) * dsdt
        else:
            dgroup = np.zeros_like(dsdt)
        if want_absorption:
            dabs = absorption_rate(self.ray, phys[..., :3], u, model=self.absorption_model,
                                   rdot=rdot, _cache=cache) * dsdt
        else:
            dabs = np.zeros_like(dsdt)
        out = np.empty(s.shape)
        out[..., :6] = f
        out[..., 6] = dsdt
        out[..., 7] = dphase
        out[..., 8] = dgroup
        out[..., 9] = dabs
        if self.doppler is not None:
            out[..., 10] = self._doppler_rate(phys, rdot, cache) * dsdt
        else:
            out[..., 10] = 0.0
        below = _alt_m(phys[..., :3]) < 120e3                  # absorption diagnostics (spec §7.3, item 4)
        out[..., 11] = np.where(below, dabs, 0.0)
        return out

    def _doppler_rate(self, phys, rdot, cache):
        """
        g dNe/dt per metre of path (spec eq. 4.8), shape (M,) for (M, 6) states. g comes from
        phase_sensitivity on the RHS's own centre values (no extra RHS call); grad(Ne) for the
        convective term is the one the RHS cached.
        """
        r = phys[..., :3]
        g = self.ray.phase_sensitivity(phys, rdot=rdot, _cache=cache)
        dne = np.zeros(r.shape[:-1])
        if self.doppler.rate is not None:
            dne = dne + np.asarray(self.doppler.rate(r), dtype=float)
        if self.doppler.drift is not None:
            vd = np.asarray(self.doppler.drift(r), dtype=float)
            dne = dne - np.sum(vd * cache["grad_ne"], axis=-1)
        return g * dne

    # ------------------------------------------------------------------
    # Ground crossing (bisection on the chord, as in RayTracer._locate_ground_crossing)
    # ------------------------------------------------------------------

    @staticmethod
    def _locate(prev, nxt, A=WGS84_A, B=WGS84_B):
        f_lo = _quadric(prev[:, :3], A, B)
        lo = np.zeros(prev.shape[0])
        hi = np.ones(prev.shape[0])
        d = nxt[:, :3] - prev[:, :3]
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            r_mid = prev[:, :3] + mid[:, None] * d
            above = _quadric(r_mid, A, B) > 0.0
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        lam = 0.5 * (lo + hi)
        res = prev + lam[:, None] * (nxt - prev)
        return np.where((f_lo <= 0.0)[:, None], prev, res)

    # ------------------------------------------------------------------
    # One segment (until ground, ceiling, budget or stall), batch
    # ------------------------------------------------------------------

    def _segment(self, st0, s_budget, launched_r, A_ceil, B_ceil, want_group, want_absorption,
                 record_every: int, rec, srec=None, grec=None, apex=None, attmax=None, alt_mask=None):
        """
        rec: per-ray path-point lists (record_every). srec: per-ray state lists (every
        accepted step). grec: per-ray list of crossing indices (filled by trace()).
        apex: (M,) running maximum radius of each ray in this hop, updated in place.
        """
        M = st0.shape[0]
        st = st0.copy()
        s0 = st[:, 6].copy()
        ds = self.ds
        reason = np.full(M, LENGTH, dtype=int)
        active = np.ones(M, dtype=bool)
        step_i = np.zeros(M, dtype=int)

        while True:
            active &= (st[:, 6] - s0) < s_budget
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            s = st[idx]
            f1 = self._rhs(s, want_group, want_absorption)
            v1 = np.linalg.norm(f1[:, :3], axis=1)
            stall = v1 < 1e-12
            if stall.any():
                reason[idx[stall]] = STALL
                active[idx[stall]] = False
            keep = ~stall
            idx, s, f1, v1 = idx[keep], s[keep], f1[keep], v1[keep]
            if idx.size == 0:
                continue

            # step with rejection, per ray (the scalar loop of up to 60 halvings)
            dtau = ds / v1
            new = np.empty_like(s)
            ok = np.zeros(idx.size, dtype=bool)
            pending = np.arange(idx.size)
            for _ in range(60):
                if pending.size == 0:
                    break
                sp = s[pending]
                f1p = f1[pending]
                dt = dtau[pending][:, None]
                f2 = self._rhs(sp + 0.5 * dt * f1p, want_group, want_absorption)
                f3 = self._rhs(sp + 0.5 * dt * f2, want_group, want_absorption)
                f4 = self._rhs(sp + dt * f3, want_group, want_absorption)
                cand = sp + (dt / 6.0) * (f1p + 2.0 * f2 + 2.0 * f3 + f4)
                dsact = np.linalg.norm(cand[:, :3] - sp[:, :3], axis=1)
                acc = dsact <= 5.0 * ds
                new[pending[acc]] = cand[acc]
                ok[pending[acc]] = True
                rej = pending[~acc]
                dtau[rej] *= 0.5
                pending = rej
            if pending.size:
                reason[idx[pending]] = STALL
                active[idx[pending]] = False

            good = np.flatnonzero(ok)
            if good.size == 0:
                continue
            gi = idx[good]
            prev = s[good]
            nxt = new[good]
            st[gi] = nxt
            step_i[gi] += 1
            if apex is not None:
                apex[gi] = np.maximum(apex[gi], _alt_m(nxt[:, :3]))
            if attmax is not None and want_absorption:
                _, att = absorption_rate(self.ray, nxt[:, :3], nxt[:, 3:6], model=self.absorption_model,
                                         return_att=True)
                attmax[gi] = np.maximum(attmax[gi], att)
            if srec is not None:
                for j, i in enumerate(gi):
                    srec[i].append(nxt[j].copy())
            if record_every:
                for j in np.flatnonzero(step_i[gi] % record_every == 0):
                    rec[gi[j]].append(nxt[j, :3].copy())

            ground = (_quadric(nxt[:, :3], WGS84_A, WGS84_B) <= 0.0) & (nxt[:, 6] - s0[gi] > 2.0 * ds)
            A_s = np.full(gi.size, WGS84_A)
            B_s = np.full(gi.size, WGS84_B)
            if self.landing_alt_m > 0.0 and alt_mask is not None and alt_mask[gi].any():
                # aircraft surface (§5.3): a descending crossing of the ellipsoid offset by h_T on the final hop
                Ao, Bo = WGS84_A + self.landing_alt_m, WGS84_B + self.landing_alt_m
                air = (alt_mask[gi] & (_quadric(prev[:, :3], Ao, Bo) > 0.0)
                       & (_quadric(nxt[:, :3], Ao, Bo) <= 0.0) & (nxt[:, 6] - s0[gi] > 2.0 * ds))
                A_s[air] = Ao
                B_s[air] = Bo
                ground = ground | air
            if ground.any():
                gg = gi[ground]
                crossing = self._locate(prev[ground], nxt[ground], A_s[ground], B_s[ground])
                st[gg] = crossing
                reason[gg] = GROUND
                active[gg] = False
                if srec is not None:
                    for j, i in enumerate(gg):
                        srec[i][-1] = crossing[j].copy()       # the step that passed the ground
            rest = ~ground
            if rest.any():
                rr = gi[rest]
                rn = np.linalg.norm(nxt[rest, :3], axis=1)
                qc = _quadric(nxt[rest, :3], A_ceil, B_ceil)
                ceil = (qc > 0.0) & (rn > launched_r[rr])
                if ceil.any():
                    reason[rr[ceil]] = CEILING
                    active[rr[ceil]] = False

        if record_every:
            for j in range(M):
                rec[j].append(st[j, :3].copy())
        return st, reason

    # ------------------------------------------------------------------
    # Full trace with hops (mirrors RayTracer.trace)
    # ------------------------------------------------------------------

    def trace(self, r0, u0, max_path_m: float, *, nhops: int = 1, max_total_path_m: float | None = None,
              ceiling_alt_m: float = 800e3, want_group: bool = True, want_absorption: bool = True,
              record_every: int = 0, record_states: bool = False, bounce_models: bool = False) -> BatchResult:
        """
        Trace N rays from one launch point r0 (3,) with directions u0 (N, 3), through up to
        nhops ground bounces. max_path_m is the budget per hop; max_total_path_m (default
        max_path_m * nhops) caps the total. record_every > 0 keeps path points; record_states
        keeps every accepted step (for samples_from_record); bounce_models adds Bounce records
        (surface constants from self.surface at each ground contact).
        """
        r0 = np.asarray(r0, dtype=float)
        u0 = np.atleast_2d(np.asarray(u0, dtype=float))
        N = u0.shape[0]
        launched_r = np.full(N, np.linalg.norm(r0))
        max_total = max_path_m * nhops if max_total_path_m is None else max_total_path_m
        A_ceil, B_ceil = WGS84_A + ceiling_alt_m, WGS84_B + ceiling_alt_m

        states = np.zeros((N, EXTENDED))
        states[:, :3] = r0
        states[:, 3:6] = u0
        hop = np.zeros(N, dtype=int)
        done = np.zeros(N, dtype=bool)
        landed = np.zeros(N, dtype=bool)
        escaped = np.zeros(N, dtype=bool)
        cap = np.zeros(N, dtype=bool)
        last_reason = np.full(N, LENGTH, dtype=int)
        bounce = [[] for _ in range(N)]
        paths = [[r0.copy()] for _ in range(N)] if record_every else None
        srec = [[states[i].copy()] for i in range(N)] if record_states else None
        hop_rec = [[0] for _ in range(N)] if record_states else None
        cross_rec = [[] for _ in range(N)] if record_states else None
        normals = [[] for _ in range(N)] if record_states else None
        bounces = [[] for _ in range(N)] if bounce_models else None
        hop_states = [[] for _ in range(N)]
        hop_apex = [[] for _ in range(N)]
        apex_all = _alt_m(states[:, :3])
        attmax_all = np.zeros(N)
        dop_cum = [[] for _ in range(N)] if self.doppler is not None else None

        while True:
            remaining = max_total - states[:, 6]
            capped = (~done) & (remaining <= 1e-6)
            cap[capped] = True
            done[capped] = True
            act = np.flatnonzero(~done)
            if act.size == 0:
                break
            s_budget = np.minimum(max_path_m, remaining[act])
            rec = [[] for _ in range(act.size)] if record_every else None
            sr = [srec[i] for i in act] if record_states else None
            n_before = [len(srec[i]) for i in act] if record_states else None
            ap = apex_all[act].copy()
            ap2 = attmax_all[act].copy()
            st_new, reason = self._segment(states[act], s_budget, launched_r[act], A_ceil, B_ceil,
                                           want_group, want_absorption, record_every, rec, srec=sr,
                                           apex=ap, attmax=ap2, alt_mask=(hop[act] == nhops - 1))
            states[act] = st_new
            apex_all[act] = ap
            attmax_all[act] = ap2
            last_reason[act] = reason
            if record_every:
                for j, i in enumerate(act):
                    paths[i].extend(rec[j])
            if record_states:
                for j, i in enumerate(act):
                    for _ in range(len(srec[i]) - n_before[j]):
                        hop_rec[i].append(int(hop[i]))

            esc = act[reason == CEILING]
            escaped[esc] = True
            done[esc] = True
            done[act[reason == STALL]] = True
            done[act[reason == LENGTH]] = True
            gro = act[reason == GROUND]
            if gro.size:
                hop[gro] += 1
                for i in gro:
                    bounce[i].append(states[i, :3].copy())
                    hop_states[i].append(states[i].copy())
                    hop_apex[i].append(apex_all[i])
                    if dop_cum is not None:
                        dop_cum[i].append(states[i, 10])
                    if record_states:
                        cross_rec[i].append(len(srec[i]) - 1)
                fin = gro[hop[gro] >= nhops]
                landed[fin] = True
                done[fin] = True
                cont = gro[hop[gro] < nhops]
                if cont.size:
                    n_hat = _outward_normal(states[cont, :3])
                    u = states[cont, 3:6]
                    if bounce_models:
                        for k, i in enumerate(cont):
                            s_in = self._direction(states[i])
                            lat, lon, _ = xyz_m_to_latlon(states[i, :3])
                            env = self.surface.at(lat, lon, None)
                            bounces[i].append(make_bounce(
                                len(srec[i]) - 1 if record_states else -1, states[i, :3], s_in,
                                n_hat[k], eps_r=float(np.atleast_1d(env["eps_r"])[0]),
                                sigma_g=float(np.atleast_1d(env["sigma_g"])[0]),
                                surface=str(np.atleast_1d(env["surface"])[0]), f0=self.f0))
                    if record_states:
                        for k, i in enumerate(cont):
                            normals[i].append(n_hat[k].copy())
                    states[cont, 3:6] = u - 2.0 * np.sum(u * n_hat, axis=1)[:, None] * n_hat
                    apex_all[cont] = _alt_m(states[cont, :3])
                    if record_states:
                        for i in cont:
                            srec[i].append(states[i].copy())
                            hop_rec[i].append(int(hop[i]))

        r_final = states[:, :3].copy()
        u_final = states[:, 3:6].copy()
        A_amp = states[:, 9].copy()
        A_D = states[:, 11].copy()
        dop = states[:, 10].copy() if self.doppler is not None else None
        records = []
        if record_states:
            for i in range(N):
                records.append(dict(states=np.array(srec[i]), hop=np.array(hop_rec[i], dtype=int),
                                    bounce_index=np.array(cross_rec[i], dtype=int),
                                    normals=np.array(normals[i]) if normals[i] else np.zeros((0, 3))))
        return BatchResult(
            landed=landed, escaped=escaped, safety_cap_hit=cap, hops=hop, reason=last_reason,
            r_final=r_final, u_final=u_final, s_total=states[:, 6].copy(),
            phase_path=states[:, 7].copy(), group_path=states[:, 8].copy(),
            absorption_amplitude=A_amp, absorption_db=8.686 * A_amp,
            ground_range_m=_great_circle_m(r_final, r0),
            bounce_points=[np.array(b) if b else np.zeros((0, 3)) for b in bounce],
            paths=[np.array(p) for p in paths] if record_every else [],
            records=records,
            bounces=bounces if bounce_models else [[] for _ in range(N)],
            hop_states=[np.asarray(h, dtype=float).reshape(-1, EXTENDED) for h in hop_states],
            hop_apex=[np.asarray(a, dtype=float) for a in hop_apex],
            absorption_D_db=8.686 * A_D,
            att_per_wavelength_max=attmax_all,
            doppler_integral=dop, doppler_hz=None if dop is None else -dop / (2.0 * np.pi),
            doppler_hops_hz=[] if dop_cum is None else [
                -np.diff(np.concatenate([[0.0], np.asarray(c, dtype=float)])) / (2.0 * np.pi) for c in dop_cum],
        )

    def _direction(self, state):
        """Physical ray direction at one state (unit vector)."""
        f = self.ray._haselgrove_physical_rhs(0.0, state[:6])[:3]
        return f / np.linalg.norm(f)


# ---------------------------------------------------------------------------
# Records -> RaySamples (spec Listing 1, Table 4) and frame transport (Algorithm 1)
# ---------------------------------------------------------------------------

def transport_frame(s_hat: np.ndarray, e_up: np.ndarray, mirror_at=(), normals=()):
    """
    Algorithm 1 (spec §4.3): transverse frame along a traced leg, initialised at the array end
    and parallel-transported (projection onto the local ray direction, no twist). At each
    ground reflection both vectors are mirrored in the local ground plane, at the sample
    after the bounce (the first sample with the reflected direction). Returns (t1, t2), each
    (N, 3). If the launch direction is vertical, t1 is east (spec note ii).
    """
    s_hat = np.asarray(s_hat, dtype=float)
    N = s_hat.shape[0]
    s0 = s_hat[0]
    t1 = np.cross(s0, e_up)
    if np.linalg.norm(t1) < 1e-9:
        t1 = np.array([0.0, 1.0, 0.0]) if abs(s0[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(t1, s0)
    t2 = t2 / np.linalg.norm(t2)
    mirror = {int(j): np.asarray(nh, dtype=float) for j, nh in zip(mirror_at, normals)}
    T1 = np.empty((N, 3))
    T2 = np.empty((N, 3))
    T1[0], T2[0] = t1, t2
    for j in range(1, N):
        if j in mirror:
            nh = mirror[j]
            t1 = t1 - 2.0 * np.dot(t1, nh) * nh
            t2 = t2 - 2.0 * np.dot(t2, nh) * nh
        sj = s_hat[j]
        t1 = t1 - np.dot(t1, sj) * sj
        t1 = t1 / np.linalg.norm(t1)
        t2 = t2 - np.dot(t2, sj) * sj - np.dot(t2, t1) * t1
        t2 = t2 / np.linalg.norm(t2)
        T1[j], T2[j] = t1, t2
    return T1, T2


def samples_from_record(tracer: BatchTracer, rec: dict, *, n_min: float = 0.1,
                        e_up: np.ndarray | None = None, transport: bool = True) -> RaySamples:
    """Build RaySamples for one ray from its record (BatchResult.records[i])."""
    ray = tracer.ray
    S = rec["states"]
    r = S[:, :3]
    u = S[:, 3:6]
    phys = S[:, :6]
    f = ray._haselgrove_physical_rhs(0.0, phys)[:, :3]
    fnorm = np.linalg.norm(f, axis=1)
    s_hat = f / np.where(fnorm > 0.0, fnorm, 1.0)[:, None]
    n = np.linalg.norm(u, axis=1)
    u_hat = u / n[:, None]

    ne, Y_vec, grads = ray._fields(r[:, 0], r[:, 1], r[:, 2], grad=True)
    X = ne * ray.X_factor
    B = Y_vec / ray.Y_factor
    grad_ne = grads[0]
    lat, lon, alt = xyz_m_to_latlon(np.moveaxis(r, -1, 0))
    A = S[:, 9]
    dA_dB = np.concatenate([[0.0], 8.686 * np.diff(A)])
    if transport:
        if e_up is None:
            e_up = r[0] / np.linalg.norm(r[0])
        t1, t2 = transport_frame(s_hat, e_up, rec["bounce_index"] + 1, rec["normals"])
    else:
        t1 = t2 = None
    return RaySamples(
        r=r.copy(), s=S[:, 6].copy(), P=S[:, 7].copy(), Pg=S[:, 8].copy(),
        s_hat=s_hat, u_hat=u_hat, n=n, X=X, B=B, Ne=ne, grad_Ne=grad_ne,
        alt=np.asarray(alt, dtype=float), hop=rec["hop"].copy(), dA_dB=dA_dB,
        absorption_db=8.686 * A, vacuum=X <= X_VACUUM_EPS, steep=n < n_min,
        t1=t1, t2=t2, bounce_index=rec["bounce_index"].copy(),
    )
