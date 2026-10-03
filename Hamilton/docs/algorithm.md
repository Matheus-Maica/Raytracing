# Hamilton — how the algorithm works

Hamilton is a Python HF (≈2–30 MHz) ionospheric ray tracer. It follows a radio ray through a
magnetised, inhomogeneous, collisionless plasma by integrating Haselgrove's (1954) Hamiltonian
ray equations, with the electron density taken from **E-CHAIM** (high latitude) and
**IRI-2020** (everywhere else), the magnetic field from **IGRF-13**, over **WGS84** geometry.
On top of the deterministic ray it computes phase/group path, non-deviative absorption,
ray-tube focusing, and a Monte Carlo model of scattering by small-scale irregularities.

This document explains **what is computed, why it is computed that way, and where in the code**.
It is written for two readers at once: physicists (the equations, the approximations, what
the numbers mean) and programmers (data flow, numerics, module boundaries, pitfalls). Sections
marked **▸ Code** map a piece of physics to the file that implements it.

> **Status of this document.** It describes the code as reviewed on 2026-09-19. Section 8 lists
> what the review found, including two defects that are *not yet fixed*
> (§8.1 — absorption is reported 2× too large in dB; §8.2 — the group path is locally wrong near
> reflection). Read those before quoting any absorption or group-delay number.

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Physical model](#2-physical-model)
3. [Environment: geometry, electron density, magnetic field](#3-environment-geometry-electron-density-magnetic-field)
4. [Numerical methods](#4-numerical-methods)
5. [Software architecture](#5-software-architecture)
6. [Using it](#6-using-it)
7. [Validation](#7-validation)
8. [Known defects, limitations, and pitfalls](#8-known-defects-limitations-and-pitfalls)
9. [References](#9-references)

---

## 1. The big picture

**Question answered.** A transmitter at a given place and time, carrier frequency $f$, and
launch angles (elevation, azimuth), in ordinary (O) or extraordinary (X) mode: *where does the
ray come back down, how long did it take, how much was it absorbed, how strongly is it focused,
and how much do irregularities smear it?*

**Pipeline.**

```
   Link(lat, lon, f, dt, mode)                       ← api.py   (one-time, expensive)
        │
        ├─ build_ionosphere_for_ray  ─► Ionosphere   ← main.py / ionosphere.py
        │        E-CHAIM (AACGM mlat ≥ 45°) else IRI-2020, on a coarse grid,
        │        trilinearly upsampled to a 2 km × 1° × 1° grid   →  Ne(x,y,z)
        │
        └─ build_igrf_for_ionosphere ─► IgrfField    ← igrf.py
                 IGRF-13 on a 5×5×5 grid, linear interpolation  →  B(x,y,z)
                                  │
                                  ▼
                       Ray(ω, mode, Ne, B)           ← main.py   the PHYSICS
                       ├─ Appleton–Hartree in Haselgrove's homogeneous form
                       └─ RHS:  (r,u) ↦ (dr/dτ, du/dτ)
                                  │
                                  ▼
                       RayTracer(ray, spectrum)      ← raytracer.py   the ENGINE
                       ├─ extended state: r, u, arc, phase path, group path,
                       │                  absorption, phase variance   (11 numbers)
                       ├─ driver A: fixed-step RK4  (+ Langevin kicks for scattering)
                       ├─ driver B: adaptive RK45 + arc-length tangent-map pass
                       └─ hop loop: ground / ceiling / budget events, specular bounce
                                  │
              ┌───────────────────┼─────────────────────┐
              ▼                   ▼                     ▼
        RayPath (trace)   ScatterResult (scatter)   LinkBudget (link_budget)
```

Everything after the constructor of `Link` reuses the same density and field patch, so tracing
another ray, another frequency, or the other mode is comparatively cheap (seconds per ray, see §4.6).

**Three design decisions worth knowing up front.**

1. *One physics function, two drivers.* The deterministic tracer, the Monte Carlo tracer, and
   the tangent-map tracer all evaluate the same right-hand side
   (`RayTracer._extended_rhs`). A change to the dispersion relation or to absorption reaches all
   of them at once; they cannot silently diverge.
2. *The ray is always traced loss-free.* Absorption and phase variance are integrals accumulated
   along the already-determined real ray ("non-deviative"). Loss never bends the path.
3. *The independent variable is not arc length.* Haselgrove's ray parameter $\tau$ has units of
   metres but $ds/d\tau = |\dot{\mathbf r}|$ is not 1 — it goes to ~0 in weak plasma. Much of the
   numerical design (§4) exists to handle that.

---

## 2. Physical model

### 2.1 Geometrical optics as a Hamiltonian system

For a wave $\propto e^{i(\mathbf k\cdot\mathbf r-\omega t)}$ in a slowly varying medium, $\omega$
and the wave-vector $\mathbf k(\mathbf r)$ satisfy a local dispersion relation. Hamilton's
equations for a ray follow from it. Hamilton uses the **refractive-index vector**

$$
\mathbf u \equiv \frac{c\,\mathbf k}{\omega},\qquad |\mathbf u| = n,\quad p \equiv n^2 = \mathbf u\cdot\mathbf u ,
$$

and a dispersion function $F(\mathbf r,\mathbf u)$ that vanishes on the dispersion surface
$F=0$ (§2.3). The ray equations are

$$
\frac{d\mathbf r}{d\tau}=\frac{\partial F}{\partial\mathbf u},\qquad
\frac{d\mathbf u}{d\tau}=-\frac{\partial F}{\partial\mathbf r}. \tag{1}
$$

$\mathbf r$ is geocentric Cartesian (ECEF) metres. $\tau$ is a bookkeeping parameter; the
physical rate is the geometric arc length $ds=|\dot{\mathbf r}|\,d\tau$. In an anisotropic medium
$\dot{\mathbf r}$ (the **ray / group-velocity direction**) is not parallel to $\mathbf u$
(the **wave normal**) — the angle between them is what the code later calls the
ray/wave-normal splitting angle.

### 2.2 Cold magnetoplasma and the dimensionless parameters

Cold, collisionless electron plasma in a static field $\mathbf B$. With $q,m_e$ the electron
charge and mass, $N_e$ the density and $\omega=2\pi f$:

$$
X=\frac{N_e q^2}{\varepsilon_0 m_e\omega^2}=\frac{f_p^2}{f^2},\qquad
\mathbf Y=\frac{q}{m_e\omega}\,\mathbf B,\quad Y=|\mathbf Y|=\frac{f_c}{f}.
$$

Only $X$ and $\mathbf Y$ depend on the position; $X\propto N_e(\mathbf r)$ and $\mathbf Y\propto\mathbf B(\mathbf r)$.
**▸ Code:** `Ray.X_factor`, `Ray.Y_factor`, `Ray.Y_vec` in `main.py`.

### 2.3 The dispersion relation as a homogeneous polynomial

In Stix notation the cold-plasma relation between $n^2$ and the angle $\theta$ between $\mathbf k$
and $\mathbf B$ is

$$
\sin^2\!\theta\,(S n^2-RL)(n^2-P)+\cos^2\!\theta\,P\,(n^2-R)(n^2-L)=0 ,
$$

with $S,P,R,L$ the usual functions of $X,Y$ (e.g. $P=1-X$, $R,L=1-X/(1\mp Y)$).
The textbook form solves this for $n^2(\theta)$ (Appleton–Hartree). Haselgrove's trick makes it
**polynomial in the components of $\mathbf u$**, so that $\partial F/\partial\mathbf u$ and
$\partial F/\partial\mathbf r$ exist in closed form: substitute
$\cos\theta = \hat{\mathbf Y}\!\cdot\!\mathbf u/\sqrt p$ and define
$Y_d\equiv\hat{\mathbf Y}\!\cdot\!\mathbf u=\sqrt p\cos\theta$. Clearing denominators gives a
quadratic in $p$:

$$
F(\mathbf r,\mathbf u)=a\,p^2+b\,p+c=0,\tag{2}
$$
$$
\begin{aligned}
a&=1-X-Y^2,\\
b&=-2X^2+XY^2Y_d^2-XY^2+4X+2Y^2-2,\\
c&=-X^3+3X^2-XY^2Y_d^2+XY^2-3X-Y^2+1 .
\end{aligned}
$$

The two roots $p_\pm=\big(-b\pm\sqrt{b^2-4ac}\big)/2a$ are the two magneto-ionic modes;
the code's `MODE` selects the sign: **`+1` ↔ O-mode (+√), `−1` ↔ X-mode (−√)**.
**▸ Code:** `Ray._appleton_hartree_p` (`main.py`).

Because $Y_d$ is itself a function of $\mathbf u$, (2) is a polynomial *surface in $\mathbf u$-space*
and eq. (1) applies to it directly. The coefficients were re-derived symbolically from the
dielectric tensor during this project (an earlier version failed the trivial vacuum limit); the
checks that pin them down are in §7.

**Two degeneracies you should know about.**

* At $X=0$: $F=(1-Y^2)(p-1)^2$ — a *double* root, $p=1$ for both modes and any $\mathbf B$.
  Then $\partial F/\partial\mathbf u=\partial F/\partial\mathbf r=0$ and eq. (1) is $0=0$.
* At $Y=0$: $F=(1-X)\,\big(p-(1-X)\big)^2$ — also a double root (the modes coincide in an
  isotropic plasma). The equations stay finite for real Earth fields ($Y\sim0.02$–$0.2$ across HF),
  but see §8.6.

### 2.4 The ray equations, explicitly

Differentiating (2) on the chosen root gives four scalars $J,K,L,M_{sc}$ (`J`, `K`, `L`, `Msc` in the code):

$$
\dot{\mathbf r}=J\,\mathbf u-K\,\mathbf Y,\qquad
\dot u_j = L\,\frac{q^2}{\varepsilon_0 m_e\omega^2}\,\partial_j N_e
+\sum_i\big(K u_i+M_{sc}Y_i\big)\,\partial_j Y_i .\tag{3}
$$

The first term of $\dot{\mathbf u}$ is refraction by the density gradient; the second is
refraction by the gradient of the geomagnetic field. Directionally, in an isotropic plasma
$\dot{\mathbf r}\parallel\mathbf u$ and $\dot{\mathbf u}\parallel\nabla N_e$.
**▸ Code:** `Ray._haselgrove_physical_rhs`.

### 2.5 The vacuum branch and the numerically stable form

Below the ionosphere, and wherever $N_e$ is a numerical residue (the interpolated density
routinely falls from $10^{-29}$-scale noise to a physical value within one grid cell), $J,K,L,M_{sc}$
are all proportional to $X$ and vanish at the double root of §2.3. Evaluating them naively
fails in two different ways, both observed during development: the ray *stalls*
($|\dot{\mathbf r}|$ underflows to exactly 0) or *overshoots* (the step $d\tau=ds/|\dot{\mathbf r}|$
explodes to compensate; $|\mathbf u|$ reached $10^{18}$ in one step). Hamilton therefore has two
safeguards:

1. **Exact vacuum branch.** For $X\le X_{\rm vac}=10^{-6}$ the RHS returns the closed-form
   vacuum solution $\dot{\mathbf r}=\mathbf u,\ \dot{\mathbf u}=0$ ($|\mathbf u|=1$, straight line).
   $X=10^{-6}$ means $f=1000\,f_p$: $n^2-1\sim10^{-6}$, immaterial for any bending the tracer resolves.
2. **The $\varepsilon$-form.** Write $p=1+\varepsilon$ and substitute into (2) *before* solving. Every
   coefficient acquires a factor of $X$:
   $$a\,\varepsilon^2+X B_\varepsilon\,\varepsilon+X^2(1-X)=0,\qquad B_\varepsilon=2-2X+Y^2Y_d^2-Y^2,$$
   so $\varepsilon=X\big(-B_\varepsilon\pm\sqrt{B_\varepsilon^2-4a(1-X)}\big)/2a$ is $X$ times an
   $O(1)$ bracket — never a difference of two nearly equal large numbers — and
   $$J=2XB_\varepsilon+4a\varepsilon,\quad K=-2XY\,Y_d\,\varepsilon,\quad
   L=X(3X-2)+\varepsilon(2X-B_\varepsilon+\varepsilon),\quad M_{sc}=2\varepsilon(X+\varepsilon).$$
   These agree with the $p$-based forms to ≥4 digits where the latter are trustworthy
   ($X\gtrsim10^{-4}$) and remain accurate down to $X\sim10^{-29}$, where the naive form is noise.

**▸ Code:** `Ray._dispersion_eps` and the guard at the top of `_haselgrove_physical_rhs`.

### 2.6 What is integrated: the extended state

The engine integrates 11 numbers per ray (`raytracer.py`, module docstring):

| idx | symbol | meaning | rate w.r.t. $\tau$ |
|---|---|---|---|
| 0–2 | $x,y,z$ | position, ECEF m | $\dot{\mathbf r}$ |
| 3–5 | $u_x,u_y,u_z$ | index vector, $\lvert\mathbf u\rvert=n$ | $\dot{\mathbf u}$ |
| 6 | $s$ | geometric arc length | $\lvert\dot{\mathbf r}\rvert$ |
| 7 | $P_\phi$ | phase path | $\mathbf u\cdot\dot{\mathbf r}$ |
| 8 | $P_g$ | group path | $n_g\lvert\dot{\mathbf r}\rvert$ |
| 9 | $A$ | absorption (one-way, integrated rate) | $\alpha\,\lvert\dot{\mathbf r}\rvert$ |
| 10 | $\sigma_\phi^2$ | phase variance, rad² | $\rho_\phi\,\lvert\dot{\mathbf r}\rvert$ |

(A tangent-map block of 36 more numbers is appended only when requested — §2.10.)
The optional accumulators can be switched off; they then contribute zero but keep their slot,
so all indexing is by fixed position.

### 2.7 Path lengths

* **Arc length** $s=\int|\dot{\mathbf r}|\,d\tau$ — the geometric path.
* **Phase path** $P_\phi=\int\mathbf u\cdot d\mathbf r=\frac{1}{k_0}\int\mathbf k\cdot d\mathbf r$ — an
  exact eikonal identity, so it needs no extra physics: $dP_\phi/d\tau=\mathbf u\cdot\dot{\mathbf r}$.
* **Group path** $P_g=c\,t_{\rm group}$ — the path length corresponding to the actual signal delay.
  The code uses $dP_g=n_g\,ds$ with $n_g=n+\omega\,\partial n/\partial\omega$, the derivative taken
  by central difference ($\delta\omega=10^{-4}\omega$) of the same Appleton–Hartree solve.
  **This is approximate and locally wrong near reflection — see §8.2.**
  `RayPath.group_delay_ms` $=P_g/c$.

**▸ Code:** `RayTracer._extended_rhs`, `RayTracer._group_index`.

### 2.8 Absorption

Absorption is **non-deviative** (the same approach as PHaRLAP's "George" option): the ray is not
altered by loss; loss is integrated along it afterwards. At each point the code re-solves the *same*
quadratic (2) but with a complex, collision-damped density parameter (Budden 1961),

$$
X\;\to\;\frac{X}{1-iZ},\qquad Z=\frac{\nu_{\rm eff}(h)}{\omega},
$$

which makes $p=n^2$, and hence $n$, complex. Wave amplitude goes as $e^{-k_0\,\mathrm{Im}(n)\,s}$, so the
**power** attenuation coefficient is

$$
\alpha=2k_0\,\mathrm{Im}\,n\ \ [\mathrm{Np/m}],\qquad
A=\int\alpha\,ds,\qquad P(s)=P_0\,e^{-A}.
$$

(The sign of the square root is chosen so that $\mathrm{Im}\,n>0$ — the physically damped branch
for the $e^{-i(kz-\omega t)}$ convention. The first version got this backwards and produced negative
absorption; this is fixed and commented in `absorption.py`. Consistency check: for small $Z$ this
reproduces the standard $\alpha=\omega_p^2\nu/[c\,n(\omega^2+\nu^2)]$.)

**Collision frequency.** IRI/E-CHAIM supply no neutral atmosphere, and no MSIS-type model is
included, so $\nu_{\rm eff}$ is a height-only exponential:

$$
\nu_{\rm eff}(h)=10^{7}\,\mathrm{s^{-1}}\;e^{-(h-70\,\mathrm{km})/6\,\mathrm{km}},\qquad 0<h\le130\ \mathrm{km},
$$

and 0 outside. **This is an illustrative placeholder, not a published fit.** Any absolute
absorption number inherits that uncertainty; pass a callable `absorption_model=fn(alt_km)` to
substitute a real profile.

**Units — read this.** The engine stores $A$ in `state[9]` and converts it in
`TraceResult.absorption_db_total`. Because $\alpha$ is a **power** coefficient, the conversion
should be $10\log_{10}e\approx4.343\,A$. The code currently applies $8.686\,A$ (which is the factor for
an *amplitude* exponent). See §8.1.

**▸ Code:** `absorption.py` (`absorption_rate`, `collision_frequency_profile`); accumulation in
`_extended_rhs`; dB conversion at the end of `RayTracer.trace`.

### 2.9 Scattering by small-scale irregularities (Monte Carlo)

Real F-region plasma has field-aligned irregularities $\xi=\delta N_e/N_e$ far below the grid scale.
They cannot be resolved, so they are treated statistically, as a **Langevin equation**: the
deterministic ray equations plus small random rotations of $\mathbf u$ whose variance is set by
the irregularity spectrum. There are no free tuning parameters beyond the spectrum itself.

**Spectrum.** Anisotropic power law with axis along $\mathbf B$ (`IrregularitySpectrum`):

$$
\Phi_\xi(\mathbf q)=C\,\big(q_0^2+q_{\perp B}^2+\alpha^2 q_{\parallel B}^2\big)^{-p/2},\qquad q_0=\frac{2\pi}{L_0},
$$

with outer scale $L_0$, spectral index $p$ (default 4), axial ratio $\alpha\ge1$ (field-aligned
elongation, default 10) and rms $\xi_{\rm rms}$ (default $5\times10^{-3}$). $C$ is fixed by
$\int\Phi_\xi\,d^3q=1$ (multiply by $\xi_{\rm rms}^2$). The high-$q$ cut-off is the **Bragg limit**
$q_{\max}=2k_0$: structures smaller than $\lambda/2$ cannot scatter a propagating wave.

**Why the fluctuation is applied to $n^2$.** $\delta(n^2)=-X\xi$ stays finite as $n\to0$
at the reflection point, whereas $\delta n=\delta(n^2)/2n$ diverges there.

**Rates.** Geometrical optics gives the transverse ray-slope perturbation
$d\theta_i/ds=g\,\partial_i\xi$ with $g=X/(2n^2)$. Only Fourier components *transverse* to the ray
accumulate deflection (small-angle Bragg selection), and in the Markov limit the angular
variance grows at

$$
\frac{d\langle\theta_i\theta_j\rangle}{ds}=g^2\xi_{\rm rms}^2\,2\pi\!\int\! d^2q_\perp\,q_iq_j\Phi_\xi(\mathbf q_\perp,q_\parallel{=}0)
\;\equiv\; g^2\xi_{\rm rms}^2\,I_{ij}(\psi),\tag{K1}
$$
$$
\frac{d\langle\sigma_\phi^2\rangle}{ds}=\Big(\frac{k_0X}{2n}\Big)^2\xi_{\rm rms}^2\,I_{\phi}(\psi),\tag{K2}
$$

where $\psi$ is the angle between the ray and $\mathbf B$. (K1) is dominated by the *smallest*
scales (log-divergent for $p=4$, cut at the Bragg limit); (K2) by the *largest* (the outer scale).
$I_{11},I_{22},I_\phi$ depend only on $\psi$, so they are **tabulated once** on 61 values of
$\psi\in[0,\pi/2]$ and looked up at run time. The canonical transverse frame is $\hat e_1$ = projection of $\hat{\mathbf B}$
perpendicular to the ray, $\hat e_2=\hat{\mathbf u}\times\hat e_1$.

**Kicks.** After each step of length $\Delta s$, $\hat{\mathbf u}\leftarrow\hat{\mathbf u}+\delta_1\hat e_1+\delta_2\hat e_2$ with
$\delta_i\sim\mathcal N(0,\,\mathrm{rate}_{ii}\,\Delta s)$, then renormalised so $|\mathbf u|$ is
unchanged (static irregularities cannot change $\omega$, so the kick is a pure rotation;
exact for an isotropic plasma, $O(Y\,\delta\theta)$ off in a magnetised one).

**Coherence.** The deterministic phase variance $\sigma_\phi^2=\int\rho_\phi\,ds$ (K2) is accumulated on *every* trace when requested
(no kicks needed). The mean field survives as $\langle E\rangle/E_0=e^{-\sigma_\phi^2/2}$, i.e. a coherent
*power* fraction $e^{-\sigma_\phi^2}$ ↔ a loss of $(10/\ln10)\,\sigma_\phi^2\approx4.34\,\sigma_\phi^2$ dB.

**What it cannot do.** Interference fringes, caustic amplitudes, or any deterministic field
statistic: phases are randomised by construction.

**▸ Code:** `mc_scattering.py` (spectrum, tabulation, `MCScatteringTracer.ensemble/summarize`);
kick rates in `RayTracer._kick_rates`; kicks in the fixed-step driver.

### 2.10 Ray-tube focusing (the tangent map)

Power along a ray is set by how a small tube of neighbouring rays converges or spreads. Hamilton
computes this from the **tangent map** $\Phi=\partial(\mathbf r,\mathbf u)_{\rm final}/\partial(\mathbf r,\mathbf u)_{\rm initial}$
($6\times6$), the linearisation of the flow of (1).

For a *point* source with a bundle of launch *angles*, the relevant block is
$\Phi_{ru}=\partial\mathbf r_{\rm final}/\partial\mathbf u_{\rm initial}$ (3×3). Let
$\hat{\mathbf t}_1,\hat{\mathbf t}_2$ span the plane ⟂ to the launch direction; then
$\mathbf v_i=\Phi_{ru}\hat{\mathbf t}_i$ are the landing-point displacements per unit launch-direction
change. Projecting them onto the plane perpendicular to the **group-velocity direction** at the
landing point (not the wave normal — the two differ in a magnetised medium) gives the cross-section
area of the ray tube, $\mathcal A=|\mathbf v_{1\perp}\times\mathbf v_{2\perp}|$. In free space
$\Phi_{ru}\hat{\mathbf t}_i=s\,\hat{\mathbf t}_i$ exactly, so $\mathcal A_{\rm free}=s^2$ with $s$ the *arc
length* (not chord, not group path). Hence

$$
G_{\rm focus}=10\log_{10}\frac{s_{\rm total}^2}{\mathcal A}\ \ [\mathrm{dB,\ power}] ,
$$

positive = focused relative to free-space spreading.

*A subtlety that turns out to be harmless:* the trace stops on a ground-crossing **event**, not at a
fixed $\tau$, so $\Phi$ is not strictly "how the landing point moves with launch angle"; the
standard event correction adds a term of the form $\dot{\mathbf r}\,\mathbf g^{\!\top}$. That term's
output is parallel to $\dot{\mathbf r}$, and the projection ⟂ $\dot{\mathbf r}$ removes exactly that
component — so it cancels and is correctly omitted (it would matter if one projected onto $\mathbf u$).

**Why the tangent map is propagated in arc length, in a second pass.** $d\Phi/d\tau=J_m\Phi$ is exact,
but a linear ODE integrated over a huge span of $\tau$ grows like $e^{J_m\,\Delta\tau}$ even when
$J_m$ is tiny — and crossing a weakly ionised layer *does* consume a huge span of $\tau$ for a short
distance. Observed: `solve_ivp` stalled tens of thousands of RHS calls with $|\Phi|\sim10^5$ from a
Jacobian entry $\lesssim4\times10^{-6}$. The fix reparametrises by arc length ($ds/ds=1$, no such
stretching). With $v=|\dot{\mathbf r}|$, $\mathbf F=(\dot{\mathbf r},\dot{\mathbf u})$ and
$\nabla v=(\dot{\mathbf r}/v)^{\!\top}J_m[:3,:]$:

$$
\frac{d\Phi}{ds}=\Big[\frac{J_m}{v}-\frac{\mathbf F\,\nabla v^{\top}}{v^{2}}\Big]\Phi .
$$

This is integrated as its own `solve_ivp` over $s\in[0,s_{\rm total}]$, looking the physical state up
in the dense output of the *already finished* base trajectory. Being a decoupled second pass, a
numerical problem in $\Phi$ can no longer corrupt the trajectory (which it did when both were
integrated together). $J_m$ is a finite-difference Jacobian of the physical RHS
(`Ray._state_jacobian_fd`, step $10^{-3}+10^{-6}\max(1,|y_j|)$).

**Caustics.** Where the ray tube collapses ($\mathcal A\to0$) geometrical optics predicts infinite
focusing; $G_{\rm focus}\to+\infty$ (the function returns `None` only if the area is exactly zero). Near caustics the
result is a large positive number that should be read as "a caustic is near", not as a
power gain; a wave-optics (Airy) treatment is outside scope.

**▸ Code:** `RayTracer._propagate_tangent_map` (`raytracer.py`), `_focusing_gain_db` (`link_budget.py`).

### 2.11 The link budget

`link_budget.compute_receiver_statistics` runs **one** adaptive trace with the tangent map, group and
phase paths, absorption and phase variance all switched on, and reports relative dB terms:

$$
\text{total}=G_{\rm focus}-L_{\rm abs}-L_{\rm coh},\qquad L_{\rm coh}=\tfrac{10}{\ln 10}\,\sigma_\phi^2 .
$$

There is **no transmitter EIRP, antenna gain, or receiver model** anywhere in Hamilton, so `total`
is a *relative* figure, not a received power. It is `None` if the ray did not land. It is
rigorous for a single hop only (§8.3).

### 2.12 Reflection at the ground and multiple hops

When a ray reaches the WGS84 ellipsoid it reflects specularly about the **ellipsoid's outward
normal** (which is exactly geodetic "up"; not the radial direction — the two differ by up to
0.192° at mid-latitude): $\mathbf u\leftarrow\mathbf u-2(\mathbf u\cdot\hat{\mathbf n})\hat{\mathbf n}$.
Each hop has its own path budget; `max_total_path_m` (default budget × hops) caps the total. The
ground is a perfect specular reflector: no reflection loss, no roughness, no Fresnel coefficients.

---

## 3. Environment: geometry, electron density, magnetic field

### 3.1 Geometry (`geometry.py`)

* **WGS84** ellipsoid throughout ($a=6378137$ m, $f=1/298.257223563$). Latitude is **geodetic**
  everywhere — what IRI, IGRF, GPS and maps mean. No geoid undulation (±100 m, negligible next to
  the ionosphere's own uncertainty).
* Geodetic → ECEF is closed form. ECEF → geodetic uses **Bowring's (1976) method** with a
  reduced-latitude seed and two fixed-point iterations: verified on 200,000 random points including
  poles, equator, antimeridian and $h=0$, max round-trip error ~$5.5\times10^{-8}$ m in position and
  $1.1\times10^{-7}$ m in height.
* Ground detection uses the implicit quadric $x^2/A^2+y^2/A^2+z^2/B^2-1$ (negative inside).
* **Ground range and cross range are spherical** (great circle on $R_{\rm mean}=(2a+b)/3$) by
  design — they are reporting quantities, not physics boundaries, and Vincenty/Karney geodesics are
  not justified by the payoff.

### 3.2 Electron density (`ionosphere.py`, `ECHAIM/`)

* **Two models, one grid, per-point selection.** E-CHAIM (Themens et al. 2017/18; an empirical,
  data-assimilating high-latitude model) is used wherever it returns a value, i.e. where the
  AACGM geomagnetic latitude is **≥ 45°**. Below that, E-CHAIM returns `NaN` (enforced in its own
  C code, with the real AACGM transform) and **IRI-2020** is used. `use_echaim=False` forces
  IRI-only; if E-CHAIM fails to load the code falls back to IRI with a printed warning.
  The selection is a hard switch, not a blend (E-CHAIM documents 45–50° as its own transition band
  for merging with a mid-latitude model); the step is smoothed by the upsampling below.
* **Building the patch.** Both models are queried **once**, at construction, on a coarse
  `n_ang × n_ang` (default 15×15) lat/lon grid and at altitudes 60–800 km in 5 km steps (IRI's
  native range). Below 60 km the density is a hard zero. The result is trilinearly upsampled onto
  the ray-tracer grid (**2 km in height × 1° × 1°**), and runtime queries are one
  `RegularGridInterpolator` call plus one Bowring transform. Outside the patch $N_e=0$.
* **Patch size.** `build_ionosphere_for_ray` sizes the patch as half-angle
  $=\mathrm{path}/R_{\rm mean}\times1.3$ (margin), clamped to [2°, 180°], centred on the launch
  point — a conservative straight-line bound. For a launch above the 800 km ceiling it is centred on
  the entry point instead.
* **E-CHAIM binding.** The vendored CHAIN C library is called through `ctypes` (`ECHAIM/echaim.py`,
  built by `ECHAIM/build.sh`). One patch was made to the vendored source: `getDir()` normally reads a
  config file from the CWD and *blocks on stdin* if it is absent; `echaim_init(dir)` sets the same global
  directly. The bundled `CHAIM_DB.db` is an index-history **snapshot** (no auto-update; `updateLocalDB.c`
  is excluded because it needs libcurl). Querying past its coverage still returns a value (the library falls
  back to medians), with reduced accuracy in the storm/precipitation terms. Options: storm off,
  precipitation on, E-CHAIM's own D-region off (absorption handles the D-region separately).
  E-CHAIM is **GPL-3.0**; its licences are in `ECHAIM/licenses/`.

### 3.3 Magnetic field (`igrf.py`)

IGRF-13 through `ppigrf` (degree 13) is evaluated once on a **coarse 5×5×5** (height, lat, lon) grid
over exactly the same patch and date as the density, transformed from east/north/up into ECEF
components, and linearly interpolated. Two design points:

* `ppigrf` re-parses its coefficient file on *every call*; the ray RHS calls $\mathbf B$ six times
  (central-difference $\nabla\mathbf Y$) plus once — tens of thousands of calls per ray — so the
  batch-then-interpolate pattern is not optional.
* Unlike density, $\mathbf B$ must never silently become zero outside the patch, so queries are
  **clamped** to the grid instead of zero-filled.

If no field is supplied, `Ray` falls back to an analytic tilted dipole ($7.8\times10^{22}$ A m², tilt 9.6°) —
adequate for synthetic tests, not for real predictions.

### 3.4 Gradients

$\nabla N_e$ and $\nabla\mathbf Y$ are **central finite differences with a 500 m step**, on the
interpolated fields. Because the density is *trilinear* on 2 km cells, its gradient inside a cell is
essentially constant and **jumps at cell faces** — the Hamiltonian is only $C^0$. This limits the
achievable convergence order of any integrator, and is why "converged" (§4.4) means agreement to
~0.1–0.15 %, not to machine precision.

---

## 4. Numerical methods

### 4.1 Two drivers, one RHS

| | fixed-step RK4 | adaptive RK45 |
|---|---|---|
| use | default; Monte Carlo ensembles | tangent map / link budget; high accuracy |
| step | $d\tau=\Delta s/\lvert\dot{\mathbf r}\rvert$, $\Delta s=$ `ds_step_km` (2 km default) | `solve_ivp`, `rtol=1e-6`, `atol=1e-9` |
| scattering kicks | yes | no |
| tangent map | no | yes (second pass, §2.10) |
| events | tested each step | ground / ceiling / length events, dense output |

### 4.2 Fixed-step RK4 and its step rejection

Each step is sized so the *nominal* arc advance is $\Delta s$: $d\tau=\Delta s/|\dot{\mathbf r}_1|$ using the
rate at the **start** of the step. That assumption fails in weakly ionised layers, where $|\dot{\mathbf r}|$
can start near $10^{-6}$ and grow by orders of magnitude within a few hundred metres. So after each RK4 step the
achieved advance $\Delta s_{\rm actual}=|\mathbf r_{\rm new}-\mathbf r_{\rm old}|$ is compared to
$5\Delta s$ and, if exceeded, $d\tau$ is halved and the step retried (up to 60 times, then `STALL`).
Without this, an early version overshot by 161,000 km in a step and returned a wrong landing point.

After each accepted step, in order: (optional) Langevin kick with variance
`rate·Δs_actual` → record every `record_every`-th point → ground test → ceiling test.

### 4.3 Stopping conditions and ground location

* **GROUND:** the quadric turns non-positive **and** more than $2\Delta s$ of arc has been travelled
  in this segment (a guard so a ray launched at $h=0$ does not "land" on step 1). The exact crossing is then found by a
  **40-step bisection** on the chord between the last two states, interpolating the *whole* state (accumulators included).
* **CEILING:** outside the ellipsoid inflated by 800 km **and** $|\mathbf r|$ larger than at launch — the ray has escaped.
* **LENGTH_BUDGET / STALL:** path budget spent, or $|\dot{\mathbf r}|<10^{-12}$ / 60 halvings failed.

The result records which one fired (`RayPath.status`: *landed* / *escaped* / *incomplete*).

### 4.4 Adaptive driver and tolerances

Two lessons are baked into the defaults:

* **No `max_step` cap.** A cap in $\tau$ sized for "a few km" is, where $ds/d\tau\to0$, a cap on arc progress of a small fraction of a
  millimetre — `solve_ivp` advanced 21 m in 21,000 RHS calls crossing the D-layer base. RK45's own error control is sufficient.
* **Tight tolerances.** SciPy's defaults (`1e-3/1e-6`) put a reflection's apex at 210 km and the range 25 % short;
  `1e-6/1e-9` and `1e-9/1e-12` agree, giving a 166 km apex. The RHS spans $X\sim10^{-29}$–1 in one trace
  and the *turning point*, which sets where the ray lands, is where default tolerances under-resolve.

Cross-check: fixed step (2 km) and adaptive agree to ~0.14 % (940.5 vs 941.8 km) on the reference ray.

### 4.5 Ray parameter and units

$\tau$ has units of metres only because $F$ was normalised so; it is **not** a length. Never
interpret $\Delta\tau$ as a distance. Distances are always the accumulated arc length $s$.

### 4.6 Cost

The state is small; the cost is the RHS. Each RHS evaluation calls the density interpolator 7 times
(value + 6 for the gradient), the field interpolator 7 times, Bowring for each, and — with group path
and absorption on — the Appleton–Hartree solve 3 more times for $n_g$ and once complex for absorption.
A correctly converged single-hop fixed-step trace is ~2,400 RHS calls: **a few to ~10 s per ray**.
A 200-ray Monte Carlo ensemble therefore takes minutes. The two-pass tangent map costs an extra
$\sim$adaptive solve plus a 12-RHS-call finite-difference Jacobian per derivative evaluation. Building the density patch is
typically a few seconds at the default resolution (memory-hungry at large `n_ang`).

---

## 5. Software architecture

### 5.1 Module map

| module | role | depends on |
|---|---|---|
| `api.py` | **public interface**: `Link`, `RayPath`, `ScatterResult`, `LinkBudget` | everything below |
| `main.py` | `Ray` (the physics RHS), `build_ionosphere_for_ray`; a legacy `__main__` demo | ionosphere, geometry |
| `raytracer.py` | `RayTracer` (extended RHS, both drivers, hop loop, tangent-map pass), `TraceResult` | absorption, geometry |
| `absorption.py` | collision-frequency profile; complex-$X$ absorption rate | `Ray._appleton_hartree_p` |
| `mc_scattering.py` | irregularity spectrum + tabulated kick statistics; `MCScatteringTracer` (thin subclass of `RayTracer`) | raytracer |
| `link_budget.py` | focusing gain + absorption + coherence → relative dB | raytracer |
| `ionosphere.py` | E-CHAIM + IRI-2020 density patch, trilinear interpolation | iricore, `ECHAIM/`, geometry |
| `ECHAIM/` | vendored C library + `ctypes` binding | libc, SQLite (vendored) |
| `igrf.py` | IGRF-13 field patch | ppigrf, geometry |
| `geometry.py` | WGS84 transforms, ellipsoid intersection, ranges | numpy |
| `visualization.py` | interactive 3-D Plotly scene, duck-typed on `.path` | plotly |
| `constants.py` | $c$, $q_e$, $m_e$, $\varepsilon_0$, $\mu_0$ | — |
| `test_iono.py` | legacy Chapman-layer test, unused | — |

Dependencies are a DAG: `geometry` was pulled out of `main.py` precisely so `raytracer` and `absorption`
could use it without a circular import.

### 5.2 Design of the public API

`api.Link` is a *transmitter*: place + moment + default frequency and mode. The constructor does all the
expensive work once; then

* `trace(elevation_deg, azimuth_deg, …)` → `RayPath`
* `scatter(elevation_deg, …, n_rays=200, seed=…)` → `ScatterResult`
* `link_budget(elevation_deg, …)` → `LinkBudget`
* `sweep_elevation / sweep_azimuth / sweep_frequency` → `list[RayPath]`
* `plot(...)`, `grid_info()`, `.ionosphere`, `.magnetic_field`

Frequency and mode can be overridden **per call**; the `Ray` for each (frequency, mode) pair is cached,
and it reuses the same patch. **Unit convention of the API:** km, MHz, degrees, dB, ms; ECEF positions are
the one exception and stay in metres (so a `RayPath` can be handed straight to `visualization.plot_ray`).

Guard rails and traps:

* The patch is sized by `max_path_km × max_nhops` at construction. Asking `trace()` for a longer path than
  the `Link` was built for **silently runs past the patch edge** (density → 0, field clamped).
* `tangent_map=True` silently switches to `method="adaptive"`.
* `RayPath.sigma_phi2` is `None` unless `phase_variance=True` (or it came from `scatter`).
* `elevation_deg` is restricted to [0, 90]; azimuth is clockwise from true north.

### 5.3 Data flow of one `trace()` call

1. `Link._direction`: (elevation, azimuth) → unit vector in the local ENU frame at the transmitter → ECEF.
2. `Link._resolve_ray`: get/create `Ray(ω, MODE, Ne, B)`.
3. `RayTracer.trace`: initial extended state; loop over hops: pick a driver, integrate one segment, stop on an event;
   on GROUND, reflect $\mathbf u$ and continue if hops remain.
4. Assemble `TraceResult` (totals, decimated path, bounce points, ground range).
5. `_build_ray_path`: unit conversion, arrival elevation (against the *ellipsoid* normal), wrap into `RayPath`.

---

## 6. Using it

```python
from datetime import datetime
from api import Link

link = Link(57.0, 10.0, frequency_mhz=8.0,
            dt=datetime(2024, 6, 21, 14, 0, 0), mode="O")   # builds Ne + B once

ray = link.trace(elevation_deg=28.0)        # deterministic ray
print(ray)                                   # landing, range, delay, absorption
budget = link.link_budget(elevation_deg=28.0)
ens = link.scatter(elevation_deg=28.0, n_rays=200, seed=1)   # minutes
ray.plot().write_html("ray.html")           # rotate/zoom in a browser
```

`example.py` is exactly this in ~20 lines. **Requirements:** Python 3, `numpy`, `scipy`, `plotly`,
`iricore`, `ppigrf` (+`pandas`), a C compiler (the E-CHAIM library is built on first import via
`ECHAIM/build.sh`), optionally `global-land-mask` (land/ocean shading) and `matplotlib` (legacy demo).
Working reference: the project `venv/`.

Low-level access (custom tracing, tests): `build_ionosphere_for_ray` → `build_igrf_for_ionosphere` →
`Ray` → `RayTracer.trace(r0, u0, max_path_m, method=…, nhops=…, scatter=…, carry_tangent_map=…)`.
`u0` is a *unit* vector (launched from vacuum, $n=1$).

---

## 7. Validation

What was checked, and how (details in the separate *Hamilton Physics Audit*):

| what | check | result |
|---|---|---|
| dispersion polynomial (2) | vacuum $X=0\Rightarrow p=1$ for any $Y$ | pass (the *old* polynomial failed this) |
| | $\theta=90°$: O-mode $n^2=1-X$; X-mode $n^2=[(1-X)^2-Y^2]/(1-X-Y^2)$ | pass |
| | $\theta=0$: whistler-type $n^2=1-X/(1\pm Y)$ | pass |
| | vertical incidence on a synthetic ramp reflects at $X=1$ | pass |
| $J,K,L,M_{sc}$ | four identities $J=2\partial_pF$, $K=-\partial_{Y_d}F/Y$, $L=-\partial_XF$, $M_{sc}=-\partial_YF/Y+Y_d\,\partial_{Y_d}F/Y^2$ cancelled symbolically | pass |
| $\varepsilon$-form | agrees with naive $p$-form to ≥4 digits for $X\gtrsim10^{-4}$; stays accurate to $X\sim10^{-29}$ | pass |
| absorption | sign ($\mathrm{Im}\,n>0$) across a wide sweep of $X,Y,Y_d,\nu$, both modes; small-$Z$ limit matches the textbook $\alpha$ (analytic check made in this review) | pass — **but the dB conversion is off by 2, §8.1** |
| geometry | Bowring round trip, 200 k points + edge cases | ≤ $1.1\times10^{-7}$ m |
| integrators | fixed (2 km) vs adaptive on the reference ray | 940.5 vs 941.8 km |
| | adaptive at `1e-6/1e-9` vs `1e-9/1e-12` | agree |
| tangent map | free-space reference equals $s^2$ exactly; projection/event-correction cancellation | verified analytically and on random matrices |
| link budget | end-to-end finite on the reference ray | pass |

**Not validated:** absolute absorption (placeholder $\nu$ profile; dB factor wrong), group path away from
isotropic/vacuum limits (§8.2), the Monte Carlo statistics against an independent code (the module docstring
promises a runnable "gold-standard" validation; that code does **not** exist in the file), multi-hop
focusing, and any comparison against PHaRLAP or measured ionosonde/oblique-sounder data.

---

## 8. Known defects, limitations, and pitfalls

### Defects found in this review (not fixed)

**8.1 Absorption in dB is 2× too large.** `absorption_rate` returns the *power* coefficient
$\alpha=2k_0\,\mathrm{Im}\,n$ (the docstring says so and the small-$Z$ limit confirms it), so
$A=\int\alpha\,ds$ is the power exponent and the loss in dB is $10\log_{10}(e^{A})=4.343\,A$.
`raytracer.py` (`result.absorption_db_total = 8.686 * float(state[9])`) and the docstrings in
`absorption.py`/`TraceResult` use $8.686=20\log_{10}e$, which is the factor for an *amplitude*
exponent. Every `absorption_db`, `LinkBudget.absorption_db` and `LinkBudget.total_db` is therefore
too large by exactly 2 for a given $\nu_{\rm eff}$ (and `absorption_nepers_total` is a *power* exponent
despite the name). Fix: one constant, plus comments.

**8.2 Group path is approximate and locally wrong near reflection.** `_group_index` takes
$n_g=n+\omega\,\partial n/\partial\omega$ by re-solving (2) at $\omega\pm\delta\omega$ while holding
$Y_d=\hat{\mathbf Y}\cdot\mathbf u$ **fixed**. The correct quantity is the derivative at fixed *wave-normal
direction* (or, equivalently in the Hamiltonian picture, $dP_g/d\tau=-\omega\,\partial F/\partial\omega|_{\mathbf k}$), and the
group delay per unit arc also carries a $\cos\alpha$ factor for the angle $\alpha$ between ray and wave
normal. In an isotropic plasma it reduces to the exact $n_g=1/n$. Spot checks of the code's $n_g$ against the
Hamiltonian-exact value at 8 MHz, $B=5\times10^{-5}$ T:

| $X$ | $\theta_{kB}$ | mode | code / exact − 1 |
|---|---|---|---|
| 0.30 | 60° | O | 0.4 % |
| 0.95 | 85° | O | 0.3 % |
| 0.50 | 60° | X | 3 % |
| 0.90 | 60° | O | 10 % |
| 0.70 | 20° | X | 26 % |

Weak plasma (most of the path) is fine; the error sits where $n$ is small — near the reflection point, where the
group delay accumulates fastest. The *integrated* effect on the total group path was not quantified.
The phase path and arc length do not depend on this and are unaffected.

### Limitations by design

**8.3 Multi-hop.** The tangent map is carried across ground bounces *without* differentiating the specular-reflection
map (it depends on $\mathbf r$ through $\hat{\mathbf n}$ and acts on $\mathbf u$). Focusing and therefore `total_db`
are rigorous for `nhops=1` only.

**8.4 Scattering.** A single, spatially uniform spectrum (no height/latitude dependence of $\xi_{\rm rms}$; typical
mid-latitude $10^{-3}$–$10^{-2}$, spread-F up to $10^{-1}$). Static irregularities only. Kicks use the rates at the
end of each step. `mc_scattering.py`'s header describes a validation harness ("run this file directly") that is not present.

**8.5 Doppler.** Not implemented; `doppler_shift_hz` is always `None` and `compute_doppler` raises `NotImplementedError`.
A physically correct Doppler needs two ionosphere epochs.

**8.6 Mode degeneracy.** Besides $X=0$ (handled), $\partial F/\partial\mathbf u\to0$ wherever the O and X roots coincide,
which includes $Y\to0$ (frequency $\gg$ the gyrofrequency — not the HF regime, where $Y\approx0.02$–$0.2$, but the
step-size compensation and the $\max(\cdot,0)$ clip on the discriminant lose accuracy as $Y$ shrinks). Also unguarded:
the resonance $a=1-X-Y^2=0$, and a negative discriminant, which is silently clipped to 0 (the ray is pushed onto the degenerate point rather than an error being reported).

**8.7 Numerical smoothness.** Density is trilinear on 2 km cells (§3.4) and the field is linear on a coarse 5×5×5 grid
(≈ 10° cells over a typical patch, ~200 km in height), so $\nabla N_e,\nabla\mathbf B$ are piecewise constant. For
the field this is benign ($\nabla\mathbf Y$ enters through $K$ and $M_{sc}$, which are small); for the density it sets
the ~0.1 % convergence floor.

**8.8 Model coverage.** E-CHAIM ↔ IRI is a hard switch at 45° AACGM; the horizontal structure below the coarse query
spacing (a few degrees) is interpolated, not modelled; storm and E-CHAIM-D-region options are off; the CHAIM index
snapshot is not auto-refreshed; the collision-frequency profile is illustrative (§2.8).

**8.9 Ground.** Smooth, perfectly reflecting ellipsoid, no terrain, no sea/land reflection coefficients (`visualization`
colours land/ocean but this is purely cosmetic).

### Practical pitfalls

* **Memory.** On a memory-constrained machine the E-CHAIM/IRI query at large `n_ang` can stall (observed as a hang at
  "Querying E-CHAIM" at `n_ang=15` under exhausted swap; `n_ang=6` was instant). This was environmental, not an algorithm issue.
* **Path budget vs patch.** See §5.2.
* **Stale/legacy code.** `main.py`'s `__main__` block still uses the low-level pipeline (75×75 query, 200-ray ensemble,
  matplotlib figure); `test_iono.py` tests a discarded Chapman-layer model; `ECHAIM/build.sh` mentions an `ECHAIM/README.md`
  that does not exist.
* **Version control.** At review time `absorption.py, api.py, example.py, geometry.py, igrf.py, link_budget.py, mc_scattering.py,
  raytracer.py` and `ECHAIM/` are untracked, and there is no `.gitignore` (`venv/`, `__pycache__/`, `ECHAIM/etc/` ≈115 MB and the built
  `libechaim.so` would all be swept in).

---

## 9. References

* Haselgrove, J. (1954). *Ray theory and a new method for ray tracing.* In Physics of the Ionosphere, Physical Society, London.
* Budden, K. G. (1961). *Radio Waves in the Ionosphere.* Cambridge University Press. (magneto-ionic theory, complex-$X$ collisional damping, group index.)
* Stix, T. H. (1992). *Waves in Plasmas.* AIP. (S, D, P, R, L notation.)
* Davies, K. (1990). *Ionospheric Radio.* Peter Peregrinus. (D/E-region collision frequencies.)
* Themens, D. R., Jayachandran, P. T., Galkin, I., Hall, C. (2017). *The Empirical Canadian High Arctic Ionospheric Model (E-CHAIM): NmF2 and hmF2.* J. Geophys. Res. Space Physics, 122. (and 2018, topside/bottomside/E-region follow-ups.)
* Bilitza, D. et al. (2022). *The International Reference Ionosphere 2020.* Space Weather / RS.
* Alken, P. et al. (2021). *International Geomagnetic Reference Field: the thirteenth generation.* Earth, Planets and Space, 73.
* Shepherd, S. G. (2014). *Altitude-adjusted corrected geomagnetic coordinates.* J. Geophys. Res. Space Physics, 119.
* Bowring, B. R. (1976). *Transformation from spatial to geographical coordinates.* Survey Review, 23(181).
* Booker, H. G. (1959). *A theory of scattering by nonisotropic irregularities…* J. Geophys. Res.; Tatarskii, V. I. (1961). *Wave Propagation in a Turbulent Medium*; Ishimaru, A. (1978). *Wave Propagation and Scattering in Random Media*, Vol. 1.
* Zabotin, N. A. et al. (2004). *Radio Science*, 39, RS3002. (HF multiple scattering.) Kontar, E. P. et al. (2019). *ApJ*, 884, 122. (Langevin Monte Carlo.)
* Coleman, C. J. (1998). *A ray tracing formulation and its application to some problems in over-the-horizon radar.* Radio Science, 33. Cervera, M. A., Harris, T. J. (2014). *Modeling ionospheric disturbance features in quasi-vertical incidence ionograms using 3-D magnetoionic ray tracing and atmospheric gravity waves.* JGR Space Physics. (PHaRLAP.)
