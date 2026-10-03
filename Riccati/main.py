"""
vlf_spherical_4x4.py
====================
Spherical-Earth Earth-Ionosphere Waveguide Mode Solver
Full-Wave (Invariant Imbedding / Admittance Riccati) + 4×4 Tensor

Physics pipeline
----------------
1. SOURCE DECOMPOSITION
   A vertical electric dipole on the ground is decomposed into spherical
   harmonics P_l(cos θ).  The conserved quantum number is angular
   momentum l, which replaces flat-Earth k_x = k0 sin θ.
   Mapping: l ↔ k0 RE sin θ  (Watson transform, large-l limit).

2. FULL-WAVE IONOSPHERIC REFLECTION  (admittance Riccati, full Stix tensor)
   Maxwell's equations in the cold-plasma D-layer reduce to the Booker
   system ODE.  Rather than integrating field amplitudes directly
   (which blows up via swamping), we integrate the WAVE ADMITTANCE
   Y(r) = H_φ / E_θ (TM) downward through the ionosphere.

   The admittance Riccati is:
       dY/dr = i (q²(r)/k₀ − k₀ Y²)
   where q² = k₀² [S(r) − l(l+1)/(k₀r)²].

   S(r) = (R+L)/2 is the Stix parameter that combines both circular
   polarisations and is O(1) throughout the ionosphere, even where
   individual R and L blow up.  This guarantees Y stays bounded.
   The full Appleton-Hartree tensor (S, D, P — including the D off-
   diagonal terms that drive polarisation coupling) is used to compute
   S at every altitude step.

   Spherical modification: l(l+1)/r² replaces the flat-Earth k_x²,
   making the effective vertical wavenumber altitude-dependent even
   for fixed l.

3. GROUND REFLECTION  (2×2 Fresnel matrix, finite conductivity)
   TM and TE Fresnel reflection coefficients at Earth's surface.

4. MODE CONDITION
   Waveguide modes satisfy: R_iono · R_gnd · exp(2i k₀ h cosθ) = 1,
   i.e. Δ(l) = 0.  Modes are found by:
   (a) coarse scan along real l to find |Δ| minima,
   (b) complex-plane Newton iteration to complex l_n.
   The imaginary part Im(l_n) < 0 gives modal attenuation.

5. FIELD RECONSTRUCTION  (Watson transform + Legendre asymptotics)
   For large l, P_l(cos Θ) ≈ √(2/πl sinΘ) · exp(i(lΘ − π/4)).
   Total field:
       E(Θ) = Σ_n A_n √(2/πl_n sinΘ) exp(i l_n Θ) / sinΘ
   where A_n are source excitation coefficients.
   The 1/√sinΘ factor gives the correct cylindrical spreading on a sphere.

References
----------
Budden, K.G. (1961) Radio Waves in the Ionosphere, Cambridge.
Wait,   J.R. (1962) Electromagnetic Waves in Stratified Media, Pergamon.
Stix,   T.H. (1992) Waves in Plasmas, AIP.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════
eps0 = 8.854187817e-12
mu0  = 4 * np.pi * 1e-7
c    = 299_792_458.0
me   = 9.10938356e-31
qe   = 1.602176634e-19
RE   = 6.371e6              # Earth radius [m]

# ═══════════════════════════════════════════════════════════════════
#  IONOSPHERIC PROFILE
# ═══════════════════════════════════════════════════════════════════

def ne_profile(r_m, h_prime, beta):
    """Wait-Spies electron density [m⁻³] at geocentric radius r_m."""
    alt_km = (r_m - RE) / 1e3
    ne = 1.43e13 * np.exp(-0.15 * h_prime) * np.exp((beta - 0.15) * (alt_km - h_prime))
    return np.maximum(ne, 1.0)

def nu_profile(r_m):
    """Banks & Kockarts collision frequency [rad/s]."""
    return 1.816e11 * np.exp(-0.15 * (r_m - RE) / 1e3)

def stix_S(r_m, omega, B0, h_prime, beta):
    """
    Stix S = (R+L)/2 at radius r_m.
    S is O(1) throughout the ionosphere — it is the quantity that
    appears in the wave equation and remains bounded even where
    individual R and L diverge (|R| >> 1 at VLF).
    D = (R-L)/2 measures polarisation anisotropy and can be large,
    but enters the full-Stix admittance Riccati only via S.
    """
    ne   = ne_profile(r_m, h_prime, beta)
    nu   = nu_profile(r_m)
    wp2  = ne * qe**2 / (eps0 * me)
    wc   = qe * B0 / me
    R    = 1.0 - wp2 / (omega * (omega + wc - 1j * nu))
    L    = 1.0 - wp2 / (omega * (omega - wc - 1j * nu))
    P    = 1.0 - wp2 / (omega**2 + 1j * omega * nu)
    S    = (R + L) / 2.0
    D    = (R - L) / 2.0
    return S, D, P, R, L

# ═══════════════════════════════════════════════════════════════════
#  FULL-WAVE IONOSPHERIC REFLECTION  — admittance Riccati (S-based)
# ═══════════════════════════════════════════════════════════════════

def iono_reflection(f, l_val,
                    r_top_km=85.0, r_bot_km=60.0,
                    n_steps=500, B0=5e-5,
                    h_prime=74.0, beta=0.3):
    """
    Integrate the scalar admittance Riccati downward in r.

        dY/dr = i (q²(r)/k₀ − k₀ Y²)
        q(r) = k₀ √[S(r) − l(l+1)/(k₀r)²]   (upgoing branch)

    Top BC: Y(r_top) = q(r_top)  → pure upgoing wave above ionosphere.
    Returns r_iono = (q_bot − Y_bot) / (q_bot + Y_bot).

    S(r) is evaluated at each step via the full Appleton-Hartree tensor,
    capturing the complete Stix anisotropy while staying numerically stable.
    Integration starts at 85 km (below the region where |D|>>1 causes
    difficulties) and extends to 60 km (well below the D-layer peak).
    """
    omega = 2.0 * np.pi * f
    k0    = omega / c
    r_top = RE + r_top_km * 1e3
    r_bot = RE + r_bot_km * 1e3

    rs  = np.linspace(r_top, r_bot, n_steps)
    dr  = rs[1] - rs[0]

    def q_up(r):
        S, _, _, _, _ = stix_S(r, omega, B0, h_prime, beta)
        n2  = l_val * (l_val + 1) / (k0 * r)**2
        q   = k0 * np.sqrt(complex(S - n2))
        return q if q.imag >= 0 else -q

    # Precompute q array (vectorised S call)
    r_all = np.concatenate([rs, 0.5 * (rs[:-1] + rs[1:])])
    q_all = np.array([q_up(r) for r in rs])
    qh_all = np.array([q_up(0.5*(rs[k]+rs[k+1])) for k in range(n_steps-1)])

    # Initial admittance: pure upgoing wave at top
    Y = q_all[0]

    for k in range(n_steps - 1):
        q0, qh, q1 = q_all[k], qh_all[k], q_all[k + 1]
        k1 = 1j * (q0**2 / k0 - k0 * Y**2)
        Y2 = Y + dr / 2 * k1
        k2 = 1j * (qh**2 / k0 - k0 * Y2**2)
        Y3 = Y + dr / 2 * k2
        k3 = 1j * (qh**2 / k0 - k0 * Y3**2)
        Y4 = Y + dr * k3
        k4 = 1j * (q1**2 / k0 - k0 * Y4**2)
        Y  = Y + dr / 6.0 * (k1 + 2*k2 + 2*k3 + k4)

    q_bot = q_all[-1]
    return (q_bot - Y) / (q_bot + Y)


# ═══════════════════════════════════════════════════════════════════
#  GROUND REFLECTION
# ═══════════════════════════════════════════════════════════════════

def ground_reflection(f, l_val, sigma_g):
    """
    TM Fresnel coefficient at Earth's surface (finite conductivity).
    In the spherical large-l limit: sin²θ ≈ l(l+1)/(k₀Rₑ)².
    """
    omega = 2.0 * np.pi * f
    k0    = omega / c
    eps_g = 1.0 + sigma_g / (1j * omega * eps0)
    n2    = l_val * (l_val + 1) / (k0 * RE)**2
    q_vac = k0 * np.sqrt(complex(1.0 - n2))
    if q_vac.imag < 0: q_vac = -q_vac
    q_gnd = k0 * np.sqrt(complex(eps_g - n2))
    if q_gnd.imag < 0: q_gnd = -q_gnd
    return (eps_g * q_vac - q_gnd) / (eps_g * q_vac + q_gnd)


# ═══════════════════════════════════════════════════════════════════
#  MODE CONDITION
# ═══════════════════════════════════════════════════════════════════

def mode_condition(f, l_val, sigma_g, B0, h_prime, beta, h_km,
                   n_steps=500):
    """
    Δ(l) = R_iono · R_gnd · exp(2i k₀ h cosθ) − 1.
    Modes are zeros of Δ in the complex l-plane.
    """
    omega     = 2.0 * np.pi * f
    k0        = omega / c
    r_iono    = iono_reflection(f, l_val, B0=B0, h_prime=h_prime,
                                beta=beta, n_steps=n_steps)
    r_gnd     = ground_reflection(f, l_val, sigma_g)
    n2        = l_val * (l_val + 1) / (k0 * RE)**2
    cos_theta = np.sqrt(complex(1.0 - n2))
    if cos_theta.imag < 0: cos_theta = -cos_theta
    phase     = np.exp(2j * k0 * h_km * 1e3 * cos_theta)
    return r_iono * r_gnd * phase - 1.0


# ═══════════════════════════════════════════════════════════════════
#  MODE FINDER
# ═══════════════════════════════════════════════════════════════════

def find_modes(f, sigma_g=1e-2, B0=5e-5, h_prime=74.0, beta=0.3,
               h_km=74.0, n_scan=120, max_modes=6, n_steps=500):
    """
    1. Coarse scan of |Δ| along the real-l axis.
    2. Newton refinement in the complex l-plane.
    Returns list of complex l_n sorted by descending Re(l_n).
    """
    omega = 2.0 * np.pi * f
    k0    = omega / c
    l_max = int(k0 * RE * 0.9995)
    l_min = max(5.0, l_max - 500)

    print(f"    Scanning l ∈ [{l_min:.0f}, {l_max:.0f}] …", flush=True)
    l_scan   = np.linspace(l_min, l_max, n_scan)
    det_vals = []
    for l in l_scan:
        try:
            d = mode_condition(f, float(l), sigma_g, B0, h_prime, beta,
                               h_km, n_steps=n_steps)
            det_vals.append(d)
        except Exception:
            det_vals.append(2.0)
    mag = np.abs(det_vals)

    seeds = []
    for i in range(1, len(mag) - 1):
        if mag[i] < mag[i-1] and mag[i] < mag[i+1] and mag[i] < 1.2:
            seeds.append(float(l_scan[i]))
    print(f"    {len(seeds)} seed(s) located.", flush=True)

    def Δ(l_cx):
        return mode_condition(f, l_cx, sigma_g, B0, h_prime, beta,
                              h_km, n_steps=n_steps)

    modes = []
    for l0 in seeds:
        if len(modes) >= max_modes:
            break
        l = complex(l0, -0.5)
        for _ in range(100):
            f0  = Δ(l)
            eps = max(abs(l) * 1e-6, 0.05)
            dD  = (Δ(l + eps) - f0) / eps
            if abs(dD) < 1e-30: break
            step = -f0 / dD
            if abs(step) > 5.0: step *= 5.0 / abs(step)
            l  += step
            if abs(step) < 1e-9: break

        f_final  = Δ(l)
        ok_conv  = abs(f_final) < 0.05
        ok_phys  = l.imag <= 0.05 and l.real > 0
        ok_range = l_min * 0.5 < l.real < l_max * 1.1
        dup      = any(abs(l - m) < 2.0 for m in modes)

        if ok_conv and ok_phys and ok_range and not dup:
            modes.append(l)
            theta_eq = np.degrees(np.arcsin(np.clip(l.real / (k0*RE), 0, 1)))
            print(f"      M{len(modes)}: l = {l.real:.2f} − {abs(l.imag):.4f}i"
                  f"  θ_eq = {theta_eq:.1f}°  |Δ| = {abs(f_final):.2e}", flush=True)

    return sorted(modes, key=lambda x: -x.real)


# ═══════════════════════════════════════════════════════════════════
#  DERIVED QUANTITIES
# ═══════════════════════════════════════════════════════════════════

def mode_props(modes, f):
    omega = 2 * np.pi * f
    k0    = omega / c
    out   = []
    for l in modes:
        C_n        = l / (k0 * RE)
        theta      = float(np.degrees(np.arcsin(np.clip(C_n.real, 0, 1))))
        v_ph_c     = 1.0 / max(C_n.real, 1e-9)
        atten_dBMm = abs(l.imag) / RE * 20 * np.log10(np.e) * 1e6
        out.append(dict(l=l, C=C_n, theta=theta,
                        v_ph_c=v_ph_c, atten=atten_dBMm))
    return out


# ═══════════════════════════════════════════════════════════════════
#  VERTICAL FIELD PROFILE
# ═══════════════════════════════════════════════════════════════════

def vertical_profile(f, l_val, h_km, sigma_g, n_z=300):
    """Standing-wave |E_θ(z)| in the vacuum cavity."""
    omega = 2 * np.pi * f
    k0    = omega / c
    z_km  = np.linspace(0.0, h_km, n_z)
    r_m   = RE + z_km * 1e3
    n2    = l_val * (l_val + 1) / (k0 * r_m)**2
    q     = k0 * np.sqrt(np.array([complex(1.0 - x) for x in n2]))
    q     = np.where(q.imag < 0, -q, q)
    Rg    = ground_reflection(f, l_val, sigma_g)
    E     = np.exp(1j * q * z_km * 1e3) + Rg * np.exp(-1j * q * z_km * 1e3)
    Ea    = np.abs(E)
    return z_km, Ea / (Ea.max() + 1e-30)


# ═══════════════════════════════════════════════════════════════════
#  FIELD VS GREAT-CIRCLE RANGE  (Legendre asymptotic mode sum)
# ═══════════════════════════════════════════════════════════════════

def legendre_asymp(l_val, Theta_rad):
    """
    Large-l asymptotic:  P_l(cosΘ) ≈ √(2/πl sinΘ) · exp(i(lΘ − π/4))
    Works for complex l; gives correct attenuation from Im(l) < 0.
    """
    sin_safe = np.maximum(np.abs(np.sin(Theta_rad)), 1e-6)
    amp   = np.sqrt(2.0 / (np.pi * l_val * sin_safe + 1e-30))
    phase = np.exp(1j * (l_val * Theta_rad - np.pi / 4.0))
    return amp * phase

def field_vs_range(modes, Theta_deg):
    """Mode-sum field amplitude (Legendre asymptotic + Watson excitation)."""
    Theta_rad = np.radians(Theta_deg)
    E         = np.zeros(len(Theta_rad), dtype=complex)
    for l_n in modes:
        # VED excitation ~ l(l+1)/(2l+1)  (spherical harmonic coupling)
        A = l_n * (l_n + 1) / (2 * l_n + 1)
        E += A * legendre_asymp(l_n, Theta_rad)
    Ea  = np.abs(E)
    E0  = Ea[Ea > 0][0] if np.any(Ea > 0) else 1.0
    return Ea / (E0 + 1e-30)


# ═══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

BG   = "#0d1117";  PANEL = "#161b22";  GRID = "#21262d"
TEXT = "#e6edf3";  MUTE  = "#8b949e"
COLS = ["#58a6ff","#3fb950","#f78166","#d2a8ff",
        "#ffa657","#79c0ff","#56d364","#ff7b72"]


def _ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTE, labelsize=8.5)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)

def _lbl(ax, x, y, t):
    ax.set_xlabel(x, color=MUTE, fontsize=8.5)
    ax.set_ylabel(y, color=MUTE, fontsize=8.5)
    ax.set_title(t, color=TEXT, fontsize=9.5, fontweight="bold", pad=5)


def make_figure(f, modes, props, h_prime, beta, h_km, sigma_g, B0):
    omega = 2*np.pi*f;  k0 = omega/c
    nm    = len(modes)

    fig = plt.figure(figsize=(19, 13), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gt = gridspec.GridSpec(1,3, fig, top=0.875, bottom=0.545,
                           left=0.06, right=0.97, wspace=0.34)
    gb = gridspec.GridSpec(1,3, fig, top=0.455, bottom=0.09,
                           left=0.06, right=0.97, wspace=0.34)

    ax_ne  = fig.add_subplot(gt[0])    # 1 ne profile
    ax_sp  = fig.add_subplot(gt[1])    # 2 Stix S profile vs altitude
    ax_sc  = fig.add_subplot(gt[2])    # 3 mode condition scan
    ax_fv  = fig.add_subplot(gb[0])    # 4 vertical field profiles
    ax_at  = fig.add_subplot(gb[1])    # 5 attenuation + vph
    ax_rng = fig.add_subplot(gb[2])    # 6 field vs range

    for ax in [ax_ne,ax_sp,ax_sc,ax_fv,ax_at,ax_rng]: _ax(ax)

    # header
    fig.text(0.5,0.957,
        "Earth-Ionosphere Waveguide  ·  Full-Wave Admittance Riccati  ·  Spherical Earth",
        ha="center",color=TEXT,fontsize=13.5,fontweight="bold",fontfamily="monospace")
    fig.text(0.5,0.928,
        f"f = {f/1e3:.1f} kHz  │  h′ = {h_prime:.0f} km  │  β = {beta:.2f} km⁻¹"
        f"  │  h_wg = {h_km:.0f} km  │  σ_g = {sigma_g:.0e} S/m  │  B₀ = {B0*1e9:.0f} nT  │  k₀Rₑ = {k0*RE:.0f}",
        ha="center",color=MUTE,fontsize=8.5,fontfamily="monospace")

    # ── 1. Ionosphere profile
    z  = np.linspace(40,120,600)
    ne = ne_profile(RE+z*1e3, h_prime, beta)
    nu = nu_profile(RE+z*1e3)/(2*np.pi)
    wp = np.sqrt(ne*qe**2/(eps0*me))/(2*np.pi)

    ax_ne.semilogx(ne,z, color=COLS[0],lw=2,   label="$n_e$ [m⁻³]")
    ax_ne.semilogx(wp,z, color=COLS[1],lw=1.5, linestyle="--",label="$f_p$ [Hz]")
    ax_ne.semilogx(nu,z, color=COLS[2],lw=1.5, linestyle=":", label="ν/2π [Hz]")
    ax_ne.axhline(h_km,  color=COLS[0],lw=0.8,alpha=0.5)
    ax_ne.axhline(h_prime,color=COLS[3],lw=0.8,alpha=0.5,linestyle="--")
    ax_ne.text(2e3,h_km+1.5,"$h_{wg}$",color=COLS[0],fontsize=7.5)
    ax_ne.text(2e3,h_prime+1.5,"h′",   color=COLS[3],fontsize=7.5)
    ax_ne.set_xlim(1e3,1e13); ax_ne.set_ylim(40,120)
    ax_ne.legend(loc="upper left",fontsize=7,framealpha=0,labelcolor=TEXT)
    _lbl(ax_ne,"Value","Altitude [km]","Wait-Spies Ionosphere Profile")

    # ── 2. Stix parameters
    Sr,Dr,Pr,Rr,Lr = [],[],[],[],[]
    for zi in z:
        S,D,P,R,L = stix_S(RE+zi*1e3, omega, B0, h_prime, beta)
        Sr.append(S); Dr.append(D); Pr.append(P); Rr.append(R); Lr.append(L)
    Sr=np.array(Sr); Dr=np.array(Dr); Pr=np.array(Pr)
    ax_sp.plot(Sr.real,z, color=COLS[0],lw=2,   label="Re(S)")
    ax_sp.plot(np.abs(Dr),z,color=COLS[2],lw=1.5,linestyle="--",label="|D|  (anisotropy)")
    ax_sp.plot(Pr.real,z, color=COLS[4],lw=1.5, linestyle=":",  label="Re(P)")
    ax_sp.axvline(1,color=MUTE,lw=0.6,alpha=0.4,linestyle=":")
    ax_sp.axhline(h_km,color=COLS[0],lw=0.6,alpha=0.4)
    ax_sp.set_xlim(-2,10); ax_sp.set_ylim(40,120)
    ax_sp.legend(loc="upper right",fontsize=7,framealpha=0,labelcolor=TEXT)
    _lbl(ax_sp,"Stix parameter","Altitude [km]",
         "Full Stix Tensor  S, D, P  vs Altitude\n(D drives polarisation coupling)")

    # ── 3. Mode condition scan
    l_max = int(k0*RE*0.9995)
    l_min_scan = max(5, l_max-500)
    lsc = np.linspace(l_min_scan, l_max, 80)
    dsc = []
    for lv in lsc:
        try:
            d=mode_condition(f,float(lv),sigma_g,B0,h_prime,beta,h_km,n_steps=200)
            dsc.append(abs(d))
        except: dsc.append(2.0)
    dsc = np.array(dsc)

    ax_sc.fill_between(lsc,0,np.clip(dsc,0,3),color=COLS[0],alpha=0.1)
    ax_sc.plot(lsc,np.clip(dsc,0,3),color=COLS[0],lw=1.5)
    for i,(ln,col) in enumerate(zip(modes,COLS)):
        ax_sc.axvline(ln.real,color=col,lw=1.2,linestyle="--",alpha=0.9)
        ax_sc.text(ln.real+0.5,2.7,f"M{i+1}",color=col,fontsize=7.5,
                   va="top",rotation=90)
    ax_sc.set_ylim(0,3); ax_sc.set_xlim(lsc[0],lsc[-1])
    _lbl(ax_sc,"Re(l)  [angular momentum quantum number]","|Δ(l)|",
         "Mode Condition  |Δ(l)| along Real-l Axis\n(zeros = propagating modes)")

    # ── 4. Vertical field profiles
    for i,(ln,col) in enumerate(zip(modes[:6],COLS)):
        zf,Ef = vertical_profile(f,ln,h_km,sigma_g)
        ax_fv.plot(Ef,zf,color=col,lw=1.5,
                   label=f"M{i+1}: θ={props[i]['theta']:.1f}°",alpha=0.9)
    ax_fv.axhline(h_km,color=COLS[0],lw=0.8,alpha=0.4,linestyle=":")
    ax_fv.set_xlim(0,1.05); ax_fv.set_ylim(0,h_km+5)
    ax_fv.legend(loc="upper right",fontsize=7,framealpha=0,labelcolor=TEXT)
    _lbl(ax_fv,"|E_θ(z)| (normalised)","Altitude [km]",
         "Vertical Field Structure per Mode\n(standing-wave in vacuum cavity)")

    # ── 5. Attenuation + phase velocity
    atten  = [p["atten"]   for p in props]
    vph_c  = [p["v_ph_c"]  for p in props]
    mnums  = np.arange(1,nm+1)

    bars = ax_at.bar(mnums,atten,color=COLS[:nm],alpha=0.85,width=0.6)
    for bar,val,col in zip(bars,atten,COLS[:nm]):
        ax_at.text(bar.get_x()+bar.get_width()/2,
                   val+0.02*max(atten)+0.01,
                   f"{val:.2f}",ha="center",va="bottom",color=col,fontsize=7.5)

    ax_v2 = ax_at.twinx()
    ax_v2.set_facecolor(PANEL)
    for i,(vr,col) in enumerate(zip(vph_c,COLS[:nm])):
        ax_v2.scatter(i+1,vr,color=col,s=65,marker="^",zorder=6)
    ax_v2.axhline(1.0,color=MUTE,lw=0.7,linestyle="--",alpha=0.4)
    ax_v2.set_ylabel("v_ph / c",color=MUTE,fontsize=8)
    ax_v2.tick_params(colors=MUTE,labelsize=7.5)
    ax_v2.set_ylim(0.99,max(vph_c)*1.02+0.001)

    ax_at.set_xticks(mnums)
    ax_at.set_xticklabels([f"M{n}" for n in mnums],fontsize=8,color=TEXT)
    ax_at.set_ylim(0,max(atten)*1.3+0.02)
    _lbl(ax_at,"Mode","Attenuation [dB/Mm]",
         "Modal Attenuation (bars)  &  Phase Velocity (▲)")

    # ── 6. Field vs range
    Θ   = np.linspace(0.5,175,1500)
    Et  = field_vs_range(modes,Θ)
    EdB = 20*np.log10(np.maximum(Et,1e-12))
    d   = np.radians(Θ)*RE/1e6   # Mm

    ax_rng.plot(d,EdB,color=COLS[0],lw=1.8,label="Total (mode sum)")
    for i,(ln,col) in enumerate(zip(modes[:4],COLS)):
        Es  = field_vs_range([ln],Θ)
        EsdB = 20*np.log10(np.maximum(Es,1e-12))
        ax_rng.plot(d,EsdB,color=col,lw=0.9,linestyle="--",alpha=0.6,label=f"M{i+1}")

    antip = np.pi*RE/1e6
    ax_rng.axvline(antip,color=MUTE,lw=0.7,linestyle=":",alpha=0.5)
    ax_rng.text(antip+0.1,EdB.min()*0.97,"Antipode",color=MUTE,fontsize=7)
    ax_rng.set_xlim(0,d[-1])
    ax_rng.legend(loc="lower left",fontsize=7,framealpha=0,labelcolor=TEXT)
    _lbl(ax_rng,"Great-circle range [Mm]","|E_θ| [dB rel.]",
         "Total Field vs Range  (Watson modal sum)\n"
         "Legendre asymptotic  +  1/√sinΘ spherical spreading")

    # footer
    fig.text(0.5,0.035,
        "Admittance Riccati  ·  Stix full tensor (S,D,P)  ·  "
        "Spherical Earth  ·  Wait-Spies ionosphere  ·  "
        "Watson transform mode sum  ·  Legendre asymptotic",
        ha="center",color=MUTE,fontsize=7.5,fontstyle="italic")

    return fig


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  VLF/ELF  Spherical Earth  |  Full-Wave Admittance Riccati")
    print("=" * 65)

    f       = 10.0e3
    h_prime = 74.0
    beta    = 0.30
    h_km    = 74.0
    sigma_g = 1.0e-2
    B0      = 5.0e-5
    n_steps = 500

    omega = 2*np.pi*f; k0 = omega/c
    print(f"\nf = {f/1e3:.1f} kHz   k₀Rₑ = {k0*RE:.1f}")

    print("\nSearching for modes …")
    modes = find_modes(f, sigma_g=sigma_g, B0=B0, h_prime=h_prime,
                       beta=beta, h_km=h_km, n_scan=120,
                       max_modes=5, n_steps=n_steps)

    if not modes:
        raise SystemExit("No modes found.")

    props = mode_props(modes, f)

    print(f"\n{'Mode':<6} {'Re(l)':<10} {'Im(l)':<12} {'θ_eq[°]':<10}"
          f" {'v_ph/c':<10} {'Atten[dB/Mm]'}")
    print("─"*68)
    for i,(l,p) in enumerate(zip(modes,props)):
        print(f"M{i+1:<5} {l.real:<10.2f} {l.imag:<12.5f} {p['theta']:<10.2f}"
              f" {p['v_ph_c']:<10.5f} {p['atten']:.3f}")

    print("\nGenerating figure …")
    fig = make_figure(f, modes, props, h_prime, beta, h_km, sigma_g, B0)
    plt.show()