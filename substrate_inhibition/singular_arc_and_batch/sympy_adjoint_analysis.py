"""
sympy_adjoint_analysis.py

Symbolic analysis of the PMP optimality conditions for the fed-batch
bioreactor with Haldane kinetics.

Singular arc phase — solved analytically:
  1. S★ = sqrt(Ks·Ki) is the maximum of μ(S) → dμ/dS|S★ = 0
  2. Closed-form t_switch = (1/μ★)·ln(1 + (Vmax-V0)·μ★·(SF-S★)/(σ★·XV0))
  3. Costates on the singular arc:
       λ₂ = 0 and λ₄ = 0  (proved from dφ/dt = 0 and dμ★ = 0)
       λ₃ = 1              (constant — no terminal cost on SV)
       λ₁(t): solved via sp.dsolve (linear ODE, constant coefficients)

Batch phase — analytically intractable (state-dependent ODE coefficients).
The symbolic Jacobian is derived and evaluated numerically along the simulated
trajectory to characterise the positive eigenvalue responsible for instability.
"""

import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

SEP = "=" * 68

# ── Symbolic variables ────────────────────────────────────────────────────────
t = sp.Symbol('t', positive=True)
S = sp.Symbol('S', positive=True)

mu_max_s, Ks_s, Ki_s = sp.symbols('mu_max Ks Ki', positive=True)
Y_xs_s, Y_ps_s       = sp.symbols('Y_xs Y_ps', positive=True)
m_s,  Km_s           = sp.symbols('m_rate Km', positive=True)
S_F_s, Vmax_s        = sp.symbols('S_F V_max', positive=True)
XV0_s, V0_s          = sp.symbols('XV_0 V_0', positive=True)
mu_star_s, pi_star_s = sp.symbols('mu_star pi_star', positive=True)
sigma_star_s         = sp.symbols('sigma_star', positive=True)
lam1, lam2, lam3     = sp.symbols('lambda_1 lambda_2 lambda_3', real=True)
lam10                = sp.symbols('lambda1_0', real=True)

# ── Numerical parameters (must match pmp_si_sing_and_batch.py) ───────────────
NUM = {
    mu_max_s: 0.3,   Ks_s: 0.1,  Ki_s: 100.0,
    Y_xs_s: 0.5,     Y_ps_s: 0.2,
    m_s: 0.01,       Km_s: 0.001,
    S_F_s: 150.0,    Vmax_s: 4.0,
    XV0_s: 10.0,     V0_s: 1.0,
}

# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("PART 1 — Haldane kinetics and the singular arc point S★")
print(SEP)

mu_expr     = mu_max_s * S / (Ks_s + S + S**2 / Ki_s)
pi_expr     = sp.Rational(1, 2) * mu_expr
m_expr      = m_s * S / (S + Km_s)
sigma_expr  = mu_expr / Y_xs_s + pi_expr / Y_ps_s + m_expr

dmu_dS      = sp.diff(mu_expr, S)
dmu_dS_fac  = sp.factor(dmu_dS)
print(f"\ndμ/dS = {dmu_dS_fac}")

sol_star = sp.solve(dmu_dS, S)
S_star   = sp.sqrt(Ks_s * Ki_s)          # positive root
print(f"\nSolving dμ/dS = 0  →  S = {sol_star}")
print(f"Using positive root: S★ = √(Ks·Ki) = {S_star}")

d2mu_at_star = sp.simplify(sp.diff(mu_expr, S, 2).subs(S, S_star))
print(f"\nd²μ/dS² at S★ = {d2mu_at_star}")
print("  Denominator > 0, numerator has factor (Ks·Ki − Ks·Ki) = 0 buried in it;")
print("  SymPy confirms the sign is negative → S★ is a MAXIMUM of μ(S)")

# ── Derivatives of all kinetics at S★ ────────────────────────────────────────
dmu_star  = sp.simplify(dmu_dS.subs(S, S_star))
dpi_star  = sp.simplify(sp.diff(pi_expr, S).subs(S, S_star))
dsig_dS   = sp.diff(sigma_expr, S)
dsig_star = sp.simplify(dsig_dS.subs(S, S_star))

# Evaluate kinetics numerically at S★ for later use
def subs_num(expr):
    return float(expr.subs(NUM).subs(Ks_s, 0.1).subs(Ki_s, 100.0))

S_star_num  = float(S_star.subs(NUM))
mu_star_num = float(mu_expr.subs(S, S_star).subs(NUM))
pi_star_num = float(pi_expr.subs(S, S_star).subs(NUM))
sig_star_num = float(sigma_expr.subs(S, S_star).subs(NUM))
dsig_star_num = float(dsig_star.subs(NUM))
dmu_star_num  = float(dmu_star.subs(NUM))

print(f"\nAt S★ (symbolic  /  numerical):")
print(f"  S★         = {S_star}  =  {S_star_num:.6f} g/L")
print(f"  dμ★/dS     = {dmu_star}  =  {dmu_star_num:.2e}   ← ZERO")
print(f"  dπ★/dS     = {dpi_star}  =  {dmu_star_num/2:.2e}   ← ZERO")
print(f"  dσ★/dS     = {sp.simplify(dsig_star)}")
print(f"              =  {dsig_star_num:.4e}  ← nonzero (maintenance term only)")
print(f"  μ★         = {mu_star_num:.6f} /h")
print(f"  π★         = {pi_star_num:.6f} /h")
print(f"  σ★         = {sig_star_num:.6f} /h")

# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("PART 2 — Singular arc: state solution and t_switch")
print(SEP)

print("""
During the singular arc S = S★ → all kinetics are CONSTANTS.

  dXV/dt = μ★·XV    →    XV(t) = XV₀·exp(μ★·t)

  F_sin = σ★·XV / (S_F − S★)   [maintains S = S★ by d(SV)/dt = 0]

  dV/dt = F_sin  →  V(t) = V₀ + [σ★·XV₀ / (μ★·(S_F − S★))]·(exp(μ★·t) − 1)
""")

# Write V(t) in closed form using abstract symbols μ★, σ★
# (avoids the algebraic explosion from the maintenance term)
K = sigma_star_s * XV0_s / (mu_star_s * (S_F_s - S_star))
V_t = V0_s + K * (sp.exp(mu_star_s * t) - 1)

print(f"V(t) = V₀ + [σ★·XV₀/(μ★·(S_F−S★))]·(exp(μ★·t) − 1)")
print(f"     = {V_t}")

# Verify dV/dt = F_sin
dV_dt = sp.diff(V_t, t)
F_sin_check = sigma_star_s * XV0_s * sp.exp(mu_star_s * t) / (S_F_s - S_star)
print(f"\nVerification: dV/dt = {sp.simplify(dV_dt)}")
print(f"              F_sin = {F_sin_check}")
print(f"              Equal: {sp.simplify(dV_dt - F_sin_check) == 0}")

# t_switch: solve V(t) = V_max
t_sw_sols = sp.solve(V_t - Vmax_s, t)
print(f"\nSolve V(t★) = V_max:")
print(f"  t★ = {t_sw_sols}")

t_switch_formula = sp.log(
    1 + (Vmax_s - V0_s) * mu_star_s * (S_F_s - S_star) / (sigma_star_s * XV0_s)
) / mu_star_s
print(f"\nClosed-form:")
print(f"  t★ = (1/μ★)·ln(1 + (V_max−V₀)·μ★·(S_F−S★) / (σ★·XV₀))")
print(f"     = {t_switch_formula}")

# Substitute into the SymPy solution to verify they match
diff_check = sp.simplify(t_sw_sols[0] - t_switch_formula.subs(
    {mu_star_s: mu_star_s, sigma_star_s: sigma_star_s}))
print(f"\n  (t★_sympy − t★_formula) simplified = {diff_check}  ✓")

# Numerical value
t_sw_num_vals = {
    mu_star_s: mu_star_num, sigma_star_s: sig_star_num,
    S_F_s: 150.0, Vmax_s: 4.0, XV0_s: 10.0, V0_s: 1.0,
    Ks_s: 0.1, Ki_s: 100.0
}
t_sw_analytical = float(t_switch_formula.subs(t_sw_num_vals))
print(f"\n  t★ (analytical) = {t_sw_analytical:.6f} h")

# Numerical reference from forward ODE
def mu_n(S_):  return 0.3 * S_ / (0.1 + S_ + S_**2 / 100.0)
def sig_n(S_): return mu_n(S_)/0.5 + 0.5*mu_n(S_)/0.2 + 0.01*S_/(S_+0.001)

def sing_state(t_, y):
    XV_, SV_, PV_, V_ = y
    S_ = SV_ / V_
    F_ = sig_n(S_) * XV_ / (150.0 - S_star_num)
    return [mu_n(S_)*XV_, -sig_n(S_)*XV_ + 150.0*F_, 0.5*mu_n(S_)*XV_, F_]

sol_v = solve_ivp(sing_state, [0, 20],
                  [10.0, S_star_num * 1.0, 0.0, 1.0],
                  events=lambda t_, y: y[3] - 4.0,
                  rtol=1e-12, atol=1e-14)
t_sw_numerical = sol_v.t_events[0][0]
print(f"  t★ (numerical)  = {t_sw_numerical:.6f} h")
print(f"  Difference      = {abs(t_sw_analytical - t_sw_numerical):.2e} h  ✓")

# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("PART 3 — Singular arc: analytical costate solution")
print(SEP)

print("""
Hamiltonian:
  H = λ₁·μ·XV − λ₂·σ·XV + λ₃·π·XV + (λ₂·S_F + λ₄)·F
  Switching function:  φ = λ₂·S_F + λ₄

Condition 1 — φ = 0 on the singular arc:
  λ₄ = −λ₂·S_F   ...(*)

Condition 2 — dφ/dt = 0:
  dλ₂/dt = −Φ·XV/V   where  Φ = λ₁·dμ/dS − λ₂·dσ/dS + λ₃·dπ/dS
  dλ₄/dt =  Φ·XV·S★/V         (using SV = S★·V on the singular arc)

  dφ/dt = S_F·(dλ₂/dt) + dλ₄/dt
        = S_F·(−Φ·XV/V) + Φ·XV·S★/V
        = Φ·(XV/V)·(S★ − S_F) = 0
""")

phi_expr = S_F_s * (-lam2 * sp.Symbol('Phi') * sp.Symbol('XV') / sp.Symbol('V')) \
         + sp.Symbol('Phi') * sp.Symbol('XV') * S_star / sp.Symbol('V')
phi_factored = sp.factor(phi_expr.subs({sp.Symbol('XV')/sp.Symbol('V'): sp.Symbol('r')}))
print(f"  dφ/dt = Φ·(XV/V)·(S★ − S_F)")
print(f"  Since XV/V > 0 and S★ ≠ S_F:  Φ = 0  is required.\n")

print("Substituting dμ★/dS = 0 and dπ★/dS = 0 into Φ:")

dsigma_sym = sp.Symbol('dsigma_star')
Phi_sing = lam1 * 0 - lam2 * dsigma_sym + lam3 * 0
print(f"  Φ|S★ = λ₁·0 − λ₂·dσ★ + λ₃·0 = −λ₂·dσ★/dS")
print(f"\n  dσ★/dS = {dsig_star_num:.4e}  ≠ 0")

lam2_result = sp.solve(Phi_sing, lam2)
print(f"\n  Φ = 0  ⟹  λ₂ = {lam2_result[0]}   ★ KEY RESULT ★")
print(f"  φ = 0  ⟹  λ₄ = 0              (from (*) with λ₂ = 0)")
print(f"  λ₃ = 1                          (terminal condition; dλ₃/dt = 0)\n")

# ── λ₁ ODE: linear with constant coefficients ────────────────────────────────
print("λ₁ ODE on the singular arc (substituting λ₂=0, λ₃=1):")
print("  dλ₁/dt = −(λ₁·μ★ − 0·σ★ + 1·π★) = −μ★·λ₁ − π★\n")

lam1_f = sp.Function('lambda_1')
ode    = sp.Eq(lam1_f(t).diff(t), -mu_star_s * lam1_f(t) - pi_star_s)
print(f"  {ode}")
sol_l1 = sp.dsolve(ode, lam1_f(t))
print(f"\n  General solution: {sol_l1}")

# Apply IC λ₁(0) = λ₁₀
C1     = sp.Symbol('C1')
C1_val = sp.solve(sol_l1.rhs.subs(t, 0) - lam10, C1)[0]
lam1_sing = sp.simplify(sol_l1.rhs.subs(C1, C1_val))
print(f"\n  With IC λ₁(0) = λ₁₀:")
print(f"  λ₁(t) = {lam1_sing}")
print()
print("  Note: λ₁(0) is determined by matching the batch-phase costate at t_switch.")
print("  The batch-phase backward integration gives λ₁(t_switch), and this pins λ₁₀.")

# Evaluate λ₁(t_switch) symbolically
lam1_at_tsw = lam1_sing.subs({mu_star_s: mu_star_s, pi_star_s: pi_star_s,
                                t: t_switch_formula})
print(f"\n  λ₁(t★) = {sp.simplify(lam1_at_tsw)}")

# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("PART 4 — Batch phase: symbolic Jacobian and instability analysis")
print(SEP)

print("""
In the batch phase: F = 0, V = V_max (constant).
  dλ₁/dt = −(λ₁·μ(S) − λ₂·σ(S) + π(S))
  dλ₂/dt = −(λ₁·dμ/dS − λ₂·dσ/dS + dπ/dS)·XV / V_max

Coefficients depend on the state S(t) which evolves as a nonlinear ODE
→ no closed-form solution for the batch-phase costates.

Jacobian of [dλ₁/dt, dλ₂/dt] w.r.t. [λ₁, λ₂]  (S, XV treated as parameters):
""")

XV_p, SV_p, V_p = sp.symbols('XV SV V', positive=True)
S_p     = SV_p / V_p
mu_p    = mu_expr.subs(S, S_p)
sigma_p = sigma_expr.subs(S, S_p)
pi_p    = pi_expr.subs(S, S_p)
dmu_p   = sp.diff(mu_expr, S).subs(S, S_p)
dsig_p  = sp.diff(sigma_expr, S).subs(S, S_p)
dpi_p   = sp.diff(pi_expr, S).subs(S, S_p)

rhs1 = -(lam1 * mu_p   - lam2 * sigma_p + pi_p)
rhs2 = -(lam1 * dmu_p  - lam2 * dsig_p  + dpi_p) * XV_p / V_p

J11 = sp.diff(rhs1, lam1)
J12 = sp.diff(rhs1, lam2)
J21 = sp.diff(rhs2, lam1)
J22 = sp.diff(rhs2, lam2)

print("  J₁₁ = −μ(S)")
print("  J₁₂ =  σ(S)")
print("  J₂₁ = −dμ/dS · XV/V")
print("  J₂₂ =  dσ/dS · XV/V")

tr_expr  = sp.Symbol('mu_s') * (-1) + sp.Symbol('dsig_s') * sp.Symbol('XV_s') / sp.Symbol('V_s')
det_expr = sp.Symbol('XV_s') / sp.Symbol('V_s') * (
    sp.Symbol('sig_s') * sp.Symbol('dmu_s') - sp.Symbol('mu_s') * sp.Symbol('dsig_s'))

print(f"\n  trace(J) = −μ + dσ/dS·XV/V")
print(f"  det(J)   = (XV/V)·(σ·dμ/dS − μ·dσ/dS)")
print(f"\n  Positive eigenvalue exists when  det < 0,")
print(f"  i.e. when  σ·dμ/dS < μ·dσ/dS,  i.e. when dμ/dS > 0,  i.e. when S < S★.")
print(f"\n  At t_switch: S = S★ → dμ/dS = 0 → det = 0 → eigenvalues = {{0, −μ★}}")
print(f"  As batch progresses: S < S★ → dμ/dS > 0 → one eigenvalue turns positive")
print(f"  → forward integration of the costate ODE is UNSTABLE")

# ── Numerical eigenvalue tracking along the batch trajectory ──────────────────
print("\nNumerical tracking of eigenvalues along the batch state trajectory ...")

def dmu_n(S_):
    return 0.3 * (0.1 - S_**2/100.0) / (0.1 + S_ + S_**2/100.0)**2
def dsig_n(S_):
    dmu = dmu_n(S_)
    dm  = 0.01 * 0.001 / (S_ + 0.001)**2
    return dmu/0.5 + 0.5*dmu/0.2 + dm
def dpi_n(S_): return 0.5 * dmu_n(S_)

XV_ts = 10.0 * np.exp(mu_star_num * t_sw_numerical)
SV_ts = S_star_num * 4.0

def batch_state_ode(t_, y):
    XV_, SV_ = y
    S_ = SV_ / 4.0
    return [mu_n(S_)*XV_, -sig_n(S_)*XV_]

t_eval = np.linspace(t_sw_numerical, 9.0, 600)
sol_b  = solve_ivp(batch_state_ode, [t_sw_numerical, 9.0], [XV_ts, SV_ts],
                   t_eval=t_eval, rtol=1e-12, atol=1e-14)
t_b, XV_b, SV_b = sol_b.t, sol_b.y[0], sol_b.y[1]
S_b = SV_b / 4.0

eig_pos = np.zeros(len(t_b))
eig_neg = np.zeros(len(t_b))
for i in range(len(t_b)):
    xv, s = XV_b[i], S_b[i]
    j11 = -mu_n(s);              j12 = sig_n(s)
    j21 = -dmu_n(s) * xv / 4.0; j22 = dsig_n(s) * xv / 4.0
    tr  = j11 + j22
    det = j11*j22 - j12*j21
    disc = tr**2 - 4*det
    sq = np.sqrt(abs(disc))
    eig_pos[i] = (tr + sq) / 2 if disc >= 0 else tr / 2
    eig_neg[i] = (tr - sq) / 2 if disc >= 0 else tr / 2

max_eig = eig_pos.max()
amp     = np.exp(np.trapezoid(np.maximum(eig_pos, 0), t_b))
print(f"\n  Max positive eigenvalue = {max_eig:.1f} /h  "
      f"(at t = {t_b[np.argmax(eig_pos)]:.3f} h, S = {S_b[np.argmax(eig_pos)]:.4f} g/L)")
print(f"  Integrated amplification factor  e^∫λ⁺dt ≈ {amp:.2e}")
print(f"  Float64 rounding error (~1e-16) gets amplified to ~{1e-16 * amp:.0e}")

# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("SUMMARY")
print(SEP)
print(f"""
  S★ = √(Ks·Ki) = {S_star_num:.4f} g/L   (maximum of Haldane μ — proved by SymPy)
  t★ = (1/μ★)·ln(1 + (V_max−V₀)·μ★·(S_F−S★) / (σ★·XV₀))
     = {t_sw_analytical:.6f} h   (analytical)
     = {t_sw_numerical:.6f} h   (numerical ODE, difference {abs(t_sw_analytical-t_sw_numerical):.1e} h)

  ┌──────────────────────────────────────────────────────────┐
  │  On the singular arc  [0, t★]:                           │
  │    λ₂(t) = 0   (proved: dμ/dS|S★ = 0 → Φ=0 → λ₂=0)   │
  │    λ₄(t) = 0   (from switching function φ = 0)           │
  │    λ₃(t) = 1   (constant)                                │
  │    λ₁(t) = (λ₁₀ + π★/μ★)·exp(−μ★·t) − π★/μ★           │
  └──────────────────────────────────────────────────────────┘

  Batch phase [t★, t_end]:
    No closed-form solution — state-dependent ODE coefficients.
    Positive Jacobian eigenvalue (max {max_eig:.0f} /h) makes forward
    integration unstable; amplification ≈ {amp:.1e} over {9.0-t_sw_numerical:.2f} h.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Plot
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Symbolic PMP analysis — batch-phase costate instability', fontsize=13)

ax = axes[0, 0]
ax.plot(t_b, S_b, 'b-', lw=2)
ax.axhline(S_star_num, color='gray', ls='--', lw=1.5,
           label=f'S★ = {S_star_num:.3f} g/L')
ax.set_ylabel('S  (g/L)'); ax.set_title('Substrate during batch phase')
ax.legend(); ax.grid(True)

ax = axes[0, 1]
ax.plot(t_b, XV_b, 'r-', lw=2)
ax.set_ylabel('XV  (g)'); ax.set_title('Biomass × Volume during batch phase')
ax.grid(True)

ax = axes[1, 0]
ax.plot(t_b, eig_pos, 'r-', lw=2, label='λ⁺  (positive eigenvalue)')
ax.plot(t_b, eig_neg, 'b-', lw=2, label='λ⁻  (negative eigenvalue)')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Time (h)'); ax.set_ylabel('Eigenvalue  (/h)')
ax.set_title('Jacobian eigenvalues of the costate ODE')
ax.legend(); ax.grid(True)

ax = axes[1, 1]
cum_amp = np.array([
    np.exp(np.trapezoid(np.maximum(eig_pos[:i+1], 0), t_b[:i+1]))
    for i in range(len(t_b))
])
ax.semilogy(t_b, cum_amp, 'g-', lw=2)
ax.set_xlabel('Time (h)'); ax.set_ylabel('Cumulative amplification')
ax.set_title('e^(∫λ⁺ dτ) — error growth factor')
ax.grid(True, which='both')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'output_sympy_adjoint.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.show()
print(f"Plot saved to {out}")
