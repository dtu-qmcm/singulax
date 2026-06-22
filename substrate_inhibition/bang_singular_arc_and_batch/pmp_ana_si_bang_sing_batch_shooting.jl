"""
pmp_si_bang_sing_batch_shooting.jl

Two-variable PMP shooting for Bang → Singular arc → Batch.
Mirrors pmp_si_bang_sing_batch.py but in Julia.

Unknowns  : t1 (bang→singular switch), l4_tf = λ₄(t_end)
Conditions: φ(t1) = 0,  φ(t2) = 0   where φ = λ₂·S_F + λ₄
"""

import Pkg
for pkg in ["NonlinearSolve", "ForwardDiff"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

using OrdinaryDiffEq
using NonlinearSolve
using LinearAlgebra
using ForwardDiff
using Printf

# ── Problem constants ─────────────────────────────────────────────────────────
const Ks    = 0.1
const Ki    = 100.0
const S0    = 1e-6
const F_MAX = 2.0
const SF    = 150.0
const VMAX  = 4.0
const TEND  = 9.0
const SSTAR = sqrt(Ks * Ki)
const y0_4  = [10.0, log(S0), 0.0, 1.0]   # [XV, log(SV), PV, V]

# ── Kinetics ──────────────────────────────────────────────────────────────────
_mu(S)    = 0.3 * S / (Ks + S + S^2 / Ki)
_pi(S)    = 0.5 * _mu(S)
_mf(S)    = 0.01 * S / (S + 0.001)
_sigma(S) = _mu(S) / 0.5 + _pi(S) / 0.2 + _mf(S)

const SIGMA_STAR = _sigma(SSTAR)

# Analytical F_sin: keeps S = S★ (from d(S·V)/dt = 0)
_Fsin(XV) = SIGMA_STAR * XV / (SF - SSTAR)

# Switching function  φ = λ₂·S_F + λ₄
_phi(y) = y[6] * SF + y[8]

# ── State RHS in log(SV) coordinates ─────────────────────────────────────────
function state_rhs(x, F)
    SV = exp(x[2]); S = SV / x[4]
    [_mu(S) * x[1],
     (-_sigma(S) * x[1] + SF * F) / SV,
     _pi(S) * x[1],
     F]
end

# ── Costate RHS via ForwardDiff ───────────────────────────────────────────────
# Hamiltonian in SV-space (original coordinates):  H = λᵀ(f_orig + g_orig·F)
function costate_rhs(y, F)
    x = y[1:4]; l = y[5:8]
    SV  = exp(x[2])
    xo  = [x[1], SV, x[3], x[4]]       # log(SV) → SV
    function H(xo_)
        S   = xo_[2] / xo_[4]
        f_o = [_mu(S)*xo_[1], -_sigma(S)*xo_[1], _pi(S)*xo_[1], 0.0]
        g_o = [0.0, SF, 0.0, 1.0]
        dot(l, f_o + g_o * F)
    end
    return -ForwardDiff.gradient(H, xo)
end

# ── ODE functions ─────────────────────────────────────────────────────────────

# 4D (forward pass, no costates)
bang_fwd!(du, u, p, t)     = (du .= state_rhs(u, F_MAX))
singular_fwd!(du, u, p, t) = (du .= state_rhs(u, _Fsin(u[1])))
batch_fwd!(du, u, p, t)    = (du .= state_rhs(u, 0.0))

# 8D (backward pass, states + costates, analytical state-based control)
# autodiff=false prevents nested AD (ForwardDiff inside ForwardDiff)
function batch_8d!(du, u, p, t)
    du[1:4] .= state_rhs(u[1:4], 0.0)
    du[5:8] .= costate_rhs(u, 0.0)
end
function singular_8d!(du, u, p, t)
    F = _Fsin(u[1])
    du[1:4] .= state_rhs(u[1:4], F)
    du[5:8] .= costate_rhs(u, F)
end
function bang_8d!(du, u, p, t)
    du[1:4] .= state_rhs(u[1:4], F_MAX)
    du[5:8] .= costate_rhs(u, F_MAX)
end

# ── Solver wrappers ───────────────────────────────────────────────────────────
const ALG_4D = Rodas5P()
const ALG_8D = Rodas5P(autodiff=AutoFiniteDiff())   # avoids nested ForwardDiff

function solve4(f!, y0, tspan)
    sol = solve(ODEProblem(f!, copy(y0), tspan), ALG_4D;
                reltol=1e-8, abstol=1e-8, save_everystep=false)
    sol.u[end]
end

function solve8(f!, y0, tspan)
    sol = solve(ODEProblem(f!, copy(y0), tspan), ALG_8D;
                reltol=1e-8, abstol=1e-8, save_everystep=false)
    sol.u[end]
end

# ── Main shooting residuals ───────────────────────────────────────────────────
function shoot(t1, l4_tf)

    # 1. Forward bang [0, t1]
    y_t1 = solve4(bang_fwd!, y0_4, (0.0, t1))

    # 2. Singular arc [t1, TEND] — stop when V = VMAX
    cb     = ContinuousCallback((u,t,i) -> u[4] - VMAX, terminate!)
    sol_ev = solve(ODEProblem(singular_fwd!, copy(y_t1), (t1, TEND)), ALG_4D;
                   reltol=1e-8, abstol=1e-8, callback=cb,
                   save_everystep=false)
    t2   = sol_ev.t[end]
    y_t2 = sol_ev.u[end]

    # 3. Forward batch [t2, TEND]
    x_tf = solve4(batch_fwd!, y_t2, (t2, TEND))

    # 4. Terminal costate  [x(tf), 0, 0, 1, l4_tf]
    y_tf = [x_tf[1], x_tf[2], x_tf[3], x_tf[4], 0.0, 0.0, 1.0, l4_tf]

    # 5. Backward batch [TEND → t2]
    y_t2_bwd = solve8(batch_8d!, y_tf, (TEND, t2))

    # 6. Backward singular [t2 → t1]
    y_t1_bwd = solve8(singular_8d!, y_t2_bwd, (t2, t1))

    # 7. Backward bang [t1 → 0]   (not needed for residuals but completes the arc)
    solve8(bang_8d!, y_t1_bwd, (t1, 0.0))

    return _phi(y_t1_bwd), _phi(y_t2_bwd), t2
end

# ── NLsolve wrapper ───────────────────────────────────────────────────────────
const t1_guess    = 0.01
const l4_tf_guess = -16.0

function residuals!(res, x, p)
    t1, l4_tf = x[1], x[2]
    try
        phi1, phi2, _ = shoot(t1, l4_tf)
        res[1] = isfinite(phi1) ? phi1 : 1e6
        res[2] = isfinite(phi2) ? phi2 : 1e6
    catch
        res[1] = 1e6
        res[2] = 1e6
    end
    @printf("  t1=%.6f  l4_tf=%.6f  →  φ(t1)=%.3e  φ(t2)=%.3e\n",
            t1, l4_tf, res[1], res[2])
end

# Initial check
phi1_0, phi2_0, t2_0 = shoot(t1_guess, l4_tf_guess)
@printf("\nInitial residuals:  φ(t1) = %.4e   φ(t2) = %.4e\n", phi1_0, phi2_0)
@printf("t2 (initial guess)  = %.4f h\n\n", t2_0)

println("Running NonlinearSolve (Newton + finite-difference Jacobian) ...")
prob   = NonlinearProblem(residuals!, [t1_guess, l4_tf_guess], nothing)
sol_nl = solve(prob, NewtonRaphson(; autodiff=AutoFiniteDiff()); abstol=1e-8)

t1_opt    = sol_nl.u[1]
l4_tf_opt = sol_nl.u[2]
@printf("\nOptimal:  t1 = %.6f h   l4_tf = %.6f\n", t1_opt, l4_tf_opt)

phi1_f, phi2_f, t2_f = shoot(t1_opt, l4_tf_opt)
@printf("Final residuals:  φ(t1) = %.2e   φ(t2) = %.2e\n", phi1_f, phi2_f)
@printf("t2 (V = VMAX)     = %.6f h\n", t2_f)
