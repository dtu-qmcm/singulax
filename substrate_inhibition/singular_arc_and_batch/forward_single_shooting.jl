"""
Forward single-shooting comparison across Julia ODE solvers.

Same problem as forward_single_shooting.py:
  - Backward shooting gives the exact initial conditions at t_switch
  - We then try forward integration of the batch phase with every
    available Julia solver and report λ₂(t_end) — expected to be 0.

All solvers should fail for the same reason: the costate ODE has a
positive eigenvalue (~641/h) that amplifies any float64 rounding error
by ~10^167 over the 0.6 h batch phase.
"""

using OrdinaryDiffEq
using OrdinaryDiffEqBDF      # FBDF, QNDF, ABDF2 + exports terminate!
using OrdinaryDiffEqSDIRK    # KenCarp4/5, Kvaerno5, TRBDF2, Hairer4
using LSODA
using Optim
using Printf
using Plots

# ── Kinetics (Haldane + maintenance) ─────────────────────────────────────────
const mu_max = 0.3
const Ks     = 0.1
const Ki     = 100.0
const Y_xs   = 0.5
const Y_ps   = 0.2
const m_rate = 0.01
const Km     = 0.001

mu_k(S)    = mu_max * S / (Ks + S + S^2/Ki)
pi_k(S)    = 0.5 * mu_k(S)
m_func(S)  = m_rate * S / (S + Km)
sigma_k(S) = mu_k(S)/Y_xs + pi_k(S)/Y_ps + m_func(S)

# Exact derivatives w.r.t. S (for costate equations)
dmu(S)    = mu_max * (Ks - S^2/Ki) / (Ks + S + S^2/Ki)^2
dpi(S)    = 0.5 * dmu(S)
dm(S)     = m_rate * Km / (S + Km)^2
dsigma(S) = dmu(S)/Y_xs + dpi(S)/Y_ps + dm(S)

# ── Problem parameters ────────────────────────────────────────────────────────
const S_F   = 150.0
const V_max = 4.0
const F_max = 2.0
const F_min = 0.0
const t_end = 9.0
const S_star = sqrt(Ks * Ki)   # optimal singular arc substrate ≈ 3.162 g/L

# Initial conditions: state = [XV, SV, PV, V]
const u0 = [10.0, S_star * 1.0, 0.0, 1.0]

# ── Singular arc feed rate ────────────────────────────────────────────────────
# Maintains S = S★ by balance: d(SV)/dt = 0  →  F = σ(S★)·XV / (S_F − S★)
F_sing(XV) = clamp(sigma_k(S_star) * XV / (S_F - S_star), F_min, F_max)

# ── ODE: state only (for finding t_switch and x_tf) ──────────────────────────
function state_sing!(du, u, p, t)
    XV, SV, PV, V = u
    F = F_sing(XV)
    S = SV / V
    du[1] = mu_k(S) * XV
    du[2] = -sigma_k(S) * XV + S_F * F
    du[3] = pi_k(S) * XV
    du[4] = F
end

function state_batch!(du, u, p, t)   # F = 0
    XV, SV, PV, V = u
    S = SV / V
    du[1] = mu_k(S) * XV
    du[2] = -sigma_k(S) * XV
    du[3] = pi_k(S) * XV
    du[4] = 0.0
end

# ── ODE: state + costate (8-dim) ─────────────────────────────────────────────
# State  y[1:4] = [XV, SV, PV, V]   (original, not log-transformed)
# Costate y[5:8] = [λ₁, λ₂, λ₃, λ₄] conjugate to [XV, SV, PV, V]
#
# Hamiltonian: H = λ₁·μ·XV + λ₂·(−σ·XV + S_F·F) + λ₃·π·XV + λ₄·F
# Costate:    dλ/dt = −∂H/∂x
function full_ode!(dy, y, p, t)
    XV, SV, PV, V  = y[1], y[2], y[3], y[4]
    λ1, λ2, λ3, λ4 = y[5], y[6], y[7], y[8]
    F = p

    S  = SV / V
    μ  = mu_k(S);     dμ = dmu(S)
    π  = pi_k(S);     dπ = dpi(S)
    σ  = sigma_k(S);  dσ = dsigma(S)

    # State equations
    dy[1] =  μ * XV
    dy[2] = -σ * XV + S_F * F
    dy[3] =  π * XV
    dy[4] =  F

    # Costate equations  dλ/dt = −∂H/∂x
    Φ = λ1*dμ - λ2*dσ + λ3*dπ   # XV·Φ = ∂H/∂S;  ∂S/∂(SV)=1/V, ∂S/∂V=−S/V
    dy[5] = -(λ1*μ - λ2*σ + λ3*π)
    dy[6] = -Φ * XV / V
    dy[7] =  0.0
    dy[8] =  Φ * XV * SV / V^2
end

# Switching function φ = λ₂·S_F + λ₄  (zero on the singular arc)
phi(y) = y[6] * S_F + y[8]

# ── Step 1: forward analytical solution → t_switch, x_tf ─────────────────────
println("Step 1 — forward analytical solution ...")

cb_Vmax = ContinuousCallback((u, t, int) -> u[4] - V_max, terminate!)
prob_sing = ODEProblem(state_sing!, u0, (0.0, t_end))
sol_sing  = solve(prob_sing, Rodas5P(); abstol=1e-10, reltol=1e-8, callback=cb_Vmax)
t_switch  = sol_sing.t[end]
x_at_tswitch_states = sol_sing.u[end]

prob_batch_ana = ODEProblem(state_batch!, x_at_tswitch_states, (t_switch, t_end))
sol_batch_ana  = solve(prob_batch_ana, Rodas5P(); abstol=1e-10, reltol=1e-8)
x_tf = sol_batch_ana.u[end]

@printf("  t_switch = %.4f h\n", t_switch)
@printf("  x_tf: XV=%.4f  S=%.4f  PV=%.4f  V=%.4f\n",
        x_tf[1], x_tf[2]/x_tf[4], x_tf[3], x_tf[4])

# ── Step 2: backward shooting → l4_tf ────────────────────────────────────────
println("Step 2 — backward shooting ...")

function shoot_residual(l4_tf)
    y_tf  = [x_tf[1], x_tf[2], x_tf[3], x_tf[4], 0.0, 0.0, 1.0, l4_tf]
    prob  = ODEProblem(full_ode!, y_tf, (t_end, t_switch), 0.0)
    sol   = solve(prob, Rodas5P(); abstol=1e-10, reltol=1e-8, dense=false)
    return phi(sol.u[end])^2
end

opt    = optimize(shoot_residual, -25.0, 0.0, Brent(); rel_tol=1e-10)
l4_tf  = Optim.minimizer(opt)
@printf("  l4_tf = %.6f  (residual = %.2e)\n", l4_tf, sqrt(Optim.minimum(opt)))

y_tf_opt = [x_tf[1], x_tf[2], x_tf[3], x_tf[4], 0.0, 0.0, 1.0, l4_tf]

# ── Step 3: backward reference integration ────────────────────────────────────
println("Step 3 — backward reference integration ...")

prob_bwd = ODEProblem(full_ode!, y_tf_opt, (t_end, t_switch), 0.0)
sol_bwd  = solve(prob_bwd, Rodas5P();
                 abstol=1e-10, reltol=1e-8,
                 saveat=range(t_end, t_switch; length=500))
y_at_tswitch = sol_bwd.u[end]

@printf("  λ₂(t_switch) backward = %.8f\n", y_at_tswitch[6])
@printf("  φ(t_switch)            = %.2e  (should be 0)\n", phi(y_at_tswitch))

# ── Step 4: forward solver comparison ────────────────────────────────────────
println()
println("="^68)
@printf("  %-30s  %8s  %14s\n", "Solver", "reltol", "λ₂(t_end)")
println("-"^68)

function try_forward(label, solver, reltol; atol=reltol*1e-2)
    prob = ODEProblem(full_ode!, y_at_tswitch, (t_switch, t_end), 0.0)
    try
        sol = solve(prob, solver;
                    abstol=atol, reltol=reltol,
                    saveat=range(t_switch, t_end; length=300),
                    maxiters=Int(1e7))
        u_end  = sol.u[end]
        l2_end = u_end[6]
        V_end  = u_end[4]
        V_drift = V_end - y_at_tswitch[4]
        @printf("  %-30s  %8.0e  %14.4e  V(t_end)=%10.6f  drift=%9.2e\n",
                label, reltol, l2_end, V_end, V_drift)
        return sol
    catch e
        @printf("  %-30s  %8.0e  %14s\n", label, reltol, "FAILED: $(typeof(e))")
        return nothing
    end
end

# Explicit solvers
println("  Explicit (non-stiff):")
s_tsit5  = try_forward("Tsit5",          Tsit5(),   1e-8)
s_vern7  = try_forward("Vern7",          Vern7(),   1e-8)
s_vern9  = try_forward("Vern9",          Vern9(),   1e-8)
s_vern8  = try_forward("Vern8",          Vern8(),   1e-8)
s_avern8 = try_forward("AutoVern8",     AutoVern8(Rodas5P()),   1e-8)

# Stiff / implicit solvers
println("  Stiff / implicit:")
s_rodas  = try_forward("Rodas5P",         Rodas5P(),  1e-8)
s_kv5    = try_forward("Kvaerno5",        Kvaerno5(), 1e-8)
s_kc4    = try_forward("KenCarp4",        KenCarp4(), 1e-8)
s_kc5    = try_forward("KenCarp5",        KenCarp5(), 1e-8)
s_qndf   = try_forward("QNDF",           QNDF(),     1e-8)
s_fbdf   = try_forward("FBDF",           FBDF(),     1e-8)
s_trbdf  = try_forward("TRBDF2",         TRBDF2(),   1e-8)
s_h4     = try_forward("Hairer4",        Hairer4(),  1e-8)

# Tight tolerances
println("  Tight tolerances (1e-12):")
s_rodast = try_forward("Rodas5P tight",   Rodas5P(),  1e-12)
s_kv5t   = try_forward("Kvaerno5 tight",  Kvaerno5(), 1e-12)
s_vern9t = try_forward("Vern9 tight",     Vern9(),    1e-12)

# LSODA with atol=0 (pure relative tolerance control)
println("  LSODA  (atol=0 — pure relative tolerance):")
s_lsoda      = try_forward("LSODA",             lsoda(), 1e-8;  atol=0.0)
s_lsoda_t    = try_forward("LSODA tight",       lsoda(), 1e-12; atol=0.0)
s_lsoda_tt   = try_forward("LSODA very tight",  lsoda(), 1e-14; atol=0.0)

println("="^68)
println("  λ₂(t_end) expected = 0.0  (terminal optimality condition)")
@printf("  V(t_switch) = %.8f  (should remain constant throughout)\n", y_at_tswitch[4])

# ── Plot ──────────────────────────────────────────────────────────────────────
t_ref  = reverse(sol_bwd.t)
ys_ref = reverse(sol_bwd.u)   # vector of 8-element state vectors

# Reference state and costate arrays
XV_ref  = [u[1]        for u in ys_ref]
S_ref   = [u[2]/u[4]   for u in ys_ref]   # S = SV/V
PV_ref  = [u[3]        for u in ys_ref]
V_ref   = [u[4]        for u in ys_ref]
λ1_ref  = [u[5]        for u in ys_ref]
λ2_ref  = [u[6]        for u in ys_ref]
λ3_ref  = [u[7]        for u in ys_ref]
λ4_ref  = [u[8]        for u in ys_ref]

# Solvers to overlay (states are stable so all agree; costates diverge)
overlay = [
    (s_tsit5,  "Tsit5",              :blue),
    (s_rodas,  "Rodas5P",            :red),
    (s_kv5,    "Kvaerno5",           :green),
    (s_vern9,  "Vern9",              :orange),
    (s_rodast, "Rodas5P 1e-12",      :purple),
]

# ── Figure 1: states ──────────────────────────────────────────────────────────
ps1 = plot(t_ref, XV_ref; lw=2.5, color=:black, label="Backward (ref)",
           ylabel="XV  (g)", title="Biomass × Volume", legend=:topleft)
ps2 = plot(t_ref, S_ref;  lw=2.5, color=:black, label="Backward (ref)",
           ylabel="S  (g/L)", title="Substrate concentration")
ps3 = plot(t_ref, PV_ref; lw=2.5, color=:black, label="Backward (ref)",
           ylabel="PV  (g)", title="Product × Volume")
ps4 = plot(t_ref, V_ref;  lw=2.5, color=:black, label="Backward (ref)",
           xlabel="Time (h)", ylabel="V  (L)", title="Volume")

for (sol, label, col) in overlay
    sol === nothing && continue
    ts = sol.t
    plot!(ps1, ts, [u[1]      for u in sol.u]; lw=1.4, ls=:dash, color=col, label=label)
    plot!(ps2, ts, [u[2]/u[4] for u in sol.u]; lw=1.4, ls=:dash, color=col, label=label)
    plot!(ps3, ts, [u[3]      for u in sol.u]; lw=1.4, ls=:dash, color=col, label=label)
    plot!(ps4, ts, [u[4]      for u in sol.u]; lw=1.4, ls=:dash, color=col, label=label)
end

fig_states = plot(ps1, ps2, ps3, ps4;
                  layout=(2, 2), size=(1100, 800),
                  plot_title="States — batch phase forward integration (Julia)\nAll solvers agree: instability is in costates only")
savefig(fig_states, "output_forward_julia.png")
println("\nState plot saved to output_forward_julia.png")

# ── Figure 2: costates ────────────────────────────────────────────────────────
pc1 = plot(t_ref, λ1_ref; lw=2.5, color=:black, label="Backward (ref)",
           ylabel="λ₁", title="λ₁")
pc2 = plot(t_ref, λ2_ref; lw=2.5, color=:black, label="Backward (ref)",
           ylabel="λ₂", title="λ₂  (clipped ±0.5)", ylim=(-0.5, 0.5))
pc3 = plot(t_ref, λ3_ref; lw=2.5, color=:black, label="Backward (ref)",
           ylabel="λ₃", title="λ₃")
pc4 = plot(t_ref, λ4_ref; lw=2.5, color=:black, label="Backward (ref)",
           xlabel="Time (h)", ylabel="λ₄", title="λ₄")

for (sol, label, col) in overlay
    sol === nothing && continue
    ts = sol.t
    plot!(pc1, ts, [u[5] for u in sol.u]; lw=1.4, ls=:dash, color=col, label=label)
    plot!(pc2, ts, clamp.([u[6] for u in sol.u], -0.5, 0.5); lw=1.4, ls=:dash, color=col, label=label)
    plot!(pc3, ts, [u[7] for u in sol.u]; lw=1.4, ls=:dash, color=col, label=label)
    plot!(pc4, ts, [u[8] for u in sol.u]; lw=1.4, ls=:dash, color=col, label=label)
end

fig_costates = plot(pc1, pc2, pc3, pc4;
                    layout=(2, 2), size=(1100, 800),
                    plot_title="Costates — batch phase forward integration (Julia)\nλ₂ explodes; other costates follow")
savefig(fig_costates, "output_forward_julia_costates.png")
println("Costate plot saved to output_forward_julia_costates.png")
