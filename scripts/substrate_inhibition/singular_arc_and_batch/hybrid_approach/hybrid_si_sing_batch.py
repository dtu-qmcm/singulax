"""
Hybrid two-stage approach to solve the case where the optimal control structure is Fsin->Fb(=0).  
"""
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx
from cyipopt import minimize_ipopt
from singulax.substrate_inhibition.shooting import solve_and_get_residuals_sing
from singulax.substrate_inhibition.odes import states_ode_phys, vf_states, vector_field, get_phi, l_to_orig, batch_phys_vf, objective_batch
from singulax.substrate_inhibition.Fsin import get_Fsin_unconstrained
from singulax.collocation import CollocParams, unpack, objective, constraints, jac_defects
import os
import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ---- PROBLEM DATA ----
Ks, Ki = 0.1, 100
S0 = jnp.sqrt(Ks * Ki)
S_F, V_max, F_max, F_min, t_end = 150.0, 4.0, 2.0, 0.0, 9.0
x0_phys = jnp.array([10.0, float(S0), 0.0, 1.0])            # [XV, SV, PV, V]
x0_log = jnp.array([10.0, jnp.log(S0), 0.0, 1.0])           # [XV, log SV, PV, V]

N = 400; tf = t_end; h = tf / N; nx = 4; nX = (N + 1) * nx; nz = nX + N

p = CollocParams(N=N, nx=nx, h=h, nX=nX, nz=nz, S_F=S_F, f=states_ode_phys)
obj_jit = jax.jit(lambda z: objective(z, p))
grad_jit = jax.jit(jax.grad(lambda z: objective(z, p)))
con_jit = jax.jit(lambda z: constraints(z, p))

if __name__ == "__main__":
    # ===== STAGE 1: physical-SV collocation (only to seed lambda(0), t_switch) =====
    print("=== STAGE 1: physical-SV collocation (maximise PV) ===")
    F_init = 0.5
    def _sim(x, _):
        return x + h * states_ode_phys(x, F_init, S_F), x
    _, X_hist = jax.lax.scan(_sim, x0_phys, None, length=N)
    z0 = np.concatenate([np.asarray(jnp.concatenate([x0_phys[None, :], X_hist])).ravel(), np.full(N, F_init)])

    lb = np.full(nz, -np.inf); ub = np.full(nz, np.inf)
    lb[:nx] = ub[:nx] = np.asarray(x0_phys)
    lb[1 + 4 * np.arange(N + 1)] = 0.0            # SV >= 0
    ub[3 + 4 * np.arange(N + 1)] = V_max          # V <= V_max
    lb[nX:] = F_min; ub[nX:] = F_max

    cons = [{"type": "eq", "fun": lambda z: np.asarray(con_jit(jnp.asarray(z))), "jac": lambda z: jac_defects(z, p)}]
    result = minimize_ipopt(
        fun=lambda z: float(obj_jit(jnp.asarray(z))),
        jac=lambda z: np.asarray(grad_jit(jnp.asarray(z))),
        x0=z0, bounds=list(zip(lb, ub)), constraints=cons,
        options={"max_iter": 3000, "tol": 1e-6, "print_level": 0})

    X_opt, U_opt = unpack(jnp.asarray(result.x), p)
    X_opt = np.asarray(X_opt); U_opt = np.asarray(U_opt)
    mult_g = np.asarray(result.info["mult_g"]).reshape(N, nx)
    lam_t0_seed = mult_g[0, :]
    t_grid = np.linspace(0, tf, N + 1)
    hit = np.where(X_opt[:, 3] >= V_max - 1e-5)[0]
    t_switch_euler = float(t_grid[hit[0]]) if len(hit) else tf
    print("lambda(0) seed   =", np.round(lam_t0_seed, 5))
    print("t_switch seed    =", round(t_switch_euler, 4))
    print("X opt=", X_opt)
    # ===== STAGE 2: forward shoot, batch as a terminal cost =====
    print("\n=== STAGE 2: forward shoot (lambda(0), t_switch); batch = terminal cost ===")
    sc = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10)
    sing_vf = lambda t, y, a: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))))

    lam0 = jnp.asarray(lam_t0_seed)
    guess = jnp.concatenate([lam0, jnp.array([t_switch_euler])])
    scale = jnp.maximum(jnp.abs(guess), 1e-3)
    args = (t_end, x0_log, S_F, V_max, F_min, F_max)
    sol = optx.least_squares(lambda u, a: solve_and_get_residuals_sing(u * scale, args),
                             optx.LevenbergMarquardt(rtol=1e-12, atol=1e-14),
                             guess / scale, args=None, max_steps=500, throw=False)
    opt_sol = sol.value * scale
    res_r = solve_and_get_residuals_sing(opt_sol, args)
    lam0_opt = np.asarray(opt_sol[:4])
    t_switch_opt = float(np.clip(opt_sol[4], 0.5, tf - 0.05))
    print("  lambda(0) seed    =", np.round(np.asarray(lam0), 6))
    print("  lambda(0) refined =", np.round(lam0_opt, 6))
    print("  t_switch refined  =", round(t_switch_opt, 6))
    print(" res =", float(jnp.linalg.norm(res_r)))
    print(" residuals =", np.round(np.asarray(res_r), 8))

    # ===== reconstruct forward: singular arc then batch, check terminal state =====
    y0 = jnp.concatenate([x0_log, jnp.asarray(lam0_opt)])
    sol_s = diffrax.diffeqsolve(diffrax.ODETerm(sing_vf), diffrax.Tsit5(),
        t0=0.0, t1=t_switch_opt, dt0=0.01, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_switch_opt, 200)),
        stepsize_controller=sc, max_steps=100000, throw=False)
    y_sw = sol_s.ys[-1]
    sol_b = diffrax.diffeqsolve(diffrax.ODETerm(batch_phys_vf),
        diffrax.Tsit5(), t0=t_switch_opt, t1=tf, dt0=0.01, y0=y_sw[:4],
        args=S_F,
        saveat=diffrax.SaveAt(ts=jnp.linspace(t_switch_opt, tf, 100)),
        stepsize_controller=sc, max_steps=100000, throw=False)
    x_tf = np.asarray(sol_b.ys[-1])
    print("\nx(t_switch) = [XV=%.3f, SV=%.4f, PV=%.3f, V=%.4f]"
          % (y_sw[0], np.exp(y_sw[1]), y_sw[2], y_sw[3]))
    print("x(tf)       = [XV=%.3f, SV=%.4g, PV=%.3f, V=%.4f]"
          % (x_tf[0], np.exp(x_tf[1]), x_tf[2], x_tf[3]))
    print("PV(tf)      =", round(float(x_tf[2]), 4))

    # ================================ PLOTS ================================
    here = os.path.dirname(os.path.abspath(__file__))

    # ---------- STAGE 1: direct-collocation solution ----------
    tX = t_grid                                 # state nodes (N+1)
    tU = t_grid[:-1]                            # control nodes (N), piecewise constant
    S1 = X_opt[:, 1] / X_opt[:, 3]              # S = SV / V
    fig1, ax = plt.subplots(2, 2, figsize=(11, 7)); ax = ax.ravel()
    ax[0].plot(tX, X_opt[:, 0], label="XV"); ax[0].plot(tX, X_opt[:, 2], label="PV")
    ax[0].set_title("biomass & product (mass)"); ax[0].set_xlabel("t [h]"); ax[0].legend()
    ax[1].plot(tX, S1, color="tab:green"); ax[1].axhline(float(S0), ls="--", color="k", lw=0.8, label="S*")
    ax[1].set_title("substrate conc. S = SV/V"); ax[1].set_xlabel("t [h]"); ax[1].legend()
    ax[2].plot(tX, X_opt[:, 3], color="tab:red"); ax[2].axhline(V_max, ls="--", color="k", lw=0.8, label="V_max")
    ax[2].set_title("volume V"); ax[2].set_xlabel("t [h]"); ax[2].legend()
    ax[3].step(tU, U_opt, where="post", color="tab:purple"); ax[3].set_title("feed F (control)")
    ax[3].set_xlabel("t [h]"); ax[3].set_ylim(F_min - 0.1, F_max + 0.1)
    fig1.suptitle("STAGE 1 — direct forward-Euler collocation (IPOPT)")
    fig1.tight_layout(); fig1.savefig(os.path.join(here, "hybrid_fwd_stage1.png"), dpi=130)

    # ---------- STAGE 2: assemble the forward hybrid trajectory ----------
    ts_a = np.asarray(sol_s.ts); y_a = np.asarray(sol_s.ys)          # arc: 200 x 8 (state + costate)
    ts_b = np.asarray(sol_b.ts); x_b = np.asarray(sol_b.ys)          # batch: 100 x 4 (state only)

    def _unpack(Y):                                                  # log-SV state block -> physical
        return Y[:, 0], np.exp(Y[:, 1]), Y[:, 2], Y[:, 3], np.exp(Y[:, 1]) / Y[:, 3]
    XVa, SVa, PVa, Va, Sa = _unpack(y_a[:, :4])
    XVb, SVb, PVb, Vb, Sb = _unpack(x_b)

    F_a = np.asarray(jax.vmap(lambda y: get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max)))(jnp.asarray(y_a)))
    F_b = np.zeros_like(ts_b)                                        # batch: F = 0

    lam_a = y_a[:, 4:]                                               # arc costates (physical convention)
    def _batch_costate(x_i, t_i):
        return l_to_orig(x_i, jax.grad(lambda x: objective_batch(x, t_i, (t_end, S_F)))(x_i))
    lam_b = np.array(jax.vmap(_batch_costate)(jnp.asarray(x_b), jnp.asarray(ts_b)))   # writable copy
    mu = lam_a[-1, 3] - lam_b[0, 3]
    lam_b[:, 3] = lam_b[:, 3] + mu

    phi_a = lam_a[:, 1] * S_F + lam_a[:, 3]                          # switching function on the arc (~0)
    phi_b = lam_b[:, 1] * S_F + lam_b[:, 3]                          # < 0 in the batch  => F = F_min = 0

    # concatenated full-trajectory arrays for the state/control/phi panels
    tt = np.concatenate([ts_a, ts_b])
    XV = np.concatenate([XVa, XVb]); PV = np.concatenate([PVa, PVb])
    S = np.concatenate([Sa, Sb]);   V = np.concatenate([Va, Vb])
    F = np.concatenate([F_a, F_b]); phi = np.concatenate([phi_a, phi_b])

    def _vsw(a):
        a.axvline(t_switch_opt, ls=":", color="gray", lw=1.0)

    fig2, ax = plt.subplots(2, 3, figsize=(15, 8)); ax = ax.ravel()
    ax[0].plot(tt, XV, label="XV"); ax[0].plot(tt, PV, label="PV")
    ax[0].set_title("biomass & product"); ax[0].set_xlabel("t [h]"); ax[0].legend(); _vsw(ax[0])
    ax[1].plot(tt, S, color="tab:green"); ax[1].axhline(float(S0), ls="--", color="k", lw=0.8, label="S*")
    ax[1].set_title("substrate conc. S"); ax[1].set_xlabel("t [h]"); ax[1].legend(); _vsw(ax[1])
    ax[2].plot(tt, V, color="tab:red"); ax[2].axhline(V_max, ls="--", color="k", lw=0.8, label="V_max")
    ax[2].set_title("volume V"); ax[2].set_xlabel("t [h]"); ax[2].legend(); _vsw(ax[2])
    ax[3].plot(tt, F, color="tab:purple"); ax[3].set_title("feed F (singular arc then batch)")
    ax[3].set_xlabel("t [h]"); ax[3].set_ylim(F_min - 0.1, F_max + 0.1); _vsw(ax[3])
    ax[4].plot(tt, phi, color="m")                                  # one colour over the whole interval
    ax[4].axhline(0.0, ls="--", color="k", lw=0.8)
    ax[4].set_title(r"switching function $\varphi=\lambda_{SV}S_F+\lambda_V$")
    ax[4].set_xlabel("t [h]"); _vsw(ax[4])
    labels = [r"$\lambda_{XV}$", r"$\lambda_{SV}$", r"$\lambda_{PV}$", r"$\lambda_V$"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    lam_full = np.concatenate([lam_a, lam_b], axis=0)               # continuous arc + batch (single line)
    ax5r = ax[5].twinx()                                            # lambda_V on its own right axis (kept black)
    for k in range(3):                                             # lambda_XV, lambda_SV, lambda_PV -> left axis
        ax[5].plot(tt, lam_full[:, k], color=colors[k], label=labels[k])
    ax5r.plot(tt, lam_full[:, 3], color=colors[3], label=labels[3])  # lambda_V -> right axis
    ax5r.set_ylabel(r"$\lambda_V$ (right)")
    ax[5].set_title("costates"); ax[5].set_xlabel("t [h]")
    ln1, lb1 = ax[5].get_legend_handles_labels(); ln2, lb2 = ax5r.get_legend_handles_labels()
    ax[5].legend(ln1 + ln2, lb1 + lb2, ncol=2, loc="best"); _vsw(ax[5])
    fig2.suptitle("STAGE 2 — forward hybrid (singular arc + batch), t_switch = %.4f" % t_switch_opt)
    fig2.tight_layout(); fig2.savefig(os.path.join(here, "hybrid_fwd_stage2.png"), dpi=130)
    print("saved: hybrid_fwd_stage1.png, hybrid_fwd_stage2.png")
