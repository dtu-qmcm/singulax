"""
Hybrid two-stage approach to solve the case where the optimal control structure is Fmax-> Fsin->Fb(=0).  
"""
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx
from cyipopt import minimize_ipopt
from fedopt.substrate_inhibition.shooting import solve_and_get_residuals_sing, solve_and_get_residuals_bang_sing_batch
from fedopt.substrate_inhibition.odes import states_ode_phys, vf_states, vector_field, get_phi, l_to_orig, batch_phys_vf, objective_batch
from fedopt.substrate_inhibition.Fsin import get_Fsin_unconstrained
from fedopt.collocation import CollocParams, unpack, objective, constraints, jac_defects, detect_switches
import os
import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ---- PROBLEM DATA ----
Ks, Ki = 0.1, 100
S0 = 1e-6
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
    t_ctrl = t_grid[:-1]
    # singular -> batch switch: active state constraint V = V_max (general, robust)
    i2 = np.where(X_opt[:, 3] >= V_max - 1e-5)[0]
    t2_switch_euler = float(t_grid[i2[0]]) if len(i2) else tf
    # interior switch(es) from the feed regime alone (works for any inhibition model)
    t_sw, seq = detect_switches(U_opt, t_ctrl, F_min, F_max)
    before_batch = t_sw[t_sw < t2_switch_euler - 1e-6]       # bang -> singular = first switch before the batch
    t1_switch_euler = float(before_batch[0]) if len(before_batch) else 0.0
    print("lambda(0) seed        =", np.round(lam_t0_seed, 5))
    print("feed-regime switches  =", np.round(t_sw, 4), " regime seq =", seq)
    print("t1 (bang->sing)  seed =", round(t1_switch_euler, 4))
    print("t2 (sing->batch) seed =", round(t2_switch_euler, 4))
    print("X opt=", X_opt)

    # ===== STAGE 2: forward shoot, batch as a terminal cost =====
    print("\n=== STAGE 2: forward shoot ===")
    sc = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10)
    sing_vf = lambda t, y, a: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))))

    lam0 = jnp.asarray(lam_t0_seed)
    guess = jnp.concatenate([lam0, jnp.array([t1_switch_euler]), jnp.array([t2_switch_euler])])
    scale = jnp.maximum(jnp.abs(guess), 1e-3)
    args = (t_end, x0_log, S_F, V_max, F_min, F_max)
    sol = optx.least_squares(lambda u, a: solve_and_get_residuals_bang_sing_batch(u * scale, args),
                             optx.LevenbergMarquardt(rtol=1e-12, atol=1e-14),
                             guess / scale, args=None, max_steps=500, throw=False)
    opt_sol = sol.value * scale
    res_r = solve_and_get_residuals_bang_sing_batch(opt_sol, args)
    lam0_opt = np.asarray(opt_sol[:4])
    t1_opt = float(np.clip(opt_sol[4], 2e-3, tf - 0.2))
    t2_opt = float(np.clip(opt_sol[5], t1_opt + 0.05, tf - 0.05))
    print("  lambda(0) seed    =", np.round(np.asarray(lam0), 6))
    print("  lambda(0) refined =", np.round(lam0_opt, 6))
    print("  t1 refined        =", round(t1_opt, 6), " (seed", round(t1_switch_euler, 4), ")")
    print("  t2 refined        =", round(t2_opt, 6), " (seed", round(t2_switch_euler, 4), ")")
    print("  ||res|| =", float(jnp.linalg.norm(res_r)))
    print("  residuals =", np.round(np.asarray(res_r), 8))

    # PLOTS
    here = os.path.dirname(os.path.abspath(__file__))
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
    fig1.tight_layout(); fig1.savefig(os.path.join(here, "hybrid_bsb_stage1.png"), dpi=130)

    # ---------- STAGE 2: reconstruct bang -> singular -> batch, then plot ----------
    sc2 = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-14)
    bang_vf = lambda t, y, a: vector_field(t, y, (S_F, F_max))
    sing_vf = lambda t, y, a: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))))
    y0r = jnp.concatenate([x0_log, jnp.asarray(lam0_opt)])
    solb = diffrax.diffeqsolve(diffrax.ODETerm(bang_vf), diffrax.Tsit5(), t0=0.0, t1=t1_opt, dt0=1e-6, y0=y0r,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1_opt, 80)), stepsize_controller=sc2, max_steps=1_000_000, throw=False)
    sols = diffrax.diffeqsolve(diffrax.ODETerm(sing_vf), diffrax.Tsit5(), t0=t1_opt, t1=t2_opt, dt0=0.01, y0=solb.ys[-1],
        saveat=diffrax.SaveAt(ts=jnp.linspace(t1_opt, t2_opt, 200)), stepsize_controller=sc2, max_steps=100000, throw=False)
    solc = diffrax.diffeqsolve(diffrax.ODETerm(batch_phys_vf), diffrax.Tsit5(), t0=t2_opt, t1=tf, dt0=0.01, y0=sols.ys[-1][:4],
        args=S_F, saveat=diffrax.SaveAt(ts=jnp.linspace(t2_opt, tf, 100)), stepsize_controller=sc2, max_steps=100000, throw=False)
    tb, yb = np.asarray(solb.ts), np.asarray(solb.ys)
    ts2, ys2 = np.asarray(sols.ts), np.asarray(sols.ys)
    tc, xc = np.asarray(solc.ts), np.asarray(solc.ys)

    # batch costates via the stable reverse-mode grad, shifted for lambda_V continuity at t2
    def _bcost(x_i, t_i): return l_to_orig(x_i, jax.grad(lambda x: objective_batch(x, t_i, (tf, S_F)))(x_i))
    lam_c = np.array(jax.vmap(_bcost)(jnp.asarray(xc), jnp.asarray(tc)))
    lam_c[:, 3] += ys2[-1, 7] - lam_c[0, 3]

    def _unp(Y): return Y[:, 0], Y[:, 2], Y[:, 3], np.exp(Y[:, 1]) / Y[:, 3]      # XV, PV, V, S
    XVb, PVb, Vb, Sb = _unp(yb[:, :4]); XVs, PVs, Vs, Ss = _unp(ys2[:, :4]); XVc, PVc, Vc, Sc = _unp(xc)
    tt = np.concatenate([tb, ts2, tc])
    XV = np.concatenate([XVb, XVs, XVc]); PV = np.concatenate([PVb, PVs, PVc])
    S = np.concatenate([Sb, Ss, Sc]); V = np.concatenate([Vb, Vs, Vc])
    Fs = np.asarray(jax.vmap(lambda y: get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max)))(jnp.asarray(ys2)))
    F = np.concatenate([np.full_like(tb, F_max), Fs, np.zeros_like(tc)])
    lam_all = np.concatenate([yb[:, 4:], ys2[:, 4:], lam_c], axis=0)
    phi = lam_all[:, 1] * S_F + lam_all[:, 3]

    def _vsw(a):
        a.axvline(t1_opt, ls=":", color="gray", lw=1); a.axvline(t2_opt, ls=":", color="gray", lw=1)
    fig2, ax = plt.subplots(2, 3, figsize=(15, 8)); ax = ax.ravel()
    ax[0].plot(tt, XV, label="XV"); ax[0].plot(tt, PV, label="PV")
    ax[0].set_title("biomass & product"); ax[0].set_xlabel("t [h]"); ax[0].legend(); _vsw(ax[0])
    ax[1].plot(tt, S, color="tab:green"); ax[1].axhline(float(np.sqrt(Ks * Ki)), ls="--", color="k", lw=0.8, label="S*")
    ax[1].set_title("substrate conc. S"); ax[1].set_xlabel("t [h]"); ax[1].legend(); _vsw(ax[1])
    ax[2].plot(tt, V, color="tab:red"); ax[2].axhline(V_max, ls="--", color="k", lw=0.8, label="V_max")
    ax[2].set_title("volume V"); ax[2].set_xlabel("t [h]"); ax[2].legend(); _vsw(ax[2])
    ax[3].plot(tt, F, color="tab:purple"); ax[3].set_title("feed F (bang / singular / batch)")
    ax[3].set_xlabel("t [h]"); ax[3].set_ylim(F_min - 0.1, F_max + 0.1); _vsw(ax[3])
    ax[4].plot(tt, phi, color="m"); ax[4].axhline(0.0, ls="--", color="k", lw=0.8)
    ax[4].set_title(r"switching function $\varphi$"); ax[4].set_xlabel("t [h]"); _vsw(ax[4])
    labels = [r"$\lambda_{XV}$", r"$\lambda_{SV}$", r"$\lambda_{PV}$", r"$\lambda_V$"]
    for k in range(4):
        ax[5].plot(tt, lam_all[:, k], label=labels[k])
    ax[5].set_title("costates"); ax[5].set_xlabel("t [h]"); ax[5].legend(ncol=2); _vsw(ax[5])
    fig2.suptitle("STAGE 2 — bang→singular→batch  (t1=%.4f, t2=%.4f, ||res||=%.3g)"
                  % (t1_opt, t2_opt, float(jnp.linalg.norm(res_r))))
    fig2.tight_layout(); fig2.savefig(os.path.join(here, "hybrid_bsb_stage2.png"), dpi=130)
    print("saved: hybrid_bsb_stage1.png, hybrid_bsb_stage2.png")
