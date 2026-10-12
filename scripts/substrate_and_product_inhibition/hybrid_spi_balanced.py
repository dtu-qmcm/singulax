import jax
import diffrax
import os
from functools import partial
import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from cycler import cycler
from cyipopt import minimize_ipopt
from singulax.substrate_product_inhibition.shooting import (
    solve_and_get_residuals_ms, switch_times_from,
)
from singulax.substrate_product_inhibition.odes import states_ode, vector_field, vf_states
from singulax.substrate_product_inhibition.Fsin import get_Fsin_unconstrained
from singulax.collocation import CollocParams, unpack, objective, constraints, jac_defects, detect_switches
from singulax.substrate_product_inhibition.kinetics import BALANCED

jax.config.update("jax_enable_x64", True)

# ---- PROBLEM DATA ----
PARAMS = BALANCED

S_F, V_max, F_max, F_min, t_end = 150.0, 10.0, 2.0, 0.0, 39.5
x0 = jnp.array([0.1, 0.001, 0.0, 1.0])            # [XV, SV, PV, V]

N = 400; tf = t_end; h = tf / N; nx = 4; nX = (N + 1) * nx; nz = nX + N

p = CollocParams(N=N, nx=nx, h=h, nX=nX, nz=nz, S_F=S_F,
                 f=partial(states_ode, params=PARAMS))
obj_jit = jax.jit(lambda z: objective(z, p))
grad_jit = jax.jit(jax.grad(lambda z: objective(z, p)))
con_jit = jax.jit(lambda z: constraints(z, p))

if __name__ == "__main__":
    # ===== STAGE 1: collocation =====
    print("=== STAGE 1: collocation (maximise PV) ===")
    F_init = 0.5
    def _sim(x, _):
        """Forward-Euler rollout at constant feed F_init; produces the Stage-1 NLP seed z0."""
        return x + h * states_ode(x, F_init, S_F, PARAMS), x
    _, X_hist = jax.lax.scan(_sim, x0, None, length=N)
    z0 = np.concatenate([np.asarray(jnp.concatenate([x0[None, :], X_hist])).ravel(), np.full(N, F_init)])

    lb = np.full(nz, -np.inf); ub = np.full(nz, np.inf)
    lb[:nx] = ub[:nx] = np.asarray(x0)
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
    print("l0 seed:", mult_g[0, :])

    # ----- switching times from Stage 1 -----
    t_grid = np.linspace(0, tf, N + 1); t_ctrl = t_grid[:-1]
    t_sw, seq = detect_switches(U_opt, t_ctrl, F_min, F_max)      # interior switches from the feed
    S = X_opt[:, 1] / X_opt[:, 3]; P = X_opt[:, 2] / X_opt[:, 3]
    print("feed-regime switches =", np.round(t_sw, 3))
    print("regime sequence      =", seq, "  (2=Fmax bang, 1=singular, 0=Fmin/batch)")

    # ---- Stage 1 plot ----
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,

    "legend.frameon": False,

    "figure.dpi": 300,
    "savefig.dpi": 600,

    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    })
    
    oi = {
    "orange":       "#E69F00",  
    "sky_blue":     "#56B4E9",  
    "bluish_green": "#009E73",  
    "vermillion":   "#D55E00",
    "black":        "#000000",
    "reddish_purple": "#CC79A7"   
    }
    here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots(
    3, 1,
    figsize=(6.5, 7),
    sharex=True,
    constrained_layout=True
    )
    ax = ax.ravel()
    ax[0].plot(t_grid, X_opt[:,0], label=r"$XV$", color=oi["sky_blue"])
    ax[0].plot(t_grid, X_opt[:,1], label=r"$SV$", color=oi["orange"])
    ax[0].plot(t_grid, X_opt[:,2]/5, label=r"$PV/5$", color=oi["bluish_green"])
    ax[0].plot(t_grid, X_opt[:,3], label=r"$V$", color=oi["vermillion"])
    ax[0].set_ylabel("Mass (g), Volume (L)")
    ax[0].legend(ncol=4, loc="best")

    ax[1].plot(t_grid, S, label=r"$S$", color=oi["orange"], ls="--", lw=1.8)
    ax[1].plot(t_grid, P, label=r"$P$", color=oi["bluish_green"], ls="--", lw=1.8)
    ax[1].set_ylabel("Concentration (g L$^{-1}$)")
    ax[1].legend(ncol=2, loc="best")

    ax[2].step(t_ctrl,U_opt,where="post",color=oi["black"])
    ax[2].set_ylabel(r"$F$ (L h$^{-1}$)")
    ax[2].set_xlabel("Time (h)")
    ax[2].set_ylim(F_min - 0.1, F_max + 0.1)

    fig.savefig(os.path.join(here, "spi_stage1_balanced.png"), dpi=600, bbox_inches="tight")
    print("saved spi_stage1_balanced.png")

    # ===== STAGE 2: shooting =====
    print("=== STAGE 2: multiple shooting ===")
    def _integ(y0, t0, t1, u):
        """Integrate one arc [t0, t1] to build a multiple-shooting node seed.
        u is the bang feed (F_max / F_min), or None for the singular arc (F_sing feedback)."""
        vf_singular = lambda t, y, a: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max, PARAMS)), PARAMS))
        vf = vf_singular if u is None else vector_field
        s = diffrax.diffeqsolve(diffrax.ODETerm(vf),
                                diffrax.Tsit5(),
                                t0=t0, t1=t1, y0=y0, dt0=0.01,
                                args=(S_F, u, PARAMS),
                                stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
                                saveat=diffrax.SaveAt(t1=True),
                                max_steps=100000,
                                throw=False)
        return s.ys[-1]

    seq3 = seq[:3]
    arc_u = [F_max if s == 2 else F_min if s == 0 else None for s in seq3]
    num_segments = len(arc_u)                                        # 3
    sw = jnp.asarray(t_sw[:num_segments])                            # [t1, t2, t3]
    shooting_nodes_guess = jnp.concatenate([jnp.array([0.0]), sw])   # [0, t1, t2, t3]

    y = jnp.concatenate([x0, lam_t0_seed])
    node_list = [y]
    for i in range(num_segments):                                    # build node seeds by forward integration
        y = _integ(y, shooting_nodes_guess[i], shooting_nodes_guess[i + 1], arc_u[i])
        node_list.append(y)
    guess0 = jnp.concatenate([jnp.stack(node_list).ravel(), sw])

    ms_args = (num_segments, S_F, V_max, F_min, F_max, tf, x0, arc_u, PARAMS)
    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
    sol = optx.least_squares(lambda u, a: solve_and_get_residuals_ms(u, a),
                             solver,
                             guess0,
                             args=ms_args,
                             max_steps=10000,
                             throw=False)
    Zopt = sol.value
    res_final = solve_and_get_residuals_ms(Zopt, ms_args)
    n_nodes = num_segments + 1
    print("Stage 2 ||res|| =", float(jnp.linalg.norm(res_final)))
    print("Stage 2 switch times =", np.round(np.asarray(Zopt[n_nodes * 8:]), 3))
    ns8 = num_segments * 8
    print("  ||defects|| =", round(float(jnp.linalg.norm(res_final[:ns8])), 4),
          " ||bc_start|| =", round(float(jnp.linalg.norm(res_final[ns8:ns8 + 4])), 4),
          " ||terminal|| =", round(float(jnp.linalg.norm(res_final[ns8 + 4:ns8 + 7])), 4),
          " V-Vmax =", round(float(res_final[ns8 + 7]), 4))
    print("  phi / phidot =", np.round(np.asarray(res_final[ns8 + 8:]), 5))
    
    # ---- Stage 2 plot ----
    nodes_opt = np.asarray(Zopt[:n_nodes * 8]).reshape(n_nodes, 8)
    # The same map the residual used -- not a second hand-written copy of it.
    sw_eff = np.asarray(switch_times_from(Zopt[n_nodes * 8:], tf))
    nodes_t = np.concatenate([[0.0], sw_eff])
    sc2 = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-12)
    vf_sing = lambda t, y, a: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max, PARAMS)), PARAMS))
    seg_t, seg_y, seg_F = [], [], []
    for i in range(num_segments):
        u = arc_u[i]
        vf = vf_sing if u is None else vector_field
        tsi = jnp.linspace(nodes_t[i], max(nodes_t[i + 1], nodes_t[i] + 1e-9), 60)
        s = diffrax.diffeqsolve(diffrax.ODETerm(vf), diffrax.Tsit5(), t0=nodes_t[i],
            t1=max(nodes_t[i + 1], nodes_t[i] + 1e-9), y0=jnp.asarray(nodes_opt[i]), dt0=0.01,
            args=(S_F, u, PARAMS), saveat=diffrax.SaveAt(ts=tsi), stepsize_controller=sc2, max_steps=100000, throw=False)
        yi = np.asarray(s.ys); seg_t.append(np.asarray(s.ts)); seg_y.append(yi)
        seg_F.append(np.asarray(jax.vmap(lambda y: get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max, PARAMS)))(jnp.asarray(yi)))
                     if u is None else np.full(len(yi), u))
    # ---- terminal batch: states FORWARD, costate BACKWARD ----
    tb = np.asarray(jnp.linspace(nodes_t[-1], tf, 100))
    f_batch = lambda t, x, a: vf_states(t, x, (S_F, 0.0, PARAMS))
    fwd = diffrax.diffeqsolve(diffrax.ODETerm(f_batch), diffrax.Tsit5(),
        t0=nodes_t[-1], t1=tf, y0=jnp.asarray(nodes_opt[-1][:4]), dt0=0.01,
        saveat=diffrax.SaveAt(ts=jnp.asarray(tb), dense=True),
        stepsize_controller=sc2, max_steps=100000, throw=False)

    def lam_rhs(t, lam, a):
        """lambda_dot = -(df/dx)^T lambda, with x read off the saved forward pass."""
        _, vjp = jax.vjp(lambda xx: f_batch(t, xx, None), fwd.evaluate(t))
        return -vjp(lam)[0]

    # lambda(tf) = [0, 0, 1, eta]: the transversality this batch is anchored on.
    bwd = diffrax.diffeqsolve(diffrax.ODETerm(lam_rhs), diffrax.Tsit5(),
        t0=tf, t1=nodes_t[-1], y0=jnp.array([0.0, 0.0, 1.0, 0.0]),
        dt0=-0.01, saveat=diffrax.SaveAt(ts=jnp.asarray(tb[::-1])),
        stepsize_controller=sc2, max_steps=100000, throw=False)
    lam_b = np.array(bwd.ys)[::-1].copy()
    lam_b[:, 3] += nodes_opt[-1][7] - lam_b[0, 3]       # lambda_V continuous at t_M
    yb = np.concatenate([np.asarray(fwd.ys), lam_b], axis=1)

    print("Final amount of product PV(tf) =", round(float(yb[-1, 2]), 2), "g")
    print("batch lambda(tf) =", np.round(yb[-1, 4:], 4), "  (expect [0, 0, 1, eta], eta free)")
    print("lambda(t_M)  backward =", np.round(yb[0, 4:7], 6),
          "  shooting =", np.round(nodes_opt[-1][4:7], 6),
          "  gap =", f"{np.max(np.abs(yb[0, 4:7] - nodes_opt[-1][4:7])):.2e}")

    # single continuous trajectory: bang / bang / singular segments + batch
    T = np.concatenate(seg_t + [tb])
    Yf = np.concatenate(seg_y + [yb], axis=0)
    Ff = np.concatenate(seg_F + [np.zeros(len(tb))])
    S_all = Yf[:, 1] / Yf[:, 3]; P_all = Yf[:, 2] / Yf[:, 3]
    phi_all = Yf[:, 5] * S_F + Yf[:, 7]

    fig, axd = plt.subplot_mosaic(
        [["state", "SP"],
         ["phi",   "F"],
         ["adj",   "adj"]],
        figsize=(7,8),
        constrained_layout=True
    )
    axd["state"].plot(T, Yf[:, 0], label=r"$XV$", color=oi["sky_blue"])
    axd["state"].plot(T, Yf[:, 1], label=r"$SV$",color=oi["orange"] )
    axd["state"].plot(T, Yf[:, 2] / 5, label=r"$PV / 5$", color=oi["bluish_green"])
    axd["state"].plot(T, Yf[:, 3], label=r"$V$", color=oi["vermillion"])
    axd["state"].set_ylabel("Mass (g), Volume (L)")
    axd["state"].legend(ncol=4, loc="best")

    axd["SP"].plot(T, S_all, label="S", color=oi["orange"], ls="--", lw=1.8)
    axd["SP"].plot(T, P_all, label="P", color=oi["bluish_green"], ls="--", lw=1.8)
    axd["SP"].set_ylabel(r"Concentration (g L$^{-1}$)")
    axd["SP"].legend(ncol=2, loc="best")

    axd["phi"].plot(T, phi_all, color=oi["reddish_purple"])
    axd["phi"].axhline(0,color=oi["black"],ls="--", lw=0.9)
    arcs = T <= sw_eff[-1]
    lim = 1.15 * np.nanmax(np.abs(phi_all[arcs]))
    axd["phi"].set_ylim(-lim, lim)
    for tsw in sw_eff:
        axd["phi"].axvline(tsw, color="0.6", ls=":", lw=0.8)
    axd["phi"].set_ylabel("$\\phi$")
    axd["phi"].set_xlabel("Time (h)")
    
    axd["F"].step(T,Ff,where="post",color=oi["black"])
    axd["F"].set_ylim(F_min - 0.1, F_max + 0.1)
    axd["F"].set_ylabel("F (L h$^{-1}$)")
    axd["F"].set_xlabel("Time (h)")

    axd["adj"].plot(T, Yf[:, 4], label=r"$\lambda_1$ ($XV$)", color=oi["sky_blue"], ls="-")
    axd["adj"].plot(T, Yf[:, 5], label=r"$\lambda_2$ ($SV$)", color=oi["orange"], ls="--")
    axd["adj"].plot(T, Yf[:, 6], label=r"$\lambda_3$ ($PV$)", color=oi["bluish_green"], ls="-.")
    axd["adj"].plot(T, Yf[:, 7], label=r"$\lambda_4$ ($V$)", color=oi["vermillion"], ls=":")

    axd["adj"].set_xlabel("Time (h)")
    axd["adj"].set_ylabel(r"$\lambda$")
    axd["adj"].legend(ncol=4, frameon=False, loc="upper right")

    # dashed vertical lines at the switching times, in every subplot
    for a in axd.values():
        for tsw in sw_eff:
            a.axvline(tsw,color="0.5",ls=":",lw=0.8)
    fig.savefig(os.path.join(here, "spi_stage2_balanced.png"), dpi=600, bbox_inches="tight")
    print("saved spi_stage2_balanced.png")

