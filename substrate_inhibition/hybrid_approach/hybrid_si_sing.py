import os
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx
from scipy.sparse import coo_array
from cyipopt import minimize_ipopt

from fedopt.odes import vector_field, get_phi, states_ode
from fedopt.Fsin import get_Fsin
from fedopt.hybrid import CollocParams, unpack, objective, constraints, jac_defects
from fedopt.shooting import solve_and_get_residuals
  
jax.config.update("jax_enable_x64", True)

# PROBLEM DATA
Ks = 0.1
Ki = 100
S0 = jnp.sqrt(Ks * Ki)          # S* = sqrt(Ks*Ki), the growth-optimal substrate level
S_F = 150.0
V_max = 5.0
F_max = 2.0
F_min = 0.0
t_end = 9.0

x0 = jnp.array([10.0, jnp.log(S0), 0.0, 1.0])   
l_tf = jnp.array([0.0, 0.0, 1.0, 0.0])          

# collocation grid
N = 400
tf = t_end
h = tf / N
nx = 4
nX = (N + 1) * nx               # states
nU = N                          # controls
nz = nX + nU

# define  grid/problem parameters and pass them into the hybrid functions
p = CollocParams(N=N, nx=nx, h=h, nX=nX, nz=nz, S_F=S_F)

obj_jit = jax.jit(lambda z: objective(z, p))
grad_jit = jax.jit(jax.grad(lambda z: objective(z, p)))
con_jit = jax.jit(lambda z: constraints(z, p))


if __name__ == "__main__":
    # ---- STAGE 1: DIRECT APPROACH ----
    print("=== STAGE 1: Simultaneous NLP  ===")

    # warm start: constant feed, forward-simulate the states from x0
    F_init = 0.5
    def _sim(x, _):
        return x + h * states_ode(x, F_init, S_F), x
    _, X_hist = jax.lax.scan(_sim, x0, None, length=N)
    X_init = jnp.concatenate([x0[None, :], X_hist])
    z0 = np.concatenate([np.asarray(X_init).ravel(), np.full(N, F_init)])

    # bounds: fix x_0, F in [F_min, F_max], and V <= V_max on every node (essential)
    lb = np.full(nz, -np.inf)
    ub = np.full(nz, np.inf)
    lb[:nx] = ub[:nx] = np.asarray(x0)
    lb[nX:] = F_min
    ub[nX:] = F_max
    V_idx = 4 * np.arange(N + 1) + 3
    ub[V_idx] = V_max

    cons = [{
        "type": "eq",
        "fun": lambda z: np.asarray(con_jit(jnp.asarray(z))),
        "jac": lambda z: jac_defects(z, p),
    }]

    result = minimize_ipopt(
        fun=lambda z: float(obj_jit(jnp.asarray(z))),
        jac=lambda z: np.asarray(grad_jit(jnp.asarray(z))),
        x0=z0,
        bounds=list(zip(lb, ub)),
        constraints=cons,
        options={"max_iter": 2000, "tol": 1e-6, "print_level": 5},
    )

    z_opt = result.x
    X_opt, U_opt = unpack(jnp.asarray(z_opt), p)
    X_opt = np.asarray(X_opt)
    U_opt = np.asarray(U_opt)

    # multipliers -> costates. MAXIMISE-PV convention => lambda = +mult_g
    mult_g = np.asarray(result.info["mult_g"]).reshape(N, nx)
    lam = mult_g
    lam0 = jnp.asarray(lam[0])

    print("PV(tf)      =", X_opt[-1, 2])
    print("x(tf)       =", X_opt[-1])
    print("lambda(0)   =", np.array(lam0))
    print("lambda(tf)  =", lam[-1], " (should be ~[0,0,1,0])")

    # ---- STAGE 2: INDIRECT APPROACH ----
    print("\n=== STAGE 2: Backward single shooting ===")
    stage2_args = (tf, x0, S_F, V_max, F_min, F_max )
    xtf_guess = jnp.asarray(X_opt[-1])
    solver = optx.LevenbergMarquardt(rtol=1e-10, atol=1e-12)
    solution = optx.least_squares(
        fn=solve_and_get_residuals,
        solver=solver,
        y0=xtf_guess,
        args=stage2_args,
        max_steps=2000,
        throw=False,
    )
    xtf_opt = solution.value
    res_opt = solve_and_get_residuals(xtf_opt, stage2_args)
    print("optimal x(tf) =", np.array(xtf_opt))
    print("residual norm =", float(jnp.linalg.norm(res_opt)))

    # re-simulate the converged solution on a dense grid for plotting: integrate
    # backward from [x(tf), lambda(tf)] and reverse to increasing time.
    ts = jnp.linspace(tf, 0.0, 500)
    ytf_opt = jnp.concatenate([xtf_opt, l_tf])
    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(lambda t, y, args: vector_field(t, y, (S_F, get_Fsin(y, (S_F, V_max, F_min, F_max))))),
        solver=diffrax.Kvaerno5(),
        t0=tf, t1=0.0, dt0=-0.01,
        y0=ytf_opt,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
        saveat=diffrax.SaveAt(ts=ts)
        )
    t_plot = np.asarray(sol.ts)[::-1]
    y_plot = np.asarray(sol.ys)[::-1]
    F_plot = np.asarray(jax.vmap(lambda y: get_Fsin(y, (S_F, V_max, F_min, F_max)))(jnp.asarray(y_plot)))
    phi_plot = np.asarray(jax.vmap(get_phi)(jnp.asarray(y_plot)))

    print("\nFinal XV =", y_plot[-1, 0])
    print("Final SV =", np.exp(y_plot[-1, 1]))
    print("Final PV =", y_plot[-1, 2])
    print("Final V  =", y_plot[-1, 3])

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- STAGE 1 PLOT ----
    t_state = np.linspace(0, tf, N + 1)
    t_ctrl = np.linspace(0, tf, N, endpoint=False) + 0.5 * h
    t_lam = t_state[1:]                                  

    fig1, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 4))
    a1.plot(t_state, X_opt[:, 0], "k-", label="XV")
    a1.plot(t_state, np.exp(X_opt[:, 1]), "r--", label="SV")
    a1.plot(t_state, X_opt[:, 2], "g:", label="PV")
    a1.plot(t_state, X_opt[:, 3], "b--", label="V")
    a1.set_xlabel("t"); a1.set_title("Stage 1: states"); a1.legend()

    a2.plot(t_ctrl, U_opt, "r")
    a2.axhline(F_max, color="k", ls="--", lw=0.8)
    a2.axhline(F_min, color="k", ls="--", lw=0.8)
    a2.set_xlabel("t"); a2.set_ylabel("F"); a2.set_title("Stage 1: feed rate")

    a3.plot(t_lam, lam[:, 0], "g-", label="λ1")
    a3.plot(t_lam, lam[:, 1], "r--", label="λ2")
    a3.plot(t_lam, lam[:, 2], "b:", label="λ3")
    a3.plot(t_lam, lam[:, 3], "k--", label="λ4")
    a3.set_xlabel("t"); a3.set_title("Stage 1: costates (= +mult_g)"); a3.legend()

    fig1.suptitle("Stage 1: Forward Euler Discretization", fontsize=16)
    plt.tight_layout()
    fig1.savefig(os.path.join(script_dir, "hybrid_si_sing_stage1.png"), dpi=110)
    print("saved -> hybrid_si_sing_stage1.png")

    # ---- STAGE 2 PLOT ----
    fig2, ((b1, b2), (b3, b4)) = plt.subplots(2, 2, figsize=(14, 8))
    b1.plot(t_plot, y_plot[:, 0], "k-", label="XV")
    b1.plot(t_plot, np.exp(y_plot[:, 1]), "r--", label="SV")
    b1.plot(t_plot, y_plot[:, 2], "g:", label="PV")
    b1.plot(t_plot, y_plot[:, 3], "b--", label="V")
    b1.set_title("States (x)"); b1.set_xlabel("t"); b1.legend()

    b2.plot(t_plot, np.exp(y_plot[:, 1]) / y_plot[:, 3], "r-", label="S")
    b2.plot(t_plot, y_plot[:, 2] / y_plot[:, 3], "g--", label="P")
    b2.axhline(float(S0), color="k", ls=":", lw=0.8, label="S*")
    b2.set_title("S and P concentration"); b2.set_xlabel("t"); b2.legend()

    b3.plot(t_plot, F_plot, "b")
    b3.axhline(F_max, color="k", ls="--", lw=0.8); b3.axhline(F_min, color="k", ls="--", lw=0.8)
    b3.set_title("Feed rate F"); b3.set_xlabel("t"); b3.set_ylabel("F")

    b4.plot(t_plot, phi_plot, "m"); b4.axhline(0, color="k", ls="--", lw=0.8)
    b4.set_title("Switching function φ"); b4.set_xlabel("t")

    fig2.suptitle("Stage 2: indirect single shooting", fontsize=16)
    plt.tight_layout()
    fig2.savefig(os.path.join(script_dir, "hybrid_si_sing_stage2.png"), dpi=110)
    print("saved -> hybrid_si_sing_stage2.png")
