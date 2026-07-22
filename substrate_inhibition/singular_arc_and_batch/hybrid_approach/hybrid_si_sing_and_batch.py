import os
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx
from cyipopt import minimize_ipopt

from fedopt.odes import f_vec_orig, g_vec_orig, vector_field, get_phi
from fedopt.Fsin import get_Fsin_unconstrained
from fedopt.si_analytical import states_analytical_sol, volume_cond
from fedopt.shooting import solve_and_get_residuals_batch, loss_fn_batch
from fedopt.hybrid import CollocParams, unpack, objective, constraints, jac_defects

jax.config.update("jax_enable_x64", True)

# ---- PROBLEM DATA ----
Ks = 0.1
Ki = 100
S0 = jnp.sqrt(Ks * Ki)
S_F = 150.0
V_max = 4.0
F_max = 2.0
F_min = 0.0
t_end = 9.0

x0_phys = jnp.array([10.0, float(S0), 0.0, 1.0])           # [XV, SV, PV, V]  (Stage 1)
x0_log = jnp.array([10.0, jnp.log(S0), 0.0, 1.0])          # [XV, log SV, PV, V]  (switch sim + Stage 2)

# collocation grid
N = 400
tf = t_end
h = tf / N
nx = 4
nX = (N + 1) * nx
nz = nX + N


def states_ode_phys(x, F, S_F):
    return f_vec_orig(x) + g_vec_orig(x, S_F) * F

p = CollocParams(N=N, nx=nx, h=h, nX=nX, nz=nz, S_F=S_F, f=states_ode_phys)

obj_jit = jax.jit(lambda z: objective(z, p))
grad_jit = jax.jit(jax.grad(lambda z: objective(z, p)))
con_jit = jax.jit(lambda z: constraints(z, p))


# reconstruction fields (log coords); clip the singular feedback to the pump limits so a
# small off-optimal excursion can't blow the plot up (on the true arc F in [0,1], never binds)
def _batch_vf(t, y, args):
    return vector_field(t, y, (S_F, 0.0))

def _singular_vf(t, y, args):
    F = jnp.clip(get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max)), F_min, F_max)
    return vector_field(t, y, (S_F, F))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

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
    x_opt_tf = X_opt[-1]
    mult_g = np.asarray(result.info["mult_g"]).reshape(N, nx)

    lam4_tf_seed = float(mult_g[-1, 3])
    print("PV(tf)               =", x_opt_tf[ 2])
    print("x(tf) [XV,SV,PV,V]   =", x_opt_tf)
    print("lambda4(tf) seed     =", round(lam4_tf_seed, 4))

    # Calculate switching time
    t_grid = np.linspace(0, tf, N + 1)
    tol = 1e-5
    indices_vmax = np.where(X_opt[:, 3] >= (V_max - tol))[0]

    idx_switch = indices_vmax[0]
    t_switch = t_grid[idx_switch]
    print(f"Switching time (t_switch) = {t_switch:.4f}")

    x_tf_log = jnp.array([x_opt_tf[0], jnp.log(x_opt_tf[1]), x_opt_tf[2], x_opt_tf[3]])
    
    # ================= STAGE 2 =================
    print("\n=== STAGE 2: backward shoot lambda4(tf) so phi(t_switch)=0 ===")
    stage2_args = (tf, x0_log, S_F, x_tf_log, t_switch, V_max, F_min, F_max)
    solver = optx.LevenbergMarquardt(rtol=1e-10, atol=1e-12)
    solution = optx.least_squares(
        fn = solve_and_get_residuals_batch,
        solver = solver,
        y0 = lam4_tf_seed,
        args = stage2_args,
        max_steps = 2000,
        throw = False
    )
    lam4_opt = float(solution.value)
    res = solve_and_get_residuals_batch(lam4_opt, stage2_args)
    print("optimal lambda4(tf) =", round(lam4_opt, 5))
    print("phi(t_switch) resid =", float(res[0]))

    # dense reconstruction (backward, reversed to increasing time) + x(0) check
    ytf = jnp.concatenate([x_tf_log, jnp.array([0.0, 0.0, 1.0, lam4_opt])])
    sc = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10)
    sol_b = diffrax.diffeqsolve(diffrax.ODETerm(_batch_vf), diffrax.Kvaerno5(),
        t0=tf, t1=t_switch, dt0=-0.01, y0=ytf,
        saveat=diffrax.SaveAt(ts=jnp.linspace(tf, t_switch, 200)),
        stepsize_controller=sc, max_steps=100000, throw=False)
    sol_s = diffrax.diffeqsolve(diffrax.ODETerm(_singular_vf), diffrax.Kvaerno5(),
        t0=t_switch, t1=0.0, dt0=-0.01, y0=sol_b.ys[-1],
        saveat=diffrax.SaveAt(ts=jnp.linspace(t_switch, 0.0, 300)),
        stepsize_controller=sc, max_steps=100000, throw=False)
    t_plot = np.concatenate([np.asarray(sol_s.ts)[::-1], np.asarray(sol_b.ts)[::-1]])
    y_plot = np.concatenate([np.asarray(sol_s.ys)[::-1], np.asarray(sol_b.ys)[::-1]])
    F_plot = np.where(t_plot < t_switch,
                      np.asarray(jax.vmap(lambda y: jnp.clip(get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max)), F_min, F_max))(jnp.asarray(y_plot))),
                      0.0)
    phi_plot = np.asarray(jax.vmap(get_phi)(jnp.asarray(y_plot)))

    print("\nRecovered x(0) = [%.3f, SV=%.3f, %.3f, %.3f]  (target [10, %.3f, 0, 1])"
          % (y_plot[0, 0], np.exp(y_plot[0, 1]), y_plot[0, 2], y_plot[0, 3], float(S0)))
    print("Final    x(tf) = [%.3f, SV=%.4g, %.3f, %.3f]"
          % (y_plot[-1, 0], np.exp(y_plot[-1, 1]), y_plot[-1, 2], y_plot[-1, 3]))

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- STAGE 1 plot ----
    t_state = np.linspace(0, tf, N + 1); t_ctrl = np.linspace(0, tf, N, endpoint=False) + 0.5 * h
    fig1, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    a1.plot(t_state, X_opt[:, 0], "k-", label="XV"); a1.plot(t_state, X_opt[:, 1], "r--", label="SV")
    a1.plot(t_state, X_opt[:, 2], "g:", label="PV"); a1.plot(t_state, X_opt[:, 3], "b-.", label="V")
    a1.axvline(t_switch, color="m", ls=":", lw=1); a1.set_title("Stage 1: states (physical)"); a1.legend(); a1.set_xlabel("t")
    a2.plot(t_ctrl, U_opt, "r"); a2.axhline(F_max, color="k", ls="--", lw=.8); a2.axhline(F_min, color="k", ls="--", lw=.8)
    a2.axvline(t_switch, color="m", ls=":", lw=1, label="t_switch (S=S* sim)")
    a2.set_title("Stage 1: feed (chatters on singular arc)"); a2.set_xlabel("t"); a2.set_ylabel("F"); a2.legend()
    fig1.tight_layout(); fig1.savefig(os.path.join(script_dir, "hybrid_sb_stage1.png"), dpi=110)
    print("saved -> hybrid_sb_stage1.png")

    # ---- STAGE 2 plot ----
    fig2, ((b1, b2), (b3, b4)) = plt.subplots(2, 2, figsize=(14, 8))
    b1.plot(t_plot, y_plot[:, 0], "k-", label="XV"); b1.plot(t_plot, np.exp(y_plot[:, 1]), "r--", label="SV")
    b1.plot(t_plot, y_plot[:, 2], "g:", label="PV"); b1.plot(t_plot, y_plot[:, 3], "b-.", label="V")
    b1.axvline(t_switch, color="m", ls=":", lw=1); b1.set_title("States"); b1.set_xlabel("t"); b1.legend()
    b2.plot(t_plot, np.exp(y_plot[:, 1]) / y_plot[:, 3], "r-", label="S"); b2.plot(t_plot, y_plot[:, 2] / y_plot[:, 3], "g--", label="P")
    b2.axhline(float(S0), color="k", ls=":", lw=.8, label="S*"); b2.set_title("S, P concentration"); b2.set_xlabel("t"); b2.legend()
    b3.plot(t_plot, F_plot, "b"); b3.axhline(F_max, color="k", ls="--", lw=.8); b3.axhline(F_min, color="k", ls="--", lw=.8)
    b3.axvline(t_switch, color="m", ls=":", lw=1); b3.set_title("Feed rate F"); b3.set_xlabel("t"); b3.set_ylabel("F")
    b4.plot(t_plot, phi_plot, "m"); b4.axhline(0, color="k", ls="--", lw=.8); b4.axvline(t_switch, color="m", ls=":", lw=1)
    b4.set_title("Switching function φ"); b4.set_xlabel("t")
    fig2.suptitle("Stage 2: singular arc + batch (backward λ4 shoot)", fontsize=15)
    fig2.tight_layout(); fig2.savefig(os.path.join(script_dir, "hybrid_sb_stage2.png"), dpi=110)
    print("saved -> hybrid_sb_stage2.png")
