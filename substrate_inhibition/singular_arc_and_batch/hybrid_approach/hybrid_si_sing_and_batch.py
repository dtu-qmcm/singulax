import os
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx
from cyipopt import minimize_ipopt

from fedopt.odes import f_vec_orig, g_vec_orig, vector_field, get_phi
from fedopt.Fsin import get_Fsin_unconstrained, get_Fsin
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
    x_opt_tf_log = jnp.array([x_opt_tf[0], jnp.log(jnp.clip(x_opt_tf[1], 1e-12, None)), x_opt_tf[2], x_opt_tf[3]])
    mult_g = np.asarray(result.info["mult_g"]).reshape(N, nx)

    lam4_tf_seed = float(mult_g[-1, 3])
    lam_t0_seed = mult_g[0, :]
    print("PV(tf)               =", x_opt_tf[ 2])
    print("x(tf) [XV,SV,PV,V]   =", x_opt_tf)
    print("lambda4(tf) seed     =", round(lam4_tf_seed, 4))
    print("lambda4(t0) seed     =", lam_t0_seed)

    t_grid = np.linspace(0, tf, N + 1)
    hit = np.where(X_opt[:, 3] >= V_max - 1e-5)[0]
    t_switch_euler = float(t_grid[hit[0]]) if len(hit) else tf
    print(f"Switching time seed = {t_switch_euler:.4f}")

    # ===== STAGE 2: shoot (t_switch, lambda4) so phi = phidot = 0 at the singular junction =====
    print("\n=== STAGE 2: shoot (t_switch, lambda4); phi = phidot = 0 at the junction ===")
    def solve_and_get_residuals_ms(guess, args):
        l0 = guess[:4]
        t_switch = guess[4]
        x0, t_end = args
        y0 = jnp.concatenate([x0, l0])
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10)
        sol_to_sw = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: vector_field(t, y, (S_F, get_Fsin(y, (S_F, V_max, F_min, F_max))))),
            solver=diffrax.Kvaerno5(),
            t0=0.0, t1=t_switch, dt0=0.01, y0=y0,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=stepsize_controller,
            max_steps=100000, throw=False,
            adjoint=diffrax.ForwardMode()
        )
        y_sw = sol_to_sw.ys[-1]
        y_sw = jnp.where(jnp.any(~jnp.isfinite(y_sw)), jnp.full(8, 1e6), y_sw)
        ydot = vector_field(t_switch, y_sw, (S_F, 0.0))     
        phi = get_phi(y_sw)                                 
        phidot = S_F * ydot[5] + ydot[7]                    # d(phi)/dt = S_F*d(l2)/dt + d(l4)/dt
        sol_to_tf = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: vector_field(t, y, (S_F, 0.0))),
            solver=diffrax.Kvaerno5(),
            t0=t_switch, t1=t_end, dt0=0.01, y0=y_sw,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=stepsize_controller,
            max_steps=100000,
            throw=False,
            adjoint=diffrax.ForwardMode()
        )
        y_tf = sol_to_tf.ys[-1]
        y_tf = jnp.where(jnp.any(~jnp.isfinite(y_tf)), jnp.full(8, 1e6), y_tf)

        bc = jnp.array([phi,
                        phidot,
                        y_tf[3]-V_max,
                        y_tf[4]-0,
                        y_tf[5]-0,
                        y_tf[6]-1
                       ])
        return bc
    
    guess = jnp.concatenate([jnp.asarray(lam_t0_seed), jnp.array([t_switch_euler])])
    solver = optx.LevenbergMarquardt(rtol=1e-10, atol=1e-12)
    solution = optx.least_squares(solve_and_get_residuals_ms, solver, guess,
                                  args=(x0_log, t_end), max_steps=500, throw=False)
    l0_opt = solution.value[:4]
    t_switch_opt = float(solution.value[4])
    
    print("optimal lambda(t0) =", np.round(np.asarray(l0_opt), 4))
    print("optimal t_switch   =", round(t_switch_opt, 4))
    res_final = solve_and_get_residuals_ms(solution.value, (x0_log, t_end))
    print("residuals          =", np.array(res_final))
