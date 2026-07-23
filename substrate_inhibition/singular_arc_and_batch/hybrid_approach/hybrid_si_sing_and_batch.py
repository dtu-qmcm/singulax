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
    print("PV(tf)               =", x_opt_tf[ 2])
    print("x(tf) [XV,SV,PV,V]   =", x_opt_tf)
    print("lambda4(tf) seed     =", round(lam4_tf_seed, 4))

    t_grid = np.linspace(0, tf, N + 1)
    hit = np.where(X_opt[:, 3] >= V_max - 1e-5)[0]
    t_switch_euler = float(t_grid[hit[0]]) if len(hit) else tf
    print(f"Switching time seed = {t_switch_euler:.4f}")

    # ===== STAGE 2: shoot (t_switch, lambda4) so phi = phidot = 0 at the singular junction =====
    print("\n=== STAGE 2: shoot (t_switch, lambda4); phi = phidot = 0 at the junction ===")
    def solve_and_get_residuals_ms(guess, args):
        t_switch, l4_tf = guess
        ytf = jnp.concatenate([x_opt_tf_log, jnp.array([0.0, 0.0, 1.0, l4_tf])])
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(vector_field),
            solver=diffrax.Kvaerno5(),
            t0=tf, t1=t_switch, dt0=-0.01, y0=ytf,
            args=(S_F, 0.0),                                # F = 0  (batch)
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
            max_steps=100000, throw=False, adjoint=diffrax.DirectAdjoint(),
        )
        y_sw = sol.ys[-1]
        y_sw = jnp.where(jnp.any(~jnp.isfinite(y_sw)), jnp.full(8, 1e6), y_sw)
        ydot = vector_field(t_switch, y_sw, (S_F, 0.0))     
        phi = get_phi(y_sw)                                 
        phidot = S_F * ydot[5] + ydot[7]                    # d(phi)/dt = S_F*d(l2)/dt + d(l4)/dt
        return jnp.array([phi, phidot])

    guess = jnp.array([t_switch_euler, lam4_tf_seed])
    solver = optx.LevenbergMarquardt(rtol=1e-10, atol=1e-12)
    solution = optx.least_squares(solve_and_get_residuals_ms, solver, guess,
                                  args=None, max_steps=500, throw=False)
    t_switch_opt, l4_opt = [float(v) for v in solution.value]
    print("optimal t_switch =", round(t_switch_opt, 4))
    print("optimal lambda4  =", round(l4_opt, 4))
    print("residuals [phi, phidot] =", np.array(solve_and_get_residuals_ms(solution.value, None)))

    # reconstruct and check x(0)
    ytf = jnp.concatenate([x_opt_tf_log, jnp.array([0.0, 0.0, 1.0, l4_opt])])
    sc = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10)
    batch_vf = lambda t, y, a: vector_field(t, y, (S_F, 0.0))
    sing_vf = lambda t, y, a: vector_field(t, y, (S_F, jnp.clip(
        get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max)), F_min, F_max)))
    sol_b = diffrax.diffeqsolve(diffrax.ODETerm(batch_vf), diffrax.Kvaerno5(),
        t0=tf, t1=t_switch_opt, dt0=-0.01, y0=ytf,
        saveat=diffrax.SaveAt(ts=jnp.linspace(tf, t_switch_opt, 200)),
        stepsize_controller=sc, max_steps=100000, throw=False)
    sol_s = diffrax.diffeqsolve(diffrax.ODETerm(sing_vf), diffrax.Kvaerno5(),
        t0=t_switch_opt, t1=0.0, dt0=-0.01, y0=sol_b.ys[-1],
        saveat=diffrax.SaveAt(ts=jnp.linspace(t_switch_opt, 0.0, 300)),
        stepsize_controller=sc, max_steps=100000, throw=False)
    y0 = np.asarray(sol_s.ys[-1])
    print("Recovered x(0) = [%.3f, SV=%.3f, %.3f, %.3f]  (target [10, %.4f, 0, 1])"
          % (y0[0], np.exp(y0[1]), y0[2], y0[3], float(S0)))
