import diffrax
import os
import optimistix as optx
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, PIDController
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import fsolve
from fedopt.substrate_inhibition.analytical import states_analytical_sol, states_analytical_sol_batch, states_analytical_sol_bang, states_analytical_sol_singular, calculate_real_feed, volume_cond, analytical_sol_bang, analytical_sol_batch, analytical_sol_singular
from fedopt.substrate_inhibition.odes import get_phi, vector_field
from fedopt.substrate_inhibition.Fsin import get_Fsin_unconstrained, get_Fsin

jax.config.update("jax_enable_x64", True)

Ks = 0.1
Ki = 100
S0 = 1e-6
F_max = 2.0
y0 = jnp.array([10, jnp.log(S0 * 1.0), 0., 1.])
S_F = 150
V_max = 4
t_end = 9.0

t1_guess = 0.01
l4_tf_guess = -16
guess = jnp.array([t1_guess, l4_tf_guess])


def solve_and_get_residuals(guess, args):
    t1_guess = guess[0]
    l4_tf_guess = guess[1]
    S_F, V_max, Fmin, Fmax, tf, l1_tf, l2_tf, l3_tf = args

    # FORWARD SIMULATION
    # Phase 1: bang (F_max) from 0 to t1_guess
    solution_to_tswitch = diffeqsolve(
        ODETerm(states_analytical_sol_bang),
        Kvaerno5(),
        t0=0, t1=t1_guess, dt0=0.01, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        args=(S_F, F_max),
    )
    y_at_switch = solution_to_tswitch.ys[0]

    # Find t_full: when V reaches V_max, starting from y_at_switch after the bang phase
    root_finder = optx.Newton(1e-6, 1e-6, optx.rms_norm)
    event = diffrax.Event(volume_cond, root_finder)

    sol = diffeqsolve(
        ODETerm(states_analytical_sol),
        Kvaerno5(),
        t0=t1_guess, t1=t_end, dt0=0.01, y0=y_at_switch,
        event=event,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
        args=(S_F, V_max),
    )
    t_full = sol.ts.squeeze()

    # Phase 2: singular arc from t1 to tfull
    solution_to_full = diffeqsolve(ODETerm(states_analytical_sol),
                              Kvaerno5(),
                              stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
                              t0=t1_guess, t1=t_full, dt0=0.01,
                              y0=y_at_switch,
                              saveat= diffrax.SaveAt(t1=True),
                              args=(S_F, V_max),
                              max_steps=int(1e6),
                              )
    y_at_full = solution_to_full.ys[0]

    # Phase 3: batch from tfull to tf
    solution_to_end = diffeqsolve(ODETerm(states_analytical_sol),
                              Kvaerno5(),
                              stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
                              t0=t_full, t1=t_end, dt0=0.01,
                              y0=y_at_full,
                              saveat= diffrax.SaveAt(t1=True),
                              args=(S_F, V_max),
                              max_steps=int(1e6),
                              )
    y_at_end = solution_to_end.ys[0]
 
    y_tf = jnp.array([y_at_end[0], y_at_end[1], y_at_end[2], y_at_end[3], 0, 0, 1, l4_tf_guess])

    # BACKWARD SIMULATION
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-12, atol=1e-12)
    sol_at_tfull_bwd = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: analytical_sol_batch(t, y, (S_F))),
        solver=solver,
        y0=y_tf,
        t0=tf,
        t1=t_full,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e6),
        throw=False,
    )
    y_at_tfull_bwd = sol_at_tfull_bwd.ys[0]

    # Phase 2: singular arc from t_full to t1_guess
    sol_at_t1_bwd = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: analytical_sol_singular(t, y, (S_F))),
        solver=solver,
        y0=y_at_tfull_bwd,
        t0=t_full,
        t1=t1_guess,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e6),
        throw=False,
    )
    y_at_t1_bwd = sol_at_t1_bwd.ys[0]

    # Phase 3: bang (Fmax) from t1 to t0
    sol_at_t0_bwd = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: analytical_sol_bang(t, y, (S_F,Fmax))),
        solver=solver,
        y0=y_at_t1_bwd,
        t0=t1_guess,
        t1=0.0,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e6),
        throw=False,
    )
    y_at_t0_bwd = sol_at_t0_bwd.ys[0]

    # Penalize diverged integrations
    y_at_tfull_bwd = jnp.where(jnp.any(jnp.isnan(y_at_tfull_bwd)), jnp.full(8, 1e6), y_at_tfull_bwd)
    y_at_t1_bwd = jnp.where(jnp.any(jnp.isnan(y_at_t1_bwd)), jnp.full(8, 1e6), y_at_t1_bwd)
    y_at_t0_bwd = jnp.where(jnp.any(jnp.isnan(y_at_t0_bwd)), jnp.full(8, 1e6), y_at_t0_bwd)
    

    return jnp.array([
        get_phi(y_at_t1_bwd),
        get_phi(y_at_tfull_bwd)
                     ])
args_res = S_F, V_max, 0.0, F_max, t_end, 0, 0, 1

res0 = solve_and_get_residuals(guess, args_res)
print(f"Initial residuals:  φ(t1) = {float(res0[0]):.4e}   φ(t2) = {float(res0[1]):.4e}")

def residuals_np(params_np):
    try:
        params = jnp.array(params_np)
        res = solve_and_get_residuals(params, args_res)
        np_res = np.array(res)
        if not np.all(np.isfinite(np_res)):
            np_res = np.array([1e6, 1e6])
    except Exception:
        np_res = np.array([1e6, 1e6])
    print(f"  t1={params_np[0]:.6f}  l4_tf={params_np[1]:.6f}"
          f"  →  φ(t1)={np_res[0]:.3e}  φ(t2)={np_res[1]:.3e}")
    return np_res

print("\nRunning fsolve ...")
sol_fsolve, info, ier, msg = fsolve(residuals_np, [t1_guess, l4_tf_guess],
                                    full_output=True, xtol=1e-8)
print(f"\nfsolve exit flag = {ier}  ({msg})")

t1_opt    = float(sol_fsolve[0])
l4_tf_opt = float(sol_fsolve[1])
print(f"Optimal:   t1 = {t1_opt:.6f} h   l4_tf = {l4_tf_opt:.6f}")
res_f = solve_and_get_residuals(jnp.array(sol_fsolve), args_res)
print(f"Final residuals:  φ(t1) = {float(res_f[0]):.2e}   φ(t2) = {float(res_f[1]):.2e}")
