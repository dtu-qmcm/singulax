import diffrax
import os
import optimistix as optx
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, PIDController
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import fsolve
from fedopt.si_analytical import states_analytical_sol, states_analytical_sol_batch, states_analytical_sol_bang, states_analytical_sol_singular, calculate_real_feed, volume_cond, analytical_sol_bang, analytical_sol_batch, analytical_sol_singular
from fedopt.odes import get_phi, vector_field
from fedopt.Fsin import get_Fsin_unconstrained, get_Fsin

jax.config.update("jax_enable_x64", True)

S0 = 1e-6
F_max = 2.0
y0 = jnp.array([10, jnp.log(S0 * 1.0), 0., 1.])
S_F = 150
V_max = 4
t_end = 9.0

def phi_cond(t,y,args, **kwargs):
    S_F = args
    return get_phi(y)

def solve_and_get_residuals(guess, args):
    x_tf_guess, t1_guess, l4_tf_guess = guess
    S_F, Fmin, Fmax, V_max, x0, tf = args
    y_tf = jnp.array([x_tf_guess[0], x_tf_guess[1], x_tf_guess[2], V_max, 0.0, 0.0, 1.0, l4_tf_guess])
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-12, atol=1e-12)
    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(phi_cond, root_finder)
    sol_full = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: vector_field(t, y, (S_F, 0.0))),
        solver=solver,
        y0=y_tf,
        t0=tf,
        t1=0.0,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e6),
        event=event,
    )
    t_full = sol_full.ts[-1]
    y_at_t_full = sol_full.ys[-1]

    sol_singular = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, Fmin, Fmax))))),
        solver=solver,
        y0=y_at_t_full,
        t0=t_full,
        t1=t1_guess,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e6),
    )
    y_at_t1 = sol_singular.ys[-1]

    sol_bang = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: vector_field(t, y, (S_F,Fmax))),
        solver=solver,
        y0=y_at_t1,
        t0=t1_guess,
        t1=0.0,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e6)
    )
    y_at_t0 = sol_bang.ys[-1]

    # Penalize diverged integrations
    y_at_t_full = jnp.where(jnp.any(jnp.isnan(y_at_t_full)), jnp.full(8, 1e6), y_at_t_full)
    y_at_t1 = jnp.where(jnp.any(jnp.isnan(y_at_t1)), jnp.full(8, 1e6), y_at_t1)
    y_at_t0 = jnp.where(jnp.any(jnp.isnan(y_at_t0)), jnp.full(8, 1e6), y_at_t0)

    return jnp.array([
        get_phi(y_at_t1),
        y_at_t0[0] - x0[0],
        y_at_t0[1] - x0[1],
        y_at_t0[2] - x0[2],
        y_at_t0[3] - x0[3]
                     ])
