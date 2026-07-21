import jax
import diffrax
import jax.numpy as jnp
from fedopt.odes import vector_field, get_phi
from fedopt.Fsin import get_Fsin, get_Fsin_unconstrained

def solve_and_get_residuals(xtf_guess, args):
    tf, x0, S_F, V_max, Fmin, Fmax = args
    ltf = jnp.array([0.0, 0.0, 1.0, 0.0])
    ytf = jnp.concatenate([xtf_guess, ltf])
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5, dtmin=1e-10)
    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: vector_field(t, y, (S_F, get_Fsin(y, (S_F, V_max, Fmin, Fmax))))),
        solver=solver,
        y0=ytf,
        t0=tf,
        t1=0,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e5),
        throw=False,
        adjoint=diffrax.DirectAdjoint(),
    )
    y_at_t0 = sol.ys[0]
    # Penalize diverged integrations
    y_at_t0 = jnp.where(jnp.any(jnp.isnan(y_at_t0)), jnp.full(8, 1e6), y_at_t0)

    # Residuals
    return jnp.array([
        y_at_t0[0] - x0[0],                         
        y_at_t0[1] - x0[1],                         
        y_at_t0[2] - x0[2],                         
        y_at_t0[3] - x0[3],                  
    ])

def solve_and_get_residuals_batch(l4_tf_guess, args):
    tf, x0, S_F, xtf, t_switch, V_max, Fmin, Fmax = args
    ltf = jnp.array([0.0, 0.0, 1.0, l4_tf_guess])
    ytf = jnp.concatenate([xtf, ltf])
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10, dtmin=1e-12, force_dtmin=True)
    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: vector_field(t, y, (S_F, 0))),
        solver=solver,
        y0=ytf,
        t0=tf,
        t1=t_switch,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e5)
    )
    y_at_tswitch = sol.ys[0]
    # Penalize diverged integrations
    y_at_tswitch = jnp.where(jnp.any(jnp.isnan(y_at_tswitch)), jnp.full(8, 1e6), y_at_tswitch)

    # Residuals
    return jnp.array([
        get_phi(y_at_tswitch) - 0
    ])

def solve_and_get_residuals_sing_and_batch(l4_tf_guess, args):
    tf, x0, S_F, xtf, t_switch, V_max, Fmin, Fmax = args
    ltf = jnp.array([0.0, 0.0, 1.0, l4_tf_guess])
    ytf = jnp.concatenate([xtf, ltf])
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10)
    sol_at_tswitch = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: vector_field(t, y, (S_F, 0))),
        solver=solver,
        y0=ytf,
        t0=tf,
        t1=t_switch,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e5),
        throw=False
    )
    y_at_tswitch = sol_at_tswitch.ys[0]

    sol_at_t0 = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(
            lambda t, y, args: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, Fmin, Fmax))))),
        solver=solver,
        y0=y_at_tswitch,
        t0=t_switch,
        t1=0.0,
        dt0=-0.01,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=int(1e5)
    )
    y_at_t0 = sol_at_t0.ys[0]
    # Penalize diverged integrations
    y_at_tswitch = jnp.where(jnp.any(jnp.isnan(y_at_tswitch)), jnp.full(8, 1e6), y_at_tswitch)
    y_at_t0 = jnp.where(jnp.any(jnp.isnan(y_at_t0)), jnp.full(8, 1e6), y_at_t0)

    # Residuals
    return jnp.array([
        get_phi(y_at_tswitch) - 0              
    ])

@jax.jit
def loss_fn(xtf_guess, args):
    res = solve_and_get_residuals(xtf_guess, args)
    return jnp.sum(res**2)

def loss_fn_batch(final_guess, args):
    res = solve_and_get_residuals_batch(final_guess, args)
    jax.debug.print("res {res}", res=res)
    return jnp.sum(res**2)

def loss_fn_sing_and_batch(final_guess, args):
    res = solve_and_get_residuals_sing_and_batch(final_guess, args)
    jax.debug.print("res{res}", res=res)
    return jnp.sum(res**2)
