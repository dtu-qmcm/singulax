import jax
import diffrax
import jax.numpy as jnp
from fedopt.odes import vector_field
from fedopt.Fsin import get_Fsin

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
        throw=False
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

@jax.jit
def loss_fn(xtf_guess, args):
    res = solve_and_get_residuals(xtf_guess, args)
    return jnp.sum(res**2)