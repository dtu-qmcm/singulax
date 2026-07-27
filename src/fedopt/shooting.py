import jax
import diffrax
import jax.numpy as jnp
from fedopt.odes import vector_field, get_phi, l_to_orig, objective_batch
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

def solve_and_get_residuals_sing(guess, args):
    tf, x0_log, S_F, V_max, F_min, F_max = args
    lam0 = guess[:4]
    t_switch = jnp.clip(guess[4], 0.5, tf - 0.05)         # keep 0 < t_switch < tf
    y0 = jnp.concatenate([x0_log, lam0])
    yd0 = vector_field(0.0, y0, (S_F, get_Fsin_unconstrained(y0, (S_F, V_max, F_min, F_max))))
    phi0 = get_phi(y0)
    phidot0 = S_F * yd0[5] + yd0[7]
    sing_vf = lambda t, y, a: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))))
    sc = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10)
    # forward singular arc 0 -> t_switch
    sol = diffrax.diffeqsolve(
            diffrax.ODETerm(sing_vf),
            diffrax.Tsit5(),
            t0=0.0, t1=t_switch, dt0=0.01, y0=y0,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=sc,
            max_steps=100000,
            throw=False,
            adjoint=diffrax.ForwardMode())
    y_sw = sol.ys[-1]
    y_sw = jnp.where(jnp.any(~jnp.isfinite(y_sw)), jnp.full(8, 1e6), y_sw)
    x_sw, lam_sw = y_sw[:4], y_sw[4:]
    grad_PV = l_to_orig(x_sw, jax.grad(lambda x: objective_batch(x, t_switch, (tf, S_F)))(x_sw))
    return jnp.array([
            lam0[2] - 1.0,                 # lambda3(0) = 1  (PV costate normalisation)
            phi0,                          # start on the singular surface
            phidot0,                       # start on the singular surface
            100*(x_sw[3] - V_max),         # V(t_switch) = V_max   <-- the switch trigger
            lam_sw[0] - grad_PV[0],        # costate continuity at the junction: XV
            lam_sw[1] - grad_PV[1],        # costate continuity at the junction: SV
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
        max_steps=int(1e5),
        adjoint=diffrax.DirectAdjoint(),
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
        throw=False,
        adjoint=diffrax.DirectAdjoint(),

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
    return jnp.sum(res**2)

def loss_fn_sing_and_batch(final_guess, args):
    res = solve_and_get_residuals_sing_and_batch(final_guess, args)
    jax.debug.print("res{res}", res=res)
    return jnp.sum(res**2)
