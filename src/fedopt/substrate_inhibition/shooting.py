import jax
import diffrax
import jax.numpy as jnp
from .odes import vector_field, get_phi, l_to_orig, objective_batch
from .Fsin import get_Fsin, get_Fsin_unconstrained

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

def solve_and_get_residuals_bang_sing_batch(guess, args):
    """Bang -> singular -> batch forward shoot.  guess = [lambda(0) (4), t_switch_1, t_switch_2].
    Phase 0 bang (F=F_max) on [0,t1] brings the state ONTO the singular surface; phase 1 singular
    (F_sing) on [t1,t2] until V=V_max; phase 2 batch (F=0) folded as a terminal cost. Residuals (6):
    lambda3(0)=1; phi(t1)=phidot(t1)=0 (tangential bang->singular entry); V(t2)=V_max; costate
    continuity (XV,SV) at the batch junction."""
    tf, x0_log, S_F, V_max, F_min, F_max = args
    lam0 = guess[:4]
    t_switch_1 = jnp.clip(guess[4], 2e-3, tf - 0.2)
    t_switch_2 = jnp.clip(guess[5], t_switch_1 + 0.05, tf - 0.05)    # keep 0 < t1 < t2 < tf
    y0 = jnp.concatenate([x0_log, lam0])
    sc = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-14)
    bang_vf = lambda t, y, a: vector_field(t, y, (S_F, F_max))                       # BANG: F = F_max
    sing_vf = lambda t, y, a: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))))

    # ---- phase 0: bang F=F_max on [0, t_switch_1] (stiff: SV=1e-6 -> S* in log coords) ----
    sol0 = diffrax.diffeqsolve(diffrax.ODETerm(bang_vf), diffrax.Tsit5(),
        t0=0.0, t1=t_switch_1, dt0=1e-6, y0=y0, saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=sc, max_steps=1_000_000, throw=False, adjoint=diffrax.ForwardMode())
    y1 = sol0.ys[-1]
    y1 = jnp.where(jnp.any(~jnp.isfinite(y1)), jnp.full(8, 1e6), y1)
    yd1 = vector_field(t_switch_1, y1, (S_F, F_max))        # phidot is F-independent (g constant)
    phi1 = get_phi(y1)
    phidot1 = S_F * yd1[5] + yd1[7]

    # ---- phase 1: singular F=F_sing on [t_switch_1, t_switch_2] ----
    sol1 = diffrax.diffeqsolve(diffrax.ODETerm(sing_vf), diffrax.Tsit5(),
        t0=t_switch_1, t1=t_switch_2, dt0=0.01, y0=y1, saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=sc, max_steps=100000, throw=False, adjoint=diffrax.ForwardMode())
    y2 = sol1.ys[-1]
    y2 = jnp.where(jnp.any(~jnp.isfinite(y2)), jnp.full(8, 1e6), y2)
    x2, lam2 = y2[:4], y2[4:]

    # ---- phase 2: batch as a terminal cost from x(t_switch_2) ----
    grad_PV = l_to_orig(x2, jax.grad(lambda x: objective_batch(x, t_switch_2, (tf, S_F)))(x2))
    return jnp.array([
            lam0[2] - 1.0,                 # lambda3(0) = 1
            phi1,                          # bang -> singular: on the surface at t1
            phidot1,                       # bang -> singular: tangential entry at t1
            (x2[3] - V_max),               # singular -> batch: V(t2) = V_max
            lam2[0] - grad_PV[0],          # costate continuity at the batch junction: XV
            lam2[1] - grad_PV[1],          # costate continuity at the batch junction: SV
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
