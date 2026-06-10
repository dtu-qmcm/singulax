import jax.numpy as jnp
from fedopt.si_kinetics import sigma
from fedopt.odes import vf_states

def states_analytical_sol(t, y, args):
    S_F, V_max = args
    Ks = 0.1
    Ki = 100

    S_star = jnp.sqrt(Ks * Ki)
    sigma_star = sigma(S_star)
    F_opt_ana = (sigma_star* y[0]) / (S_F-S_star)
    F = jnp.where(y[3]>= V_max, 0.0, F_opt_ana)

    vf = vf_states(t, y, (S_F, F))

    return vf

def volume_cond(t, y, args, **kwargs):
    S_F, V_max = args
    return y[3]-V_max

def states_analytical_sol_singular(t, y, args):
    S_F = args
    Ks = 0.1
    Ki = 100

    S_star = jnp.sqrt(Ks * Ki)
    sigma_star = sigma(S_star)
    F_opt_ana = (sigma_star* y[0]) / (S_F-S_star)

    vf = vf_states(t, y, (S_F, F_opt_ana))

    return vf

def states_analytical_sol_batch(t, y, args):
    S_F = args
    vf = vf_states(t, y, (S_F, 0.0))
    return vf

def states_analytical_sol_bang(t, y, args):
    S_F, F_max = args
    return vf_states(t, y, (S_F, F_max))

def S_star_cond(t, y, args, **kwargs):
    Ks = 0.1
    Ki = 100.0
    S_star = jnp.sqrt(Ks * Ki)
    SV = jnp.exp(y[1])
    V = y[3]
    return SV / V - S_star

def calculate_real_feed(states, args):
    S_F, V_max, Ks, Ki = args
    
    S_star = jnp.sqrt(Ks * Ki)
    F_opt_ana = (sigma(S_star) * states[0]) / (S_F - S_star)

    return jnp.where(states[3] < V_max, F_opt_ana, 0.0)
