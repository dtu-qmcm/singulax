"""Closed-form ('analytical') feed-rate reference trajectories and event conditions for
the substrate inhibition case.

On the singular arc the optimum holds S at S* = sqrt(Ks*Ki), giving the analytical feed
    F = sigma(S*) * XV / (S_F - S*).
Naming: states_analytical_sol_* are STATE-only RHS (4-dim, via vf_states); analytical_sol_*
are the AUGMENTED state+costate RHS (8-dim, via vector_field); the *_singular / *_batch /
*_bang suffix picks the feed for that arc (F_sing / 0 / F_max).
"""
import jax.numpy as jnp
from singulax.substrate_inhibition.kinetics import sigma
from singulax.substrate_inhibition.odes import vf_states, vector_field

def states_analytical_sol(t, y, args):
    """State RHS under the analytical singular feed, switched off once V >= V_max. args = (S_F, V_max)."""
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
    """Event / residual V - V_max (the tank-full trigger). args = (S_F, V_max)."""
    S_F, V_max = args
    return y[3]-V_max

def states_analytical_sol_singular(t, y, args):
    """State RHS under the analytical singular feed F = sigma(S*)*XV/(S_F-S*). args = S_F."""
    S_F = args
    Ks = 0.1
    Ki = 100

    S_star = jnp.sqrt(Ks * Ki)
    sigma_star = sigma(S_star)
    F_opt_ana = (sigma_star* y[0]) / (S_F-S_star)

    vf = vf_states(t, y, (S_F, F_opt_ana))

    return vf

def analytical_sol_singular(t, y, args):
    """Augmented (state+costate) RHS under the analytical singular feed. args = S_F."""
    S_F = args
    Ks = 0.1
    Ki = 100

    S_star = jnp.sqrt(Ks * Ki)
    sigma_star = sigma(S_star)
    F_opt_ana = (sigma_star* y[0]) / (S_F-S_star)

    vf = vector_field(t, y, (S_F, F_opt_ana))

    return vf
    
def states_analytical_sol_batch(t, y, args):
    """State RHS for the batch arc (F = 0). args = S_F."""
    S_F = args
    vf = vf_states(t, y, (S_F, 0.0))
    return vf

def analytical_sol_batch(t, y, args):
    """Augmented RHS for the batch arc (F = 0). args = S_F."""
    S_F = args
    vf = vector_field(t, y, (S_F, 0.0))
    return vf

def states_analytical_sol_bang(t, y, args):
    """State RHS for the bang arc (F = F_max). args = (S_F, F_max)."""
    S_F, F_max = args
    return vf_states(t, y, (S_F, F_max))

def analytical_sol_bang(t, y, args):
    """Augmented RHS for the bang arc (F = F_max). args = (S_F, F_max)."""
    S_F, F_max = args
    vf = vector_field(t, y, (S_F, F_max))
    return vf

def S_star_cond(t, y, args, **kwargs):
    """Event / residual S - S* with S = SV/V, S* = sqrt(Ks*Ki) (singular-entry trigger)."""
    Ks = 0.1
    Ki = 100.0
    S_star = jnp.sqrt(Ks * Ki)
    SV = jnp.exp(y[1])
    V = y[3]
    return SV / V - S_star

def calculate_real_feed(states, args):
    """Analytical singular feed value F = sigma(S*)*XV/(S_F-S*), zeroed once V >= V_max.
    args = (S_F, V_max, Ks, Ki)."""
    S_F, V_max, Ks, Ki = args
    
    S_star = jnp.sqrt(Ks * Ki)
    F_opt_ana = (sigma(S_star) * states[0]) / (S_F - S_star)

    return jnp.where(states[3] < V_max, F_opt_ana, 0.0)
