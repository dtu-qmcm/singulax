"""Singular feed rate for the substrate+product inhibition case.

On the singular arc phi = phi_dot = phi_ddot = 0; solving phi_ddot = 0 for F gives
    F_sing = - lambda^T [f,[f,g]] / lambda^T [g,[f,g]],
with the Lie-bracket convention  [a, b] = (db/dx) a - (da/dx) b.
"""
import jax
import jax.numpy as jnp
from substrate_product_inhibition.odes import f_vec, g_vec

def lie_bracket_fg(x_val, S_F):
    """First Lie bracket [f, g] = (dg/dx) f - (df/dx) g. Here dg/dx = 0 (g is constant)."""
    df_dx = jax.jacobian(f_vec)(x_val)
    dg_dx = jax.jacobian(g_vec)(x_val, S_F) # This is 0 in our case
    return dg_dx @ f_vec(x_val) - df_dx @ g_vec(x_val, S_F)

def inner_bracket(x_val, S_F):
    """Alias for the inner bracket [f, g] (the inner term of [f,[f,g]] and [g,[f,g]])."""
    return lie_bracket_fg(x_val, S_F)

def get_Fsin_unconstrained(y, args):
    """Singular feed from phi_ddot = 0:  F = - lambda^T[f,[f,g]] / lambda^T[g,[f,g]].

    The numerator [f,[f,g]] = (d[f,g]/dx) f - (df/dx) [f,g]. Result is clipped to [Fmin, Fmax].
    y = [x, lambda];  args = (S_F, V_max, Fmin, Fmax)."""
    S_F, V_max, Fmin, Fmax = args
    x, l = jnp.split(y, 2)
    
    # Term A (Numerator): l.T * [f, [f, g]] = l.T * (d[f,g]/dx @ f  -  df/dx @ [f,g])
    term_A = l @ (jax.jacobian(inner_bracket)(x, S_F) @ f_vec(x)
                  - jax.jacobian(f_vec)(x) @ inner_bracket(x, S_F))
    
    # Term B (Denominator): l.T * [g, [f, g]]
    term_B = l @ (jax.jacobian(inner_bracket)(x, S_F) @ g_vec(x, S_F))

    # Final Singular Feed: F = -Term_A / Term_B
    F_sin = -term_A / (term_B + 1e-9)
    F_sin_clip = jnp.clip(F_sin, Fmin, Fmax)
        
    return F_sin_clip

def get_Fsin(y, args):
    """As get_Fsin_unconstrained, but forced to 0 once V >= V_max (tank full -> batch).
    args = (S_F, V_max, Fmin, Fmax)."""
    S_F, V_max, Fmin, Fmax = args
    F = get_Fsin_unconstrained(y, args)
    F_sin = jnp.where(y[3] >= V_max, 0.0, F)        

    return F_sin
