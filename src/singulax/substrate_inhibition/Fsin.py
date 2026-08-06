"""Singular feed rate for the SUBSTRATE inhibition case, in PHYSICAL-SV coordinates
(the log-SV state y[1] is mapped to SV via exp before the brackets).

    F_sing = - lambda^T [f,[f,g]] / lambda^T [g,[f,g]]   (from phi_ddot = 0),
with the Lie-bracket convention  [a, b] = (db/dx) a - (da/dx) b.
"""
import jax
import jax.numpy as jnp
from substrate_inhibition.odes import f_vec_orig, g_vec_orig, l_to_orig

def lie_bracket_fg(x_val, S_F):
    """First Lie bracket [f, g] = (dg/dx) f - (df/dx) g in physical-SV coords (dg/dx = 0)."""
    df_dx = jax.jacobian(f_vec_orig)(x_val)
    dg_dx = jax.jacobian(g_vec_orig)(x_val, S_F) # This is 0 in our case
    return dg_dx @ f_vec_orig(x_val) - df_dx @ g_vec_orig(x_val, S_F)

def inner_bracket(x_val, S_F):
    """Alias for the inner bracket [f, g] (the inner term of [f,[f,g]] and [g,[f,g]])."""
    return lie_bracket_fg(x_val, S_F)

def get_Fsin_unconstrained(y, args):
    """Singular feed from phi_ddot = 0:  F = - lambda^T[f,[f,g]] / lambda^T[g,[f,g]]
    (physical-SV coords; the log-SV state y[1] is mapped to SV via exp). Clipped to [Fmin, Fmax].

    BUG / TODO: term_A below computes only (d[f,g]/dx) f and is MISSING the second term
    -(df/dx)[f,g] of [f,[f,g]] -- the same omission already fixed in the SPI version
    (fedopt.substrate_product_inhibition.Fsin). Applying the fix here left PV(tf) unchanged
    but blew up the shoot residual (~3e-4 -> ~4e-2), so it needs a re-tuned seed before
    re-enabling. y = [x, lambda];  args = (S_F, V_max, Fmin, Fmax)."""
    S_F, V_max, Fmin, Fmax = args
    x, l = jnp.split(y, 2)
    SV = jnp.exp(x[1])
    x_orig = x.at[1].set(SV)
    
    # Term A (Numerator): l.T * [f, [f, g]]
    term_A = l @ (jax.jacobian(inner_bracket)(x_orig, S_F) @ f_vec_orig(x_orig))
    
    # Term B (Denominator): l.T * [g, [f, g]]
    term_B = l @ (jax.jacobian(inner_bracket)(x_orig, S_F) @ g_vec_orig(x_orig, S_F))

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
