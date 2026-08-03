import jax
import jax.numpy as jnp
from .odes import f_vec_orig, g_vec_orig, l_to_orig

def lie_bracket_fg(x_val, S_F):
    df_dx = jax.jacobian(f_vec_orig)(x_val)
    dg_dx = jax.jacobian(g_vec_orig)(x_val, S_F) # This is 0 in our case
    return dg_dx @ f_vec_orig(x_val) - df_dx @ g_vec_orig(x_val, S_F)

def inner_bracket(x_val, S_F): return lie_bracket_fg(x_val, S_F)

def get_Fsin_unconstrained(y, args):
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
    S_F, V_max, Fmin, Fmax = args
    F = get_Fsin_unconstrained(y, args)
    F_sin = jnp.where(y[3] >= V_max, 0.0, F)        

    return F_sin
