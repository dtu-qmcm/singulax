import jax
import jax.numpy as jnp
from fedopt.odes import f_vec, g_vec

def lie_bracket_fg(x_val, S_F):
    df_dx = jax.jacobian(f_vec)(x_val)
    dg_dx = jax.jacobian(g_vec)(x_val, S_F) # This is 0 in our case
    return dg_dx @ f_vec(x_val) - df_dx @ g_vec(x_val, S_F)

def inner_bracket(x_val, S_F): return lie_bracket_fg(x_val, S_F)

def get_Fsin(y, args):
    S_F, V_max, Fmin, Fmax = args
    x, l = jnp.split(y, 2)
    
    # Term A (Numerator): l.T * [f, [f, g]]
    term_A = l @ (jax.jacobian(inner_bracket)(x, S_F) @ f_vec(x))
    
    # Term B (Denominator): l.T * [g, [f, g]]
    term_B = l @ (jax.jacobian(inner_bracket)(x, S_F) @ g_vec(x, S_F))

    # Final Singular Feed: F = -Term_A / Term_B
    F_sin = -term_A / (term_B + 1e-9)
    F_sin_clip = jnp.clip(F_sin, Fmin, Fmax)
    F = jnp.where(y[3] >= V_max, 0.0, F_sin_clip)
        
    return F