import jax
import jax.numpy as jnp
from fedopt.si_kinetics import mu, sigma, pi

def f_vec(x_val):
    SV = jnp.exp(x_val[1])
    S = SV/ x_val[3]

    return jnp.array([
        mu(S) * x_val[0],
        (-sigma(S) * x_val[0])/SV,
        pi(S) * x_val[0],
        0.0
    ])

def g_vec(x_val, S_F):
    SV = jnp.exp(x_val[1])
    return jnp.array([0.0, S_F/SV, 0.0, 1.0])

def vf_states(t,y, args):
    S_F, F = args
    x, l = jnp.split(y, 2)
    dx_dt = f_vec(x) + g_vec(x, S_F) * F
    return dx_dt

def hamiltonian(t, y, args):
    S_F, F = args
    x, l = jnp.split(y, 2)
    f = f_vec(x)
    g = g_vec(x, S_F)
    return l @ (f + g * F)

def get_dldt(t, y, args):
    S_F, F = args
    x, l = jnp.split(y, 2)

    # Adjoint equations: dl/dt = -dH/dx
    dl_dt = -jax.grad(hamiltonian)(x, S_F)
    return dl_dt

def vector_field(t, y, args):
    x, l = jnp.split(y, 2)
    S_F, F = args

    dx_dt = f_vec(x) + g_vec(x, S_F) * F
    dl_dt = get_dldt(t,y, args)

    return jnp.concatenate([dx_dt, dl_dt])