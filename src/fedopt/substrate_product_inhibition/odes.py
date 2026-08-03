import jax
import diffrax
import jax.numpy as jnp
from .kinetics import mu, sigma, pi

def f_vec(x_val):
    S = x_val[1] / x_val[3]
    P = x_val[2] / x_val[3]

    return jnp.array([
        mu(S, P) * x_val[0],
        (-sigma(S, P) * x_val[0]),
        pi(S, P) * x_val[0],
        0.0
    ])

def g_vec(x_val, S_F):
    return jnp.array([0.0, S_F, 0.0, 1.0])

def vf_states(t,y, args):
    S_F, F = args
    dx_dt = f_vec(y) + g_vec(y, S_F) * F
    return dx_dt

def get_dldt(t, y, args):
    S_F, F = args
    x, l = jnp.split(y, 2)
    # Define the Hamiltonian: H = lambda^T * (f(x) + g(x)*F)
    def hamiltonian(x, S_F):
        f = f_vec(x)
        g = g_vec(x, S_F)
        return l @ (f + g * F)
    
    # Adjoint equations: dl/dt = -dH/dx
    dl_dt = -jax.grad(hamiltonian)(x, S_F)
    
    return dl_dt

def get_phi(y):
    x, l = jnp.split(y, 2)
    S_F = 150.0
    return l[1] * S_F + l[3]

def vector_field(t, y, args):
    x, l = jnp.split(y, 2)
    S_F, F = args

    dx_dt = f_vec(x) + g_vec(x, S_F) * F
    dl_dt = get_dldt(t,y, args)

    return jnp.concatenate([dx_dt, dl_dt])

# Flat convention where x and F are distinct differentiable arguments
def states_ode(x, F, S_F):
    return vf_states(0.0, x, (S_F, F))

def costate_ode(y, S_F, F):
    return get_dldt(0.0, y, (S_F, F))
