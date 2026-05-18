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
    dx_dt = f_vec(y) + g_vec(y, S_F) * F
    return dx_dt

def get_dldt(t, y, args):
    S_F, F = args
    x, l = jnp.split(y, 2)
    
    # Define the Hamiltonian: H = lambda^T * (f(x) + g(x)*F)
    def hamiltonian(x_vars, S_F):
        f = f_vec(x_vars)
        g = g_vec(x_vars, S_F)
        return l @ (f + g * F)
    
    # Adjoint equations: dl/dt = -dH/dx
    dl_dt = -jax.grad(hamiltonian)(x, S_F)
    return dl_dt

def get_phi(y):
    x, l = jnp.split(y, 2)
    SV = jnp.exp(x[1])
    S_F = 150.0
    return l[1] * (S_F/SV) + l[3]

def vector_field(t, y, args):
    x, l = jnp.split(y, 2)
    S_F, F = args

    dx_dt = f_vec(x) + g_vec(x, S_F) * F
    dl_dt = get_dldt(t,y, args)

    return jnp.concatenate([dx_dt, dl_dt])

def hybrid_fedbatch(t, y, args):
    S_F, Ks, Ki = args

    S_star = jnp.sqrt(Ks * Ki)
    sigma_star = sigma(S_star)
    F_opt_ana = (sigma_star* y[0]) / (S_F-S_star)

    return vector_field(t, y, (S_F, F_opt_ana))
