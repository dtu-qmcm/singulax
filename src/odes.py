import jax.numpy as jnp

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