import jax
import diffrax
import jax.numpy as jnp
from fedopt.si_kinetics import mu, sigma, pi

def f_vec(x_val):
    SV = jnp.exp(x_val[1])
    S = SV/ x_val[3]

    return jnp.array([
        mu(S) * x_val[0],
        (-sigma(S) * x_val[0]/SV),
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

def f_vec_orig(x_val):
    S = x_val[1] / x_val[3]
    return jnp.array([
        mu(S) * x_val[0],
        -sigma(S) * x_val[0],
         pi(S) * x_val[0],
         0.0
    ])

def g_vec_orig(x_val, S_F):
    return jnp.array([0.0, S_F, 0.0, 1.0])

def l_to_orig(x, l):
    """Convert l[1] from log(SV)-conjugate to SV-conjugate."""
    SV = jnp.exp(x[1])
    return l.at[1].set(l[1] / SV)

def get_dldt(t, y, args):
    S_F, F = args
    x, l = jnp.split(y, 2)
    SV = jnp.exp(x[1])
    # Define the Hamiltonian: H = lambda^T * (f(x) + g(x)*F)
    def hamiltonian(x_orig, S_F):
        f_orig = f_vec_orig(x_orig)
        g_orig = g_vec_orig(x_orig, S_F)
        return l @ (f_orig + g_orig * F)
    
    # Adjoint equations: dl/dt = -dH/dx
    x_orig = x.at[1].set(SV)
    dl_dt = -jax.grad(hamiltonian)(x_orig, S_F)
    
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

def hybrid_fedbatch(t, y, args):
    S_F, Ks, Ki = args

    S_star = jnp.sqrt(Ks * Ki)
    sigma_star = sigma(S_star)
    F_opt_ana = (sigma_star* y[0]) / (S_F-S_star)

    return vector_field(t, y, (S_F, F_opt_ana))

def batch_phys_vf(t, x, args):
    S_F = args
    x = x.at[1].set(jnp.maximum(x[1], jnp.log(1e-12)))
    return vf_states(t, x, (S_F, 0.0))

def objective_batch(x_sw, t_switch, args):
    tf, S_F = args
    sc = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(batch_phys_vf),
        diffrax.Tsit5(),
        t0=t_switch, t1=tf, dt0=0.01, y0=x_sw,
        args=S_F,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=sc,
        max_steps=100000,
        throw=False,
        adjoint=diffrax.DirectAdjoint())
    return sol.ys[-1][2]   

# Flat convention where x and F are distinct differentiable arguments
def states_ode(x, F, S_F):
    return vf_states(0.0, x, (S_F, F))

def costate_ode(y, S_F, F):
    return get_dldt(0.0, y, (S_F, F))
