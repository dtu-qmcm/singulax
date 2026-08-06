"""State, costate and switching-function machinery for the SUBSTRATE inhibition case.

Two coordinate conventions coexist:
  * f_vec / g_vec / vf_states use LOG-substrate state  x = [XV, log(SV), PV, V]
    (so SV = exp(x[1])); this keeps SV strictly positive so the Stage-1 Euler NLP
    cannot drive it negative (see the singulax.collocation module docstring).
  * f_vec_orig / g_vec_orig are the PHYSICAL-SV versions x = [XV, SV, PV, V], used
    for the costate and the Lie brackets; l_to_orig maps the SV-costate between them.

Kinetics (.kinetics) are substrate-inhibited (Haldane); the productivity-optimal
substrate is S* = sqrt(Ks*Ki). Sign convention: PV(tf) is MAXIMISED, so the costate
obeys lambda_dot = -dH/dx and phi = S_F*lambda_SV + lambda_V picks F_max when phi > 0.
"""
import jax
import diffrax
import jax.numpy as jnp
from singulax.substrate_inhibition.kinetics import mu, sigma, pi

def f_vec(x_val):
    """Drift f(x) in LOG-SV coords x = [XV, log(SV), PV, V] (SV = exp(x[1]))."""
    SV = jnp.exp(x_val[1])
    S = SV/ x_val[3]

    return jnp.array([
        mu(S) * x_val[0],
        (-sigma(S) * x_val[0]/SV),
        pi(S) * x_val[0],
        0.0
    ])

def g_vec(x_val, S_F):
    """Control vector g in LOG-SV coords (g[1] = S_F/SV)."""
    SV = jnp.exp(x_val[1])
    return jnp.array([0.0, S_F/SV, 0.0, 1.0])

def vf_states(t,y, args):
    """State RHS xdot = f(x) + g(x)*F in log-SV coords; args = (S_F, F)."""
    S_F, F = args
    dx_dt = f_vec(y) + g_vec(y, S_F) * F
    return dx_dt

def f_vec_orig(x_val):
    """Drift f(x) in PHYSICAL-SV coords x = [XV, SV, PV, V] (used for the costate + brackets)."""
    S = x_val[1] / x_val[3]
    return jnp.array([
        mu(S) * x_val[0],
        -sigma(S) * x_val[0],
         pi(S) * x_val[0],
         0.0
    ])

def g_vec_orig(x_val, S_F):
    """Control vector g = [0, S_F, 0, 1] in physical-SV coords."""
    return jnp.array([0.0, S_F, 0.0, 1.0])

def states_ode_phys(x, F, S_F):
    return f_vec_orig(x) + g_vec_orig(x, S_F) * F

def l_to_orig(x, l):
    """Convert l[1] from log(SV)-conjugate to SV-conjugate."""
    SV = jnp.exp(x[1])
    return l.at[1].set(l[1] / SV)

def get_dldt(t, y, args):
    """Costate RHS lambda_dot = -dH/dx, evaluated in PHYSICAL-SV coords (the log-SV state
    is mapped to SV via exp first). H = lambda^T (f_orig + g_orig F); args = (S_F, F)."""
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
    """Switching function phi = S_F*lambda_SV + lambda_V, for y = [x, lambda].
    Convention (maximisation): phi > 0 -> F_max, < 0 -> F_min, = 0 -> singular. S_F hardcoded 150."""
    x, l = jnp.split(y, 2)
    S_F = 150.0
    return l[1] * S_F + l[3]

def vector_field(t, y, args):
    """Augmented RHS [xdot, lambda_dot] for y = [x, lambda] (state in log-SV, costate physical)."""
    x, l = jnp.split(y, 2)
    S_F, F = args

    dx_dt = f_vec(x) + g_vec(x, S_F) * F
    dl_dt = get_dldt(t,y, args)

    return jnp.concatenate([dx_dt, dl_dt])

def hybrid_fedbatch(t, y, args):
    """Integrate with the ANALYTICAL singular feed that holds S at S* = sqrt(Ks*Ki):
    F = sigma(S*)*XV / (S_F - S*).  args = (S_F, Ks, Ki)."""
    S_F, Ks, Ki = args

    S_star = jnp.sqrt(Ks * Ki)
    sigma_star = sigma(S_star)
    F_opt_ana = (sigma_star* y[0]) / (S_F-S_star)

    return vector_field(t, y, (S_F, F_opt_ana))

def batch_phys_vf(t, x, args):
    """Physical-SV batch RHS (F = 0), with log(SV) floored to avoid -inf as SV -> 0."""
    S_F = args
    x = x.at[1].set(jnp.maximum(x[1], jnp.log(1e-12)))
    return vf_states(t, x, (S_F, 0.0))

def objective_batch(x_sw, t_switch, args):
    """Batch terminal cost: integrate F=0 from t_switch to tf and return PV(tf) = x[2](tf).
    Differentiated (jax.grad) to get the batch costate at the singular exit. args = (tf, S_F)."""
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
    """State RHS with x, F, S_F as separate differentiable args (collocation/AD convention)."""
    return vf_states(0.0, x, (S_F, F))

def costate_ode(y, S_F, F):
    """Costate RHS with (y, S_F, F) as separate args; thin wrapper over get_dldt."""
    return get_dldt(0.0, y, (S_F, F))
