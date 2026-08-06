"""State, costate and switching-function machinery for the substrate+product
inhibition case, in PHYSICAL coordinates x = [XV, SV, PV, V].

Control-affine dynamics  xdot = f(x) + g(x) F  (F = feed rate):
    f(x) = [mu*XV, -sigma*XV, pi*XV, 0],   g = [0, S_F, 0, 1]
with S = SV/V, P = PV/V and the kinetics from `.kinetics`.

Sign convention: PV(tf) is MAXIMISED, so the Hamiltonian H = lambda^T (f + gF) is
maximised, the costate obeys lambda_dot = -dH/dx, and the switching function
phi = dH/dF = S_F*lambda_SV + lambda_V selects F_max when phi > 0 (opposite sign to
Shin & Lim, who minimise -PV).
"""
import jax
import diffrax
import jax.numpy as jnp
from substrate_product_inhibition.kinetics import mu, sigma, pi

def f_vec(x_val):
    """Drift term f(x) of the control-affine dynamics; x = [XV, SV, PV, V] (physical)."""
    S = x_val[1] / x_val[3]
    P = x_val[2] / x_val[3]

    return jnp.array([
        mu(S, P) * x_val[0],
        (-sigma(S, P) * x_val[0]),
        pi(S, P) * x_val[0],
        0.0
    ])

def g_vec(x_val, S_F):
    """Control vector g = [0, S_F, 0, 1] (constant): the feed adds substrate and volume."""
    return jnp.array([0.0, S_F, 0.0, 1.0])

def vf_states(t,y, args):
    """State RHS xdot = f(x) + g(x)*F for a diffrax solve; args = (S_F, F)."""
    S_F, F = args
    dx_dt = f_vec(y) + g_vec(y, S_F) * F
    return dx_dt

def get_dldt(t, y, args):
    """Costate RHS lambda_dot = -dH/dx for the augmented state y = [x, lambda],
    with H = lambda^T (f(x) + g(x) F); args = (S_F, F). Computed by autodiff of H."""
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
    """Switching function phi = dH/dF = S_F*lambda_SV + lambda_V, for y = [x, lambda].
    Sign convention (maximisation): phi > 0 -> F_max, phi < 0 -> F_min, phi = 0 -> singular.
    NOTE: S_F is hardcoded to 150.0 here."""
    x, l = jnp.split(y, 2)
    S_F = 150.0
    return l[1] * S_F + l[3]

def vector_field(t, y, args):
    """Augmented RHS [xdot, lambda_dot] for the state-costate vector y = [x, lambda]
    (used by the shooting integrations); F is fixed in args = (S_F, F)."""
    x, l = jnp.split(y, 2)
    S_F, F = args

    dx_dt = f_vec(x) + g_vec(x, S_F) * F
    dl_dt = get_dldt(t,y, args)

    return jnp.concatenate([dx_dt, dl_dt])

# Flat convention where x and F are distinct differentiable arguments
def states_ode(x, F, S_F):
    """State RHS with x, F, S_F as separate differentiable args (collocation/AD convention)."""
    return vf_states(0.0, x, (S_F, F))

def costate_ode(y, S_F, F):
    """Costate RHS with (y, S_F, F) as separate args; thin wrapper over get_dldt."""
    return get_dldt(0.0, y, (S_F, F))
