import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

def F_calc(x, args):
    x1, x2 = x[0], x[1]
    u = args
    c = jnp.cos(x1)
    s = jnp.sin(x1)
    e = 0.5
    D  = 1-e*c**2

    F2 = -(e*x2**2*c + s+ c*u)/D
    F4 = -(e*s*(x2**2 + c)+ u)/ D

    return jnp.array([F2, F4])

def states_ode(x, args):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    Fs = F_calc(x, args)
    F2, F4 = Fs[0], Fs[1]
    
    dx1_dt = x2
    dx2_dt = F2
    dx3_dt = x4
    dx4_dt = F4
    return jnp.array([dx1_dt, dx2_dt, dx3_dt, dx4_dt])

def hamiltonian(x, u, l):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    c = jnp.cos(x1)
    s = jnp.sin(x1)
    e = 0.5
    D  = 1-e*c**2
    Fs = F_calc(x, u)
    F2, F4 = Fs[0], Fs[1]
    H = l[0]*x2 + l[1]*F2 +l[2]*x4 + l[3]*F4
    return H

def switching_fn(x, l):
    x1 = x[0]
    c = jnp.cos(x1)
    e = 0.5
    D = 1 - e*c**2
    return -(c*l[1] + l[3]) / D

def costate_ode(x, u, lam):
    return -jax.grad(hamiltonian, argnums=0)(x, u, lam)

def u_sing(x, l, t):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    l1, l2, l3, l4 = l[0], l[1], l[2], l[3]
    c = jnp.cos(x1)
    s = jnp.sin(x1)
    e = 0.5
    D  = 1-e*c**2
    S = jnp.sin(2*x1)
    
    A = ((2-2*e-D)*c*x2**2-2*s**2+D)/(S)
    B = (D*x2*l1)/(c*l2)
    u_sin = A - B
    return u_sin

def simulate(u_vec):
    def step(x, u):
        dx = states_ode(x, u)
        return x + h * dx, None
    x_final, _ = jax.lax.scan(step, jnp.zeros(4), u_vec)
    return x_final

def objective(v_vec, args):
    u_vec = jnp.tanh(v_vec)
    x_final = simulate(u_vec)
    diff = x_final - xf
    return 0.5 * jnp.dot(diff, diff)

N = 1000
t_f = 4.912
h = t_f / N
xf = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    obj_jit  = jax.jit(lambda v: objective(v, None))
    grad_jit = jax.jit(jax.grad(lambda v: objective(v, None)))
    result = scipy.optimize.minimize(
        fun    = lambda v: np.array(obj_jit(jnp.array(v))),
        jac    = lambda v: np.array(grad_jit(jnp.array(v))),
        x0     = np.zeros(N),
        method = ipyopt.optimize.ipopt,
        constraints = jnp.array([-1.0, 1.0])
    )
    u_opt = np.tanh(result.x)

    def simulate_all(u_vec):
        def step(x, u):
            dx = states_ode(x, u)
            x_next = x + h * dx
            return x_next, x_next
        _, x_hist = jax.lax.scan(step, jnp.zeros(4), u_vec)
        return x_hist

    x_hist = np.array(simulate_all(jnp.array(u_opt)))
    t = np.linspace(0, t_f, N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(t, x_hist[:, 0], label='x1')
    ax1.plot(t, x_hist[:, 1], label='x2')
    ax1.plot(t, x_hist[:, 2], label='x3')
    ax1.plot(t, x_hist[:, 3], label='x4')
    ax1.set_xlabel('t')
    ax1.set_title('Stage 1: states')
    ax1.legend()

    ax2.plot(t, u_opt)
    ax2.axhline( 1, color='k', linestyle='--', linewidth=0.8)
    ax2.axhline(-1, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('t')
    ax2.set_ylabel('u')
    ax2.set_title('Stage 1: optimal control')

    plt.tight_layout()
    plt.show()
