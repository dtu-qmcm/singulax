import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from cyipopt import minimize_ipopt
import diffrax

jax.config.update("jax_enable_x64", True)

# ---- FUNCTIONS ----

def F_calc(x, args):
    x1, x2 = x[0], x[1]
    u = args
    c = jnp.cos(x1)
    s = jnp.sin(x1)
    e = 0.5
    D  = 1-e*c**2

    F2 = -((e*x2**2*s*c + s + c*u)/D)
    F4 = ((e*s*(x2**2 + c) + u)/D)

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

def switching_fn(y):
    x, l = jnp.split(y, 2)
    x1 = x[0]
    c = jnp.cos(x1)
    e = 0.5
    D = 1 - e*c**2
    return (-c*l[1] + l[3]) / D

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

def vector_field(t, y, args):
    x, l = jnp.split(y, 2)
    dx_dt = states_ode(x, args)
    dl_dt = costate_ode(x, args, l)
    return jnp.concatenate([dx_dt, dl_dt])

def vector_field_singular(t, y, args):
    x, l = y[:4], y[4:]
    u = u_sing(x, l, t)              # recomputed at each step
    dxdt = states_ode(x, u)
    dldt = costate_ode(x, u, l)
    return jnp.concatenate([dxdt, dldt])

def simulate(u_vec):
    def step(x, u):
        dx = states_ode(x, u)
        return x + h * dx, None
    x_final, _ = jax.lax.scan(step, jnp.zeros(4), u_vec)
    return x_final

def objective(u_vec, args):
    x_final = simulate(u_vec)
    diff = x_final - xf
    return 0.5 * jnp.dot(diff, diff)

def simulate_all(u_vec):
    def step(x, u):
        dx = states_ode(x, u)
        x_next = x + h * dx
        return x_next, x_next
    _, x_hist = jax.lax.scan(step, jnp.zeros(4), u_vec)
    return x_hist

def J_from_x0(x0):
    def step(x, u): return x + h * states_ode(x, u), None
    x_final, _ = jax.lax.scan(step, x0, u_opt_jnp)
    return 0.5 * jnp.dot(x_final - xf, x_final - xf)

def solve_and_get_residuals(guess, args):
    l0_guess = guess[:4]
    t1_guess, t2_guess, t3_guess, t4_guess = guess[4], guess[5], guess[6], guess[7]
    x0, xf, tf = args
    y0 = jnp.concatenate([x0, l0_guess])
    
    solution_to_t1 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Kvaerno5(),
        t0=0., t1=t1_guess, dt0=0.01, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        args=1.0,
        max_steps=10000,
    )
    y_at_t1 = solution_to_t1.ys[0]

    solution_to_t2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field_singular),
        diffrax.Kvaerno5(),
        t0=t1_guess, t1=t2_guess, dt0=0.01, y0=y_at_t1,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        args=None,
        max_steps=10000,
    )
    y_at_t2 = solution_to_t2.ys[0]

    solution_to_t3 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Kvaerno5(),
        t0=t2_guess, t1=t3_guess, dt0=0.01, y0=y_at_t2,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        args=-1.0,
        max_steps=10000,
    )
    y_at_t3 = solution_to_t3.ys[0]

    solution_to_t4 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Kvaerno5(),
        t0=t3_guess, t1=t4_guess, dt0=0.01, y0=y_at_t3,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        args=1.0,
        max_steps=10000,
    )
    y_at_t4 = solution_to_t4.ys[0]

    solution_to_tf = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Kvaerno5(),
        t0=t4_guess, t1=tf, dt0=0.01, y0=y_at_t4,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        args=-1.0,
        max_steps=10000,
    )
    y_at_tf = solution_to_tf.ys[0]

    # Penalize diverged integrations
    y_at_t1 = jnp.where(jnp.any(jnp.isnan(y_at_t1)), jnp.full(8, 1e6), y_at_t1)
    y_at_t2 = jnp.where(jnp.any(jnp.isnan(y_at_t2)), jnp.full(8, 1e6), y_at_t2)
    y_at_t3 = jnp.where(jnp.any(jnp.isnan(y_at_t3)), jnp.full(8, 1e6), y_at_t3)
    y_at_t4 = jnp.where(jnp.any(jnp.isnan(y_at_t4)), jnp.full(8, 1e6), y_at_t4)
    y_at_tf = jnp.where(jnp.any(jnp.isnan(y_at_tf)), jnp.full(8, 1e6), y_at_tf)
    
    x_at_tf, l_at_tf = jnp.split(y_at_tf, 2)
    
    return jnp.concatenate([
        jnp.array([
            switching_fn(y_at_t1),
            switching_fn(y_at_t2),
            switching_fn(y_at_t3),
            switching_fn(y_at_t4),
        ]),
        x_at_tf - xf,
    ])

@jax.jit
def loss_fn(guess, args):
    res = solve_and_get_residuals(guess, args)
    return jnp.sum(res**2)

# ----STAGE 1: DIRECT APPROACH----
N = 1000
t_f = 4.912
h = t_f / N
xf = jnp.array([jnp.pi, 0.0, 0.0, 0.0])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    obj_jit  = jax.jit(lambda u: objective(u, None))
    grad_jit = jax.jit(jax.grad(lambda u: objective(u, None)))

    result = minimize_ipopt(
        fun    = lambda u: float(obj_jit(jnp.array(u))),
        jac    = lambda u: np.array(grad_jit(jnp.array(u))),
        x0     = np.zeros(N),
        bounds = [(-1.0, 1.0)] * N,
        options = {'max_iter': 300, 'tol': 1e-6},
    )
    u_opt = result.x
    u_opt_jnp = jnp.array(u_opt)

    print("x(tf) =", np.array(simulate(jnp.array(u_opt))))
    print("J     =", result.fun)
    lam0 = jax.grad(J_from_x0)(jnp.zeros(4))
    print("λ(0) =", np.array(lam0))

    x_hist = np.array(simulate_all(jnp.array(u_opt)))
    t = np.linspace(0, t_f, N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4))

    ax1.plot(t, x_hist[:, 0], color='black',  linestyle='-',   label='x1')
    ax1.plot(t, x_hist[:, 1], color='red',    linestyle='--',  label='x2')
    ax1.plot(t, x_hist[:, 2], color='green',  linestyle=':',   label='x3')
    ax1.plot(t, x_hist[:, 3], color='blue',   linestyle='--',  label='x4')
    ax1.set_xlabel('t')
    ax1.set_title('Stage 1: states')
    ax1.legend()

    ax2.plot(t, u_opt, color = 'red')
    ax2.axhline( 1, color='k', linestyle='--', linewidth=0.8)
    ax2.axhline(-1, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('t')
    ax2.set_ylabel('u')
    ax2.set_title('Stage 1: optimal control')

    plt.tight_layout()
    plt.show()

#----STAGE 2: INDIRECT APPROACH----
