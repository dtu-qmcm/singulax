import jax
import matplotlib.pyplot as plt
import diffrax
import optimistix as optx
from optimistix import BFGS
import jax.numpy as jnp
import numpy as np
from scipy.signal import medfilt
from scipy.sparse import coo_array
from cyipopt import minimize_ipopt

jax.config.update("jax_enable_x64", True)

def F_calc(x, u):
    x1, x2 = x[0], x[1]
    c = jnp.cos(x1)
    s = jnp.sin(x1)
    e = 0.5
    D = 1 - e * c**2
    F2 = -((e * x2**2 * s * c + s + c * u) / D)
    F4 = ((e * s * (x2**2 + c) + u) / D)
    return jnp.array([F2, F4])

def states_ode(x, u):
    x2, x4 = x[1], x[3]
    F2, F4 = F_calc(x, u)
    return jnp.array([x2, F2, x4, F4])

def unpack(z):
    X = z[:nX].reshape(N + 1, nx)
    U = z[nX:]
    return X, U

def objective(z):
    X, _ = unpack(z)
    d = X[-1] - xf
    return 0.5 * jnp.dot(d, d)

# equality constraints: dynamics defects d_k = x_{k+1} - x_k - h f(x_k,u_k) ----
def constraints(z):
    X, U = unpack(z)
    xk = X[:-1]                        # (N, 4)
    xk1 = X[1:]                        # (N, 4)
    f = jax.vmap(states_ode)(xk, U)   # (N, 4)
    return (xk1 - xk - h * f).ravel()  # (4N,)

# sparse Jacobian of the constraints: each defect d_k touches only x_k, x_{k+1},
# u_k, so autodiff the 3 per-step blocks and scatter them into a COO matrix
def _single_defect(xk, xk1, uk):
    return xk1 - xk - h * states_ode(xk, uk)

def jac_defects(z):
    X, U = unpack(jnp.asarray(z))
    A, B, C = _blocks(X[:-1], X[1:], U)
    data = np.concatenate([np.asarray(A).ravel(), np.asarray(B).ravel(), np.asarray(C).ravel()])
    return coo_array((data, (_ROWS, _COLS)), shape=(nx * N, nz))
 

def detect_switching_times(u, tf, eps=0.05, min_singular=15):
    """Switching times of a bang-singular-bang control on N intervals over [0, tf].

    eps          : |u| > 1-eps  == the control is sitting on a bound (+-1)
    min_singular : an interior run longer than this many points is a singular arc
                   (2 switches: enter & exit); a shorter one is a bang-bang
                   switch (1 switch, at the zero crossing).
    """
    u = np.asarray(u)
    n = len(u)
    hh = tf / n
    tc = (np.arange(n) + 0.5) * hh                          # control-sample times
    cls = np.where(u > 1 - eps, 1, np.where(u < -1 + eps, -1, 0))  # +1 / -1 / interior
    cls = medfilt(cls, 5).astype(int)                      # kill 1-2 pt band-edge flicker

    seg = []                                               # contiguous arcs [state, start, end]
    s, i0 = cls[0], 0
    for i in range(1, n):
        if cls[i] != s:
            seg.append((s, i0, i - 1))
            s, i0 = cls[i], i
    seg.append((s, i0, n - 1))

    ts = []
    for s, a, b in seg:
        if s != 0:                                         # bang arc -> no interior switch
            continue
        if (b - a + 1) > min_singular:                     # singular arc: enter + exit
            ts += [tc[a], tc[b]]
        else:                                              # bang-bang: sub-grid zero crossing
            lo, hi = max(a - 1, 0), min(b + 1, n - 1)
            zc = np.where(np.diff(np.sign(u[lo:hi + 1])) != 0)[0]
            if len(zc):
                j = lo + zc[0]
                ts.append(tc[j] + (u[j] / (u[j] - u[j + 1])) * hh)
            else:
                ts.append(tc[(a + b) // 2])
    return np.sort(np.array(ts))


def switching_fn(y):
    x, l = jnp.split(y, 2)
    x1 = x[0]
    c = jnp.cos(x1)
    e = 0.5
    D = 1 - e*c**2
    return (l[3] - c*l[1]) / D

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


def solve_and_get_residuals__ms(guess, args):
    num_segments, num_states = args
    n_nodes = num_segments + 1
    params = guess[:n_nodes * num_states].reshape(n_nodes, num_states)
    switch_times = guess[n_nodes * num_states:]                     # t1..t4 (now optimized)
    shooting_nodes = jnp.concatenate([jnp.array([0.0]), switch_times, jnp.array([tf])])

    # Extract states and costates
    x = params[:, :4] # x1, x2, x3, x4
    l = params[:, 4:]     # l1, l2, l3, l4

    states = jnp.concatenate([x, l], axis=1)

    res = []
    solver_ode = diffrax.Kvaerno5()
    arc_u = [1.0, None, -1.0, 1.0, -1.0]   # control per arc; None = singular arc

    for i in range(num_segments):
        t_start, t_end = shooting_nodes[i], shooting_nodes[i+1]
        s_initial = states[i]

        t_span = diffrax.SaveAt(ts=[t_start, t_end])
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
   
        vfield = vector_field_singular if arc_u[i] is None else vector_field
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(vfield),
            solver=solver_ode,
            t0=t_start,
            t1=t_end,
            y0=s_initial,
            saveat=t_span,
            dt0=0.01,
            args=arc_u[i],
            stepsize_controller=stepsize_controller,
            max_steps=50000,
            throw=False,
            adjoint=diffrax.ForwardMode(),
        )

        s_final_integrated = sol.ys[-1]
        s_final_integrated = jnp.where(jnp.any(~jnp.isfinite(s_final_integrated)),
                                       1e6, s_final_integrated)   # penalize a diverged arc
        s_next_segment_initial = states[i+1]

        defect = s_final_integrated - s_next_segment_initial
        res.append(defect)

    bc_end = jnp.array([
        states[-1][4] - states[-1][0] + xf[0],             
        states[-1][5] - states[-1][1] + xf[1],              
        states[-1][6] - states[-1][2] + xf[2],       
        states[-1][7] - states[-1][3] + xf[3],    
        ])

    bc_start = states[0][:4] - x0    # x(0) = 0
    sw_res = jnp.array([switching_fn(states[k]) for k in range(1, num_segments)])  # sigma = 0
    flat_defects = jnp.concatenate([r for r in res])
    return jnp.concatenate([flat_defects, bc_start, bc_end, sw_res])


# PROBLEM DATA
N = 1000
tf = 4.912
h = tf / N
xf = jnp.array([jnp.pi, 0.0, 0.0, 0.0])
x0 = jnp.zeros(4)
nx = 4 # nx=4 refers to the 4 state variables

# decision-vector packing:  z = [ x_0..x_N (flattened) , u_0..u_{N-1} ] ----
nX = (N + 1) * nx          # states
nU = N                     # controls (control sits between the points)
nz = nX + nU

obj_jit = jax.jit(objective)
grad_jit = jax.jit(jax.grad(objective))
con_jit = jax.jit(constraints)

# start by declaring every variable free (+-inf), then overwrite only the entries
# that actually have limits:
lb = np.full(nz, -np.inf)
ub = np.full(nz, np.inf)
lb[:nx] = ub[:nx] = 0.0        # fix x_0 = 0
lb[nX:] = -1.0                 # u lower
ub[nX:] = 1.0                  # u upper

_blocks = jax.jit(jax.vmap(jax.jacrev(_single_defect, argnums=(0, 1, 2))))

_k = np.arange(N)[:, None, None]
_i = np.arange(nx)[None, :, None]
_j = np.arange(nx)[None, None, :]
_rowsAB = np.broadcast_to(4 * _k + _i, (N, nx, nx)).ravel()
_colsA  = np.broadcast_to(4 * _k + _j, (N, nx, nx)).ravel()             # cols of x_k
_colsB  = np.broadcast_to(4 * (_k + 1) + _j, (N, nx, nx)).ravel()       # cols of x_{k+1}
_rowsC  = (4 * np.arange(N)[:, None] + np.arange(nx)[None, :]).ravel()
_colsC  = np.broadcast_to(nX + np.arange(N)[:, None], (N, nx)).ravel()  # col of u_k
_ROWS = np.concatenate([_rowsAB, _rowsAB, _rowsC])
_COLS = np.concatenate([_colsA, _colsB, _colsC])


# ---STAGE 1: DIRECT APPROACH---
if __name__ == "__main__":
    # states at 0, controls at 0
    z0 = np.zeros(nz)

    constraints = [{
        "type": "eq",
        "fun": lambda z: np.asarray(con_jit(jnp.asarray(z))),
        "jac": jac_defects,
    }]

    print("=== Simultaneous NLP (states + controls), multipliers = costates ===")
    result = minimize_ipopt(
        fun=lambda z: float(obj_jit(jnp.asarray(z))),
        jac=lambda z: np.asarray(grad_jit(jnp.asarray(z))),
        x0=z0,
        bounds=list(zip(lb, ub)),
        constraints=constraints,
        options={"max_iter": 500, "tol": 1e-8, "print_level": 5},
    )

    z_opt = result.x
    X_opt, U_opt = unpack(jnp.asarray(z_opt))
    X_opt = np.asarray(X_opt)
    U_opt = np.asarray(U_opt)

    # the multipliers for the whole trajectory
    mult_g = np.asarray(result.info["mult_g"]).reshape(N, nx)   # mu_k, k=0..N-1
    lam = -mult_g                                               # continuous costate ~ -mult_g
    lam0 = lam[0]
    print("lambda(0) guess =", lam0)

    # switching times from the bang-singular structure of the control
    ts_guess = detect_switching_times(U_opt, tf)
    print("switching times =", ts_guess.round(4))

# ---STAGE 2: INDIRECT APPROACH---
    arc_u = [1.0, None, -1.0, 1.0, -1.0]
    shooting_nodes_guess = jnp.concatenate([jnp.array([0.0]), jnp.asarray(ts_guess), jnp.array([tf])])

    def _integ(y0, t0, t1, u):
        vf = vector_field_singular if u is None else vector_field
        s = diffrax.diffeqsolve(diffrax.ODETerm(vf), diffrax.Kvaerno5(),
            t0=t0, t1=t1, y0=y0, dt0=0.01, args=u,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            saveat=diffrax.SaveAt(t1=True), max_steps=50000,
            throw=False, adjoint=diffrax.ForwardMode())
        return s.ys[0]

    y = jnp.concatenate([x0, jnp.asarray(lam0)])
    node_list = [y]
    for i in range(5):                                   # build all 6 node seeds
        y = _integ(y, shooting_nodes_guess[i], shooting_nodes_guess[i + 1], arc_u[i])
        node_list.append(y)
    guess0 = jnp.concatenate([jnp.stack(node_list).ravel(), jnp.asarray(ts_guess)])  # 52

    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
    sol = optx.least_squares(solve_and_get_residuals__ms, solver, guess0,
                             args=(5, 8), max_steps=2000, throw=False)
    Zopt = sol.value
    print("λ(0) optimized =", np.array(Zopt[4:8]))
    print("optimized switching times =", np.array(Zopt[48:]).round(4))
    print("final residual norm =", float(jnp.linalg.norm(solve_and_get_residuals__ms(Zopt, (5, 8)))))

    # STAGE 1 PLOT
    t = np.linspace(0, tf, N + 1)         # state grid (N+1 points)
    t_u = np.linspace(0, tf, N)           # control grid
    t_lam = t[1:]                         # mult_g[k] lives at t_{k+1}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    ax1.plot(t, X_opt[:, 0], "k-", label="x1")
    ax1.plot(t, X_opt[:, 1], "r--", label="x2")
    ax1.plot(t, X_opt[:, 2], "g:", label="x3")
    ax1.plot(t, X_opt[:, 3], "b--", label="x4")
    ax1.set_xlabel("t"); ax1.set_title("States (x)"); ax1.legend()

    ax2.plot(t_u, U_opt, "r")
    ax2.axhline(1, color="k", ls="--", lw=0.8)
    ax2.axhline(-1, color="k", ls="--", lw=0.8)
    ax2.set_xlabel("t"); ax2.set_ylabel("u"); ax2.set_title("Control (u)")
    for tsw in ts_guess:
        ax2.axvline(tsw, color="b", ls=":", lw=1)

    ax3.plot(t_lam, lam[:, 0], "g-", label="λ1")
    ax3.plot(t_lam, lam[:, 1], "r--", label="λ2")
    ax3.plot(t_lam, lam[:, 2], "b:", label="λ3")
    ax3.plot(t_lam, lam[:, 3], "k--", label="λ4")
    ax3.set_xlabel("t"); ax3.set_title("Costates (= -mult_g)"); ax3.legend()
    
    fig.suptitle("Stage 1 Optimization Results", fontsize = 16)
    plt.tight_layout()
    plt.savefig("lagrange_multiplier_stage1.png", dpi=110)
    print("\nsaved plot -> lagrange_multiplier_stage1.png")

    # STAGE 2 PLOT: reconstruct dense trajectories from the converged solution
    nodes_opt = Zopt[:6 * 8].reshape(6, 8)
    ts_opt = Zopt[48:]
    node_t = jnp.concatenate([jnp.array([0.0]), ts_opt, jnp.array([tf])])

    def _integ_dense(y0, t0, t1, u, npts=200):
        vf = vector_field_singular if u is None else vector_field
        s = diffrax.diffeqsolve(diffrax.ODETerm(vf), diffrax.Kvaerno5(),
            t0=t0, t1=t1, y0=y0, dt0=0.01, args=u,
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
            saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, npts)),
            max_steps=100000, throw=False, adjoint=diffrax.ForwardMode())
        return np.asarray(s.ts), np.asarray(s.ys)

    ts_seg, ys_seg, u_seg = [], [], []
    for i in range(5):
        ti, yi = _integ_dense(nodes_opt[i], node_t[i], node_t[i + 1], arc_u[i])
        if arc_u[i] is None:                                  # singular arc -> u_sing
            ui = np.asarray(jax.vmap(lambda y: u_sing(y[:4], y[4:], 0.0))(jnp.asarray(yi)))
        else:                                                 # bang arc -> +-1
            ui = np.full(yi.shape[0], arc_u[i])
        ts_seg.append(ti); ys_seg.append(yi); u_seg.append(ui)
    t2 = np.concatenate(ts_seg)
    Y2 = np.concatenate(ys_seg, axis=0)
    u2 = np.concatenate(u_seg)
    sigma2 = np.asarray(jax.vmap(switching_fn)(jnp.asarray(Y2)))
    ts_opt_np = np.array(ts_opt)

    fig2, ((b1, b2), (b3, b4)) = plt.subplots(2, 2, figsize=(14, 8))

    b1.plot(t2, Y2[:, 0], "k-", label="x1"); b1.plot(t2, Y2[:, 1], "r--", label="x2")
    b1.plot(t2, Y2[:, 2], "g:", label="x3"); b1.plot(t2, Y2[:, 3], "b--", label="x4")
    b1.set_title("States (x)"); b1.set_xlabel("t"); b1.legend()

    b2.plot(t2, Y2[:, 4], "g-", label="λ1"); b2.plot(t2, Y2[:, 5], "r--", label="λ2")
    b2.plot(t2, Y2[:, 6], "b:", label="λ3"); b2.plot(t2, Y2[:, 7], "k--", label="λ4")
    b2.set_title("Costates (λ)"); b2.set_xlabel("t"); b2.legend()

    b3.plot(t2, u2, "r")
    b3.axhline(1, color="k", ls="--", lw=0.8); b3.axhline(-1, color="k", ls="--", lw=0.8)
    b3.set_ylim(-1.3, 1.3); b3.set_title("Control (u)"); b3.set_xlabel("t"); b3.set_ylabel("u")

    b4.plot(t2, sigma2, "m"); b4.axhline(0, color="k", ls="--", lw=0.8)
    b4.set_title("Switching function σ"); b4.set_xlabel("t")

    for tsw in ts_opt_np:                                     # mark the 4 switching times
        b3.axvline(tsw, color="b", ls=":", lw=1)
        b4.axvline(tsw, color="b", ls=":", lw=1)

    fig2.suptitle("Stage 2 Optimization Results", fontsize=16)

    plt.tight_layout()
    plt.savefig("lagrange_multiplier_stage2.png", dpi=110)
    print("saved plot -> lagrange_multiplier_stage2.png")
