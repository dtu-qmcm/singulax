"""Two-stage hybrid solve of the substrate + product inhibition fed-batch problem,
as a callable taking a kinetic parameter set.

Stage 1 (direct): a forward-Euler collocation NLP (IPOPT) that gives an approximate
trajectory -- enough to read off the bang/singular/batch structure and to seed the
initial costate lambda(0) from the constraint multipliers.

Stage 2 (indirect): PMP multiple shooting, initialized from Stage 1, that refines the
switch times and adjoints to high accuracy; the terminal batch enters as a terminal
condition (V = V_max at the last node) rather than as a shooting segment.

`params` defaults to DEFAULT_PARAMS, which is the carbon/redox-BALANCED model --
pass REFERENCE explicitly to reproduce the published Shin & Lim results.

Nothing here prints unless `verbose=True`, and nothing raises on a failed draw: a
solve that does not converge comes back with `ok=False` and a `status` string, so a
sampling loop can record it and carry on.
"""
from typing import NamedTuple, Optional
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx
from cyipopt import minimize_ipopt

from singulax.collocation import (
    CollocParams, unpack, objective, constraints, jac_defects, detect_switches,
)
from singulax.substrate_product_inhibition.kinetics import DEFAULT_PARAMS, REFERENCE
from singulax.substrate_product_inhibition.odes import states_ode, vector_field
from singulax.substrate_product_inhibition.Fsin import get_Fsin_unconstrained
from singulax.substrate_product_inhibition.shooting import solve_and_get_residuals_ms


class ProblemConfig(NamedTuple):
    """Operating data and solver settings -- everything that is NOT kinetics."""
    S_F: float = 150.0              # feed substrate concentration [g/L]
    V_max: float = 10.0             # tank volume [L]
    F_max: float = 2.0              # max feed rate [L/h]
    F_min: float = 0.0              # min feed rate [L/h]
    t_end: float = 39.5             # final time [h]
    x0: tuple = (0.1, 0.001, 0.0, 1.0)   # [XV, SV, PV, V] at t = 0
    N: int = 400                    # collocation intervals
    F_init: float = 0.5             # constant feed used to build the Stage-1 seed
    ipopt_max_iter: int = 3000
    ipopt_tol: float = 1e-6
    ls_max_steps: int = 2000        # Levenberg-Marquardt steps in Stage 2
    res_tol: float = 1e-6           # ||residual|| below this counts as converged
    defect_tol: float = 1e-4        # Stage-1 ||defects|| below this counts as converged


DEFAULT_CONFIG = ProblemConfig()


class Stage1Result(NamedTuple):
    """Collocation output: the trajectory, the arc structure and the costate seed."""
    X: np.ndarray                   # (N+1, 4) states
    U: np.ndarray                   # (N,) feed
    t_grid: np.ndarray              # (N+1,)
    t_sw: np.ndarray                # interior switch times
    seq: tuple                      # regime sequence: 2 = F_max, 1 = singular,
                                    # 0 = F_min/batch
    lam0: np.ndarray                # lambda(0) seed from the constraint multipliers
    PV: float                       # PV(tf) from the collocation trajectory
    defect_norm: float              # ||forward-Euler defects|| at the solution
    ipopt_message: str              # why IPOPT stopped -- see `ok` below
    ok: bool                        # FEASIBLE (defects small), not provably optimal:
                                    # a draw that hit max_iter passes this test, so
                                    # check ipopt_message before trusting its PV


class HybridResult(NamedTuple):
    """What one parameter set produced. `PV` and `ok` are what a sampling loop needs."""
    PV: float                       # PV(tf); nan when the solve failed
    ok: bool                        # both stages converged with a usable arc structure
    status: str                     # "ok", or why not
    structure: tuple                # Stage-1 arc sequence actually found
    res_norm: float                 # ||Stage-2 residual||
    switch_times: np.ndarray
    stage1: Optional[Stage1Result]
    T: Optional[np.ndarray]         # reconstructed trajectory (for plotting)
    Y: Optional[np.ndarray]
    F: Optional[np.ndarray]
    Zopt: Optional[np.ndarray]
    arc_u: Optional[tuple]


def _sizes(cfg):
    nx = 4
    h = cfg.t_end / cfg.N
    nX = (cfg.N + 1) * nx
    return nx, h, nX, nX + cfg.N


def _require_x64():
    """float32 does not fail loudly here -- it just makes every solve miss tolerance,
    so check once rather than let a sampling run return a column of nan."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "singulax needs float64: call jax.config.update('jax_enable_x64', True) "
            "before creating any array. In float32 the shooting residuals cannot "
            "reach their convergence tolerance and every solve fails.")


def solve_stage1(params=DEFAULT_PARAMS, cfg=DEFAULT_CONFIG, verbose=False):
    """Collocation NLP. Returns the trajectory, the arc structure and the
    lambda(0) seed."""
    _require_x64()
    nx, h, nX, nz = _sizes(cfg)
    x0 = jnp.asarray(cfg.x0)
    f_p = partial(states_ode, params=params)
    p = CollocParams(N=cfg.N, nx=nx, h=h, nX=nX, nz=nz, S_F=cfg.S_F, f=f_p)

    obj_jit = jax.jit(lambda z: objective(z, p))
    grad_jit = jax.jit(jax.grad(lambda z: objective(z, p)))
    con_jit = jax.jit(lambda z: constraints(z, p))

    def _sim(x, _):
        """Forward-Euler rollout at constant feed F_init; produces the
        Stage-1 NLP seed z0."""
        return x + h * states_ode(x, cfg.F_init, cfg.S_F, params), x
    _, X_hist = jax.lax.scan(_sim, x0, None, length=cfg.N)
    z0 = np.concatenate([np.asarray(jnp.concatenate([x0[None, :], X_hist])).ravel(),
                         np.full(cfg.N, cfg.F_init)])

    lb = np.full(nz, -np.inf); ub = np.full(nz, np.inf)
    lb[:nx] = ub[:nx] = np.asarray(x0)
    lb[1 + 4 * np.arange(cfg.N + 1)] = 0.0            # SV >= 0
    ub[3 + 4 * np.arange(cfg.N + 1)] = cfg.V_max      # V <= V_max
    lb[nX:] = cfg.F_min; ub[nX:] = cfg.F_max

    cons = [{"type": "eq",
             "fun": lambda z: np.asarray(con_jit(jnp.asarray(z))),
             "jac": lambda z: jac_defects(z, p)}]
    result = minimize_ipopt(
        fun=lambda z: float(obj_jit(jnp.asarray(z))),
        jac=lambda z: np.asarray(grad_jit(jnp.asarray(z))),
        x0=z0, bounds=list(zip(lb, ub)), constraints=cons,
        options={"max_iter": cfg.ipopt_max_iter, "tol": cfg.ipopt_tol, "print_level": 0})

    X_opt, U_opt = unpack(jnp.asarray(result.x), p)
    X_opt = np.asarray(X_opt); U_opt = np.asarray(U_opt)
    mult_g = np.asarray(result.info["mult_g"]).reshape(cfg.N, nx)
    lam0 = mult_g[0, :]

    t_grid = np.linspace(0, cfg.t_end, cfg.N + 1)
    t_sw, seq = detect_switches(U_opt, t_grid[:-1], cfg.F_min, cfg.F_max)

    # IPOPT's own flag is too strict here: on harder kinetics it reports
    # "maximum iterations exceeded" while sitting on a perfectly good point
    # (||defects|| ~ 1e-7). Judge the solution, not the exit code.
    defect_norm = float(np.linalg.norm(np.asarray(con_jit(jnp.asarray(result.x)))))
    ok = bool(np.all(np.isfinite(X_opt)) and defect_norm < cfg.defect_tol)

    # That test is FEASIBILITY, not optimality, so keep IPOPT's own verdict: a
    # point abandoned at max_iter satisfies the defects while possibly sitting
    # short of the optimum, which biases its PV low. Without this field a
    # sampling run cannot tell those draws apart afterwards.
    msg = getattr(result, "message", "")
    ipopt_message = msg.decode() if isinstance(msg, bytes) else str(msg)

    if verbose:
        print("l0 seed:", lam0)
        print("feed-regime switches =", np.round(t_sw, 3))
        print("regime sequence      =", list(seq),
              "  (2=Fmax bang, 1=singular, 0=Fmin/batch)")
        print(f"||defects|| = {defect_norm:.2e}   IPOPT: {ipopt_message}")

    return Stage1Result(X=X_opt, U=U_opt, t_grid=t_grid, t_sw=t_sw, seq=tuple(seq),
                        lam0=lam0, PV=float(X_opt[-1, 2]),
                        defect_norm=defect_norm, ipopt_message=ipopt_message, ok=ok)


def _reconstruct(Zopt, arc_u, params, cfg):
    """Re-integrate the optimal arcs plus the terminal batch into one continuous
    trajectory (T, Y, F) -- the same reconstruction the Stage-2 figure plots."""
    num_segments = len(arc_u)
    n_nodes = num_segments + 1
    S_F, V_max, F_min, F_max, tf = cfg.S_F, cfg.V_max, cfg.F_min, cfg.F_max, cfg.t_end
    nodes_opt = np.asarray(Zopt[:n_nodes * 8]).reshape(n_nodes, 8)
    sw_eff = np.maximum.accumulate(
        np.clip(np.asarray(Zopt[n_nodes * 8:]), 1e-2, tf - 1e-2))
    nodes_t = np.concatenate([[0.0], sw_eff])
    sc2 = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-12)
    fsin_args = (S_F, V_max, F_min, F_max, params)
    vf_sing = lambda t, y, a: vector_field(
        t, y, (S_F, get_Fsin_unconstrained(y, fsin_args), params))

    seg_t, seg_y, seg_F = [], [], []
    for i in range(num_segments):
        u = arc_u[i]
        vf = vf_sing if u is None else vector_field
        t1 = max(nodes_t[i + 1], nodes_t[i] + 1e-9)
        s = diffrax.diffeqsolve(
            diffrax.ODETerm(vf), diffrax.Tsit5(), t0=nodes_t[i], t1=t1,
            y0=jnp.asarray(nodes_opt[i]), dt0=0.01, args=(S_F, u, params),
            saveat=diffrax.SaveAt(ts=jnp.linspace(nodes_t[i], t1, 60)),
            stepsize_controller=sc2, max_steps=100000, throw=False)
        yi = np.asarray(s.ys)
        seg_t.append(np.asarray(s.ts)); seg_y.append(yi)
        seg_F.append(
            np.asarray(jax.vmap(lambda y: get_Fsin_unconstrained(y, fsin_args))(
                jnp.asarray(yi)))
            if u is None else np.full(len(yi), u))

    sb = diffrax.diffeqsolve(diffrax.ODETerm(vector_field), diffrax.Tsit5(),
        t0=nodes_t[-1], t1=tf, y0=jnp.asarray(nodes_opt[-1]), dt0=0.01,
        args=(S_F, 0.0, params),
        saveat=diffrax.SaveAt(ts=jnp.linspace(nodes_t[-1], tf, 100)),
        stepsize_controller=sc2, max_steps=100000, throw=False)
    tb, yb = np.asarray(sb.ts), np.asarray(sb.ys)

    T = np.concatenate(seg_t + [tb])
    Y = np.concatenate(seg_y + [yb], axis=0)
    F = np.concatenate(seg_F + [np.zeros(len(tb))])
    return T, Y, F, nodes_opt, sw_eff


def solve_hybrid(params=DEFAULT_PARAMS, cfg=DEFAULT_CONFIG, verbose=False, stage1=None):
    """Run both stages for one kinetic parameter set.

    Returns a HybridResult; `PV` is nan and `ok` is False when a stage fails or the
    Stage-1 arc structure cannot be shot (fewer than three interior switches). Pass a
    precomputed `stage1` to reuse a collocation solve.
    """
    S_F, V_max, F_min, F_max, tf = cfg.S_F, cfg.V_max, cfg.F_min, cfg.F_max, cfg.t_end
    _require_x64()
    x0 = jnp.asarray(cfg.x0)
    blank = dict(switch_times=np.array([]), T=None, Y=None, F=None, Zopt=None, arc_u=None)

    if verbose:
        print("=== STAGE 1: collocation (maximise PV) ===")
    try:
        s1 = stage1 if stage1 is not None else solve_stage1(params, cfg, verbose)
    except Exception as exc:                                     # IPOPT / trace failure
        return HybridResult(PV=np.nan, ok=False,
                            status=f"stage1 raised: {type(exc).__name__}",
                            structure=(), res_norm=np.nan, stage1=None, **blank)
    if not s1.ok:
        return HybridResult(PV=np.nan, ok=False, status="stage1 did not converge",
                            structure=s1.seq, res_norm=np.nan, stage1=s1, **blank)
    # The residual in shooting.py is square for ANY number of segments; what it
    # genuinely requires is that the LAST shot arc be the singular one and that a
    # terminal batch follow it (that batch enters as a terminal cost, not a
    # segment). So shoot every arc except the trailing batch.
    if len(s1.seq) < 3 or s1.seq[-1] != 0 or s1.seq[-2] != 1:
        return HybridResult(PV=np.nan, ok=False,
                            status=f"arc structure not shootable: {s1.seq} "
                                   f"(need ... -> singular -> batch)",
                            structure=s1.seq, res_norm=np.nan, stage1=s1, **blank)

    if verbose:
        print("=== STAGE 2: multiple shooting ===")
    arc_u = tuple(F_max if s == 2 else F_min if s == 0 else None for s in s1.seq[:-1])
    num_segments = len(arc_u)
    n_nodes = num_segments + 1
    sw = jnp.asarray(s1.t_sw[:num_segments])
    nodes_guess = jnp.concatenate([jnp.array([0.0]), sw])
    fsin_args = (S_F, V_max, F_min, F_max, params)

    def _integ(y0, t0, t1, u):
        """Integrate one arc [t0, t1] to build a multiple-shooting node seed."""
        vf_singular = lambda t, y, a: vector_field(
            t, y, (S_F, get_Fsin_unconstrained(y, fsin_args), params))
        vf = vf_singular if u is None else vector_field
        s = diffrax.diffeqsolve(
            diffrax.ODETerm(vf), diffrax.Tsit5(),
            t0=t0, t1=t1, y0=y0, dt0=0.01, args=(S_F, u, params),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            saveat=diffrax.SaveAt(t1=True), max_steps=100000, throw=False)
        return s.ys[-1]

    try:
        y = jnp.concatenate([x0, jnp.asarray(s1.lam0)])
        node_list = [y]
        for i in range(num_segments):
            y = _integ(y, nodes_guess[i], nodes_guess[i + 1], arc_u[i])
            node_list.append(y)
        guess0 = jnp.concatenate([jnp.stack(node_list).ravel(), sw])

        ms_args = (num_segments, S_F, V_max, F_min, F_max, tf, x0, arc_u, params)
        solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
        sol = optx.least_squares(
            lambda u, a: solve_and_get_residuals_ms(u, a), solver, guess0,
                                 args=ms_args, max_steps=cfg.ls_max_steps, throw=False)
        Zopt = sol.value
        res_final = solve_and_get_residuals_ms(Zopt, ms_args)
        res_norm = float(jnp.linalg.norm(res_final))
        T, Y, F, nodes_opt, sw_eff = _reconstruct(Zopt, arc_u, params, cfg)
    except Exception as exc:
        return HybridResult(PV=np.nan, ok=False,
                            status=f"stage2 raised: {type(exc).__name__}",
                            structure=s1.seq, res_norm=np.nan, stage1=s1, **blank)

    PV = float(Y[-1, 2])
    if not np.isfinite(res_norm):
        status = "stage2 residual is not finite"
    elif res_norm >= cfg.res_tol:
        status = f"stage2 residual {res_norm:.3e} > {cfg.res_tol:g}"
    elif not np.isfinite(PV):
        status = "stage2 residual converged but the reconstruction returned nan"
    else:
        status = "ok"
    converged = status == "ok"
    if verbose:
        ns8 = num_segments * 8
        print("Stage 2 ||res|| =", res_norm)
        print("Stage 2 switch times =", np.round(np.asarray(Zopt[n_nodes * 8:]), 3))
        print("  ||defects|| =", round(float(jnp.linalg.norm(res_final[:ns8])), 4),
              " ||bc_start|| =", round(float(jnp.linalg.norm(res_final[ns8:ns8 + 4])), 4),
              " ||terminal|| =",
              round(float(jnp.linalg.norm(res_final[ns8 + 4:ns8 + 7])), 4),
              " V-Vmax =", round(float(res_final[ns8 + 7]), 4))
        print("  phi / phidot =", np.round(np.asarray(res_final[ns8 + 8:]), 5))
        # The published 764.47 g is only a benchmark for the REFERENCE
        # coefficients; printing it next to a balanced PV invites the comparison
        # it does not support.
        is_ref = (params.c_mu, params.c_pi) == (REFERENCE.c_mu, REFERENCE.c_pi)
        bench = "   (Shin & Lim: 764.47 g)" if is_ref else "   (balanced model)"
        print("Final amount of product PV(tf) =", round(PV, 2), "g" + bench)
        print("batch lambda(tf) =", np.round(Y[-1, 4:], 4),
              "  (expect [0, 0, 1, eta], eta free)")

    return HybridResult(PV=PV, ok=bool(converged), status=status, structure=s1.seq,
                        res_norm=res_norm, switch_times=np.asarray(sw_eff), stage1=s1,
                        T=T, Y=Y, F=F, Zopt=np.asarray(Zopt), arc_u=arc_u)
