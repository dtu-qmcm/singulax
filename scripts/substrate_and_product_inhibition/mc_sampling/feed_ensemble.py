"""Optimal FEED PROFILES across a kinetic-parameter ensemble.

Companion to sampling_spi.py. That script asks how much PV(tf) moves; this one asks
how much the POLICY that achieves it moves -- the larger and more interesting spread.

Why the hybrid and not Stage 1 alone: the collocation feed chatters on the SV >= 0
bound, so it is not an implementable control. The Stage-2 feed on the singular arc
comes from the feedback law get_Fsin, so it is smooth by construction.

On batching with jax.vmap -- a negative result worth recording. Stage 1 cannot be
batched at all (IPOPT through cyipopt is a C library behind Python callbacks,
outside JAX). Stage 2 is JAX all the way down and vmap does work over the whole
Levenberg-Marquardt solve, giving numerically identical answers. It is just SLOWER:
on a 3-draw group, 1151 s batched against 839 s sequential, a 0.7x "speedup". See
stage2_group for why -- both optimistix and diffrax run their loops in lockstep
across the batch, so the batch costs max(steps) * B where the loop costs sum(steps).
Sequential is therefore the default here, and `time_it=True` reproduces the
comparison for the paper.

One constraint shapes the rest: `arc_u` and `num_segments` are STATIC, so draws are
grouped by the arc structure Stage 1 discovered and each group is handled
separately. That is not a workaround -- different structures are genuinely
different boundary-value problems.
"""
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)   # REQUIRED before any array exists

import diffrax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optimistix as optx

from singulax.substrate_product_inhibition.hybrid import (
    ProblemConfig, solve_stage1, _reconstruct,
)
from singulax.substrate_product_inhibition.kinetics import BALANCED, KineticParams
from singulax.substrate_product_inhibition.odes import vector_field
from singulax.substrate_product_inhibition.Fsin import get_Fsin_unconstrained
from singulax.substrate_product_inhibition.shooting import solve_and_get_residuals_ms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampling_spi import CV, NAMES, params_from_row, sample_X   # noqa: E402

CONFIG = ProblemConfig(F_init=0.5, ls_max_steps=10000)

N = 20
SEED = 1234
HERE = Path(__file__).resolve().parent
_TAG = f"balanced_cv{round(CV * 100):02d}"
FEEDS_NPZ = HERE / f"feed_ensemble_{_TAG}.npz"
FIGURE = HERE / f"feed_ensemble_{_TAG}.png"
N_PLOT = 400                             # points on the common plotting grid


def shootable(seq):
    """Can shooting.py represent this arc structure?

    Its residual folds the terminal batch in as a terminal cost rather than
    integrating it, so it needs the last SHOT arc to be the singular one with a
    batch after it. Anything else is a different boundary-value problem.
    """
    return len(seq) >= 2 and seq[-1] == 0 and seq[-2] == 1


def stage1_ensemble(X, cfg=CONFIG, verbose=True):
    """Stage 1 for every draw, sequentially. IPOPT cannot be batched."""
    out, t0 = [], time.time()
    for i, row in enumerate(X):
        try:
            s1 = solve_stage1(params_from_row(row), cfg)
        except Exception as exc:                                  # noqa: BLE001
            s1 = None
            if verbose:
                print(f"[{i + 1:3d}/{len(X)}] raised {type(exc).__name__}", flush=True)
        if s1 is not None and verbose:
            eta = (time.time() - t0) / (i + 1) * (len(X) - i - 1)
            print(f"[{i + 1:3d}/{len(X)}] PV={s1.PV:8.2f}  {s1.seq}  "
                  f"ok={s1.ok}  ETA {eta / 60:4.1f} min", flush=True)
        out.append(s1)
    return out


def _batched_params(rows):
    """Rows of sampled kinetics -> one KineticParams whose fields are (B,) arrays.

    KineticParams is a NamedTuple, so it is a pytree and vmap maps over it
    field-by-field -- but every field needs the batch axis, including the two
    stoichiometric coefficients, which are identical across draws.
    """
    B = len(rows)
    vals = {n: jnp.asarray([float(r[n]) for r in rows]) for n in NAMES}
    vals["c_mu"] = jnp.full(B, BALANCED.c_mu)
    vals["c_pi"] = jnp.full(B, BALANCED.c_pi)
    return KineticParams(**vals)


def _seed_one(params, lam0, sw, arc_u, cfg):
    """Forward-integrate the arcs from lambda(0) to build this draw's node seed."""
    S_F, V_max, F_min, F_max = cfg.S_F, cfg.V_max, cfg.F_min, cfg.F_max
    x0 = jnp.asarray(cfg.x0)
    fsin_args = (S_F, V_max, F_min, F_max, params)
    vf_sing = lambda t, y, a: vector_field(                       # noqa: E731
        t, y, (S_F, get_Fsin_unconstrained(y, fsin_args), params))
    nodes_t = jnp.concatenate([jnp.array([0.0]), sw])
    y = jnp.concatenate([x0, lam0])
    nodes = [y]
    for i, u in enumerate(arc_u):
        vf = vf_sing if u is None else vector_field
        s = diffrax.diffeqsolve(
            diffrax.ODETerm(vf), diffrax.Tsit5(),
            t0=nodes_t[i], t1=nodes_t[i + 1], y0=y, dt0=0.01,
            args=(S_F, u, params),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            saveat=diffrax.SaveAt(t1=True), max_steps=100000, throw=False)
        y = s.ys[-1]
        nodes.append(y)
    return jnp.concatenate([jnp.stack(nodes).ravel(), sw])


def _solve_one(params, guess, arc_u, cfg):
    """One Stage-2 shoot. Pure JAX -- this is the function vmap batches."""
    args = (len(arc_u), cfg.S_F, cfg.V_max, cfg.F_min, cfg.F_max,
            cfg.t_end, jnp.asarray(cfg.x0), arc_u, params)
    sol = optx.least_squares(
        lambda u, a: solve_and_get_residuals_ms(u, a),
        optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8),
        guess, args=args, max_steps=cfg.ls_max_steps, throw=False)
    res = solve_and_get_residuals_ms(sol.value, args)
    return sol.value, jnp.linalg.norm(res)


def stage2_group(rows, lam0s, sws, arc_u, cfg=CONFIG, batched=False,
                 progress=True, label=""):
    """Stage 2 for one structure-homogeneous group. Returns (Zopt, res_norm, seconds).

    Sequential by default, and that is a measured choice rather than a default left
    unexamined. On a 3-draw group vmap took 1151 s against the loop's 839 s -- 0.7x,
    i.e. slower. Two lockstep effects cause it, and they compound:

      * optimistix runs ONE while-loop for the whole batch, so it iterates until
        every member has converged. Sequentially the work is sum(steps_i); batched
        it is max(steps_i) * B. With uneven step counts -- which is the normal case
        -- the batch does strictly more work.
      * diffrax batches adaptive stepping the same way: every draw pays for every
        other draw's stiff patches.

    On CPU nothing offsets that; vectorising a handful of ODE solves is just more
    arithmetic on one core. vmap would need a GPU, or true process parallelism, to
    pay off. `batched=True` is kept so the comparison can be reproduced.
    """
    params = _batched_params(rows)
    lam0s, sws = jnp.asarray(lam0s), jnp.asarray(sws)

    seed = lambda p, l, s: _seed_one(p, l, s, arc_u, cfg)         # noqa: E731
    solve = lambda p, g: _solve_one(p, g, arc_u, cfg)             # noqa: E731

    if batched:
        guesses = jax.vmap(seed)(params, lam0s, sws)
        t0 = time.time()
        Z, rn = jax.vmap(solve)(params, guesses)
        Z.block_until_ready()
        return np.asarray(Z), np.asarray(rn), time.time() - t0

    B = lam0s.shape[0]
    rows_p = [jax.tree.map(lambda a, i=i: a[i], params) for i in range(B)]
    t0, Zs, rns = time.time(), [], []
    for i, p in enumerate(rows_p):
        ti = time.time()
        g = seed(p, lam0s[i], sws[i])
        z, r = solve(p, g)
        z.block_until_ready()
        Zs.append(z); rns.append(float(r))
        if progress:                       # per draw, so a long group visibly moves
            done, el = i + 1, time.time() - t0
            print(f"      {label}{done:2d}/{B}  ||res|| {float(r):.1e}  "
                  f"{time.time() - ti:5.1f}s   ETA {el / done * (B - done) / 60:4.1f} min",
                  flush=True)
    return np.asarray(jnp.stack(Zs)), np.asarray(rns), time.time() - t0


def run(n=N, seed=SEED, cfg=CONFIG, out=FEEDS_NPZ, time_it=False):
    """Stage 1 sequentially, then Stage 2 per structure group.

    `time_it=True` additionally runs each group through jax.vmap to
    reproduce the batched-vs-sequential comparison. It doubles the runtime,
    so it is off by default.
    """
    X = sample_X(n, seed)
    print(f"=== STAGE 1: {n} collocation solves (sequential -- IPOPT is not JAX) ===")
    s1s = stage1_ensemble(X, cfg)

    groups = {}
    for i, s1 in enumerate(s1s):
        if s1 is None or not s1.ok or not shootable(s1.seq):
            continue
        groups.setdefault(s1.seq, []).append(i)

    n_ok = sum(s1 is not None and s1.ok for s1 in s1s)
    print(f"\n=== STAGE 2: {sum(len(v) for v in groups.values())} of {n_ok} "
          f"converged draws are shootable, in {len(groups)} structure groups ===")

    T_grid = np.linspace(0.0, cfg.t_end, N_PLOT)
    feeds, structs, res_norms, pvs, timings = [], [], [], [], []

    for seq, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        arc_u = tuple(cfg.F_max if s == 2 else cfg.F_min if s == 0 else None
                      for s in seq[:-1])
        rows = [dict(zip(NAMES, X[i])) for i in idx]
        lam0s = [s1s[i].lam0 for i in idx]
        sws = [s1s[i].t_sw[:len(arc_u)] for i in idx]

        tag = "-".join(map(str, seq))
        print(f"   ({tag})  n={len(idx):2d}", flush=True)
        Z, rn, t_loop = stage2_group(rows, lam0s, sws, arc_u, cfg, batched=False)
        line = f"   ({tag})  n={len(idx):2d}  loop {t_loop:7.1f}s"
        if time_it and len(idx) > 1:       # only for the paper's benchmark section
            _, _, t_vmap = stage2_group(rows, lam0s, sws, arc_u, cfg, batched=True)
            line += f"   vmap {t_vmap:7.1f}s   speedup {t_loop / t_vmap:4.1f}x"
            timings.append((seq, len(idx), t_vmap, t_loop))
        print(line + f"   ||res|| {np.min(rn):.1e}-{np.max(rn):.1e}", flush=True)

        # Reconstruction is cheap next to the solve, so it stays a plain loop.
        for k, i in enumerate(idx):
            p = params_from_row(X[i])
            try:
                T, Y, F, _, _ = _reconstruct(jnp.asarray(Z[k]), arc_u, p, cfg)
            except Exception:                                     # noqa: BLE001
                continue
            feeds.append(np.interp(T_grid, T, F))
            structs.append("-".join(map(str, seq)))
            res_norms.append(float(rn[k]))
            pvs.append(float(Y[-1, 2]))

    feeds = np.asarray(feeds)
    np.savez_compressed(out, F=feeds, t=T_grid, structure=np.asarray(structs),
                        res_norm=np.asarray(res_norms), PV=np.asarray(pvs),
                        X=X)
    print(f"\nwrote {out}   ({len(feeds)} profiles)")
    if timings:
        tot_v = sum(t[2] for t in timings); tot_l = sum(t[3] for t in timings)
        print(f"Stage 2 total: loop {tot_l:.1f}s vs vmap {tot_v:.1f}s "
              f"-> {tot_l / tot_v:.2f}x "
              f"({'vmap wins' if tot_v < tot_l else 'vmap LOSES -- see stage2_group'})")
    return feeds, T_grid, np.asarray(structs)


def plot(path=FEEDS_NPZ, figure=FIGURE):
    """The ensemble of optimal feed profiles, one colour per draw."""
    z = np.load(path, allow_pickle=True)
    F, t = z["F"], z["t"]
    # A draw whose shoot did not converge has no optimal feed to plot -- its
    # trajectory is whatever the reconstruction made of an unconverged Zopt.
    # Dropping them is not cherry-picking: the yield is reported, and a
    # non-solution is not a datum.
    good = np.asarray(z["res_norm"]) < CONFIG.res_tol
    if (~good).sum():
        bad = np.asarray(z["res_norm"])[~good]
        print(f"dropping {int((~good).sum())} of {len(good)} draws that did not "
              f"converge (||res|| {bad.min():.1e}-{bad.max():.1e}); "
              f"{int(good.sum())} remain")
    F, structure = F[good], z["structure"][good]
    if len(F) == 0:
        print("no converged draws to plot"); return

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
        "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 600,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    # Colour by the arc structure Stage 1 discovered, most common family first.
    # Draws within a family share a bang/singular/batch pattern and differ only
    # in switch times; draws in different families are qualitatively different
    # policies, which is the thing worth seeing.
    oi = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#0072B2"]
    fams = [s for s, _ in sorted(((s, int((structure == s).sum()))
                                  for s in set(structure.tolist())),
                                 key=lambda kv: -kv[1])]
    colour = {s: oi[k % len(oi)] for k, s in enumerate(fams)}

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    for f, s_ in zip(F, structure):
        ax.plot(t, f, color=colour[s_], lw=1.0, alpha=0.75)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"$F$ (L h$^{-1}$)")
    ax.set_title(f"Optimal feed, {len(F)} sampled parameter sets", loc="left")
    ax.legend(handles=[plt.Line2D([], [], color=colour[s_], lw=2,
                                  label=f"({s_})  n={int((structure == s_).sum())}")
                       for s_ in fams], loc="upper right")
    fig.savefig(figure, dpi=600, bbox_inches="tight")
    print(f"saved {figure}")


if __name__ == "__main__":
    if "--plot" not in sys.argv:         # --plot re-reads the npz without re-solving
        run()
    plot()
