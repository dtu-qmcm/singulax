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
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# One compute thread per process, and ONE LLVM codegen partition. These must be
# set BEFORE jax is imported: XLA reads XLA_FLAGS once, at initialisation, so
# setting them inside run() reached only the spawned children -- the parent,
# which runs Stage 1 and the screen, kept the full pools every time.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("XLA_FLAGS",
                      "--xla_cpu_multi_thread_eigen=false "
                      "--xla_cpu_parallel_codegen_split_count=1")

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

CONFIG = ProblemConfig(F_init=0.5, ls_max_steps=10000, ipopt_max_iter=20000)

N = 100
SEED = 1234
HERE = Path(__file__).resolve().parent
_TAG = f"balanced_cv{round(CV * 100):02d}"
FEEDS_NPZ = HERE / f"feed_ensemble_{_TAG}.npz"
FIGURE = HERE / f"feed_ensemble_{_TAG}.png"
NOMINAL_NPZ = HERE / "feed_nominal_balanced.npz"   # one solve, cached
SCREEN_CSV = HERE / f"screening_{_TAG}.csv"        # per-draw exclusion record
FSING_MIN = 1.0e-3
SEED_OVERFLOW = 1.0e6
N_PLOT = 400                             # points on the common plotting grid

#: Override with run(workers=N) if you have measured your own machine.
GB_PER_WORKER = 5.0

def _default_workers():
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    try:                                   # physical cores, not SMT threads
        cores = len({ln.split(":")[1].strip()
                     for ln in open("/proc/cpuinfo")
                     if ln.startswith("core id")}) or (os.cpu_count() or 2) // 2
    except OSError:
        cores = (os.cpu_count() or 2) // 2
    return max(1, min(cores - 2, int(mem_gb // GB_PER_WORKER))), mem_gb, cores


WORKERS, _MEM_GB, _CORES = _default_workers()


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



def _stage2_draw(job):
    """Seed, shoot and reconstruct ONE draw. Module-level so it can be pickled.

    Workers are spawned, not forked -- JAX is not fork-safe -- so each re-imports
    this module and gets jax_enable_x64 from the top of the file.
    """
    i, row, seq, t_sw, lam0 = job
    cfg = CONFIG
    arc_u = tuple(cfg.F_max if v == 2 else cfg.F_min if v == 0 else None
                  for v in seq[:-1])
    M = len(arc_u)
    p = params_from_row(row)
    t0 = time.time()
    g = _seed_one(p, jnp.asarray(lam0), jnp.asarray(t_sw[:M]), arc_u, cfg)
    Z, rn = _solve_one(p, g, arc_u, cfg)
    try:
        T, Y, F, _, _ = _reconstruct(Z, arc_u, p, cfg)
        feed = np.interp(np.linspace(0.0, cfg.t_end, N_PLOT),
                         np.asarray(T), np.asarray(F))
        PV = float(Y[-1, 2])
    except Exception:                                             # noqa: BLE001
        feed, PV = np.full(N_PLOT, np.nan), np.nan
    return i, feed, float(rn), PV, "-".join(map(str, seq)), time.time() - t0


def _fingerprint(n, seed, cfg):
    """What a saved ensemble must agree with before any of it can be reused.

    Resume is only safe if the rows on disk came from the SAME problem. These are
    the things that silently change the answer: the sample, the stoichiometry and
    the solver settings that select which optimum Stage 1 lands in.
    """
    from singulax.substrate_product_inhibition.kinetics import BALANCED
    return np.array([n, seed, CV, float(BALANCED.c_mu), float(BALANCED.c_pi),
                     cfg.t_end, cfg.F_init, cfg.ls_max_steps, cfg.S_F, cfg.V_max],
                    dtype=float)


def _save(out, feeds, T_grid, structs, res_norms, pvs, X, draws, fp):
    """Write the ensemble so far. Called after each draw so a long run is resumable."""
    np.savez_compressed(out, F=np.asarray(feeds), t=T_grid,
                        structure=np.asarray(structs),
                        res_norm=np.asarray(res_norms), PV=np.asarray(pvs), X=X,
                        draw=np.asarray(draws), fingerprint=fp)


def screen(X, s1s, groups, cfg=CONFIG, out=SCREEN_CSV,
           fsing_min=FSING_MIN, seed_max=SEED_OVERFLOW):
    """Reject draws whose Stage-2 seed is already hopeless, before paying for LM.

    The seed is not an arbitrary guess: it is a forward simulation of the system
    under the arc structure Stage 1 found, starting from lambda(0) = the NLP
    multipliers. Evaluating the residual there is therefore a PHYSICS check
    wearing a numerical costume -- it is large exactly when the arc structure and
    the singular feedback are mutually inconsistent, which is also exactly when
    Levenberg-Marquardt will grind for ~800 s and fail. One evaluation costs ~6 s.

    Two mechanisms show up, in different blocks of the same residual:

      starved singular entry   the arc opens at t ~ 0.2 h with the tank still at
                               its initial litre and S < 0.15 g/L, where F_sing
                               collapses to ~1e-4 L/h. V - Vmax comes back at
                               -7 to -9: the tank cannot fill, and no choice of
                               costates fixes that.
      costate seed overflow    lambda(0) integrated forward through five or more
                               arcs overflows (the adjoint is unstable forward),
                               so ||seed|| ~ 1e19 before LM starts. Here the
                               volume is fine; the numbers are not.

    Every draw is written to `out` with its diagnostics and verdict, so the
    exclusions can be reported with a mechanism attached rather than as a count.
    """
    import csv

    rows, kept = [], {}
    t0 = time.time()
    for seq, idx in groups.items():
        arc_u = tuple(cfg.F_max if v == 2 else cfg.F_min if v == 0 else None
                      for v in seq[:-1])
        M = len(arc_u)
        k_sing = seq.index(1)
        for i in idx:
            s1 = s1s[i]
            p = params_from_row(X[i])
            sw = jnp.asarray(s1.t_sw[:M])
            g = _seed_one(p, jnp.asarray(s1.lam0), sw, arc_u, cfg)
            args = (M, cfg.S_F, cfg.V_max, cfg.F_min, cfg.F_max, cfg.t_end,
                    jnp.asarray(cfg.x0), arc_u, p)
            r = np.asarray(solve_and_get_residuals_ms(g, args))
            finite = bool(np.all(np.isfinite(r)))
            norm = float(np.linalg.norm(r)) if finite else np.inf
            v_gap = float(r[M * 8 + 7]) if finite else np.nan

            t_entry = float(s1.t_sw[k_sing - 1]) if k_sing > 0 else 0.0
            j = int(np.searchsorted(s1.t_grid, t_entry))
            x = s1.X[min(j, len(s1.X) - 1)]
            y = jnp.concatenate([jnp.asarray(x), jnp.asarray(s1.lam0)])
            F_sing = float(get_Fsin_unconstrained(
                y, (cfg.S_F, cfg.V_max, cfg.F_min, cfg.F_max, p)))

            # Test the two mechanisms separately. They are orthogonal: the
            # overflow draw has F_sing = 0.063, perfectly healthy, and the
            # starved draws seed at 8.8-12, nowhere near the overflow cutoff.
            if F_sing < fsing_min:
                verdict, why = "reject", "starved singular entry"
            elif not finite or norm > seed_max:
                verdict, why = "reject", "costate seed overflow"
            else:
                verdict, why = "keep", ""
                kept.setdefault(seq, []).append(i)
            rows.append(dict(draw=i, structure="-".join(map(str, seq)), M=M,
                             seed_norm=norm, v_gap=v_gap, S_entry=float(x[1] / x[3]),
                             F_sing=F_sing, t_entry=t_entry,
                             verdict=verdict, reason=why))

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n_rej = sum(r["verdict"] == "reject" for r in rows)
    print(f"   screened {len(rows)} draws in {time.time() - t0:.0f} s "
          f"({(time.time() - t0) / max(len(rows), 1):.1f} s each); "
          f"{n_rej} rejected, {len(rows) - n_rej} kept")
    for why in sorted({r["reason"] for r in rows if r["verdict"] == "reject"}):
        n = sum(r["reason"] == why for r in rows)
        print(f"      {n:3d} x  {why}")
    # Either margin thinning means the sample has found something the 20-draw
    # calibration did not. Say so rather than silently trusting the cutoff.
    near_f = [r for r in rows if fsing_min < r["F_sing"] < fsing_min * 100]
    near_s = [r for r in rows if np.isfinite(r["seed_norm"])
              and seed_max / 1e4 < r["seed_norm"] < seed_max]
    for tag, near, fmt in [("F_sing", near_f, "{:.1e}"),
                           ("||seed||", near_s, "{:.1f}")]:
        if near:
            detail = ", ".join(f"#{r['draw']} at " + fmt.format(
                r["F_sing"] if tag == "F_sing" else r["seed_norm"]) for r in near)
            print(f"      NOTE: {len(near)} draw(s) near the {tag} cutoff "
                  f"({detail}) -- check {out.name}")
    print(f"   wrote {out.name}")
    return kept, rows


def run(n=N, seed=SEED, cfg=CONFIG, out=FEEDS_NPZ, workers=WORKERS,
        resume=False):
    """Stage 1 sequentially, then Stage 2 per structure group.

    Stage 1 and the screen run in this process; Stage 2 is spread over
    `workers` processes, one draw each. stage2_group() is kept for the
    batched-vs-sequential benchmark but is no longer on this path.
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
    print(f"\n=== SCREEN: {sum(len(v) for v in groups.values())} of {n_ok} "
          f"converged draws are shootable, in {len(groups)} structure groups ===")
    groups, screened = screen(X, s1s, groups, cfg)

    draws = [i for idx in groups.values() for i in idx]
    fp = _fingerprint(n, seed, cfg)
    done, feeds0, structs0, rn0, pv0 = {}, [], [], [], []
    if resume and Path(out).exists():
        # Reuse earlier rows only if they came from the same problem. Default is
        # OFF: a plain run() is always a clean run, so a changed c_mu or F_init
        # cannot quietly mix two models in one ensemble.
        z = np.load(out, allow_pickle=True)
        if "draw" not in z.files:
            print(f"   resume: {Path(out).name} predates draw indexing, ignoring")
        elif not np.allclose(z["fingerprint"], fp):
            print(f"   resume: {Path(out).name} is from a DIFFERENT configuration, "
                  f"ignoring it")
        else:
            for k, i in enumerate(z["draw"]):
                done[int(i)] = True
                feeds0.append(z["F"][k]); structs0.append(str(z["structure"][k]))
                rn0.append(float(z["res_norm"][k])); pv0.append(float(z["PV"][k]))
            print(f"   resume: reusing {len(done)} draws already in "
                  f"{Path(out).name}")
    draws = [i for i in draws if i not in done]
    peak = workers * GB_PER_WORKER
    print(f"\n=== STAGE 2: {len(draws)} draws on {workers} processes ===",
          flush=True)
    print(f"    memory budget: ~{peak:.0f} GB of {_MEM_GB:.0f} GB "
          f"({GB_PER_WORKER:.0f} GB/worker, measured 3.2-4.0)", flush=True)
    if peak > 0.85 * _MEM_GB:
        # Say it before spending an hour: OOM here kills the desktop session too.
        print(f"    WARNING: that is {100 * peak / _MEM_GB:.0f}% of RAM. Pass a "
              f"smaller workers= or expect the OOM killer.", flush=True)
    T_grid = np.linspace(0.0, cfg.t_end, N_PLOT)
    feeds, structs, res_norms, pvs = feeds0, structs0, rn0, pv0
    kept_draws = [i for i in done]
    jobs = [(i, X[i], s1s[i].seq, np.asarray(s1s[i].t_sw),
             np.asarray(s1s[i].lam0)) for i in draws]

    t0 = time.time()
    # Thread and codegen limits are set at module scope, above `import jax`, so
    # they apply to this parent AND to every spawned child when it re-imports.
    # spawn, not fork: JAX is not fork-safe.
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, (i, feed, rn, PV, seq_s, dt) in enumerate(
                pool.imap_unordered(_stage2_draw, jobs, chunksize=1)):
            feeds.append(feed); structs.append(seq_s)
            res_norms.append(rn); pvs.append(PV)
            kept_draws.append(i)
            el = time.time() - t0
            print(f"   {k + 1:3d}/{len(jobs)}  draw {i:3d}  ({seq_s})  "
                  f"||res|| {rn:.1e}  {dt:6.1f}s   elapsed {el / 60:5.1f} min   "
                  f"ETA {el / (k + 1) * (len(jobs) - k - 1) / 60:5.1f} min",
                  flush=True)
            # Save after every draw: an N=100 run is hours, and writing only on
            # completion loses everything if it is interrupted.
            _save(out, feeds, T_grid, structs, res_norms, pvs, X,
                  kept_draws, fp)

    print(f"\nwrote {out}   ({len(feeds)} profiles)")
    return feeds, T_grid, np.asarray(structs)


def nominal_feed(cfg=CONFIG, out=NOMINAL_NPZ, force=False):
    """Feed profile of the BALANCED model at the literature kinetics.

    A single hybrid solve, cached, so it can be overlaid on the ensemble as a
    reference without re-running the sampling. This is the trajectory every
    figure in the deterministic sections refers to.
    """
    if out.exists() and not force:
        z = np.load(out)
        return z["t"], z["F"], float(z["PV"]), float(z["res_norm"])

    from singulax.substrate_product_inhibition.hybrid import solve_hybrid
    from singulax.substrate_product_inhibition.kinetics import BALANCED

    print("solving the nominal balanced case (one hybrid solve)...", flush=True)
    r = solve_hybrid(BALANCED, cfg)
    if not r.ok:
        print(f"   nominal solve did NOT converge: {r.status}")
        return None, None, np.nan, r.res_norm
    T_grid = np.linspace(0.0, cfg.t_end, N_PLOT)
    F = np.interp(T_grid, r.T, r.F)
    np.savez_compressed(out, t=T_grid, F=F, PV=r.PV, res_norm=r.res_norm,
                        structure="-".join(map(str, r.structure)))
    print(f"   PV = {r.PV:.2f} g, ||res|| = {r.res_norm:.2e}, "
          f"structure {r.structure};  cached to {out.name}")
    return T_grid, F, r.PV, r.res_norm


def plot(path=FEEDS_NPZ, figure=FIGURE, nominal=True):
    """The ensemble of optimal feed profiles, one colour per draw."""
    z = np.load(path, allow_pickle=True)
    F, t = z["F"], z["t"]
    # A draw whose shoot did not converge has no optimal feed to plot -- its
    # trajectory is whatever the reconstruction made of an unconverged Zopt.
    # Dropping them is not cherry-picking: the yield is reported, and a
    # non-solution is not a datum.
    good = ((np.asarray(z["res_norm"]) < CONFIG.res_tol)
            & np.isfinite(np.asarray(z["PV"]))
            & np.isfinite(F).all(axis=1))
    if (~good).sum():
        # Say WHICH test dropped each draw. Reporting them all as "did not
        # converge" is false for the ones whose residual reached 1e-12 and whose
        # reconstruction then returned nan -- two different failures.
        rn = np.asarray(z["res_norm"])
        n_res = int((rn >= CONFIG.res_tol).sum())
        n_rec = int((~good & (rn < CONFIG.res_tol)).sum())
        if n_res:
            bad = rn[rn >= CONFIG.res_tol]
            print(f"dropping {n_res} draw(s) whose shoot did not converge "
                  f"(||res|| {bad.min():.1e}-{bad.max():.1e})")
        if n_rec:
            print(f"dropping {n_rec} draw(s) that converged but reconstructed "
                  f"to a non-finite trajectory")
        print(f"   {int(good.sum())} of {len(good)} remain")
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
    oi = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#0072B2"]
    fams = [s for s, _ in sorted(((s, int((structure == s).sum()))
                                  for s in set(structure.tolist())),
                                 key=lambda kv: -kv[1])]
    colour = {s: oi[k % len(oi)] for k, s in enumerate(fams)}

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    for f, s_ in zip(F, structure):
        ax.plot(t, f, color=colour[s_], lw=1.0, alpha=0.75)
    handles = [plt.Line2D([], [], color=colour[s_], lw=2,
                          label=f"({s_})  n={int((structure == s_).sum())}")
               for s_ in fams]
    if nominal:
        t_n, F_n, pv_n, _ = nominal_feed()
        if F_n is not None:
            ax.plot(t_n, F_n, color="black", lw=2.0, zorder=5)
            handles.append(plt.Line2D([], [], color="black", lw=2.0,
                                      label=f"literature kinetics ({pv_n:.0f} g)"))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"$F$ (L h$^{-1}$)")
    ax.set_title(f"Optimal feed, {len(F)} sampled parameter sets", loc="left")
    ax.legend(handles=handles, loc="upper right")
    fig.savefig(figure, dpi=600, bbox_inches="tight")
    print(f"saved {figure}")


if __name__ == "__main__":
    if "--plot" not in sys.argv:         # --plot re-reads the npz without re-solving
        run()
    plot()
