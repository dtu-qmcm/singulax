"""The 23 draws the pipeline dropped, plotted next to the 77 it kept.

Two different losses, so two panels -- they are not comparable curves:

  left   12 draws never reached Stage 2, because Stage 1 labelled a structure the
         screen will not shoot. These are STAGE 1 feeds (piecewise constant on the
         N = 400 collocation grid), re-solved here since feed_ensemble does not
         cache them.
  right  11 draws were shot and failed: 3 whose residual never came down, 8 that
         converged to ~1e-12 and then reconstructed to a non-finite trajectory.
         These are STAGE 2 feeds, read back from the ensemble npz.

The 77 usable draws sit behind both in grey, for shape comparison only (they are
Stage 2 feeds in both panels).

Also prints where V reaches V_max on the left-hand 12. vmax_probe.py measured that
on `ipopt_max_iter = 3000` solutions -- solves IPOPT had abandoned -- and found the
bound touched only at the final node. At 20000 Stage 1 converges, and Stage 2's
answer rides V = V_max for 5.26 h; if Stage 1 now does too, everything past the
entry node is degenerate (every F = 0 is optimal there) and that is where the
trailing (2,1,0,...) chatter comes from.

    uv run python plot_rejected.py                          # solves 12, then plots
    uv run python plot_rejected.py --plot                   # re-plot from the cache
    uv run python plot_rejected.py mc_sampling/other.npz    # a different ensemble
"""
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MC = HERE / "mc_sampling"
sys.path.insert(0, str(MC))

from singulax.substrate_product_inhibition.hybrid import (   # noqa: E402
    solve_stage1, DEFAULT_CONFIG)
from sampling_spi import params_from_row, sample_X           # noqa: E402

#: feed_ensemble.py's draws. LHS is NOT nested -- sample_X(10, s) is a different set.
N_DRAWS, SEED = 100, 1234
CFG = DEFAULT_CONFIG._replace(F_init=0.5, ls_max_steps=10000, ipopt_max_iter=20000)
RES_TOL = CFG.res_tol

CACHE = MC / "stage1_rejected.npz"
FIGURE = HERE / "rejected_feeds.png"

#: 0.1 mL in a 10 L tank. hybrid.VMAX_TOL = 1e-6 is tighter than IPOPT's own
#: convergence, so it reports "no boundary arc" on solutions that have one.
BOUNDARY_TOL = 1.0e-3

#: The 12 dropped before Stage 2: every Stage 1 label that was not exactly
#: (2, 0, 1, 0), minus draw 60, which WAS shot and failed at ||res|| 1.2e-02.
#: Split by what the label actually says, which is the point of the left panel.
TRAILING = [1, 2, 7, 9, 11, 16, 23, 29, 49, 99]   # (2,0,1,0) + extra arcs after it
DIFFERENT = [26, 40]                              # (2,0) and (1,) -- no singular exit
REJECTED = sorted(TRAILING + DIFFERENT)

#: Okabe-Ito, by failure class rather than by exact label: 9 distinct labels across
#: 12 draws would need 9 hues, and the classes are what the figure is about.
C_TRAIL, C_DIFF = "#E69F00", "#CC79A7"
C_RES, C_RECON = "#D55E00", "#0072B2"
C_BG = "#9a9a9a"


def stage1_feeds(which=REJECTED, cache=CACHE):
    """Stage 1 for the rejected draws, cached -- ~30 s each, so solve them once."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["draw"]) == list(which):
            print(f"cache hit: {cache.name} ({len(which)} draws)")
            return {int(i): (z["U"][k], z["V"][k], str(z["seq"][k]))
                    for k, i in enumerate(z["draw"])}
        print(f"{cache.name} holds different draws -- re-solving")

    X = sample_X(N_DRAWS, SEED)
    U, V, seqs = [], [], []
    print(f"solving Stage 1 for {len(which)} rejected draws "
          f"(max_iter = {CFG.ipopt_max_iter})...\n")
    print("%-6s %-28s %9s %9s %9s %12s %7s"
          % ("draw", "seq", "max V", "enter", "arc (h)", "max|F| after", "time"))
    h = CFG.t_end / CFG.N
    for i in which:
        t0 = time.time()
        r = solve_stage1(params_from_row(X[i]), CFG)
        Vi = np.asarray(r.X)[:, 3]
        U.append(np.asarray(r.U)); V.append(Vi); seqs.append("-".join(map(str, r.seq)))

        on = np.flatnonzero(Vi >= CFG.V_max - BOUNDARY_TOL)
        if len(on):
            k = int(on[0])
            after = float(np.abs(np.asarray(r.U)[k:]).max()) if k < CFG.N else 0.0
            enter, arc = f"{k * h:.2f} h", f"{(CFG.N - k) * h:.2f}"
        else:
            after, enter, arc = 0.0, "never", "-"
        print("%-6d %-28s %9.5f %9s %9s %12.2e %6.0fs"
              % (i, str(tuple(r.seq)), Vi.max(), enter, arc, after, time.time() - t0))

    np.savez_compressed(cache, draw=np.asarray(which), U=np.asarray(U),
                        V=np.asarray(V), seq=np.asarray(seqs))
    print(f"\ncached to {cache}")
    print("An 'enter' well before 39.50 h means Stage 1 DOES ride V_max at 20000 and\n"
          "the chatter after it is the degenerate tail. 'never' means the\n"
          "3000-iteration picture still holds and the chatter is something else.")
    return {int(i): (U[k], V[k], seqs[k]) for k, i in enumerate(which)}


def load_ensemble(path):
    """The shot draws, split into usable / residual failure / reconstruction failure."""
    z = np.load(path, allow_pickle=True)
    F, t, draw = z["F"], z["t"], np.asarray(z["draw"]).astype(int)
    rn, pv = np.asarray(z["res_norm"]), np.asarray(z["PV"])
    finite = np.isfinite(F).all(axis=1) & np.isfinite(pv)
    good = (rn < RES_TOL) & finite
    bad_res = rn >= RES_TOL                      # never converged
    bad_rec = (rn < RES_TOL) & ~finite           # converged, then reconstructed to inf
    print(f"\n{path.name}: {len(draw)} shot -> {good.sum()} usable, "
          f"{bad_res.sum()} residual, {bad_rec.sum()} reconstruction")
    return t, F, draw, rn, good, bad_res, bad_rec


def plot(path, s1, figure=FIGURE):
    t, F, draw, rn, good, bad_res, bad_rec = load_ensemble(path)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
        "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 600,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True,
                                   constrained_layout=True)

    for ax in (axL, axR):                        # the kept 77, for shape only
        for f in F[good]:
            ax.plot(t, f, color=C_BG, lw=0.7, alpha=0.30, zorder=1)
        ax.set_xlabel("Time (h)")
    axL.set_ylabel(r"$F$ (L h$^{-1}$)")

    # ---- left: the 12 that never reached Stage 2 ---------------------------
    t1 = np.linspace(0.0, CFG.t_end, CFG.N + 1)[:-1]
    for i in REJECTED:
        U, _, _ = s1[i]
        c = C_DIFF if i in DIFFERENT else C_TRAIL
        axL.step(t1, U, where="post", color=c, lw=1.0, alpha=0.85,
                 zorder=3 if i in DIFFERENT else 2)
    for i in DIFFERENT:                          # the contrast WARN wants labels
        U, _, s_ = s1[i]
        k = int(np.argmax(U < 1e-6)) if (U < 1e-6).any() else len(U) - 1
        axL.annotate(f"#{i}  ({s_.replace('-', ',')})", (t1[k], U[k]),
                     textcoords="offset points", xytext=(6, 6), fontsize=7,
                     color=C_DIFF)
    axL.set_title(f"Rejected at Stage 1 ({len(REJECTED)}) -- collocation feed", loc="left")
    axL.legend(handles=[
        plt.Line2D([], [], color=C_TRAIL, lw=2,
                   label=f"(2,0,1,0) + trailing arcs   n={len(TRAILING)}"),
        plt.Line2D([], [], color=C_DIFF, lw=2,
                   label=f"different structure   n={len(DIFFERENT)}"),
        plt.Line2D([], [], color=C_BG, lw=2, alpha=0.6,
                   label=f"usable, Stage 2   n={int(good.sum())}"),
    ], loc="upper right")

    # ---- right: the 11 that were shot and failed ---------------------------
    for mask, c, z in ((bad_rec, C_RECON, 2), (bad_res, C_RES, 3)):
        for f in F[mask]:
            if np.isfinite(f).any():
                axR.plot(t, f, color=c, lw=1.0, alpha=0.85, zorder=z)
    n_blank = int(sum(not np.isfinite(f).any() for f in F[bad_rec]))
    for i, f, r in zip(draw[bad_res], F[bad_res], rn[bad_res]):
        if np.isfinite(f).any():
            k = int(np.nanargmax(f))
            axR.annotate(f"#{i}  {r:.0e}", (t[k], f[k]), textcoords="offset points",
                         xytext=(5, 4), fontsize=7, color=C_RES)
    axR.set_title(f"Shot and failed ({int(bad_res.sum() + bad_rec.sum())}) -- "
                  "indirect feed", loc="left")
    axR.legend(handles=[
        plt.Line2D([], [], color=C_RES, lw=2,
                   label=f"residual did not converge   n={int(bad_res.sum())}"),
        plt.Line2D([], [], color=C_RECON, lw=2,
                   label=f"reconstruction non-finite   n={int(bad_rec.sum())}"
                         + (f" ({n_blank} all-NaN)" if n_blank else "")),
        plt.Line2D([], [], color=C_BG, lw=2, alpha=0.6,
                   label=f"usable, Stage 2   n={int(good.sum())}"),
    ], loc="upper right")

    fig.savefig(figure, dpi=600, bbox_inches="tight")
    print(f"saved {figure}")


def main():
    args = [a for a in sys.argv[1:] if a != "--plot"]
    path = Path(args[0]) if args else None
    if path is None:
        for cand in (MC / "feed_ensemble_it20k.npz",
                     MC / "feed_ensemble_balanced_cv20.npz"):
            if cand.exists():
                path = cand
                break
    if path is None or not path.exists():
        sys.exit(f"no ensemble npz found -- copy barbie's to {MC/'feed_ensemble_it20k.npz'}")
    plot(path, stage1_feeds())


if __name__ == "__main__":
    main()
