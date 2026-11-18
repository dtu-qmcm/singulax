"""Stage 1 vs Stage 3 feed for the literature (BALANCED) kinetics.

The argument the hybrid method makes, in one picture: the DIRECT stage finds the arc
STRUCTURE but returns a feed that is piecewise constant on the mesh and chatters on the
singular arc; the INDIRECT stage takes that structure as given and solves the boundary
value problem, returning exact switching times and a smooth singular control.

Side by side on a SHARED FEED AXIS, to sit under the poster's own
"STAGE 1 - DIRECT" / "STAGE 3 - INDIRECT" headings. No titles for the same reason:
the poster supplies them, and a figure that repeats them drifts out of step with the
template the first time either is reworded.

Stage 3 is read from the cache feed_ensemble.py already wrote, so only Stage 1 is
solved here (~20 s at ipopt_max_iter = 20000).

    uv run python plot_nominal_stages.py
"""
import sys
from pathlib import Path

# Must precede every other singulax import and any array creation: solve_stage1
# calls _require_x64() and refuses float32, in which the shooting residuals
# cannot reach 1e-12.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MC = HERE / "mc_sampling"
sys.path.insert(0, str(MC))

from singulax.substrate_product_inhibition.hybrid import (            # noqa: E402
    ProblemConfig, solve_stage1)
from singulax.substrate_product_inhibition.kinetics import BALANCED   # noqa: E402
from crossfeed import simulate                                        # noqa: E402

#: Same configuration feed_ensemble.py uses. The iteration limit is not incidental:
#: at the default 3000 IPOPT is abandoned feasible but not optimal and the detected
#: structure degrades, so a figure made at 3000 would not show the method working.
CFG = ProblemConfig(F_init=0.5, ls_max_steps=10000, ipopt_max_iter=20000)

NOMINAL = MC / "feed_nominal_balanced.npz"
FIG_DIR = HERE.parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)
FIGURE = FIG_DIR / "spi_nominal_stages.png"

#: Okabe-Ito in the roles the other figures use: sky blue for the direct result,
#: bluish green for the refined one.
OI = {"orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
      "vermillion": "#D55E00", "black": "#000000", "reddish_purple": "#CC79A7"}


def main():
    if not NOMINAL.exists():
        sys.exit(f"missing {NOMINAL} -- run feed_ensemble.py first")
    z = np.load(NOMINAL, allow_pickle=True)
    t3, F3, pv3 = z["t"], z["F"], float(z["PV"])

    print(f"solving Stage 1 for the literature kinetics "
          f"(max_iter = {CFG.ipopt_max_iter})...", flush=True)
    s1 = solve_stage1(BALANCED, CFG)
    F1 = np.append(s1.U, s1.U[-1])          # U is per-interval; close the last one

    # s1.PV is the forward-Euler trajectory's own estimate at N = 400, so it carries
    # the mesh's discretisation error. Quoting it against Stage 3's accurately
    # integrated PV would attribute that error to the method. Re-integrate the SAME
    # Stage 1 feed with the same solver crossfeed.py uses, and the two numbers become
    # comparable: what is left is the product the refined switching times actually buy.
    pv1 = float(simulate(jnp.asarray(s1.t_grid), jnp.asarray(F1), BALANCED))

    print(f"   structure {s1.seq}   defects {s1.defect_norm:.1e}")
    print(f"   {s1.ipopt_message}")
    print(f"\nStage 1  collocation's own PV   {s1.PV:8.2f} g   (forward Euler, N={len(s1.U)})")
    print(f"         same feed, integrated  {pv1:8.2f} g   <- comparable to Stage 3")
    print(f"Stage 3  PV                     {pv3:8.2f} g")
    print(f"         gain from refinement   {pv3 - pv1:+8.2f} g")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
        "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 600,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharex=True,
                                   sharey=True, constrained_layout=True)

    def note(ax, text):
        ax.annotate(text, (0.97, 0.94), xycoords="axes fraction", ha="right",
                    va="top", fontsize=7.5, color=OI["black"], linespacing=1.4)

    # Stage 1 is piecewise constant on the mesh, so it is drawn as a STEP. A smooth
    # line here would quietly show the reader a control the method never produced,
    # and would hide the chatter that is the whole reason Stage 3 exists.
    axA.step(s1.t_grid, F1, where="post", color=OI["sky_blue"], zorder=3)
    axB.plot(t3, F3, color=OI["bluish_green"], zorder=3)

    for ax in (axA, axB):
        ax.set_xlabel("Time (h)")
        ax.set_xlim(0.0, CFG.t_end)
        ax.set_ylim(-0.1, CFG.F_max * 1.12)
        ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axA.set_ylabel(r"$F$ (L h$^{-1}$)")

    fig.savefig(FIGURE, dpi=600, bbox_inches="tight")
    print(f"saved {FIGURE}")


if __name__ == "__main__":
    main()
