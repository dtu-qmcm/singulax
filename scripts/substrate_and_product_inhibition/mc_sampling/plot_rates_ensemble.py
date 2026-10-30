"""The specific rates mu, pi and sigma for every sampled model -- no optimisation.

What the 20% parameter uncertainty does to the KINETICS, before any control problem is
posed. Purely evaluating the rate laws on the sampled constants, so it runs in seconds.

Both rates are Monod in S and inhibited by P:

    mu    = mu_max * S / ((1 + P/K_p)    * (K_s + S))
    pi    = pi_max * S / ((1 + P/K_p_pi) * (K_s_pi + S))
    sigma = c_mu*mu + c_pi*pi

Note what that does and does not say. They are monotonically INCREASING in S -- there is
no S^2/K_I term, so this model is not substrate-inhibited despite the module name, and
the singular arc comes from PRODUCT inhibition. Arguments of the form "non-monotonic
kinetics fix the arc sequence a priori" (Modak & Lim) do not apply here.

The S panels use a log axis on purpose: K_s = 0.22 and K_s_pi = 0.44 g/L, so on a linear
0-30 g/L axis every bit of curvature collapses into the first millimetre and all 100
curves look identical.

    uv run python plot_rates_ensemble.py           100 draws
    uv run python plot_rates_ensemble.py 500       more, for a smoother envelope
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from singulax.substrate_product_inhibition.kinetics import BALANCED, mu, pi, sigma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampling_spi import CV, NAMES, sample_X                      # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = 1234
HERE = Path(__file__).resolve().parent
FIGURE = HERE / f"rates_ensemble_balanced_cv{round(CV * 100):02d}.png"

#: Ranges the optimal trajectories actually visit: S peaks near 26 g/L at the singular
#: entry, P reaches ~57 g/L at t_f (PV 566 g in 10 L).
S_GRID = np.logspace(-2, np.log10(50.0), 400)
P_GRID = np.linspace(0.0, 60.0, 400)
S_REF = 20.0        # representative substrate level on the singular arc
P_REF = 0.0         # start of fermentation


def main():
    X = sample_X(N, SEED)
    cols = {n: X[:, j][:, None] for j, n in enumerate(NAMES)}      # (N, 1) for broadcast
    nom = {n: getattr(BALANCED, n) for n in NAMES}

    def rates(S, P, p):
        """mu, pi, sigma from the real rate laws -- one source of truth, not retyped."""
        m = mu(S, P, p["mu_max"], p["k_p"], p["k_s"])
        q = pi(S, P, p["pi_max"], p["k_p_pi"], p["k_s_pi"])
        s = sigma(S, P, BALANCED.c_mu, BALANCED.c_pi,
                  p["mu_max"], p["k_p"], p["k_s"],
                  p["pi_max"], p["k_p_pi"], p["k_s_pi"])
        return m, q, s

    ens_S = rates(S_GRID[None, :], P_REF, cols)       # (N, 400) each
    nom_S = rates(S_GRID, P_REF, nom)
    ens_P = rates(S_REF, P_GRID[None, :], cols)
    nom_P = rates(S_REF, P_GRID, nom)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.8,
        "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 600,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "grid.linewidth": 0.5,
    })
    oi = {"orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
          "vermillion": "#D55E00", "black": "#000000", "reddish_purple": "#CC79A7"}

    fig, ax = plt.subplots(2, 3, figsize=(9.5, 5.6), constrained_layout=True)
    labels = [r"$\mu$ (h$^{-1}$)",
              r"$\pi$ (g g$^{-1}$ h$^{-1}$)",
              r"$\sigma$ (g g$^{-1}$ h$^{-1}$)"]
    titles = ["growth", "production", "substrate demand"]

    for j in range(3):
        # --- top row: vs S at P = 0 ---
        a = ax[0, j]
        a.plot(S_GRID, ens_S[j].T, color=oi["sky_blue"], lw=0.6, alpha=0.18)
        a.plot(S_GRID, nom_S[j], color=oi["black"], lw=1.6, zorder=5)
        a.set_xscale("log")
        a.set_xlabel(r"$S$ (g L$^{-1}$)")
        a.set_ylabel(labels[j])
        a.set_title(f"({'abc'[j]}) {titles[j]} vs $S$   ($P$ = {P_REF:.0f} g L$^{{-1}}$)",
                    loc="left")

        # --- bottom row: vs P at a singular-arc substrate level ---
        b = ax[1, j]
        b.plot(P_GRID, ens_P[j].T, color=oi["sky_blue"], lw=0.6, alpha=0.18)
        b.plot(P_GRID, nom_P[j], color=oi["black"], lw=1.6, zorder=5)
        b.set_xlabel(r"$P$ (g L$^{-1}$)")
        b.set_ylabel(labels[j])
        b.set_title(f"({'def'[j]}) {titles[j]} vs $P$   ($S$ = {S_REF:.0f} g L$^{{-1}}$)",
                    loc="left")

    # Identity is never colour alone: two series, so label them.
    ax[0, 0].legend(handles=[
        plt.Line2D([], [], color=oi["sky_blue"], lw=1.4, alpha=0.6,
                   label=f"{N} draws, CV = {CV:.0%}"),
        plt.Line2D([], [], color=oi["black"], lw=1.6, label="nominal (Shin & Lim)")],
        loc="lower right")

    fig.suptitle(f"Specific rates under {CV:.0%} parameter uncertainty "
                 f"(carbon- and redox-balanced model)", fontsize=10)
    fig.savefig(FIGURE, dpi=600, bbox_inches="tight")
    print(f"saved {FIGURE}")

    # The numbers behind the picture, so the figure can be described in text.
    print(f"\nspread over {N} draws (2.5th - 97.5th percentile):")
    for tag, S, P, ens, nm in (("at S = 0.5, P = 0  ", 0.5, 0.0, None, None),
                               (f"at S = {S_REF:.0f},  P = 0  ", S_REF, 0.0, None, None),
                               (f"at S = {S_REF:.0f},  P = 50 ", S_REF, 50.0, None, None)):
        m, q, s = rates(S, P, cols)
        mn, qn, sn = rates(S, P, nom)
        print(f"   {tag}  mu {np.percentile(m, 2.5):6.3f}-{np.percentile(m, 97.5):6.3f} "
              f"(nom {float(mn):6.3f})   "
              f"pi {np.percentile(q, 2.5):6.3f}-{np.percentile(q, 97.5):6.3f} "
              f"(nom {float(qn):6.3f})   "
              f"sigma {np.percentile(s, 2.5):6.3f}-{np.percentile(s, 97.5):6.3f} "
              f"(nom {float(sn):6.3f})")


if __name__ == "__main__":
    main()
