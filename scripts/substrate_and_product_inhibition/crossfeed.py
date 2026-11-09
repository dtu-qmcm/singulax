"""What does it cost to apply an optimal feed to the wrong kinetics?

The ensemble says the ACHIEVABLE product is insensitive to the kinetics: every draw's
own optimum lands near 570 g. That is not the same as saying the kinetics do not
matter. Each draw reaches its optimum with a DIFFERENT feed -- singular exit times span
16 to 39 h -- so the open-loop question is separate and is the one a practitioner
faces: you have literature kinetics, you compute one feed, you run it on an organism
whose kinetics differ.

So this applies the NOMINAL feed -- the one computed from literature kinetics -- to
each sampled model, which is what a practitioner actually does.

The volume constraint cannot be violated: Vdot = F, so V(t) depends only on the feed
and is identical under any kinetics. Nothing goes infeasible; the only cost is lost
product. For draw i the loss is

    loss_i = 1 - PV_i(F*_nominal) / PV_i(F*_i)

with PV_i(F*_i) read from the ensemble -- that draw's own converged optimum.

No optimisation runs here, only integration: 77 ODE solves, seconds after the JIT
warm-up.

    uv run python crossfeed.py
"""
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import diffrax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MC = HERE / "mc_sampling"
sys.path.insert(0, str(MC))

from singulax.substrate_product_inhibition.hybrid import DEFAULT_CONFIG   # noqa: E402
from singulax.substrate_product_inhibition.odes import f_vec, g_vec       # noqa: E402
from sampling_spi import params_from_row                                  # noqa: E402

CFG = DEFAULT_CONFIG
ENSEMBLE = MC / "feed_ensemble_balanced_cv20.npz"
NOMINAL = MC / "feed_nominal_balanced.npz"
FIGURE = HERE / "crossfeed.png"

#: Okabe-Ito, in the SAME ROLES sampling_spi.py assigns them: sky blue carries the
#: data, vermillion the contrast series, black the reference lines. Matching the roles
#: and not merely the palette is what makes the figures read as one set -- a reader who
#: has learnt that vermillion means "the costly case" on one figure keeps that reading.
OI = {"orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
      "vermillion": "#D55E00", "black": "#000000", "reddish_purple": "#CC79A7"}
C_OK, C_BAD, C_INK = OI["sky_blue"], OI["vermillion"], OI["black"]

#: Loss above which a draw counts as materially hurt by the nominal feed (%).
THRESH = 5.0


def _rhs(t, y, args):
    """xdot = f(x) + g(x) F(t), with F read off the stored profile.

    The stored feed is piecewise constant on a 400-point grid; jnp.interp makes it
    piecewise LINEAR, which rounds each switch over one grid spacing (0.099 h). That
    is deliberate -- an adaptive solver handles a kink far better than a jump, and
    0.099 h against a 39.5 h horizon moves PV by far less than the effect being
    measured.
    """
    t_grid, F_grid, S_F, p = args
    return f_vec(y, p) + g_vec(y, S_F) * jnp.interp(t, t_grid, F_grid)


@jax.jit
def simulate(t_grid, F_grid, p):
    """PV(t_f) from applying this feed to these kinetics. No optimisation."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_rhs), diffrax.Tsit5(),
        t0=0.0, t1=CFG.t_end, y0=jnp.asarray(CFG.x0), dt0=0.01,
        args=(t_grid, F_grid, CFG.S_F, p),
        stepsize_controller=diffrax.PIDController(rtol=1e-9, atol=1e-9),
        saveat=diffrax.SaveAt(t1=True), max_steps=1000000, throw=False)
    return sol.ys[-1][2]


def main():
    for f in (ENSEMBLE, NOMINAL):
        if not f.exists():
            sys.exit(f"missing {f}")

    z = np.load(ENSEMBLE, allow_pickle=True)
    F, t, draw = z["F"], z["t"], np.asarray(z["draw"]).astype(int)
    rn, pv_own, X = np.asarray(z["res_norm"]), np.asarray(z["PV"]), z["X"]
    good = (rn < CFG.res_tol) & np.isfinite(pv_own) & np.isfinite(F).all(axis=1)
    F, draw, pv_own = F[good], draw[good], pv_own[good]
    rows = [params_from_row(X[i]) for i in draw]
    n = len(draw)

    zn = np.load(NOMINAL, allow_pickle=True)
    t_nom, F_nom, pv_nom_own = zn["t"], zn["F"], float(zn["PV"])
    tj, Fnj = jnp.asarray(t_nom), jnp.asarray(F_nom)

    print(f"{n} usable draws;  nominal optimum PV = {pv_nom_own:.2f} g\n")

    # The nominal feed applied to each sampled model.
    pv_b = np.array([float(simulate(tj, Fnj, p)) for p in rows])
    loss_b = 100.0 * (1.0 - pv_b / pv_own)

    q1, q2, q3 = np.percentile(loss_b, [25, 50, 75])
    print("nominal feed -> sampled models")
    print(f"    PV    {pv_b.mean():7.2f} +- {pv_b.std(ddof=1):5.2f} g   "
          f"[{pv_b.min():.1f}, {pv_b.max():.1f}]")
    # Strongly right-skewed (median ~0, long tail), so mean +- sd describes neither
    # the typical case nor the risk. Quote quartiles.
    print(f"    loss  median {q2:5.2f} %   IQR [{q1:.2f}, {q3:.2f}]   "
          f"p90 {np.percentile(loss_b, 90):5.2f} %   worst {loss_b.max():5.2f} %")
    print(f"    mean  {loss_b.mean():5.2f} +- {loss_b.std(ddof=1):.2f} %  "
          f"(skewed -- report the median)")
    print(f"    above {THRESH:g} % loss: {(loss_b > THRESH).sum()} of {n}\n")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
        "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 600,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)

    costly = loss_b > THRESH
    n_bad = int(costly.sum())

    # --- (a) own optimum vs nominal feed, draw by draw --------------------------
    # Equal aspect with the identity line is the point: every marker must fall ON or
    # BELOW it, because no feed can beat a model's own optimum. The empty upper-left
    # triangle is the sanity check, so it is kept rather than cropped away.
    lo = min(pv_own.min(), pv_b.min()) - 12
    hi = max(pv_own.max(), pv_b.max()) + 12
    axL.plot([lo, hi], [lo, hi], color=C_INK, lw=0.9, ls=(0, (5, 3)), zorder=1,
             label="no loss ($y = x$)")
    axL.scatter(pv_own[~costly], pv_b[~costly], s=17, color=C_OK, alpha=0.85,
                edgecolor="white", linewidth=0.4, zorder=3,
                label=f"loss $\\leq$ {THRESH:g}% ({n - n_bad} draws)")
    axL.scatter(pv_own[costly], pv_b[costly], s=21, color=C_BAD, alpha=0.9,
                marker="D", edgecolor="white", linewidth=0.4, zorder=4,
                label=f"loss $>$ {THRESH:g}% ({n_bad} draws)")
    axL.set_xlabel(r"$PV$ under the model's own optimal feed (g)")
    axL.set_ylabel(r"$PV$ under the nominal feed (g)")
    axL.set_title("(a)  Nominal feed applied to each sampled model", loc="left")
    axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
    axL.set_aspect("equal", adjustable="box")
    axL.legend(loc="upper left", handletextpad=0.5, borderpad=0.2, labelspacing=0.35)

    # --- (b) the loss distribution ----------------------------------------------
    # Bin width divides THRESH exactly, so the threshold is a bin EDGE and no bar
    # straddles it. Without that, the bar spanning 4-5.5 % sits half on each side of
    # the line and the split stops being readable off the bars. Colour carries the
    # same 5 % split as panel (a): these bars are those diamonds.
    bw = THRESH / 4.0
    edges = np.arange(0.0, np.ceil(loss_b.max() / bw) * bw + bw, bw)
    axR.hist([loss_b[~costly], loss_b[costly]], bins=edges, stacked=True,
             color=[C_OK, C_BAD], edgecolor="white", linewidth=0.4, zorder=2,
             label=[f"loss $\\leq$ {THRESH:g}%  ({n - n_bad} draws)",
                    f"loss $>$ {THRESH:g}%  ({n_bad} draws)"])
    axR.axvline(THRESH, color=C_INK, lw=0.9, ls=(0, (5, 3)), zorder=4)
    axR.annotate(f"median {q2:.2f}%\nworst {loss_b.max():.1f}%",
                 (0.975, 0.62), xycoords="axes fraction", ha="right", va="top",
                 fontsize=7.5, color=C_INK)
    axR.set_xlabel("Product loss vs. the model's own optimum (%)")
    axR.set_ylabel("Draws")
    axR.set_title("(b)  Cost of using the nominal feed", loc="left")
    axR.set_xlim(-bw, edges[-1])
    axR.legend(loc="upper right", handletextpad=0.5, borderpad=0.2,
               labelspacing=0.35)

    fig.savefig(FIGURE, dpi=600, bbox_inches="tight")
    print(f"saved {FIGURE}")


if __name__ == "__main__":
    main()
