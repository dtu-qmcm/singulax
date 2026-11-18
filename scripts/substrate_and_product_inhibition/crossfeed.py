"""What does it cost to apply an optimal feed to the wrong kinetics?

The ensemble says the ACHIEVABLE product is insensitive to the kinetics: every draw's
own optimum lands near 570 g. That is not the same as saying the kinetics do not
matter. Each draw reaches its optimum with a DIFFERENT feed -- singular exit times span
16 to 39 h -- so the open-loop question is separate and is the one a practitioner
faces: you have literature kinetics, you compute one feed, you run it on an organism
whose kinetics differ.

So this applies the NOMINAL feed -- the one computed from literature kinetics -- to
each sampled model, which is what a practitioner actually does.

Vdot = F, so V(t) depends only on the feed and is identical under any kinetics. The
re-simulation cuts the feed at V_max (see _rhs) so that the discretised profile
cannot overfill the reactor. Nothing goes infeasible; the only cost is lost product. For draw i the loss is

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
FIG_DIR = HERE.parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)
FIGURE = FIG_DIR / "spi_crossfeed.png"

#: Okabe-Ito, in the SAME ROLES sampling_spi.py assigns them: black for reference
#: lines, and the bluish-green / vermillion pair for good vs bad -- exactly how the SRC
#: tornado already reads them ("increases PV" / "decreases PV"). Matching the roles and
#: not merely the palette is what makes the figures read as one set.
#:
#: Green against vermillion is the tightest pair in the set under red-green CVD: OKLab
#: dE is 25.8 normal, 13.0 protanopia, 11.0 deuteranopia. Above the dE >= 8 target, but
#: with less headroom than sky blue would give, so the second encoding is not optional
#: -- panel (a) separates the series by MARKER SHAPE and panel (b) by position either
#: side of the threshold. Keep both if these colours are ever restyled.
OI = {"orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
      "vermillion": "#D55E00", "black": "#000000", "reddish_purple": "#CC79A7"}
C_OK, C_BAD, C_INK = OI["bluish_green"], OI["vermillion"], OI["black"]

#: Loss above which a draw counts as materially hurt by the nominal feed (%).
THRESH = 5.0


def _rhs(t, y, args):
    """xdot = f(x) + g(x) F(t), with F read off the stored profile.

    The stored feed is piecewise constant on a 400-point grid; jnp.interp makes it
    piecewise LINEAR, which rounds each switch over one grid spacing (0.099 h). That
    is deliberate -- an adaptive solver handles a kink far better than a jump, and
    0.099 h against a 39.5 h horizon moves PV by far less than the effect being
    measured.

    The feed is cut once the reactor is full (V >= V_max), smoothly over 1 mL. The
    stored 400-point profile integrates to slightly MORE than the exact optimum's
    volume (V(t_f) ~ 10.07 L, not 10 L), and the extra feed is extra substrate and
    extra product: without the cut the nominal feed on the nominal model gives
    570.7 g against its own optimum of 566.5 g, and 39 of 82 draws showed a
    NEGATIVE loss -- the reference feed "beating" the draw's own optimum, which is
    impossible. With it the nominal case reproduces 566.4 g and no loss is < 0.
    A real reactor stops feeding when full, so this is also the physical answer.
    """
    t_grid, F_grid, S_F, p = args
    full = jnp.clip((CFG.V_max - y[3]) / 1e-3, 0.0, 1.0)     # 1 below V_max, 0 at it
    return f_vec(y, p) + g_vec(y, S_F) * jnp.interp(t, t_grid, F_grid) * full


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
    # Sanity checks: the nominal feed must reproduce its own optimum on the nominal
    # model, and no feed may beat a draw's own optimum (the histogram starts at 0).
    from singulax.substrate_product_inhibition.kinetics import BALANCED
    pv_check = float(simulate(tj, Fnj, BALANCED))
    print(f"check: nominal feed on nominal model {pv_check:.2f} g "
          f"(stored optimum {pv_nom_own:.2f} g)")
    if (loss_b < 0).any():
        print(f"   WARNING: {(loss_b < 0).sum()} negative loss(es), min "
              f"{loss_b.min():.3f} % -- these fall outside the histogram")

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
    # Single panel: the loss distribution. (The PV-vs-PV scatter that used to sit
    # beside it was dropped from the paper; the numbers above still report it.)
    fig, axR = plt.subplots(figsize=(5.0, 3.3), constrained_layout=True)

    costly = loss_b > THRESH
    n_bad = int(costly.sum())

    # --- (b) the loss distribution ----------------------------------------------
    # Bin width divides THRESH exactly, so the threshold is a bin EDGE and no bar
    # straddles it. Without that, the bar spanning 4-5.5 % sits half on each side of
    # the line and the split stops being readable off the bars. Colour carries the
    # 5 % split reported in the console summary.
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
    # axes meet at (0, 0): no margin before the first bin or under the bars
    axR.set_xlim(0.0, edges[-1])
    axR.set_ylim(bottom=0)
    axR.margins(x=0, y=0)
    axR.set_ylim(bottom=0, top=axR.get_ylim()[1] * 1.05)
    axR.legend(loc="upper right", handletextpad=0.5, borderpad=0.2,
               labelspacing=0.35)

    fig.savefig(FIGURE, dpi=600, bbox_inches="tight")
    print(f"saved {FIGURE}")


if __name__ == "__main__":
    main()
