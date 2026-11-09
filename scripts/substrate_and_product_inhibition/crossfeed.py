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

#: Okabe-Ito. One series, so one hue; the identity line is ink, not a series.
C_B, C_INK = "#0072B2", "#444444"


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
    print(f"    above 5 % loss: {(loss_b > 5).sum()} of {n}\n")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
        "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 600,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)

    lo = min(pv_own.min(), pv_b.min()) - 10
    hi = max(pv_own.max(), pv_b.max()) + 10
    axL.plot([lo, hi], [lo, hi], color=C_INK, lw=1.0, ls="--", zorder=1)
    axL.scatter(pv_own, pv_b, s=16, color=C_B, alpha=0.8, zorder=3,
                edgecolor="white", linewidth=0.4)
    axL.annotate("no loss", (hi, hi), textcoords="offset points", xytext=(-4, -12),
                 ha="right", fontsize=7.5, color=C_INK)
    axL.set_xlabel(r"$PV$ under the model's own optimal feed (g)")
    axL.set_ylabel(r"$PV$ under the nominal feed (g)")
    axL.set_title("Nominal feed applied to each sampled model", loc="left")
    axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
    axL.set_aspect("equal", adjustable="box")

    # One series, so no legend box -- the title names it and the median is annotated.
    axR.hist(loss_b, bins=24, color=C_B, alpha=0.8)
    axR.axvline(5.0, color=C_INK, lw=1.0, ls="--", zorder=3)
    axR.annotate(f"median {q2:.2f} %\n{(loss_b > 5).sum()} of {n} above 5 %",
                 (0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                 fontsize=7.5, color=C_INK)
    axR.set_xlabel("Product loss vs. the model's own optimum (%)")
    axR.set_ylabel("Draws")
    axR.set_title("Cost of using the nominal feed", loc="left")

    fig.savefig(FIGURE, dpi=600, bbox_inches="tight")
    print(f"saved {FIGURE}")


if __name__ == "__main__":
    main()
