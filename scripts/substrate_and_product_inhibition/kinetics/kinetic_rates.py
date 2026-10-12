"""Specific rates against substrate concentration."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from singulax.substrate_product_inhibition.kinetics import BALANCED, mu, pi, sigma

P_LEVELS = [0.0, 20.0, 40.0, 60.0, 80.0]
S_LIN = np.linspace(0.0, 120.0, 800)
S_LOG = np.logspace(-3, np.log10(120.0), 800)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 600,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
oi = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]


def curves(S, P):
    """(mu, pi, sigma) on a substrate grid at fixed product."""
    return mu(S, P), pi(S, P), sigma(S, P, BALANCED.c_mu, BALANCED.c_pi)


fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6), constrained_layout=True)
labels = [r"$\mu$ (h$^{-1}$)",
          r"$\pi$ (g EtOH (g cell h)$^{-1}$)",
          r"$\sigma$ (g glucose (g cell h)$^{-1}$)"]

for row, (S, scale) in enumerate([(S_LIN, "linear"), (S_LOG, "log")]):
    for k, P in enumerate(P_LEVELS):
        m, p, s = curves(S, P)
        axes[row, 0].plot(S, m, color=oi[k], label=f"{P:.0f}")
        axes[row, 1].plot(S, p, color=oi[k])
        axes[row, 2].plot(S, s, color=oi[k])
    for ax, lab in zip(axes[row], labels):
        ax.set_xscale(scale)
        ax.set_xlabel("S (g L$^{-1}$)")
        ax.set_ylabel(lab)

# Mark where the kinetics actually turn over -- invisible on the linear row.
for ax in axes[1]:
    ax.axvline(BALANCED.k_s, color="0.5", ls=":", lw=0.8)
    ax.axvline(BALANCED.k_s_pi, color="0.5", ls="-.", lw=0.8)

axes[0, 0].legend(title="P (g L$^{-1}$)", loc="lower right")
axes[1, 0].legend(handles=[plt.Line2D([], [], color="0.5", ls=":", label=r"$K_s$"),
                           plt.Line2D([], [], color="0.5", ls="-.", label=r"$K_{s,\pi}$")],
                  loc="lower right")
axes[0, 0].set_title("(a) linear S -- the operating range", loc="left")
axes[1, 0].set_title("(b) log S -- where the Monod term lives", loc="left")

fig.suptitle("Specific rates, carbon- and redox-balanced model", fontsize=10)
fig.savefig("rates_vs_S.png", dpi=600, bbox_inches="tight")
print("saved rates_vs_S.png")

# ---- the numbers behind the panels, so the figure can be checked by eye ----
print(f"\nplateau (S -> inf)")
print(f"{'P':>4}{'mu':>9}{'pi':>9}{'sigma':>9}")
for P in P_LEVELS:
    m, p, s = curves(np.array([1e6]), P)
    print(f"{P:>4.0f}{m[0]:>9.4f}{p[0]:>9.4f}{s[0]:>9.4f}")
