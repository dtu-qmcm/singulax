import csv
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)   # REQUIRED, and before any array exists:
                                            # in float32 the shooting residuals never
                                            # reach tolerance and every solve fails.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, norm, qmc, uniform
from singulax.substrate_product_inhibition import (
    solve_stage1, BALANCED, stoichiometry as st,
)

NOMINAL_STRUCTURE = (0, 1, 0)

CV = 0.20  # relative sd assumed for every parameter


def _lognormal(mu_log, sigma_log):
    """Lognormal from the mean and sd of ln(X)."""
    return lognorm(s=sigma_log, scale=np.exp(mu_log))


def _lognormal_mean(mean, cv=CV):
    """Lognormal with this mean and coefficient of variation."""
    sigma_log = np.sqrt(np.log(1 + cv**2))
    return _lognormal(np.log(mean) - sigma_log**2 / 2, sigma_log)


# Priors -- means are the Shin & Lim example 13.E.3 values, sd = 10% of the mean.
# Y_xs is NOT here: under the balanced model the biomass yield is an OUTPUT of
# the carbon/redox balance, not a free parameter (see .stoichiometry).
mu_max = _lognormal_mean(0.408)          # max specific growth rate [1/h]
k_p = _lognormal_mean(16.0)              # ethanol that halves growth [g/L]
k_s = _lognormal_mean(0.22)              # half-saturation glucose, growth [g/L]
pi_max = _lognormal_mean(1.0)            # max specific productivity [g EtOH/(g cell h)]
k_p_pi = _lognormal_mean(71.5)           # ethanol that halves productivity [g/L]
k_s_pi = _lognormal_mean(0.44)           # half-saturation glucose, production [g/L]

NAMES = ['mu_max', 'k_p', 'k_s', 'pi_max', 'k_p_pi', 'k_s_pi']
DISTS = [mu_max, k_p, k_s, pi_max, k_p_pi, k_s_pi]
M = len(DISTS)                           # number of input parameters


def sample_X(n, seed, method='lhs'):
    """Draw n scenarios -> array of shape (n, M) in real parameter units.

    method='lhs' gives a Latin hypercube design, 'mc' plain random sampling.
    """
    if method == 'lhs':
        u = qmc.LatinHypercube(d=M, seed=seed).random(n=n)
    elif method == 'mc':
        u = np.random.default_rng(seed).random((n, M))
    else:
        raise ValueError(f"unknown method {method!r}, expected 'lhs' or 'mc'")

    return np.stack([d.ppf(u[:, j]) for j, d in enumerate(DISTS)], axis=1)


N = 50                                   # Stage-1 solves; each takes ~20-90 s
SEED = 1234
HERE = Path(__file__).resolve().parent
# The CV is in the filenames: runs at different assumed uncertainty must not
# overwrite each other, which is exactly what happened the first time.
_TAG = f"balanced_cv{round(CV * 100):02d}"
RESULTS_CSV = HERE / f"sampling_spi_{_TAG}.csv"
FIGURE = HERE / f"sampling_spi_{_TAG}.png"
NOMINAL_PV = 608.05                      # PV(tf) of the BALANCED model at the literature kinetics

COLS = (["i", "ok", "status", "PV", "defect_norm", "structure", "seconds"] + NAMES
        + ["Y_xs_overall", "Y_ps_overall", "carbon_accounted"])


def params_from_row(row):
    """A sampled row -> KineticParams, keeping the BALANCED stoichiometry."""
    return BALANCED._replace(**dict(zip(NAMES, row)))


def _ledger(s1):
    """Overall yields from the collocation trajectory's mass ledger.

    Y_ps is the quantity the balance bounds: it can never exceed Gay-Lussac,
    whatever the kinetics. Y_xs is an output worth watching against the
    0.08-0.12 g/g typical of anaerobic yeast.
    """
    XV, SV, PV, V = s1.X[-1]
    consumed = s1.X[0, 1] + 150.0 * (V - s1.X[0, 3]) - SV
    if consumed <= 0:
        return np.nan, np.nan, np.nan
    Y_xs = (XV - s1.X[0, 0]) / consumed
    Y_ps = (PV - s1.X[0, 2]) / consumed
    return Y_xs, Y_ps, st.carbon_accounted(XV - s1.X[0, 0], PV, consumed)


def run_sampling(n=N, seed=SEED, out=RESULTS_CSV, method="lhs"):
    """Solve Stage 1 once per sampled parameter set, appending a row per sample."""
    X = sample_X(n, seed, method)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(COLS); fh.flush()
        t_run = time.time()
        for i, row in enumerate(X):
            t0 = time.time()
            try:
                s1 = solve_stage1(params_from_row(row))
                status = "ok" if s1.ok else f"defects {s1.defect_norm:.2e}"
                pv, dn, seq = s1.PV, s1.defect_norm, s1.seq
                yx, yp, ca = _ledger(s1)
            except Exception as exc:
                status = f"raised: {type(exc).__name__}"
                s1 = None; pv = dn = yx = yp = ca = np.nan; seq = ()
            dt = time.time() - t0
            w.writerow([i, int(s1.ok) if s1 is not None else 0, status, pv, dn,
                        "-".join(str(v) for v in seq), round(dt, 1)]
                       + list(row) + [yx, yp, ca])
            fh.flush()
            done, eta = i + 1, (time.time() - t_run) / (i + 1) * (n - i - 1)
            print(f"[{done:3d}/{n}] PV={pv:8.2f}  Y_ps={yp:.4f}  {seq}  "
                  f"{dt:5.1f}s  ETA {eta/60:5.1f} min   {'' if status=='ok' else status}",
                  flush=True)
    print(f"\nwrote {out}")
    return X


def _src(X, Y):
    """Standardised regression coefficients on log(X) -> log(Y), and the fit's R^2.

    For a monotone, near-linear response SRC^2 approximates the first-order Sobol
    index, at a fraction of the samples. R^2 is the check on that: high R^2 means
    the ranking is trustworthy, low R^2 means interactions matter and a proper
    variance-based method is needed.
    """
    lx = np.log(X); lx = (lx - lx.mean(0)) / lx.std(0)
    ly = np.log(Y); ly = (ly - ly.mean()) / ly.std()
    A = np.c_[np.ones(len(ly)), lx]
    beta, *_ = np.linalg.lstsq(A, ly, rcond=None)
    r2 = 1 - ((ly - A @ beta) ** 2).sum() / (ly ** 2).sum()
    return beta[1:], float(r2)


def analyse(path=RESULTS_CSV, figure=FIGURE):
    """Summary table + figure from a results CSV. Safe to re-run without re-solving."""
    with open(path, newline="") as fh:                # csv, not genfromtxt: status
        rows = list(csv.DictReader(fh))               # strings contain commas
    ok = np.array([r["ok"] == "1" for r in rows])
    status = np.array([r["status"] for r in rows])
    structure = np.array([r["structure"] for r in rows])
    PV = np.array([float(r["PV"]) for r in rows])
    X_all = np.array([[float(r[n]) for n in NAMES] for r in rows])
    Y_ps = np.array([float(r["Y_ps_overall"]) for r in rows])
    Y_xs = np.array([float(r["Y_xs_overall"]) for r in rows])
    C_acc = np.array([float(r["carbon_accounted"]) for r in rows])
    Y, X = PV[ok], X_all[ok]
    d = dict(ok=ok, status=status, structure=structure, PV=PV, X=X_all)

    print(f"\n=== {len(rows)} samples: {ok.sum()} converged, {(~ok).sum()} failed ===")
    for msg in np.unique(status[~ok]):
        print(f"   {int((status[~ok] == msg).sum()):3d} x  {msg}")

    print("\narc structure (2=Fmax bang, 1=singular, 0=Fmin/batch):")
    counts = {s: int((structure == s).sum()) for s in np.unique(structure)}
    modal = max(counts, key=counts.get)
    nominal = "-".join(str(k) for k in NOMINAL_STRUCTURE)
    for s, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        flag = "  <- most common" if s == modal else ""
        flag += "  (literature kinetics)" if s == nominal else ""
        print(f"   {c:3d} x  ({s}){flag}")

    if ok.sum() < 3:
        print("\nfewer than 3 converged draws -- nothing to summarise. "
              "Check the status column before re-running.")
        return d, None, None

    print(f"\nPV(tf) over the {ok.sum()} converged draws:")
    for label, v in [("mean", Y.mean()), ("sd", Y.std(ddof=1)), ("cv", Y.std(ddof=1) / Y.mean()),
                     ("min", Y.min()), ("2.5%", np.percentile(Y, 2.5)),
                     ("median", np.median(Y)), ("97.5%", np.percentile(Y, 97.5)), ("max", Y.max())]:
        print(f"   {label:>7s} = {v:9.3f}")

    print("\nbalance diagnostics (these are what the reference model could not satisfy):")
    print(f"   ethanol yield Y_ps   {np.nanmin(Y_ps[ok]):.4f} - {np.nanmax(Y_ps[ok]):.4f} g/g"
          f"   (Gay-Lussac {st.GAY_LUSSAC:.4f}; violations: {int((Y_ps[ok] > st.GAY_LUSSAC + 1e-9).sum())})")
    print(f"   biomass yield Y_xs   {np.nanmin(Y_xs[ok]):.4f} - {np.nanmax(Y_xs[ok]):.4f} g/g"
          f"   (an OUTPUT; anaerobic yeast 0.08-0.12)")
    print(f"   carbon as X+P        {np.nanmin(C_acc[ok]):.4f} - {np.nanmax(C_acc[ok]):.4f}"
          f"   (rest is CO2; must stay below 1)")

    beta, r2 = _src(X, Y)
    n, m = X.shape
    r2_adj = 1 - (1 - r2) * (n - 1) / max(n - m - 1, 1)
    order = np.argsort(-beta ** 2)
    print(f"\nsensitivity (R^2 = {r2:.4f}, adjusted R^2 = {r2_adj:.4f}, "
          f"sum of SRC^2 = {np.sum(beta ** 2):.3f}):")
    print(f"   {'parameter':10s} {'SRC':>8s} {'SRC^2':>8s}   share of explained variance")
    for i in order:
        bar = "#" * int(round(40 * beta[i] ** 2 / max(beta ** 2)))
        print(f"   {NAMES[i]:10s} {beta[i]:+8.3f} {beta[i]**2:8.3f}   {bar}")
    if n < 3 * m:
        print(f"   WARNING: {n} converged draws for {m} parameters. The regression is nearly")
        print("            saturated, so R^2 is inflated and the ranking is not yet reliable.")
        print(f"            Aim for at least {3 * m} converged draws, ideally {10 * m}.")
    if np.sum(beta ** 2) > 1.1:
        print(f"   WARNING: sum of SRC^2 = {np.sum(beta ** 2):.2f} > 1. With independent inputs this")
        print("            should be <= 1; a value above it means the fit is overdetermined.")
    elif r2_adj < 0.9:
        print(f"   NOTE: adjusted R^2 = {r2_adj:.2f} < 0.9 -- interactions are material and SRC^2")
        print("         understates them; a variance-based (Sobol) analysis is warranted.")

    _plot(X, Y, beta, r2, figure)
    return d, beta, r2


def _plot(X, Y, beta, r2, figure):
    """Three views: where PV landed, what moved it, and the shape of each
    dependence. The feed-profile ensemble is a separate study -- see
    feed_ensemble.py, which uses the hybrid rather than Stage 1."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.8, "legend.frameon": False,
        "figure.dpi": 300, "savefig.dpi": 600,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-", "grid.linewidth": 0.5,
    })
    oi = {"orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
          "vermillion": "#D55E00", "black": "#000000", "reddish_purple": "#CC79A7"}

    # One column per parameter, so panel (c) fits on a single row whatever the
    # number of parameters. (a) and (b) split the top row between them.
    m = X.shape[1]
    fig = plt.figure(figsize=(2.0 * m, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, m)

    # (a) where PV landed
    ax = fig.add_subplot(gs[0, :m // 2])
    ax.hist(Y, bins=max(8, int(np.sqrt(len(Y)))), color=oi["sky_blue"], edgecolor="white", linewidth=0.6)
    ax.axvline(Y.mean(), color=oi["black"], lw=1.2, label=f"mean {Y.mean():.0f} g")
    ax.axvline(NOMINAL_PV, color=oi["vermillion"], lw=1.2, ls="--", label=f"nominal {NOMINAL_PV:.0f} g")
    ax.set_xlabel(r"$PV(t_f)$ (g)"); ax.set_ylabel("count")
    ax.set_title(f"(a) optimal product (balanced), n = {len(Y)}   CV = {Y.std(ddof=1)/Y.mean():.3f}", loc="left")
    ax.legend(loc="best")

    # (b) what moved it -- sign is real information, so colour encodes direction only
    ax = fig.add_subplot(gs[0, m // 2:])
    order = np.argsort(beta ** 2)
    cols = [oi["bluish_green"] if beta[i] > 0 else oi["vermillion"] for i in order]
    ax.barh(range(len(order)), (beta ** 2)[order], color=cols, height=0.65)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[i] for i in order])
    for k, i in enumerate(order):
        ax.text((beta[i] ** 2) + 0.006, k, f"{beta[i]:+.2f}", va="center", fontsize=7.5)
    ax.set_xlabel(r"$\mathrm{SRC}^2$  (share of variance)")
    ax.set_xlim(0, max(beta ** 2) * 1.22)
    ax.set_title(f"(b) sensitivity,  $R^2$ = {r2:.3f}", loc="left")
    ax.legend(handles=[plt.Line2D([], [], color=oi["bluish_green"], lw=6, label="increases PV"),
                       plt.Line2D([], [], color=oi["vermillion"], lw=6, label="decreases PV")],
              loc="lower right")

    # (c) the shape of each dependence
    rank = np.argsort(-beta ** 2)
    for k, i in enumerate(rank):
        ax = fig.add_subplot(gs[1, k])
        ax.plot(X[:, i], Y, "o", ms=3.5, color=oi["sky_blue"], mew=0)
        c = np.polyfit(X[:, i], Y, 1)
        xs = np.linspace(X[:, i].min(), X[:, i].max(), 20)
        ax.plot(xs, np.polyval(c, xs), color=oi["black"], lw=1.0)
        ax.set_xlabel(NAMES[i]); ax.set_title(f"SRC {beta[i]:+.2f}", loc="left", fontsize=8)
        if k == 0:
            ax.set_ylabel(r"$PV(t_f)$ (g)")

    fig.suptitle("Kinetic parameter sampling: carbon- and redox-balanced model", fontsize=10)
    fig.savefig(figure, dpi=600, bbox_inches="tight")
    print(f"\nsaved {figure}")


if __name__ == "__main__":
    if "--analyse" not in sys.argv:      # --analyse re-reads the CSV without re-solving
        run_sampling()
    analyse()
