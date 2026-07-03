import jax
import os
import jax.numpy as jnp
import numpy as np
import diffrax
import interpax
from diffrax import ODETerm, Kvaerno5, diffeqsolve, PIDController, SaveAt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from fedopt.odes import vf_states
from fedopt.si_kinetics import sigma

jax.config.update("jax_enable_x64", True)

S0         = 1e-6
Fmax       = 2.0
S_F        = 150.0
V_max      = 4.0
t_f        = 9.0
y0         = jnp.array([10.0, jnp.log(S0), 0.0, 1.0])

Ks, Ki     = 0.1, 100.0
S_star     = float(jnp.sqrt(Ks * Ki))
sigma_star = float(sigma(jnp.array(S_star)))

n_knots    = 10
knot_times = jnp.linspace(0.0, t_f, n_knots)

def get_F(t, t1, F_knots, F_max):
    cs = interpax.CubicSpline(knot_times, F_knots, check=False)
    F_sing = jnp.clip(cs(t), 0.0, F_max)
    return jnp.where(t < t1, F_max, F_sing)

def ode_rhs(t, y, args):
    opt_vars, fixed = args
    t1      = opt_vars[0]
    F_knots = opt_vars[1:]
    S_F, V_max, F_max, _ = fixed
    F = jnp.where(y[3] < V_max, get_F(t, t1, F_knots, F_max), 0.0)
    return vf_states(t, y, (S_F, F))

def _solve(opt_vars, fixed, saveat, max_steps=int(1e5)):
    _, _, _, t_f = fixed
    return diffeqsolve(
        ODETerm(ode_rhs),
        Kvaerno5(),
        t0=0.0, t1=t_f, dt0=1e-6,
        y0=y0,
        args=(opt_vars, fixed),
        stepsize_controller=PIDController(rtol=1e-6, atol=1e-6,
                                          dtmin=1e-12, dtmax=0.5),
        saveat=saveat,
        max_steps=max_steps,
        throw=False,
    )

def simulate_final(opt_vars, fixed):
    y_tf = _solve(opt_vars, fixed, SaveAt(t1=True)).ys[0]
    return jnp.where(jnp.any(jnp.isnan(y_tf)), jnp.zeros(4), y_tf)

def simulate_full(opt_vars, fixed, n_pts=500):
    _, _, _, t_f = fixed
    ts = jnp.linspace(0.0, t_f, n_pts)
    sol = _solve(opt_vars, fixed, SaveAt(ts=ts), max_steps=int(5e5))
    return sol.ts, sol.ys

@jax.jit
def objective(opt_vars, fixed):
    return -simulate_final(opt_vars, fixed)[2]

fixed = (S_F, V_max, Fmax, t_f)

t1_guess  = 0.01
#Fk_guess  = jnp.array([0.1, 0.15, 0.2, 0.23, 0.25, 0.4, 0.5, 0.6, 0.8, 1.0])
Fk_guess  = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
opt_guess = jnp.concatenate([jnp.array([t1_guess]), Fk_guess])
print(f"Initial guess: t1 = {t1_guess} h,  F_knots = {np.round(np.array(Fk_guess), 3)}")

bounds = [(0.005, 0.5)] + [(0.05, Fmax)] * n_knots

print("Compiling ... ", end="", flush=True)
_ = objective(opt_guess, fixed)
print("done.")
print(f"Initial PV(tf) = {-float(objective(opt_guess, fixed)):.4f} g")

iteration = [0]
best_val = [float(objective(opt_guess, fixed))]
best_params = [np.array(opt_guess)]

def obj_np(p_np, fixed):
    val = float(objective(jnp.array(p_np), fixed))
    if val < best_val[0]:
        best_val[0]    = val
        best_params[0] = p_np.copy()
    iteration[0] += 1
    if iteration[0] % 10 == 0 or iteration[0] == 1:
        print(f"  iter {iteration[0]:4d}   PV(tf) = {-val:.4f} g   "
              f"best = {-best_val[0]:.4f} g   t1 = {p_np[0]:.5f} h")
    return val

print(f"\nRunning L-BFGS-B  ({n_knots} knots + t1, scipy finite-diff gradient) ...")
result = minimize(
    obj_np,
    np.array(opt_guess),
    args=(fixed,),
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-4, "eps": 1e-4},
)

params_opt = jnp.array(best_params[0])
t1_opt     = float(params_opt[0])
F_opt      = params_opt[1:]

print(f"\nOptimal solution found!")
print(f"t1        = {t1_opt:.6f} h   (PMP result: ~0.011114 h)")
print(f"Knot values (L/h): {np.round(np.array(F_opt), 4)}")

y_tf  = simulate_final(params_opt, fixed)
PV_tf = float(y_tf[2])
V_tf  = float(y_tf[3])
print(f"\nPV(t_f) = {PV_tf:.4f} g   P(t_f) = {PV_tf/V_tf:.4f} g/L   V(t_f) = {V_tf:.4f} L")

direct_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "direct_si_opt.npy")
np.save(direct_save, np.concatenate([np.array(y_tf), np.array([t1_opt])]))
print(f"Direct result saved → {direct_save}")

ts, ys = simulate_full(params_opt, fixed)
ts = np.array(ts)
ys = np.array(ys)

XV = ys[:, 0]
SV = np.exp(ys[:, 1])
PV = ys[:, 2]
V  = ys[:, 3]
S  = np.divide(SV, V, out=np.zeros_like(SV), where=V > 0)
P  = np.divide(PV, V, out=np.zeros_like(PV), where=V > 0)

F_plot = np.array([
    float(jnp.where(V[i] < V_max,
                    get_F(jnp.array(ts[i]), params_opt[0], F_opt, jnp.array(Fmax)),
                    0.0))
    for i in range(len(ts))
])

F_sin_ana = np.clip(sigma_star * XV / (S_F - S_star), 0.0, Fmax)

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9,  "figure.titlesize": 14,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f"Direct method — cubic spline  (N={n_knots} knots + t₁)\n"
    f"PV(t_f) = {PV_tf:.2f} g    t₁ = {t1_opt:.4f} h",
    fontsize=13,
)

ax = axes[0, 0]
ax.plot(ts, XV, label="XV (g)",  color="royalblue")
ax.plot(ts, SV, label="SV (g)",  color="orange")
ax.plot(ts, PV, label="PV (g)",  color="green")
ax.plot(ts, V,  label="V (L)",   color="purple")
ax.set_xlabel("Time (h)"); ax.set_ylabel("State")
ax.set_title("Extensive state variables"); ax.legend()

ax = axes[0, 1]
ax.plot(ts, S, label="S (g/L)", color="orange")
ax.plot(ts, P, label="P (g/L)", color="green")
ax.axhline(S_star, color="gray", linestyle="--", lw=1.2,
           label=f"S* = {S_star:.3f} g/L")
ax.set_xlabel("Time (h)"); ax.set_ylabel("Concentration (g/L)")
ax.set_title("Substrate & product"); ax.legend()

ax = axes[1, 0]
ax.plot(ts, F_plot,    color="tomato", lw=2,   label="F(t) direct (opt)")
ax.plot(ts, F_sin_ana, color="navy",   lw=1.5, linestyle="--",
        label="F_sin analytical (PMP)")
ax.scatter(np.array(knot_times), np.array(F_opt),
           color="red", zorder=5, s=60, label="Knots")
ax.axvline(t1_opt, color="black", linestyle=":", lw=1.2,
           label=f"t₁ = {t1_opt:.4f} h")
ax.set_ylim(-0.05, Fmax + 0.3)
ax.set_xlabel("Time (h)"); ax.set_ylabel("F (L/h)")
ax.set_title("Feed rate — direct vs PMP"); ax.legend()

ax = axes[1, 1]
ax.plot(ts, V, color="purple", lw=2)
ax.axhline(V_max, color="black", linestyle="--", lw=1.2,
           label=f"V_max = {V_max} L")
ax.set_xlabel("Time (h)"); ax.set_ylabel("V (L)")
ax.set_title("Reactor volume"); ax.legend()

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "direct_si_result.png")
plt.savefig(out, dpi=150)
plt.show()

