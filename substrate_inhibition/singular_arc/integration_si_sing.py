import jax
import diffrax
import os
import jax.numpy as jnp
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve
import matplotlib.pyplot as plt
from fedopt.si_analytical import states_analytical_sol, calculate_real_feed
from fedopt.odes import hybrid_fedbatch, get_phi


jax.config.update("jax_enable_x64", True)

Ks = 0.1
Ki = 100
S0 = jnp.sqrt(Ks*Ki)
S_F = 150
V_max = 5.0
F_max = 2.0
F_min = 0.0
t_end = 9.0

# Get x_switch directly from forward analytical solution
y0 = jnp.array([10, jnp.log(S0*1), 0., 1.])
sol_ana = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=0, t1=t_end, dt0=0.01, y0=y0,
    args=(S_F, V_max),
    stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
    saveat=SaveAt(ts=jnp.linspace(0, t_end, 2000))
)
XV_tf = sol_ana.ys[-1, 0]
SV_tf = sol_ana.ys[-1, 1]
PV_tf = sol_ana.ys[-1, 2]
V_tf = sol_ana.ys[-1, 3]

x_tf =  jnp.array([XV_tf, SV_tf, PV_tf, V_tf])
l_tf = jnp.array([0.0, 0.0, 1.0, 0.0])
y_tf = jnp.concatenate([x_tf, l_tf])

# Singular arc phase
ts_fb_back = jnp.linspace(t_end, 0.0, 500)
sol_fb_back = diffeqsolve(
    ODETerm(hybrid_fedbatch), Kvaerno5(),
    t0=t_end, t1=0.0, dt0=-0.01,
    y0=y_tf,
    args=(S_F, Ks, Ki),
    stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
    saveat=SaveAt(ts=ts_fb_back)
)

y0_recovered = sol_fb_back.ys[-1]

print(f"\nRecovered x(0) = {y0_recovered[:4]}")
print(f"True      x(0) = [10.0, {float(S0):.4f}, 0.0, 1.0]")
print(f"Costates  λ(0) = {y0_recovered[4:]}")

t_plot   = sol_fb_back.ts[::-1]         
y_plot   = sol_fb_back.ys[::-1]

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'gray'
fig, axes = plt.subplots(4, 1, figsize=(14, 15))
fig.suptitle('ODE integration for S=S* using analytical solution', fontsize=18)

# Plot State variables
axes[0].plot(t_plot, y_plot[:,0], label='(XV)', color='navy')
axes[0].plot(t_plot, y_plot[:,1], label='(SV)', color='darkorange')
axes[0].plot(t_plot, y_plot[:,2], label='(PV)', color='darkgreen')
axes[0].plot(t_plot, y_plot[:,3], label='V', color='rebeccapurple')
axes[0].set_ylabel('State (g or V)')
axes[0].set_xlabel('Time (h)')
axes[0].set_title('State variables')
axes[0].grid(True)
axes[0].legend()

# Plot S and P concentration
axes[1].plot(t_plot, jnp.array(y_plot[:,1]) / jnp.array(y_plot[:,3]), label='S', color='darkorange')
axes[1].plot(t_plot, jnp.array(y_plot[:,2]) / jnp.array(y_plot[:,3]), label='P', color='darkgreen')
axes[1].set_ylabel('Concentration (g/L)')
axes[1].set_xlabel('Time (h)')
axes[1].set_title('S and P concentration')
axes[1].grid(True)
axes[1].legend()


# Plot feed and phi
args_F = S_F, V_max, Ks, Ki
F_actual = jax.vmap(lambda y: calculate_real_feed(y[:4], args_F))(y_plot)
phi_plot = jax.vmap(get_phi)(y_plot)
axes[2].plot(t_plot, F_actual, label='Feed Rate', color='navy')
axes[2].set_ylabel('Feed rate (L/h)')
axes[2].set_xlabel('Time (h)')
axes[2].set_title('Feed rate')
axes[2].grid(True)

axes[2].plot(t_plot, phi_plot, color='rebeccapurple', label='$\phi(t)$')
axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[2].set_title('Switching Function')
axes[2].legend()

axes[3].plot(t_plot, y_plot[:,4], label='l1', color='navy', linestyle='--')
axes[3].plot(t_plot, y_plot[:,5], label='l2', color='darkorange', linestyle='--')
axes[3].plot(t_plot, y_plot[:,6], label='l3', color='darkgreen', linestyle='--')
axes[3].plot(t_plot, y_plot[:,7], label='l4', color='rebeccapurple', linestyle='--')
axes[3].set_ylabel('Costates')
axes[3].set_xlabel('Time (h)')
axes[3].set_title('Adjoints variables')
axes[3].grid(True)
axes[3].legend()

print("Final XV value", y_plot[-1, 0])
print("Final SV value", jnp.exp(y_plot[-1, 1]))
print("Final PV value", y_plot[-1, 2])
print("Final V value", y_plot[-1, 3])

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "output_integration_si_sing.png"
output_path = os.path.join(script_dir, file_name)
plt.savefig(output_path)