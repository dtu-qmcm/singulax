import jax
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, PIDController
from fedopt.si_analytical import states_analytical_sol, calculate_real_feed

jax.config.update("jax_enable_x64", True)

Ks = 0.1
Ki = 100
S0 = jnp.sqrt(Ks*Ki)
y0 = jnp.array([10, jnp.log(S0*1), 0., 1.])

S_F = 150
V_max = 5
term = ODETerm(states_analytical_sol)
stepsize_controller = PIDController(rtol=1e-6, atol=1e-6, dtmin=1e-12)
solver = Kvaerno5()
t1 = 9.0
ts = np.linspace(0, t1, 50)
saveat = SaveAt(ts=ts) # number of time points solved that we want to be shown

solution = diffeqsolve(term, solver, stepsize_controller=stepsize_controller,max_steps=65536, t0=0, t1=t1, dt0=0.1, y0=y0, args=(S_F, V_max), saveat=saveat)
t_max = solution.ts
y_max = solution.ys

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 26
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'gray'
fig, axes = plt.subplots(3, 1, figsize=(14, 15))
fig.suptitle('Analytical solution for S=S*', fontsize=28)

# Plot State variables
axes[0].plot(t_max, y_max[:,0], label='(XV)', color='royalblue')
axes[0].plot(t_max, jnp.exp(y_max[:,1]), label='(SV)', color='orange')
axes[0].plot(t_max, y_max[:,2], label='(PV)', color='green')
axes[0].plot(t_max, y_max[:,3], label='V', color='purple')
axes[0].set_ylabel('State (g or V)')
axes[0].set_xlabel('Time (h)')
axes[0].set_title('State variables')
axes[0].grid(True)
axes[0].legend()

# Plot S and P concentration
axes[1].plot(t_max, jnp.array(jnp.exp(y_max[:,1])) / y_max[:,3], label='S', color='orange')
axes[1].plot(t_max, jnp.array(y_max[:,2]) / jnp.array(y_max[:,3]), label='P', color='green')
axes[1].set_ylabel('Concentration (g/L)')
axes[1].set_xlabel('Time (h)')
axes[1].set_title('S and P concentration')
axes[1].grid(True)
axes[1].legend()

# Plot feed
args = S_F, V_max, Ks, Ki
F_actual = jax.vmap(lambda y: calculate_real_feed(jnp.array([y[0], jnp.exp(y[1]), y[2], y[3]]), args))(y_max)
axes[2].plot(t_max, F_actual, label='Feed Rate', color='royalblue')
axes[2].set_ylabel('Feed rate (L/h)')
axes[2].set_xlabel('Time (h)')
axes[2].set_title('Feed rate')
axes[2].grid(True)
axes[2].legend()

# Print value of final states
print("Final XV value", y_max[-1, 0])
print("Final SV value", jnp.exp(y_max[-1, 1]))
print("Final PV value", y_max[-1, 2])
print("Final V value", y_max[-1, 3])

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "output_ana_so_sing.png"
output_path = os.path.join(script_dir, file_name)
plt.savefig(output_path)
