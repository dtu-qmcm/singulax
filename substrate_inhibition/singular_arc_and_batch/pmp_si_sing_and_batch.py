import optimistix as optx
import jax
import os
import diffrax
import jax.numpy as jnp
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, Dopri5
import matplotlib.pyplot as plt
from fedopt.si_analytical import states_analytical_sol, calculate_real_feed, volume_cond
from fedopt.shooting import loss_fn_batch
from fedopt.odes import vector_field, get_phi
from fedopt.Fsin import get_Fsin_unconstrained, get_Fsin

jax.config.update("jax_enable_x64", True)

Ks = 0.1
Ki = 100
S0 = jnp.sqrt(Ks*Ki)
y0 = jnp.array([10, jnp.log(S0*1), 0., 1.])
S_F = 150
V_max = 4.0
F_max = 2.0
F_min = 0.0
t_end = 9.0

# Get x_tf directly from forward analytical solution
root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event = diffrax.Event(volume_cond, root_finder)

# Run the SINGULAR phase to find t_switch
sol = diffeqsolve(
    ODETerm(states_analytical_sol), 
    Kvaerno5(),
    t0=0, t1=t_end, dt0=0.01, y0=y0,
    event=event,
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    args=(S_F, V_max)
)
t_switch = sol.ts.squeeze()

print(f"Switching time:", t_switch)

# Solve ODE in two phases
solution_to_tswitch = diffeqsolve(
    ODETerm(states_analytical_sol), 
    Kvaerno5(),
    t0=0, t1=t_switch, dt0=0.01, y0=y0,
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_switch, 500)),
    args=(S_F, V_max)
)
y_at_switch = solution_to_tswitch.ys[-1]

solution_to_end = diffeqsolve(ODETerm(states_analytical_sol),
                              Kvaerno5(),
                              stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
                              t0=t_switch, t1=t_end, dt0=0.01,
                              y0=y_at_switch,
                              saveat= diffrax.SaveAt(ts=jnp.linspace(t_switch, t_end, 500)),
                              args=(S_F, V_max),
                              max_steps=int(1e6)                              
                              )

XV_tf = solution_to_end.ys[-1, 0]
SV_tf = solution_to_end.ys[-1, 1]
PV_tf = solution_to_end.ys[-1, 2]
V_tf = solution_to_end.ys[-1, 3]

x_tf =  jnp.array([XV_tf,SV_tf, PV_tf, V_tf])
print("Final states gotten from analytical solution:", jnp.array([XV_tf,jnp.exp(SV_tf), PV_tf, V_tf]))

get_Fs = jax.vmap(lambda y: calculate_real_feed(jnp.array([y[0], jnp.exp(y[1]), y[2], y[3]]), (S_F, V_max, Ks, Ki)))
Fs = get_Fs(solution_to_end.ys)

# Singular arc phase
if __name__ == "__main__":
    print("=== Optimal Control Optimization for Singular Arc ===\n")
    l4_tf_guess = -0.1
    print("Final guess l4tf = ",l4_tf_guess)

    solver = optx.BFGS(rtol=1e-5, atol=1e-6)
    print("\nStarting optimization")
    solution = optx.minimise(
        fn = loss_fn_batch,
        solver=solver,
        y0=l4_tf_guess,
        args=(t_end, y0, S_F, x_tf, t_switch, V_max, F_min, F_max),
        max_steps=500
    )

    if solution is not None:
        optimal_l4_guess = solution.value
        print("Optimal solution found!:", optimal_l4_guess)
        y_tf_opt = jnp.array([x_tf[0], x_tf[1], x_tf[2], x_tf[3], 0.0, 0.0, 1.0, optimal_l4_guess])
        sol_to_tswitch= diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: vector_field(t, y, (S_F, 0.0))),
            solver=diffrax.Kvaerno5(),
            t0=t_end,
            t1=t_switch,
            dt0=-0.01,
            y0=y_tf_opt,
            stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8),
            max_steps=int(1e5),
            saveat=diffrax.SaveAt(ts=jnp.linspace(t_end, t_switch, 500))
            )
        y_at_tswitch = sol_to_tswitch.ys[-1]

        sol_to_t0= diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))))),
            solver=diffrax.Kvaerno5(),
            t0=t_switch,
            t1=0.0,
            dt0=-0.01,
            y0=y_at_tswitch,
            stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8),
            max_steps=int(1e5),
            saveat=diffrax.SaveAt(ts=jnp.linspace(t_switch, 0.0, 500))
            )
        y_at_t0 = sol_to_t0.ys[-1]

        t_plot = jnp.concatenate([sol_to_tswitch.ts, sol_to_t0.ts])
        y_plot = jnp.concatenate([sol_to_tswitch.ys, sol_to_t0.ys])

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.titlesize'] = 12
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.facecolor'] = 'white'
        plt.rcParams['legend.edgecolor'] = 'gray'
        fig, axes = plt.subplots(5, 1, figsize=(12, 16), layout='constrained')
        fig.suptitle('PMP shooting during singular arc for S=S*', fontsize=28)

        # Plot State variables
        axes[0].plot(t_plot, y_plot[:,0], label='(XV)', color='black', linestyle='--')
        axes[0].plot(t_plot, jnp.exp(y_plot[:,1]), label='(SV)', color='black', marker='s', linestyle='-', markevery=50)
        axes[0].plot(t_plot, y_plot[:,2], label='(PV)', color='black', linestyle='-')
        axes[0].plot(t_plot, y_plot[:,3], label='V', color='black', marker='o', linestyle='-', markevery=50)
        axes[0].set_ylabel('State (g or V)')
        axes[0].set_xlabel('Time (h)')
        axes[0].set_title('State variables')
        axes[0].grid(True)
        axes[0].legend()

        # Plot S and P concentration
        axes[1].plot(t_plot, jnp.array(jnp.exp(y_plot[:,1])) / jnp.array(y_plot[:,3]), label='S', color='black', linestyle='-')
        axes[1].plot(t_plot, jnp.array(y_plot[:,2]) / jnp.array(y_plot[:,3]), label='P', color='black', linestyle='--')
        axes[1].set_ylabel('Concentration (g/L)')
        axes[1].set_xlabel('Time (h)')
        axes[1].set_title('S and P concentration')
        axes[1].grid(True)
        axes[1].legend()

        phi_plot = jax.vmap(get_phi)(y_plot)
        F_sin_val = jax.vmap(lambda y: get_Fsin(y, (S_F, V_max, F_min, F_max)))(y_plot)
        F_plot = jnp.where(
            t_plot<t_switch,
            F_sin_val,
            F_min
        )
        axes[2].plot(t_plot, F_plot, label='Feed Rate', color='black')
        axes[2].set_ylabel('F (L/h)')
        axes[2].set_xlabel('Time (h)')
        axes[2].set_title('Feed rate')
        axes[2].grid(True)

        axes[3].plot(t_plot, phi_plot, color='black', label=r'$\phi(t)$ y', linestyle = '--')
        axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_title('Switching Function')
        axes[3].set_ylabel(r'$\phi(t)$')

        ax4 = axes[4]
        ax4b = ax4.twinx()
        axes[4].plot(t_plot, y_plot[:,4], label=r'$\lambda_1$', color='black', linestyle='--')
        axes[4].plot(t_plot, y_plot[:,5], label=r'$\lambda_2$', color='black', marker = 's', linestyle='-', markevery=50)
        axes[4].plot(t_plot, y_plot[:,6], label=r'$\lambda_3$', color='black', linestyle='-')
        ax4b.plot(t_plot, y_plot[:,7], label=r'$\lambda_4$', color='black', marker = 'o', linestyle='-', markevery=50)
        axes[4].set_ylabel(r'$\lambda_1, \lambda_2, \lambda_3$')
        ax4b.set_ylabel(r'$\lambda_4$')
        axes[4].set_xlabel('Time (h)')
        axes[4].set_title('Adjoints variables')
        axes[4].grid(True)
        axes[4].legend(loc='upper left')
        ax4b.legend(loc='upper right')

        print("Final XV value", y_plot[-1, 0])
        print("Final SV value", jnp.exp(y_plot[-1, 1]))
        print("Final PV value", y_plot[-1, 2])
        print("Final V value", y_plot[-1, 3])

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = "output_pmp_sing_and_batch.png"
        output_path = os.path.join(script_dir, file_name)
        plt.savefig(output_path)
        plt.show()
        
        # Create a new figure with 4 rows and 1 column
        fig_lam, axes_lam = plt.subplots(4, 1, figsize=(10, 12), layout='constrained')
        fig_lam.suptitle('Adjoint Variables (Lambdas) Detail', fontsize=20)
        
        # Lambda 1
        axes_lam[0].plot(t_plot, y_plot[:,4], color='black', linestyle='-')
        axes_lam[0].set_ylabel(r'$\lambda_1$')
        axes_lam[0].set_title('Adjoint for XV')
        axes_lam[0].grid(True)
        
        # Lambda 2
        axes_lam[1].plot(t_plot, y_plot[:,5], color='black', marker='s', linestyle='-', markevery=50)
        axes_lam[1].set_ylabel(r'$\lambda_2$')
        axes_lam[1].set_title('Adjoint for SV')
        axes_lam[1].grid(True)
        
        # Lambda 3
        axes_lam[2].plot(t_plot, y_plot[:,6], color='black', linestyle='--')
        axes_lam[2].set_ylabel(r'$\lambda_3$')
        axes_lam[2].set_title('Adjoint for PV')
        axes_lam[2].grid(True)
        
        # Lambda 4
        axes_lam[3].plot(t_plot, y_plot[:,7], color='black', marker='o', linestyle='-', markevery=50)
        axes_lam[3].set_ylabel(r'$\lambda_4$')
        axes_lam[3].set_title('Adjoint for V')
        axes_lam[3].set_xlabel('Time (h)')
        axes_lam[3].grid(True)
        
        file_name_lam = "output_lambdas_detail.png"
        output_path_lam = os.path.join(script_dir, file_name_lam)
        fig_lam.savefig(output_path_lam)
        plt.show()

