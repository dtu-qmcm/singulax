"""Substrate inhibition case (Haldane-type kinetics)."""
from substrate_inhibition.kinetics import mu, pi, m_func, sigma
from substrate_inhibition.odes import (
    f_vec, g_vec, vf_states, get_dldt, get_phi,
    vector_field, hybrid_fedbatch, states_ode, costate_ode,
)
from substrate_inhibition.Fsin import lie_bracket_fg, inner_bracket, get_Fsin, get_Fsin_unconstrained
from substrate_inhibition.analytical import (
    states_analytical_sol, states_analytical_sol_singular,
    states_analytical_sol_batch, states_analytical_sol_bang,
    volume_cond, S_star_cond, calculate_real_feed,
)
from substrate_inhibition.shooting import (
    solve_and_get_residuals, solve_and_get_residuals_sing,
    solve_and_get_residuals_batch, solve_and_get_residuals_sing_and_batch,
    solve_and_get_residuals_bang_sing_batch,
    loss_fn, loss_fn_batch, loss_fn_sing_and_batch,
)

__all__ = [
    "mu", "pi", "m_func", "sigma",
    "f_vec", "g_vec", "vf_states", "get_dldt", "get_phi",
    "vector_field", "hybrid_fedbatch", "states_ode", "costate_ode",
    "lie_bracket_fg", "inner_bracket", "get_Fsin", "get_Fsin_unconstrained",
    "states_analytical_sol", "states_analytical_sol_singular",
    "states_analytical_sol_batch", "states_analytical_sol_bang",
    "volume_cond", "S_star_cond", "calculate_real_feed",
    "solve_and_get_residuals", "solve_and_get_residuals_sing",
    "solve_and_get_residuals_batch", "solve_and_get_residuals_sing_and_batch",
    "solve_and_get_residuals_bang_sing_batch",
    "loss_fn", "loss_fn_batch", "loss_fn_sing_and_batch",
]
