"""Substrate + product inhibition case (Shin & Lim ethanol fermentation)."""
from singulax.substrate_product_inhibition.kinetics import mu, sigma, pi
from singulax.substrate_product_inhibition.odes import (
    f_vec, g_vec, vf_states, get_dldt, get_phi,
    vector_field, states_ode, costate_ode,
)
from singulax.substrate_product_inhibition.Fsin import lie_bracket_fg, inner_bracket, get_Fsin, get_Fsin_unconstrained
from singulax.substrate_product_inhibition.shooting import solve_and_get_residuals_ms

__all__ = [
    "mu", "sigma", "pi",
    "f_vec", "g_vec", "vf_states", "get_dldt", "get_phi",
    "vector_field", "states_ode", "costate_ode",
    "lie_bracket_fg", "inner_bracket", "get_Fsin", "get_Fsin_unconstrained",
    "solve_and_get_residuals_ms",
]
