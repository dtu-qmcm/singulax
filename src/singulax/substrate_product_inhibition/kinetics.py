"""Specific rates -- growth (mu), substrate uptake (sigma) and product formation (pi)
-- for the substrate+product inhibition case. The kinetic constants are Shin & Lim's,
but the substrate demand defaults to the carbon/redox-balanced coefficients;
pass REFERENCE to reproduce the published results."""

from typing import NamedTuple
from jaxtyping import ScalarLike
from singulax.substrate_product_inhibition.stoichiometry import (C_MU, C_PI, REFERENCE_C_MU, REFERENCE_C_PI)

class KineticParams(NamedTuple):
    """The six kinetic constants plus the two stoichiometric coefficients.

    The six kinetic ones are what a sampling study varies. `c_mu` and `c_pi` are
    NOT uncertain parameters -- they select which substrate-demand model is in
    force, and in the balanced case they come from elemental formulas (see
    `.stoichiometry`). Both models are the same equation:

        REFERENCE  sigma = 10.0000*mu + 0.0000*pi   ethanol costs no substrate
        BALANCED   sigma =  1.2802*mu + 1.9553*pi   carbon and redox close

    so the reference model is the c_pi = 0 corner of a stoichiometric family.

    Defaults are the mass balanced values obtained from stoichiometry with the
    BALANCED coefficients """

    mu_max: ScalarLike = 0.408      # max specific growth rate [1/h]
    k_p: ScalarLike = 16.0          # ethanol that halves growth [g/L]
    k_s: ScalarLike = 0.22          # half-saturation glucose, growth [g/L]
    pi_max: ScalarLike = 1.0        # max specific productivity [g EtOH/(g cell h)]
    k_p_pi: ScalarLike = 71.5       # ethanol that halves productivity [g/L]
    k_s_pi: ScalarLike = 0.44       # half-saturation glucose, production [g/L]
    c_mu: ScalarLike = C_MU   # g glucose per g biomass
    c_pi: ScalarLike = C_PI   # g glucose per g ethanol


#: The published model: substrate charged for growth only (c_pi = 0).
REFERENCE = KineticParams(c_mu=REFERENCE_C_MU, c_pi=REFERENCE_C_PI)
#: Carbon- and redox-closed: substrate charged for both products.
BALANCED = KineticParams()

DEFAULT_PARAMS = BALANCED


# SUBSTRATE-PRODUCT INHIBITION
def mu(S: ScalarLike, P: ScalarLike, mu_max=BALANCED.mu_max, K_p=BALANCED.k_p, K_s=BALANCED.k_s) -> ScalarLike:
    """Growth rate function.

    Taken from textbook "Fed Batch Cultures" by Shin and Lim (specifically example 13.E.3 "ethanol from glucose by s. cerevisiae").

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.

    Returns:
        float: Specific growth rate of the microorganism.
    """
    return mu_max * S / ((1 + P / K_p) * (K_s + S))


def sigma(S: ScalarLike, P: ScalarLike,
          c_mu=BALANCED.c_mu, c_pi=BALANCED.c_pi,
          mu_max=BALANCED.mu_max, K_p=BALANCED.k_p, K_s=BALANCED.k_s,
          pi_max=BALANCED.pi_max, K_p_pi=BALANCED.k_p_pi, K_s_pi=BALANCED.k_s_pi) -> ScalarLike:
    """Substrate consumption function.

    sigma = c_mu*mu + c_pi*pi -- the glucose bill for the two products, at
    c_mu g glucose per g biomass and c_pi g glucose per g ethanol.

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.
        c_mu (float): g glucose consumed per g biomass formed.
        c_pi (float): g glucose consumed per g product formed.
        mu_max, K_p, K_s (float): passed through to `mu`.
        pi_max, K_p_pi, K_s_pi (float): passed through to `pi`.

    Returns:
        float: Specific substrate consumption rate.
    """
    return c_mu * mu(S, P, mu_max, K_p, K_s) + c_pi * pi(S, P, pi_max, K_p_pi, K_s_pi)


def pi(S: ScalarLike, P: ScalarLike, pi_max=BALANCED.pi_max, K_p_pi=BALANCED.k_p_pi, K_s_pi=BALANCED.k_s_pi) -> ScalarLike:
    """Specific productivity function.

    Taken from textbook "Fed Batch Cultures" by Shin and Lim (specifically example 13.E.3 "ethanol from glucose by s. cerevisiae").

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.

    Returns:
        float: Specific productivity rate.
    """
    return pi_max * S / ((1 + P / K_p_pi) * (K_s_pi + S))
