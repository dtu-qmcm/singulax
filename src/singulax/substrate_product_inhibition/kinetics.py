"""Specific rates -- growth (mu), substrate uptake (sigma) and product formation (pi)
-- for the substrate+product inhibition case. From Shin & Lim, "Fed-Batch Cultures",
example 13.E.3 (ethanol from glucose by S. cerevisiae). Each is a function of the
substrate S and product P concentrations.
"""
import jax
from jaxtyping import Array, Float, ScalarLike
import jax.numpy as jnp

# SUBSTRATE-PRODUCT INHIBITION
def mu(S: ScalarLike, P: ScalarLike) -> ScalarLike:
    """Growth rate function.

    Taken from textbook "Fed Batch Cultures" by Shin and Lim (specifically example 13.E.3 "ethanol from glucose by s. cerevisiae").

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.

    Returns:
        float: Specific growth rate of the microorganism.
    """
    return 0.408 * S / ((1 + P / 16) * (0.22 + S))


def sigma(S: ScalarLike, P: ScalarLike) -> ScalarLike:
    """Substrate consumption function.

    Taken from textbook "Fed Batch Cultures" by Shin and Lim (specifically example 13.E.3 "ethanol from glucose by s. cerevisiae").

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.

    Returns:
        float: Specific substrate consumption rate.
    """
    return 10 * mu(S, P)


def pi(S: ScalarLike, P: ScalarLike) -> ScalarLike:
    """Specific productivity function.

    Taken from textbook "Fed Batch Cultures" by Shin and Lim (specifically example 13.E.3 "ethanol from glucose by s. cerevisiae").

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.

    Returns:
        float: Specific productivity rate.
    """
    return S / ((1 + P / 71.5) * (0.44 + S))
