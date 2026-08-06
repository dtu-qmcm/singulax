"""singulax -- hybrid direct-indirect optimal control for fed-batch fermentation.

Shared collocation utilities live in ``singulax.hybrid``. Each inhibition case is a
self-contained subpackage (kinetics / odes / singular control / shooting):

    singulax.substrate_inhibition          -- substrate inhibition (Haldane)
    singulax.substrate_product_inhibition  -- substrate + product inhibition

Case-specific symbols (``mu``, ``sigma``, ``get_phi``, ``vector_field``,
``get_Fsin``, ...) exist in *both* cases and therefore are NOT re-exported at the
top level; import them from the relevant subpackage instead, e.g.::

    from singulax.substrate_product_inhibition import get_phi, solve_and_get_residuals_ms
    from singulax.hybrid import CollocParams
"""
from singulax import collocation
from singulax import substrate_inhibition
from singulax import substrate_product_inhibition

__all__ = ["collocation", "substrate_inhibition", "substrate_product_inhibition"]
