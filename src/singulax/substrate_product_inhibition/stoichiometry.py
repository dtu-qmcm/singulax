"""
Elemental and redox stoichiometry for glucose -> ethanol + biomass + glycerol.

The two coefficients in the substrate demand

    sigma = c_mu * mu  +  c_pi * pi          [g glucose /(g cell h)]

are derived here from elemental formulas.

Method: write the two macro-reactions on a C-mol basis and close them with carbon
plus the degree of reduction. After Schulze (1995, PhD thesis, DTU):

    R1 catabolic:  1.5 CH2O -> CH3O0.5 + 0.5 CO2
    R2 anabolic :  2 CH2O + 0.2 NH3 -> CH1.8O0.6N0.2 + (12/14) CH8/3O
                                       + (2/14) CO2 + 0.2571 H2O
"""
from typing import NamedTuple

# atomic masses
_C, _H, _O, _N = 12.011, 1.008, 15.999, 14.007


class Species(NamedTuple):
    """A compound written per carbon atom (C-mol basis)."""
    name: str
    C: float
    H: float
    O: float
    N: float

    @property
    def M(self) -> float:
        """C-molar mass [g/C-mol]."""
        return self.C * _C + self.H * _H + self.O * _O + self.N * _N

    @property
    def gamma(self) -> float:
        """Degree of reduction: available electrons per C-mol."""
        return 4 * self.C + self.H - 2 * self.O - 3 * self.N


GLUCOSE  = Species("glucose",  1, 2.000, 1.0,  0.0)     # CH2O
BIOMASS  = Species("biomass",  1, 1.800, 0.6,  0.2)     # CH1.8O0.6N0.2
ETHANOL  = Species("ethanol",  1, 3.000, 0.5,  0.0)     # CH3O0.5
CO2      = Species("CO2",      1, 0.000, 2.0,  0.0)
WATER    = Species("water",    0, 2.000, 1.0,  0.0)
GLYCEROL = Species("glycerol", 1, 8 / 3, 1.0,  0.0)     # CH8/3O
AMMONIA  = Species("ammonia",  0, 3.000, 0.0,  1.0)

GLYCEROL_K = 6.0 / 7.0

def glucose_per_gram(product: Species,
                     yield_cmol: float,
                     substrate: Species = GLUCOSE) -> float:
    """g substrate consumed per g of `product`, given the reaction's C-mol yield."""
    return (1.0 / yield_cmol) * (substrate.M / product.M)

def r1_yield() -> float:
    """C-mol ethanol per C-mol glucose in the catabolic reaction."""
    return GLUCOSE.gamma / ETHANOL.gamma                      # 4/6 = 0.6667


def r2_yield(k: float = GLYCEROL_K) -> float:
    """C-mol biomass per C-mol glucose in the anabolic reaction."""
    return GLUCOSE.gamma / (BIOMASS.gamma + k * GLYCEROL.gamma)


def r2_yields(k: float = GLYCEROL_K) -> tuple:
    """(Yx, Yg, Yc) per C-mol glucose: biomass, glycerol, CO2."""
    Yx = r2_yield(k)
    return Yx, k * Yx, 1.0 - Yx * (1.0 + k)


def coefficients(k: float = GLYCEROL_K) -> tuple:
    """(c_mu, c_pi) for sigma = c_mu*mu + c_pi*pi, in g glucose /(g cell h)."""
    return (glucose_per_gram(BIOMASS, r2_yield(k)),
            glucose_per_gram(ETHANOL, r1_yield()))

C_MU, C_PI = coefficients()

# The reference (Hong / Shin & Lim) model in the same form: substrate is charged
# for growth only, at 1/Y_xs = 10, and ethanol is produced free of charge.
REFERENCE_C_MU, REFERENCE_C_PI = 10.0, 0.0
GAY_LUSSAC = 1.0 / C_PI

def carbon_accounted(mu_val, pi_val, sigma_val: float):
    """Fraction of substrate carbon appearing as biomass + ethanol.

    The remainder leaves as CO2, so this is strictly less than 1 for any real
    fermentation. A value ABOVE 1 means the products contain more carbon than
    the substrate supplied -- carbon created from nothing.
    """
    c_in = sigma_val / GLUCOSE.M
    return (mu_val / BIOMASS.M + pi_val / ETHANOL.M) / c_in if c_in else float("nan")
