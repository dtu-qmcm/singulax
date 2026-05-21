"""
Kinetic model for microbial growth, product formation, and substrate consumption with inhibition and maintenance effects.
"""

def mu(S):
    """
    Calculate the specific growth rate as a function of substrate concentration.

    Uses a Monod-type expression with substrate inhibition.

    Args:
        S : float
            Substrate concentration.

    Returns: 
        float
            Specific growth rate.
    """
    mu_max = 0.3
    Ks = 0.1
    Ki = 100
    mu_val = (mu_max * S) / (Ks + S + S**2/Ki)

    return mu_val

def pi(S):
    """
    Calculate the product formation rate.

    Assumes growth-associated product formation proportional to growth rate.

    Args:
        S : float
            Substrate concentration.

    Returns:
        float
            Product formation rate.
    """
    mu_val = mu(S)
    pi_val = 0.5* mu_val

    return pi_val

def m_func(S):
    """
    Calculate substrate consumption due to cellular maintenance.

    Modeled using a saturation function similar to Monod kinetics.

    Args: 
        S : float
            Substrate concentration.

    Returns:
        float
            Maintenance-related substrate consumption rate.
    """
    m = 0.01
    Km = 0.001
    return m * (S / (S + Km))

def sigma(S):
    """
    Calculate the total substrate consumption rate.

    Combines contributions from:
    - Biomass growth
    - Product formation
    - Cellular maintenance

    Args:
        S : float
            Substrate concentration.

    Returns:
        float
            Total substrate consumption rate.
    """
    Y_xs = 0.5
    Y_ps = 0.2
    mu_val = mu(S)
    pi_val = pi(S)
    sigma_val = mu_val/Y_xs + pi_val/Y_ps + m_func(S)

    return sigma_val
