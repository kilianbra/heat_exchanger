import numpy as np


def dp_friction_only(
    area_friction_over_minimum_flow, mass_velocity, one_over_rho_mean, friction_factor
):
    return (
        1
        / 2
        * mass_velocity**2
        * friction_factor
        * area_friction_over_minimum_flow
        * one_over_rho_mean
    )


def dp_tube_bank(
    area_friction_over_minimum_flow,
    mass_velocity,
    rho_in,
    rho_out,
    sigma,
    friction_factor,
    one_over_rho_mean=None,
):
    if one_over_rho_mean is None:
        one_over_rho_mean = (1 / rho_in + 1 / rho_out) / 2
    dp_momentum = 1 / 2 * mass_velocity**2 * (1 + sigma**2) * (1 / rho_out - 1 / rho_in)
    dp_friction = (
        1
        / 2
        * mass_velocity**2
        * friction_factor
        * area_friction_over_minimum_flow
        * one_over_rho_mean
    )
    return dp_momentum + dp_friction


def epsilon(ntu, c_ratio):
    """Calculate effectiveness of counterflow heat_exchanger.

    Parameters
    ----------
    ntu : float
        Number of transfer units.
    c_ratio : float
        Ratio of heat capacity fluxes of the two fluids.

    Returns
    -------
    effectiveness : float
        Effectiveness of the heat exchanger.
    """
    tol = 1e-9
    if c_ratio < 1 - tol:
        return 1 - np.exp(-ntu * (1 - c_ratio)) / (1 - c_ratio * np.exp(-ntu * (1 - c_ratio)))
    else:
        return ntu / (ntu + 1)


def ntu(
    stanton_1,
    stanton_2,
    area_ratio_1_q_over_o,
    area_ratio_2_q_over_o,
    heat_capacity_flux_1,
    heat_capacity_flux_2,
):
    """
    Calculate the number of transfer units (NTU) for a heat exchanger.

    Parameters
    ----------
    stanton_1 : float
        Stanton number for fluid 1.
    stanton_2 : float
        Stanton number for fluid 2.
    area_ratio_1_q_over_o : float
        Ratio of heat transfer area to outside area for fluid 1.
    area_ratio_2_q_over_o : float
        Ratio of heat transfer area to outside area for fluid 2.
    heat_capacity_flux_1 : float
        Heat capacity flux for fluid 1.
    heat_capacity_flux_2 : float
        Heat capacity flux for fluid 2.

    Returns
    -------
    ntu : float
        Number of transfer units.
    """
    c_min = np.minimum(heat_capacity_flux_1, heat_capacity_flux_2)
    inv_ntu = (
        1 / stanton_1 / area_ratio_1_q_over_o * c_min / heat_capacity_flux_1
        + 1 / stanton_2 / area_ratio_2_q_over_o * c_min / heat_capacity_flux_2
    )
    return 1 / inv_ntu
