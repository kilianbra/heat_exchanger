import numpy as np


def dp(four_L_over_dh, mass_velocity, one_over_rho_mean, friction_factor):
    return 1 / 2 * mass_velocity**2 * friction_factor * four_L_over_dh * one_over_rho_mean


def ntu(
    stanton_1,
    stanton_2,
    area_ratio_1_q_over_o,
    area_ratio_2_q_over_o,
    heat_capacity_flux_1,
    heat_capacity_flux_2,
):
    c_min = np.minimum(heat_capacity_flux_1, heat_capacity_flux_2)
    inv_ntu = (
        1 / stanton_1 / area_ratio_1_q_over_o * c_min / heat_capacity_flux_1
        + 1 / stanton_2 / area_ratio_2_q_over_o * c_min / heat_capacity_flux_2
    )
    return 1 / inv_ntu


def epsilon(ntu, c_ratio):
    """Calculate effectiveness of counterflow heat_exchanger"""
    tol = 1e-9
    if c_ratio < 1 - tol:
        return 1 - np.exp(-ntu * (1 - c_ratio)) / (1 - c_ratio * np.exp(-ntu * (1 - c_ratio)))
    else:
        return ntu / (ntu + 1)
