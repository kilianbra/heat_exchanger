import numpy as np


def sigma_tube_bank(tube_spacing_longitudinal_dimensionless):
    """This is the ratio of minimum free flow area to frontal area"""
    return (tube_spacing_longitudinal_dimensionless - 1) / tube_spacing_longitudinal_dimensionless


def area_frontal_bank(total_diameter_outer, total_diameter_inner):
    return np.pi * (total_diameter_outer**2 - total_diameter_inner**2) / 4


def area_free_flow_bank(area_frontal_bank, sigma_tube_bank):
    return area_frontal_bank * sigma_tube_bank


def area_free_flow_in_tubes(tube_diameter_inner, n_tubes_per_pass):
    return np.pi * tube_diameter_inner**2 * n_tubes_per_pass


def area_heat_transfer_bank(tube_diameter_outer, tube_length, n_rows, n_tubes_per_row):
    return np.pi * tube_diameter_outer * tube_length * n_rows * n_tubes_per_row


def area_heat_transfer_in_tubes(tube_diameter_inner, tube_length, n_tubes):
    return np.pi * tube_diameter_inner**2 * tube_length * n_tubes
