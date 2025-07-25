import numpy as np


def area_minimum_flow(area_frontal, sigma):
    return sigma * area_frontal


def sigma_tube_bank(tube_spacing_longitudinal_dimensionless):
    return (tube_spacing_longitudinal_dimensionless - 1) / tube_spacing_longitudinal_dimensionless


def area_frontal_bank(total_diameter_outer, total_diameter_inner):
    return np.pi * (total_diameter_outer**2 - total_diameter_inner**2) / 4


def area_heat_transfer_bank(tube_diameter_outer, tube_length, n_rows, n_tubes_per_row):
    return np.pi * tube_diameter_outer * tube_length * n_rows * n_tubes_per_row
