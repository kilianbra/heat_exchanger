import numpy as np

from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    tube_bank_nusselt_number_and_friction_factor,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu
from heat_exchanger.fluid_properties import (
    CombustionProductsProperties,
    CoolPropProperties,
    PerfectGasProperties,
    RefPropProperties,
)
from heat_exchanger.geometry_tube_bank import (
    area_free_flow_bank,
    area_free_flow_in_tubes,
    area_frontal_bank,
    area_heat_transfer_bank,
    area_heat_transfer_in_tubes,
    axial_involute_tube_length,
    sigma_tube_bank,
)
from heat_exchanger.hex_basic import dp_tube_bank, ntu

mdot_hot = 1144 / 60  # 19.07  # kg/s
mdot_cold = 9.95 / 60  # 0.166  # kg/s

model = "RP"  # "CP" or "PG"

match model:
    case "RP":
        hot_air = CombustionProductsProperties(fuel_type="H2", FAR_mass=9.95 / (1144 - 9.95), prefer_refprop=True)
        cold_hydrogen = RefPropProperties(fluid_name="PARAHYDROGEN")
    case "CP":
        hot_air = CoolPropProperties(fluid_name="Air")
        cold_hydrogen = CoolPropProperties(fluid_name="Hydrogen")
    case "PG":
        hot_air = PerfectGasProperties(molecular_weight=28.97, gamma=1.4, Pr=0.7, mu_ref=1.8e-5, T_ref=300.0, S=110.4)
        cold_hydrogen = PerfectGasProperties(
            molecular_weight=2.016, gamma=1.4, Pr=0.7, mu_ref=8.4e-6, T_ref=273.15, S=110.4
        )


# Brewer recuperator values
temp_hot_in = 778  # K
temp_cold_in = 264  # K

p_hot_in = 4e4  # Pa
p_cold_in = 17.3e5  # Pa

total_diameter_outer = 1.265  # m
total_diameter_inner = 0.564  # m
spacing_trans = 6.0  # out of correlation, overruled correlation checks
spacing_long = 1.25

tube_diameter_outer = 0.478e-2  # m - 4.78 mm
t_tubes = 0.03e-2  # m -  300 microns or 0.3 mm
tube_diameter_inner = tube_diameter_outer - 2 * t_tubes

n_passes_cold = 8

n_tubes_per_row = 62  # approx np.pi * D_i**2 / (Xt* * d_o)
n_rows = 32

n_tubes_per_pass = n_tubes_per_row * n_rows / n_passes_cold

area_frontal = area_frontal_bank(total_diameter_outer, total_diameter_inner)

sigma = sigma_tube_bank(spacing_trans)
area_free_flow_hot = area_free_flow_bank(area_frontal, sigma)
tube_length = axial_involute_tube_length(total_diameter_outer, total_diameter_inner)
area_heat_transfer_hot = area_heat_transfer_bank(tube_diameter_outer, tube_length, n_rows, n_tubes_per_row)
area_heat_transfer_cold = area_heat_transfer_in_tubes(tube_diameter_inner, tube_length, n_tubes_per_row * n_rows)

area_free_flow_cold = area_free_flow_in_tubes(tube_diameter_inner, n_tubes_per_pass)

print(f"heat transfer areas hot & cold: {area_heat_transfer_hot:.2f} & {area_heat_transfer_cold:.2f} m^2")

print(f"free flow areas hot & cold: {area_free_flow_hot:.2f} & {area_free_flow_cold:.4f} m^2")

print(f"Hot viscosity: {hot_air.get_viscosity(temp_hot_in, p_hot_in):.2e} Pa.s")

reynolds_hot_in = mdot_hot / area_free_flow_hot / hot_air.get_viscosity(temp_hot_in, p_hot_in) * tube_diameter_outer
reynolds_cold_in = (
    mdot_cold / area_free_flow_cold / cold_hydrogen.get_viscosity(temp_cold_in, p_cold_in) * tube_diameter_inner
)

print(f"reynolds numbers hot & cold: {reynolds_hot_in:.2e} & {reynolds_cold_in:.2e}")
tube_bank_correction_factor_hot = 1  # 0.05 / 0.13
print(f"tube bank correction factor hot: {tube_bank_correction_factor_hot}")
nusselt_hot, f_hot = tube_bank_nusselt_number_and_friction_factor(
    reynolds_hot_in, spacing_long, spacing_trans, prandtl=0.7, inline=True, n_rows=n_rows
)
f_hot = f_hot * tube_bank_correction_factor_hot
nusselt_hot = nusselt_hot * tube_bank_correction_factor_hot  # keep j/f propto Nu/f cst
nusselt_cold = circular_pipe_nusselt(reynolds_cold_in)
f_cold = circular_pipe_friction_factor(reynolds_cold_in)

print(f"nusselt numbers hot & cold: {nusselt_hot:.2f} & {nusselt_cold:.2f}")
print(f"friction factors hot & cold: {f_hot:.2f} & {f_cold:.2f}")

stanton_hot = nusselt_hot / reynolds_hot_in / 0.7
stanton_cold = nusselt_cold / reynolds_cold_in / 0.7

heat_capacity_flux_hot = mdot_hot * hot_air.get_cp(temp_hot_in, p_hot_in)
heat_capacity_flux_cold = mdot_cold * cold_hydrogen.get_cp(temp_cold_in, p_cold_in)

area_ratio_q_over_o_hot = area_heat_transfer_hot / area_free_flow_hot
area_ratio_q_over_o_cold = area_heat_transfer_cold / area_free_flow_cold

print(
    f"heat transfer to minimum flow area ratio (4L/d_h with K&L def): {area_ratio_q_over_o_hot:.2f} & {area_ratio_q_over_o_cold:.2f}"
)

ntu = ntu(
    stanton_hot,
    stanton_cold,
    area_ratio_q_over_o_hot,
    area_ratio_q_over_o_cold,
    heat_capacity_flux_hot,
    heat_capacity_flux_cold,
)

if heat_capacity_flux_hot > heat_capacity_flux_cold:
    c_min = heat_capacity_flux_cold
    c_ratio = heat_capacity_flux_cold / heat_capacity_flux_hot
    # cold is minimum and hot is mixed
    flow_type_description = "Cmax_mixed"
else:
    c_min = heat_capacity_flux_hot
    c_ratio = heat_capacity_flux_hot / heat_capacity_flux_cold
    # hot is minimum and cold is mixed
    flow_type_description = "Cmin_mixed"

epsilon = epsilon_ntu(
    ntu,
    c_ratio,
    exchanger_type="cross_flow",
    flow_type=flow_type_description,
    n_passes=n_passes_cold,
)

heat_transfer = epsilon * c_min * (temp_hot_in - temp_cold_in)  # not caring about enthalpy yet

temp_hot_out = temp_hot_in - heat_transfer / heat_capacity_flux_hot
temp_cold_out = temp_cold_in + heat_transfer / heat_capacity_flux_cold

rho_hot_in = hot_air.get_density(temp_hot_in, p_hot_in)
rho_hot_out_approx = hot_air.get_density(temp_hot_out, p_hot_in)

dp_hot = dp_tube_bank(
    area_ratio_q_over_o_hot,
    mdot_hot / area_free_flow_hot,
    rho_hot_in,
    rho_hot_out_approx,
    sigma,
    f_hot,
)

print(f"NTU: {ntu:.2f}")
print(f"effectiveness: {epsilon:.2%} (80.43% in Brewer)")
print(f"dp_hot: {dp_hot / p_hot_in:.2%} of inlet pressure (3.2% in Brewer)")


# Checks with Brewer data
rho_wall = 7930  # kg/m^3 304 Stainless Steel (CRES -> Corrosion Resistant?)
# https://ssmalloys.com/density-of-stainless-steel-304/
sigma_yield_wall = 205e6  # Pa MPa
thermal_expansion_coefficient_wall = 16e-6  # K^-1
# (high temperature strentgh from -200 C to 870 C) i.e. from 73.15 K to 1143.15 K
conductivity_wall = 14  # W/m.K https://www.azom.com/properties.aspx?ArticleID=965

wall_volume = n_rows * n_tubes_per_row * np.pi * (tube_diameter_outer**2 - tube_diameter_inner**2) * tube_length / 4
wall_mass = wall_volume * rho_wall

print(f"calculated wall mass: {wall_mass:.2f} kg (vs 38.3 kg in Brewer) so {wall_mass / 38.3 - 1:.2%} difference")
# Calculate the f_hot that would result in same pressure drop as Brewer
dp_hot_brewer = 3.2 / 100 * p_hot_in  # Pa

T_hot_out_brewer = 733
T_cold_out_breter = 677
rho_hot_out_brewer = hot_air.get_density(T_hot_out_brewer, p_hot_in)

dp_momentum_hot = (
    0.5 * (mdot_hot / area_free_flow_hot) ** 2 * (1 + sigma**2) * (1 / rho_hot_out_brewer - 1 / rho_hot_in)
)

dp_brewer_hot_friction = dp_hot_brewer - dp_momentum_hot
one_over_rho_mean_hot = (1 / rho_hot_in + 1 / rho_hot_out_brewer) / 2

f_hot_brewer = (
    2
    * dp_brewer_hot_friction
    / ((mdot_hot / area_free_flow_hot) ** 2 * area_ratio_q_over_o_hot * one_over_rho_mean_hot)
)

print(
    f"f_hot that would result in same pressure drop as Brewer: {f_hot_brewer:.2f} vs {f_hot:.2f} in model at Re_in = {reynolds_hot_in:.2e}"
)

# Calculate the nusselt number that would result in same heat transfer as Brewer
