import numpy as np

from fluid_files.H2_recirc import calculate_recirc_fraction_coolprop
from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    tube_bank_nusselt_number_and_friction_factor,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu
from heat_exchanger.fluid_properties import (
    CombustionProductsProperties,
    #CoolPropProperties,
    #PerfectGasProperties,
    RefPropProperties,
)
from heat_exchanger.fluids.protocols import PerfectGasFluid, CoolPropFluid
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

mdot_hot = 64.316  # 19.07  # kg/s
mdot_cold = 0.427 * (1+0.787)  # If get H2 hot then can afford to only recirculate 79% of core
#But now need more. 0.787  0.166  # kg/s

model = "PG"  # "CP" or "PG"

match model:
    case "RP":
        pass # Need to implement
        hot_air = CombustionProductsProperties(fuel_type="H2", FAR_mass=9.95 / (1144 - 9.95), prefer_refprop=True)
        cold_hydrogen = RefPropProperties(fluid_name="PARAHYDROGEN")
    case "CP":
        hot_air = CoolPropFluid("air")
        cold_hydrogen = CoolPropFluid("para_h2")
    case "PG":
        #hot_air = PerfectGasProperties(molecular_weight=28.97, gamma=1.4, Pr=0.7, mu_ref=1.8e-5, T_ref=300.0, S=110.4)
        #cold_hydrogen = PerfectGasProperties(molecular_weight=2.016, gamma=1.4, Pr=0.7, mu_ref=8.4e-6, T_ref=273.15, S=110.4)
        hot_air = PerfectGasFluid.from_name("h2comb_ahje")
        cold_hydrogen = PerfectGasFluid.from_name("para_h2")

# Brewer recuperator values
temp_hot_in = 574  # K
temp_cold_in = 275  # K Alistair 275, before 287 K, just ensure air doesn't freeze

p_hot_in = 0.368e5  # Pa
p_cold_in = 25e5  # Pa

total_diameter_outer = 2 * 0.871  # m
total_diameter_inner = 2 * 0.541  # m
spacing_trans = 2.5 
spacing_long = 1.5

tube_diameter_outer = 1.0e-3 + 2 * 0.040e-3 # 1.067e-3 m - 4.78 mm
t_tubes = 0.040e-3  # 0.129e-3 m -  300 microns or 0.3 mm
tube_diameter_inner = tube_diameter_outer - 2 * t_tubes

n_passes_cold = 1 # 8

n_tubes_per_row = round(
    np.pi * total_diameter_inner**2 / (spacing_trans * tube_diameter_outer)
)  # now 287 tubes per row (Axially)  old 62  # approx np.pi * D_i**2 / (Xt* * d_o)
n_rows = 6 #32

n_tubes_per_pass = n_tubes_per_row * n_rows / n_passes_cold

print(f"Axial length: {n_rows * spacing_long * tube_diameter_outer:.2f} m (N_tubes = {n_tubes_per_row * n_rows})")

area_frontal = area_frontal_bank(total_diameter_outer, total_diameter_inner)

sigma = sigma_tube_bank(spacing_trans)
print(
    f"Axial length: {n_rows * spacing_long * tube_diameter_outer:.2f} m (N_tubes = {n_tubes_per_row * n_rows}), sigma = {sigma:.2f}"
)
area_free_flow_hot = area_free_flow_bank(area_frontal, sigma)
tube_length = axial_involute_tube_length(total_diameter_outer, total_diameter_inner)
area_heat_transfer_hot = area_heat_transfer_bank(tube_diameter_outer, tube_length, n_rows, n_tubes_per_row)
area_heat_transfer_cold = area_heat_transfer_in_tubes(tube_diameter_inner, tube_length, n_tubes_per_row * n_rows)

area_free_flow_cold = area_free_flow_in_tubes(tube_diameter_inner, n_tubes_per_pass)

print(f"heat transfer areas hot & cold: {area_heat_transfer_hot:.2f} & {area_heat_transfer_cold:.2f} m^2")

print(f"free flow areas hot & cold: {area_free_flow_hot:.2f} & {area_free_flow_cold:.4f} m^2")

hot_in = hot_air.state(temp_hot_in, p_hot_in)
cold_in = cold_hydrogen.state(temp_cold_in, p_cold_in)

print(f"Hot viscosity: {hot_in.mu:.2e} Pa.s")

reynolds_hot_in = mdot_hot / area_free_flow_hot / hot_in.mu * tube_diameter_outer
reynolds_cold_in = mdot_cold / area_free_flow_cold / cold_in.mu * tube_diameter_inner

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

heat_capacity_flux_hot = mdot_hot * hot_in.cp
heat_capacity_flux_cold = mdot_cold * cold_in.cp

area_ratio_q_over_o_hot = area_heat_transfer_hot / area_free_flow_hot
area_ratio_q_over_o_cold = area_heat_transfer_cold / area_free_flow_cold

print(
    f"heat transfer to minimum flow area ratio (4L/d_h with K&L def): {area_ratio_q_over_o_hot:.2f} & {area_ratio_q_over_o_cold:.2f}"
)


c_min = np.minimum(heat_capacity_flux_hot, heat_capacity_flux_cold)
inv_h = 1 / stanton_hot / area_ratio_q_over_o_hot * c_min / heat_capacity_flux_hot
inv_c = 1 / stanton_cold / area_ratio_q_over_o_cold * c_min / heat_capacity_flux_cold
total_resistance = inv_h + inv_c
ntu = 1 / total_resistance

# Percentage contribution to total thermal resistance for each fluid
perc_hot = inv_h / total_resistance * 100
perc_cold = inv_c / total_resistance * 100

print(f"Thermal resistance percentage contribution hot: {perc_hot:.0f}%  cold: {perc_cold:.0f}%")

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
print(f"heat_transfer: {heat_transfer / 1e6:.2f} MW")

temp_hot_out = temp_hot_in - heat_transfer / heat_capacity_flux_hot
temp_cold_out = temp_cold_in + heat_transfer / heat_capacity_flux_cold

print(f"temp_hot_out: {temp_hot_out:.2f} K, temp_cold_out: {temp_cold_out:.2f} K")

recirc_fraction = calculate_recirc_fraction_coolprop(40, temp_cold_in, temp_cold_out, p_cold_in/1e5)  
print(f"recirc_fraction: {recirc_fraction[0]:.2f}")

rho_hot_in = hot_in.rho
hot_out_approx = hot_air.state(temp_hot_out, p_hot_in)
rho_hot_out_approx = hot_out_approx.rho

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
hot_out_brewer = hot_air.state(T_hot_out_brewer, p_hot_in)
rho_hot_out_brewer = hot_out_brewer.rho

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
