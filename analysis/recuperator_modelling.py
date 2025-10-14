from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    tube_bank_nusselt_number_and_friction_factor,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu
from heat_exchanger.fluid_properties import CoolPropProperties, PerfectGasProperties
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

try:
    hot_air = CoolPropProperties(fluid_name="Air")
    cold_hydrogen = CoolPropProperties(fluid_name="Hydrogen")
    print("Using CoolPropProperties")
except Exception as e:
    print(f"Error using CoolPropProperties: {e}")
    print("Using PerfectGasProperties")

    hot_air = PerfectGasProperties(
        molecular_weight=28.97, gamma=1.4, Pr=0.7, mu_ref=1.8e-5, T_ref=300.0, S=110.4
    )
    cold_hydrogen = PerfectGasProperties(
        molecular_weight=2.016, gamma=1.4, Pr=0.7, mu_ref=8.4e-6, T_ref=273.15, S=110.4
    )


# Brewer recuperator values
temp_hot_in = 778  # K
temp_cold_in = 264  # K

mdot_hot = 19.07  # kg/s
mdot_cold = 0.166  # kg/s

p_hot_in = 4e4  # Pa
p_cold_in = 17.3e5  # Pa

total_diameter_outer = 1.265  # m
total_diameter_inner = 0.564  # m
spacing_trans = 6.0  # out of correlation, overruled correlation checks
spacing_long = 1.25

tube_diameter_outer = 0.478e-2  # m
t_tubes = 0.03e-2  # 300 microns
tube_diameter_inner = tube_diameter_outer - 2 * t_tubes

n_passes_cold = 8

n_tubes_per_row = 62  # approx np.pi * D_i**2 / (Xt* * d_o)
n_rows = 32

n_tubes_per_pass = n_tubes_per_row * n_rows / n_passes_cold

area_frontal = area_frontal_bank(total_diameter_outer, total_diameter_inner)

sigma = sigma_tube_bank(spacing_trans)
area_free_flow_hot = area_free_flow_bank(area_frontal, sigma)
tube_length = axial_involute_tube_length(total_diameter_outer, total_diameter_inner)
area_heat_transfer_hot = area_heat_transfer_bank(
    tube_diameter_outer, tube_length, n_rows, n_tubes_per_row
)
area_heat_transfer_cold = area_heat_transfer_in_tubes(
    tube_diameter_inner, tube_length, n_tubes_per_row * n_rows
)

area_free_flow_cold = area_free_flow_in_tubes(tube_diameter_inner, n_tubes_per_pass)

print(
    f" heat transfer areas hot & cold: {area_heat_transfer_hot:.2f} & {area_heat_transfer_cold:.2f} m^2"
)

print(f"free flow areas hot & cold: {area_free_flow_hot:.2f} & {area_free_flow_cold:.2f} m^2")


reynolds_hot_in = (
    mdot_hot
    / area_free_flow_hot
    / hot_air.get_viscosity(temp_hot_in, p_hot_in)
    * tube_diameter_outer
)
reynolds_cold_in = (
    mdot_cold
    / area_free_flow_cold
    / cold_hydrogen.get_viscosity(temp_cold_in, p_cold_in)
    * tube_diameter_inner
)

print(f"reynolds numbers hot & cold: {reynolds_hot_in:.2e} & {reynolds_cold_in:.2e}")

nusselt_hot, f_hot = tube_bank_nusselt_number_and_friction_factor(
    reynolds_hot_in, spacing_long, spacing_trans, prandtl=0.7, inline=True, n_rows=n_rows
)
nusselt_cold = circular_pipe_nusselt(reynolds_cold_in)
f_cold = circular_pipe_friction_factor(reynolds_cold_in)

stanton_hot = nusselt_hot / reynolds_hot_in / 0.7
stanton_cold = nusselt_cold / reynolds_cold_in / 0.7

heat_capacity_flux_hot = mdot_hot * hot_air.get_cp(temp_hot_in, p_hot_in)
heat_capacity_flux_cold = mdot_cold * cold_hydrogen.get_cp(temp_cold_in, p_cold_in)

area_ratio_q_over_o_hot = area_heat_transfer_hot / area_free_flow_hot
area_ratio_q_over_o_cold = area_heat_transfer_cold / area_free_flow_cold

print(
    f"heat transfer to minimum flow area ratio: {area_ratio_q_over_o_hot:.2f} & {area_ratio_q_over_o_cold:.2f}"
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
    area_ratio_q_over_o_hot, mdot_hot, rho_hot_in, rho_hot_out_approx, sigma, f_hot
)

print(f"NTU: {ntu:.2f}")
print(f"effectiveness: {epsilon:.2%}")
print(f"dp_hot: {dp_hot / p_hot_in:.2%} of inlet pressure")
