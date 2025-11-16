"""Brewer recuperator model using the new tube_bank_normal geometry and solver."""

import logging

import numpy as np

from heat_exchanger.fluids.protocols import CoolPropFluid, FluidInputs, PerfectGasFluid, RefPropFluid
from heat_exchanger.geometries.tube_bank_normal import TubeBankNormalSpec, tube_bank_normal_0d_solver
from heat_exchanger.logging_utils import configure_logging

# Configure logging
configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

# Mass flow rates
mdot_hot = 1144 / 60  # 19.07 kg/s 
mdot_cold = 9.95 / 60  # 0.166 kg/s

# Fluid model selection
model = "PG"  # "CP", "PG", or "RP"

match model:
    case "RP":
        hot_air = CombustionProductsProperties(fuel_type="H2", FAR_mass=9.95 / (1144 - 9.95), prefer_refprop=True)
        cold_hydrogen = RefPropProperties(fluid_name="PARAHYDROGEN")
    case "CP":
        hot_air = CoolPropFluid("Air")
        cold_hydrogen = CoolPropFluid("Hydrogen")
    case "PG":
        hot_air = PerfectGasFluid.from_name("h2comb_ahje")
        cold_hydrogen = PerfectGasFluid.from_name("para_h2")

# Brewer recuperator values
temp_hot_in = 778  # K
temp_cold_in = 264  # K

p_hot_in = 4e4  # Pa
p_cold_in = 17.3e5  # Pa

# Geometry parameters
total_diameter_outer = 1.265  # m
total_diameter_inner = 0.564  # m
spacing_trans = 6.0  # out of correlation, overruled correlation checks
spacing_long = 1.25

tube_diameter_outer = 0.478e-2  # m - 4.78 mm
t_tubes = 0.03e-2  # m - 300 microns or 0.3 mm

n_passes_cold = 8

n_tubes_per_row = 62  # approx np.pi * D_i**2 / (Xt* * d_o)
n_rows = 32

n_rows_per_pass = int(n_rows / n_passes_cold)  # rows per pass

# Calculate frontal area
area_frontal = np.pi * (total_diameter_outer**2 - total_diameter_inner**2) / 4

# Create geometry object
geom = TubeBankNormalSpec(
    tube_outer_diam=tube_diameter_outer,
    tube_thick=t_tubes,
    tube_spacing_trv=spacing_trans,
    tube_spacing_long=spacing_long,
    staggered=False,  # inline
    n_rows_per_pass=n_rows_per_pass,
    n_passes=n_passes_cold,
    n_tubes_per_row=n_tubes_per_row,
    frontal_area_outer=area_frontal,
    annular_not_box=True,
    total_diameter_inner=total_diameter_inner,  # Use actual inner diameter for accurate tube length
    total_diameter_outer=total_diameter_outer,  # Use actual outer diameter for accurate tube length
)

logger.info(
    "Axial length: %.2f m (N_tubes = %d)",
    geom.axial_length,
    geom.n_tubes_total,
)
logger.info(
    "Tube length: %.4f m, n_rows_total: %d, n_tubes_per_row: %d",
    geom.tube_length,
    geom.n_rows_total,
    geom.n_tubes_per_row,
)
logger.info(
    "Heat transfer areas: hot=%.2f m², cold=%.2f m²",
    geom.area_heat_transfer_outer_total,
    geom.area_heat_transfer_inner_total,
)

# Create fluid inputs
f_in = FluidInputs(
    hot=hot_air,
    cold=cold_hydrogen,
    m_dot_hot=mdot_hot,
    m_dot_cold=mdot_cold,
    Th_in=temp_hot_in,
    Ph_in=p_hot_in,
    Tc_in=temp_cold_in,
    Pc_in=p_cold_in,
)

# Tube bank correction factor
tube_bank_correction_factor_hot = 0.048 / 0.1374

# MTO conditions for thermal expansion and thickness calculations
p_cold_MTO = None  # Pa - set if MTO conditions are available
p_hot_MTO = None  # Pa
T_hot_MTO = None  # K
T_base = None  # K

# Material properties for wall calculations
sigma_yield_wall = 205e6  # Pa - 304 Stainless Steel
thermal_expansion_coefficient_wall = 16e-6  # K^-1
rho_wall = 7930  # kg/m^3 - 304 Stainless Steel

# Solve using 0D solver
result = tube_bank_normal_0d_solver(
    geom=geom,
    f_in=f_in,
    tube_bank_correction_factor_hot=tube_bank_correction_factor_hot,
    p_cold_MTO=p_cold_MTO,
    p_hot_MTO=p_hot_MTO,
    T_hot_MTO=T_hot_MTO,
    T_base=T_base,
    sigma_yield_wall=sigma_yield_wall,
    thermal_expansion_coefficient_wall=thermal_expansion_coefficient_wall,
    rho_wall=rho_wall,
)

# Extract results
Th_out = result["Th_out"]
Tc_out = result["Tc_out"]
Ph_out = result["Ph_out"]
epsilon = result["epsilon"]
NTU = result["NTU"]
dp_hot = result["dP_hot"]
dp_hot_pct = result["dP_hot_pct"]

logger.info("NTU: %.2f", NTU)
logger.info("Effectiveness: %.2f%% (80.43%% in Brewer)", epsilon * 100)
logger.info("dP_hot: %.2f%% of inlet pressure (3.2%% in Brewer)", dp_hot_pct)

# Checks with Brewer data
# Calculate the f_hot that would result in same pressure drop as Brewer
dp_hot_brewer = 3.2 / 100 * p_hot_in  # Pa

T_hot_out_brewer = 733  # K
T_cold_out_brewer = 677  # K
hot_out_brewer = hot_air.state(T_hot_out_brewer, p_hot_in)
rho_hot_out_brewer = hot_out_brewer.rho
hot_in_state = hot_air.state(temp_hot_in, p_hot_in)
rho_hot_in = hot_in_state.rho

# Calculate friction factor that would give Brewer's pressure drop
area_free_flow_hot = geom.area_free_flow_outer
sigma = geom.sigma_outer
area_ratio_q_over_o_hot = geom.area_heat_transfer_outer_total / area_free_flow_hot

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

# Get the actual friction factor from diagnostics
f_hot_actual = result["diagnostics"]["f_hot"]
reynolds_hot_in = result["diagnostics"]["Re_hot"]

logger.info(
    "f_hot that would result in same pressure drop as Brewer: %.4f vs %.4f in model at Re_in = %.2e",
    f_hot_brewer,
    f_hot_actual,
    reynolds_hot_in,
)

# Wall mass comparison
wall_mass = result["diagnostics"]["wall_mass"]
wall_mass_brewer = 38.3  # kg
logger.info(
    "Calculated wall mass: %.2f kg (vs %.2f kg in Brewer) so %.2f%% difference",
    wall_mass,
    wall_mass_brewer,
    (wall_mass / wall_mass_brewer - 1) * 100,
)

# Calculate the nusselt number that would result in same heat transfer as Brewer
# (This would require iterating to match the heat transfer, so we'll leave it as a placeholder)
logger.info("Brewer comparison complete.")
