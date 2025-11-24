import logging

from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.radial_spiral import RadialSpiralSpec, spiral_hex_solver
from heat_exchanger.logging_utils import configure_logging

# Configure logging
configure_logging(logging.INFO)

# ============================================================================
# SELECT CASE: Change this to 1 or 2 to switch between cases
# ============================================================================
CASE = 1

# ============================================================================
# CASE 1
# ============================================================================
geom_case1 = RadialSpiralSpec(
    tube_outer_diam=0.00108,
    tube_thick=0.00004,
    tube_spacing_trv=4.0,  # chg from 6 to 4 for H2 slower
    tube_spacing_long=1.5,
    staggered=False,
    n_headers=13,
    n_rows_per_header=1,
    n_tubes_per_row=170,  # chg from 113 to 170 for H2 slower
    radius_outer_hex=0.65,
    inv_angle_deg=360.0,
    wall_conductivity=14.0,
    ext_fluid_flows_radially_inwards=True,
)

# Additional geometry attributes:
# radius_inner_hex=6.289400e-01
# frontal_area_inner=2.679284e+00
# frontal_area_outer=2.990520e+00
# area_heat_transfer_outer_total=2.002633e+01
# volume_total=6.196006e-02

f_in_case1 = FluidInputs(
    hot=PerfectGasFluid.from_name("Air"),
    cold=PerfectGasFluid.from_name("Para_Hydrogen"),
    m_dot_hot=64.327297380,
    m_dot_cold=0.76633099552 / 2,
    Tc_in=275.0,
    Pc_in=2790000.0,
    Th_in=574.3318,
    Ph_out=22631.0,
)

# ============================================================================
# CASE 2
# ============================================================================
geom_case2 = RadialSpiralSpec(
    tube_outer_diam=0.00108,
    tube_thick=0.00004,
    tube_spacing_trv=6.0,
    tube_spacing_long=1.5,
    staggered=False,
    n_headers=6,
    n_rows_per_header=1,
    n_tubes_per_row=113,
    radius_outer_hex=0.65,
    inv_angle_deg=360.0,
    wall_conductivity=14.0,
    ext_fluid_flows_radially_inwards=True,
)

# Additional geometry attributes:
# radius_inner_hex=6.402800e-01
# frontal_area_inner=2.727593e+00
# frontal_area_outer=2.990520e+00
# area_heat_transfer_outer_total=9.324776e+00
# volume_total=2.885051e-02

f_in_case2 = f_in_case1  # same
# FluidInputs(
#     hot=PerfectGasFluid.from_name("Air"),
#     cold=PerfectGasFluid.from_name("Para_Hydrogen"),
#     m_dot_hot=64.336045634,
#     m_dot_cold=0.78194734443,
#     Tc_in=275.0,
#     Pc_in=2790000.0,
#     Th_in=574.3318,
#     Ph_out=22631.0,
# )

# ============================================================================
# RUN SELECTED CASE
# ============================================================================
if CASE == 1:
    geom = geom_case1
    f_in = f_in_case1
    case_name = "Case 1"
elif CASE == 2:
    geom = geom_case2
    f_in = f_in_case2
    case_name = "Case 2"
else:
    raise ValueError(f"Invalid CASE value: {CASE}. Must be 1 or 2.")

print(f"\n{'=' * 70}")
print(f"Running {case_name}")
print(f"{'=' * 70}\n")

result = spiral_hex_solver(geom, f_in, method="1d")

# Extract diagnostics
diag = result["diagnostics"]

# Print results
print(f"\n{'=' * 70}")
print(f"RESULTS for {case_name}")
print("=" * 70)
print(f"Effectiveness (ε):        {diag.get('epsilon', float('nan')) * 100.0:.4f} %")
print(f"NTU:                       {diag.get('NTU', float('nan')):.4f}")
print(f"Heat Transfer (Q_total):   {diag.get('Q_total', float('nan')) / 1e6:.6f} MW")
print("\nPressure Drops:")
print(
    f"  Hot side (ΔP_hot):       {diag.get('dP_hot_pct', float('nan')):.4f} %  ({diag.get('dP_hot', float('nan')):.2f} Pa)"
)
print(
    f"  Cold side (ΔP_cold):     {diag.get('dP_cold_pct', float('nan')):.4f} %  ({diag.get('dP_cold', float('nan')):.2f} Pa)"
)
print("\nOutlet Temperatures:")
print(f"  Th_out:                  {diag.get('Th_out', float('nan')):.2f} K")
print(f"  Tc_out:                  {diag.get('Tc_out', float('nan')):.2f} K")
print(f"\nCapacity Ratio (Cr):      {diag.get('Cr', float('nan')):.6f}")
print("=" * 70 + "\n")
