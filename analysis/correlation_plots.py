import numpy as np
from tabulate import tabulate

from heat_exchanger.correlations import (
    calculate_new_Hg,
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    rectangular_duct_friction_factor,
    rectangular_duct_nusselt,
    tube_bank_friction_factor,
    tube_bank_nusselt_from_hagen,
    tube_bank_nusselt_number_and_friction_factor,
)

# Create Reynolds number range from 1e3 to 5e5 with 5 values
reynolds_list = np.logspace(3, 5, 5)

# Initialize lists to store results
circular_pipe_f_list = []
circular_pipe_nusselt_list = []
annular_pipe_f_list = []
annular_pipe_nusselt_list = []
rectangular_duct_f_list = []
rectangular_duct_nusselt_list = []

r_ratio = 0.2
a_over_b = 0.7

# Loop through each Reynolds number since functions don't accept vectors
for Re in reynolds_list:
    circular_pipe_f_list.append(circular_pipe_friction_factor(Re))
    circular_pipe_nusselt_list.append(circular_pipe_nusselt(Re))
    annular_pipe_f_list.append(circular_pipe_friction_factor(Re, r_ratio=r_ratio))
    annular_pipe_nusselt_list.append(circular_pipe_nusselt(Re, r_ratio=r_ratio))
    rectangular_duct_f_list.append(rectangular_duct_friction_factor(Re, a_over_b=a_over_b))
    rectangular_duct_nusselt_list.append(rectangular_duct_nusselt(Re, a_over_b=a_over_b))

# Create table data
table_data = []
for i, Re in enumerate(reynolds_list):
    table_data.append(
        [
            f"{Re:.0f}",
            f"{circular_pipe_f_list[i]:.4f}",
            f"{annular_pipe_f_list[i]:.4f}",
            f"{rectangular_duct_f_list[i]:.4f}",
            f"{circular_pipe_nusselt_list[i]:.2f}",
            f"{annular_pipe_nusselt_list[i]:.2f}",
            f"{rectangular_duct_nusselt_list[i]:.2f}",
        ]
    )

# Print the table
headers = ["Re", "Circ f", "Ann f", "Rect f", "Circ Nu", "Ann Nu", "Rect Nu"]
print("\n" + "=" * 80)
print("HEAT EXCHANGER CORRELATION RESULTS")
print("=" * 80)
print(tabulate(table_data, headers=headers, tablefmt="grid"))
print("=" * 80)
print(
    f"Note: \n Annular pipe calculations uses r_inner/r_outer = {r_ratio}, \n Rectangular duct calculations use a/b = {a_over_b}"
)
print("=" * 80)


# Data for Kays and London inline surface from Fig 10-12: I1.50 - 1.25(a)
spacing_trans = 1.5
spacing_long = 1.25
n_rows = 15

Re_values = [12627, 10101, 7576, 6313, 5051, 3788, 3157, 2525, 1894, 1515, 1263, 1010]

f_exp_k_and_l = [
    0.0505,
    0.0525,
    0.0549,
    0.0558,
    0.0562,
    0.0554,
    0.0535,
    0.0497,
    0.0410,
    0.0331,
    0.0281,
    0.0265,
]

j_exp_knl = [
    0.00752,
    0.00820,
    0.00900,
    0.00958,
    0.01020,
    0.01080,
    0.01095,
    0.01075,
    0.00960,
    0.00778,
    0.00750,
    0.00790,
]

# Calculate correlation values using tube_bank_friction_factor
f_corr_list = []
for Re in Re_values:
    f_corr = tube_bank_friction_factor(Re, spacing_long, spacing_trans, inline=True, n_rows=n_rows)
    f_corr_list.append(f_corr)

# Create comparison table
tube_bank_table_data = []
for i, Re in enumerate(Re_values):
    tube_bank_table_data.append(
        [
            f"{Re:.2e}",  # Reynolds in scientific notation with 2 sig figs
            f"{f_exp_k_and_l[i]:.2e}",  # Experimental in scientific notation with 2 sig figs
            f"{f_corr_list[i]:.2e}",  # Correlation in scientific notation with 2 sig figs
        ]
    )

# Print the tube bank comparison table
print("\n" + "=" * 80)
print("TUBE BANK FRICTION FACTOR COMPARISON")
print("=" * 80)
print(
    f"Configuration: Inline tubes, Xt* = {spacing_trans}, Xl* = {spacing_long}, N_rows = {n_rows}"
)
print("Data source: Kays & London Fig 10-12: I1.50 - 1.25(a)")
print("=" * 80)
tube_bank_headers = ["Reynolds", "f_experimental", "f_correlation"]
print(tabulate(tube_bank_table_data, headers=tube_bank_headers, tablefmt="grid"))
print("=" * 80)


# Example calculation using the new functions
print("\n" + "=" * 80)
print("EXAMPLE CALCULATION USING NEW FUNCTIONS")
print("=" * 80)
Re_example = 1e4
Pr_example = 0.7
print(f"Example: Re_d = {Re_example:.0e}, Pr = {Pr_example}")
print(
    f"Configuration: Inline tubes, Xt* = {spacing_trans}, Xl* = {spacing_long}, N_rows = {n_rows}"
)

# Step 1: Calculate Hagen number
hagen_number = calculate_new_Hg(Re_example, spacing_long, spacing_trans, inline=True, Nr=n_rows)
print(f"Hagen number: {hagen_number:.2f}")

# Step 2: Calculate Nusselt number from Hagen number
nusselt_from_hagen = tube_bank_nusselt_from_hagen(
    hagen_number, Re_example, spacing_long, spacing_trans, Pr_example, inline=True
)
print(f"Nusselt number: {nusselt_from_hagen:.2f}")

# Step 3: Calculate j-factor
j_factor_example = nusselt_from_hagen / (Re_example * Pr_example ** (1 / 3))
print(f"j-factor: {j_factor_example:.4f}")
print("=" * 80)

# Complete analysis using tube_bank_nusselt_number_and_friction_factor
print("\n" + "=" * 80)
print("COMPLETE TUBE BANK ANALYSIS")
print("=" * 80)
print(
    f"Configuration: Inline tubes, Xt* = {spacing_trans}, Xl* = {spacing_long}, N_rows = {n_rows}"
)
print("Data source: Kays & London Fig 10-12: I1.50 - 1.25(a)")
print("=" * 80)

# Calculate correlation values for all Reynolds numbers
nusselt_corr_list = []
f_corr_list_new = []
j_corr_list = []

Pr = 0.7  # Prandtl number

for Re in Re_values:
    nusselt, f_corr = tube_bank_nusselt_number_and_friction_factor(
        Re, spacing_long, spacing_trans, Pr, inline=True, n_rows=n_rows
    )
    j_corr = nusselt / (Re * Pr ** (1 / 3))

    nusselt_corr_list.append(nusselt)
    f_corr_list_new.append(f_corr)
    j_corr_list.append(j_corr)

# Calculate experimental j-factors from j_exp_knl data
j_exp_list = j_exp_knl  # These are already j-factors from the experimental data

# Create comprehensive comparison table
comprehensive_table_data = []
for i, Re in enumerate(Re_values):
    comprehensive_table_data.append(
        [
            f"{Re:.1e}",  # Reynolds in scientific notation
            f"{f_exp_k_and_l[i]:.4f}",  # Experimental friction factor
            f"{f_corr_list_new[i]:.4f}",  # Correlation friction factor
            f"{j_exp_list[i]:.5f}",  # Experimental j-factor
            f"{j_corr_list[i]:.5f}",  # Correlation j-factor
        ]
    )

# Print the comprehensive comparison table
comprehensive_headers = ["Reynolds", "f_exp", "f_corr", "j_exp", "j_corr"]
print(tabulate(comprehensive_table_data, headers=comprehensive_headers, tablefmt="grid"))
print("=" * 80)

# Excel-friendly output for all values
print("\n" + "=" * 80)
print("EXCEL COPY-PASTE FORMAT - COMPREHENSIVE DATA")
print("=" * 80)

print("Tab-separated for easy Excel paste (Reynolds\tf_exp\tf_corr\tj_exp\tj_corr):")
for i, Re in enumerate(Re_values):
    print(
        f"{Re}\t{f_exp_k_and_l[i]:.6f}\t{f_corr_list_new[i]:.6f}\t{j_exp_list[i]:.6f}\t{j_corr_list[i]:.6f}"
    )
