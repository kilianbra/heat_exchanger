import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    rectangular_duct_friction_factor,
    rectangular_duct_nusselt,
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
