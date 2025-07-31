import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    rectangular_duct_friction_factor,
    rectangular_duct_nusselt,
)

reynolds_list = 5e3

circular_pipe_f_list = circular_pipe_friction_factor(reynolds_list)
circular_pipe_Nu_list = circular_pipe_nusselt(reynolds_list)

r_ratio = 0.2

annular_pipe_f_list = circular_pipe_friction_factor(reynolds_list, r_ratio=r_ratio)
annular_pipe_Nu_list = circular_pipe_nusselt(reynolds_list, r_ratio=r_ratio)

print(f"Circular pipe f: {circular_pipe_f_list:.4f}")
print(f"Circular pipe Nu: {circular_pipe_Nu_list:.2f}")
print(f"Annular pipe f: {annular_pipe_f_list:.4f}")
print(f"Annular pipe Nu: {annular_pipe_Nu_list:.2f}")
