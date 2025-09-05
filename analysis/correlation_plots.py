import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from heat_exchanger.correlations import (
    calculate_Hg,
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    rectangular_duct_friction_factor,
    rectangular_duct_nusselt,
    tube_bank_corrected_xi_gunter_and_shaw,
    tube_bank_friction_factor,
    tube_bank_nusselt_from_hagen,
    tube_bank_nusselt_number_and_friction_factor,
)

# ---------------------------
# Quick plot configuration
# ---------------------------
# Choose which plot to generate:
#   - "tube_bank_compare": Kays & London experimental vs Gaddis & Gnielinski correlation
#   - "drag_vs_re": xi = 2 Hg / Re^2 vs Re for different spacing_trans
#   - "j_over_f_v_re": j/f ratio vs Re for experimental and correlation data
PLOT_CASE = "drag_vs_re"


# Toggle twin y-axis for j/f ratio on tube bank compare plot
SHOW_TWIN_J_OVER_F_AXIS = True

# General matplotlib style bits (optional)
DEFAULT_DPI = 120


# ---------------------------
# Tabular quick check (kept, but can be ignored when plotting)
# ---------------------------
def _print_quick_tables():
    reynolds_list = np.logspace(3, 5, 5)

    circular_pipe_f_list = []
    circular_pipe_nusselt_list = []
    annular_pipe_f_list = []
    annular_pipe_nusselt_list = []
    rectangular_duct_f_list = []
    rectangular_duct_nusselt_list = []

    r_ratio = 0.2
    a_over_b = 0.7

    for Re in reynolds_list:
        circular_pipe_f_list.append(circular_pipe_friction_factor(Re))
        circular_pipe_nusselt_list.append(circular_pipe_nusselt(Re))
        annular_pipe_f_list.append(circular_pipe_friction_factor(Re, r_ratio=r_ratio))
        annular_pipe_nusselt_list.append(circular_pipe_nusselt(Re, r_ratio=r_ratio))
        rectangular_duct_f_list.append(rectangular_duct_friction_factor(Re, a_over_b=a_over_b))
        rectangular_duct_nusselt_list.append(rectangular_duct_nusselt(Re, a_over_b=a_over_b))

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

    headers = ["Re", "Circ f", "Ann f", "Rect f", "Circ Nu", "Ann Nu", "Rect Nu"]
    print("\n" + "=" * 80)
    print("HEAT EXCHANGER CORRELATION RESULTS")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 80)
    print(
        "Note:\n Annular pipe calculations uses r_inner/r_outer = 0.2,\n Rectangular duct calculations use a/b = 0.7"
    )
    print("=" * 80)


# Data for Kays and London inline surface from
spacing_trans = 1.5
spacing_long = 1.25
n_rows = 15
# Table 10-2 I1.50-1.25(s)  (corresponds to Fig 10-12: I1.50 - 1.25(a))
# These Reynolds numbers are based on the tube diameter (in KnL they are based on d_h)
Re_hydraulic = [10_000, 8_000, 6_000, 5_000, 4_000, 3_000, 2_500, 2_000, 1_500, 1_200, 1_000, 800]
# These reynolds are based on the tube outer diameter
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
re_gaddis_vdi = [
    1.162e0,
    2.667e0,
    1.313e1,
    1.313e1,
    4.537e1,
    1.464e2,
    3.015e2,
    5.491e2,
    9.865e2,
    2.204e3,
    2.204e3,
    4.476e3,
    4.476e3,
    8.967e3,
    8.967e3,
    1.633e4,
    1.633e4,
    4.297e4,
    4.297e4,
    1.115e5,
    1.115e5,
    3.272e5,
    3.272e5,
]

xi_gaddis_vdi = [
    1.812e2,
    8.091e1,
    1.704e1,
    1.704e1,
    5.333e0,
    1.888e0,
    1.078e0,
    7.353e-1,
    6.073e-1,
    5.444e-1,
    5.444e-1,
    5.297e-1,
    5.297e-1,
    4.880e-1,
    4.880e-1,
    4.684e-1,
    4.684e-1,
    4.199e-1,
    4.199e-1,
    3.764e-1,
    3.764e-1,
    3.329e-1,
    3.329e-1,
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


def _print_tube_bank_comparison_table():
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


def _print_example_calc():
    print("\n" + "=" * 80)
    print("EXAMPLE CALCULATION USING NEW FUNCTIONS")
    print("=" * 80)
    Re_example = 1e4
    Pr_example = 0.7
    print(f"Example: Re_d = {Re_example:.0e}, Pr = {Pr_example}")
    print(
        f"Configuration: Inline tubes, Xt* = {spacing_trans}, Xl* = {spacing_long}, N_rows = {n_rows}"
    )

    hagen_number = calculate_Hg(Re_example, spacing_long, spacing_trans, inline=True, Nr=n_rows)
    print(f"Hagen number: {hagen_number:.2f}")

    nusselt_from_hagen = tube_bank_nusselt_from_hagen(
        hagen_number, Re_example, spacing_long, spacing_trans, Pr_example, inline=True
    )
    print(f"Nusselt number: {nusselt_from_hagen:.2f}")

    j_factor_example = nusselt_from_hagen / (Re_example * Pr_example ** (1 / 3))
    print(f"j-factor: {j_factor_example:.4f}")
    print("=" * 80)


def _print_complete_analysis_header():
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
            f"{Re:.4e}",  # Reynolds in scientific notation
            f"{f_exp_k_and_l[i]:.4f}",  # Experimental friction factor
            f"{f_corr_list_new[i]:.4f}",  # Correlation friction factor
            f"{j_exp_list[i]:.5f}",  # Experimental j-factor
            f"{j_corr_list[i]:.5f}",  # Correlation j-factor
        ]
    )


def _print_comprehensive_table():
    comprehensive_headers = ["Reynolds", "f_exp", "f_corr", "j_exp", "j_corr"]
    print(tabulate(comprehensive_table_data, headers=comprehensive_headers, tablefmt="grid"))
    print("=" * 80)


def _print_excel_friendly():
    print("\n" + "=" * 80)
    print("EXCEL COPY-PASTE FORMAT - COMPREHENSIVE DATA")
    print("=" * 80)
    print("Tab-separated for easy Excel paste\n (Reynolds\tf_exp\tf_corr\tj_exp\tj_corr):")
    for i, Re in enumerate(Re_values):
        print(
            f"{Re}\t{f_exp_k_and_l[i]:.6f}\t{f_corr_list_new[i]:.6f}\t{j_exp_list[i]:.6f}\t{j_corr_list[i]:.6f}"
        )


# ---------------------------
# Plotting utilities
# ---------------------------
def _plot_tube_bank_compare(show_twin_j_over_f_axis: bool = True):
    plt.figure(dpi=DEFAULT_DPI)
    ax = plt.gca()

    # f and j vs Re on same (left) axis with log-log scaling
    ax.scatter(Re_values, f_exp_k_and_l, label="f exp (K&L)", color="#1f77b4", marker="o")
    ax.plot(Re_values, f_corr_list_new, label="f corr (G&G)", color="#1f77b4", linestyle="--")
    ax.scatter(Re_values, j_exp_list, label="j exp (K&L)", color="#d62728", marker="s")
    ax.plot(Re_values, j_corr_list, label="j corr (G&G)", color="#d62728", linestyle="-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re (based on tube diameter)")
    ax.set_ylabel("f and j")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.set_ylim(0.006, 0.06)

    handles, labels = ax.get_legend_handles_labels()

    if show_twin_j_over_f_axis:
        ax2 = ax.twinx()
        # Calculate j/f ratios
        j_over_f_exp = [j / f for j, f in zip(j_exp_list, f_exp_k_and_l, strict=True)]
        j_over_f_corr = [j / f for j, f in zip(j_corr_list, f_corr_list_new, strict=True)]

        ax2.scatter(Re_values, j_over_f_exp, label="j/f exp (K&L)", color="black", marker="^")
        ax2.plot(Re_values, j_over_f_corr, label="j/f corr (G&G)", color="black", linestyle=":")
        ax2.set_ylabel("j/f ratio")
        ax2.set_yscale("linear")
        ax2.set_ylim(0.0, 0.5)

        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")

    title = f"Inline tube bank: Xt*={spacing_trans}, Xl*={spacing_long}, N_rows={n_rows}"
    plt.title(title)
    plt.tight_layout()


def _plot_drag_vs_re():
    plt.figure(dpi=DEFAULT_DPI)
    ax = plt.gca()

    reynolds = np.logspace(1, 5.5, 100)
    n_rows_loc = 15
    spacing_trans_list = [1.25, 1.5, 2.5, 5.0]
    spacing_long_default = 1.25  # keep within validated inline range when possible

    for xt in spacing_trans_list:
        xi_vals = []
        valid_any = False
        for Re in reynolds:
            try:
                Hg = calculate_Hg(Re, spacing_long_default, xt, inline=True, Nr=n_rows_loc)
                xi = 2 * Hg / Re**2
                # xi = tube_bank_corrected_xi_gunter_and_shaw(Re, spacing_long_default, xt)
                xi_vals.append(xi)
                valid_any = True
            except AssertionError:
                xi_vals.append(np.nan)

        if valid_any:
            ax.plot(reynolds, xi_vals, label=f"Xt*={xt}")
    xi_exp = [np.pi / (spacing_trans - 1) * f_exp for f_exp in f_exp_k_and_l]
    ax.plot(
        re_gaddis_vdi, xi_gaddis_vdi, label="exp (? Gaddis) Xt*=1.25", color="C0", linestyle="--"
    )
    ax.scatter(Re_values, xi_exp, label="exp (K&L) Xt*=1.5", color="C1", marker="x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re (based on tube diameter)")
    ax.set_ylabel(r"Drag factor $\xi = 2\rho \Delta p / (N_r G^2) =  \pi f_o/(X_t^*-1)$")
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlim(1e1, 10**5.5)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best", title="Transverse spacing")
    plt.title("Drag factor vs Reynolds (inline, Xl*=1.25)")
    plt.tight_layout()


def _plot_j_over_f_v_re():
    plt.figure(dpi=DEFAULT_DPI)
    ax = plt.gca()

    # Reynolds range for correlation
    reynolds = np.logspace(1, 5.5, 100)
    pr = 0.7
    n_rows_loc = 15
    spacing_trans_list = [1.25, 2.5, 5.0]
    spacing_long_default = 1.25  # keep within validated inline range when possible

    # Plot experimental data (single point for reference)
    j_over_f_exp = [j / f for j, f in zip(j_exp_list, f_exp_k_and_l, strict=True)]
    ax.scatter(
        Re_values, j_over_f_exp, label="j/f exp Xt*=1.5", color="blue", marker="^", s=50, zorder=5
    )

    # Plot correlation for different spacing_trans
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green
    for i, xt in enumerate(spacing_trans_list):
        j_over_f_vals = []
        valid_reynolds = []

        for Re in reynolds:
            try:
                nusselt, f_corr = tube_bank_nusselt_number_and_friction_factor(
                    Re, spacing_long_default, xt, pr, inline=True, n_rows=n_rows_loc
                )
                j_factor = nusselt / (Re * pr ** (1 / 3))
                j_over_f = j_factor / f_corr
                j_over_f_vals.append(j_over_f)
                valid_reynolds.append(Re)
            except AssertionError:
                # Skip invalid Reynolds numbers
                continue

        if valid_reynolds:
            ax.plot(
                valid_reynolds,
                j_over_f_vals,
                label=f"j/f corr Xt*={xt}",
                color=colors[i],
                linestyle="-",
                linewidth=2,
            )

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_xlabel("Re (based on tube diameter)")
    ax.set_ylabel("j/f ratio")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best", title=f"Xl*={spacing_long_default}")

    title = "Dimensionless heat transfer to friction ratio of inline tube banks"
    plt.title(title)
    plt.tight_layout()


if __name__ == "__main__":
    # Optional: keep the quick prints for context
    # _print_quick_tables()
    # _print_tube_bank_comparison_table()
    # _print_example_calc()
    # _print_complete_analysis_header()
    # _print_comprehensive_table()
    # _print_excel_friendly()

    if PLOT_CASE == "tube_bank_compare":
        _plot_tube_bank_compare(SHOW_TWIN_J_OVER_F_AXIS)
        plt.show()
    elif PLOT_CASE == "drag_vs_re":
        _plot_drag_vs_re()
        plt.show()
    elif PLOT_CASE == "j_over_f_v_re":
        _plot_j_over_f_v_re()
        plt.show()
    else:
        raise ValueError(f"Unknown PLOT_CASE '{PLOT_CASE}'")
