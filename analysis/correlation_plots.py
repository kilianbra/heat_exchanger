from contextlib import suppress

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
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
    tube_bank_nusselt_gnielinski_vdi,
    tube_bank_nusselt_number_and_friction_factor,
    tube_bank_stanton_number_from_murray,
)

# ---------------------------
# Quick plot configuration
# ---------------------------
# Choose which plot to generate:
#   - "tube_bank_inline_compare": Kays & London experimental vs Gaddis & Gnielinski correlation (inline)
#   - "tube_bank_staggered_compare": Gaddis & Gnielinski correlation (staggered)
#   - "drag_vs_re": xi = 2 Hg / Re^2 vs Re for different spacing_trans
#   - "j_over_f_v_re": j/f ratio vs Re for experimental and correlation data
PLOT_CASE = "tube_bank_interactive"


# Toggle twin y-axis for j/f ratio on tube bank compare plot
SHOW_TWIN_J_OVER_F_AXIS = True

# General matplotlib style bits (optional)
DEFAULT_DPI = 120

# Shared Reynolds array for correlation curves
reynolds = np.logspace(1, 5.5, 100)


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
Re_hydraulic_inline = [
    10_000,
    8_000,
    6_000,
    5_000,
    4_000,
    3_000,
    2_500,
    2_000,
    1_500,
    1_200,
    1_000,
    800,
]
# These reynolds are based on the tube outer diameter
Re_values_inline = [12627, 10101, 7576, 6313, 5051, 3788, 3157, 2525, 1894, 1515, 1263, 1010]

f_exp_k_and_l_inline = [
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

# 2D CFD results for inline Xl*=1.25 Xt*=1.5 configuration
Re_cfd_inline = [380, 508, 658, 848, 1100, 1515, 1945, 2461, 2752, 3015]
f_cfd_inline = [0.041, 0.034, 0.028, 0.025, 0.022, 0.0196, 0.0174, 0.0163, 0.016, 0.01702]
# 39k cells 3D CFD - probs not mesh independent
# Re_cfd_inline_3D = [880, 1094, 1433, 1884, 2456, 2497, 4332, 4089]
# f_cfd_inline_3D = [0.0352, 0.0327, 0.0306, 0.0286, 0.0272, 0.0271, 0.0250, 0.0252]

re_cfd_inline_3d = [2494, 3000, 4989, 5997, 8980] + [1164, 1013, 750, 470, 241, 78]
f_cfd_inline_3d = [0.0181, 0.0170, 0.017936, 0.017934, 0.0183] + [
    0.0254,
    0.0269,
    0.0308,
    0.0396,
    0.0609,
    0.1482,
]


j_exp_knl_inline = [
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
re_gaddis_vdi_inline = [
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

xi_gaddis_vdi_inline = [
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


# Staggered placeholders to be populated later
Re_hydraulic_staggered = [
    15000,
    12000,
    10000,
    8000,
    6000,
    5000,
    4000,
    3000,
    2500,
    2000,
    1500,
    1200,
    1000,
    800,
]
X_l_knl = 1.25
X_t_knl = 1.5
Re_values_staggered = [rh * 4 * X_l_knl * (X_t_knl - 1) / np.pi for rh in Re_hydraulic_staggered]
j_exp_knl_staggered = [
    0.00632,
    0.00698,
    0.00755,
    0.00832,
    0.00941,
    0.0102,
    0.0112,
    0.0127,
    0.0137,
    0.0149,
    0.0166,
    0.0178,
    0.0189,
    0.0201,
]
f_exp_k_and_l_staggered = [
    0.0508,
    0.0530,
    0.0550,
    0.0578,
    0.0614,
    0.0640,
    0.0670,
    0.0702,
    0.0725,
    0.0750,
    0.0780,
    0.0800,
    0.0812,
    0.0827,
]
re_gaddis_vdi_staggered = []
xi_gaddis_vdi_staggered = []

# Calculate correlation values (inline) using tube_bank_friction_factor
f_corr_list_inline = []
for Re in Re_values_inline:
    f_corr = tube_bank_friction_factor(Re, spacing_long, spacing_trans, inline=True, n_rows=n_rows)
    f_corr_list_inline.append(f_corr)

# Create comparison table
tube_bank_table_data = []
for i, Re in enumerate(Re_values_inline):
    tube_bank_table_data.append(
        [
            f"{Re:.2e}",  # Reynolds in scientific notation with 2 sig figs
            f"{f_exp_k_and_l_inline[i]:.2e}",  # Experimental in scientific notation with 2 sig figs
            f"{f_corr_list_inline[i]:.2e}",  # Correlation in scientific notation with 2 sig figs
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


# Calculate correlation values for all Reynolds numbers (inline)
nusselt_corr_list = []
f_corr_list_new = []
j_corr_list = []

Pr = 0.7  # Prandtl number

for Re in Re_values_inline:
    nusselt, f_corr = tube_bank_nusselt_number_and_friction_factor(
        Re, spacing_long, spacing_trans, Pr, inline=True, n_rows=n_rows
    )
    j_corr = nusselt / (Re * Pr ** (1 / 3))

    nusselt_corr_list.append(nusselt)
    f_corr_list_new.append(f_corr)
    j_corr_list.append(j_corr)

# Calculate experimental j-factors from j_exp_knl data
j_exp_list = j_exp_knl_inline  # These are already j-factors from the experimental data

# Create comprehensive comparison table
comprehensive_table_data = []
for i, Re in enumerate(Re_values_inline):
    comprehensive_table_data.append(
        [
            f"{Re:.4e}",  # Reynolds in scientific notation
            f"{f_exp_k_and_l_inline[i]:.4f}",  # Experimental friction factor
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
    for i, Re in enumerate(Re_values_inline):
        print(
            f"{Re}\t{f_exp_k_and_l_inline[i]:.6f}\t{f_corr_list_new[i]:.6f}\t{j_exp_list[i]:.6f}\t{j_corr_list[i]:.6f}"
        )


# ---------------------------
# Plotting utilities
# ---------------------------
def _plot_tube_bank_inline_compare(show_twin_j_over_f_axis: bool = True):
    plt.figure(dpi=DEFAULT_DPI)
    ax = plt.gca()

    # f and j vs Re on same (left) axis with log-log scaling
    ax.scatter(
        Re_values_inline, f_exp_k_and_l_inline, label="f exp (K&L)", color="#1f77b4", marker="o"
    )
    ax.plot(
        Re_values_inline, f_corr_list_new, label="f corr (G&G)", color="#1f77b4", linestyle="--"
    )
    ax.scatter(Re_values_inline, j_exp_list, label="j exp (K&L)", color="#d62728", marker="s")
    ax.plot(Re_values_inline, j_corr_list, label="j corr (G&G)", color="#d62728", linestyle="-")
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
        j_over_f_exp = [j / f for j, f in zip(j_exp_list, f_exp_k_and_l_inline, strict=True)]
        j_over_f_corr = [j / f for j, f in zip(j_corr_list, f_corr_list_new, strict=True)]

        ax2.scatter(
            Re_values_inline, j_over_f_exp, label="j/f exp (K&L)", color="black", marker="^"
        )
        ax2.plot(
            Re_values_inline, j_over_f_corr, label="j/f corr (G&G)", color="black", linestyle=":"
        )
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


def _plot_tube_bank_staggered_compare(show_twin_j_over_f_axis: bool = True):
    plt.figure(dpi=DEFAULT_DPI)
    ax = plt.gca()

    # Compute correlation curves over shared reynolds array for staggered layout
    f_corr_vals = []
    j_corr_vals = []
    Pr_local = 0.7
    for Re in reynolds:
        try:
            nusselt, f_corr = tube_bank_nusselt_number_and_friction_factor(
                Re, spacing_long, spacing_trans, Pr_local, inline=False, n_rows=n_rows
            )
            j_factor = nusselt / (Re * Pr_local ** (1 / 3))
            f_corr_vals.append(f_corr)
            j_corr_vals.append(j_factor)
        except AssertionError:
            f_corr_vals.append(np.nan)
            j_corr_vals.append(np.nan)

    ax.plot(reynolds, f_corr_vals, label="f corr (G&G)", color="#1f77b4", linestyle="--")
    ax.plot(reynolds, j_corr_vals, label="j corr (G&G)", color="#d62728", linestyle="-")
    ax.scatter(
        Re_values_staggered,
        j_exp_knl_staggered,
        label="j exp (K&L)",
        color="#d62728",
        marker="s",
    )
    ax.scatter(
        Re_values_staggered,
        f_exp_k_and_l_staggered,
        label="f exp (K&L)",
        color="#1f77b4",
        marker="o",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re (based on tube diameter)")
    ax.set_ylabel("f and j")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    if show_twin_j_over_f_axis:
        ax2 = ax.twinx()
        with np.errstate(invalid="ignore", divide="ignore"):
            j_over_f_corr = [j / f for j, f in zip(j_corr_vals, f_corr_vals, strict=True)]
        ax2.plot(reynolds, j_over_f_corr, label="j/f corr (G&G)", color="black", linestyle=":")
        ax2.set_ylabel("j/f ratio")
        ax2.set_yscale("linear")
        ax2.set_ylim(0.0, 0.5)

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
    else:
        ax.legend(loc="best")

    title = f"Staggered tube bank: Xt*={spacing_trans}, Xl*={spacing_long}, N_rows={n_rows}"
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
    xi_exp = [np.pi / (spacing_trans - 1) * f_exp for f_exp in f_exp_k_and_l_inline]
    ax.plot(
        re_gaddis_vdi_inline,
        xi_gaddis_vdi_inline,
        label="exp (? Gaddis) Xt*=1.25",
        color="C0",
        linestyle="--",
    )
    ax.scatter(Re_values_inline, xi_exp, label="exp (K&L) Xt*=1.5", color="C1", marker="x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re (based on tube diameter)")
    ax.set_ylabel(r"Drag factor $\xi = 2\rho \Delta p / (N_r G^2) =  \pi f_o/(X_t^*-1)$")
    ax.set_ylim(1e-3, 1e2)
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
    j_over_f_exp = [j / f for j, f in zip(j_exp_list, f_exp_k_and_l_inline, strict=True)]
    ax.scatter(
        Re_values_inline,
        j_over_f_exp,
        label="j/f exp Xt*=1.5",
        color="blue",
        marker="^",
        s=50,
        zorder=5,
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


def _plot_tube_bank_interactive():
    plt.figure(figsize=(12, 8), dpi=DEFAULT_DPI)

    # Main plot area - leave space on the right for controls
    ax = plt.axes([0.1, 0.1, 0.6, 0.8])

    # Controls layout - all on the right side
    axcolor = "lightgoldenrodyellow"
    # Widget axes positioned on the right side
    rax_metric = plt.axes([0.75, 0.7, 0.2, 0.15], facecolor=axcolor)
    rax_layout = plt.axes([0.75, 0.52, 0.2, 0.12], facecolor=axcolor)
    sax_xt = plt.axes([0.75, 0.4, 0.2, 0.03], facecolor=axcolor)
    sax_xl = plt.axes([0.75, 0.45, 0.2, 0.03], facecolor=axcolor)

    # Defaults
    default_xt = 1.5
    default_xl = 1.25
    pr_local = 0.7

    # Radio buttons
    metric_radio = RadioButtons(rax_metric, ("Friction coeff.", "Stanton number", "j/f"), active=0)
    layout_radio = RadioButtons(rax_layout, ("Inline", "Staggered"), active=0)

    # Sliders - will be updated based on layout selection
    xt_slider = Slider(sax_xt, "Xt*", 1.25, 6.0, valinit=default_xt, valstep=0.01)
    xl_slider = Slider(sax_xl, "Xl*", 1.2, 3.0, valinit=default_xl, valstep=0.01)

    # Lines
    (line_gg,) = ax.plot([], [], label="Gaddis and Gnielinski", color="#1f77b4", linestyle="-")
    (line_gs,) = ax.plot([], [], label="Gunter and Shaw", color="#9467bd", linestyle="--")
    (line_mr,) = ax.plot([], [], label="Murray (REL)", color="#2ca02c", linestyle=":", marker="o")
    (line_gn,) = ax.plot([], [], label="Gnielinski VDI", color="#ff7f0e", linestyle="-.")
    (line_zk,) = ax.plot([], [], label="Zukauskas (1972)", color="#8c564b", linestyle="-.")

    # Experimental scatters (created upfront, toggled visible)
    exp_scatter = ax.scatter([], [], label="Kays & London (exp)", color="black", marker="x")
    cfd_scatter = ax.scatter([], [], label="2D CFD", color="red", marker="o", s=50)
    cfd_3d_scatter = ax.scatter(
        [], [], label="3D CFD (249k cells)", color="orange", marker="s", s=50
    )

    # Brewer point (only shown for specific Xt*, Xl*)
    brewer_scatter = ax.scatter([], [], label="Brewer", color="black", marker="+", s=70)

    # Handle for shaded CFD agreement band (managed to avoid duplicates)
    cfd_band = None

    ax.set_xscale("log")
    ax.set_xlabel("Re (based on tube diameter)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    # State - experimental data shows when sliders match defaults

    def update_slider_bounds(is_inline: bool):
        """Update slider bounds based on layout selection"""
        # Store current values
        current_xt = xt_slider.val
        current_xl = xl_slider.val

        if is_inline:
            # Inline bounds: Xt* [1.25, 3.0], Xl* [1.2, 3.0]
            xt_min, xt_max = 1.25, 6.0
            xl_min, xl_max = 1.2, 3.0
        else:
            # Staggered bounds: Xt* [1.25, 3.0], Xl* [0.6, 3.0]
            xt_min, xt_max = 1.25, 3.0
            xl_min, xl_max = 0.6, 3.0

        # Clamp current values to new bounds
        new_xt = max(xt_min, min(xt_max, current_xt))
        new_xl = max(xl_min, min(xl_max, current_xl))

        # Update sliders with new bounds by recreating them
        xt_slider.valmin = xt_min
        xt_slider.valmax = xt_max
        xt_slider.set_val(new_xt)

        xl_slider.valmin = xl_min
        xl_slider.valmax = xl_max
        xl_slider.set_val(new_xl)

    def mask_bounds(
        values: list[float], xl_val: float, xt_val: float, is_inline: bool
    ) -> list[float]:
        if is_inline:
            valid = (xt_val >= 1.25) and (xt_val <= 6.0) and (xl_val >= 1.2) and (xl_val <= 3.0)
        else:
            valid = (xt_val >= 1.25) and (xt_val <= 3.0) and (xl_val >= 0.6) and (xl_val <= 3.0)
        if not valid:
            return [np.nan] * len(values)
        return values

    def compute_series(metric_name: str, layout_name: str, xl_val: float, xt_val: float):
        is_inline = layout_name == "Inline"

        # Initialize arrays for all correlations
        gg_y = []
        gs_y = []
        mr_y = []
        gn_y = []
        zk_y = []

        for Re in reynolds:  # based on tube diameter
            try:
                nu, f_gg = tube_bank_nusselt_number_and_friction_factor(
                    Re, xl_val, xt_val, pr_local, inline=is_inline, n_rows=n_rows
                )
                xi = tube_bank_corrected_xi_gunter_and_shaw(
                    Re, xl_val, xt_val, use_outside_bounds=False
                )
                f_gs = (xt_val - 1) / np.pi * xi

                # Gnielinski VDI correlation
                nu_gn = tube_bank_nusselt_gnielinski_vdi(
                    Re,
                    xl_val,
                    xt_val,
                    prandtl=pr_local,
                    inline=is_inline,
                    n_rows=11,
                    use_outside_bounds=False,
                )

                # Murray correlation - only for staggered and within valid range
                if not is_inline:
                    St_mr, f_mr = tube_bank_stanton_number_from_murray(
                        Re, xl_val, xt_val, prandtl=pr_local, use_outside_bounds=False
                    )

                else:
                    St_mr, f_mr = np.nan, np.nan

                # Zukauskas (1972) correlation for tube banks (external crossflow)
                # NOTE: Ignore the row correction (C2 = 1). Valid for 1e3 <= Re <= 2e5.
                # Velocity for Re in Zukauskas is based on the mean velocity in the minimum free cross-section,
                # i.e. G_max = G_inf * Xt / (Xt - 1); Re uses the tube diameter (Eq. 7 in Zukauskas 1972; nomenclature there).
                if 1e3 <= Re <= 2e5:
                    if is_inline:
                        # Inline default: C1 = 0.27, m = 0.63 (Zukauskas)
                        zuk_C1 = 0.27
                        zuk_m = 0.63
                        # Override m for specific (Xt*, Xl*) pairs (Fig. 51 guided values)
                        # NOTE: would need to also get values of C1 but not shown in Zuk 1972
                        overrides = {
                            (1.30, 2.60): 0.60,
                            (1.30, 2.00): 0.60,
                            (1.30, 1.30): 0.63,
                            (2.50, 2.00): 0.63,
                            (2.50, 1.30): 0.65,
                            (2.50, 1.10): 0.73,
                            (1.65, 2.00): 0.62,
                            (2.00, 2.00): 0.635,
                            (1.95, 1.30): 0.645,
                        }
                        for (xt_o, xl_o), m_o in overrides.items():
                            if np.isclose(xt_val, xt_o, rtol=0, atol=1e-3) and np.isclose(
                                xl_val, xl_o, rtol=0, atol=1e-3
                            ):
                                zuk_m = m_o
                                break
                    else:
                        # Staggered: spacing dependence per Zukauskas
                        if xt_val / xl_val < 2:
                            zuk_C1 = 0.35 * (xt_val / xl_val) ** 0.2
                            zuk_m = 0.6
                        else:
                            zuk_C1 = 0.4
                            zuk_m = 0.6
                    nu_zk = zuk_C1 * (Re**zuk_m) * (pr_local**0.36)
                else:
                    nu_zk = np.nan

                if metric_name == "Friction coeff.":
                    gg_y.append(f_gg)
                    gs_y.append(f_gs)
                    mr_y.append(f_mr)
                    gn_y.append(np.nan)  # Gnielinski VDI doesn't provide friction
                    zk_y.append(np.nan)  # Zukauskas shown only for Stanton number
                elif metric_name == "Stanton number":
                    St_gg = nu / (Re * pr_local)
                    St_gn = nu_gn / (Re * pr_local)
                    gg_y.append(St_gg)
                    # Gunter & Shaw provides only friction (via xi); Stanton not defined → NaN
                    gs_y.append(np.nan)
                    mr_y.append(St_mr)
                    gn_y.append(St_gn)
                    zk_y.append(nu_zk / (Re * pr_local) if np.isfinite(nu_zk) else np.nan)
                else:  # j/f
                    j_gg = nu / (Re * pr_local ** (1 / 3))
                    gg_y.append(j_gg / f_gg if f_gg else np.nan)
                    # Convert Murray St to j then divide by f
                    j_mr = St_mr * pr_local ** (2 / 3)
                    mr_y.append(j_mr / f_mr if f_mr else np.nan)
                    # Gunter & Shaw j is not defined; set NaN
                    gs_y.append(np.nan)
                    # Gnielinski VDI doesn't provide friction, so j/f not available
                    gn_y.append(np.nan)
                    zk_y.append(np.nan)  # Zukauskas shown only for Stanton number
            except (AssertionError, Exception):
                gg_y.append(np.nan)
                gs_y.append(np.nan)
                mr_y.append(np.nan)
                gn_y.append(np.nan)
                zk_y.append(np.nan)

        gg_y = mask_bounds(gg_y, xl_val, xt_val, is_inline)
        gs_y = mask_bounds(gs_y, xl_val, xt_val, is_inline)
        mr_y = mask_bounds(mr_y, xl_val, xt_val, is_inline)
        gn_y = mask_bounds(gn_y, xl_val, xt_val, is_inline)
        zk_y = mask_bounds(zk_y, xl_val, xt_val, is_inline)
        return gg_y, gs_y, mr_y, gn_y, zk_y, is_inline

    def update_plot(_=None):
        nonlocal cfd_band
        metric = metric_radio.value_selected
        layout_name = layout_radio.value_selected
        xl_val = xl_slider.val
        xt_val = xt_slider.val

        gg_y, gs_y, mr_y, gn_y, zk_y, is_inline = compute_series(
            metric, layout_name, xl_val, xt_val
        )
        line_gg.set_data(reynolds, gg_y)
        line_gs.set_data(reynolds, gs_y)
        line_mr.set_data(reynolds, mr_y)
        line_gn.set_data(reynolds, gn_y)
        line_zk.set_data(reynolds, zk_y)

        # Show/hide lines based on layout and metric
        line_mr.set_visible(not is_inline)  # Murray only for staggered
        # Gnielinski VDI only for Stanton number
        line_gn.set_visible(metric == "Stanton number")
        # Zukauskas only for Stanton number
        line_zk.set_visible(metric == "Stanton number")

        # Y-axis scaling
        ax.set_yscale("log" if metric != "j/f" else "linear")
        if metric == "Friction coeff.":
            ax.set_ylabel("f")
        elif metric == "Stanton number":
            ax.set_ylabel("St")
        else:
            ax.set_ylabel("j/f")

        # Experimental overlay when sliders match defaults
        show_exp = abs(xl_val - default_xl) < 1e-9 and abs(xt_val - default_xt) < 1e-9
        if show_exp:
            if is_inline:
                re_exp = Re_values_inline
                f_exp = f_exp_k_and_l_inline
                j_exp = j_exp_knl_inline
            else:
                re_exp = Re_values_staggered
                f_exp = f_exp_k_and_l_staggered
                j_exp = j_exp_knl_staggered

            if metric == "Friction coeff.":
                exp_y = f_exp
            elif metric == "Stanton number":
                # Convert j to St: St = j * Pr^(2/3)
                exp_y = [jv * pr_local ** (2 / 3) for jv in j_exp]
            else:
                # j/f from experiment: j_exp / f_exp
                exp_y = [jv / fv for jv, fv in zip(j_exp, f_exp, strict=True)]
            exp_scatter.set_offsets(np.column_stack((re_exp, exp_y)))
            exp_scatter.set_visible(True)
        else:
            exp_scatter.set_visible(False)

        # CFD data overlay when sliders match defaults AND inline layout AND friction coefficient
        show_cfd = show_exp and is_inline and metric == "Friction coeff."
        if show_cfd:
            # 2D CFD data
            cfd_y = f_cfd_inline
            cfd_scatter.set_offsets(np.column_stack((Re_cfd_inline, cfd_y)))
            cfd_scatter.set_visible(True)

            # 3D CFD data
            cfd_3d_y = f_cfd_inline_3d
            cfd_3d_scatter.set_offsets(np.column_stack((re_cfd_inline_3d, cfd_3d_y)))
            cfd_3d_scatter.set_visible(True)

            # Manage shaded agreement band between f_GG and 0.6 * f_GG without accumulating artists
            lower_band = [0.6 * fv if np.isfinite(fv) else np.nan for fv in gg_y]
            if cfd_band is None:
                cfd_band = ax.fill_between(
                    reynolds,
                    lower_band,
                    gg_y,
                    color="blue",
                    alpha=0.1,
                    label="G&G -40%",
                    zorder=0,
                )
            else:
                # Replace existing band with updated data
                with suppress(Exception):
                    cfd_band.remove()
                cfd_band = ax.fill_between(
                    reynolds,
                    lower_band,
                    gg_y,
                    color="blue",
                    alpha=0.1,
                    label="G&G -40%",
                    zorder=0,
                )
            cfd_band.set_visible(True)
        else:
            cfd_scatter.set_visible(False)
            cfd_3d_scatter.set_visible(False)
            if cfd_band is not None:
                cfd_band.set_visible(False)

        # Brewer point overlay when Inline, Friction coeff., Xt*=6.0 and Xl*=1.25
        show_brewer = (
            is_inline
            and metric == "Friction coeff."
            and abs(xl_val - 1.25) < 1e-6
            and abs(xt_val - 6.0) < 1e-6
        )
        if show_brewer:
            brewer_scatter.set_offsets(np.array([[3000.0, 0.05]]))
            brewer_scatter.set_visible(True)
        else:
            brewer_scatter.set_visible(False)

        ax.relim()
        if metric == "j/f":
            ax.set_ylim(0.0, 0.5)
        elif metric == "Friction coeff.":
            ax.set_ylim(0.01, 2e0)  # Include CFD data down to ~0.016
        else:
            ax.autoscale(axis="y")
        # Dynamic legend: include only visible artists
        legend_artists = [
            art
            for art in [
                line_gg,
                line_gs,
                line_mr,
                line_gn,
                line_zk,
                exp_scatter,
                cfd_scatter,
                cfd_3d_scatter,
                brewer_scatter,
            ]
            if art.get_visible()
        ]
        if cfd_band is not None and cfd_band.get_visible():
            legend_artists.append(cfd_band)
        if legend_artists:
            ax.legend(handles=legend_artists, loc="best")
        else:
            # Fallback to default legend behavior if everything is hidden (unlikely)
            ax.legend(loc="best")
        plt.draw()

    def on_layout_change(_):
        """Handle layout radio button change"""
        layout_name = layout_radio.value_selected
        is_inline = layout_name == "Inline"
        update_slider_bounds(is_inline)
        update_plot()

    def on_slider_change(_):
        update_plot()

    metric_radio.on_clicked(update_plot)
    layout_radio.on_clicked(on_layout_change)
    xt_slider.on_changed(on_slider_change)
    xl_slider.on_changed(on_slider_change)

    # Initial setup and bounds
    update_slider_bounds(True)  # Start with inline bounds
    update_plot()

    plt.show()


if __name__ == "__main__":
    # Optional: keep the quick prints for context
    # _print_quick_tables()
    # _print_tube_bank_comparison_table()
    # _print_example_calc()
    # _print_complete_analysis_header()
    # _print_comprehensive_table()
    # _print_excel_friendly()

    if PLOT_CASE == "tube_bank_inline_compare":
        _plot_tube_bank_inline_compare(SHOW_TWIN_J_OVER_F_AXIS)
        plt.show()
    elif PLOT_CASE == "tube_bank_staggered_compare":
        _plot_tube_bank_staggered_compare(SHOW_TWIN_J_OVER_F_AXIS)
        plt.show()
    elif PLOT_CASE == "drag_vs_re":
        _plot_drag_vs_re()
        plt.show()
    elif PLOT_CASE == "j_over_f_v_re":
        _plot_j_over_f_v_re()
        plt.show()
    elif PLOT_CASE == "tube_bank_interactive":
        _plot_tube_bank_interactive()
        plt.show()
    else:
        raise ValueError(f"Unknown PLOT_CASE '{PLOT_CASE}'")
