import os

import matplotlib.pyplot as plt
import numpy as np
import xflow
from fig_paths import JOURNAL_PLOTS, ensure_fig_dirs
from plot_colors import MARKER_SIZE_LATEX, TITLE_FONTSIZE
from xflow import calculate_pressure_drop_ratio, create_plot

ensure_fig_dirs()
save_dir = JOURNAL_PLOTS

# Default modeling assumptions (match fig8/9)
DEFAULT_C_COLD_OVER_C_HOT = 1.0
DEFAULT_ST_OVER_F = 0.4
# DEFAULT_F_C_OVER_F_H = 0.25  # old: compensated missing f in inlet_density ratio
DEFAULT_F_C_OVER_F_H = 1.0
# DEFAULT_D_R = 0.257
DEFAULT_D_R = 0.25
DEFAULT_GAMMA = 1.4
# DEFAULT_MACH_IN = 0.1
# DEFAULT_MACH_IN = 0.11  # Mh_in (old, with f_c/f_h=0.25)
DEFAULT_MACH_IN = 0.1362  # Mh_in from xflow Helicopte_retrofit (f_c/f_h=1)
DEFAULT_G2_H = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2

DEFAULT_DP_MAX = 0.2

# PLOT_TRIPLE_MACH = [0.04, 0.08, 0.17]
# PLOT_TRIPLE_MACH = [0.11, 0.06, 0.04]
PLOT_TRIPLE_MACH = [0.1362, 0.06, 0.04]
PLOT_TRIPLE_G2 = [0.5 * DEFAULT_GAMMA * m**2 for m in PLOT_TRIPLE_MACH]

# Framework parameters (match fig8/9, T_ratios from xflow 106-109)
# DEFAULT_T = 898 / 588
DEFAULT_T = 908 / 588
# DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.04
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.064
# DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.04
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.064
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_MOLAR_MASS_RATIO = 1.0
# DEFAULT_A_R = 1.0
DEFAULT_A_R = 0.92

# A = A_ref when NTU = NTU_MATCH; A/A_ref = NTU/NTU_MATCH
NTU_MATCH = 1.479


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig6c_lengthening",
    plot_triple_g2=None,
    t=DEFAULT_T,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
    plot_area_ratio_ref=False,
    plot_ref_plus=False,
):
    """
    Save figures as SVG, TIFF, and HD PNG for practical framework with multiple g^2 values.

    Parameters:
        plot_triple_g2: List of g^2 values to plot (e.g., [1e-5, 2e-5, 5e-5])
    """
    # Set font sizes to match Word (10pt = 10 points)
    font_size = 8
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": font_size,
            "mathtext.fontset": "stix",
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
        }
    )

    # Calculate pressure drop ratio based on assumption
    # Calculate sigma_r from d_r * A_r
    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption,
        c_cold_over_c_hot,
        t,
        d_r,
        molar_mass_ratio,
        sigma_r,
        p_cold_in_over_p_hot_in,
        f_c_over_f_h=f_c_over_f_h,
    )

    # Ensure SHOW_CUBIC is False for this figure
    xflow.SHOW_CUBIC = False

    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))  # IF DOUBLE COLUMN FIGURE, USE THIS
    fig = plt.figure(figsize=((6) / 2.54, 7 / 2.54))  # IF TRIPPLE COLUMN FIGURE, USE THIS
    ax = plt.subplot(111)
    ax_twin = ax.twinx()  # Created but will be hidden for practical framework

    _, line_list, ax, ax_twin = create_plot(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=15.0,
        dp_max=dp_max,
        ax=ax,
        ax_twin=ax_twin,
        plot_triple_g2=plot_triple_g2,
        framework="practical",
        t=t,
        t_dead_over_t_cold_in=1.0,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )

    # Negate y-data to plot change in availability (negative of unavailable energy)
    for line in line_list:
        line.set_ydata(-np.array(line.get_ydata()))

    # Remove title if present
    ax.set_title("")

    if plot_area_ratio_ref:
        base_name = base_name.replace(
            "_lengthening", "_A_A_ref_w_plus" if plot_ref_plus else "_A_A_ref"
        )
        # Transform x from NTU to A/A_ref = NTU/NTU_MATCH
        for line in line_list:
            line.set_xdata(np.array(line.get_xdata()) / NTU_MATCH)
        for ax_use in (ax, ax_twin):
            for line in ax_use.get_lines():
                if line not in line_list:
                    line.set_xdata(np.array(line.get_xdata()) / NTU_MATCH)
            for coll in ax_use.collections:
                if hasattr(coll, "get_offsets") and coll.get_offsets().size > 0:
                    off = coll.get_offsets().copy()
                    off[:, 0] = off[:, 0] / NTU_MATCH
                    coll.set_offsets(off)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
        ax.set_xlabel(r"Heat Transfer Area $A/A_\mathrm{ref}$ [-]")
        # Add + marker at reference design (A/A_ref=1) on practical availability line (baseline Mach only)
        if plot_ref_plus and line_list and len(line_list) > 2:
            x_ref = 1.0
            line_m011 = line_list[2]  # [0]=M0.04, [1]=M0.06, [2]=M0.1362
            x_ln, y_ln = line_m011.get_xdata(), line_m011.get_ydata()
            if np.min(x_ln) <= x_ref <= np.max(x_ln):
                y_at_ref = np.interp(x_ref, x_ln, y_ln)
                ax.scatter(
                    x_ref, y_at_ref, marker="+", s=MARKER_SIZE_LATEX,
                    linewidths=1, color="black", zorder=5
                )
    else:
        if plot_ref_plus:
            base_name = base_name.replace("_lengthening", "_NTU_w_plus")
        ax.set_xticks([0, 5, 10, 15])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}"))
        ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
        # Add + marker at reference design (NTU=NTU_MATCH) on practical availability line (baseline Mach only)
        if plot_ref_plus and line_list and len(line_list) > 2:
            x_ref = NTU_MATCH
            line_m011 = line_list[2]
            x_ln, y_ln = line_m011.get_xdata(), line_m011.get_ydata()
            if np.min(x_ln) <= x_ref <= np.max(x_ln):
                y_at_ref = np.interp(x_ref, x_ln, y_ln)
                ax.scatter(
                    x_ref, y_at_ref, marker="+", s=MARKER_SIZE_LATEX,
                    linewidths=1, color="black", zorder=5
                )
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(round(x * 100))}%"))

    # Remove existing legend and recreate using line_list order (matches newfig1: highest g^2 at top)
    legend = ax.get_legend()
    if legend:
        legend.remove()
    # # Use line_list order directly - lines are plotted in reversed order (8e-3, 5e-3, 1e-3)
    # if line_list:
    #     handles = line_list
    #     labels = [line.get_label() for line in line_list]
    #     ax.legend(handles, labels, loc="upper right")

    # Add new legend matching newfig1 style with custom labels and title
    if line_list:
        # Add optimum point markers (minimum) for each line
        for line in line_list:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            # Filter out invalid/masked data
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if np.any(valid_mask):
                x_plot = x_data[valid_mask]
                y_plot = y_data[valid_mask]
                # Find maximum (optimum = max availability)
                arg_y_max = np.argmax(y_plot)
                ax.scatter(
                    x_plot[arg_y_max],
                    y_plot[arg_y_max],
                    color="white",
                    marker="o",
                    zorder=5,
                    facecolor="black",
                    s=50,
                )

    if not plot_area_ratio_ref:
        ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax.set_ylabel(
        r"Change in Availability ($\sum_{i} \; \Delta \dot{W}^{\mathrm{M}}_{\mathrm{A},i}/\dot{Q}_{\mathrm{max}}$ [%])"
    )
    plt.tight_layout(pad=0.5)
    ax.yaxis.labelpad = -4  # After tight_layout: reduce distance between ticks and ylabel

    # Top-center label with arrow indicating increasing length
    ax.text(
        0.5,
        0.98,
        "increasing length",
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=font_size,
    )
    ax.annotate(
        "",
        xy=(0.7, 0.92),
        xytext=(0.3, 0.92),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", linewidth=0.75),
    )

    # Save as SVG
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.svg"),
        dpi=300,
        facecolor="white",
        format="svg",
        bbox_inches=None,
        pad_inches=0,
    )

    # Save as TIFF
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.tiff"),
        dpi=300,
        facecolor="white",
        format="tiff",
        bbox_inches=None,
        pad_inches=0,
    )

    # Save as HD PNG
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.png"),
        dpi=300,
        facecolor="white",
        format="png",
        bbox_inches=None,
        pad_inches=0,
    )

    # Save as HD PDF
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.pdf"),
        dpi=300,
        facecolor="white",
        format="pdf",
        bbox_inches=None,
        pad_inches=0,
    )

    plt.close(fig)
    print(f"Saved figures: {base_name}.svg, {base_name}.tiff, {base_name}.png, {base_name}.pdf")


if __name__ == "__main__":
    PLOT_AREA_RATIO_REF = False  # Set False for standard NTU x-axis; True saves fig6c_A_A_ref.svg etc.
    PLOT_REF_PLUS = True  # Set True to add + at reference design; saves fig6c_NTU_w_plus.svg when NTU axis
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="fig6c_lengthening",
        plot_triple_g2=PLOT_TRIPLE_G2,
        t=DEFAULT_T,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
        plot_area_ratio_ref=PLOT_AREA_RATIO_REF,
        plot_ref_plus=PLOT_REF_PLUS,
    )
