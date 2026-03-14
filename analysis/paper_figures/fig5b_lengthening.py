import os

import matplotlib.pyplot as plt
import numpy as np
import xflow
from scipy.signal import find_peaks
from xflow import calculate_pressure_drop_ratio, create_plot

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figs_current")

# Default modeling assumptions (match fig8/9)
DEFAULT_C_COLD_OVER_C_HOT = 1.0
DEFAULT_ST_OVER_F = 0.4
# DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_F_C_OVER_F_H = 0.25
# DEFAULT_D_R = 0.257
DEFAULT_D_R = 0.25
DEFAULT_GAMMA = 1.4
# DEFAULT_MACH_IN = 0.1
DEFAULT_MACH_IN = 0.11  # Mh_in
DEFAULT_G2_H = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2

DEFAULT_DP_MAX = 0.2

# PLOT_TRIPLE_MACH = [0.04, 0.08, 0.17]
PLOT_TRIPLE_MACH = [0.11, 0.06, 0.04]
PLOT_TRIPLE_G2 = [0.5 * DEFAULT_GAMMA * m**2 for m in PLOT_TRIPLE_MACH]

# Framework parameters (match fig8/9, T_ratios from xflow 106-109)
# DEFAULT_T = 898 / 588
DEFAULT_T = 907 / 588
DEFAULT_T_DEAD_OVER_T_COLD_IN = 288 / 588
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
# DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.04
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.064
DEFAULT_MOLAR_MASS_RATIO = 1.0
# DEFAULT_A_R = 1.0
DEFAULT_A_R = 0.92


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig5b_lengthening",
    plot_triple_g2=None,
    t=DEFAULT_T,
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
):
    """
    Save figures as SVG, TIFF, and HD PNG for classical framework with multiple g^2 values.

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
            "axes.titlesize": font_size,
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
        pressure_drop_assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
    )
    print(f"Pressure drop ratio (inlet_density): (dp_c/p_cin)/(dp_h/p_hin) = {pressure_drop_ratio:.6f}")

    # Ensure SHOW_CUBIC is False for this figure
    xflow.SHOW_CUBIC = False
    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))  # IF DOUBLE COLUMN FIGURE, USE THIS
    fig = plt.figure(figsize=((6) / 2.54, 7 / 2.54))  # IF TRIPPLE COLUMN FIGURE, USE THIS
    ax = plt.subplot(111)
    ax_twin = ax.twinx()  # Created but will be hidden for classical framework

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
        framework="classical",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=1.0,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )

    # Negate y-data to plot change in availability (negative of unavailable energy)
    for line in line_list:
        line.set_ydata(-np.array(line.get_ydata()))

    # Remove title if present
    ax.set_title("")

    ax.set_xticks([0, 5, 10, 15])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}"))
    ax.set_ylim(-0.1, 0)
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
        # Add optimum point markers (maximum after first trough, since we plot availability).
        # Skip if optimum is at lowest NTU (index 0) - not a real optimum, e.g. highest Mach case.
        opt_xy = []
        for line in line_list:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if np.any(valid_mask):
                x_plot = x_data[valid_mask]
                y_plot = y_data[valid_mask]
                troughs, _ = find_peaks(-y_plot)
                if len(troughs) > 0:
                    first_trough = troughs[0]
                    y_opt_idx = first_trough + np.argmax(y_plot[first_trough:])
                else:
                    y_opt_idx = np.argmax(y_plot)
                # Skip if optimum is at first (lowest NTU) point - boundary artefact
                if y_opt_idx == 0:
                    continue
                x_opt, y_opt = x_plot[y_opt_idx], y_plot[y_opt_idx]
                opt_xy.append((x_opt, y_opt))
                ax.scatter(
                    x_opt,
                    y_opt,
                    color="white",
                    marker="o",
                    zorder=5,
                    s=50,
                    facecolor="black",
                )
        # Arrows from "optimal designs" label to the valid optimum points; text in empty space above -0.025
        if len(opt_xy) >= 2:
            x_text = 5
            y_text = -0.012  # In empty space above -0.025, within ylim (-0.1, 0)
            arrow_kw = dict(arrowstyle="->", color="black", lw=1, shrinkB=12)
            ax.annotate(
                "optimal designs",
                xy=opt_xy[0],
                xytext=(x_text, y_text),
                fontsize=font_size,
                ha="left",
                arrowprops=arrow_kw,
            )
            ax.annotate(
                "",
                xy=opt_xy[1],
                xytext=(6.8, y_text),
                arrowprops=arrow_kw,
            )

    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax.set_ylabel(r"Change in Availability ($\sum_{i} \; \Delta \dot{W}_{\mathrm{A},i}/\dot{Q}_{\mathrm{max}}$ [%])")
    plt.tight_layout(pad=0.5)
    ax.yaxis.labelpad = -4  # After tight_layout: reduce distance between ticks and ylabel

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
    # Save figures with classical framework
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="fig5b_lengthening",
        plot_triple_g2=PLOT_TRIPLE_G2,
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
    )
