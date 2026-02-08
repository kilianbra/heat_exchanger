import os

import matplotlib.pyplot as plt
import numpy as np
import xflow
from scipy.signal import find_peaks
from xflow import calculate_pressure_drop_ratio, create_plot

save_dir = os.path.dirname(os.path.abspath(__file__))

# Default modeling assumptions
DEFAULT_C_COLD_OVER_C_HOT = 1.0  # C_cold / C_hot
DEFAULT_ST_OVER_F = 0.4  # Assumed same for both fluids
DEFAULT_F_C_OVER_F_H = 1.0  # f_c/f_h
DEFAULT_D_R = 1.0  # d_r = sigma_r/A_r (cold/hot ratio)
DEFAULT_G2_H = 1e-5  # g2_h

DEFAULT_DP_MAX = 0.3

# Three g^2 values for plotting
PLOT_TRIPLE_G2 = [1e-3, 5e-3, 20e-3]

# Framework-specific parameters for classical
DEFAULT_T = 2.0  # T_hot_in / T_cold_in
DEFAULT_T_DEAD_OVER_T_COLD_IN = 1.1
DEFAULT_GAMMA = 1.4

# Pressure drop assumption: "dp_c=dp_h", "dp_c<<dp_h", or "inlet_density"
# For inlet_density, also need:
DEFAULT_PRESSURE_DROP_ASSUMPTION = "dp_c=dp_h"  # Essential input for accurate cold pressure drop
if DEFAULT_PRESSURE_DROP_ASSUMPTION == "inlet_density":
    DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 10.0  # p_cold_in / p_hot_in (needed for inlet_density calculation)
    DEFAULT_MOLAR_MASS_RATIO = 1.0  # M_cold / M_hot (cold/hot)
    DEFAULT_A_R = 0.1  # A_r (cold/hot) - sigma_r is calculated as d_r * A_r
else:
    DEFAULT_P_COLD_IN_OVER_P_HOT_IN = None
    DEFAULT_MOLAR_MASS_RATIO = None
    DEFAULT_A_R = None


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig2b",
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

    # Ensure SHOW_CUBIC is False for this figure
    xflow.SHOW_CUBIC = False
    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54)) # IF DOUBLE COLUMN FIGURE, USE THIS
    fig = plt.figure(figsize=((6) / 2.54, 7 / 2.54)) # IF TRIPPLE COLUMN FIGURE, USE THIS
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

    # Update ylabel and x-axis formatting
    ax.set_ylabel(r"HEx $\Delta Q_0/Q_{\mathrm{max}}$")

    # Remove title if present
    ax.set_title("")

    ax.set_xticks([0, 5, 10, 15])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}"))
    ax.set_ylim(0, 0.3)

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
        # Add optimum point markers (minimum after first peak) for each line
        opt_xy = []
        for line in line_list:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            # Filter out invalid/masked data
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if np.any(valid_mask):
                x_plot = x_data[valid_mask]
                y_plot = y_data[valid_mask]
                # Find peaks in the data
                peaks, _ = find_peaks(y_plot)
                if len(peaks) > 0:
                    first_peak = peaks[0]
                    # Find minimum after first peak
                    y_local_min_idx = first_peak + np.argmin(y_plot[first_peak:])
                    x_opt, y_opt = x_plot[y_local_min_idx], y_plot[y_local_min_idx]
                    opt_xy.append((x_opt, y_opt))
                    ax.scatter(
                        x_opt,
                        y_opt,
                        color="black",
                        marker="o",
                        zorder=5,
                    )
        # Two arrows from one label "optimal designs" at y=0.15 to the two optimum points
        if len(opt_xy) >= 2:
            x_text = np.mean([p[0] for p in opt_xy]) + 3.5
            x_text = 5
            y_text = 0.12
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
                xytext=(8.8, y_text-0.005),
                arrowprops=arrow_kw,
            )

    plt.tight_layout(pad=0.5)

    # Independent control for figure
    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax.set_ylabel(r"Change in Unavailable Energy ($\Delta \dot{Q}_0/\dot{Q}_{\mathrm{max}}$)")

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
        base_name="fig2b",
        plot_triple_g2=PLOT_TRIPLE_G2,
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
    )
