import os

import matplotlib.pyplot as plt
import numpy as np
from xflow import calculate_pressure_drop_ratio, create_plot

save_dir = os.path.dirname(os.path.abspath(__file__))

# Default modeling assumptions
DEFAULT_C_COLD_OVER_C_HOT = 1.0  # C_cold / C_hot
DEFAULT_ST_OVER_F = 0.4  # Assumed same for both fluids
DEFAULT_F_C_OVER_F_H = 1.0  # f_c/f_h
DEFAULT_D_R = 1.0  # d_r = sigma_r/A_r (cold/hot ratio)
DEFAULT_G2_H = 1e-5  # g2_h

DEFAULT_DP_MAX = 0.2

# Three g^2 values for plotting
PLOT_TRIPLE_G2 = [1e-3, 5e-3, 8e-3]

# Framework-specific parameters for practical
DEFAULT_T = 2.0  # T_hot_in / T_cold_in
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 10.0  # p_cold_in / p_hot_in
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.1  # p_hot_in / p_dead (slider value)
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD  # p_dead / p_hot_in (for calculations)
DEFAULT_GAMMA = 1.4

# Pressure drop assumption: "dp_c=dp_h", "dp_c<<dp_h", or "inlet_density"
# For inlet_density, also need:
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"  # Essential input for accurate cold pressure drop
if DEFAULT_PRESSURE_DROP_ASSUMPTION == "inlet_density":
    DEFAULT_MOLAR_MASS_RATIO = 1.0  # M_cold / M_hot (cold/hot)
    DEFAULT_A_R = 0.1  # A_r (cold/hot) - sigma_r is calculated as d_r * A_r
else:
    DEFAULT_MOLAR_MASS_RATIO = None
    DEFAULT_A_R = None


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="newfig3",
    plot_triple_g2=None,
    t=DEFAULT_T,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
):
    """
    Save figures as SVG, TIFF, and HD PNG for practical framework with multiple g^2 values.

    Parameters:
        plot_triple_g2: List of g^2 values to plot (e.g., [1e-5, 2e-5, 5e-5])
    """
    # Set font sizes to match Word (10pt = 10 points)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 10,
        }
    )

    # Calculate pressure drop ratio based on assumption
    # Calculate sigma_r from d_r * A_r
    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
    )

    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))
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

    # Update ylabel and x-axis formatting
    ax.set_ylabel(r"HEX $\Delta Q_0^M/Q_{\mathrm{max}}$")
    ax.set_xticks([0, 5, 10, 15])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}"))

    # Remove existing legend and recreate using line_list order (matches newfig1: highest g^2 at top)
    legend = ax.get_legend()
    if legend:
        legend.remove()
    # Use line_list order directly - lines are plotted in reversed order (8e-3, 5e-3, 1e-3)
    if line_list:
        handles = line_list
        labels = [line.get_label() for line in line_list]
        ax.legend(handles, labels, loc="upper right")

        # Add optimum point markers (minimum) for each line
        for line in line_list:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            # Filter out invalid/masked data
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if np.any(valid_mask):
                x_plot = x_data[valid_mask]
                y_plot = y_data[valid_mask]
                # Find minimum
                arg_y_min = np.argmin(y_plot)
                ax.scatter(
                    x_plot[arg_y_min],
                    y_plot[arg_y_min],
                    color="black",
                    marker="o",
                    zorder=5,
                )

    plt.tight_layout(pad=0.5)

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

    plt.close(fig)
    print(f"Saved figures: {base_name}.svg, {base_name}.tiff, {base_name}.png")


if __name__ == "__main__":
    # Save figures with practical framework
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="newfig3",
        plot_triple_g2=PLOT_TRIPLE_G2,
        t=DEFAULT_T,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
    )
