import os

import matplotlib.pyplot as plt
import numpy as np
import xflow
from xflow import calculate_pressure_drop_ratio, create_plot

save_dir = os.path.dirname(os.path.abspath(__file__))

# Helicopter defaults (hardcoded from xflow.py)
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_C_COLD_OVER_C_HOT = 0.95  # C_cold / C_hot
DEFAULT_D_R = 0.44
DEFAULT_G2_H = 2e-2
DEFAULT_ST_OVER_F = 0.4  # Assumed same for both fluids
DEFAULT_F_C_OVER_F_H = 1.0  # f_c/f_h
DEFAULT_T = 1.7  # 980/576
DEFAULT_T_DEAD_OVER_T_COLD_IN = 0.52  # 300/576
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 7.2
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.03
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_GAMMA = 1.4
TARGET_EPS = 0.6
NTU_MATCH = 1.479  # From Helicopter case in xflow.py
DEFAULT_NTU_MAX = 2.0
SHOW_CUBIC = True

# For inlet_density assumption
DEFAULT_MOLAR_MASS_RATIO = 1.0  # M_cold / M_hot (cold/hot) - default when not specified
DEFAULT_A_R = 1.0  # A_r (cold/hot) - default when not specified

DEFAULT_DP_MAX = 0.2

# Single g^2 value for plotting
PLOT_TRIPLE_G2 = None


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="newfig4",
    t=DEFAULT_T,
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
):
    """
    Save figures as SVG, TIFF, and HD PNG combining agnostic and practical frameworks.
    Two subplots side by side: left shows agnostic (epsilon + pressure drops), right shows practical.
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

    # Set SHOW_CUBIC and NTU_MATCH in xflow module before calling create_plot
    xflow.SHOW_CUBIC = SHOW_CUBIC
    xflow.NTU_MATCH = NTU_MATCH

    # Calculate pressure drop ratio based on assumption
    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
    )

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9 / 2.54, 7 / 2.54))
    ax1_twin = ax1.twinx()  # Will be used but both pressure drops go on left axis

    # Left subplot: Agnostic framework
    # Plot epsilon and both pressure drops on left y-axis
    create_plot(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=2.0,  # NTU 0-2 for left subplot
        dp_max=dp_max,
        ax=ax1,
        ax_twin=ax1_twin,
        plot_triple_g2=PLOT_TRIPLE_G2,
        framework="agnostic",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )
    ax1.set_ylabel("")  # Remove the ylabel
    ax1.set_title(r"$\varepsilon$, $\Delta p / p_\mathrm{in}$")  # Set new title

    # Remove twin axis visibility and move pressure drops to main axis
    # Get the pressure drop lines from twin axis and replot on main axis
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()

    # Remove existing legends
    legend1 = ax1.get_legend()
    if legend1:
        legend1.remove()
    legend1_twin = ax1_twin.get_legend()
    if legend1_twin:
        legend1_twin.remove()

    # Replot pressure drops on main axis (if they exist)
    # We need to get the data from the lines
    for handle in handles2:
        x_data = handle.get_xdata()
        y_data = handle.get_ydata()
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        if np.any(valid_mask):
            label = handle.get_label()
            color = handle.get_color()
            linestyle = handle.get_linestyle()
            ax1.plot(x_data[valid_mask], y_data[valid_mask], color=color, linestyle=linestyle, label=label, zorder=1)

    # Hide twin axis
    ax1_twin.set_visible(False)

    # Set x-axis limits to 0-2
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 0.6)
    ax1.set_xlabel("NTU [-]")

    # Add grey vertical line at NTU_MATCH
    ax1.axvline(x=NTU_MATCH, color="grey", linestyle="--", linewidth=1, zorder=2)

    # Right subplot: Practical framework
    ax2_twin = ax2.twinx()  # Created but will be hidden
    create_plot(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=2.0,  # NTU 0-2 for right subplot
        dp_max=dp_max,
        ax=ax2,
        ax_twin=ax2_twin,
        plot_triple_g2=PLOT_TRIPLE_G2,
        framework="practical",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )

    # Move y-axis to right side
    # Get ylabel and ylim from left axis (where create_plot set them)
    ylim = ax2.get_ylim()
    # Remove the current title from ax2
    ax2.set_title(r"HEx $\Delta Q_0^M/Q_{\mathrm{max}}$")
    # Remove the current legend from ax2, if it exists
    legend2 = ax2.get_legend()
    if legend2 is not None:
        legend2.remove()

    # Reuse ax2_twin (which was hidden by create_plot) as the right axis
    ax2_twin.set_visible(True)
    ax2_twin.set_ylim(ylim)
    ax2_twin.spines["right"].set_visible(True)
    ax2_twin.yaxis.set_visible(True)
    ax2_twin.tick_params(axis="y", right=True, labelright=True)

    # Hide left y-axis completely
    ax2.yaxis.set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_ylabel("")  # Clear ylabel from left axis

    # Set x-axis limits to 0-2
    ax2.set_xlim(0, 2)
    ax2.set_xlabel("NTU [-]")

    lines = [line for line in ax2.get_lines()]

    # Add grey vertical line at NTU_MATCH
    ax2.axvline(x=NTU_MATCH, color="grey", linestyle="--", linewidth=1, zorder=2)
    # Scatter an x at the point (NTU_MATCH, y_closest) on the plotted line in ax2
    # Find the first Line2D artist in ax2 that is not a vline or hline

    if lines:
        line = lines[0]
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        # Find index where x data is closest to NTU_MATCH
        idx = (np.abs(xdata - NTU_MATCH)).argmin()
        x_closest = xdata[idx]
        y_closest = ydata[idx]
        ax2.scatter(x_closest, y_closest, marker="x", color="grey", zorder=10, s=60)

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
    # Save figures with Helicopter defaults
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="newfig4",
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
    )
