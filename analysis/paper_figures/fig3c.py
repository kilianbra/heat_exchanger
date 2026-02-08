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

DEFAULT_DP_MAX = 0.3

# Single g^2 value for plotting
PLOT_TRIPLE_G2 = None


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig3c",
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
    Save figures as SVG, TIFF, and HD PNG for practical framework (right subplot from newfig4).
    Shows HEx ΔQ_0^M/Q_max.
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

    # Set SHOW_CUBIC and NTU_MATCH in xflow module before calling create_plot
    xflow.SHOW_CUBIC = SHOW_CUBIC
    xflow.NTU_MATCH = NTU_MATCH

    # Calculate pressure drop ratio based on assumption
    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
    )

    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(9 / 2.54, 7 / 2.54))
    ax_twin = ax.twinx()  # Created but will be hidden

    create_plot(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=2.0,  # NTU 0-2
        dp_max=dp_max,
        ax=ax,
        ax_twin=ax_twin,
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
    ylim = ax.get_ylim()
    # Remove the current title from ax
    ax.set_title("")
    # Remove the current legend from ax, if it exists
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    # Reuse ax_twin (which was hidden by create_plot) as the right axis
    ax_twin.set_visible(True)
    ax_twin.set_ylim(ylim)
    ax_twin.spines["right"].set_visible(True)
    ax_twin.yaxis.set_visible(True)
    ax_twin.tick_params(axis="y", right=True, labelright=True)

    # Hide left y-axis completely
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("")  # Clear ylabel from left axis

    # Set x-axis limits to 0-2
    ax.set_xlim(0, 2)
    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax_twin.set_ylabel(r"Change in Unavailable Energy ($\Delta \dot{Q}_0^\mathrm{M}/\dot{Q}_{\mathrm{max}}$)")

    # NTU reference and optimum values
    NTU_REF = 1.48
    NTU_OPT = 1.18

    # Get all lines from ax and add markers
    lines = [line for line in ax.get_lines() if hasattr(line, 'get_xdata')]

    # Add grey crosses at NTU_REF and black filled circles at NTU_OPT on all lines
    for line in lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        if np.any(valid_mask):
            x_plot = x_data[valid_mask]
            y_plot = y_data[valid_mask]
            
            # Find y-value at NTU_REF
            idx_ref = (np.abs(x_plot - NTU_REF)).argmin()
            y_ref = y_plot[idx_ref]
            ax.scatter(NTU_REF, y_ref, marker="x", color="grey", zorder=10, s=60)
            
            # Find y-value at NTU_OPT
            idx_opt = (np.abs(x_plot - NTU_OPT)).argmin()
            y_opt = y_plot[idx_opt]
            ax.scatter(NTU_OPT, y_opt, marker="o", color="black", zorder=10, s=60, facecolor="black")

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
        base_name="fig3c",
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
    )
