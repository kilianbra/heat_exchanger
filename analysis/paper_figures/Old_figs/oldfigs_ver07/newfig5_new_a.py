import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xflow_ver07
from xflow_ver07 import calculate_pressure_drop_ratio, plot_unavailable_energy_breakdown

save_dir = os.path.dirname(os.path.abspath(__file__))

# Helicopter defaults (hardcoded from xflow.py) - same as newfig4
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

# Optimum NTU from practical framework (set after running newfig5_new_b)
NTU_OPTIMUM = 1.2075  # Will be set based on output from newfig5_new_b


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="newfig5_new_a",
    t=DEFAULT_T,
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
    ntu_optimum=None,
):
    """
    Save figures as SVG, TIFF, and HD PNG showing classical unavailable energy breakdown.
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

    # Set SHOW_CUBIC and NTU_MATCH in xflow module
    xflow_ver07.SHOW_CUBIC = SHOW_CUBIC
    xflow_ver07.NTU_MATCH = NTU_MATCH

    # Calculate pressure drop ratio based on assumption
    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
    )

    # Create figure
    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)

    # Plot unavailable energy breakdown for classical framework
    line_no_dp, line_with_dp, ax = plot_unavailable_energy_breakdown(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=DEFAULT_NTU_MAX,
        dp_max=dp_max,
        ax=ax,
        framework="classical",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )

    # Get data from the lines
    ntu_no_dp = line_no_dp.get_xdata()
    y_no_dp = line_no_dp.get_ydata()
    ntu_with_dp = line_with_dp.get_xdata()
    y_with_dp = line_with_dp.get_ydata()

    # Mask thermal dissipation line to only show where pressure drop line exists
    # Find the valid range from the pressure drop line
    ntu_min_valid = ntu_with_dp.min()
    ntu_max_valid = ntu_with_dp.max()

    # Create mask for thermal dissipation line
    mask_thermal = (ntu_no_dp >= ntu_min_valid) & (ntu_no_dp <= ntu_max_valid)
    ntu_no_dp_masked = ntu_no_dp[mask_thermal]
    y_no_dp_masked = y_no_dp[mask_thermal]

    # Remove the original line and replot with masked data
    line_no_dp.remove()
    line_no_dp = ax.plot(
        ntu_no_dp_masked,
        y_no_dp_masked,
        "k-",
        label="_nolegend_",  # Hide from legend, we'll add custom handles
        zorder=2,
    )[0]

    # Hide the "With pressure drop" line from legend
    line_with_dp.set_label("_nolegend_")

    # Shade area between x-axis (y=0) and thermal dissipation line (using masked data)
    # Grey fill for thermal dissipation (classical)
    ax.fill_between(ntu_no_dp_masked, 0, y_no_dp_masked, alpha=0.3, color="gray", zorder=1)

    # Find overlapping x-range for shading between the two lines
    # Use the masked range
    ntu_common = np.linspace(ntu_min_valid, ntu_max_valid, 200)

    # Interpolate both curves to common x values
    y_no_dp_interp = np.interp(ntu_common, ntu_no_dp_masked, y_no_dp_masked)
    y_with_dp_interp = np.interp(ntu_common, ntu_with_dp, y_with_dp)

    # Shade area between the two lines with hatching (viscous dissipation)
    # Transparent background with black hatching
    ax.fill_between(
        ntu_common,
        y_no_dp_interp,
        y_with_dp_interp,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
        hatch="///",
        zorder=1,
        label="_nolegend_",  # Hide from legend, we'll add custom handles
    )

    # Add 'x' marker at NTU_MATCH on pressure drop line (total)
    # Find closest point to NTU_MATCH on pressure drop line
    idx_pressure = np.argmin(np.abs(ntu_with_dp - NTU_MATCH))
    x_pressure = ntu_with_dp[idx_pressure]
    y_pressure = y_with_dp[idx_pressure]
    ax.scatter(x_pressure, y_pressure, marker="x", color="grey", s=60, zorder=5)

    # Add circle marker at optimum NTU if provided
    if ntu_optimum is not None:
        # Find closest point to optimum NTU on pressure drop line
        idx_opt = np.argmin(np.abs(ntu_with_dp - ntu_optimum))
        x_opt = ntu_with_dp[idx_opt]
        y_opt = y_with_dp[idx_opt]
        ax.scatter(x_opt, y_opt, marker="o", color="black", s=60, zorder=5)

    # Remove title if present
    ax.set_title("Classical Availability")

    # Set x-axis limits to 0-2 (matching newfig4 scaling)
    ax.set_xlim(0, 2)
    ax.set_xlabel("NTU [-]")

    # Create custom legend with boxes instead of lines
    # Solid grey box for thermal dissipation
    patch_thermal = mpatches.Patch(facecolor="gray", alpha=0.3, edgecolor="black", label="Thermal dissipation")
    # Transparent box with black hatching for viscous dissipation
    patch_viscous = mpatches.Patch(
        facecolor="none", edgecolor="black", hatch="///", linewidth=1.5, label="Viscous dissipation"
    )

    # Update legend with custom patches
    ax.legend(
        handles=[patch_thermal, patch_viscous],
        loc="upper left",
        labelspacing=0.05,
        edgecolor="black",
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        fancybox=True,
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
    # Save figures with Helicopter defaults
    # Use NTU_OPTIMUM if set, otherwise use None (no circle will be plotted)
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="newfig5_new_a",
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
        ntu_optimum=NTU_OPTIMUM,
    )
