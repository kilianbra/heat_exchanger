import os

import matplotlib.pyplot as plt
import numpy as np
import xflow
from xflow import calculate_pressure_drop_ratio, plot_unavailable_energy_breakdown

save_dir = os.path.dirname(os.path.abspath(__file__))

# Helicopter defaults (hardcoded from xflow.py) - same as fig5c
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


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig6b",
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
    Save figures as SVG, TIFF, and HD PNG showing practical unavailable energy breakdown.
    Matches fig5c style (labels, markers, legend box, no title).
    """
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

    xflow.SHOW_CUBIC = SHOW_CUBIC
    xflow.NTU_MATCH = NTU_MATCH

    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
    )

    # Match fig5c figure size (triple column 6 cm)
    fig = plt.figure(figsize=(6 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)

    line_no_dp, line_with_dp, ax = plot_unavailable_energy_breakdown(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=DEFAULT_NTU_MAX,
        dp_max=dp_max,
        ax=ax,
        framework="practical",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )

    ntu_no_dp = line_no_dp.get_xdata()
    y_no_dp = line_no_dp.get_ydata()
    ntu_with_dp = line_with_dp.get_xdata()
    y_with_dp = line_with_dp.get_ydata()

    ntu_min_valid = ntu_with_dp.min()
    ntu_max_valid = ntu_with_dp.max()

    mask_thermal = (ntu_no_dp >= ntu_min_valid) & (ntu_no_dp <= ntu_max_valid)
    ntu_no_dp_masked = ntu_no_dp[mask_thermal]
    y_no_dp_masked = y_no_dp[mask_thermal]

    line_no_dp.remove()
    line_no_dp = ax.plot(
        ntu_no_dp_masked,
        y_no_dp_masked,
        "k-",
        label="_nolegend_",
        zorder=2,
    )[0]

    line_with_dp.set_label("_nolegend_")

    ax.fill_between(ntu_no_dp_masked, y_no_dp_masked, 0, alpha=0.3, color="gray", zorder=1)

    ntu_common = np.linspace(ntu_min_valid, ntu_max_valid, 200)
    y_no_dp_interp = np.interp(ntu_common, ntu_no_dp_masked, y_no_dp_masked)
    y_with_dp_interp = np.interp(ntu_common, ntu_with_dp, y_with_dp)

    ax.fill_between(
        ntu_common,
        y_with_dp_interp,
        y_no_dp_interp,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
        hatch="///",
        zorder=1,
        label="_nolegend_",
    )

    # Optimal marker: circle, black, s=50, no double edge (match fig5c)
    idx_optimum = np.argmin(y_with_dp)
    x_optimum = ntu_with_dp[idx_optimum]
    y_optimum = y_with_dp[idx_optimum]
    ax.scatter(x_optimum, y_optimum, marker="o", facecolor="black", edgecolor="white", zorder=5, s=50)

    print(f"Optimum NTU: {x_optimum:.4f}")

    # Reference marker: diamond (D), black face, white edge, s=50 (match fig5c)
    idx_pressure = np.argmin(np.abs(ntu_with_dp - NTU_MATCH))
    x_pressure = ntu_with_dp[idx_pressure]
    y_pressure = y_with_dp[idx_pressure]
    ax.scatter(x_pressure, y_pressure, marker="D", facecolor="black", edgecolor="white", zorder=5, s=50)

    ax.set_title("")
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.3, 0)
    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax.set_ylabel(r"Change in Unavailable Energy ($\Delta \dot{Q}_0^\mathrm{M}/\dot{Q}_{\mathrm{max}}$)")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

    # No legend
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    plt.tight_layout(pad=0.5)

    fig.savefig(os.path.join(save_dir, f"{base_name}.svg"), dpi=300, facecolor="white", format="svg", bbox_inches=None, pad_inches=0)
    fig.savefig(os.path.join(save_dir, f"{base_name}.tiff"), dpi=300, facecolor="white", format="tiff", bbox_inches=None, pad_inches=0)
    fig.savefig(os.path.join(save_dir, f"{base_name}.png"), dpi=300, facecolor="white", format="png", bbox_inches=None, pad_inches=0)
    fig.savefig(os.path.join(save_dir, f"{base_name}.pdf"), dpi=300, facecolor="white", format="pdf", bbox_inches=None, pad_inches=0)

    plt.close(fig)
    print(f"Saved figures: {base_name}.svg, {base_name}.tiff, {base_name}.png, {base_name}.pdf")

    return x_optimum


if __name__ == "__main__":
    optimum_ntu = save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="fig6b",
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
    )
    print(f"\nUse this NTU_OPTIMUM value in fig6a: {optimum_ntu:.4f}")
