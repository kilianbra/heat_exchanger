"""
fig2b_visc: Same structure as fig2b but with g2=5e-3 only, plotting thermal and viscous
dissipation breakdown in the style of fig5bvisc.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xflow
from scipy.signal import find_peaks
from xflow import calculate_pressure_drop_ratio, plot_unavailable_energy_breakdown

save_dir = os.path.dirname(os.path.abspath(__file__))

# fig2b defaults (classical framework, generic case)
DEFAULT_C_COLD_OVER_C_HOT = 1.0
DEFAULT_ST_OVER_F = 0.4
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_D_R = 1.0
G2_H = 5e-3  # Single value as requested (was one of [1e-3, 5e-3, 20e-3] in fig2b)
DEFAULT_DP_MAX = 0.3

DEFAULT_T = 2.0
DEFAULT_T_DEAD_OVER_T_COLD_IN = 1.1
DEFAULT_GAMMA = 1.4

DEFAULT_PRESSURE_DROP_ASSUMPTION = "dp_c=dp_h"
DEFAULT_NTU_MAX = 15.0

if DEFAULT_PRESSURE_DROP_ASSUMPTION == "inlet_density":
    DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 10.0
    DEFAULT_MOLAR_MASS_RATIO = 1.0
    DEFAULT_A_R = 0.1
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
    base_name="fig2b_visc",
    t=DEFAULT_T,
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=1.0,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
    ntu_max=DEFAULT_NTU_MAX,
):
    """Save figures with thermal/viscous breakdown, matching fig5bvisc plotting style."""
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

    # fig2b uses SHOW_CUBIC = False
    xflow.SHOW_CUBIC = False

    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption,
        c_cold_over_c_hot,
        t,
        d_r,
        molar_mass_ratio,
        sigma_r,
        p_cold_in_over_p_hot_in,
    )

    # Match fig2b figure size (triple column)
    fig = plt.figure(figsize=(6 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)

    line_no_dp, line_with_dp, ax = plot_unavailable_energy_breakdown(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=ntu_max,
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

    # fig5bvisc style: gray fill for thermal
    ax.fill_between(ntu_no_dp_masked, 0, y_no_dp_masked, alpha=0.3, color="gray", zorder=1)

    # fig5bvisc style: hatched fill for viscous
    ntu_common = np.linspace(ntu_min_valid, ntu_max_valid, 200)
    y_no_dp_interp = np.interp(ntu_common, ntu_no_dp_masked, y_no_dp_masked)
    y_with_dp_interp = np.interp(ntu_common, ntu_with_dp, y_with_dp)

    ax.fill_between(
        ntu_common,
        y_no_dp_interp,
        y_with_dp_interp,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
        hatch="///",
        zorder=1,
        label="_nolegend_",
    )

    # Optimal design: minimum after first peak (fig2b style)
    valid_mask = np.isfinite(ntu_with_dp) & np.isfinite(y_with_dp)
    x_plot = ntu_with_dp[valid_mask]
    y_plot = y_with_dp[valid_mask]
    peaks, _ = find_peaks(y_plot)
    if g2_h == 5e-3:
        peak_loc = (2, 0.22)
    else:
        peak_loc = (5, 0.12)
    if len(peaks) > 0:
        first_peak = peaks[0]
        y_local_min_idx = first_peak + np.argmin(y_plot[first_peak:])
        x_opt, y_opt = x_plot[y_local_min_idx], y_plot[y_local_min_idx]
        ax.scatter(x_opt, y_opt, marker="o", facecolor="black", edgecolor="white", zorder=5, s=50)
        arrow_kw = dict(arrowstyle="->", color="black", lw=1, shrinkB=10)
        ax.annotate(
            "optimal design",
            xy=(x_opt, y_opt),
            xytext=peak_loc,
            fontsize=font_size,
            ha="left",
            arrowprops=arrow_kw,
        )

    ax.set_title("")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 0.3)
    ax.set_xticks([0, 5, 10, 15])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}"))
    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax.set_ylabel(r"Change in Unavailable Energy ($\Delta \dot{Q}_0/\dot{Q}_{\mathrm{max}}$)")

    patch_thermal = mpatches.Patch(facecolor="gray", alpha=0.3, edgecolor="black", label="Thermal dissipation")
    patch_viscous = mpatches.Patch(
        facecolor="none", edgecolor="black", hatch="///", linewidth=1.5, label="Viscous dissipation"
    )

    ax.legend(
        handles=[patch_thermal, patch_viscous],
        loc="upper left",
        labelspacing=0.05,
        edgecolor="black",
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
    )

    plt.tight_layout(pad=0.5)

    fig.savefig(
        os.path.join(save_dir, f"{base_name}.svg"),
        dpi=300,
        facecolor="white",
        format="svg",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.tiff"),
        dpi=300,
        facecolor="white",
        format="tiff",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.png"),
        dpi=300,
        facecolor="white",
        format="png",
        bbox_inches=None,
        pad_inches=0,
    )
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
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="fig2b_visc",
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=1.0,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
        ntu_max=DEFAULT_NTU_MAX,
    )
