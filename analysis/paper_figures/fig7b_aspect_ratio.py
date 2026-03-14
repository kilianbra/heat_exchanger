import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import xflow
from xflow import calculate_pressure_drop_ratio, plot_unavailable_energy_breakdown

from plot_colors import COLOR_THERMAL, COLOR_VISC_HOT, MARKER_SIZE_LATEX

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figs_current")

# Defaults (match fig8/9, ref: eps=0.6, dp_h/pin=0.06, dpc/pcin=0.04)
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_C_COLD_OVER_C_HOT = 1.0
# DEFAULT_D_R = 0.257
DEFAULT_D_R = 0.25
DEFAULT_GAMMA = 1.4
# DEFAULT_MACH_IN = 0.1
DEFAULT_MACH_IN = 0.11  # Mh_in
DEFAULT_G2_H = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2
DEFAULT_ST_OVER_F = 0.4
# DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_F_C_OVER_F_H = 0.25
# DEFAULT_T = 898 / 588
DEFAULT_T = 907 / 588  # T_ratios from xflow 106-109
DEFAULT_T_DEAD_OVER_T_COLD_IN = 288 / 588
# DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.04
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.064
# DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.04
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.064
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_MOLAR_MASS_RATIO = 1.0
# DEFAULT_A_R = 1.0
DEFAULT_A_R = 0.92
# NTU_MATCH = 1.824
NTU_MATCH = 1.479
DEFAULT_NTU_MAX = 8.0
SHOW_CUBIC = True
DEFAULT_DP_MAX = 0.2

# Optimum NTU from practical framework (run fig7c first to get value)
# NTU_OPTIMUM = 1.6482  # From fig7c with fig8/9 defaults (previous)
NTU_OPTIMUM = 1.2513  # From fig7c with Mh_in=0.11, f_c/f_h=0.25


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig7b_aspect_ratio",
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
    Matches fig5b style (labels, markers, legend box, no title).
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
            "hatch.linewidth": 0.5,
        }
    )

    xflow.SHOW_CUBIC = SHOW_CUBIC
    xflow.NTU_MATCH = NTU_MATCH

    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
    )

    # Match fig5b figure size (triple column)
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
        framework="classical",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )

    ntu_no_dp = line_no_dp.get_xdata()
    y_no_dp = -np.array(line_no_dp.get_ydata())  # Negate for change in availability
    ntu_with_dp = line_with_dp.get_xdata()
    y_with_dp = -np.array(line_with_dp.get_ydata())  # Negate for change in availability

    ntu_min_valid = ntu_with_dp.min()
    ntu_max_valid = ntu_with_dp.max()

    mask_thermal = (ntu_no_dp >= ntu_min_valid) & (ntu_no_dp <= ntu_max_valid)
    ntu_no_dp_masked = ntu_no_dp[mask_thermal]
    y_no_dp_masked = y_no_dp[mask_thermal]

    line_no_dp.remove()
    line_with_dp.remove()

    ntu_common = np.linspace(ntu_min_valid, ntu_max_valid, 200)
    y_no_dp_interp = np.interp(ntu_common, ntu_no_dp_masked, y_no_dp_masked)
    y_with_dp_interp = np.interp(ntu_common, ntu_with_dp, y_with_dp)

    # Viscous alone = total - thermal = (viscous + thermal) - just_thermal
    y_viscous_interp = y_with_dp_interp - y_no_dp_interp

    # Stack: first viscous (baseline to viscous alone), then thermal (viscous to total)
    # Thin outlines (0.5) like bar chart; main line drawn on top
    edge_lw = 0.5
    ax.fill_between(
        ntu_common,
        0,
        y_viscous_interp,
        facecolor=COLOR_VISC_HOT,
        edgecolor="black",
        linewidth=edge_lw,
        zorder=0,
        label="_nolegend_",
    )
    ax.fill_between(
        ntu_common,
        y_viscous_interp,
        y_with_dp_interp,
        facecolor=COLOR_THERMAL,
        edgecolor="black",
        linewidth=edge_lw,
        hatch="///",
        zorder=1,
    )
    ax.plot(ntu_with_dp, y_with_dp, "k-", label="_nolegend_", zorder=3, linewidth=1.5)

    # Baseline marker: + (baseline design)
    idx_pressure = np.argmin(np.abs(ntu_with_dp - NTU_MATCH))
    x_pressure = ntu_with_dp[idx_pressure]
    y_pressure = y_with_dp[idx_pressure]
    ax.scatter(x_pressure, y_pressure, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=5)

    # Optimal marker: circle, black, s=50, white edge (match fig5b)
    if ntu_optimum is not None:
        idx_opt = np.argmin(np.abs(ntu_with_dp - ntu_optimum))
        x_opt = ntu_with_dp[idx_opt]
        y_opt = y_with_dp[idx_opt]
        ax.scatter(x_opt, y_opt, marker="o", facecolor="black", edgecolor="white", zorder=5, s=MARKER_SIZE_LATEX)

    # Annotations with arrows: baseline design, optimal design (text within ylim)
    arrow_kw_opt = dict(arrowstyle="->", color="black", lw=1, shrinkB=10)
    arrow_kw_ref = dict(arrowstyle="->", color="black", lw=1, shrinkB=10)
    ax.annotate(
        "baseline design",
        xy=(x_pressure, y_pressure),
        xytext=(0.15, -0.075),
        fontsize=font_size,
        ha="left",
        arrowprops=arrow_kw_ref,
    )
    if ntu_optimum is not None:
        ax.annotate(
            "optimal design",
            xy=(x_opt, y_opt),
            xytext=(0.10, -0.06),
            fontsize=font_size,
            ha="left",
            arrowprops=arrow_kw_opt,
        )

    ax.set_title("")
    ax.set_xlim(0, 2.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylim(-0.1, 0)
    ax.set_yticks([-0.1, -0.08, -0.06, -0.04, -0.02, 0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(round(x * 100))}%"))
    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax.set_ylabel(r"Change in Availability ($\sum_{i} \; \Delta \dot{W}_{\mathrm{A},i}/\dot{Q}_{\mathrm{max}}$ [%])")

    patch_thermal = mpatches.Patch(
        facecolor=COLOR_THERMAL, edgecolor="black", hatch="///", linewidth=0.5, label="Thermal dissipation"
    )
    patch_viscous = mpatches.Patch(
        facecolor=COLOR_VISC_HOT, edgecolor="black", linewidth=0.5, label="Viscous dissipation"
    )

    # Legend box style to match fig5b (frameon, facecolor white, edge black, fancybox=False)
    ax.legend(
        handles=[patch_thermal, patch_viscous],
        loc="lower left",
        labelspacing=0.05,
        edgecolor="black",
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
    )

    plt.tight_layout(pad=0.5)
    ax.yaxis.labelpad = -4  # After tight_layout: reduce distance between ticks and ylabel
    fig.subplots_adjust(left=0.25)

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
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="fig7b_aspect_ratio",
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
