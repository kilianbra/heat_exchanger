import os

import matplotlib.pyplot as plt
import numpy as np
import xflow
from xflow import calculate_pressure_drop_ratio, create_plot

from plot_colors import MARKER_SIZE_LATEX
from fig_paths import JOURNAL_PLOTS, ensure_fig_dirs

ensure_fig_dirs()
save_dir = JOURNAL_PLOTS

# A = A_ref when NTU = NTU_MATCH; A/A_ref = NTU/NTU_MATCH
NTU_MATCH = 1.479

# Default modeling assumptions (match fig8/9 + fig7a)
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_C_COLD_OVER_C_HOT = 1.0
DEFAULT_ST_OVER_F = 0.4
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_D_R = 0.25
DEFAULT_GAMMA = 1.4
DEFAULT_MACH_IN = 0.1362  # Mh_in from xflow Helicopte_retrofit (f_c/f_h=1)
DEFAULT_G2_H = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2
DEFAULT_DP_MAX = 0.2
DEFAULT_T = 907 / 588
DEFAULT_T_DEAD_OVER_T_COLD_IN = 288 / 588
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.064
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.064
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 0.92


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig6a_lengthening",
    plot_area_ratio_ref=False,
    plot_ref_plus=False,
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
    Epsilon (left) + hot/cold pressure drop (right), styled like fig7a.
    Single baseline Mach; both Δp sides via inlet_density ratio.
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

    xflow.SHOW_CUBIC = False
    xflow.NTU_MATCH = NTU_MATCH

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

    fig = plt.figure(figsize=(7.1 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)
    ax_twin = ax.twinx()

    line_eps, _, ax, ax_twin = create_plot(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=15.0,
        dp_max=dp_max,
        ax=ax,
        ax_twin=ax_twin,
        plot_triple_g2=None,
        framework="conventional",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )
    ax.set_title("")

    legend = ax.get_legend()
    if legend:
        legend.remove()
    legend_twin = ax_twin.get_legend()
    if legend_twin:
        legend_twin.remove()

    # Left axis: epsilon in red (match fig7a)
    line_eps.set_color("r")
    ax.spines["left"].set_color("r")
    ax.yaxis.label.set_color("r")
    ax.tick_params(axis="y", colors="r")

    # Pressure drop: both black; hot dotted, cold dashed (match fig7a)
    lines_dp = ax_twin.get_lines()
    if len(lines_dp) >= 2:
        lines_dp[0].set_color("k")
        lines_dp[0].set_linestyle(":")
        lines_dp[1].set_color("k")
        lines_dp[1].set_linestyle("--")
        ax.legend(
            handles=lines_dp,
            labels=["Hot side", "Cold side"],
            loc="center right",
            bbox_to_anchor=(1, 0.62),
            frameon=True,
            facecolor="white",
            edgecolor="black",
            fancybox=False,
        )

    ax.set_ylabel(r"Heat Transfer Effectiveness ($\varepsilon$ [%])")
    ax_twin.set_ylabel(r"Pressure Drop ($\Delta p/p_{\mathrm{in}}$ [%])")

    if plot_area_ratio_ref:
        base_name = base_name.replace("_lengthening", "_A_A_ref_w_plus" if plot_ref_plus else "_A_A_ref")
        for ax_use in (ax, ax_twin):
            for line in ax_use.get_lines():
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
        x_ref = 1.0
    else:
        if plot_ref_plus:
            base_name = base_name.replace("_lengthening", "_NTU_w_plus")
        ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
        ax.set_xticks([0, 5, 10, 15])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}"))
        x_ref = NTU_MATCH

    if plot_ref_plus and line_eps is not None:
        x_eps, y_eps = np.array(line_eps.get_xdata()), np.array(line_eps.get_ydata())
        if np.min(x_eps) <= x_ref <= np.max(x_eps):
            y_at_ref = float(np.interp(x_ref, x_eps, y_eps))
            ax.scatter(x_ref, y_at_ref, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=5)
            ax.annotate(
                "baseline\n design",
                xy=(x_ref, y_at_ref),
                xytext=(0.08, 0.88) if not plot_area_ratio_ref else (0.08, 0.88),
                fontsize=font_size,
                ha="left",
                zorder=6,
                arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkB=12),
            )
        for line in lines_dp:
            x_dp, y_dp = np.array(line.get_xdata()), np.array(line.get_ydata())
            if len(x_dp) and np.min(x_dp) <= x_ref <= np.max(x_dp):
                y_dp_at_ref = float(np.interp(x_ref, x_dp, y_dp))
                ax_twin.scatter(
                    x_ref, y_dp_at_ref, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=5
                )

    plt.tight_layout(pad=0.5)

    for ext in ("svg", "tiff", "png", "pdf"):
        fig.savefig(
            os.path.join(save_dir, f"{base_name}.{ext}"),
            dpi=300,
            facecolor="white",
            format=ext,
            bbox_inches=None,
            pad_inches=0,
        )

    plt.close(fig)
    print(f"Saved figures: {base_name}.svg, {base_name}.tiff, {base_name}.png, {base_name}.pdf")


if __name__ == "__main__":
    PLOT_AREA_RATIO_REF = False
    PLOT_REF_PLUS = True
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="fig6a_lengthening",
        plot_area_ratio_ref=PLOT_AREA_RATIO_REF,
        plot_ref_plus=PLOT_REF_PLUS,
    )
