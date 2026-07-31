"""
Figure 6 variants: Mach = 0.1362 only, fig6a–c axis bounds, thermal/viscous breakdown on b and c (fig7 style).

Outputs (SVG, TIFF, PNG, PDF) in Figs_current/explore_ideas/:
  fig6a_oneMach_th_n_visc
  fig6b_oneMach_th_n_visc
  fig6c_oneMach_th_n_visc
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import xflow
from xflow import calculate_epsilon_ntu_curve, calculate_pressure_drop_ratio, plot_unavailable_energy_breakdown

from plot_colors import COLOR_THERMAL, COLOR_VISC_HOT, MARKER_SIZE_LATEX

from fig_paths import EXPLORE_IDEAS, JOURNAL_PLOTS, ensure_fig_dirs

ensure_fig_dirs()
save_dir = JOURNAL_PLOTS

# Defaults (match fig6/fig8/9)
DEFAULT_C_COLD_OVER_C_HOT = 1.0
DEFAULT_ST_OVER_F = 0.4
# DEFAULT_F_C_OVER_F_H = 0.25  # old: compensated missing f in inlet_density ratio
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_D_R = 0.25
DEFAULT_GAMMA = 1.4
# DEFAULT_MACH_IN = 0.11  # Mh_in (old, with f_c/f_h=0.25)
DEFAULT_MACH_IN = 0.1362  # Mh_in from xflow Helicopte_retrofit (f_c/f_h=1)
DEFAULT_G2_H = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2
DEFAULT_DP_MAX = 0.2
DEFAULT_NTU_MAX = 5.0

DEFAULT_T = 907 / 588
DEFAULT_T_DEAD_OVER_T_COLD_IN = 288 / 588
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.064
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.064
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 0.92

NTU_MATCH = 1.479

FONT_SIZE = 8


def _setup_rcparams():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": FONT_SIZE,
            "mathtext.fontset": "stix",
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "hatch.linewidth": 0.5,
        }
    )


def _pressure_drop_ratio(c_cold_over_c_hot, t, d_r, p_cold_in_over_p_hot_in, a_r, f_c_over_f_h=DEFAULT_F_C_OVER_F_H):
    sigma_r = d_r * a_r if a_r is not None else None
    return calculate_pressure_drop_ratio(
        DEFAULT_PRESSURE_DROP_ASSUMPTION,
        c_cold_over_c_hot,
        t,
        d_r,
        DEFAULT_MOLAR_MASS_RATIO,
        sigma_r,
        p_cold_in_over_p_hot_in,
        f_c_over_f_h=f_c_over_f_h,
    )


def _set_ntu_axis(ax):
    ax.set_xlim(0, DEFAULT_NTU_MAX)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}"))
    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")


def _save_figure(fig, base_name):  # ("tiff", "tiff"),
    for ext, fmt in [("svg", "svg"), ("png", "png"), ("pdf", "pdf")]:
        fig.savefig(
            os.path.join(save_dir, f"{base_name}.{ext}"),
            dpi=300,
            facecolor="white",
            format=fmt,
            bbox_inches=None,
            pad_inches=0,
        )
    plt.close(fig)
    print(f"Saved figures: {base_name}.svg, {base_name}.png, {base_name}.pdf")


def save_fig6a(
    c_cold_over_c_hot=DEFAULT_C_COLD_OVER_C_HOT,
    st_over_f=DEFAULT_ST_OVER_F,
    f_c_over_f_h=DEFAULT_F_C_OVER_F_H,
    d_r=DEFAULT_D_R,
    g2_h=DEFAULT_G2_H,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig6a_oneMach_th_n_visc",
):
    """Conventional: epsilon (red) to NTU max; hot dp (black) stops at dp_max (match fig6a triple-Mach colors)."""
    _setup_rcparams()
    xflow.SHOW_CUBIC = False

    fig = plt.figure(figsize=(7.1 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)
    ax_twin = ax.twinx()

    # Epsilon: full NTU sweep (no dp cutoff on left axis)
    ntu_eps, epsilon, _, _, _ = calculate_epsilon_ntu_curve(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=DEFAULT_NTU_MAX,
        dp_max=1.0,
        pressure_drop_percent_ratio_cold_over_hot=0.0,
    )
    # Pressure drop: stop when hot side exceeds dp_max
    ntu_dp, _, dp_hot, _, mask_dp = calculate_epsilon_ntu_curve(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=DEFAULT_NTU_MAX,
        dp_max=dp_max,
        pressure_drop_percent_ratio_cold_over_hot=0.0,
    )

    line_eps = ax.plot(ntu_eps, epsilon, "-", color="r", zorder=3)[0]
    line_dp = ax_twin.plot(ntu_dp[mask_dp], dp_hot[mask_dp], "k-", zorder=1)[0]

    ax.set_title("")
    _set_ntu_axis(ax)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel(r"Heat Transfer Effectiveness ($\varepsilon$ [%])")
    ax_twin.set_ylim(0, dp_max)
    ax_twin.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_twin.set_ylabel(r"Hot Pressure Drop ($\Delta p/p_{\mathrm{in}}$ [%])")

    # Baseline markers at NTU_MATCH
    x_ref = NTU_MATCH
    x_eps, y_eps = line_eps.get_xdata(), line_eps.get_ydata()
    if np.min(x_eps) <= x_ref <= np.max(x_eps):
        y_at_ref = np.interp(x_ref, x_eps, y_eps)
        ax.scatter(x_ref, y_at_ref, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=5)
    x_dp, y_dp = line_dp.get_xdata(), line_dp.get_ydata()
    if len(x_dp) > 0 and np.min(x_dp) <= x_ref <= np.max(x_dp):
        y_dp_at_ref = np.interp(x_ref, x_dp, y_dp)
        ax_twin.scatter(x_ref, y_dp_at_ref, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=5)

    plt.tight_layout(pad=0.5)
    _save_figure(fig, base_name)


def _save_availability_breakdown(
    framework,
    ylim,
    ylabel,
    base_name,
    *,
    c_cold_over_c_hot=DEFAULT_C_COLD_OVER_C_HOT,
    st_over_f=DEFAULT_ST_OVER_F,
    f_c_over_f_h=DEFAULT_F_C_OVER_F_H,
    d_r=DEFAULT_D_R,
    g2_h=DEFAULT_G2_H,
    dp_max=DEFAULT_DP_MAX,
    t=DEFAULT_T,
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
    a_r=DEFAULT_A_R,
    show_increasing_length=False,
    annotate_optimum=False,
):
    """Classical or practical availability with thermal/viscous stacked regions (fig7 style)."""
    _setup_rcparams()
    xflow.SHOW_CUBIC = False

    pressure_drop_ratio = _pressure_drop_ratio(
        c_cold_over_c_hot, t, d_r, p_cold_in_over_p_hot_in, a_r, f_c_over_f_h=f_c_over_f_h
    )

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
        framework=framework,
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=DEFAULT_GAMMA,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )

    ntu_no_dp = line_no_dp.get_xdata()
    y_no_dp = -np.array(line_no_dp.get_ydata())
    ntu_with_dp = line_with_dp.get_xdata()
    y_with_dp = -np.array(line_with_dp.get_ydata())

    ntu_min_valid = ntu_with_dp.min()
    ntu_max_valid = ntu_with_dp.max()
    mask_thermal = (ntu_no_dp >= ntu_min_valid) & (ntu_no_dp <= ntu_max_valid)
    ntu_no_dp_masked = ntu_no_dp[mask_thermal]
    y_no_dp_masked = y_no_dp[mask_thermal]

    line_no_dp.remove()
    line_with_dp.remove()

    ntu_common = np.linspace(ntu_min_valid, ntu_max_valid, 400)
    y_no_dp_interp = np.interp(ntu_common, ntu_no_dp_masked, y_no_dp_masked)
    y_with_dp_interp = np.interp(ntu_common, ntu_with_dp, y_with_dp)
    y_viscous_interp = y_with_dp_interp - y_no_dp_interp

    # Match fig7b (classical): viscous below, thermal on top.
    # Match fig7c (practical): thermal below, viscous on top for visibility.
    edge_lw = 0.5
    if framework == "practical":
        ax.fill_between(
            ntu_common,
            y_viscous_interp,
            y_with_dp_interp,
            facecolor=COLOR_THERMAL,
            edgecolor="black",
            linewidth=edge_lw,
            hatch="///",
            zorder=0,
        )
        ax.fill_between(
            ntu_common,
            0,
            y_viscous_interp,
            facecolor=COLOR_VISC_HOT,
            edgecolor="black",
            linewidth=edge_lw,
            zorder=1,
            label="_nolegend_",
        )
    else:
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

    ntu_with_dp_orig = np.array(ntu_with_dp)

    # Baseline marker
    idx_ref = np.argmin(np.abs(ntu_with_dp_orig - NTU_MATCH))
    x_ref = ntu_with_dp_orig[idx_ref]
    y_ref = y_with_dp[idx_ref]
    ax.scatter(
        x_ref,
        y_ref,
        marker="+",
        s=MARKER_SIZE_LATEX,
        linewidths=1,
        color="black",
        zorder=5,
    )
    if framework == "classical":
        ax.annotate(
            "baseline design",
            xy=(x_ref, y_ref),
            xytext=(x_ref, y_ref + 0.15),
            fontsize=FONT_SIZE,
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkB=8),
        )

    # Optimum marker (practical only; classical omits circle)
    if False:  # framework == "practical":
        idx_opt = np.argmax(y_with_dp)
        x_opt = ntu_with_dp_orig[idx_opt]
        y_opt = y_with_dp[idx_opt]
        ax.scatter(
            x_opt,
            y_opt,
            marker="o",
            facecolor="black",
            edgecolor="white",
            zorder=5,
            s=MARKER_SIZE_LATEX,
        )
        if annotate_optimum:
            ax.annotate(
                "optimal design",
                xy=(x_opt, y_opt),
                xytext=(3, 0.25),
                fontsize=FONT_SIZE,
                ha="left",
                arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkB=10),
            )

    ax.set_title("")
    _set_ntu_axis(ax)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(round(x * 100))}%"))
    ax.set_ylabel(ylabel)

    patch_thermal = mpatches.Patch(
        facecolor=COLOR_THERMAL, edgecolor="black", hatch="///", linewidth=0.5, label="Thermal"
    )
    patch_viscous = mpatches.Patch(facecolor=COLOR_VISC_HOT, edgecolor="black", linewidth=0.5, label="Viscous")
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

    if show_increasing_length:
        ax.text(
            0.5,
            0.98,
            "increasing length",
            transform=ax.transAxes,
            va="top",
            ha="center",
            fontsize=FONT_SIZE,
        )
        ax.annotate(
            "",
            xy=(0.7, 0.92),
            xytext=(0.3, 0.92),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=0.75),
        )

    plt.tight_layout(pad=0.5)
    ax.yaxis.labelpad = -4
    fig.subplots_adjust(left=0.25)
    _save_figure(fig, base_name)


def save_fig6b(base_name="fig6b_oneMach_th_n_visc"):
    _save_availability_breakdown(
        framework="classical",
        ylim=(-0.2, 0.3),
        ylabel=r"Change in Availability ($\sum_{i} \; \Delta \dot{W}_{\mathrm{A},i}/\dot{Q}_{\mathrm{max}}$ [%])",
        base_name=base_name,
        show_increasing_length=True,
    )


def save_fig6c(base_name="fig6c_oneMach_th_n_visc"):
    _save_availability_breakdown(
        framework="practical",
        ylim=(-0.2, 0.3),
        ylabel=(
            r"Change in Availability ($\sum_{i} \; \Delta \dot{W}^{\mathrm{M}}_{\mathrm{A},i}/\dot{Q}_{\mathrm{max}}$ [%])"
        ),
        base_name=base_name,
        annotate_optimum=True,
    )


if __name__ == "__main__":
    save_fig6a()
    save_fig6b()
    save_fig6c()
