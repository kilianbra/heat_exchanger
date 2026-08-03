import os

import matplotlib.pyplot as plt
import numpy as np
import xflow
from fig_paths import JOURNAL_PLOTS, ensure_fig_dirs
from matplotlib.ticker import MultipleLocator
from plot_colors import MARKER_SIZE_LATEX, TITLE_FONTSIZE
from xflow import calculate_pressure_drop_ratio, create_plot

ensure_fig_dirs()
save_dir = JOURNAL_PLOTS

# Defaults (match fig8/9)
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_C_COLD_OVER_C_HOT = 1.0
DEFAULT_D_R = 0.25  # 0.257
DEFAULT_GAMMA = 1.4
# DEFAULT_MACH_IN = 0.11  # Mh_in (old, with f_c/f_h=0.25)
DEFAULT_MACH_IN = 0.1362  # Mh_in from xflow Helicopte_retrofit (f_c/f_h=1)
DEFAULT_G2_H = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2
DEFAULT_ST_OVER_F = 0.4
# DEFAULT_F_C_OVER_F_H = 0.25  # old: compensated missing f in inlet_density ratio
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_T = 908 / 588  # 898 / 588
DEFAULT_T_DEAD_OVER_T_COLD_IN = 288 / 588
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.064  # 9.0 / 1.04
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.064  # 1.04
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 0.92  # 1.0
TARGET_EPS = 0.65
NTU_MATCH = 1.479  # 1.824
# Ao propto NTU^-0.704; Ao_ref/Ao = (NTU/NTU_MATCH)^0.704
AO_REF_OVER_AO_EXP = 0.704
DEFAULT_NTU_MAX = 8.0
SHOW_CUBIC = True
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
    base_name="fig7a_aspect_ratio",
    t=DEFAULT_T,
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
    plot_area_ratio_ref=False,
):
    """
    Save figures as SVG, TIFF, and HD PNG for conventional framework (left subplot from newfig4).
    Shows epsilon and pressure drops.
    """
    # Set font sizes to match Word (10pt = 10 points)
    font_size = 8
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": font_size,
            "mathtext.fontset": "stix",
            "axes.titlesize": TITLE_FONTSIZE,
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
        pressure_drop_assumption,
        c_cold_over_c_hot,
        t,
        d_r,
        molar_mass_ratio,
        sigma_r,
        p_cold_in_over_p_hot_in,
        f_c_over_f_h=f_c_over_f_h,
    )

    # Create figure with single subplot (match fig2a: triple column 7.1 cm width)
    fig, ax = plt.subplots(1, 1, figsize=(7.1 / 2.54, 7 / 2.54))
    ax_twin = ax.twinx()  # Will be used but both pressure drops go on left axis

    # Plot epsilon and both pressure drops; capture line_eps for styling and markers
    line_eps, _, ax, ax_twin = create_plot(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=3.0,  # NTU 0-2
        dp_max=dp_max,
        ax=ax,
        ax_twin=ax_twin,
        plot_triple_g2=PLOT_TRIPLE_G2,
        framework="conventional",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )
    # Remove title if present
    ax.set_title("")

    # Remove grey vertical line at NTU_MATCH (drawn by xflow when SHOW_CUBIC is True)
    for line in list(ax.get_lines()):
        x_data = line.get_xdata()
        if len(x_data) >= 2 and np.allclose(x_data, x_data[0]):
            line.remove()
            break

    # Remove existing legends (keep pressure drop on right axis, match fig2a)
    legend1 = ax.get_legend()
    if legend1:
        legend1.remove()
    legend1_twin = ax_twin.get_legend()
    if legend1_twin:
        legend1_twin.remove()

    # Left axis red like fig2a: epsilon line and axis
    line_eps.set_color("r")
    ax.spines["left"].set_color("r")
    ax.yaxis.label.set_color("r")
    ax.tick_params(axis="y", colors="r")

    # Pressure drop lines: both black, hot solid and cold dashed; legend Hot side / Cold side
    lines_dp = ax_twin.get_lines()
    if len(lines_dp) >= 2:
        lines_dp[0].set_color("k")
        lines_dp[0].set_linestyle(":")
        lines_dp[1].set_color("k")
        lines_dp[1].set_linestyle("--")
        ax.legend(
            handles=lines_dp,
            labels=["Hot side", "Cold side"],
            loc="upper left",
            # bbox_to_anchor=(1.0, 0.5),
            frameon=True,
            facecolor="white",
            edgecolor="black",
            fancybox=False,
        )

    # Set x-axis limits and ticks
    ax.set_xlim(0, 2.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax.set_ylim(0, 0.7)
    ax.set_ylabel(r"Heat Transfer Effectiveness ($\varepsilon$ [%])")
    ax_twin.set_ylabel(r"Pressure Loss ($\Delta p/p_{\mathrm{in}}$ [%])")

    # NTU reference and optimum values (match fig8/9)
    NTU_REF = NTU_MATCH
    NTU_OPT = 1.2513  # From fig7c; refresh after changing Mh_in / f_c/f_h

    # Markers: + for baseline design, o for optimal design
    x_eps = np.array(line_eps.get_xdata())
    y_eps = np.array(line_eps.get_ydata())
    idx_ref = (np.abs(x_eps - NTU_REF)).argmin()
    ax.scatter(NTU_REF, y_eps[idx_ref], marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=10)
    idx_opt = (np.abs(x_eps - NTU_OPT)).argmin()
    # ax.scatter(
    #    NTU_OPT, y_eps[idx_opt], marker="o", color="white", zorder=10, s=MARKER_SIZE_LATEX, facecolor="black", edgecolor="white"
    # )

    # Same markers on right-axis (pressure drop) lines only (hot and cold, not other artists)
    for line in lines_dp:
        x_data = np.array(line.get_xdata())
        y_data = np.array(line.get_ydata())
        idx_ref = (np.abs(x_data - NTU_REF)).argmin()
        y_ref = y_data[idx_ref]
        ax_twin.scatter(NTU_REF, y_ref, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=10)
        idx_opt = (np.abs(x_data - NTU_OPT)).argmin()
        y_opt = y_data[idx_opt]
        # ax_twin.scatter(
        #    NTU_OPT, y_opt, marker="o", color="white", zorder=10, s=MARKER_SIZE_LATEX, facecolor="black", edgecolor="white"
        # )

    if plot_area_ratio_ref:
        base_name = base_name.replace("_aspect_ratio", "_Ao_Aoref")

        # Transform x from NTU to Ao_ref/Ao = (NTU/NTU_MATCH)^0.704
        def _ao_ref_over_ao(ntu):
            return (np.asarray(ntu) / NTU_MATCH) ** AO_REF_OVER_AO_EXP

        for line in list(ax.get_lines()) + list(ax_twin.get_lines()):
            line.set_xdata(_ao_ref_over_ao(line.get_xdata()))
        for ax_use in (ax, ax_twin):
            for coll in ax_use.collections:
                if hasattr(coll, "get_offsets") and coll.get_offsets().size > 0:
                    off = coll.get_offsets().copy()
                    off[:, 0] = _ao_ref_over_ao(off[:, 0])
                    coll.set_offsets(off)
        ax.set_xlim(0.2, 1.2)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_xlabel(r"Inverse of Free-Flow Area $A_{\mathrm{o,ref}}/A_\mathrm{o}$ [-]")

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
    PLOT_AREA_RATIO_REF = True  # Set False for standard NTU x-axis; True saves fig7a_Ao_Aoref.svg etc.
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="fig7a_aspect_ratio",
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
        plot_area_ratio_ref=PLOT_AREA_RATIO_REF,
    )
