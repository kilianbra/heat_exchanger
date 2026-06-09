import os

import numpy as np
import matplotlib.pyplot as plt
import xflow
from xflow import create_plot

from plot_colors import MARKER_SIZE_LATEX

from fig_paths import FINAL_CONF_PAPER, ensure_fig_dirs

ensure_fig_dirs()
save_dir = FINAL_CONF_PAPER

# A = A_ref when NTU = NTU_MATCH; A/A_ref = NTU/NTU_MATCH
NTU_MATCH = 1.479

# Default modeling assumptions (match fig8/9)
DEFAULT_C_COLD_OVER_C_HOT = 1.0  # C_cold / C_hot
DEFAULT_ST_OVER_F = 0.4  # Assumed same for both fluids
# DEFAULT_F_C_OVER_F_H = 1.0  # f_c/f_h
DEFAULT_F_C_OVER_F_H = 0.25
# DEFAULT_D_R = 0.257  # d_r = sigma_r/A_r (cold/hot ratio), match fig8/9
DEFAULT_D_R = 0.25
DEFAULT_GAMMA = 1.4
# DEFAULT_MACH_IN = 0.1  # Reference Mach; g2 = 0.5 * gamma * M^2
DEFAULT_MACH_IN = 0.11  # Mh_in
DEFAULT_G2_H = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2

DEFAULT_DP_MAX = 0.2

# PLOT_TRIPLE_MACH = [0.04, 0.08, 0.17]
PLOT_TRIPLE_MACH = [0.11, 0.06, 0.04]
PLOT_TRIPLE_G2 = [0.5 * DEFAULT_GAMMA * m**2 for m in PLOT_TRIPLE_MACH]


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig6a_lengthening",
    plot_triple_g2=None,
    plot_area_ratio_ref=False,
    plot_ref_plus=False,
):
    """
    Save figures as SVG, TIFF, and HD PNG for given parameter values.

    Parameters:
        plot_triple_g2: If None, use g2_h, else except array of g2 values to plot
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

    # Ensure SHOW_CUBIC is False for this figure
    xflow.SHOW_CUBIC = False
    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))  # IF DOUBLE COLUMN FIGURE, USE THIS
    fig = plt.figure(figsize=((7.1) / 2.54, 7 / 2.54))  # IF TRIPPLE COLUMN FIGURE, USE THIS
    ax = plt.subplot(111)
    ax_twin = ax.twinx()

    line_eps, line_dp_list, ax, ax_twin = create_plot(
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
        framework="conventional",  # Default framework for saved figures
        t=2.0,
        t_dead_over_t_cold_in=1.0,
        p_cold_in_over_p_hot_in=1.0,
        p_dead_over_p_hot_in=1.0,
        gamma=1.4,
    )
    # Remove existing legend if present
    legend = ax.get_legend()
    if legend:
        legend.remove()
    legend_twin = ax_twin.get_legend()
    if legend_twin:
        legend_twin.remove()

    # Add legend at new location
    # handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax_twin.get_legend_handles_labels()
    # ax.legend(handles1 + handles2, labels1 + labels2, loc="center right") # incl eps
    # ax.legend(handles2, labels2, loc="center right")  # just have g^2 values
    # Add new legend using handles2 with custom labels High, Medium, Low and a title

    # independet control for figure
    if plot_area_ratio_ref:
        base_name = base_name.replace("_lengthening", "_A_A_ref_w_plus" if plot_ref_plus else "_A_A_ref")
        # Transform x from NTU to A/A_ref = NTU/NTU_MATCH
        for ax_use in (ax, ax_twin):
            for line in ax_use.get_lines():
                line.set_xdata(np.array(line.get_xdata()) / NTU_MATCH)
            for coll in ax_use.collections:
                if hasattr(coll, "get_offsets") and coll.get_offsets().size > 0:
                    off = coll.get_offsets()
                    off[:, 0] = off[:, 0] / NTU_MATCH
                    coll.set_offsets(off)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}"))
        ax.set_xlabel(r"Heat Transfer Area $A/A_\mathrm{ref}$ [-]")
        # Add + markers at reference design (A/A_ref=1) on heat transfer and hot pressure drop (Mach 0.11 only)
        if plot_ref_plus and line_eps is not None and line_dp_list is not None:
            x_ref = 1.0
            x_eps, y_eps = line_eps.get_xdata(), line_eps.get_ydata()
            if np.min(x_eps) <= x_ref <= np.max(x_eps):
                y_at_ref = np.interp(x_ref, x_eps, y_eps)
                ax.scatter(x_ref, y_at_ref, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=5)
                ax.annotate(
                    "baseline\n design",
                    xy=(x_ref, y_at_ref),
                    xytext=(0.08, 0.88),
                    fontsize=font_size,
                    ha="left",
                    zorder=6,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkB=12),
                )
            # line_dp_list: [0]=M0.04, [1]=M0.06, [2]=M0.11
            line_dp_m011 = line_dp_list[2] if len(line_dp_list) > 2 else None
            if line_dp_m011 is not None:
                x_dp, y_dp = line_dp_m011.get_xdata(), line_dp_m011.get_ydata()
                if np.min(x_dp) <= x_ref <= np.max(x_dp):
                    y_dp_at_ref = np.interp(x_ref, x_dp, y_dp)
                    ax_twin.scatter(
                        x_ref, y_dp_at_ref, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=5
                    )
    else:
        if plot_ref_plus:
            base_name = base_name.replace("_lengthening", "_NTU_w_plus")
        ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
        # Add + markers at reference design (NTU=NTU_MATCH) when plot_ref_plus
        if plot_ref_plus and line_eps is not None and line_dp_list is not None:
            x_ref = NTU_MATCH
            x_eps, y_eps = line_eps.get_xdata(), line_eps.get_ydata()
            if np.min(x_eps) <= x_ref <= np.max(x_eps):
                y_at_ref = np.interp(x_ref, x_eps, y_eps)
                ax.scatter(x_ref, y_at_ref, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=5)
                ax.annotate(
                    "baseline\n design",
                    xy=(x_ref, y_at_ref),
                    xytext=(0.08, 0.88),
                    fontsize=font_size,
                    ha="left",
                    zorder=6,
                    arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkB=12),
                )
            line_dp_m011 = line_dp_list[2] if len(line_dp_list) > 2 else None
            if line_dp_m011 is not None:
                x_dp, y_dp = line_dp_m011.get_xdata(), line_dp_m011.get_ydata()
                if np.min(x_dp) <= x_ref <= np.max(x_dp):
                    y_dp_at_ref = np.interp(x_ref, x_dp, y_dp)
                    ax_twin.scatter(
                        x_ref, y_dp_at_ref, marker="+", s=MARKER_SIZE_LATEX, linewidths=1, color="black", zorder=5
                    )
    ax.set_ylabel(r"Heat Transfer Effectiveness ($\varepsilon$ [%])")
    ax_twin.set_ylabel(r"Hot Pressure Drop ($\Delta p/p_{\mathrm{in}}$ [%])")
    h_right, _ = ax_twin.get_legend_handles_labels()
    # Reverse so highest Mach (0.11) is on top in legend, matching visual order (last plotted = on top)
    legend = ax_twin.legend(
        h_right[::-1],
        [r"$0.11$", r"$0.06$", r"$0.04$"],
        loc="center right",
        bbox_to_anchor=(1, 0.62),
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
        title=("Inlet Mach \n" + "Number (hot)"),
    )
    legend.get_frame().set_alpha(1.0)
    legend.get_title().set_ha("center")
    legend._legend_box.align = "center"
    for t in legend.get_texts():
        t.set_ha("center")

    # Remove title if present
    ax.set_title("")

    # Ensure x-axis shows only 0, 5, 10, 15 with integer formatting (unless plot_area_ratio_ref)
    if not plot_area_ratio_ref:
        ax.set_xticks([0, 5, 10, 15])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}"))

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
    PLOT_AREA_RATIO_REF = False  # Set False for standard NTU x-axis; True saves fig6a_A_A_ref.svg etc.
    PLOT_REF_PLUS = True  # Set True to add + at reference design; saves fig6a_NTU_w_plus.svg when NTU axis
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="fig6a_lengthening",
        plot_triple_g2=PLOT_TRIPLE_G2,
        plot_area_ratio_ref=PLOT_AREA_RATIO_REF,
        plot_ref_plus=PLOT_REF_PLUS,
    )
