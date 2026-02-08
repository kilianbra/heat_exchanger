import os

import matplotlib.pyplot as plt
import numpy as np
import xflow
from scipy.signal import find_peaks
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
SHOW_CUBIC = True  # Enable cubic case

# For inlet_density assumption
DEFAULT_MOLAR_MASS_RATIO = 1.0  # M_cold / M_hot (cold/hot) - default when not specified
DEFAULT_A_R = 1.0  # A_r (cold/hot) - default when not specified

DEFAULT_DP_MAX = 0.3

PLOT_TRIPLE_G2 = None


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig5b",
    plot_triple_g2=None,
    t=DEFAULT_T,
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
):
    """
    Save figures as SVG, TIFF, and HD PNG for classical framework with multiple g^2 values.
    Similar to fig2b but for cubic case (SHOW_CUBIC = True).

    Parameters:
        plot_triple_g2: List of g^2 values to plot (e.g., [1e-3, 5e-3, 20e-3])
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

    # Calculate pressure drop ratio based on assumption
    # Calculate sigma_r from d_r * A_r
    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
    )

    # Set SHOW_CUBIC to True for cubic case
    xflow.SHOW_CUBIC = SHOW_CUBIC
    xflow.NTU_MATCH = NTU_MATCH

    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))  # IF DOUBLE COLUMN FIGURE, USE THIS
    fig = plt.figure(figsize=((6) / 2.54, 7 / 2.54))  # IF TRIPLE COLUMN FIGURE, USE THIS
    ax = plt.subplot(111)
    ax_twin = ax.twinx()  # Created but will be hidden for classical framework

    _, line_list, ax, ax_twin = create_plot(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=2.0,
        dp_max=dp_max,
        ax=ax,
        ax_twin=ax_twin,
        plot_triple_g2=plot_triple_g2,
        framework="classical",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=1.0,
        gamma=gamma,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )

    # Update ylabel and x-axis formatting
    ax.set_ylabel(r"HEx $\Delta Q_0/Q_{\mathrm{max}}$")

    # Remove title if present
    ax.set_title("")

    # Remove grey vertical line at NTU_MATCH (drawn by xflow when SHOW_CUBIC is True)
    for line in list(ax.get_lines()):
        x_data = line.get_xdata()
        if len(x_data) >= 2 and np.allclose(x_data, x_data[0]):
            line.remove()
            break

    ax.set_xticks([0, 1, 2])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}"))
    ax.set_ylim(0, 0.15)
    ax.set_yticks([0, 0.05, 0.10, 0.15])

    # Remove existing legend and recreate using line_list order (matches newfig1: highest g^2 at top)
    legend = ax.get_legend()
    if legend:
        legend.remove()

    # NTU reference and optimum values
    NTU_REF = 1.48
    NTU_OPT = 1.18

    # Add new legend matching newfig1 style with custom labels and title
    y_ref_first = None
    y_opt_first = None
    if line_list:
        # Add grey crosses at NTU_REF and black filled circles at NTU_OPT on all lines
        for line in line_list:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            # Filter out invalid/masked data
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if np.any(valid_mask):
                x_plot = x_data[valid_mask]
                y_plot = y_data[valid_mask]

                # Find y-value at NTU_REF
                idx_ref = (np.abs(x_plot - NTU_REF)).argmin()
                y_ref = y_plot[idx_ref]
                if y_ref_first is None:
                    y_ref_first = y_ref
                ax.scatter(NTU_REF, y_ref, marker="D", color="white", zorder=10, s=50, facecolor="black")


                # Find y-value at NTU_OPT
                idx_opt = (np.abs(x_plot - NTU_OPT)).argmin()
                y_opt = y_plot[idx_opt]
                if y_opt_first is None:
                    y_opt_first = y_opt
                ax.scatter(NTU_OPT, y_opt, marker="o", color="white", zorder=10, s=50, facecolor="black")

        # Annotations with arrows (like fig2b): reference design at grey x, optimal design at circle
        arrow_kw = dict(arrowstyle="->", color="black", lw=1, shrinkB=10)
        if y_ref_first is not None:
            ax.annotate(
                "reference design",
                xy=(NTU_REF, y_ref_first),
                xytext=(0.2, 0.08),
                fontsize=font_size,
                ha="left",
                arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkB=10),
            )
        if y_opt_first is not None:
            ax.annotate(
                "optimal design",
                xy=(NTU_OPT, y_opt_first),
                xytext=(1.1, 0.035),
                fontsize=font_size,
                ha="left",
                arrowprops=arrow_kw,
            )

    plt.tight_layout(pad=0.5)

    # Independent control for figure
    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax.set_ylabel(r"Change in Unavailable Energy ($\Delta \dot{Q}_0/\dot{Q}_{\mathrm{max}}$)")

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
    # Save figures with classical framework for cubic case
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="fig5b",
        plot_triple_g2=PLOT_TRIPLE_G2,
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
    )
