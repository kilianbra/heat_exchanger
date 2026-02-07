import os

import matplotlib.pyplot as plt
import xflow
from xflow import create_plot

save_dir = os.path.dirname(os.path.abspath(__file__))

# Default modeling assumptions
DEFAULT_C_COLD_OVER_C_HOT = 1.0  # C_cold / C_hot
DEFAULT_ST_OVER_F = 0.4  # Assumed same for both fluids
DEFAULT_F_C_OVER_F_H = 1.0  # f_c/f_h
DEFAULT_D_R = 1.0  # d_r = sigma_r/A_r (cold/hot ratio)
DEFAULT_G2_H = 1e-5  # g2_h

DEFAULT_DP_MAX = 0.2

PLOT_TRIPLE_G2 = [1e-3, 5e-3, 20e-3]  # or None for single g2_h


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="newfig1",
    plot_triple_g2=None,
):
    """
    Save figures as SVG, TIFF, and HD PNG for given parameter values.

    Parameters:
        plot_triple_g2: If None, use g2_h, else except array of g2 values to plot
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

    # Ensure SHOW_CUBIC is False for this figure
    xflow.SHOW_CUBIC = False

    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)
    ax_twin = ax.twinx()

    create_plot(
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
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_twin.get_legend_handles_labels()
    # ax.legend(handles1 + handles2, labels1 + labels2, loc="center right") # incl eps
    # ax.legend(handles2, labels2, loc="center right")  # just have g^2 values
    # Add new legend using handles2 with custom labels High, Medium, Low and a title
    custom_labels = ["High", "Medium", "Low"]
    legend = ax.legend(
        handles=handles2,
        labels=custom_labels,
        title="Mass Velocity",
        loc="center right",
        bbox_to_anchor=(1, 0.62),  # Move legend up (y value: 0=bottom, 1=top)
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fancybox=True,
    )

    # Remove title if present
    ax.set_title("")

    # Ensure x-axis shows only 0, 5, 10, 15 with integer formatting
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

    plt.close(fig)
    print(f"Saved figures: {base_name}.svg, {base_name}.tiff, {base_name}.png")


if __name__ == "__main__":
    # Save figures
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="newfig1",
        plot_triple_g2=PLOT_TRIPLE_G2,
    )
