import matplotlib.pyplot as plt
import os
from xflow import create_plot

save_dir = os.path.dirname(os.path.abspath(__file__))

# Default modeling assumptions
DEFAULT_C_COLD_OVER_C_HOT = 1.0  # C_cold / C_hot
DEFAULT_ST_OVER_F = 0.4  # Assumed same for both fluids
DEFAULT_F_C_OVER_F_H = 1.0  # f_c/f_h
DEFAULT_D_R = 1.0  # d_r = sigma_r/A_r (cold/hot ratio)
DEFAULT_G2_H = 1e-5  # g2_h

DEFAULT_DP_MAX = 0.2

# Three g^2 values for plotting
PLOT_TRIPLE_G2 = [1e-5, 2e-5, 5e-5]

# Framework-specific parameters for classical
DEFAULT_T = 2.0  # T_hot_in / T_cold_in
DEFAULT_T_DEAD_OVER_T_COLD_IN = 1.1
DEFAULT_GAMMA = 1.4


def save_figures(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    dp_max=DEFAULT_DP_MAX,
    base_name="newfig2",
    plot_triple_g2=None,
    t=DEFAULT_T,
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
    gamma=DEFAULT_GAMMA,
):
    """
    Save figures as SVG, TIFF, and HD PNG for classical framework with multiple g^2 values.

    Parameters:
        plot_triple_g2: List of g^2 values to plot (e.g., [1e-5, 2e-5, 5e-5])
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

    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)
    ax_twin = ax.twinx()  # Created but will be hidden for classical framework

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
        framework="classical",
        t=t,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=1.0,
        p_dead_over_p_hot_in=1.0,
        gamma=gamma,
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
    # Save figures with classical framework
    save_figures(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        dp_max=DEFAULT_DP_MAX,
        base_name="newfig2",
        plot_triple_g2=PLOT_TRIPLE_G2,
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        gamma=DEFAULT_GAMMA,
    )
