import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.widgets import Button, Slider
import os

from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.dirname(os.path.abspath(__file__))


class ExpSlider(Slider):
    """Custom slider class that handles exponential steps.
    Input bounds should be the actual desired values (must be positive).
    The slider will internally work in log space to create exponential steps."""

    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valstep=None, valfmt=None, **kwargs):
        # Validate inputs
        if valmin <= 0 or valmax <= 0 or valinit <= 0:
            raise ValueError("All values must be positive for exponential slider")

        # Store original bounds
        self.valmin = valmin
        self.valmax = valmax

        # Convert to log space for internal use
        self.log_valmin = np.log(valmin)
        self.log_valmax = np.log(valmax)
        self.log_valinit = np.log(valinit)

        # Convert valstep to log space if provided
        if valstep is not None:
            self.log_valstep = np.log(1 + valstep)  # This creates multiplicative steps
        else:
            self.log_valstep = None

        # Create the slider in log space
        super().__init__(
            ax,
            label,
            self.log_valmin,
            self.log_valmax,
            valinit=self.log_valinit,
            valstep=self.log_valstep,
            valfmt=valfmt,
            **kwargs,
        )

    def _format(self, val):
        """Override the format method to display the exponential value in scientific notation"""
        if self.valfmt:
            return self.valfmt % np.exp(val)
        return f"{np.exp(val):.1e}"

    @property
    def val(self):
        """Override val property to return exponential value"""
        return np.exp(self._val)

    @val.setter
    def val(self, val):
        self._val = val  # val is already in log space from the slider


# Default modeling assumptions
DEFAULT_C_COLD_OVER_C_HOT = 1.0  # C_cold / C_hot
DEFAULT_ST_OVER_F = 0.4  # Same for both fluids
DEFAULT_F_C_OVER_F_H = 1.0  # f_c/f_h
DEFAULT_D_R = 1.0  # d_r = sigma_r/A_r (cold/hot ratio)
DEFAULT_G2_H = 1e-5  # g2_h

# Parameter ranges for sliders
C_COLD_OVER_C_HOT_RANGE = (0.1, 10.0)  # Will use exponential slider
ST_OVER_F_RANGE = (0.2, 0.5)
F_C_OVER_F_H_RANGE = (0.1, 10.0)  # Will use exponential slider
D_R_RANGE = (0.1, 10.0)  # Will use exponential slider
G2_H_RANGE = (1e-5, 5e-3)  # Will use exponential slider

# NTU sweep range
NTU_SWEEP = np.linspace(0.1, 15, 200)


def calculate_capacity_ratios(c_cold_over_c_hot):
    """
    Calculate C_min/C_hot and C_min/C_cold based on C_cold/C_hot ratio.

    Returns:
        C_min_over_C_hot: C_min/C_hot
        C_min_over_C_cold: C_min/C_cold
        C_hot_over_C_cold: C_hot/C_cold
    """
    C_hot_over_C_cold = 1.0 / c_cold_over_c_hot

    if c_cold_over_c_hot <= 1.0:
        # C_cold <= C_hot, so C_min = C_cold
        C_min_over_C_hot = c_cold_over_c_hot  # C_cold/C_hot
        C_min_over_C_cold = 1.0  # C_cold/C_cold
    else:
        # C_cold > C_hot, so C_min = C_hot
        C_min_over_C_hot = 1.0  # C_hot/C_hot
        C_min_over_C_cold = C_hot_over_C_cold  # C_hot/C_cold

    return C_min_over_C_hot, C_min_over_C_cold, C_hot_over_C_cold


def calculate_epsilon_ntu_curve(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    ntu_array=NTU_SWEEP,
    ntu_max=None,
    dp_max=0.2,
):
    """
    Calculate epsilon-NTU curve and pressure drop for given parameters.

    Parameters:
        c_cold_over_c_hot: C_cold / C_hot ratio
        st_over_f: St/f ratio (same for both fluids)
        f_c_over_f_h: f_c / f_h ratio
        d_r: d_r = sigma_r/A_r (cold/hot ratio) - note: d_r = sigma_r/A_r where it's always cold/hot in the ratios
        g2_h: g2_h parameter
        ntu_array: Array of NTU values to evaluate
        ntu_max: Maximum NTU to plot (None for no limit)
        dp_max: Maximum pressure drop fraction (default 0.2 = 20%)

    Returns:
        ntu: NTU values
        epsilon: Effectiveness values
        dp_over_p_in_hot: Hot side pressure drop (as fraction)
        validity_mask: Boolean mask for valid points (dp < dp_max and ntu <= ntu_max if specified)
    """
    # Generate NTU array up to ntu_max if specified, otherwise use full range
    if ntu_max is not None:
        ntu_array = np.linspace(0.1, ntu_max, 200)
    else:
        ntu_array = NTU_SWEEP

    # Calculate capacity ratios
    C_min_over_C_hot, C_min_over_C_cold, C_hot_over_C_cold = calculate_capacity_ratios(c_cold_over_c_hot)

    # C_ratio for epsilon-NTU calculation (C_min/C_max)
    if c_cold_over_c_hot <= 1.0:
        C_ratio = c_cold_over_c_hot  # C_min/C_max = C_cold/C_hot
    else:
        C_ratio = 1.0 / c_cold_over_c_hot  # C_min/C_max = C_hot/C_cold

    # Calculate effectiveness using epsilon-NTU method for counterflow
    epsilon = epsilon_ntu(ntu_array, C_ratio, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)

    # Calculate hot side pressure drop
    # dp_over_p_in_hot = g2_h * NTU * (1/St_over_f_h * Cmin/C_h + 1/f_c_over_f_h * 1/St_over_f_c * d_r* C_min/C_c)
    # Note: St_over_f_h = St_over_f_c = st_over_f (same for both fluids)
    # d_r = sigma_r/A_r where it's always cold/hot in the ratios

    st_over_f_h = st_over_f
    st_over_f_c = st_over_f

    # Pressure drop coefficient (constant part)
    dp_coeff = g2_h * (
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
    )

    # Pressure drop varies linearly with NTU
    dp_over_p_in_hot_array = dp_coeff * ntu_array

    # Create validity mask: stop when dp/p_in >= dp_max
    validity_mask = dp_over_p_in_hot_array < dp_max

    return ntu_array, epsilon, dp_over_p_in_hot_array, validity_mask


def create_plot(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    ntu_max=None,
    dp_max=0.2,
    ax=None,
    ax_twin=None,
    plot_triple_g2=False,
):
    """
    Create or update the plot with given parameters.

    Parameters:
        plot_triple_g2: If True, plot three g^2 values (1e-5, 2e-5, 5e-5) instead of single g2_h

    Returns:
        line_eps: Line object for epsilon curve
        line_dp: Line object(s) for pressure drop curve(s) - list if triple mode, single if not
    """
    if plot_triple_g2:
        # Use three g^2 values: 1e-5, 2e-5, 5e-5
        g2_values = [1e-5, 2e-5, 5e-5]
    else:
        # Use single g^2 value
        g2_values = [g2_h]

    # Calculate epsilon (same for all g^2 values)
    ntu, epsilon, _, validity_mask = calculate_epsilon_ntu_curve(
        c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, g2_values[0], ntu_max=ntu_max, dp_max=dp_max
    )

    if ax is None:
        fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))
        ax = plt.subplot(111)
        ax_twin = ax.twinx()

    # Clear existing lines
    ax.clear()
    ax_twin.clear()

    # Plot epsilon on left axis (only valid points)
    eps_label = r"$\varepsilon$ (all)" if plot_triple_g2 else r"$\varepsilon$"
    line_eps = ax.plot(ntu[validity_mask], epsilon[validity_mask], "-", label=eps_label, color="black", zorder=3)[0]
    ax.set_ylim(0, 1)
    ax.set_xlabel("NTU [-]")
    ax.set_ylabel(r"$\varepsilon$ [%]")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    # Always fix NTU max at 15
    ax.set_xlim(0, 15)

    # Plot pressure drop lines
    line_dp_list = []
    dark_blue = "b"  # Dark blue color from Fig 3

    if plot_triple_g2:
        # Plot three lines: lowest g^2 (dotted), middle (dashed), highest (dot-dashed)
        # Order: highest g^2 first (for legend ordering)
        linestyles = ["-.", "--", ":"]
        for i, g2_val in enumerate(reversed(g2_values)):  # Reverse to plot highest first
            ntu_dp, _, dp_over_p_in_hot, validity_mask_dp = calculate_epsilon_ntu_curve(
                c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, g2_val, ntu_max=ntu_max, dp_max=dp_max
            )
            linestyle_idx = len(g2_values) - 1 - i  # Reverse linestyle order too
            label = rf"hot $\Delta p/p_{{\mathrm{{in}}}}$ ($g^2$ = {g2_val:.0e})"
            line_dp = ax_twin.plot(
                ntu_dp[validity_mask_dp],
                dp_over_p_in_hot[validity_mask_dp] * 100,
                color=dark_blue,
                linestyle=linestyles[linestyle_idx],
                label=label,
                zorder=1,
            )[0]
            line_dp_list.append(line_dp)
    else:
        # Single pressure drop line
        ntu_dp, _, dp_over_p_in_hot, validity_mask_dp = calculate_epsilon_ntu_curve(
            c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, g2_h, ntu_max=ntu_max, dp_max=dp_max
        )
        line_dp = ax_twin.plot(
            ntu_dp[validity_mask_dp],
            dp_over_p_in_hot[validity_mask_dp] * 100,
            "r--",
            label=r"$\Delta p/p_{\mathrm{in}}$",
            zorder=1,
        )[0]
        line_dp_list = [line_dp]

    # Always fix upper bound at 20%
    ax_twin.set_ylim(0, dp_max)
    ax_twin.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    # Set ylabel and ensure it's on the right side
    ax_twin.set_ylabel(r"hot $\Delta p/p_{\mathrm{in}}$ (%)")
    ax_twin.yaxis.set_label_position("right")

    # Make pressure drop axis labels, ticks, and tick labels same color (dark blue if triple, red if single)
    if plot_triple_g2:
        ax_twin.spines["right"].set_color(dark_blue)
        ax_twin.yaxis.label.set_color(dark_blue)
        ax_twin.tick_params(axis="y", colors=dark_blue)
    else:
        ax_twin.spines["right"].set_color("r")
        ax_twin.yaxis.label.set_color("r")
        ax_twin.tick_params(axis="y", colors="r")

    # Combine legend handles and labels
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_twin.get_legend_handles_labels()
    legend = ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower right",
        labelspacing=0.05,
        edgecolor="black",
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        fancybox=True,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("black")

    if plot_triple_g2:
        return line_eps, line_dp_list, ax, ax_twin
    else:
        return line_eps, line_dp_list[0], ax, ax_twin


def save_figures(
    c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, g2_h, dp_max=0.2, base_name="xflow", plot_triple_g2=True
):
    """
    Save figures as SVG, TIFF, and HD PNG for given parameter values.

    Parameters:
        plot_triple_g2: If True, plot three g^2 values (1e-5, 2e-5, 5e-5) instead of single g2_h
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
    import sys

    # Boolean to control triple g^2 mode (True = three lines, False = single line)
    PLOT_TRIPLE_G2 = False

    # Check if we should use sliders or save figures
    use_sliders = False
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        use_sliders = False

    if use_sliders:
        # Create figure with space for sliders on the right
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        ax_twin = ax.twinx()
        plt.subplots_adjust(right=0.6)  # Make room for sliders on the right

        # Set font sizes
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

        # Default NTU max fixed at 15
        DEFAULT_NTU_MAX = 15.0
        DEFAULT_DP_MAX = 0.2

        # Initial plot
        create_plot(
            DEFAULT_C_COLD_OVER_C_HOT,
            DEFAULT_ST_OVER_F,
            DEFAULT_F_C_OVER_F_H,
            DEFAULT_D_R,
            DEFAULT_G2_H,
            ntu_max=DEFAULT_NTU_MAX,
            dp_max=DEFAULT_DP_MAX,
            ax=ax,
            ax_twin=ax_twin,
            plot_triple_g2=PLOT_TRIPLE_G2,
        )

        # Create sliders on the right side
        slider_height = 0.03
        slider_spacing = 0.04
        start_y = 0.8
        slider_left = 0.8
        slider_width = 0.1

        y_pos = start_y

        # C_cold/C_hot slider (exponential)
        slider_c_cold_over_c_hot = ExpSlider(
            plt.axes([slider_left, y_pos, slider_width, slider_height]),
            "C_cold/C_hot",
            C_COLD_OVER_C_HOT_RANGE[0],
            C_COLD_OVER_C_HOT_RANGE[1],
            valinit=DEFAULT_C_COLD_OVER_C_HOT,
            valstep=0.01,
            valfmt="%.2f",
        )
        y_pos -= slider_spacing

        # St_over_f slider (linear)
        slider_st_over_f = Slider(
            plt.axes([slider_left, y_pos, slider_width, slider_height]),
            "St/f",
            ST_OVER_F_RANGE[0],
            ST_OVER_F_RANGE[1],
            valinit=DEFAULT_ST_OVER_F,
            valfmt="%.2f",
        )
        y_pos -= slider_spacing

        # f_c/f_h slider (exponential)
        slider_f_c_over_f_h = ExpSlider(
            plt.axes([slider_left, y_pos, slider_width, slider_height]),
            "f_c/f_h",
            F_C_OVER_F_H_RANGE[0],
            F_C_OVER_F_H_RANGE[1],
            valinit=DEFAULT_F_C_OVER_F_H,
            valstep=0.01,
            valfmt="%.2f",
        )
        y_pos -= slider_spacing

        # d_r slider (exponential)
        slider_d_r = ExpSlider(
            plt.axes([slider_left, y_pos, slider_width, slider_height]),
            "d_r",
            D_R_RANGE[0],
            D_R_RANGE[1],
            valinit=DEFAULT_D_R,
            valstep=0.01,
            valfmt="%.2f",
        )
        y_pos -= slider_spacing

        # g2_h slider (exponential)
        slider_g2_h = ExpSlider(
            plt.axes([slider_left, y_pos, slider_width, slider_height]),
            "g2_h",
            G2_H_RANGE[0],
            G2_H_RANGE[1],
            valinit=DEFAULT_G2_H,
            valstep=1e-5,
            valfmt="%.3e",
        )

        def update_plot(val=None):
            """Update the plot when any slider changes"""
            # ExpSlider.val already returns the actual value (no conversion needed)
            c_cold_over_c_hot = slider_c_cold_over_c_hot.val
            st_over_f = slider_st_over_f.val
            f_c_over_f_h = slider_f_c_over_f_h.val
            d_r = slider_d_r.val
            g2_h = slider_g2_h.val
            # NTU max is always fixed at 15
            ntu_max = DEFAULT_NTU_MAX

            # Clear and recreate plot with current settings
            ax.clear()
            ax_twin.clear()

            create_plot(
                c_cold_over_c_hot,
                st_over_f,
                f_c_over_f_h,
                d_r,
                g2_h,
                ntu_max=ntu_max,
                dp_max=DEFAULT_DP_MAX,
                ax=ax,
                ax_twin=ax_twin,
                plot_triple_g2=PLOT_TRIPLE_G2,
            )

            fig.canvas.draw_idle()

        # Connect sliders to update function
        slider_c_cold_over_c_hot.on_changed(update_plot)
        slider_st_over_f.on_changed(update_plot)
        slider_f_c_over_f_h.on_changed(update_plot)
        slider_d_r.on_changed(update_plot)
        slider_g2_h.on_changed(update_plot)

        plt.show()

    else:
        # Boolean to control triple g^2 mode (True = three lines, False = single line)
        PLOT_TRIPLE_G2 = True
        # Save figures with default values
        save_figures(
            DEFAULT_C_COLD_OVER_C_HOT,
            DEFAULT_ST_OVER_F,
            DEFAULT_F_C_OVER_F_H,
            DEFAULT_D_R,
            DEFAULT_G2_H,
            dp_max=0.2,
            base_name="xflow",
            plot_triple_g2=PLOT_TRIPLE_G2,
        )
