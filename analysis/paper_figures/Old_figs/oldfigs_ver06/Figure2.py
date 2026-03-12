import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.general_counterflow import rate_hex_simple

save_dir = os.path.dirname(os.path.abspath(__file__))

# Fluid models (global)
FLUID_HOT = PerfectGasFluid.from_name("kerocomb_helicopter")
FLUID_COLD = PerfectGasFluid.from_name("air")

# Preset A configuration
PRESET_A = {
    "t_over_dhc": 0.02,
    "sigma_r": 9.0,
    "sigma_w": 2.0,
    "d_h_c": 4e-3,
    "ls_over_dh": 5.0,
    "aq_baseline": 43.0 * 2,
    "a_fr_baseline": 0.12,
    "m_dot_hot": 1.6,
    "m_dot_cold": 1.6,
    "th_in": 980,
    "ph_in": 1.06e5,
    "tc_in": 576,
    "pc_in": 7.2e5,
    "dp_max": 0.2,
    "plot_aq_sweep_not_afr": True,
    "plot_pdot_not_dp": False,
}

# Preset B configuration
PRESET_B = {
    "t_over_dhc": 0.02,
    "sigma_r": 2.0,
    "sigma_w": 2.0,
    "d_h_c": 4e-3,
    "ls_over_dh": 5.0,
    "aq_baseline": 43.0,
    "a_fr_baseline": 1.0,
    "m_dot_hot": 1.6,
    "m_dot_cold": 1.6,
    "th_in": 980,
    "ph_in": 1.06e5,
    "tc_in": 576,
    "pc_in": 7.2e5,
    "dp_max": 0.3,
    "plot_aq_sweep_not_afr": False,
    "plot_pdot_not_dp": False,
}


def calculate_plot(
    plot_aq_sweep_not_afr,
    plot_pdot_not_dp,
    t_over_dhc,
    sigma_r,
    sigma_w,
    d_h_c,
    ls_over_dh,
    aq_baseline,
    a_fr_baseline,
    f_in,
    dp_max,
):
    """Calculate and return plot data"""
    # Generate x values
    if plot_aq_sweep_not_afr:
        x = np.geomspace(2, 5000, 100)
        Aq = x
        A_fr = a_fr_baseline
        title = "Heat transfer area variation Aq (m²) (HEx mass, length changes)"
    else:
        # Start from larger frontal area to get lower NTU values
        # Use a_fr_baseline * 10 as starting point to extend range to lower NTU
        # x = np.geomspace(a_fr_baseline, 0.001, 100) # original
        x = np.geomspace(a_fr_baseline * 10, 0.0001, 100)
        Aq = aq_baseline
        A_fr = x
        title = "Frontal area variation A_fr (m²) (Velocity, g^2)"

    # Calculate results
    r_s = rate_hex_simple(
        A_fr=A_fr,
        A_q=Aq,
        f_in=f_in,
        d_h_c=d_h_c,
        sigma_r=sigma_r,
        sigma_w=sigma_w,
        t_over_dhc=t_over_dhc,
        ls_over_dh=ls_over_dh,
    )

    # Find validity mask
    dp_hot = r_s["dp_hot"]
    dp_cold = r_s["dp_cold"]
    eps = r_s["eps"]
    if plot_pdot_not_dp:
        hot_in = f_in.hot.state(T=f_in.Th_in, P=f_in.Ph_in)
        cold_in = f_in.cold.state(T=f_in.Tc_in, P=f_in.Pc_in)
        Cmin = min(f_in.m_dot_hot * hot_in.cp, f_in.m_dot_cold * cold_in.cp)
        Q_max = Cmin * (f_in.Th_in - f_in.Tc_in)
        y_hot = f_in.m_dot_hot * dp_hot * f_in.Ph_in / hot_in.rho / Q_max
        y_cold = f_in.m_dot_cold * dp_cold * f_in.Pc_in / cold_in.rho / Q_max
    else:
        y_hot = dp_hot
        y_cold = dp_cold

    over_limit = (dp_hot >= dp_max) | (dp_cold >= dp_max)
    if np.any(over_limit):
        first_exceed = np.argmax(over_limit)
        validity_mask = np.zeros_like(dp_hot, dtype=bool)
        validity_mask[:first_exceed] = True
    else:
        validity_mask = np.ones_like(dp_hot, dtype=bool)

    x_plot = r_s["ntu"]
    x_title = "NTU"

    return x_plot, eps, y_hot, y_cold, validity_mask, x_title, title, r_s


def get_preset(preset_switch):
    """Get preset configuration based on switch value (0 = A, 1 = B)"""
    return PRESET_B if preset_switch == 1 else PRESET_A


def create_fluid_inputs(preset):
    """Create FluidInputs from preset configuration"""
    return FluidInputs(
        hot=FLUID_HOT,
        cold=FLUID_COLD,
        m_dot_hot=preset["m_dot_hot"],
        m_dot_cold=preset["m_dot_cold"],
        Th_in=preset["th_in"],
        Ph_in=preset["ph_in"],
        Tc_in=preset["tc_in"],
        Pc_in=preset["pc_in"],
    )


if __name__ == "__main__":
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

    # Set figure size BEFORE creating plot (in inches)
    fig_width = 9 / 2.54  # 9 cm to inches
    fig_height = 7 / 2.54  # 7.5 cm to inches
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.subplot(111)

    # Compute Preset A (base plot)
    f_in_A = create_fluid_inputs(PRESET_A)
    xA, epsA, y_hotA, _, maskA, x_titleA, titleA, r_sA = calculate_plot(
        PRESET_A["plot_aq_sweep_not_afr"],
        PRESET_A["plot_pdot_not_dp"],
        PRESET_A["t_over_dhc"],
        PRESET_A["sigma_r"],
        PRESET_A["sigma_w"],
        PRESET_A["d_h_c"],
        PRESET_A["ls_over_dh"],
        PRESET_A["aq_baseline"],
        PRESET_A["a_fr_baseline"],
        f_in_A,
        PRESET_A["dp_max"],
    )

    # Compute Preset B (only dp line overlay)
    f_in_B = create_fluid_inputs(PRESET_B)
    xB, epsB, y_hotB, _, maskB, x_titleB, titleB, r_sB = calculate_plot(
        PRESET_B["plot_aq_sweep_not_afr"],
        PRESET_B["plot_pdot_not_dp"],
        PRESET_B["t_over_dhc"],
        PRESET_B["sigma_r"],
        PRESET_B["sigma_w"],
        PRESET_B["d_h_c"],
        PRESET_B["ls_over_dh"],
        PRESET_B["aq_baseline"],
        PRESET_B["a_fr_baseline"],
        f_in_B,
        PRESET_B["dp_max"],
    )

    # Create plot
    ax.plot(xA[maskA], epsA[maskA], "-", label=r"$\varepsilon$ (both)", color="black", zorder=3)
    ax.set_ylim(0, 1)
    ax_twin = ax.twinx()
    y_lab = r"$\dot{P}/Q_{\mathrm{max}}$" if PRESET_A["plot_pdot_not_dp"] else r"$\Delta p/p_{\mathrm{in}}$ [%]"
    # Preset A dp
    ax_twin.plot(xA[maskA], y_hotA[maskA], "r--", label=r"$\Delta p/p_{\mathrm{in}}$ (fix $A_o$)", zorder=1)
    # Preset B dp overlay
    ax_twin.plot(xB[maskB], y_hotB[maskB], "b-.", label=r"$\Delta p/p_{\mathrm{in}}$ (fix $A$)", zorder=2)
    max_dp = max(PRESET_A["dp_max"], PRESET_B["dp_max"])
    max_dp = 0.2
    ax_twin.set_ylim(0, max_dp)

    # Combine legend handles and labels from ax and ax_twin
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
    # Force opaque legend background and border
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("black")

    ax.set_xlabel("NTU [-]")
    ax.set_ylabel(r"$\varepsilon$ [%]")
    ax_twin.set_ylabel(y_lab)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_twin.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # Set second y-axis ticks: 0, 5, 10, 15, 20 (as percentages: 0, 0.05, 0.10, 0.15, 0.20)
    # set ticks based on max_dp
    ax_twin.set_yticks(np.linspace(0, max_dp, 5))
    # ax.set_title(f"Preset {'B' if INIT_PRESET_SWITCH == 1 else 'A'}: {title}")

    # Set xlim before tight_layout based on Preset A range (primary)
    ax.set_xlim(0, 15)

    # Apply tight layout to optimize spacing (after all labels and limits are set)
    # Use larger padding to ensure ylabel is included
    plt.tight_layout(pad=0.5)

    # Save as SVG with exact dimensions (no bbox_inches='tight' which crops)
    fig.savefig(
        os.path.join(save_dir, "figure2.svg"), dpi=300, facecolor="white", format="svg", bbox_inches=None, pad_inches=0
    )

    plt.show()
