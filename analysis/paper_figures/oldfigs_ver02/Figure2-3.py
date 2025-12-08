import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.general_counterflow import rate_hex_simple
import os
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
    "aq_baseline": 43.0,
    "a_fr_baseline": 0.1,
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
    "a_fr_baseline": 0.5,
    "m_dot_hot": 1.6,
    "m_dot_cold": 1.6,
    "th_in": 980,
    "ph_in": 1.06e5,
    "tc_in": 576,
    "pc_in": 7.2e5,
    "dp_max": 0.2,
    "plot_aq_sweep_not_afr": False,
    "plot_pdot_not_dp": False,
}

# Preset switch: Change this to 0 for Preset A, or 1 for Preset B
INIT_PRESET_SWITCH = 1

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
        x = np.geomspace(2, 2000 / a_fr_baseline, 100)
        Aq = x
        A_fr = a_fr_baseline
        title = "Heat transfer area variation Aq (m²) (HEx mass, length changes)"
    else:
        # Start from larger frontal area to get lower NTU values
        # Use a_fr_baseline * 10 as starting point to extend range to lower NTU
        # x = np.geomspace(a_fr_baseline, 0.001, 100) # original
        x = np.geomspace(a_fr_baseline * 10, 0.001, 100)
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
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 10
    })
    
    # Set figure size BEFORE creating plot (in inches)
    fig_width = 9 / 2.54  # 9 cm to inches
    fig_height = 7 / 2.54  # 7.5 cm to inches
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.subplot(111)
    
    # Get preset based on switch (0 = Preset A, 1 = Preset B)
    preset = get_preset(INIT_PRESET_SWITCH)
    f_in = create_fluid_inputs(preset)
    
    # Calculate data
    x_plot, eps, y_hot, y_cold, validity_mask, x_title, title, r_s = calculate_plot(
        preset["plot_aq_sweep_not_afr"],
        preset["plot_pdot_not_dp"],
        preset["t_over_dhc"],
        preset["sigma_r"],
        preset["sigma_w"],
        preset["d_h_c"],
        preset["ls_over_dh"],
        preset["aq_baseline"],
        preset["a_fr_baseline"],
        f_in,
        preset["dp_max"],
    )
    
    # Create plot
    ax.plot(x_plot[validity_mask], eps[validity_mask], "-", label=r"$\varepsilon$", color="blue")
    ax.set_ylim(0, 1)
    ax_twin = ax.twinx()
    y_lab = r"$\dot{P}/Q_{\mathrm{max}}$" if preset["plot_pdot_not_dp"] else r"$\Delta p/p_{\mathrm{in}}$ [%]"
    ax_twin.plot(x_plot[validity_mask], y_hot[validity_mask], "r--", label=r"$\Delta p/p_{\mathrm{in}}$")
    # ax_twin.plot(x_plot[validity_mask], y_cold[validity_mask], "g--", label=f"{y_lab}_cold")
    ax_twin.set_ylim(0, preset["dp_max"])
    
    # Combine legend handles and labels from ax and ax_twin
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left",labelspacing = 0.05)
    
    ax.set_xlabel("NTU [-]")
    ax.set_ylabel(r"$\varepsilon$ [%]")
    ax_twin.set_ylabel(y_lab)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_twin.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    
    # Set second y-axis ticks: 0, 5, 10, 15, 20 (as percentages: 0, 0.05, 0.10, 0.15, 0.20)
    ax_twin.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    # ax.set_title(f"Preset {'B' if INIT_PRESET_SWITCH == 1 else 'A'}: {title}")

    # Set xlim before tight_layout
    if not preset["plot_aq_sweep_not_afr"]:
        ax.set_xlim(0, x_plot[validity_mask][-1])
    else:
        ax.set_xlim(0, 10)
    
    # Apply tight layout to optimize spacing (after all labels and limits are set)
    # Use larger padding to ensure ylabel is included
    plt.tight_layout(pad=0.5)
    
    if not preset["plot_aq_sweep_not_afr"]:
        # Save as SVG with exact dimensions (no bbox_inches='tight' which crops)
        fig.savefig(os.path.join(save_dir, "figure3.svg"), dpi=300, facecolor='white', 
                   format='svg', bbox_inches=None, pad_inches=0)
    else:
        # Save as SVG with exact dimensions (no bbox_inches='tight' which crops)
        fig.savefig(os.path.join(save_dir, "figure2.svg"), dpi=300, facecolor='white', 
                   format='svg', bbox_inches=None, pad_inches=0)

    plt.show()

