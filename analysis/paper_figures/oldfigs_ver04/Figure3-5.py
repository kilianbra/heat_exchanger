import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks
from wp_sliders import calculate_plot

import os
save_dir = os.path.dirname(os.path.abspath(__file__))

from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid

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

# Fluid models (global)
FLUID_HOT = PerfectGasFluid.from_name("kerocomb_helicopter")
FLUID_COLD = PerfectGasFluid.from_name("air")

# Initial values for sliders
T_OVER_DHC = 0.02  # t/d_h_c = 0.02 (85 micron t over 4 mm walls)
SIGMA_R = 6  # Ratio of free flow areas (Ao_h/Ao_c)
SIGMA_W = 2.0  # Ratio of heat transfer areas (Ah/Ac)
D_H_C = 4e-3  # m, cold side hydraulic diameter
LS_OVER_DH = 5.0  # Strip length to hydraulic diameter ratio
AQ_BASELINE = 43.0
A_FR_BASELINE = 5.0

# Fluid inputs initial values
M_DOT_HOT = 1.6  # kg/s
M_DOT_COLD = 1.6  # kg/s
TH_IN = 980  # K
PH_IN = 1.06e5  # Pa (1.06 bar)
TC_IN = 576  # K
PC_IN = 7.2e5  # Pa (7.2 bar)
DP_MAX = 0.2

F_IN = FluidInputs(
    hot=FLUID_HOT,
    cold=FLUID_COLD,
    m_dot_hot=M_DOT_HOT,
    m_dot_cold=M_DOT_COLD,
    Th_in=TH_IN,
    Ph_in=PH_IN,
    Tc_in=TC_IN,
    Pc_in=PC_IN,
)

A_fr_over_Ao_c = (1 + SIGMA_R) + 2 * T_OVER_DHC * (1 + SIGMA_W)
Ao_c = A_FR_BASELINE / A_fr_over_Ao_c
Ao_h = Ao_c * SIGMA_R


def percent_without_symbol(decimals: int = 0) -> FuncFormatter:
    """
    Format fractional values as percentages without appending a % symbol.
    Ensures existing axis labels containing [%] remain the sole unit indicator.
    """
    decimals = 0 if decimals is None else decimals
    fmt = f"{{:.{decimals}f}}"
    return FuncFormatter(lambda x, _: fmt.format(x * 100))

# Initial boolean values
PLOT_AQ_SWEEP_NOT_AFR = True
PLOT_ENTROPY_NOT_EUERGY = True
# T & T = Fig 4
# F & T = Fig 5
# T & F = Fig 6
# F & F = Fig 7


colors = ["r", "b", "k"]
lines = ["-.", "--", "-"]

# Set figure size BEFORE creating the plot to ensure proper layout
# Size: 9 cm × 7.5 cm (matching Figure2-3.py)
fig_width = 9 / 2.54  # 9 cm to inches
fig_height = 7.5 / 2.54  # 7.5 cm to inches
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

if PLOT_AQ_SWEEP_NOT_AFR:
    A_fr_values = [0.07, 0.09, 0.14]

    for i, A_fr in enumerate(A_fr_values):
        x_plot, y_plot, validity_mask, x_title, label, color, deci, title, r_s, Aq_over_A_fr = calculate_plot(
            PLOT_AQ_SWEEP_NOT_AFR,
            PLOT_ENTROPY_NOT_EUERGY,
            T_OVER_DHC,
            SIGMA_R,
            SIGMA_W,
            D_H_C,
            LS_OVER_DH,
            AQ_BASELINE,
            A_fr,
            F_IN,
            DP_MAX,
        )
        ax.plot(
            x_plot[validity_mask], y_plot[validity_mask], colors[i] + lines[i], label=fr"$g^2$ = {r_s['g2_hot']:.0e}"
        )
        if PLOT_ENTROPY_NOT_EUERGY:
            peaks, _ = find_peaks(y_plot[validity_mask])
            if len(peaks) > 0:
                first_peak = peaks[0]
                y_local_min_idx = first_peak + np.argmin(y_plot[validity_mask][first_peak:])
                ax.scatter(
                    x_plot[validity_mask][y_local_min_idx],
                    y_plot[validity_mask][y_local_min_idx],
                    color=colors[i],
                    marker="o",
                )
        else:
            arg_y_min = np.argmin(y_plot[validity_mask])
            ax.scatter(x_plot[validity_mask][arg_y_min], y_plot[validity_mask][arg_y_min], color=colors[i], marker="o")
    ax.legend(labelspacing=0.05, edgecolor='black', frameon=True, loc='upper right')
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, 20)
    ax.yaxis.set_major_formatter(percent_without_symbol(decimals=deci))
    ax.set_xlabel(r"$N_{\mathrm{tu}}$ [-]")
    # ax.set_ylabel("dW_pot / Q_max")
    # ax.set_ylabel(r"$\Delta \mathcal{E}$")
    
    if PLOT_ENTROPY_NOT_EUERGY:
        ax.set_ylim(0, 0.03 / 100)
        # ax.set_yticks(np.arange(0, 0.03, 0.01))
        ax.set_yticks((0, 0.01/ 100, 0.02/ 100, 0.03/100))
        ax.yaxis.set_major_formatter(percent_without_symbol(decimals=2))
        # ax.set_title("Exergy destr/ Entropy gen (3 Ao, varying Aq)", fontsize=10)        
        ax.set_ylabel(r"$T_d\dot{S}_\mathrm{gen} / \dot{Q}_{\mathrm{max}}$ [%]")
    else:
        ax.set_ylim(-40 / 100, 0)
        ax.set_yticks((-40/ 100, -30/ 100, -20/ 100, -10/ 100, 0))
        ax.yaxis.set_major_formatter(percent_without_symbol(decimals=0))
        # ax.set_title("Euergy destruction (3 Ao, varying Aq)", fontsize=10)
        ax.set_ylabel(r"$\Delta \dot{W}_0^M / \dot{Q}_{\mathrm{max}}$ [%]")
    
    # Apply tight layout to optimize spacing (after ylabel is set)
    # Use larger padding to ensure ylabel is included
    plt.tight_layout(pad=0.5)
    
    if PLOT_ENTROPY_NOT_EUERGY:
        # Save with exact dimensions (no bbox_inches='tight' which crops)
        fig.savefig(os.path.join(save_dir, "figure3.tiff"), dpi=300, facecolor='white', 
                   format='tiff', bbox_inches=None, pad_inches=0)
    else:
        # Save with exact dimensions (no bbox_inches='tight' which crops)
        fig.savefig(os.path.join(save_dir, "figure5.tiff"), dpi=300, facecolor='white', 
                   format='tiff', bbox_inches=None, pad_inches=0)
        fig.savefig(os.path.join(save_dir, "figure5.pdf"), dpi=300, facecolor='white', 
                   format='pdf', bbox_inches=None, pad_inches=0)
else:
    A_q_values = [10, 32, 60]
    for i, A_q in enumerate(A_q_values):
        x_plot, y_plot, validity_mask, x_title, label, color, deci, title, r_s, Aq_over_A_fr = calculate_plot(
            PLOT_AQ_SWEEP_NOT_AFR,
            PLOT_ENTROPY_NOT_EUERGY,
            T_OVER_DHC,
            SIGMA_R,
            SIGMA_W,
            D_H_C,
            LS_OVER_DH,
            A_q,
            A_FR_BASELINE,
            F_IN,
            DP_MAX,
        )

        A_c = 2 * A_q / (1 + SIGMA_W)
        A_h = A_c * SIGMA_W
        Aoh_over_Aq = 1 / (Aq_over_A_fr * A_fr_over_Ao_c * SIGMA_R)
        new_g2_h = r_s["g2_hot"] * np.power(Aoh_over_Aq, 2)

        print(f"Inlet parameters: g2_h = {r_s['g2_hot'][0]:.2e}, new_g2_h = {new_g2_h[0]:.2e}")
        print(
            f"at smallest A_fr: g2_h = {r_s['g2_hot'][validity_mask][-1]:.2e}, new_g2_h = {new_g2_h[validity_mask][-1]:.2e}"
        )

        ax.plot(
            x_plot[validity_mask],
            y_plot[validity_mask],
            colors[i] + lines[i],
            label=fr"$g^2 (A_o/A)^2$ = {new_g2_h[0]:.1e}",
        )

        if PLOT_ENTROPY_NOT_EUERGY:
            peaks, _ = find_peaks(y_plot[validity_mask])
            if len(peaks) > 0:
                first_peak = peaks[0]
                y_local_min_idx = first_peak + np.argmin(y_plot[validity_mask][first_peak:])
                ax.scatter(
                    x_plot[validity_mask][y_local_min_idx],
                    y_plot[validity_mask][y_local_min_idx],
                    color=colors[i],
                    marker="o",
                )
        else:
            arg_y_min = np.argmin(y_plot[validity_mask])
            if arg_y_min != 0:
                ax.scatter(
                    x_plot[validity_mask][arg_y_min], y_plot[validity_mask][arg_y_min], color=colors[i], marker="o"
                )
    ax.legend(labelspacing=0.15, edgecolor='black', frameon=True)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, 5)
    ax.yaxis.set_major_formatter(percent_without_symbol(decimals=deci))
    ax.set_xlabel("NTU [-]")
    # ax.set_ylabel("dW_pot / Q_max")
    
    if PLOT_ENTROPY_NOT_EUERGY:
        ax.set_ylim(0, 0.03 / 100)
        ax.set_yticks((0, 0.01/ 100, 0.02/ 100, 0.03/100))
        ax.yaxis.set_major_formatter(percent_without_symbol(decimals=2))
        # ax.set_title("Exergy destr/ Entropy gen (3 Aq, varying Ao)", fontsize=10)
        ax.set_ylabel(r"$\Delta \dot{W}_0 / \dot{Q}_{\mathrm{max}}$ [%]")
    else:
        ax.set_ylim(-40 / 100, 0)
        ax.set_yticks((-40/ 100, -30/ 100, -20/ 100, -10/ 100, 0))
        ax.yaxis.set_major_formatter(percent_without_symbol(decimals=0))
        # ax.set_title("Euergy destruction (3 Aq, varying Ao)", fontsize=10)
        ax.set_ylabel(r"$\Delta \dot{W}_0^M / \dot{Q}_{\mathrm{max}}$ [%]")
    
    # Apply tight layout to optimize spacing (after ylabel is set)
    # Use larger padding to ensure ylabel is included
    plt.tight_layout(pad=0.5)
    
    # if PLOT_ENTROPY_NOT_EUERGY:
        # Save with exact dimensions (no bbox_inches='tight' which crops)
        # fig.savefig(os.path.join(save_dir, "figure3.tiff"), dpi=300, facecolor='white', 
                #    format='tiff', bbox_inches=None, pad_inches=0)
    # else:
        # Save with exact dimensions (no bbox_inches='tight' which crops)
        # fig.savefig(os.path.join(save_dir, "figure5.tiff"), dpi=300, facecolor='white', 
                #    format='tiff', bbox_inches=None, pad_inches=0)

plt.show()
