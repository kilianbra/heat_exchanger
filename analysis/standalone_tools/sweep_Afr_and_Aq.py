import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from heat_exchanger.correlations import general_hex_friction_factor, general_hex_j_factor
from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.general_counterflow import (
    calculate_pressure_ratio,
    rate_hex_simple,
)

# region fixed inputs
# Fluid models (global)
FLUID_HOT = PerfectGasFluid.from_name("kerocomb_helicopter")
FLUID_COLD = PerfectGasFluid.from_name("air")

# Fluid inputs (global) - all other values derived from this
F_IN = FluidInputs(
    hot=FLUID_HOT,
    cold=FLUID_COLD,
    m_dot_hot=1.6,  # kg/s
    m_dot_cold=1.6,  # kg/s
    Th_in=980,  # K
    Ph_in=1.06e5,  # Pa (1.06 bar)
    Tc_in=576,  # K
    Pc_in=7.2e5,  # Pa (7.2 bar)
)

# Reference conditions (for non-dimensionalization)
TD = 300  # K
PD = 1e5  # Pa

# Geometry parameters
LS_OVER_DH = 5.0  # Strip length to hydraulic diameter ratio


results = []
A_fr_list = []
Aq_list = []
# endregion

# Geometry
T_OVER_DHC = 0.02  # t/d_h_c = 0.02 (85 micron t over 4 mm walls)
SIGMA_R = 2.0  # Ratio of free flow areas (Ao_h/Ao_c)
SIGMA_W = 1.0  # Ratio of heat transfer areas (Ah/Ac)
A_FR_OVER_AO_C = (1 + SIGMA_R) + 2 * T_OVER_DHC * (1 + SIGMA_W)  # Ratio of frontal area to cold side free flow area
D_H_C = 4e-3  # m, cold side hydraulic diameter

Afr_start = 0.5
A_fr_decrease_ratio_start = 0.999
DP_MAX = 0.2
AQ_BASELINE = 43.0  # m², baseline total heat transfer area (Ah + Ac)
PLOT_EUERGY_NOT_EXERGY = True
Aq_sweep = np.linspace(20, 80, 30)


for Aq in Aq_sweep:
    A_fr = Afr_start
    A_fr_decrease_ratio = A_fr_decrease_ratio_start
    it = 0
    while it < 10:
        r_s = rate_hex_simple(
            A_fr=A_fr,
            A_q=Aq,
            f_in=F_IN,
            d_h_c=D_H_C,
            sigma_r=SIGMA_R,
            sigma_w=SIGMA_W,
            t_over_dhc=T_OVER_DHC,
            ls_over_dh=LS_OVER_DH,
        )

        if r_s["dp_hot"] > DP_MAX or r_s["dp_cold"] > DP_MAX:
            break
        else:
            # r_s["eps"]
            if PLOT_EUERGY_NOT_EXERGY:
                results.append(-r_s["dW_pot_Eu_norm"])
            else:
                results.append(-r_s["dW_pot_Ex_norm"])
            A_fr_list.append(A_fr)
            Aq_list.append(Aq)
        if it < 9:
            A_fr = A_fr * A_fr_decrease_ratio
            it += 1
        else:
            it = 0
            A_fr_decrease_ratio = A_fr_decrease_ratio * A_fr_decrease_ratio_start
            A_fr = A_fr * A_fr_decrease_ratio


# Only plot if there are results available
if len(results) > 0:
    print(f"Number of results: {len(results)}")
    # Convert lists to arrays
    Aq_array = np.array(Aq_list)
    Afr_array = np.array(A_fr_list)
    eps_array = np.array(results)

    # Make a grid for contouring
    xi = np.linspace(np.min(Aq_array), np.max(Aq_array), 100)
    yi = np.linspace(np.min(Afr_array), np.max(Afr_array), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate scattered data to grid
    from scipy.interpolate import griddata

    Zi = griddata((Aq_array, Afr_array), eps_array, (Xi, Yi), method="linear")

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(Xi, Yi, Zi, cmap="viridis", levels=20)

    cbar = plt.colorbar(cp, label="Work Potential creation / Q_max", format=PercentFormatter(xmax=1.0, decimals=1))
    plt.xlabel("Aq (m²)")
    plt.ylabel("A_fr (m²)")
    # Add optimal (maximum) results/Zi for each individual Aq value
    # For each column in Xi (corresponding to fixed Aq), find the max(Zi) and its index, ignoring nans
    max_Zi_for_Aq = np.nanmax(Zi, axis=0)
    idx_max = np.nanargmax(Zi, axis=0)  # For each column (Aq value), row index of max
    y_at_max = yi[idx_max]  # yi is the array of A_fr (y axis)
    # Plot the (Aq, A_fr) where max occurs as a red line
    if PLOT_EUERGY_NOT_EXERGY:
        plt.title("Contour plot of Euergy creation / Q_max vs Aq and A_fr")
        plt.plot(xi, y_at_max, "r--", lw=2, label="A_fr at max Euergy for each Aq")
        # Optionally, plot the value of the max as function of Aq for reference (e.g., as a secondary axis)
        # plt.plot(xi, max_Zi_for_Aq, 'k:', lw=1, label="Max Euergy for Aq (value)")

    else:
        plt.title("Contour plot of Exergy creation / Q_max vs Aq and A_fr")
        plt.plot(xi, y_at_max, "r--", lw=2, label="A_fr at max Exergy for each Aq")
    plt.legend()
    plt.tight_layout()
    plt.show()
