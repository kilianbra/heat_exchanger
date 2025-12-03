import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.general_counterflow import (
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

Cmin = min(
    F_IN.m_dot_hot * F_IN.hot.state(T=F_IN.Th_in, P=F_IN.Ph_in).cp,
    F_IN.m_dot_cold * F_IN.cold.state(T=F_IN.Tc_in, P=F_IN.Pc_in).cp,
)
Q_max = Cmin * (F_IN.Th_in - F_IN.Tc_in)

# Reference conditions (for non-dimensionalization)
TD = 300  # K
PD = 1e5  # Pa

# Geometry parameters
LS_OVER_DH = 5.0  # Strip length to hydraulic diameter ratio


results_euergy = []
results_exergy = []
A_fr_list = []
Aq_list = []
# endregion

# plotting options
PLOT_EUERGY_NOT_EXERGY = False
PLOT_BOTH_PER_AQ = False
PLOT_DIMENSIONAL = False

# Geometry
T_OVER_DHC = 0.02  # t/d_h_c = 0.02 (85 micron t over 4 mm walls)
SIGMA_R = 2.0  # Ratio of free flow areas (Ao_h/Ao_c)
SIGMA_W = 2.0  # Ratio of heat transfer areas (Ah/Ac)
A_FR_OVER_AO_C = (1 + SIGMA_R) + 2 * T_OVER_DHC * (1 + SIGMA_W)  # Ratio of frontal area to cold side free flow area
D_H_C = 4e-3  # m, cold side hydraulic diameter
RHO_WALL_T = 8000 * T_OVER_DHC * D_H_C  # kg/m³, wall material density * thickness -> weight per m² of wall
MASS_ENGINE = 72  # kg, mass of the engine from TUM paper

A_FR_OVER_AO_H = SIGMA_R * A_FR_OVER_AO_C
scale_everything = 1

if PLOT_EUERGY_NOT_EXERGY:
    Afr_start = 0.2 * 2.5 * scale_everything
    Aq_sweep = np.linspace(20, 80, 30) * scale_everything
else:
    Afr_start = 1.5 * scale_everything
    Aq_sweep = np.linspace(20, 100, 50) * scale_everything
AOH_OVER_AFR_RATIO = SIGMA_R / A_FR_OVER_AO_C
Ao_h_start = Afr_start * AOH_OVER_AFR_RATIO
g_in2_start = (F_IN.m_dot_hot / Ao_h_start) ** 2 / 4 / F_IN.Ph_in / F_IN.hot.state(T=F_IN.Th_in, P=F_IN.Ph_in).rho


A_fr_decrease_ratio_start = 0.999
DP_MAX = 0.2
AQ_BASELINE = 43.0 * scale_everything  # m², baseline total heat transfer area (Ah + Ac)


print(f"g_in2_start: {g_in2_start:.2e} , m_hex_base/m_engine: {AQ_BASELINE * RHO_WALL_T / MASS_ENGINE * 100:.2f} %")


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
            results_euergy.append(-r_s["dW_pot_Eu_norm"])
            results_exergy.append(-r_s["dW_pot_Ex_norm"])
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
if len(results_euergy) > 0:
    print(f"Number of results: {len(results_euergy)}")
    # Convert lists to arrays
    Aq_array = np.array(Aq_list)
    Afr_array = np.array(A_fr_list)
    euergy_array = np.array(results_euergy)
    exergy_array = np.array(results_exergy)

    # Make a grid for contouring
    xi = np.linspace(np.min(Aq_array), np.max(Aq_array), 100)
    yi = np.linspace(np.min(Afr_array), np.max(Afr_array), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate scattered data to grid
    from scipy.interpolate import griddata

    Zi_euergy = griddata((Aq_array, Afr_array), euergy_array, (Xi, Yi), method="linear")
    Zi_exergy = griddata((Aq_array, Afr_array), exergy_array, (Xi, Yi), method="linear")

    max_euergy_creation_for_Aq = np.nanmax(Zi_euergy, axis=0)
    idx_max_euergy_creation = np.nanargmax(Zi_euergy, axis=0)  # For each column (Aq value), row index of max
    y_at_max_euergy_creation = yi[idx_max_euergy_creation]  # yi is the array of A_fr (y axis)
    # Plot the (Aq, A_fr) where max occurs as a red line

    min_exergy_destr_for_Aq = np.nanmax(Zi_exergy, axis=0)
    idx_min_exergy_destr = np.nanargmax(Zi_exergy, axis=0)  # For each column (Aq value), row index of min
    y_at_min_exergy_destr = yi[idx_min_exergy_destr]  # yi is the array of A_fr (y axis)
    # Filter out points where y >= threshold by masking the y-values (not the indices)
    afr_threshold = Afr_start * A_fr_decrease_ratio_start**2
    mask_exergy = y_at_min_exergy_destr >= afr_threshold
    y_at_min_exergy_destr_filtered = y_at_min_exergy_destr.copy()
    y_at_min_exergy_destr_filtered[mask_exergy] = np.nan

    if PLOT_EUERGY_NOT_EXERGY:
        Zi = Zi_euergy
    else:
        Zi = Zi_exergy

    plt.figure(figsize=(8, 6))

    if PLOT_BOTH_PER_AQ:
        # Euergy at euergy-optimal A_fr for each Aq (already computed)
        euergy_at_euergy_optimal = max_euergy_creation_for_Aq

        # Euergy at exergy-optimal A_fr for each Aq
        # Extract Zi_euergy values at the indices that maximize exergy
        euergy_at_exergy_optimal = Zi_euergy[idx_min_exergy_destr, np.arange(len(xi))]

        # Apply the same filtering for exergy-optimal (where A_fr is at boundary)
        euergy_at_exergy_optimal_filtered = euergy_at_exergy_optimal.copy()
        euergy_at_exergy_optimal_filtered[mask_exergy] = np.nan

        plt.plot(
            xi,
            euergy_at_euergy_optimal / (xi * RHO_WALL_T) * Q_max / 1000,
            "r--",
            lw=2,
            label="Euergy (euergy-optimal A_fr)",
        )
        plt.plot(
            xi,
            euergy_at_exergy_optimal_filtered / (xi * RHO_WALL_T) * Q_max / 1000,
            "b--",
            lw=2,
            label="Euergy (exergy-optimal A_fr)",
        )

        plt.title("Work Potential creation per kg of core HEx mass vs Aq")
        plt.xlabel("Aq (m²)")
        plt.ylabel("Work Potential creation per kg of core HEx mass (kW/kg)")

    else:
        if PLOT_DIMENSIONAL:
            dimensionalisation_x = 1
            plt.ylabel("A_fr (m²)")
            plt.xlabel("Aq (m²)")

            y_eu = y_at_max_euergy_creation
            y_ex = y_at_min_exergy_destr_filtered
            yi = Yi

        else:
            dimensionalisation_x = RHO_WALL_T / MASS_ENGINE
            plt.ylabel("A_q/A_o_h")
            plt.xlabel("m_hex / m_engine")
            plt.ylim(0, 1000)

            y_eu = 1 / (y_at_max_euergy_creation * AOH_OVER_AFR_RATIO / xi)
            y_ex = 1 / (y_at_min_exergy_destr_filtered * AOH_OVER_AFR_RATIO / xi)
            yi = 1 / (Yi * AOH_OVER_AFR_RATIO / xi)

        cp = plt.contourf(Xi * dimensionalisation_x, yi, Zi, cmap="viridis", levels=20)

        # Add optimal (maximum) results/Zi for each individual Aq value
        # For each column in Xi (corresponding to fixed Aq), find the max(Zi) and its index, ignoring nans

        if PLOT_EUERGY_NOT_EXERGY:
            plt.title("Contour plot of Euergy creation / Q_max vs Aq and A_fr")

            plt.plot(
                xi * dimensionalisation_x,
                y_eu,
                "r--",
                lw=2,
                label="A_fr at max Euergy for each Aq",
            )
            deci = 1

            # Optionally, plot the value of the max as function of Aq for reference (e.g., as a secondary axis)
            # plt.plot(xi, max_Zi_for_Aq, 'k:', lw=1, label="Max Euergy for Aq (value)")

        else:
            plt.title("Contour plot of Exergy creation / Q_max vs Aq and A_fr")
            plt.plot(
                xi * dimensionalisation_x,
                y_ex,
                "r--",
                lw=2,
                label="A_fr at min Exergy for each Aq",
            )
            deci = 3
        cbar = plt.colorbar(
            cp, label="Work Potential creation / Q_max", format=PercentFormatter(xmax=1.0, decimals=deci)
        )

    plt.legend()
    plt.tight_layout()
    plt.show()
