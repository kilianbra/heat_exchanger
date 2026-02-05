import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata

from heat_exchanger.correlations import general_hex_friction_factor, general_hex_j_factor
from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.general_counterflow import rate_hex_simple

save_dir = os.path.dirname(os.path.abspath(__file__))

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

# region fixed inputs
case = "Heli"  # "Brewer"
if case == "Heli":
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
    # Geometry
    LS_OVER_DH = 60.0  # Strip length to hydraulic diameter ratio
    T_OVER_DHC = 0.02  # t/d_h_c = 0.02 (85 micron t over 4 mm walls)
    SIGMA_R = 2.0  # Ratio of free flow areas (Ao_h/Ao_c)
    SIGMA_W = 2.0  # Ratio of heat transfer areas (Ah/Ac)
    A_FR_OVER_AO_C = (1 + SIGMA_R) + 2 * T_OVER_DHC * (1 + SIGMA_W)  # Ratio of frontal area to cold side free flow area
    D_H_C = 4e-3  # m, cold side hydraulic diameter
    RHO_WALL_T = 8000 * T_OVER_DHC * D_H_C  # kg/m³, wall material density * thickness -> weight per m² of wall
    MASS_ENGINE = 250  # kg, mass of the engine from TUM paper

    MISSION_HOURS = 2
    LHV_KWH_PER_KG_FUEL = 43.2 / 3.6
    FUEL_PER_HEAT = MISSION_HOURS / LHV_KWH_PER_KG_FUEL

    ETA_OV_OVER_ETA_TURB = 0.2 / 0.88
    ETA_OV_RECUP_MAX_OVER_ETA_TURB = (1 - F_IN.Tc_in / F_IN.Th_in) / 0.88

    KG_HEX_FIXED = 4.7  # kg of hex per kg/s of air
    ALPHA_HEX_KG = 0.9  # kg of hex packaging per kg of matrix

    A_fr_start = 0.5

elif case == "Brewer":
    # Fluid models (global)
    FLUID_COLD = PerfectGasFluid.from_name("para_h2")
    FLUID_HOT = PerfectGasFluid(
        M=27.5, S=150.0, T_ref=350.0, mu_ref=1.12e-5, gamma=1.37, Pr=0.74, cp=1170.0
    )  # 1170 from dT

    # Fluid inputs (global) - all other values derived from this
    F_IN = FluidInputs(
        hot=FLUID_HOT,
        cold=FLUID_COLD,
        m_dot_hot=19.07,  # kg/s
        m_dot_cold=0.166,  # kg/s
        Th_in=778,  # K
        Ph_in=0.388e5,  # Pa (1.06 bar)
        Tc_in=264,  # K
        Pc_in=16.8e5,  # Pa (7.2 bar)
    )
    # Geometry
    LS_OVER_DH = 5.0  # Strip length to hydraulic diameter ratio
    T_OVER_DHC = 0.06  # t/d_h_c = 0.02 (85 micron t over 4 mm walls)
    SIGMA_R = 11.0  # Ratio of free flow areas (Ao_h/Ao_c)
    SIGMA_W = 1.14  # Ratio of heat transfer areas (Ah/Ac)
    A_FR_OVER_AO_C = (1 + SIGMA_R) + 2 * T_OVER_DHC * (1 + SIGMA_W)  # Ratio of frontal area to cold side free flow area
    D_H_C = 4.7e-2  # m, cold side hydraulic diameter
    RHO_WALL_T = 8000 * T_OVER_DHC * D_H_C  # kg/m³, wall material density * thickness -> weight per m² of wall
    MASS_ENGINE = 6000  # kg, mass of the engine from TUM paper

    MISSION_HOURS = 10
    LHV_KWH_PER_KG_FUEL = 120 / 3.6
    FUEL_PER_HEAT = MISSION_HOURS / LHV_KWH_PER_KG_FUEL

    ETA_OV_OVER_ETA_TURB = 0.363 / 0.88
    ETA_OV_RECUP_MAX_OVER_ETA_TURB = (1 - F_IN.Tc_in / F_IN.Th_in) / 0.88
    A_fr_start = 10.0


Cmin = min(
    F_IN.m_dot_hot * F_IN.hot.state(T=F_IN.Th_in, P=F_IN.Ph_in).cp,
    F_IN.m_dot_cold * F_IN.cold.state(T=F_IN.Tc_in, P=F_IN.Pc_in).cp,
)
Q_max = Cmin * (F_IN.Th_in - F_IN.Tc_in)

# Reference conditions (for non-dimensionalization)
TD = 300  # K
PD = 1e5  # Pa

# Geometry parameters


results_euergy = []
results_exergy = []
A_fr_list = []
Aq_list = []
# endregion

# plotting options
PLOT_EUERGY_NOT_EXERGY = True
PLOT_BOTH_PER_AQ = False # false for fig 6 and true for fig 7
PLOT_DIMENSIONAL = False


A_FR_OVER_AO_H = SIGMA_R * A_FR_OVER_AO_C
scale_everything = 1

if PLOT_EUERGY_NOT_EXERGY:
    Afr_start = A_fr_start * scale_everything
    Aq_sweep = np.linspace(0.5, 40, 500) * scale_everything
else:
    Afr_start = A_fr_start * scale_everything
    Aq_sweep = np.linspace(2, 100, 50) * scale_everything
AOH_OVER_AFR_RATIO = SIGMA_R / A_FR_OVER_AO_C
Ao_h_start = Afr_start * AOH_OVER_AFR_RATIO
RHO_IN_HOT = F_IN.hot.state(T=F_IN.Th_in, P=F_IN.Ph_in).rho
g_in2_start = (F_IN.m_dot_hot / Ao_h_start) ** 2 / F_IN.Ph_in / RHO_IN_HOT


A_fr_decrease_ratio_start = 0.999
DP_MAX = 0.2
AQ_BASELINE = 43.0 * scale_everything  # m², baseline total heat transfer area (Ah + Ac)


print(f"g_in2_start: {g_in2_start:.2e} , m_hex_base/m_fuel: {AQ_BASELINE * RHO_WALL_T / MASS_ENGINE * 100:.2f} %")


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

    Zi_euergy = griddata((Aq_array, Afr_array), euergy_array, (Xi, Yi), method="linear")
    Zi_exergy = griddata((Aq_array, Afr_array), exergy_array, (Xi, Yi), method="linear")

    # For each Aq value find best A_fr
    max_euergy_creation_for_Aq = np.nanmax(Zi_euergy, axis=0)
    idx_max_euergy_creation = np.nanargmax(Zi_euergy, axis=0)  # For each column (Aq value), row index of max
    y_at_max_euergy_creation = yi[idx_max_euergy_creation]  # yi is the array of A_fr (y axis)
    # Plot the (Aq, A_fr) where max occurs as a red line

    max_euergy_creation_for_Afr = np.nanmax(Zi_euergy, axis=1)
    idx_max_euergy_creation_for_Afr = np.nanargmax(Zi_euergy, axis=1)  # For each row (A_fr value), column index of max
    x_at_max_euergy_creation_for_Afr = xi[idx_max_euergy_creation_for_Afr]  # xi corresponds to A_q (x axis)

    A_fr_opt_min = y_at_max_euergy_creation[0]

    print(f"For Area {xi[0]:.2f} m² and mass of {xi[0] * RHO_WALL_T:.2f} kg, opt{A_fr_opt_min:.2f} m2 frontal area")

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

    # Set figure size BEFORE creating the plot to ensure proper layout
    # Size: 9 cm × 7.5 cm (matching other figures)
    fig_width = 9 / 2.54  # 9 cm to inches
    fig_height = 7.5 / 2.54  # 7.5 cm to inches
    fig = plt.figure(figsize=(fig_width, fig_height))
    dimensionalisation_x = RHO_WALL_T / F_IN.m_dot_cold
    x_label = "m_hex / mdot_air (kg/(kg/s))"

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
            (xi * dimensionalisation_x * (1 + ALPHA_HEX_KG) + KG_HEX_FIXED),
            euergy_at_euergy_optimal
            / (xi * dimensionalisation_x * (1 + ALPHA_HEX_KG) + KG_HEX_FIXED)
            * FUEL_PER_HEAT
            / 1000
            * Q_max
            / ETA_OV_OVER_ETA_TURB,
            "r--",
            lw=2,
            label=r"$\eta_{ov} = 20\%$",  # "Euergy (euergy-optimal A_fr)",
        )

        plt.plot(
            (xi * dimensionalisation_x * (1 + ALPHA_HEX_KG) + KG_HEX_FIXED),
            euergy_at_euergy_optimal
            / (xi * dimensionalisation_x * (1 + ALPHA_HEX_KG) + KG_HEX_FIXED)
            * FUEL_PER_HEAT
            / 1000
            * Q_max
            / ETA_OV_RECUP_MAX_OVER_ETA_TURB,
            "k-.",
            lw=2,
            label=r"$\eta_{ov} = 40\%$",  # "Euergy (euergy-optimal A_fr)",
        )
        x_label = "m_hex_overall / mdot_air (kg/(kg/s))"
        # plt.plot(
        #     xi * dimensionalisation_x,
        #     euergy_at_exergy_optimal_filtered / (xi * RHO_WALL_T) * FUEL_PER_HEAT / 1000 * Q_max / ETA_OV_OVER_ETA_TURB,
        #     "b--",
        #     lw=2,
        #     label="Euergy (exergy-optimal A_fr)",
        # )

        # plt.axhline(y=ETA_OV_OVER_ETA_TURB / ETA_OV_OVER_ETA_TURB, color="g", linestyle="-", label="Break even")
        # plt.axhline(
        #     y=ETA_OV_RECUP_MAX_OVER_ETA_TURB / ETA_OV_OVER_ETA_TURB,
        #     color="y",
        #     linestyle="-",
        #     label="Ideal Recup break even",
        # )

        # plt.title("Work Potential creation per kg of core HEx mass vs Aq")
        # plt.ylabel("Engine fuel mass avoided per unit mass of HEx (-)")
        plt.ylabel(r"$-\Delta m_{\mathrm{fuel}}/m_{\mathrm{HEx,tot}}$ [-]")
        
        # Set xlabel before tight_layout
        # plt.xlabel(x_label)
        plt.xlabel(r"$m_{\mathrm{HEx,tot}}/\dot{m}_{\mathrm{air}}$ [kg/(kg/s)]")
        
        # Apply tight layout to optimize spacing (after all labels are set)
        # Use larger padding to ensure ylabel is included
        plt.tight_layout(pad=0.5)
        plt.legend(labelspacing=0.05, edgecolor='black', frameon=True, loc='lower right')
        plt.ylim(0, 6)
        plt.xlim(5, 30)
        
        # Save with exact dimensions (no bbox_inches='tight' which crops)
        fig.savefig(os.path.join(save_dir, "figure7.tiff"), dpi=300, facecolor='white', 
                   format='tiff', bbox_inches=None, pad_inches=0)


    else:
        if PLOT_DIMENSIONAL:
            dimensionalisation_x = 1
            x_label = "Aq (m²)"
            plt.ylabel("A_fr (m²)")

            y_eu = y_at_max_euergy_creation
            y_ex = y_at_min_exergy_destr_filtered
            yi = Yi

        else:  # y starts out as A_fr now want to convert it to g^2
            # plot hex aspect ratio on y axis i.e. A_q/A_o_h

            # y_eu = 1 / (y_at_max_euergy_creation * AOH_OVER_AFR_RATIO / xi)
            # y_ex = 1 / (y_at_min_exergy_destr_filtered * AOH_OVER_AFR_RATIO / xi)
            # yi = 1 / (Yi * AOH_OVER_AFR_RATIO / xi)
            # plt.ylabel("A_q/A_o_h")

            # plot g^2_in on y axis
            y_eu = (F_IN.m_dot_hot / y_at_max_euergy_creation) ** 2 / F_IN.Ph_in / RHO_IN_HOT
            y_ex = (F_IN.m_dot_hot / y_at_min_exergy_destr_filtered) ** 2 / F_IN.Ph_in / RHO_IN_HOT
            y_i_nd = (F_IN.m_dot_hot / yi) ** 2 / F_IN.Ph_in / RHO_IN_HOT
            yi = (F_IN.m_dot_hot / Yi) ** 2 / F_IN.Ph_in / RHO_IN_HOT
            plt.ylabel("g^2_in (-)")

        # Fix color scaling at creation so colorbar matches data limits
        cp = plt.contourf(
            Xi * dimensionalisation_x,
            yi,
            Zi,
            cmap="binary_r",
            levels=20,
            vmin=-0.10,
            vmax=0.30,
        )

        # Add optimal (maximum) results/Zi for each individual Aq value
        # For each column in Xi (corresponding to fixed Aq), find the max(Zi) and its index, ignoring nans

        if PLOT_EUERGY_NOT_EXERGY:
            # plt.title("Contour plot of Euergy creation / Q_max vs Aq and A_fr")

            plt.plot(
                xi * dimensionalisation_x,
                y_eu,
                "r--",
                lw=2,
                label="Fixed $m_{\mathrm{HEx,core}}$",
            )
            plt.plot(
                x_at_max_euergy_creation_for_Afr * dimensionalisation_x,
                y_i_nd,
                "b-",
                lw=2,
                label=r"Fixed $g^2$",
            )
            deci = 0

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
        # Create the colorbar with fixed ticks/labels
        def percent_formatter(x, p):
            return f"{x*100:.0f}"

        cbar = plt.colorbar(
            cp,
            label=r"$\Delta \dot{W}_0^M / \dot{Q}_{\mathrm{max}}$ [%]",
            format=FuncFormatter(percent_formatter),
            ticks=[-0.10, 0, 0.10, 0.20, 0.299],
        )
        # Ensure colorbar uses full range and shows top tick
        # cbar.mappable.set_clim(vmin=-0.10, vmax=0.299)
        # cbar.ax.set_ylim(-0.10, 0.299)
        # plt.xlabel(x_label)
        plt.xlabel(r"$m_{\mathrm{HEx,tot}}/\dot{m}_{\mathrm{air}}$ [kg/(kg/s)]")
        plt.ylabel(r"$g^2_{\mathrm{in}}$ [-]")
        plt.xlim(0, 15)
        plt.ylim(0, 0.1)
        if not PLOT_DIMENSIONAL:
            plt.ylim(0, 0.1)
        plt.legend(labelspacing=0.05, edgecolor='black', frameon=True, loc='upper right')
        # Apply tight layout to optimize spacing (after all labels are set)
        # Use larger padding to ensure ylabel is included
        plt.tight_layout(pad=0.5)
        # Save with exact dimensions (no bbox_inches='tight' which crops)
        fig.savefig(os.path.join(save_dir, "figure6.tiff"), dpi=300, facecolor='white', 
                   format='tiff', bbox_inches=None, pad_inches=0)
    plt.show()

r_s = rate_hex_simple(
    A_fr=A_fr_opt_min,
    A_q=Aq_sweep[0],
    f_in=F_IN,
    d_h_c=D_H_C,
    sigma_r=SIGMA_R,
    sigma_w=SIGMA_W,
    t_over_dhc=T_OVER_DHC,
    ls_over_dh=LS_OVER_DH,
)

print(
    f" Re_hot: {r_s['re_hot']:.2e}, Re_cold: {r_s['re_cold']:.2e}, g2_hot: {r_s['g2_hot']:.2e}, g2_cold: {r_s['g2_cold']:.2e}"
)


j_hot = general_hex_j_factor(r_s["re_hot"], LS_OVER_DH)
f_hot = general_hex_friction_factor(r_s["re_hot"], LS_OVER_DH)
j_cold = general_hex_j_factor(r_s["re_cold"], LS_OVER_DH)
f_cold = general_hex_friction_factor(r_s["re_cold"], LS_OVER_DH)

print(f" j/f hot: {j_hot / f_hot:.2e}, j/f cold: {j_cold / f_cold:.2e}")

cutoff_g2 = j_hot / f_hot / 2 * 0.7 ** (-2 / 3) * 0.03 * (F_IN.Th_in / F_IN.Tc_in - 1)

print(f" Single stream cutoff g^2: {cutoff_g2:.2e}")
