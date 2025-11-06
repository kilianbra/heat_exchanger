"""
Model description:
- This is a model of a microtube HX with an involute geometry.
- The approach is to model a single sector of the HX (assuming involute angle is a multiple of 90 degrees)
- The model will solve row by row moving radially down, inspired by the method of He et al. (2024)
- Initial objective is to replicate the HX design for VIPER engine

Version info:
- Ver01: NTU method, my own try [failed]
- Ver02: LMTD difference (not similar to chinese) [failed]
- Ver03: NTU method, another attempt [failed]
- Ver04: Use KB packages for htc and dp calculations [failed]
- Ver05: Fixed some bugs, and added few other methods (He, Murray) as well as switches for different designs [tbd]
- Ver06: Added ability to return results, and added few more results to return [tbd]
- Ver07: changed to function so callable from other scripts [tbd]
"""

""" Importing KB packages """

# Now you can import the functions from hex_basic.py
from heat_exchanger.hex_basic import dp_friction_only, dp_tube_bank, ntu  # importing dp functions
from heat_exchanger.correlations import (
    tube_bank_nusselt_number_and_friction_factor,
    circular_pipe_nusselt,
    circular_pipe_friction_factor,
    htc_murray,
)


""" Importing public libraries """
from matplotlib import pyplot as plt
from matplotlib import colormaps
import matplotlib
import numpy as np
import time
import CoolProp as CP
from CoolProp.Plots import PropertyPlot, Common
from CoolProp.CoolProp import PropsSI
from CoolProp.CoolProp import PhaseSI
from CoolProp.CoolProp import AbstractState
# os.system('cls')

""" Switches for preset """
preset_custom = 0  # 1 for custom design
preset_AHJEB = 0  # 1 for AHJE version B
preset_AHJEB_H2TOCv2_ExPHT = 0  # 1 for AHJE version B H2TOCv2 ExPHT design
preset_AHJEB_H2TOCv2_ExPHT_Outboard = 0  # for AHJE version B H2TOCv2 ExPHT design, outboard design
preset_viper = 1  # 1 for VIPER design
preset_chinese = 0  # 1 for chinese design
preset_JMHX = 0  # 1 for JMHX design
outboard = 0

""" Switches for solvers """
solver_A = 0  # custom
solver_B = 1  # KB packages
solver_C = 0  # chinese papers
solver_D = 0  # murray correlations

""" Switches for extra calc options """
calc_entropy = 1  # 1 for calculating entropy generation
calc_tsfc = 0  # 1 for calculating TSFC

""" Switches for printing options """
print_info_model = 0  # printing info about the model
print_info_initial = 0  # printing info about the initialisation
print_results_inner = 0  # printing text for inner iteration loops
print_results_outer = 0  # printing text for outer iteration loops
print_convergence = 0  # printing text for convergence
print_results_entropy = 0  # printing text for entropy generation
print_results_tsfc = 0  # printing text for TSFC
print_results = 0  # printing text for results

""" Switches for plotting options """
plot_results_outer = 1  # plotting results
plot_results_entropy = 0  # plotting results
plot_results_tsfc = 0  # plotting results

""" Switches for tuning factors """
qh_tuning_factor = 1
qc_tuning_factor = 1
dp_tuning_factor = 1
qhtc_tuning_factor = 1

input_rect_HX = 0  # 1 for rectangular HX, 0 for annular HX

""" Input parameters (custom design) """
if preset_custom == 1:
    fluid_h = "air"  # hot stream fluid
    fluid_c = "parahydrogen"  # cold stream fluid
    # Flow conditions
    input_Th_in = 500  # hot stream inlet temperature [K]
    input_Ph_in = 1.02e5  # hot stream inlet pressure [Pa]
    input_Tc_in = 40  # cold stream inlet temperature [K]
    input_Pc_in = 50e5  # cold stream inlet pressure [Pa]
    input_mflow_h_total = 60 * 0.2  # total hot stream mass flow rate [kg/s]
    input_mflow_c_total = 0.3  # total cold stream mass flow rate [kg/s]
    # Tube specifications
    input_T_OD = 0.98e-3  # outer diameter of tube [m]
    input_T_t = 0.04e-3  # tube thickness [m]
    input_T_ID = input_T_OD - 2 * input_T_t  # inner diameter of tube [m]
    input_T_sts = 2.5  # norm. transverse spacing between tube centers [-]
    input_T_sls = 1.5  # norm. longitudinal spacing between tube centers [-]
    input_staggered = 1  # 1 for staggered arrangement, 0 for inline arrangement
    # Header specifications
    input_H_number = 21  # number of headers [-]
    input_H_rrows = 4  # radial rows of tubes per header [-]
    input_Inv_angle = 360  # angle of involute [deg]
    # Geometry parameters
    input_HX_rmin = 325e-3  # inner radius of HX [m]
    # input_HX_rmax = 450e-3 # outer radius of HX [m]
    input_HX_rmax = input_HX_rmin + input_H_rrows * input_H_number * input_T_sls * input_T_OD  # outer radius of HX [m]
    input_HX_dX = 540e-3  # axial length of HX [m]
    # targets for plotting
    Tc_in_target = np.nan  # cold inlet target
    Th_out_target = np.nan  # hot exit target
    # Material properties
    k_name = "stainless_steel"  # material name

""" Input parameters (AHJE version B) """
if preset_AHJEB == 1:
    fluid_h = "air"  # hot stream fluid
    fluid_c = "parahydrogen"  # cold stream fluid
    # Flow conditions
    input_Th_in = 500  # hot stream inlet temperature [K]
    input_Ph_in = 1.02e5  # hot stream inlet pressure [Pa]
    input_Tc_in = 40  # cold stream inlet temperature [K]
    input_Pc_in = 50e5  # cold stream inlet pressure [Pa]
    input_mflow_h_total = 12  # total hot stream mass flow rate [kg/s]
    input_mflow_c_total = 0.3  # total cold stream mass flow rate [kg/s]
    # Tube specifications
    input_T_OD = 1.067e-3  # outer diameter of tube [m] based on 19gT/W from needleworks
    input_T_t = 0.129e-3  # tube thickness [m] based on 19gT/W from needleworks
    input_T_ID = input_T_OD - 2 * input_T_t  # inner diameter of tube [m]
    input_T_sts = 2.5  # norm. transverse spacing between tube centers [-]
    input_T_sls = 1.5  # norm. longitudinal spacing between tube centers [-]
    input_staggered = 1  # 1 for staggered arrangement, 0 for inline arrangement
    # Header specifications
    input_H_number = 21  # number of headers [-]
    input_H_rrows = 4  # radial rows of tubes per header [-]
    input_Inv_angle = 360  # angle of involute [deg]
    # Geometry parameters
    # input_HX_rmin = 300e-3 # inner radius of HX [m]
    # input_HX_rmax = input_HX_rmin + input_H_rrows * input_H_number * input_T_sls * input_T_OD # outer radius of HX [m]
    input_HX_rmax = 460e-3  # outer radius of HX [m]
    input_HX_rmin = input_HX_rmax - input_H_rrows * input_H_number * input_T_sls * input_T_OD  # outer radius of HX [m]
    input_HX_dX = 690e-3  # axial length of HX [m]
    # targets for plotting
    Tc_in_target = np.nan  # cold inlet target
    Th_out_target = np.nan  # hot exit target
    # Material properties
    k_name = "stainless_steel"  # material name

""" Input parameters (AHJE version B, H2TOCv2_ExPHT spec) """
if preset_AHJEB_H2TOCv2_ExPHT == 1:
    fluid_h = "air"  # hot stream fluid
    fluid_c = "parahydrogen"  # cold stream fluid
    # Flow conditions
    input_Th_in = 574  # hot stream inlet temperature [K]
    input_Ph_in = 0.368e5  # hot stream inlet pressure [Pa]
    input_Tc_in = 287  # cold stream inlet temperature [K]
    input_Pc_in = 150e5  # cold stream inlet pressure [Pa]
    input_mflow_h_total = 12  # total hot stream mass flow rate [kg/s]
    input_mflow_c_total = 0.76  # total cold stream mass flow rate [kg/s]
    # Tube specifications
    input_T_OD = 1.067e-3  # outer diameter of tube [m] based on 19gT/W from needleworks
    input_T_t = 0.129e-3  # tube thickness [m] based on 19gT/W from needleworks
    input_T_ID = input_T_OD - 2 * input_T_t  # inner diameter of tube [m]
    input_T_sts = 3  # norm. transverse spacing between tube centers [-]
    input_T_sls = 1.5  # norm. longitudinal spacing between tube centers [-]
    input_staggered = 1  # 1 for staggered arrangement, 0 for inline arrangement
    # Header specifications
    input_H_number = 21  # number of headers [-]
    input_H_rrows = 4  # radial rows of tubes per header [-]
    input_Inv_angle = 360  # angle of involute [deg]
    # Geometry parameters
    # input_HX_rmin = 320e-3 # inner radius of HX [m] # from AM regarding radial space
    # input_HX_rmax = input_HX_rmin + input_H_rrows * input_H_number * input_T_sls * input_T_OD # outer radius of HX [m]
    input_HX_rmax = 460e-3  # outer radius of HX [m]
    input_HX_rmin = input_HX_rmax - input_H_rrows * input_H_number * input_T_sls * input_T_OD  # outer radius of HX [m]
    input_HX_dX = 690e-3  # axial length of HX [m]
    # targets for plotting
    Tc_in_target = np.nan  # cold inlet target
    Th_out_target = np.nan  # hot exit target
    dR_available = 140e-3  # available radial space, from AM [m]
    dX_available = 690e-3  # axial length of HX, from AM [m]
    # Material properties
    k_name = "stainless_steel"  # material name

    reverse = 0
    if reverse == 1:
        input_HX_rmin = input_HX_rmax
        input_HX_rmax = (
            input_HX_rmin + input_H_rrows * input_H_number * input_T_sls * input_T_OD
        )  # inner radius of HX [m]

""" Input parameters (AHJE version B, H2TOCv2_ExPHT spec, outboard design) """
if preset_AHJEB_H2TOCv2_ExPHT_Outboard == 1:
    fluid_h = "air"  # hot stream fluid
    fluid_c = "parahydrogen"  # cold stream fluid
    # Flow conditions
    input_Th_in = 574  # hot stream inlet temperature [K]
    input_Ph_in = 0.368e5  # hot stream inlet pressure [Pa]
    input_Tc_in = 287  # cold stream inlet temperature [K]
    input_Pc_in = 150e5  # cold stream inlet pressure [Pa]
    input_mflow_h_total = 12  # total hot stream mass flow rate [kg/s]
    input_mflow_c_total = 0.76  # total cold stream mass flow rate [kg/s]
    # Tube specifications
    input_T_OD = 1.067e-3  # outer diameter of tube [m] based on 19gT/W from needleworks
    input_T_t = 0.129e-3  # tube thickness [m] based on 19gT/W from needleworks
    input_T_ID = input_T_OD - 2 * input_T_t  # inner diameter of tube [m]
    input_T_sts = 3  # norm. transverse spacing between tube centers [-]
    input_T_sls = 1.5  # norm. longitudinal spacing between tube centers [-]
    input_staggered = 1  # 1 for staggered arrangement, 0 for inline arrangement
    # Header specifications
    input_H_number = 21  # number of headers [-]
    input_H_rrows = 4  # radial rows of tubes per header [-]
    input_Inv_angle = 360  # angle of involute [deg]
    # Geometry parameters
    input_HX_rmin = 680e-3  # inner radius of HX [m]
    input_HX_rmax = input_HX_rmin + input_H_rrows * input_H_number * input_T_sls * input_T_OD  # outer radius of HX [m]
    input_HX_dX = 690e-3  # axial length of HX [m]
    # targets for plotting
    Tc_in_target = np.nan  # cold inlet target
    Th_out_target = np.nan  # hot exit target
    dR_available = 140e-3  # available radial space, from AM [m]
    dX_available = 690e-3  # axial length of HX, from AM [m]
    # Material properties
    k_name = "stainless_steel"  # material name

    outboard = 1


""" Input parameters VIPER REL"""
if preset_viper == 1:
    fluid_h = "air"  # hot stream fluid
    fluid_c = "helium"  # cold stream fluid
    # Flow conditions
    input_Th_in = 298  # hot stream inlet temperature [K]
    input_Ph_in = 1.02e5  # hot stream inlet pressure [Pa]
    input_Tc_in = 96  # cold stream inlet temperature [K]
    input_Pc_in = 150e5  # cold stream inlet pressure [Pa]
    # Tube specifications
    input_T_OD = 0.98e-3  # outer diameter of tube [m]
    input_T_t = 0.04e-3  # tube thickness [m]
    input_T_ID = input_T_OD - 2 * input_T_t  # inner diameter of tube [m]
    input_T_sts = 2.5  # norm. transverse spacing between tube centers [-]
    input_T_sls = 1.1  # norm. longitudinal spacing between tube centers [-]
    input_staggered = 1  # 1 for staggered arrangement, 0 for inline arrangement
    # Header specifications
    input_H_number = 21 * 1  # number of headers [-]
    input_H_rrows = 4  # radial rows of tubes per header [-] # if testing, half this double above
    input_Inv_angle = 360  # angle of involute [deg]
    # Geometry parameters
    # input_HX_rmin = 354e-3 # inner radius of HX [m]
    # input_HX_rmax = input_HX_rmin + input_H_rrows * input_H_number * input_T_sls * input_T_OD # outer radius of HX [m]
    input_HX_rmax = 478e-3  # outer radius of HX [m]
    input_HX_rmin = input_HX_rmax - input_H_rrows * input_H_number * input_T_sls * input_T_OD  # inner radius of HX [m]
    input_HX_dX = 200 * input_T_OD * input_T_sts  # axial length of HX [490e-3 m]
    # input_HX_dX = 540e-3 # axial length of HX [m]
    # targets for plotting
    Tc_in_target = 289  # cold inlet target
    Th_out_target = 135  # hot exit target
    # Material properties
    k_name = "inconel_718"  # material name

    input_inlet_vel = 7  # inlet velocity [m/s]
    input_mflow_h_total = (
        1.19 * input_inlet_vel * (2 * np.pi * input_HX_rmax) * input_HX_dX
    )  # * 0.6 # total hot stream mass flow rate [kg/s]
    input_mflow_c_total = 0.15868 * input_mflow_h_total  # = 1.111 [kg/s], total cold stream mass flow rate

""" Input parameters K. He et al. 2024 """
if preset_chinese == 1:
    fluid_h = "air"  # hot stream fluid
    fluid_c = "parahydrogen"  # cold stream fluid
    # Flow conditions
    input_Th_in = 734  # hot stream inlet temperature [K]
    input_Ph_in = 2.62e5  # hot stream inlet pressure [Pa]
    input_Tc_in = 90  # cold stream inlet temperature [K]
    input_Pc_in = 150e5  # cold stream inlet pressure [Pa]
    input_mflow_h_total = 24  # total hot stream mass flow rate [kg/s]
    input_mflow_c_total = 2  # total cold stream mass flow rate [kg/s]
    # Geometry parameters
    input_HX_rmin = 0.112  # inner radius of HX [m]
    # input_HX_rmax = 478e-3 # outer radius of HX [m]
    # input_HX_dX = 540e-3 # axial length of HX [m]
    # Tube specifications
    input_T_OD = 1e-3  # outer diameter of tube [m]
    input_T_t = 0.07e-3  # tube thickness [m]
    input_T_ID = input_T_OD - 2 * input_T_t  # inner diameter of tube [m]
    input_T_sts = 2  # norm. transverse spacing between tube centers [-]
    input_T_sls = 1.5  # norm. longitudinal spacing between tube centers [-]
    input_staggered = 1  # 1 for staggered arrangement, 0 for inline arrangement
    # Header specifications
    input_H_number = 8  # number of headers [-]
    input_H_rrows = 4  # radial rows of tubes per header [-]
    input_Inv_angle = 360  # angle of involute [deg]
    # new specs from paper
    input_tube_arows = 1040  # number of tube rows in the axial direction
    input_HX_dX = input_tube_arows * input_T_sts * input_T_OD  # axial length of HX [m]
    input_HX_rmax = input_HX_rmin + input_H_rrows * input_H_number * input_T_sls * input_T_OD  # outer radius of HX [m]
    # targets for plotting
    Tc_in_target = 460  # cold inlet target
    Th_out_target = 306  # hot exit target
    # Material properties
    k_name = "inconel_718"  # material name

""" Input parameters JMHX (murray thesis) """
if preset_JMHX == 1:
    fluid_h = "nitrogen"  # hot stream fluid
    fluid_c = "nitrogen"  # cold stream fluid
    # Flow conditions
    input_Th_in = 793  # hot stream inlet temperature [K]
    input_Ph_in = 1.05e5  # hot stream inlet pressure [Pa]
    input_Tc_in = 188  # cold stream inlet temperature [K]
    input_Pc_in = 75e5  # cold stream inlet pressure [Pa]
    input_mflow_h_total = 0.0223  # total hot stream mass flow rate [kg/s]
    input_mflow_c_total = 0.0263  # total cold stream mass flow rate [kg/s]
    # Tube specifications
    input_T_OD = 0.38e-3  # outer diameter of tube [m]
    input_T_t = 0.05e-3  # tube thickness [m]
    input_T_ID = input_T_OD - 2 * input_T_t  # inner diameter of tube [m]
    input_T_sts = 2.5  # norm. transverse spacing between tube centers [-]
    input_T_sls = 1.1  # norm. longitudinal spacing between tube centers [-]
    input_staggered = 1  # 1 for staggered arrangement, 0 for inline arrangement
    # Header specifications
    input_H_number = 10  # number of headers [-]
    input_H_rrows = 1  # radial rows of tubes per header [-]
    input_Inv_angle = 360  # angle of involute [deg]
    # Geometry parameters
    input_HX_rmin = 40e-3  # inner radius of HX [m]
    # input_HX_rmax = 478e-3 # outer radius of HX [m]
    input_HX_rmax = input_HX_rmin + input_H_rrows * input_H_number * input_T_sls * input_T_OD  # outer radius of HX [m]
    input_HX_dX = 40e-3  # axial length of HX [m]
    # targets for plotting
    Tc_in_target = 395  # cold inlet target
    Th_out_target = 550  # hot exit target
    # Material properties
    k_name = "stainless_steel"  # material name
    # Special properties
    input_rect_HX = 1  # 1 for rectangular HX, 0 for annular HX

""" Geometry parameters """
print(f"input_HX_rmin: {input_HX_rmin}, r_min/r_max: {input_HX_rmin / input_HX_rmax}")
# Design space
HX_dX = input_HX_dX  # axial length of HX [m]
HX_rmin = input_HX_rmin  # inner radius of HX [m]
HX_rmax = input_HX_rmax  # outer radius of HX [m]
HX_dR = abs(HX_rmax - HX_rmin)  # height of HX [m]
HX_area = np.pi * (HX_rmax**2 - HX_rmin**2)  # frontal area (in the axial direction) of HX [m^2]
if input_rect_HX == 1:
    HX_area = HX_dR * HX_rmax  # frontal area (in the axial direction) of HX [m^2]
HX_volume = HX_dX * HX_area  # total volume of HX [= 0.175 m3]

# Tube specifications
T_OD = input_T_OD  # outer diameter of tube [m]
T_t = input_T_t  # tube thickness [m]
T_ID = T_OD - 2 * T_t  # inner diameter of tube [m]
Dh_c = T_ID  # cold stream hydraulic diameter [m]

# Header specifications
H_number = input_H_number  # number of headers [-]
H_rrows = input_H_rrows  # radial rows of tubes per header [-]
HX_sectors = H_number  # number of periodic sectors in the HX [-]

# Involute specifications
Inv_angle = input_Inv_angle  # angle of involute [deg]
Inv_sector_angle = Inv_angle / HX_sectors  # angle of involute sector [= 17.14 deg]
Inv_b = (HX_rmax - HX_rmin) / np.deg2rad(Inv_angle)  # radial growth per radian [m/rad]
Inv_theta_vals = np.linspace(0, np.deg2rad(Inv_angle), 50)  # theta values
Inv_r_vals = HX_rmin + Inv_b * Inv_theta_vals  # r values
Inv_length = round(
    np.trapezoid(np.sqrt(Inv_r_vals**2 + Inv_b**2), Inv_theta_vals), 1
)  # calculation using arc length integral [= 2.6 m]

# Microtube arrangement
T_sts = input_T_sts  # norm. transverse spacing between tube centers [-]
# T_sls_VIPER = round(HX_dR / (H_number*H_rrows*T_OD),1) # calculating longitudinal spacing in VIPER HX [= 1.5]
T_sls = input_T_sls  # norm. longitudinal spacing between tube centers [-]
T_st = T_sts * T_OD  # transverse spacing between tube centers [m]
T_sl = T_sls * T_OD  # longitudinal spacing between tube centers [m]
if input_staggered == 1:
    T_sd = np.sqrt(T_sl**2 + (0.5 * T_st) ** 2)
    T_sds = T_sd / T_OD
T_rrows = round(HX_dR / T_sl)  # number of radial rows of tubes in the HX [= 84]
# dR_step = HX_dR / T_rrows # radial step size [m]
T_arows = round(HX_dX / T_st)  # number of axial rows of tubes in the HX [= 220]
T_num = T_rrows * T_arows  # number of tubes in the HX [= 18480]

# HT areas (global)
T_htA_hot = np.pi * T_OD * Inv_length  # external heat transfer area of one tube [m2]
T_htA_cold = np.pi * T_ID * Inv_length  # internal heat transfer area of one tube [m2]
if input_rect_HX == 1:
    T_htA_hot = np.pi * T_OD * HX_dX  # external heat transfer area of one tube [m2]
    T_htA_cold = np.pi * T_ID * HX_dX  # internal heat transfer area of one tube [m2]
T_htA_hot_total = T_htA_hot * T_num  # total external heat transfer area of all tubes [= 148 m2]
T_htA_cold_total = T_htA_cold * T_num  # total external heat transfer area of all tubes [= 136 m2]

# Free area ratios
Ah_sigma = (T_st - T_OD) / T_st  # free-flow area / frontal area for hot stream
if input_staggered == 1:
    Ah_sigma_norm = (T_st - T_OD) / T_st
    Ah_sigma_diag = 2 * (T_sd - T_OD) / T_st
    Ah_sigma = min(Ah_sigma_norm, Ah_sigma_diag)
Ac_sigma = np.pi * T_ID**2 / (4 * T_st * T_sl)  # free-flow area / frontal area for cold stream

""" Inlet conditions """
mflow_h_total = input_mflow_h_total  # total hot stream mass flow rate [kg/s] - GUESS
mflow_c_total = input_mflow_c_total  # total cold stream mass flow rate [kg/s] - GUESSx2
mflow_h = mflow_h_total / HX_sectors  # hot stream mass flow per sector modelled [kg/s]
mflow_c_header = mflow_c_total / H_number  # cold stream mass flow per header [kg/s]
mflow_c = mflow_c_header  # cold stream mass flow per tube TESTING THIS FOR NOW..
# mflow_c = mflow_c_header / H_rrows # cold stream mass flow per tube
Th_in = input_Th_in  # hot stream inlet temperature [K]
Ph_in = input_Ph_in  # hot stream inlet pressure [Pa]
Tc_in = input_Tc_in  # cold stream inlet temperature [K]
Pc_in = input_Pc_in  # cold stream inlet pressure [Pa]

""" Material properties and weight """
k_tube, k_rho, material_desc = 14, 7930, "304 Stainless Steel"
vol_tube = np.pi * ((T_OD / 2) ** 2 - (T_ID / 2) ** 2) * Inv_length
mass_per_tube = vol_tube * k_rho
mass_tubes = mass_per_tube * T_num
if preset_viper:
    H_OD = 16.5e-3  # header OD based on needleworks thickest tube (4 gauge)
    H_t = 0.98e-3  # header thickness
    H_ID = H_OD - 2 * H_t
else:
    H_OD = 6e-3  # header OD based on needleworks thickest tube (4 gauge)
    H_t = 0.46e-3  # header thickness
    H_ID = H_OD - 2 * H_t
vol_header = np.pi * ((H_OD / 2) ** 2 - (H_ID / 2) ** 2) * HX_dX
mass_per_header = vol_header * k_rho
mass_headers = mass_per_header * H_number
# need to find mass of manifolds too

if print_info_model == 1:
    print(f"Involute length: {Inv_length:.1f} m, with {Inv_angle:.1f} deg rotation")
    print(f"{T_num} tubes, {T_rrows} radial rows, {T_arows} axial rows")
    print(
        f"Heat transfer area: {T_htA_hot_total:.1f} m^2 hot-side (external), {T_htA_cold_total:.1f} m^2 cold-side (internal)"
    )
    print(
        f"Heat transfer density: {T_htA_hot_total / HX_volume:.1f} m^2/m^3 hot-side (external), {T_htA_cold_total / HX_volume:.1f} m^2/m^3 cold-side (internal)"
    )
    print(f"Free flow areas: {Ah_sigma:.3f} hot stream, {Ac_sigma:.3f} cold stream")
    print(f"Inlet conditions - Hot: {Th_in:.1f} K, {Ph_in / 1e5:.3f} bar, Cold: {Tc_in:.1f} K, {Pc_in / 1e5:.3f} bar")
    print(f"Total mass flow rates - Hot: {mflow_h_total:.1f} kg/s, Cold: {mflow_c_total:.1f} kg/s")
    print(f"Mass flow rates per sector - Hot: {mflow_h:.3f} kg/s, Cold: {mflow_c:.3f} kg/s")
    print(f"Cold stream distribution - Per header: {mflow_c_header:.3f} kg/s, Per tube: {mflow_c:.3f} kg/s")
    print(f"Number of headers: {H_number}, Rows per header: {H_rrows}, Total tubes: {T_num}")
    print(f"Using material: {material_desc} (k={k_tube} W/m/K, ρ={k_rho} kg/m³)")

""" Initialising model """
# Initialize arrays for each radial layer
# Each layer represents a row of tubes moving radially outward
# n = T_rrows  # number of radial layers
n = round(
    T_rrows / H_rrows
)  # number of radial layers (each layer has H_rrows rows of tubes based on header specification)
dR_step = HX_dR / n  # radial step size [m]

# Hot stream arrays (flows from outermost to innermost: 0→n)
Th = np.zeros(n + 1)  # temperature at each node [K]
Ph = np.zeros(n + 1)  # pressure at each node [Pa]
Hh = np.zeros(n + 1)  # enthalpy at each node [J/kg]
rho_h = np.zeros(n + 1)  # hot stream density [kg/m³]

# Cold stream arrays (flows from innermost to outermost: n→0)
Tc = np.zeros(n + 1)  # temperature at each node [K]
Pc = np.zeros(n + 1)  # pressure at each node [Pa]
Hc = np.zeros(n + 1)  # enthalpy at each node [J/kg]
rho_c = np.zeros(n + 1)  # cold stream density [kg/m³]

# Geometry parameters for each layer (indexed from inner to outer: 0=innermost, n=outermost)
r = np.zeros(n + 1)  # radius at each layer [m]
theta = np.zeros(n + 1)  # involute angle at each layer [rad]
tube_length = np.zeros(n)  # tube length in each layer [m]
Aht_hot = np.zeros(n)  # hot side heat transfer area in each layer [m2]
Aht_cold = np.zeros(n)  # cold side heat transfer area in each layer [m2]
Afr_hot = np.zeros(n)  # frontal area for hot stream in each layer [m2]
Afr_cold = np.zeros(n)  # frontal area for cold stream in each layer [m2]
Aff_hot = np.zeros(n)  # free flow area for hot stream in each layer [m2]
Aff_cold = np.zeros(n)  # free flow area for cold stream in each layer [m2]
Dh_hot = np.zeros(n)  # hot stream hydraulic diameter in each layer [m]
Dh_cold = np.zeros(n)  # cold stream hydraulic diameter in each layer [m]

# Vars for testing
Amin_hot = np.zeros(n)  # free flow area for hot stream in each layer X. He 2024 style [m2]
Ah_sigma_He = np.zeros(n)
NTU_kb = np.zeros(n)
Tw = np.zeros(n)  # wall temperature in each layer [K]
Re_h_OD = np.zeros(n)  # hot stream Reynolds number [-]
Dh_hot_OD = np.zeros(n)  # hot stream hydraulic diameter [-]

# Initialise arrays for fluid properties
mu_h = np.zeros(n)  # hot stream viscosity [Pa·s]
mu_c = np.zeros(n)  # cold stream viscosity [Pa·s]
k_h = np.zeros(n)  # hot stream thermal conductivity [W/m/K]
k_c = np.zeros(n)  # cold stream thermal conductivity [W/m/K]
cp_h = np.zeros(n)  # hot stream specific heat capacity [J/kg/K]
cp_c = np.zeros(n)  # cold stream specific heat capacity [J/kg/K]
Pr_h = np.zeros(n)  # hot stream Prandtl number [-]
Pr_c = np.zeros(n)  # cold stream Prandtl number [-]
# rho_h = np.zeros(n) # hot stream density [kg/m³]
# rho_c = np.zeros(n) # cold stream density [kg/m³]

# Initialise arrays for calculated values
G_h = np.zeros(n)  # hot stream mass flux [kg/m²/s]
G_c = np.zeros(n)  # cold stream mass flux [kg/m²/s]
Re_h = np.zeros(n)  # hot stream Reynolds number [-]
Re_c = np.zeros(n)  # cold stream Reynolds number [-]
Vel_h = np.zeros(n)  # hot stream velocity [m/s]
Vel_c = np.zeros(n)  # cold stream velocity [m/s]
f_h = np.zeros(n)  # hot stream friction factor [-]
f_c = np.zeros(n)  # cold stream friction factor [-]
Nu_h = np.zeros(n)  # hot stream Nusselt number [-]
Nu_c = np.zeros(n)  # cold stream Nusselt number [-]
h_h = np.zeros(n)  # hot stream heat transfer coefficient [W/m²/K]
h_c = np.zeros(n)  # cold stream heat transfer coefficient [W/m²/K]
St_h = np.zeros(n)  # hot stream Stanton number [-]
St_c = np.zeros(n)  # cold stream Stanton number [-]
J_h = np.zeros(n)  # hot stream Colburn j-factor [-]
J_c = np.zeros(n)  # cold stream Colburn j-factor [-]
U_h = np.zeros(n)  # overall heat transfer coefficient (hot side basis) [W/m²/K]
U_c = np.zeros(n)  # overall heat transfer coefficient (cold side basis) [W/m²/K]
C_h = np.zeros(n)  # heat capacity rate for hot stream [W/K]
C_c = np.zeros(n)  # heat capacity rate for cold stream [W/K]
C_min = np.zeros(n)  # minimum heat capacity rate [W/K]
C_max = np.zeros(n)  # maximum heat capacity rate [W/K]
C_ratio = np.zeros(n)  # capacity ratio [-]
NTU_h = np.zeros(n)  # number of transfer units [-]
NTU_c = np.zeros(n)  # number of transfer units [-]
eps = np.zeros(n)  # effectiveness [-]
NTU = np.zeros(n)  # number of transfer units [-]
q_max = np.zeros(n)  # maximum heat transfer rate [W]
q = np.zeros(n)  # actual heat transfer rate [W]
dP_h = np.zeros(n)  # hot stream pressure drop [Pa]
dP_c = np.zeros(n)  # cold stream pressure drop [Pa]

if print_info_initial == 1:
    print(f"\nModel initialized with {n} radial layers")
    print(f"Layers range from r={HX_rmin * 1000:.1f}mm to r={HX_rmax * 1000:.1f}mm")
    print(f"Average tube length per layer: {np.mean(tube_length):.3f}m")

""" Solving models """
""" Future additions planned 
- Add feature to store convergence history
- Add outer loop
"""
# Start timer for single run (exclude plotting)
t0 = time.perf_counter()
# Calculate geometry for each layer (inner to outer indexing)
for j in range(n + 1):
    # Calculate radius and angle for each layer
    r[j] = HX_rmin + j * dR_step
    theta[j] = (r[j] - HX_rmin) / Inv_b

iter_max = 200  # max numebr of iterations for outer loop
iter_tol_Th = 1  # i.e. 1K tolerance
iter_tol_Ph = Ph_in * 0.001  # i.e. 0.1% tolerance
iter_tol_Tc = 1  # i.e. 1K tolerance
iter_tol_Pc = Pc_in * 0.001  # i.e. 0.1% tolerance
Th_out_guess = 0.5 * Th_in
Ph_out_guess = 0.97 * Ph_in
Tc_out_guess = 0.9 * Th_in
Pc_out_guess = 0.97 * Pc_in

for i in range(iter_max):
    i = 0
    # Both streams flow from innermost to outermost
    # Hot stream: inlet at outermost layer (node 0), outlet at innermost layer (node n)
    Th_out_estimate = Th_out_guess  # hot stream outlet temperature [K] - initial guess
    Ph_out_estimate = Ph_out_guess  # hot stream outlet pressure [Pa] - initial guess
    Th[0] = Th_out_estimate  # hot stream inlet temperature [K] (outermost layer)
    Ph[0] = Ph_out_estimate  # hot stream inlet pressure [Pa] (outermost layer)
    Hh[0] = PropsSI("H", "T", Th_out_estimate, "P", Ph_out_estimate, fluid_h)  # hot stream inlet enthalpy [J/kg]
    # Cold stream: inlet at innermost layer (node 0), outlet at innermost layer (node n)
    Tc[0] = Tc_in  # cold stream inlet temperature [K] (innermost layer)
    Pc[0] = Pc_in  # cold stream inlet pressure [Pa] (innermost layer)
    Hc[0] = PropsSI("H", "T", Tc_in, "P", Pc_in, fluid_c)  # cold stream inlet enthalpy [J/kg]
    if outboard == 1:
        """ 
        For outboard design, the hot stream is innermost, and the cold stream inlet is outermost
        This means that the cold stream outlet [-1] is convered to rather than the hot stream as before (for inboard)
        """
        Th[0] = Th_in
        Ph[0] = Ph_in
        Hh[0] = PropsSI("H", "T", Th_in, "P", Ph_in, fluid_h)
        Tc[0] = Tc_out_guess
        Pc[0] = Pc_out_guess
        Hc[0] = PropsSI("H", "T", Tc_out_guess, "P", Pc_out_guess, fluid_c)

    for j in range(n):  # k = 0, 1, 2, ..., n-1
        # Both streams flow from outermost to innermost
        # Hot stream: node j → j+1 (outermost to innermost)
        # Cold stream: node j → j+1 (outermost to innermost)
        # j = n - 1 - k # j = n-1, n-2, ..., 0 (outermost to innermost)

        # 0. Calculate tube length and areas for each layer
        # Arc length calculation for this layer
        tube_length[j] = np.trapezoid(np.sqrt(r[j : j + 2] ** 2 + Inv_b**2), theta[j : j + 2])
        if input_rect_HX == 1:
            tube_length[j] = HX_dX

        # Heat transfer areas for this layer
        tubes_in_layer = T_arows * H_rrows  # number of tubes in this radial layer in the axial direction
        Aht_hot[j] = np.pi * T_OD * tube_length[j] * tubes_in_layer  # hot stream heat transfer area [m2]
        Aht_cold[j] = np.pi * T_ID * tube_length[j] * tubes_in_layer  # cold stream heat transfer area [m2]
        # Calculate frontal area for this layer
        Afr_hot[j] = HX_dX * 2 * np.pi * r[j] / HX_sectors  # hot stream frontal area [m2]
        if input_rect_HX == 1:
            Afr_hot[j] = HX_dX * HX_rmin  # hot stream frontal area [m2]
        Afr_cold[j] = HX_dX * dR_step  # cold stream frontal area [m2]
        if input_rect_HX == 1:
            Afr_cold[j] = HX_dX * dR_step  # cold stream frontal area [m2]
        # Calculate free flow areas for this layer
        Aff_hot[j] = Afr_hot[j] * Ah_sigma  # hot stream free flow area [m2]
        Aff_cold[j] = Afr_cold[j] * Ac_sigma  # cold stream free flow area [m2]
        # Calculate free flow areas for this layer (K. He 2024 style): GIVES IDENTICAL ANSWERS
        Amin_hot[j] = Afr_hot[j] * min((T_sts - 1) / T_sts, 2 * (np.sqrt(T_sls**2 + T_sts**2 / 4) - 1) / T_sts)
        Ah_sigma_He[j] = Amin_hot[j] / Afr_hot[j]
        # Calculate hydraulic diameters for this layer
        # Dh_hot[j] = 4 * (T_st - T_OD) * T_sl / (np.pi * T_OD) / 2 # hot stream hydraulic diameter [m]
        Lf = H_rrows * T_sls * T_OD  # flow length according to K. He (2024) [m]
        Dh_hot[j] = (4 * Aff_hot[j] * Lf) / Aht_hot[j]  # hot stream hydraulic diameter according to K. He (2024) [m]
        Dh_hot_OD[j] = T_OD  # hot stream hydraulic diameter according to K&L for correlations [m]
        Dh_cold[j] = Dh_c  # cold stream hydraulic diameter [m]

        # 1. Calculate fluid properties
        mu_h[j] = PropsSI("V", "T", Th[j], "P", Ph[j], fluid_h)  # hot stream viscosity [Pa·s]
        mu_c[j] = PropsSI("V", "T", Tc[j], "P", Pc[j], fluid_c)  # cold stream viscosity [Pa·s]
        k_h[j] = PropsSI("L", "T", Th[j], "P", Ph[j], fluid_h)  # hot stream thermal conductivity [W/m/K]
        k_c[j] = PropsSI("L", "T", Tc[j], "P", Pc[j], fluid_c)  # cold stream thermal conductivity [W/m/K]
        cp_h[j] = PropsSI("C", "T", Th[j], "P", Ph[j], fluid_h)  # hot stream specific heat capacity [J/kg/K]
        cp_c[j] = PropsSI("C", "T", Tc[j], "P", Pc[j], fluid_c)  # cold stream specific heat capacity [J/kg/K]
        Pr_h[j] = PropsSI("Prandtl", "T", Th[j], "P", Ph[j], fluid_h)  # hot stream Prandtl number [-]
        Pr_c[j] = PropsSI("Prandtl", "T", Tc[j], "P", Pc[j], fluid_c)  # cold stream Prandtl number [-]
        rho_h[j] = PropsSI("D", "T", Th[j], "P", Ph[j], fluid_h)  # hot stream density [kg/m³]
        rho_c[j] = PropsSI("D", "T", Tc[j], "P", Pc[j], fluid_c)  # cold stream density [kg/m³]

        # 2. Calculate mass fluxs and Reynolds numbers
        Vel_h[j] = mflow_h / (Aff_hot[j] * rho_h[j])  # hot stream velocity [m/s]
        Vel_c[j] = mflow_c / (Aff_cold[j] * rho_c[j])  # cold stream velocity [m/s]
        G_h[j] = mflow_h / Aff_hot[j]  # hot stream mass flux [kg/m²/s]
        G_c[j] = mflow_c / Aff_cold[j]  # cold stream mass flux [kg/m²/s]
        Re_h[j] = G_h[j] * Dh_hot[j] / mu_h[j]  # hot stream Reynolds number [-]
        Re_h_OD[j] = G_h[j] * Dh_hot_OD[j] / mu_h[j]  # hot stream Reynolds number [-]
        Re_c[j] = G_c[j] * Dh_cold[j] / mu_c[j]  # cold stream Reynolds number [-]

        # 3.1.a. Calculate coolant stream HT and friction parameters
        if solver_A == 1:
            if Re_c[j] > 2300:  # turbulent flow (includes transition)
                f_c[j] = (0.790 * np.log(Re_c[j]) - 1.64) ** (-2)  # friction factor [-]
                Nu_c[j] = ((f_c[j] / 8) * (Re_c[j] - 1000) * Pr_c[j]) / (
                    1 + 12.7 * np.sqrt(f_c[j] / 8) * (Pr_c[j] ** (2 / 3) - 1)
                )  # Nusselt number [-]
            else:  # laminar flow for circular cross-section tubes
                f_c[j] = 64 / Re_c[j]  # friction factor [-]
                Nu_c[j] = 3.66
            h_c[j] = Nu_c[j] * k_c[j] / Dh_cold[j]  # cold stream heat transfer coefficient [W/m²/K]

        # 3.1.b. Calculate coolant stream HT and friction parameters (KB packages)
        if solver_B == 1 or preset_JMHX == 1:
            Nu_c[j] = circular_pipe_nusselt(Re_c[j], 0, prandtl=Pr_c[j])
            f_c[j] = circular_pipe_friction_factor(Re_c[j], 0)
            h_c[j] = Nu_c[j] * k_c[j] / Dh_cold[j]  # cold stream heat transfer coefficient [W/m²/K]

        # 3.1.c. Calculate coolant stream HT and friction parameters (K. He et al. 2024)
        # This is Jackson [47 in K. He 2024] Nusselt in tube
        if solver_C == 1:
            if Re_c[j] <= 2300:
                Nu_c[j] = 4.364  # laminar flow
            elif Re_c[j] >= 2300 and Re_c[j] < 8000:
                Nu_lam = 4.364
                Nu_lam_dash = 3.66  # @ constant wall temp
                Nu_turb = 0.0183 * Re_c[j] ** 0.82 * Pr_c[j] ** 0.5
                Nu_c[j] = (2 ** (-16) * (Nu_lam + Nu_lam_dash) ** 16 + Nu_turb) ** (1 / 16)
            elif Re_c[j] >= 8000:
                Nu_c[j] = (
                    0.0183 * Re_c[j] ** 0.82 * Pr_c[j] ** 0.5
                )  # turbulent flow, and without damping term at the end
            h_c[j] = Nu_c[j] * k_c[j] / Dh_cold[j]  # cold stream heat transfer coefficient [W/m²/K]

        # 3.2.b. Calculate hot stream HT and friction parameters (KB packages)
        if solver_B == 1:
            if input_staggered == 1:
                Nu_h[j], f_h[j] = tube_bank_nusselt_number_and_friction_factor(
                    Re_h_OD[j], T_sls, T_sts, Pr_h[j], inline=False, n_rows=T_rrows
                )
            elif input_staggered == 0:
                Nu_h[j], f_h[j] = tube_bank_nusselt_number_and_friction_factor(
                    Re_h_OD[j], T_sls, T_sts, Pr_h[j], inline=True, n_rows=T_rrows
                )
            h_h[j] = Nu_h[j] * k_h[j] / Dh_hot_OD[j]  # hot stream heat transfer coefficient [W/m²/K]

        # 3.2.c. Calculate hot stream HT and friction parameters (K. He et al. 2024)
        # Zukauskas 1972 equations 40 and 41: , valid for Re > 1e3 and Re < 2e5 (Mixed Subcritical flow regime)
        if solver_C == 1:
            if T_st / T_sl < 2:
                zuk_C1 = 0.35 * (T_st / T_sl) ** 0.2
                zuk_m = 0.6
            else:
                zuk_C1 = 0.4
                zuk_m = 0.6
            if (n - j - 1) * H_rrows <= 16:  # Correction for <16 rows from Fig 65 in Zukauskas 1972
                zuk_C2_x = [1, 2, 3, 4, 5, 7, 10, 13, 16]
                zuk_C2_y = [0.64, 0.76, 0.84, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99]
                zuk_C2 = np.interp((n - j - 1) * H_rrows, zuk_C2_x, zuk_C2_y)
            else:
                zuk_C2 = 1
            # Zukauskas correlation for external flow (damping term removed as Tw unknown, also effect small as ^0.25 tf ~1/2%)
            Nu_h[j] = zuk_C2 * (zuk_C1 * Re_h_OD[j] ** zuk_m * Pr_h[j] ** (0.36))
            h_h[j] = Nu_h[j] * k_h[j] / Dh_hot_OD[j]  # hot stream heat transfer coefficient [W/m²/K]

            # # 3.2.b.2 Calculating wall temperatures (K. He et al. 2024)
            # Tw_guess = 0.5 * (Th[j] + Tc[j])
            # iter_max_Tw = 100 # max number of iterations for wall temperature calculation
            # iter_tol_Tw = 5 # tolerance for wall temperature convergence [K]
            # for i_Tw in range(iter_max_Tw):
            #     # COLD STREAM CORRECTIONS
            #     if Re_c[j] <= 2300:
            #         Nu_c[j] = 4.364 # laminar flow
            #     elif Re_c[j] >= 2300 and Re_c[j] < 8000:
            #         Pr_c_Tw = PropsSI('Prandtl', 'T', Tw_guess, 'P', Pc[j], fluid_c)
            #         Nu_lam = 4.364
            #         Nu_lam_dash = 3.66 # @ constant wall temp
            #         Nu_turb = 0.0183 * Re_c[j]**0.82 * Pr_c[j]**0.5 * (Pr_c[j]/Pr_c_Tw)**(0.30)
            #         Nu_c[j] = (2**(-16) * (Nu_lam + Nu_lam_dash)**16 + Nu_turb)**(1/16)
            #     elif Re_c[j] >= 8000:
            #         Pr_c_Tw = PropsSI('Prandtl', 'T', Tw_guess, 'P', Pc[j], fluid_c)
            #         Nu_c[j] = 0.0183 * Re_c[j]**0.82 * Pr_c[j]**0.5 * (Pr_c[j]/Pr_c_Tw)**(0.30) # turbulent flow, and without damping term at the end
            #     h_c[j] = Nu_c[j] * k_c[j] / Dh_cold[j] # cold stream heat transfer coefficient [W/m²/K]
            #     # HOT STREAM CORRECTIONS
            #     Pr_h_Tw = PropsSI('Prandtl', 'T', Tw_guess, 'P', Ph[j], fluid_h)
            #     Nu_h[j] = zuk_C2 * (zuk_C1 * Re_h[j]**zuk_m * Pr_h[j]**(0.36)) * (Pr_h[j]/Pr_h_Tw)**(1/4)
            #     h_h[j] = Nu_h[j] * k_h[j] / Dh_hot_OD[j] # hot stream heat transfer coefficient [W/m²/K]
            #     # CALCULATE WALL TEMPERATURE
            #     Tw[j] = Th[j] - (h_c[j]*T_ID*(Th[j]-Tc[j]))/(h_h[j]*T_OD + h_c[j]*T_ID)
            #     # LOOP UNTIL CONVERGED
            #     error_Tw = abs(Tw[j] - Tw_guess)
            #     if error_Tw < iter_tol_Tw:
            #         print(f"Converged in {i+1} iterations! Errors: {error_Tw:.1f} K")
            #         break
            #     else:
            #         Tw_guess = Tw[j]

        if solver_D == 1:
            h_h[j], f_h[j] = htc_murray(G_h[j], cp_h[j], Re_h[j], Pr_h[j], T_sls, T_sts, T_OD)
            # if Re_c[j] <= 2300:
            #     Nu_c[j] = 4.364 # laminar flow
            # elif Re_c[j] >= 2300 and Re_c[j] < 8000:
            #     Nu_lam = 4.364
            #     Nu_lam_dash = 3.66 # @ constant wall temp
            #     Nu_turb = 0.023 * Re_c[j]**0.8 * Pr_c[j]**0.4
            #     Nu_c[j] = (2**(-16) * (Nu_lam + Nu_lam_dash)**16 + Nu_turb)**(1/16)
            # elif Re_c[j] >= 8000:
            #     Nu_c[j] = 0.023 * Re_c[j]**0.8 * Pr_c[j]**0.4 # turbulent flow, and without damping term at the end
            # h_c[j] = Nu_c[j] * k_c[j] / Dh_cold[j] # cold stream heat transfer coefficient [W/m²/K]

        # htc enhancement
        h_h[j] = h_h[j] * qhtc_tuning_factor

        # CALCULATE WALL TEMPERATURE
        Tw[j] = Th[j] - (h_c[j] * T_ID * (Th[j] - Tc[j])) / (h_h[j] * T_OD + h_c[j] * T_ID)

        # 3.3. Calculate Stanton numbers
        St_h[j] = h_h[j] / (cp_h[j] * G_h[j])
        St_c[j] = h_c[j] / (cp_c[j] * G_c[j])
        J_h[j] = St_h[j] * Pr_h[j] ** (2 / 3)
        J_c[j] = St_c[j] * Pr_c[j] ** (2 / 3)

        # 4.1. Calculate overall heat transfer analysis
        U_h[j] = 1 / (
            1 / h_h[j] + 1 / h_c[j] * (Aht_hot[j] / Aht_cold[j])
        )  # overall heat transfer coefficient (hot side basis) [W/m²/K]
        U_c[j] = 1 / (
            1 / h_c[j] + 1 / h_h[j] * (Aht_cold[j] / Aht_hot[j])
        )  # overall heat transfer coefficient (cold side basis) [W/m²/K]
        # 4.1.b. Consider wall properties too
        U_h[j] = 1 / (
            1 / h_h[j] + 1 / h_c[j] * (T_OD / T_ID) + T_OD / (2 * k_tube) * np.log(T_OD / T_ID)
        )  # overall heat transfer coefficient (hot side basis) [W/m²/K]

        # 4.2. Calculate heat capacity rates
        C_h[j] = mflow_h * cp_h[j]  # heat capacity rate for hot stream [W/K]
        C_c[j] = mflow_c * cp_c[j]  # heat capacity rate for cold stream [W/K]
        C_min[j] = min(C_h[j], C_c[j])  # minimum heat capacity rate [W/K]
        C_max[j] = max(C_h[j], C_c[j])  # maximum heat capacity rate [W/K]
        C_ratio[j] = C_min[j] / C_max[j]  # capacity ratio [-]

        # 4.3. Calculate number of transfer units
        NTU_h[j] = U_h[j] * Aht_hot[j] / C_min[j]  # number of transfer units [-]
        NTU_c[j] = U_c[j] * Aht_cold[j] / C_min[j]  # number of transfer units [-]
        NTU[j] = NTU_h[
            j
        ]  # forcing this as sometimes NTU_h != NTU_c, can see where if you debug and run without this line (inv. later)

        # 4.3.b. Calculate number of transfer units (KB packages)
        NTU_kb[j] = ntu(St_h[j], St_c[j], Aht_hot[j] / Aff_hot[j], Aht_cold[j] / Aff_cold[j], C_h[j], C_c[j])
        # Tested: identical to original NTU calculation

        # 4.4. Calculate effectiveness (assumes counterflow NTU-eps relation)
        if C_ratio[j] < 1:
            eps[j] = (1 - np.exp(-NTU[j] * (1 - C_ratio[j]))) / (
                1 - C_ratio[j] * np.exp(-NTU[j] * (1 - C_ratio[j]))
            )  # counter flow
            # eps[j] = (1 / C_ratio[j] * (1 - np.exp(-C_ratio[j] * (1 - np.exp(-NTU[j]))))) # cross flow, cmax mixed
        else:
            eps[j] = NTU / (NTU + 1)

        # 4.5. Calculate heat transfer rate
        if i < 1:  # for first iteration, use same nodes for qmax
            q_max[j] = C_min[j] * (Th[j] - Tc[j])  # maximum heat transfer rate [W]
        else:  # after first iteration, use next node for qmax
            if outboard == 0:
                q_max[j] = C_min[j] * (Th[j + 1] - Tc[j])  # maximum heat transfer rate [W]
                # q_max[j] = C_min[j] * (Th[j] - Tc[j])  # maximum heat transfer rate [W]
            elif outboard == 1:
                # q_max[j] = C_min[j] * (Th[j] - Tc[j+1])  # maximum heat transfer rate [W]
                q_max[j] = C_min[j] * (Th[j] - Tc[j])  # maximum heat transfer rate [W]
        q[j] = eps[j] * q_max[j]  # actual heat transfer rate [W]

        # 4.6. Calculate outlet temperatures
        Th[j + 1] = Th[j] + (q[j] * qh_tuning_factor) / C_h[j]  # hot stream outlet temperature [K]
        Tc[j + 1] = Tc[j] + (q[j] * qc_tuning_factor) / C_c[j]  # cold stream outlet temperature [K]
        if outboard == 1:
            Th[j + 1] = Th[j] - (q[j] * qh_tuning_factor) / C_h[j]  # hot stream outlet temperature [K]
            Tc[j + 1] = Tc[j] - (q[j] * qc_tuning_factor) / C_c[j]  # cold stream outlet temperature [K]

        # 5.a. Calculate pressure drop (basic)
        # dP_h[j] = 0.0001 * Ph[j] # hot stream pressure drop [Pa]
        # dP_c[j] = 0.0001 * Pc[j] # cold stream pressure drop [Pa]

        # 5.b. Calculate pressure drop (KB packages, friction only)
        dP_h[j] = (
            dp_friction_only(Aht_hot[j] / Aff_hot[j], G_h[j], 1 / rho_h[j], f_h[j]) * dp_tuning_factor
        )  # !!! check the x4!!!
        dP_c[j] = dp_friction_only(Aht_cold[j] / Aff_cold[j], G_c[j], 1 / rho_c[j], f_c[j])
        Ph[j + 1] = Ph[j] + dP_h[j]  # hot stream outlet pressure [Pa]
        Pc[j + 1] = Pc[j] - dP_c[j]  # cold stream outlet pressure [Pa]
        if outboard == 1:
            Ph[j + 1] = Ph[j] - dP_h[j]  # hot stream outlet pressure [Pa]
            Pc[j + 1] = Pc[j] + dP_c[j]  # cold stream outlet pressure [Pa]

        # 5.c. Calculate pressure drop (KB packages, momentum included)
        rho_h[j + 1] = PropsSI("D", "T", Th[j + 1], "P", Ph[j], fluid_h)
        rho_c[j + 1] = PropsSI("D", "T", Tc[j + 1], "P", Pc[j], fluid_c)
        dP_h[j] = (
            dp_tube_bank(Aht_hot[j] / Aff_hot[j], G_h[j], rho_h[j], rho_h[j + 1], Ah_sigma, f_h[j]) * dp_tuning_factor
        )
        Ph[j + 1] = Ph[j] + dP_h[j]  # hot stream outlet pressure [Pa]
        if outboard == 1:
            Ph[j + 1] = Ph[j] - dP_h[j]  # hot stream outlet pressure [Pa]

        # if you want to print
        if print_results_inner == 1:
            print(f"\nLayer {j} (r = {r[j] * 1000:.1f}mm to {r[j + 1] * 1000:.1f}mm)")
            print(f"Hot stream: mass flux = {G_h[j]:.1f} kg/m²/s, Re = {Re_h[j]:.1f}, Re_OD = {Re_h_OD[j]:.1f}")
            print(f"Cold stream: mass flux = {G_c[j]:.1f} kg/m²/s, Re = {Re_c[j]:.1f}")
            print(f"Nussel numbers - Hot: {Nu_h[j]:.1f}, Cold: {Nu_c[j]:.1f}")
            print(f"Heat transfer coefficients - Hot: {h_h[j]:.0f} W/m²/K, Cold: {h_c[j]:.0f} W/m²/K")
            print(f"Stanton numbers - Hot: {St_h[j]:.4f}, Cold: {St_c[j]:.4f}")
            print(f"Colburn j-factors - Hot: {J_h[j]:.4f}, Cold: {J_c[j]:.4f}")
            print(f"Friction factors - Hot: {f_h[j]:.4f}, Cold: {f_c[j]:.4f}")
            print(f"Overall heat transfer coefficients - Hot: {U_h[j]:.0f} W/m²/K, Cold: {U_c[j]:.0f} W/m²/K")
            print(
                f"Heat capacity rates - Hot: {C_h[j]:.0f} W/K, Cold: {C_c[j]:.0f} W/K, Capacity ratio: {C_ratio[j]:.3f}"
            )
            print(f"Effectiveness: {eps[j]:.3f}, NTU: {NTU[j]:.3f}")
            print(f"Heat transfer - q_max: {q_max[j] / 1000:.1f} kW, q_actual: {q[j] / 1000:.1f} kW")
            print(f"Outlet temperatures - Hot: {Th[j + 1]:.1f} K, Cold: {Tc[j + 1]:.1f} K")
            print(f"Pressure drops - Hot: {dP_h[j] / Ph[j] * 100:.3f}%, Cold: {dP_c[j] / Pc[j] * 100:.3f}%")
            print(f"Outlet pressures - Hot: {Ph[j + 1] / 1e5:.3f}bar, Cold: {Pc[j + 1] / 1e5:.3f}bar")

    # Check if outlet temperatures and pressures are within tolerance
    if outboard == 0:
        error_Th = Th[-1] - Th_in
        error_Ph = Ph[-1] - Ph_in
        error_Tc = Tc[0] - Tc_in
        error_Pc = Pc[0] - Pc_in
    elif outboard == 1:
        error_Th = Th[0] - Th_in
        error_Ph = Ph[0] - Ph_in
        error_Tc = Tc[-1] - Tc_in
        error_Pc = Pc[-1] - Pc_in

    # NEW METHOD
    if outboard == 0:
        if abs(error_Th) < iter_tol_Th:
            # print(f"Converged temperature in {i+1} iterations! Errors: {error_Th:.1f} K, {error_Ph:.1f} Pa ({error_Ph/Ph_in*100:.2f} %)")
            if abs(error_Ph) < iter_tol_Ph:
                # print(f"Converged pressure too in {i+1} iterations! Errors: {error_Th:.1f} K, {error_Ph:.1f} Pa ({error_Ph/Ph_in*100:.2f} %)")
                break
            else:
                Ph_out_guess = Ph_out_guess - 0.1 * error_Ph
        else:
            Th_out_guess = Th_out_guess - 0.1 * error_Th
            if abs(error_Ph) > iter_tol_Ph:
                Ph_out_guess = Ph_out_guess - 0.1 * error_Ph
    elif outboard == 1:
        if abs(error_Tc) < iter_tol_Tc:
            # print(f"Converged temperature in {i+1} iterations! Errors: {error_Th:.1f} K, {error_Ph:.1f} Pa ({error_Ph/Ph_in*100:.2f} %)")
            if abs(error_Pc) < iter_tol_Pc:
                # print(f"Converged pressure too in {i+1} iterations! Errors: {error_Th:.1f} K, {error_Ph:.1f} Pa ({error_Ph/Ph_in*100:.2f} %)")
                break
            else:
                Pc_out_guess = Pc_out_guess - 0.1 * error_Pc
        else:
            Tc_out_guess = Tc_out_guess - 0.1 * error_Tc
            if abs(error_Pc) > iter_tol_Pc:
                Pc_out_guess = Pc_out_guess - 0.1 * error_Pc

# Energy balance check based on enthalpies
if outboard == 0:
    h_h_in = PropsSI("H", "T", Th[-1], "P", Ph[-1], fluid_h)
    h_h_out = PropsSI("H", "T", Th[0], "P", Ph[0], fluid_h)
    h_c_in = PropsSI("H", "T", Tc[0], "P", Pc[0], fluid_c)
    h_c_out = PropsSI("H", "T", Tc[-1], "P", Pc[-1], fluid_c)
elif outboard == 1:
    h_h_in = PropsSI("H", "T", Th[0], "P", Ph[0], fluid_h)  # hot inlet at node 0
    h_h_out = PropsSI("H", "T", Th[-1], "P", Ph[-1], fluid_h)  # hot outlet at node -1
    h_c_in = PropsSI("H", "T", Tc[-1], "P", Pc[-1], fluid_c)  # cold inlet at node -1
    h_c_out = PropsSI("H", "T", Tc[0], "P", Pc[0], fluid_c)  # cold outlet at node 0
Q_h_enthalpy = input_mflow_h_total * (h_h_in - h_h_out)
Q_c_enthalpy = input_mflow_c_total * (h_c_out - h_c_in)

# Calculate overall effectiveness
eps_total = np.sum(q) / (C_min[0] * (Th[-1] - Tc[0]))
eps_total = (Q_h_enthalpy / HX_sectors) / (np.mean(C_min) * (Th[-1] - Tc[0]))
eps_total = (Q_c_enthalpy / HX_sectors) / (np.mean(C_min) * (Th[-1] - Tc[0]))
# eps_total = np.sum(q) / np.sum(q_max)
# Checks for effectiveness
eps_hot_temp = (Th[-1] - Th[0]) / (Th[-1] - Tc[0])
eps_cold_temp = (Tc[-1] - Tc[0]) / (Th[-1] - Tc[0])
eps_hot = C_h[-1] / min(C_h[-1], C_c[0]) * eps_hot_temp
eps_cold = C_c[0] / min(C_h[-1], C_c[0]) * eps_cold_temp

if outboard == 0:
    pressure_drop_hot = (Ph[-1] - Ph[0]) / Ph[-1] * 100
    eps_total = np.sum(q) / (C_min[0] * (Th[-1] - Tc[0]))
    eps_total = (Q_h_enthalpy / HX_sectors) / (np.mean(C_min) * (Th[-1] - Tc[0]))
    eps_total = (Q_c_enthalpy / HX_sectors) / (np.mean(C_min) * (Th[-1] - Tc[0]))
    # Checks for effectiveness
    eps_hot_temp = (Th[-1] - Th[0]) / (Th[-1] - Tc[0])
    eps_cold_temp = (Tc[-1] - Tc[0]) / (Th[-1] - Tc[0])
    eps_hot = C_h[-1] / min(C_h[-1], C_c[0]) * eps_hot_temp
    eps_cold = C_c[0] / min(C_h[-1], C_c[0]) * eps_cold_temp
elif outboard == 1:
    pressure_drop_hot = (Ph[0] - Ph[-1]) / Ph[0] * 100
    eps_total = np.sum(q) / (C_min[0] * (Th[0] - Tc[-1]))
    eps_total = (Q_h_enthalpy / HX_sectors) / (np.mean(C_min) * (Th[0] - Tc[-1]))
    eps_total = (Q_c_enthalpy / HX_sectors) / (np.mean(C_min) * (Th[0] - Tc[-1]))
    # Checks for effectiveness
    eps_hot_temp = (Th[0] - Th[-1]) / (Th[0] - Tc[-1])
    eps_cold_temp = (Tc[0] - Tc[-1]) / (Th[0] - Tc[-1])
    eps_hot = C_h[0] / min(C_h[0], C_c[-1]) * eps_hot_temp
    eps_cold = C_c[-1] / min(C_h[0], C_c[-1]) * eps_cold_temp

# Summary of results
if print_results_outer == 1:
    # print(f"\n--- Enthalpy-based balance check ---")
    # print(f"Hot side Q_h = {Q_h_enthalpy/1e6:.3f} MW")
    # print(f"Cold side Q_c = {Q_c_enthalpy/1e6:.3f} MW")
    # print(f"Imbalance = {(Q_h_enthalpy - Q_c_enthalpy)/Q_h_enthalpy*100:.2f} %\n")

    print(f"Overall effectiveness (accurate, local): {eps_total * 100:.1f}%")
    print(
        f"Hot stream temp. effectiveness: {eps_hot_temp * 100:.1f}%, Cold stream temp. effectiveness: {eps_cold_temp * 100:.1f}%"
    )
    print(f"Hot stream effectiveness (global): {eps_hot * 100:.1f}%, Cold stream effectiveness: {eps_cold * 100:.1f}%")

    print(f"\nSummary of results:")
    print(f"Hot stream:")
    if outboard == 0:
        print(f"  Inlet temperature: {Th[-1]:.1f} K")
        print(f"  Outlet temperature: {Th[0]:.1f} K ({Th[0] - Th[-1]:.1f} K drop)")
        print(f"  Outlet pressure drop: {(Ph[-1] - Ph[0]) / Ph[-1] * 100:.1f}% ({(Ph[-1] - Ph[0]) / 1e3:.2f} kPa)")
    elif outboard == 1:
        print(f"  Inlet temperature: {Th[0]:.1f} K")
        print(f"  Outlet temperature: {Th[-1]:.1f} K ({Th[-1] - Th[0]:.1f} K drop)")
        print(f"  Outlet pressure drop: {(Ph[0] - Ph[-1]) / Ph[0] * 100:.1f}% ({(Ph[0] - Ph[-1]) / 1e3:.2f} kPa)")
    print(f"Cold stream:")
    if outboard == 0:
        print(f"  Inlet temperature: {Tc[0]:.1f} K")
        print(f"  Outlet temperature: {Tc[-1]:.1f} K ({Tc[-1] - Tc[0]:.1f} K rise)")
        print(f"  Outlet pressure drop: {(Pc[0] - Pc[-1]) / Pc[0] * 100:.1f}% ({(Pc[0] - Pc[-1]) / 1e3:.2f} kPa)")
    elif outboard == 1:
        print(f"  Inlet temperature: {Tc[-1]:.1f} K")
        print(f"  Outlet temperature: {Tc[0]:.1f} K ({Tc[0] - Tc[-1]:.1f} K rise)")
        print(f"  Outlet pressure drop: {(Pc[-1] - Pc[0]) / Pc[-1] * 100:.1f}% ({(Pc[-1] - Pc[0]) / 1e3:.2f} kPa)")
    # print(f"Overall effectiveness: {max(eps_hot_temp, eps_cold_temp)*100:.1f}%")
    print(f"Overall effectiveness: {eps_total * 100:.1f}%\n")
    # if preset_AHJEB_H2TOCv2_ExPHT == 1:
    # print(f"Radial and axial spaces used: {(input_HX_rmax-input_HX_rmin)/dR_available*100:.1f}%, {input_HX_dX/dX_available*100:.1f}%\n")

# Print runtime excluding plotting
t_run = time.perf_counter() - t0
print(f"Single run runtime: {t_run:.3f} s")

# Plot temperature profiles
if plot_results_outer == 1:
    plt.figure(figsize=(12, 8))
    plt.title(
        (
            f"Temperature profiles {fluid_h} hot, {fluid_c} cold, "
            f"{input_mflow_h_total:.1f} kg/s hot, {input_mflow_c_total:.1f} kg/s cold, {mflow_c:.3f} kg/s per layer\n"
            f"{input_Th_in:.1f} K hot inlet ({input_Ph_in / 1e5:.1f} bar), {input_Tc_in:.1f} K cold inlet ({input_Pc_in / 1e5:.1f} bar), "
            f"{input_HX_rmin * 1000:.1f} mm inner radius, {input_HX_rmax * 1000:.1f} mm outer radius\n"
            f"Tube OD = {input_T_OD * 1000:.1f} mm, Tube t = {input_T_t * 1000:.3f} mm, Tube sts = {input_T_sts}, Tube sls = {input_T_sls}, Total number of tubes = {T_arows * T_rrows}\n"
            f"Number of headers = {input_H_number}, Number of rows = {input_H_rrows}, Involute angle = {Inv_angle:.0f}°, Mean J/f = {np.mean(J_h / f_h):.3f} & Re_OD = {np.mean(Re_h_OD):.0f}\n"
            f"Effectiveness (global) = {eps_total * 100:.1f}%, Hot side pressure drop = {pressure_drop_hot:.1f}%, Mass of tubes = {mass_tubes:.1f} kg"
        )
    )
    plt.plot(r * 1000, Th, "r-o", linewidth=2, markersize=4, label="Hot stream")
    plt.plot(r * 1000, Tc, "b-o", linewidth=2, markersize=4, label="Cold stream")
    plt.axhline(Tc_in, color="b", linestyle="--", linewidth=2, label="Cold stream inlet")
    plt.axhline(Th_in, color="r", linestyle="--", linewidth=2, label="Hot stream inlet")
    plt.axhline(Tc_in_target, color="b", linestyle=":", linewidth=2, label="Cold stream outlet target")
    plt.axhline(Th_out_target, color="r", linestyle=":", linewidth=2, label="Hot stream outlet target")
    plt.xlabel("Radial location [mm]")
    plt.ylabel("Temperature [K]")
    plt.legend(loc="upper left")
    # Adding a second y-axis for hot side pressure drop in %
    ax2 = plt.gca().twinx()
    if outboard == 0:
        ax2.plot(
            r * 1000,
            (Ph / Ph[-1] - 1) * 100,
            "k-x",
            linewidth=2,
            markersize=4,
            label="Hot side pressure drop (%)",
        )
    elif outboard == 1:
        ax2.plot(
            r * 1000,
            (Ph / Ph[0] - 1) * 100,
            "k-x",
            linewidth=2,
            markersize=4,
            label="Hot side pressure drop (%)",
        )
    ax2.set_ylabel("Hot side pressure drop [%]")
    ax2.set_ylim(top=0.2)  # Adjust the y-axis limits as needed
    ax2.legend(loc="center left")
    plt.show()

# ==== Entropy Generation Calculation ====
if calc_entropy == 1:
    Sgen_total_thermal = 0.0
    Sgen_total_both = 0.0
    Sgen_total_cool = 0.0
    Sgen_local_thermal = []
    Sgen_local_both = []
    Sgen_local_cool = []
    for j in range(len(Th) - 1):
        # Thermal entropy generation
        dS_h_thermal = input_mflow_h_total * cp_h[j] * np.log(Th[j] / Th[j + 1])
        dS_c_thermal = input_mflow_c_total * cp_c[j] * np.log(Tc[j + 1] / Tc[j])
        dS_gen_thermal = dS_h_thermal + dS_c_thermal
        Sgen_local_thermal.append(dS_gen_thermal)
        Sgen_total_thermal += dS_gen_thermal
        # Including pressure drop
        dS_h_both = input_mflow_h_total * (
            PropsSI("S", "T", Th[j], "P", Ph[j], fluid_h) - PropsSI("S", "T", Th[j + 1], "P", Ph[j + 1], fluid_h)
        )
        dS_c_both = input_mflow_c_total * (
            PropsSI("S", "T", Tc[j + 1], "P", Pc[j + 1], fluid_c) - PropsSI("S", "T", Tc[j], "P", Pc[j], fluid_c)
        )
        dS_gen_both = dS_h_both + dS_c_both
        Sgen_local_both.append(dS_gen_both)
        Sgen_total_both += dS_gen_both
        Sgen_local_cool.append(dS_c_both)
        Sgen_total_cool += dS_c_both

    # ==== Pinch Point Calculation ====
    deltaT_profile = Th - Tc
    pinch_point = np.min(deltaT_profile)

    # ==== Exergy (Second-Law) Efficiency ====
    # Example: Set flight altitude in feet and convert to meters
    flight_altitude_ft = 39000  # flight altitude in feet
    flight_altitude_m = flight_altitude_ft * 0.3048  # convert feet to meters
    flight_altitude_m = 11000
    # Calculate reference environment temperature (T0) based on altitude
    # Use International Standard Atmosphere (ISA) lapse rate for troposphere (up to 11,000 m)
    # T0 = 288.15 - 0.0065 * h (h in meters), but for altitudes above 11,000 m, T0 is constant at 216.65 K
    if flight_altitude_m <= 11000:
        T0 = 288.15 - 0.0065 * flight_altitude_m
    elif flight_altitude_m > 11000 and flight_altitude_m <= 20000:
        T0 = 216.65  # isothermal stratosphere
    elif flight_altitude_m > 20000:
        print("Altitude above 20,000 m not supported")
        exit()
    P0 = 101325 * (1 - 0.0065 * flight_altitude_m / 288.15) ** 5.2561
    # T0 = 298.0  # Reference environment temperature [K]
    Q_total = np.sum(q) * H_number  # total heat transferred [W]
    eta_II_both = 1 - (T0 * Sgen_total_both) / Q_total
    eta_II_thermal = 1 - (T0 * Sgen_total_thermal) / Q_total
    eta_II_cool = 1 - (T0 * Sgen_total_thermal) / (Q_total - Sgen_total_cool)

    if print_results_entropy == 1:
        # print(f"Total entropy generation (thermal): {Sgen_total_thermal:.1f} W/K")
        print(f"Total entropy generation (including pressure drop): {Sgen_total_both:.1f} W/K")
        # print(f"Total entropy generation (cooling): {Sgen_total_cool:.1f} W/K")
        print(f"Pinch point (minimum ΔT): {pinch_point:.3f} K")
        print(f"Flight altitude: {flight_altitude_ft} feet ({flight_altitude_m:.1f} m)")
        print(f"Reference environment temperature (T0) at altitude: {T0:.2f} K")
        print(f"Reference environment pressure (P0) at altitude: {P0 / 1e5:.3f} bar")
        print(f"Second-law (exergy) efficiency: {eta_II_both * 100:.2f} %")
        # print(f"Second-law (exergy) efficiency (thermal): {eta_II_thermal*100:.2f} %")
        # print(f"Second-law (exergy) efficiency (cooling): {eta_II_cool*100:.2f} %")

    # Optional: plot entropy generation vs. radius/layer
    if plot_results_entropy == 1:
        fig, ax1 = plt.subplots()
        # Plot entropy generation
        ax1.plot(
            r[1:] * 1000,
            Sgen_local_thermal,
            "-x",
            color="b",
            label="Local entropy generation (thermal) [W/K]",
        )
        ax1.plot(
            r[1:] * 1000,
            Sgen_local_both,
            "-o",
            color="b",
            label="Local entropy generation (both) [W/K]",
        )
        ax1.set_xlabel("Radial position [mm]")
        ax1.set_ylabel("Local entropy generation [W/K]", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        plt.legend(loc="upper left")
        ax1.grid(True)
        # Create a second y-axis for deltaT_profile
        ax2 = ax1.twinx()
        ax2.plot(r[0:] * 1000, deltaT_profile, "-x", color="r", label="Temperature Difference [K]")
        ax2.set_ylabel("Temperature Difference [K]", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        plt.title("Entropy Generation Distribution and Temperature Difference in HX")
        fig.tight_layout()  # To ensure the right y-label is not slightly clipped
        plt.show()

if calc_tsfc == 1:
    # mflow splits
    m_core = 60.3
    BPR = 14.15
    m_bypass = BPR * m_core
    m_combustor = 48.2  # air mass flow into combustor after accounting for cooling bleed air
    m_split_core = 0.8 * m_core
    m_split_hx = 0.2 * m_core

    # states for cycle
    p0 = P0
    s0 = PropsSI("S", "T", T0, "P", p0, fluid_h)  # T0 is static, others total I think.. need to confirm
    T1 = 248
    p1 = 0.362e5
    s1 = PropsSI("S", "T", T1, "P", p1, fluid_h)
    T2 = 907
    p2 = 25.37e5
    s2 = PropsSI("S", "T", T2, "P", p2, fluid_h)
    T3 = 1610
    # p3 = 21.2e5 # actual
    p3 = p2  # assume no combustor losses
    s3 = PropsSI("S", "T", T3, "P", p3, fluid_h)
    T4 = 575
    p4 = 0.368e5
    s4 = PropsSI("S", "T", T4, "P", p4, fluid_h)
    T4h2 = Th[0]
    # p4h2 = Ph[0] # need to fix this..
    p4h2 = p4 * (1 - (Ph[-1] - Ph[0]) / Ph[-1])
    # p4h2 = p4*0.95
    s4h2 = PropsSI("S", "T", T4h2, "P", p4h2, fluid_h)

    # isobar curves
    qT = np.linspace(0, 2000, 100)  # 0-2000 K queries
    qS0 = PropsSI("S", "T", qT, "P", p0, fluid_h)
    qS1 = PropsSI("S", "T", qT, "P", p1, fluid_h)
    qS2 = PropsSI("S", "T", qT, "P", p2, fluid_h)
    # cycle curve
    s_cycle = np.array(
        [s0] + [s1] + [PropsSI("S", "T", t, "P", p2, fluid_h) for t in np.linspace(T2, T3, 10)] + [s4] + [s4h2]
    )
    T_cycle = np.array([T0] + [T1] + list(np.linspace(T2, T3, 10)) + [T4] + [T4h2])

    Mflight = 0.85  # flight Mach number
    V0 = Mflight * np.sqrt(1.4 * 287 * T0)

    def calc_vjet(p0in, T0in):
        # p0 = 0.226 bar, atm static pressure
        gamma = 1.4
        pr_crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        if p0 / p0in < pr_crit:  # flow choked
            pe = p0in * pr_crit
            Me = 1
            Te = T0in / (1 + (gamma - 1) / 2)
            Ve = np.sqrt(gamma * 287 * Te)
        else:
            pe = p0
            Me = np.sqrt(2 / (gamma - 1) * ((p0 / p0in) ** ((gamma - 1) / (-gamma)) - 1))
            Te = T0in / (1 + (gamma - 1) / 2 * Me**2)
            ae = np.sqrt(gamma * 287 * Te)
            Ve = Me * ae
        return Ve, pe, Te

    V4, p4e, T4e = calc_vjet(p4, T4)
    V4h2, p4h2e, T4h2e = calc_vjet(p4h2, T4h2)
    pbp = 0.45e5
    Tbp = 276
    Vbp, pbpe, Tbpe = calc_vjet(pbp, Tbp)

    # rough method for example
    # V4 = np.sqrt(2 * PropsSI('C','T',T4,'P',p4,fluid_h) * (T4-T0))
    # V4h2 = np.sqrt(2 * PropsSI('C','T',T4,'P',p4h2,fluid_h) * (T4h2-T0))
    # Vbp = np.sqrt(2 * PropsSI('C','T',Tbp,'P',Tbp,fluid_h) * (Tbp-T0))

    # area splits from JY
    Abypass = 6.78  # bypass area [m2]
    Acore = 1.024  # core area [m2]
    Asplit_main = 0.816  # main split area [m2]
    Asplit_hx = 0.184  # hx split area [m2]

    # thrust calculations
    Fnet_bypass = m_bypass * (Vbp - V0) + Abypass * (pbpe - p0)
    Fnet_baseline = m_core * (V4 - V0) + Acore * (p4e - p0)
    Fnet_preheated = (
        m_split_core * (V4 - V0) + m_split_hx * (V4h2 - V0) + Asplit_main * (p4e - p0) + Asplit_hx * (p4h2e - p0)
    )
    Fnet_total_baseline = Fnet_baseline + Fnet_bypass
    Fnet_total_preheated = Fnet_preheated + Fnet_bypass
    # thrust changes
    dFnet_core = 1 - Fnet_preheated / Fnet_baseline
    dFnet_total = 1 - Fnet_total_preheated / Fnet_total_baseline

    fuel_LHV = 120e6  # lower heating value of hydrogen, J/kg
    # baseline
    heat_addition = m_combustor * (
        T3 * PropsSI("C", "T", T3, "P", p3, fluid_h) - T2 * PropsSI("C", "T", T2, "P", p2, fluid_h)
    )
    fuel_massflow = heat_addition / fuel_LHV  # mf*LCV = m_combustor*heat_addition, tf mf in kg/s
    fuel_massflow_frac = fuel_massflow / m_combustor
    tsfc_baseline_core = fuel_massflow / Fnet_baseline
    tsfc_baseline_total = fuel_massflow / Fnet_total_baseline

    # preheated
    Tc_inlet = 40
    H2_q = Tc[-1] * PropsSI("C", "T", Tc[-1], "P", Pc[-1], fluid_c) - Tc_inlet * PropsSI(
        "C", "T", Tc_inlet, "P", Pc[0], fluid_c
    )
    fuel_massflow_preheated = heat_addition / (fuel_LHV + H2_q)  # mf*LCV = m_combustor*heat_addition, tf mf in kg/s
    heat_frac = H2_q / fuel_LHV
    fuel_massflow_preheated_frac = fuel_massflow_preheated / m_combustor
    tsfc_preheated_core = fuel_massflow_preheated / Fnet_preheated
    tsfc_preheated_total = fuel_massflow_preheated / Fnet_total_preheated

    # change in tsfc and fuel flow
    d_fuel_massflow = (fuel_massflow_preheated / fuel_massflow - 1) * 100
    d_tsfc_core = (tsfc_preheated_core / tsfc_baseline_core - 1) * 100
    d_tsfc_total = (tsfc_preheated_total / tsfc_baseline_total - 1) * 100

    if print_results_tsfc == 1:
        print(f"V0: {V0:.0f} m/s, V4: {V4:.0f} m/s, V4h2: {V4h2:.0f} m/s, Vbp: {Vbp:.0f} m/s")
        print(f"T0: {T0:.0f} K, T4e: {T4e:.0f} K, T4h2e: {T4h2e:.0f} K, Tbpe: {Tbpe:.0f} K")
        print(
            f"Fnet_bypass: {Fnet_bypass / 1e3:.0f} kN, Fnet_baseline: {Fnet_baseline / 1e3:.0f} kN, Fnet_preheated (split): {Fnet_preheated / 1e3:.0f} kN"
        )
        print(
            f"Change in thrust, core only (exc. bypass): {dFnet_core * 100:.0f}%, total (inc. bypass): {dFnet_total * 100:.0f}%"
        )
        print(
            f"fuel/air: {fuel_massflow_frac * 100:.2f}%, fuel mass flow: {fuel_massflow:.2f} kg/s, tsfc_baseline_core: {tsfc_baseline_core:.2f} kg/s/N, tsfc_baseline_total: {tsfc_baseline_total:.2f} kg/s/N"
        )
        print(
            f"fuel/air: {fuel_massflow_preheated_frac * 100:.2f}%, fuel mass flow: {fuel_massflow_preheated:.2f} kg/s, tsfc_baseline_core: {tsfc_preheated_core:.2f} kg/s/N, tsfc_baseline_total: {tsfc_preheated_total:.2f} kg/s/N"
        )
        print(f"fraction of sensible heat pick up to fuel LHV: {heat_frac * 100:.2f}%")
        print(
            f"Change in fuel mass flow: {d_fuel_massflow:.2f}%, change in core tsfc: {d_tsfc_core:.2f}%, change in total tsfc (inc. bypass): {d_tsfc_total:.2f}%"
        )

    # pressure drop sensitivity study, constant T4h2, vary p4h2 and plot dV (V4-V0)
    n_samples = 20
    qp4h2 = np.zeros(n_samples)
    qV4h2 = np.zeros(n_samples)
    for jj in range(n_samples):
        qp4h2[jj] = p4 * (1 - jj * 0.025)
        qV4h2[jj] = calc_vjet(qp4h2[jj], T4h2)[0]
    dVel = qV4h2 - V0

    if plot_results_tsfc == 1:
        plt.figure()
        plt.title("T-s diagram")
        plt.scatter([s0, s1, s2, s3, s4, s4h2], [T0, T1, T2, T3, T4, T4h2], c="black", marker="o")
        plt.plot(s_cycle, T_cycle, c="black", label="Core + Split streams")
        plt.plot(qS0, qT, c="black", linestyle="--", linewidth=0.5, label="Atm")
        plt.plot(qS1, qT, c="black", linestyle="--", linewidth=0.5, label="ramPressure")
        plt.plot(qS2, qT, c="black", linestyle="--", linewidth=0.5, label="CPR")
        plt.xlabel("Entropy")
        plt.ylabel("Temperature")
        plt.xlim(round(s1 - 500, -2), round(s4 + 500, -2))
        plt.ylim(0, 2000)
        plt.legend()

        fig, ax1 = plt.subplots(figsize=(7, 5))
        xvals = (1 - qp4h2 / p4) * 100
        ax1.set_title("Pressure drop sensitivity (fixed air-side temp. drop) \nDuct-B at TOC")
        ax1.set_xlabel("Air-side pressure drop [%]")
        ax1.set_ylabel(r"Jet velocity excess $(V_{\mathrm{EXIT,HX}}-V_{\mathrm{flight}})$ [m/s]", color="black")
        (l1,) = ax1.plot(xvals, dVel, color="black", label="dVel", linewidth=2)
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(True, color="black")  # Add grid lines to the primary axis
        # Second y-axis (right)
        qFnet_HX = m_split_hx * dVel
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"HX exhaust net thrust ($F_{\mathrm{NET,HX}}$) [N]", color="blue")
        (l2,) = ax2.plot(xvals, qFnet_HX, color="blue", label="Thrust change", alpha=0)
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.grid(True, color="blue")  # Add grid lines to the second axis
        # Third y-axis (right, offset, flipped)
        qFnet_preheated = m_split_core * (V4 - V0) + qFnet_HX
        qd_tsfc = ((fuel_massflow_preheated / (qFnet_preheated + Fnet_bypass)) / tsfc_baseline_total - 1) * 100
        ax3 = ax1.twinx()
        # Offset the third axis further to the right
        ax3.spines["right"].set_position(("axes", 1.25))
        ax3.spines["right"].set_visible(True)
        ax3.set_ylabel("tsfc change from baseline [%]", color="tab:red")
        (l3,) = ax3.plot(
            xvals, qd_tsfc, color="tab:red", linestyle="--", label="tsfc change", alpha=0
        )  # Make the line invisible
        ax3.tick_params(axis="y", labelcolor="tab:red")
        ax3.invert_yaxis()  # Flip the third y axis
        ax3.grid(True, color="red")  # Add grid lines to the third axis and make them red
    plt.tight_layout()

# Collect results for return
results = {}
if outboard == 0:
    results["Th_in, K"] = round(Th[-1])
    results["Th_out, K"] = round(Th[0])
    results["Tc_in, K"] = round(Tc[0])
    results["Tc_out, K"] = round(Tc[-1])
    results["Ph_in, bar"] = round(Ph[-1] / 1e5, 1)
    results["Ph_out, bar"] = round(Ph[0] / 1e5, 1)
    results["Pc_in, bar"] = round(Pc[0] / 1e5, 1)
    results["Pc_out, bar"] = round(Pc[-1] / 1e5, 1)
    results["pressure_drop_hot, %"] = round((Ph[-1] - Ph[0]) / Ph[-1] * 100, 1)
    results["pressure_drop_cold, %"] = round((Pc[0] - Pc[-1]) / Pc[0] * 100, 1)
elif outboard == 1:
    results["Th_in, K"] = round(Th[0])
    results["Th_out, K"] = round(Th[-1])
    results["Tc_in, K"] = round(Tc[-1])
    results["Tc_out, K"] = round(Tc[0])
    results["Ph_in, bar"] = round(Ph[0] / 1e5, 1)
    results["Ph_out, bar"] = round(Ph[-1] / 1e5, 1)
    results["Pc_in, bar"] = round(Pc[-1] / 1e5, 1)
    results["Pc_out, bar"] = round(Pc[0] / 1e5, 1)
    results["pressure_drop_hot, %"] = round((Ph[0] - Ph[-1]) / Ph[0] * 100, 1)
    results["pressure_drop_cold, %"] = round((Pc[-1] - Pc[0]) / Pc[-1] * 100, 1)
results["effectiveness, %"] = round(eps_total * 100)
results["total_heat_transfer, kW"] = round(np.sum(q) * H_number / 1e3, 1)
results["Q_h_enthalpy, kW"] = round(Q_h_enthalpy / 1e3, 1)
results["Q_c_enthalpy, kW"] = round(Q_c_enthalpy / 1e3, 1)
results["mass_tubes, kg"] = round(mass_tubes)
results["entropy_generation, W/K"] = round(Sgen_total_both) if calc_entropy == 1 else None
results["pinch_point, K"] = round(pinch_point) if calc_entropy == 1 else None
results["exergy_efficiency, %"] = round(eta_II_both * 100) if calc_entropy == 1 else None
# 'tsfc_baseline_core': tsfc_baseline_core if calc_tsfc == 1 else None,
# 'tsfc_baseline_total': tsfc_baseline_total if calc_tsfc == 1 else None,
# 'tsfc_preheated_core': tsfc_preheated_core if calc_tsfc == 1 else None,
# 'tsfc_preheated_total': tsfc_preheated_total if calc_tsfc == 1 else None,
results["d_fuel_massflow, %"] = round(d_fuel_massflow, 1) if calc_tsfc == 1 else None
results["d_tsfc_core, %"] = round(d_tsfc_core, 1) if calc_tsfc == 1 else None
results["d_tsfc_total, %"] = round(d_tsfc_total, 1) if calc_tsfc == 1 else None

# Convert numpy float64 values to standard Python floats for easier handling
for key, value in results.items():
    if isinstance(value, np.float64):
        results[key] = float(value)

if print_results == 1:
    print(results)

""" Debugging, only hot side! """

print(f"Number of iterations: {i + 1} of max {iter_max}")

print(eps)

print(f"Spacing, T_sts, T_sls: {T_sts:.2f}, {T_sls:.2f}")
print(f"    Rmax, Rmin, dR: {HX_rmax * 1e3:.2f}, {HX_rmin * 1e3:.2f}, {HX_dR * 1e3:.2f} mm")
print(f"    Inlet pressure: {Ph[-1] / 1e5:.3f} bar, Outlet pressure: {Ph[0] / 1e5:.3f} bar")
print(f"    T_htA_hot: {T_htA_hot:.4f}, T_htA_hot_total: {T_htA_hot_total:.0f}, Involute length: {Inv_length:.2f} m")
print(f"    Tube numbers: {T_num:.0f}")
print(
    f"    sigma: {Ah_sigma:.2f}, sigma_norm: {Ah_sigma_norm:.2f}, sigma_diag: {Ah_sigma_diag:.2f}, sigma_He: {Ah_sigma_He[-1]:.2f}"
)
print(f"    Free-flow area: {Aff_hot[-1]:.4f}, Dh: {Dh_hot[-1] * 1e3:.2f} mm, Dh_OD: {Dh_hot_OD[-1] * 1e3:.2f} mm")
print(f"    Mass flux: {G_h[-1]:.1f}, Re: {Re_h[-1]:.1f}, Re_OD: {Re_h_OD[-1]:.1f}")
print(
    f"    Mean Vel, rho, mu, cp, Pr, and k in HX: {np.mean(Vel_h):.2f}, {np.mean(rho_h):.3f}, {np.mean(mu_h):.2e}, {np.mean(cp_h):.0f}, {np.mean(Pr_h):.3f}, {np.mean(k_h):.2f}"
)
print(f"    h_h: {h_h[-1]:.0f}, j_h: {J_h[-1]:.3f}, f_h: {f_h[-1]:.3f}, j/f: {J_h[-1] / f_h[-1]:.3f}")
print(f"    NTU: {NTU[-1]:.3f}")
print(
    f"    Mean h_h: {np.mean(h_h):.0f}, Mean j_h: {np.mean(J_h):.3f}, Mean NTU: {np.mean(NTU):.3f}, Mean St_h: {np.mean(St_h):.4f}"
)
print(f"    Eps_total: {eps_total * 100:.2f}%, dP_h: {pressure_drop_hot:.1f}%, Q_h: {Q_h_enthalpy / 1e3:.1f} kW")
print(f"    Pinch point: {pinch_point:.1f} K")
print(f"    Aht: {Aht_hot[-1]:.3f}, Afr: {Afr_hot[-1]:.3f}, Aff: {Aff_hot[-1]:.3f}")
