"""
Reusable involute HX solver and study tools.

Notes:
- This re-implements the same equations and method used in
  `analysis/RInvolute_Design_Ver07_Outboard.py`, but organized
  as functions with two modes:
  1) debug=True: prints and plots akin to the original script
  2) debug=False: input-output mode returning summary metrics

- A sweep utility is provided to vary HX_rmax while keeping Inv_b constant
  (i.e., same angle change per radius), recomputing the involute angle for
  each case. Intended for plotting epsilon–NTU curves.

Dependencies: numpy, matplotlib, CoolProp, and project packages
  `heat_exchanger.hex_basic` and `heat_exchanger.correlations`.
"""

from typing import Dict, Any, List, Tuple

import numpy as np
import time
from matplotlib import pyplot as plt
from heat_exchanger.hex_basic import dp_friction_only, dp_tube_bank, ntu
from heat_exchanger.correlations import (
    tube_bank_nusselt_number_and_friction_factor,
    circular_pipe_nusselt,
    circular_pipe_friction_factor,
)
from heat_exchanger.conservation import update_static_properties
from heat_exchanger.epsilon_ntu import epsilon_ntu
from heat_exchanger.fluid_properties import CoolPropProperties, PerfectGasProperties


def get_preset_params(case: str = "viper") -> Dict[str, Any]:
    """Return a dictionary of parameters for a named case.

    Currently supports a VIPER baseline case; add more as needed.
    """
    case = case.lower()
    if case == "viper":
        # Baseline VIPER (mirrors the preset_viper block in colleague's script)
        params = {
            "fluid_h": PerfectGasProperties(molecular_weight=28.97, mu_ref=1.716e-5, T_ref=273.0, S=111.0),
            "fluid_c": PerfectGasProperties(molecular_weight=4.003, mu_ref=1.9e-5, T_ref=273.0, S=79.4),
            # "fluid_h": CoolPropProperties("air"),  # "air",
            # "fluid_c": CoolPropProperties("helium"),  # "helium",
            # Flow conditions
            "Th_in": 298.0,
            "Ph_in": 1.02e5,
            "Tc_in": 96.0,
            "Pc_in": 150e5,
            # Tube specifications
            "T_OD": 0.98e-3,
            "T_t": 0.04e-3,  # wall thickness, one side
            # Dimensionless tube Spacings
            "T_sts": 2.5,
            "T_sls": 1.1,
            "staggered": True,
            # Headers and involute
            "H_number": 31,  # number of headers/modules
            "H_rrows": 4,  # rows per header
            "Inv_angle_deg": 360.0,
            # Geometry
            "HX_rmax": 2.5 * 478e-3,
            # rmin = rmax - H_rrows * H_number * T_sls * T_OD
            # matches colleague script logic for VIPER
            # Computed below when assembling derived values
            # Axial length
            "HX_dX": None,  # calculated below consistent with VIPER block
            # Targets (unused in IO mode, used in debug plots)
            "Tc_in_target": 289.0,
            "Th_out_target": 135.0,
            # Material
            "k_tube": None,  # use Inconel 718 equivalent from colleague script
            "k_rho": 7930,  # fallback density; not used unless mass estimates are needed
            # Mass flows
            "mflow_h_total": 12.26,  # 1.19 * 7.0 * (2 * np.pi * 478e-3) * 200 * 0.98e-3 * 2.5
            "mflow_c_total": 1.945,  # 0.15868 * 12.26,
            # "inlet_vel": 7.0,  # used to compute hot mass flow as in colleague script
            # Mode: outboard/inboard (VIPER uses inboard in that block, but solver supports both)
            "outboard": 0,
            # Switches
            "input_rect_HX": 0,  # 1 for rectangular HX, 0 for annular HX
        }

        # Derived fields to match colleague's VIPER setup
        T_OD = params["T_OD"]
        T_t = params["T_t"]
        T_ID = T_OD - 2 * T_t
        params["T_ID"] = T_ID
        params["Dh_c"] = T_ID

        H_number = params["H_number"]
        H_rrows = params["H_rrows"]
        T_sls = params["T_sls"]
        HX_rmax = params["HX_rmax"]

        HX_rmin = HX_rmax - H_rrows * H_number * T_sls * T_OD
        params["HX_rmin"] = HX_rmin
        params["HX_dX"] = 200 * T_OD * params["T_sts"]  # per colleague script

        # Mass flows like colleague's VIPER block
        # inlet_vel = params["inlet_vel"]
        # rho_air_nom = 1.19  # as used implicitly via colleague's mflow formula
        # params["mflow_h_total"] = rho_air_nom * inlet_vel * (2 * np.pi * HX_rmax) * params["HX_dX"]
        # params["mflow_c_total"] = 0.15868 * params["mflow_h_total"]
        # Material conductivity representative (Inconel 718)
        params["k_tube"] = 14.0

        return params
    else:
        raise ValueError(f"Unsupported case preset: {case}")


def _solve_involute_hx(params: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """Core solver replicating equations/method of colleague's script.

    Returns a dictionary with summary results.
    If debug=True, prints summary and renders plots similar to original.
    """
    # Unpack inputs
    fluid_h = params["fluid_h"]
    fluid_c = params["fluid_c"]

    Th_in = float(params["Th_in"])  # K
    Ph_in = float(params["Ph_in"])  # Pa
    Tc_in = float(params["Tc_in"])  # K
    Pc_in = float(params["Pc_in"])  # Pa

    mflow_h_total = float(params["mflow_h_total"])  # kg/s
    mflow_c_total = float(params["mflow_c_total"])  # kg/s

    T_OD = float(params["T_OD"])  # m
    T_t = float(params["T_t"])  # m
    T_ID = float(params["T_ID"])  # m

    T_sts = float(params["T_sts"])  # -
    T_sls = float(params["T_sls"])  # -
    staggered = bool(params["staggered"])  # True/False

    H_number = int(params["H_number"])  # -
    H_rrows = int(params["H_rrows"])  # -
    Inv_angle_deg = float(params.get("Inv_angle_deg", 360.0))  # deg

    HX_rmin = float(params["HX_rmin"])  # m
    HX_rmax = float(params["HX_rmax"])  # m
    HX_dX = float(params["HX_dX"])  # m

    outboard = int(params.get("outboard", 0))  # 0 inboard, 1 outboard
    input_rect_HX = int(params["input_rect_HX"])  # 1 for rectangular HX, 0 for annular HX

    # Tuning & modes
    qh_tuning_factor = float(params.get("qh_tuning_factor", 1.0))
    qc_tuning_factor = float(params.get("qc_tuning_factor", 1.0))
    dp_tuning_factor = float(params.get("dp_tuning_factor", 1.0))
    qhtc_tuning_factor = float(params.get("qhtc_tuning_factor", 1.0))

    # Material conductivity (tube wall)
    k_tube = float(params.get("k_tube", 14.0))

    # Geometry and arrays (match colleague script flow)
    HX_dR = abs(HX_rmax - HX_rmin)
    HX_area = np.pi * (HX_rmax**2 - HX_rmin**2)
    if input_rect_HX == 1:
        HX_area = HX_dR * HX_rmax

    HX_sectors = H_number
    Inv_angle = Inv_angle_deg
    Inv_sector_angle = Inv_angle / HX_sectors
    Inv_b = (HX_rmax - HX_rmin) / np.deg2rad(Inv_angle_deg)

    # Microtube arrangement
    T_st = T_sts * T_OD
    T_sl = T_sls * T_OD
    if staggered:
        T_sd = np.sqrt(T_sl**2 + (0.5 * T_st) ** 2)
        T_sds = T_sd / T_OD

    T_rrows = int(round(HX_dR / T_sl))
    T_arows = int(round(HX_dX / T_st))
    T_num = T_rrows * T_arows

    # Heat transfer areas (global single-tube)
    if input_rect_HX == 1:
        T_htA_hot = np.pi * T_OD * HX_dX
        T_htA_cold = np.pi * T_ID * HX_dX
    else:
        # Use full involute length as single-tube area proxy; local layer areas computed later
        theta_vals = np.linspace(0, np.deg2rad(Inv_angle), 50)
        r_vals = HX_rmin + Inv_b * theta_vals
        Inv_length = float(np.trapezoid(np.sqrt(r_vals**2 + Inv_b**2), theta_vals))
        T_htA_hot = np.pi * T_OD * Inv_length
        T_htA_cold = np.pi * T_ID * Inv_length

    # Free area ratios (global)
    Ah_sigma = (T_st - T_OD) / T_st
    if staggered:
        Ah_sigma_norm = (T_st - T_OD) / T_st
        Ah_sigma_diag = 2 * (T_sd - T_OD) / T_st
        Ah_sigma = min(Ah_sigma_norm, Ah_sigma_diag)
    Ac_sigma = np.pi * T_ID**2 / (4 * T_st * T_sl)

    # Mass flow splits
    mflow_h = mflow_h_total / HX_sectors
    mflow_c_header = mflow_c_total / H_number
    mflow_c = mflow_c_header  # per layer approach per colleague script

    # Discretization: combine H_rrows in one radial layer (as in colleague script)
    n = int(round((T_rrows) / H_rrows))
    dR_step = HX_dR / n

    # Geometry nodes (static across outer iterations)
    r = np.zeros(n + 1)
    theta = np.zeros(n + 1)
    for j in range(n + 1):
        r[j] = HX_rmin + j * dR_step
        theta[j] = (r[j] - HX_rmin) / Inv_b

    # Outer shooting loop to match boundary conditions
    outer_max_iter = int(params.get("outer_max_iter", 15))
    outer_tol_T = float(params.get("outer_tol_T", 1.0))  # K
    outer_rel_tol_p = float(params.get("outer_rel_tol_p", 1e-3))  # fraction of inlet P
    guess_hot_boundary = outboard == 0
    guess_cold_boundary = outboard == 1
    if guess_hot_boundary:
        Th_out_guess = 0.5 * Th_in
        Ph_out_guess = 0.97 * Ph_in
        # allow external initial guesses
        init_hot = params.get("outer_init_hot")
        if init_hot:
            Th_out_guess = float(init_hot.get("T", Th_out_guess))
            Ph_out_guess = float(init_hot.get("P", Ph_out_guess))
    if guess_cold_boundary:
        # initial cold outlet guess at node 0 (will shoot to match Tc_in at node -1)
        Tc_out_guess = 0.9 * Th_in
        Pc_out_guess = 0.97 * Pc_in
        # allow external initial guesses
        init_cold = params.get("outer_init_cold")
        if init_cold:
            Tc_out_guess = float(init_cold.get("T", Tc_out_guess))
            Pc_out_guess = float(init_cold.get("P", Pc_out_guess))

    # 0D two-step initial guess (counterflow, average properties)
    try:
        # Frontal/free areas at mid radius and total cold frontal area
        r_mid = 0.5 * (HX_rmin + HX_rmax)
        Afr_hot_mid = (HX_dX * 2 * np.pi * r_mid / HX_sectors) if input_rect_HX == 0 else (HX_dX * HX_rmin)
        Aff_hot_mid = max(Afr_hot_mid * Ah_sigma, 1e-12)
        Afr_cold_total = HX_dX * (HX_rmax - HX_rmin)
        Aff_cold_total = max(Afr_cold_total * Ac_sigma, 1e-12)

        # Total areas based on tube counts
        T_num0 = max(T_rrows * T_arows, 1)
        A_total_hot0 = T_htA_hot * T_num0
        A_total_cold0 = T_htA_cold * T_num0

        # Mass fluxes (sector/header based)
        G_h0 = mflow_h / Aff_hot_mid
        G_c0 = mflow_c / Aff_cold_total

        def _0d_step(Th_s: float, Ph_s: float, Tc_s: float, Pc_s: float):
            rho_h_s, cp_h_s, mu_h_s, k_h_s = fluid_h.get_transport_properties(Th_s, Ph_s)
            rho_c_s, cp_c_s, mu_c_s, k_c_s = fluid_c.get_transport_properties(Tc_s, Pc_s)
            Pr_h_s = mu_h_s * cp_h_s / max(k_h_s, 1e-16)
            Pr_c_s = mu_c_s * cp_c_s / max(k_c_s, 1e-16)
            Re_h_OD_s = max(G_h0 * T_OD / max(mu_h_s, 1e-16), 1e-6)
            Re_c_s = max(G_c0 * T_ID / max(mu_c_s, 1e-16), 1e-6)
            Nu_h_s, f_h_s = tube_bank_nusselt_number_and_friction_factor(
                Re_h_OD_s, T_sls, T_sts, Pr_h_s, inline=(not staggered), n_rows=T_rrows
            )
            Nu_c_s = circular_pipe_nusselt(Re_c_s, 0, prandtl=Pr_c_s)
            f_c_s = circular_pipe_friction_factor(Re_c_s, 0)
            h_h_s = Nu_h_s * k_h_s / T_OD
            h_c_s = Nu_c_s * k_c_s / T_ID
            U_s = 1.0 / (1.0 / h_h_s + 1.0 / h_c_s * (T_OD / T_ID) + T_OD / (2 * k_tube) * np.log(T_OD / T_ID))
            C_h_tot_s = mflow_h_total * cp_h_s
            C_c_tot_s = mflow_c_total * cp_c_s
            C_min_s = min(C_h_tot_s, C_c_tot_s)
            C_max_s = max(C_h_tot_s, C_c_tot_s)
            Cr_s = C_min_s / max(C_max_s, 1e-16)
            NTU_s = U_s * A_total_hot0 / max(C_min_s, 1e-16)
            eps_s = epsilon_ntu(NTU_s, Cr_s, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)
            Q_s = eps_s * C_min_s * (Th_in - Tc_in)
            Th_out_s = Th_in - Q_s / max(C_h_tot_s, 1e-16)
            Tc_out_s = Tc_in + Q_s / max(C_c_tot_s, 1e-16)
            # simple dp estimates
            dp_h_s = f_h_s * (A_total_hot0 / Aff_hot_mid) * (G_h0**2) / (2 * max(rho_h_s, 1e-12))
            dp_c_s = f_c_s * (A_total_cold0 / Aff_cold_total) * (G_c0**2) / (2 * max(rho_c_s, 1e-12))
            Ph_out_s = Ph_in - dp_h_s
            Pc_out_s = Pc_in - dp_c_s
            return Th_out_s, Tc_out_s, Ph_out_s, Pc_out_s

        Th_o1, Tc_o1, Ph_o1, Pc_o1 = _0d_step(Th_in, Ph_in, Tc_in, Pc_in)
        Th_mean0 = 0.5 * (Th_in + Th_o1)
        Tc_mean0 = 0.5 * (Tc_in + Tc_o1)
        Ph_mean0 = 0.5 * (Ph_in + Ph_o1)
        Pc_mean0 = 0.5 * (Pc_in + Pc_o1)
        Th_o2, Tc_o2, Ph_o2, Pc_o2 = _0d_step(Th_mean0, Ph_mean0, Tc_mean0, Pc_mean0)

        if guess_hot_boundary:
            Th_out_guess = float(Th_o2)
            Ph_out_guess = float(Ph_o2)
        if guess_cold_boundary:
            Tc_out_guess = float(Tc_o2)
            Pc_out_guess = float(Pc_o2)
    except Exception as e:
        print("Error in 0D step")
        print(e)
        pass

    # Simple relaxation factors (step size control)
    # Separate, fixed scale factors for temperature and pressure
    relax_T = float(params.get("outer_relax_T", 0.10))
    relax_P = float(params.get("outer_relax_P", 0.03))
    pressure_boost = float(params.get("outer_pressure_boost", 3.0))  # used when |eT| < tol

    for _outer in range(outer_max_iter):
        # Show current guesses at start of iteration
        if guess_hot_boundary:
            print(f"[outer {_outer + 1}] hot guess -> Th[0]={Th_out_guess:.2f} K, Ph[0]={Ph_out_guess:.2e} Pa")
        elif guess_cold_boundary:
            print(f"[outer {_outer + 1}] cold guess -> Tc[0]={Tc_out_guess:.2f} K, Pc[0]={Pc_out_guess:.2e} Pa")
        # region Allocate arrays fresh each pass
        Th = np.zeros(n + 1)
        Ph = np.zeros(n + 1)
        Tc = np.zeros(n + 1)
        Pc = np.zeros(n + 1)

        tube_length = np.zeros(n)
        Aht_hot = np.zeros(n)
        Aht_cold = np.zeros(n)
        Afr_hot = np.zeros(n)
        Afr_cold = np.zeros(n)
        Aff_hot = np.zeros(n)
        Aff_cold = np.zeros(n)
        Dh_hot = np.zeros(n)

        mu_h = np.zeros(n)
        mu_c = np.zeros(n)
        k_h = np.zeros(n)
        k_c = np.zeros(n)
        cp_h = np.zeros(n)
        cp_c = np.zeros(n)
        Pr_h = np.zeros(n)
        Pr_c = np.zeros(n)
        rho_h = np.zeros(n + 1)
        rho_c = np.zeros(n + 1)

        G_h = np.zeros(n)
        G_c = np.zeros(n)
        Re_h = np.zeros(n)
        Re_h_OD = np.zeros(n)
        Re_c = np.zeros(n)
        f_h = np.zeros(n)
        f_c = np.zeros(n)
        Nu_h = np.zeros(n)
        Nu_c = np.zeros(n)
        h_h = np.zeros(n)
        h_c = np.zeros(n)
        U_h = np.zeros(n)
        U_c = np.zeros(n)
        C_h = np.zeros(n)
        C_c = np.zeros(n)
        C_min = np.zeros(n)
        C_max = np.zeros(n)
        C_ratio = np.zeros(n)
        NTU_local = np.zeros(n)
        NTU_kb = np.zeros(n)
        eps_local = np.zeros(n)
        q_max = np.zeros(n)
        q = np.zeros(n)
        dP_h = np.zeros(n)
        dP_c = np.zeros(n)
        # endregion

        # Apply boundary (guess one side, fix the other to known inlet)
        Th[0] = Th_out_guess if guess_hot_boundary else Th_in
        Ph[0] = Ph_out_guess if guess_hot_boundary else Ph_in
        Tc[0] = Tc_out_guess if guess_cold_boundary else Tc_in
        Pc[0] = Pc_out_guess if guess_cold_boundary else Pc_in

        # region Single forward pass
        for j in range(n):
            # Geometry per layer
            tube_length[j] = (
                HX_dX if input_rect_HX == 1 else np.trapezoid(np.sqrt(r[j : j + 2] ** 2 + Inv_b**2), theta[j : j + 2])
            )
            tubes_in_layer = T_arows * H_rrows
            Aht_hot[j] = np.pi * T_OD * tube_length[j] * tubes_in_layer
            Aht_cold[j] = np.pi * T_ID * tube_length[j] * tubes_in_layer
            Afr_hot[j] = (HX_dX * 2 * np.pi * r[j] / HX_sectors) if input_rect_HX == 0 else (HX_dX * HX_rmin)
            Afr_cold[j] = HX_dX * dR_step
            Aff_hot[j] = Afr_hot[j] * Ah_sigma
            Aff_cold[j] = Afr_cold[j] * Ac_sigma
            Lf = H_rrows * T_sls * T_OD
            Dh_hot[j] = (4 * Aff_hot[j] * Lf) / Aht_hot[j]

            # Properties (at current node)
            rho_h[j], cp_h[j], mu_h[j], k_h[j] = fluid_h.get_transport_properties(Th[j], Ph[j])
            rho_c[j], cp_c[j], mu_c[j], k_c[j] = fluid_c.get_transport_properties(Tc[j], Pc[j])
            Pr_h[j] = mu_h[j] * cp_h[j] / k_h[j]
            Pr_c[j] = mu_c[j] * cp_c[j] / k_c[j]

            # Fluxes and Reynolds
            G_h[j] = mflow_h / Aff_hot[j]
            G_c[j] = mflow_c / Aff_cold[j]
            Re_h[j] = G_h[j] * Dh_hot[j] / mu_h[j]
            Re_h_OD[j] = G_h[j] * T_OD / mu_h[j]
            Re_c[j] = G_c[j] * T_ID / mu_c[j]

            # Cold side HT & friction (circular pipe)
            Nu_c[j] = circular_pipe_nusselt(Re_c[j], 0, prandtl=Pr_c[j])
            f_c[j] = circular_pipe_friction_factor(Re_c[j], 0)
            h_c[j] = Nu_c[j] * k_c[j] / T_ID

            # Hot side HT & friction (tube bank)
            Nu_h[j], f_h[j] = tube_bank_nusselt_number_and_friction_factor(
                Re_h_OD[j], T_sls, T_sts, Pr_h[j], inline=(not staggered), n_rows=T_rrows
            )
            h_h[j] = Nu_h[j] * k_h[j] / T_OD

            # Enhancement
            h_h[j] *= qhtc_tuning_factor

            # Overall U (hot-side basis with wall conduction)
            U_h[j] = 1.0 / (1.0 / h_h[j] + 1.0 / h_c[j] * (T_OD / T_ID) + T_OD / (2 * k_tube) * np.log(T_OD / T_ID))
            U_c[j] = 1.0 / (1.0 / h_c[j] + 1.0 / h_h[j] * (T_ID / T_OD) + T_OD / (2 * k_tube) * np.log(T_OD / T_ID))

            # Capacity rates
            C_h[j] = mflow_h * cp_h[j]
            C_c[j] = mflow_c * cp_c[j]
            C_min[j] = min(C_h[j], C_c[j])
            C_max[j] = max(C_h[j], C_c[j])
            C_ratio[j] = C_min[j] / C_max[j]

            # NTU local
            NTU_local[j] = U_h[j] * Aht_hot[j] / C_min[j]
            # KB NTU (check parity)
            NTU_kb[j] = ntu(
                h_h[j] / (cp_h[j] * G_h[j]),
                h_c[j] / (cp_c[j] * G_c[j]),
                Aht_hot[j] / Aff_hot[j],
                Aht_cold[j] / Aff_cold[j],
                C_h[j],
                C_c[j],
            )

            # Effectiveness local (counterflow)
            C_mixed = C_h[j]
            C_unmixed = C_c[j]

            Cr = C_ratio[j]

            # eps_local[j] = (1 - np.exp(-NTU_local[j] * (1 - Cr))) / (1 - Cr * np.exp(-NTU_local[j] * (1 - Cr)))
            flow_type = "Cmax_mixed" if C_mixed > C_unmixed else "Cmin_mixed"
            eps_local[j] = epsilon_ntu(NTU_local[j], Cr, exchanger_type="cross_flow", flow_type=flow_type, n_passes=1)

            # Heat transfer and outlet temps
            q_max[j] = C_min[j] * (Th[j] - Tc[j])
            q[j] = eps_local[j] * q_max[j]

            dh0_hot = -q[j] / mflow_h
            dh0_cold = q[j] / mflow_c
            tau_dA_A_c_hot = f_h[j] * (Aht_hot[j] / Aff_hot[j]) * (G_h[j] ** 2) / (2 * rho_h[j])
            tau_dA_A_c_cold = f_c[j] * (Aht_cold[j] / Aff_cold[j]) * (G_c[j] ** 2) / (2 * rho_c[j])

            loop_iterations = 20
            if outboard == 0:  # if inboard
                local_hot_in = j + 1  # hot is going from high r to low r i.e. against j
                local_hot_out = j
                local_cold_in = j  # cold is going from low r to high r i.e. with j
                local_cold_out = j + 1

                Th[local_hot_in], Ph[local_hot_in], _ = update_static_properties(
                    fluid_h,
                    G_h[j],
                    dh0_hot,
                    tau_dA_A_c_hot,
                    Th[local_hot_out],
                    rho_h[local_hot_out],
                    Ph[local_hot_out],
                    a_is_in=False,
                    b_is_in=False,
                    max_iter=loop_iterations,
                )
                # endregion
                Tc[local_cold_out], Pc[local_cold_out], _ = update_static_properties(
                    fluid_c,
                    G_c[j],
                    dh0_cold,
                    tau_dA_A_c_cold,
                    Tc[local_cold_in],
                    rho_c[local_cold_in],
                    Pc[local_cold_in],
                    a_is_in=True,
                    b_is_in=True,
                    max_iter=loop_iterations,
                )
            else:  # outboard
                local_hot_in = j  # hot is going from low r to high r i.e. with j
                local_hot_out = j + 1
                local_cold_in = j + 1  # cold is going from high r to low r i.e. against j
                local_cold_out = j
                Th[local_hot_out], Ph[local_hot_out], _ = update_static_properties(
                    fluid_h,
                    G_h[j],
                    dh0_hot,
                    tau_dA_A_c_hot,
                    Th[local_hot_in],
                    rho_h[local_hot_in],
                    Ph[local_hot_in],
                    a_is_in=True,
                    b_is_in=True,
                    max_iter=loop_iterations,
                )
                Tc[local_cold_in], Pc[local_cold_in], _ = update_static_properties(
                    fluid_c,
                    G_c[j],
                    dh0_cold,
                    tau_dA_A_c_cold,
                    Tc[local_cold_out],
                    rho_c[local_cold_out],
                    Pc[local_cold_out],
                    a_is_in=False,
                    b_is_in=False,
                    max_iter=loop_iterations,
                )
        # Boundary matching and guess update (after forward pass)
        if guess_hot_boundary:
            # Hot enters at high r: match computed inlet at index -1 to specified
            eT = Th[-1] - Th_in
            eP = Ph[-1] - Ph_in
            if abs(eT) < outer_tol_T and abs(eP) < outer_rel_tol_p * max(Ph_in, 1.0):
                break
            # diagnostics: print both if both are not converged, otherwise print only the one not converged
            printed = False
            if abs(eT) >= outer_tol_T:
                print(
                    "Calculated Hot stream inlet temperature "
                    f"{Th[-1]:.2f} K does not match imposed inlet temperature {Th_in:.2f} K "
                    f"with initial exit temperature guess {Th_out_guess:.2f} K"
                )
                printed = True
            if abs(eP) >= outer_rel_tol_p * max(Ph_in, 1.0):
                print(
                    "Calculated Hot stream inlet pressure "
                    f"{Ph[-1]:.2e} Pa does not match inlet pressure {Ph_in:.2e} Pa "
                    f"with initial guess {Ph_out_guess:.2e} Pa"
                )
            # Update both with fixed, separate scale factors
            Th_out_guess -= relax_T * eT
            # Pressure: relative/log-space update to keep positivity
            local_relax_P = relax_P * (pressure_boost if abs(eT) < outer_tol_T else 1.0)
            p_log = float(np.log(max(Ph_out_guess, 100.0)))
            p_log -= local_relax_P * (eP / max(Ph_in, 1.0))
            Ph_out_guess = float(np.exp(p_log))
            # print(f"[outer {_outer + 1}] updated hot guess -> Th[0]={Th_out_guess:.2f} K, Ph[0]={Ph_out_guess:.2f} Pa")
        else:
            # Cold enters at high r: match computed cold inlet at index -1
            eT_c = Tc[-1] - Tc_in
            eP_c = Pc[-1] - Pc_in
            if abs(eT_c) < outer_tol_T and abs(eP_c) < outer_rel_tol_p * max(Pc_in, 1.0):
                break
            # diagnostics: print both if both are not converged, otherwise print only the one not converged
            if abs(eT_c) >= outer_tol_T:
                print(
                    "Calculated Cold stream inlet temperature "
                    f"{Tc[-1]:.2f} K does not match inlet temperature {Tc_in:.2f} K "
                    f"with initial guess {Tc_out_guess:.2f} K"
                )
            if abs(eP_c) >= outer_rel_tol_p * max(Pc_in, 1.0):
                print(
                    "Calculated Cold stream inlet pressure "
                    f"{Pc[-1]:.2e} Pa does not match inlet pressure {Pc_in:.2e} Pa "
                    f"with initial guess {Pc_out_guess:.2e} Pa"
                )
            # Update both with fixed, separate scale factors
            Tc_out_guess -= relax_T * eT_c
            local_relax_P_c = relax_P * (pressure_boost if abs(eT_c) < outer_tol_T else 1.0)
            p_log_c = float(np.log(max(Pc_out_guess, 100.0)))
            p_log_c -= local_relax_P_c * (eP_c / max(Pc_in, 1.0))
            Pc_out_guess = float(np.exp(p_log_c))
        # print(f"[outer {_outer + 1}] updated cold guess -> Tc[0]={Tc_out_guess:.2f} K, Pc[0]={Pc_out_guess:.2f} Pa")

    # End of outer loop: calculate overall HEx metrics
    # Enthalpy-balance based totals (align with colleague's summary approach)
    if outboard == 0:
        hot_in_total = -1
        hot_out_total = 0
        cold_in_total = 0
        cold_out_total = -1
    else:
        hot_in_total = 0
        hot_out_total = -1
        cold_in_total = -1
        cold_out_total = 0

    h_h_in = fluid_h.get_specific_enthalpy(Th[hot_in_total], Ph[hot_in_total])
    h_h_out = fluid_h.get_specific_enthalpy(Th[hot_out_total], Ph[hot_out_total])
    h_c_in = fluid_c.get_specific_enthalpy(Tc[cold_in_total], Pc[cold_in_total])
    h_c_out = fluid_c.get_specific_enthalpy(Tc[cold_out_total], Pc[cold_out_total])
    pressure_drop_hot_pct = (1 - Ph[hot_out_total] / Ph[hot_in_total]) * 100.0

    Q_h_enthalpy = mflow_h_total * (h_h_in - h_h_out)
    Q_c_enthalpy = mflow_c_total * (h_c_out - h_c_in)

    # Global NTU via total UA and a representative C_min
    UA_total = float(np.sum(U_h * Aht_hot))
    C_hot_global = mflow_h_total * (h_h_in - h_h_out) / (Th[hot_in_total] - Th[hot_out_total])
    C_cold_global = mflow_c_total * (h_c_out - h_c_in) / (Tc[cold_out_total] - Tc[cold_in_total])

    C_min_global = min(C_hot_global, C_cold_global)
    C_max_global = max(C_hot_global, C_cold_global)
    C_ratio_global = C_min_global / C_max_global
    NTU_global = UA_total / (C_min_global + 1e-16)

    # Overall effectiveness: use enthalpy and inlet temp diff consistent with orientation

    eps_total = Q_h_enthalpy / C_min_global / (Th[hot_in_total] - Tc[cold_in_total])

    # Areas
    HX_area_front = HX_area
    A_total_hot = float(np.sum(Aht_hot))

    results = {
        "Th_in_K": float(Th_in if outboard == 0 else Th[0]),
        "Th_out_K": float(Th[0] if outboard == 0 else Th[-1]),
        "Tc_in_K": float(Tc[0] if outboard == 0 else Tc[-1]),
        "Tc_out_K": float(Tc[-1] if outboard == 0 else Tc[0]),
        "Ph_in_bar": float((Ph[-1] if outboard == 0 else Ph[0]) / 1e5),
        "Ph_out_bar": float((Ph[0] if outboard == 0 else Ph[-1]) / 1e5),
        "Pc_in_bar": float((Pc[0] if outboard == 0 else Pc[-1]) / 1e5),
        "Pc_out_bar": float((Pc[-1] if outboard == 0 else Pc[0]) / 1e5),
        "pressure_drop_hot_pct": float(pressure_drop_hot_pct),
        "pressure_drop_cold_pct": float(
            ((Pc[0] - Pc[-1]) / Pc[0] * 100.0) if outboard == 0 else ((Pc[-1] - Pc[0]) / Pc[-1] * 100.0)
        ),
        "effectiveness": float(eps_total),
        "NTU_global": float(NTU_global),
        "UA_total_W_per_K": float(UA_total),
        "A_total_hot_m2": float(A_total_hot),
        "C_ratio": float(C_ratio_global),
        "HX_frontal_area_m2": float(HX_area_front),
        "Q_h_enthalpy_W": float(Q_h_enthalpy),
        "Q_c_enthalpy_W": float(Q_c_enthalpy),
        # For plotting/debug
        "r_profile_m": r,
        "Th_profile_K": Th,
        "Tc_profile_K": Tc,
        "Ph_profile_Pa": Ph,
        "Pc_profile_Pa": Pc,
        "Re_h_OD": Re_h_OD,
        "J_over_f_mean": float(np.mean((h_h / (cp_h * G_h)) / (f_h + 1e-16))) if np.all(f_h) else np.nan,
    }

    if debug:
        print(f"Overall effectiveness: {results['effectiveness'] * 100:.1f}%")
        print(f"Hot-side pressure drop: {results['pressure_drop_hot_pct']:.2f}%")
        print(f"Global NTU: {results['NTU_global']:.3f}")

        # Plot temperature and hot-side pressure ratios similar to colleague script
        plt.figure(figsize=(12, 8))
        title = (
            f"Temperature profiles {fluid_h} hot, {fluid_c} cold, "
            f"{mflow_h_total:.2f} kg/s hot, {mflow_c_total:.2f} kg/s cold\n"
            f"{Th_in:.1f} K hot inlet ({Ph_in / 1e5:.2f} bar), {Tc_in:.1f} K cold inlet ({Pc_in / 1e5:.2f} bar)\n"
            f"r_min={HX_rmin * 1e3:.1f} mm, r_max={HX_rmax * 1e3:.1f} mm, T_OD={T_OD * 1e3:.2f} mm, sts={T_sts}, sls={T_sls}\n"
            f"Headers={H_number}, rows/header={H_rrows}, Inv_angle={Inv_angle:.0f}°, NTU={results['NTU_global']:.3f}, eps={results['effectiveness'] * 100:.1f}%"
        )
        plt.title(title)
        plt.plot(r * 1000, Th, "r-o", linewidth=2, markersize=4, label="Hot stream")
        plt.plot(r * 1000, Tc, "b-o", linewidth=2, markersize=4, label="Cold stream")
        plt.axhline(Tc_in, color="b", linestyle="--", linewidth=1.5, label="Cold inlet")
        plt.axhline(Th_in, color="r", linestyle="--", linewidth=1.5, label="Hot inlet")
        plt.xlabel("Radial location [mm]")
        plt.ylabel("Temperature [K]")
        plt.legend(loc="upper left")
        ax2 = plt.gca().twinx()
        if outboard == 0:
            ax2.plot(
                r * 1000,
                (Ph / Ph[-1] - 1) * 100,
                "k-x",
                linewidth=2,
                markersize=4,
                label="Hot dP (%)",
            )
        else:
            ax2.plot(
                r * 1000,
                (Ph / Ph[0] - 1) * 100,
                "k-x",
                linewidth=2,
                markersize=4,
                label="Hot dP (%)",
            )
        ax2.set_ylabel("Hot side pressure drop [%]")
        ax2.legend(loc="center left")
        plt.show()

    return results


def compute_involute_hx(params: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """Public API to compute HX performance. Returns summary results.

    - debug=True will print and plot for diagnostics.
    - debug=False returns only the results dict (input-output mode).
    """
    return _solve_involute_hx(params, debug=debug)


def sweep_rmax_keep_b(
    params: Dict[str, Any], dr: float = 0.01, steps: int = 5, debug_each: bool = False
) -> List[Dict[str, Any]]:
    """Generate a sweep increasing HX_rmax by dr for `steps`, keeping Inv_b constant.

    Adjusts `Inv_angle_deg` each step so that (HX_rmax - HX_rmin)/rad(Inv_angle) remains constant.
    Returns a list of results dicts for baseline and each step.
    """
    base = dict(params)  # shallow copy OK
    rmin = float(base["HX_rmin"])
    rmax0 = float(base["HX_rmax"])
    Inv_angle_deg0 = float(base.get("Inv_angle_deg", 360.0))
    b0 = (rmax0 - rmin) / np.deg2rad(Inv_angle_deg0)

    out: List[Dict[str, Any]] = []

    # Compute baseline first
    t0 = time.perf_counter()
    # Use gentle relax for baseline unless provided
    base_run = dict(base)
    base_run.setdefault("outer_relax_T", 0.02)
    base_run.setdefault("outer_relax_P", 0.02)
    base_run.setdefault("outer_max_iter", 10)
    res0 = compute_involute_hx(base_run, debug=debug_each)
    res0["elapsed_s"] = time.perf_counter() - t0
    res0["label"] = "baseline"
    res0["HX_rmax_m"] = rmax0
    out.append(res0)
    prev = res0

    # Steps
    for k in range(1, steps + 1):
        cfg = dict(base)
        rmax_new = rmax0 + k * dr
        angle_rad_new = (rmax_new - rmin) / b0
        angle_deg_new = float(np.rad2deg(angle_rad_new))
        cfg["HX_rmax"] = rmax_new
        cfg["Inv_angle_deg"] = angle_deg_new
        # Warm-start: use previous step's outlet as this step's initial guesses on the guessed boundary
        if int(params.get("outboard", 0)) == 0:
            cfg["outer_init_hot"] = {
                "T": float(prev.get("Th_out_K", np.nan)),
                "P": float(prev.get("Ph_out_bar", np.nan) * 1e5)
                if np.isfinite(prev.get("Ph_out_bar", np.nan))
                else None,
            }
        else:
            cfg["outer_init_cold"] = {
                "T": float(prev.get("Tc_out_K", np.nan)),
                "P": float(prev.get("Pc_out_bar", np.nan) * 1e5)
                if np.isfinite(prev.get("Pc_out_bar", np.nan))
                else None,
            }
        # Slightly higher relax for faster settling; allow more outer iters
        cfg.setdefault("outer_relax_T", 0.03)
        cfg.setdefault("outer_relax_P", 0.03)
        cfg.setdefault("outer_max_iter", 12)
        # Other geometry derived fields remain valid (rmin constant). Discretization follows from lengths.
        tk0 = time.perf_counter()
        resk = compute_involute_hx(cfg, debug=debug_each)
        resk["elapsed_s"] = time.perf_counter() - tk0
        resk["label"] = f"rmax+{k}"
        resk["HX_rmax_m"] = rmax_new
        out.append(resk)
        prev = resk

    return out


def summarize_sweep(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a compact table with requested metrics and relative changes vs baseline."""
    if not results:
        return []
    base = results[0]
    A0 = base.get("A_total_hot_m2") or base.get("A_total_hot_m2", base.get("A_total_hot_m2"))
    NTU0 = base["NTU_global"]
    rows: List[Dict[str, Any]] = []
    for r in results:
        A = r["A_total_hot_m2"]
        NTU = r["NTU_global"]
        eps = r["effectiveness"]
        C_ratio = r["C_ratio"]
        rows.append(
            {
                "label": r.get("label", ""),
                "rmax_m": r.get("HX_rmax_m", np.nan),
                "A_hot_m2": A,
                "A_inc_pct": (A / A0 - 1.0) * 100.0 if A0 else np.nan,
                "NTU": NTU,
                "NTU_pct": (NTU / NTU0 - 1) * 100 if NTU0 else np.nan,
                "eps": eps,
                "Cflow": epsilon_ntu(NTU, C_ratio, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1),
                "C_ratio": C_ratio,
                "dP_h_pct": r["pressure_drop_hot_pct"],
                "dP_c_pct": r["pressure_drop_cold_pct"],
            }
        )
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    """Pretty-print a compact table without external deps."""
    if not rows:
        print("(no results)")
        return
    headers = [
        "label",
        "rmax_m",
        "A_hot_m2",
        "A_inc_pct",
        "NTU",
        "NTU_pct",
        "eps",
        "Cflow",
        "dP_h_pct",
        "dP_c_pct",
    ]
    headers = ["NTU", "eps", "C_ratio"]  # uncomment this line to only print NTU and eps
    col_widths = {
        h: max(
            len(h),
            *(
                len(f"{row.get(h, ''):.4g}")
                if isinstance(row.get(h, None), (int, float, np.floating))
                else len(str(row.get(h, "")))
                for row in rows
            ),
        )
        for h in headers
    }
    # Header
    line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    # Rows
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, (int, float, np.floating)):
                if h in ("eps", "Cflow"):
                    s = f"{v:.4f}"
                elif h in ("A_inc_pct", "dP_h_pct", "dP_c_pct", "NTU_pct"):
                    s = f"{v:.2f}"
                else:
                    s = f"{v:.6g}"
            else:
                s = str(v)
            vals.append(s.ljust(col_widths[h]))
        print(" | ".join(vals))


def plot_eps_vs_ntu(rows: List[Dict[str, Any]], title: str = "Epsilon vs NTU") -> None:
    if not rows:
        return
    x = [r["NTU"] for r in rows]
    y = [r["epsilon"] for r in rows]
    labels = [r["label"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, "o-")
    for xi, yi, lbl in zip(x, y, labels):
        plt.annotate(lbl, (xi, yi), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("NTU (global)")
    plt.ylabel("Effectiveness (epsilon)")
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example usage on VIPER baseline; easy to switch case by changing get_preset_params argument
    params = get_preset_params("viper")

    # 1) Input-Output mode (no plots)
    t0 = time.perf_counter()
    res = compute_involute_hx(params, debug=False)
    t1 = time.perf_counter()
    print(
        f"VIPER baseline -> epsilon={res['effectiveness'] * 100:.2f}%, NTU={res['NTU_global']:.3f}, dP_hot={res['pressure_drop_hot_pct']:.2f}%"
    )
    print(f"compute_involute_hx() runtime: {t1 - t0:.3f} s")

    # 2) Debug mode (plots/prints akin to colleague's)
    # compute_involute_hx(params, debug=True)

    # 3) Sweep r_max while keeping Inv_b constant; 5 steps of +0.01 m
    """t_sweep0 = time.perf_counter()
    sweep_results = sweep_rmax_keep_b(params, dr=0.02, steps=10, debug_each=False)
    t_sweep_total = time.perf_counter() - t_sweep0
    for r in sweep_results:
        dur = r.get("elapsed_s", float("nan"))
        print(f"sweep element '{r.get('label', '')}' runtime: {dur:.3f} s")
    table = summarize_sweep(sweep_results)
    print_table(table)
    print(f"Total sweep runtime: {t_sweep_total:.3f} s") """
    # plot_eps_vs_ntu(table, title="VIPER: Epsilon vs NTU (r_max sweep, Inv_b constant)")
