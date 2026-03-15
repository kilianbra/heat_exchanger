"""
Fig8-style cycle-coupled HEx analysis.
Uses xflow helicopter parameters, g2h = 0.5*gamma*M^2, solves for mdot at constant shaft power.
Three weight-delta lines vs baseline unrecuperated engine: delta fuel, delta fuel+hex, delta fuel+hex+engine.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import root
from tabulate import tabulate
from xflow import (
    calculate_capacity_ratios,
    calculate_pressure_drop_ratio,
    practical_unavailable_creation_hex,
)

from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figs_current")


# effectiveness = 0.65
effectiveness = 0.6
# dp_hot = 0.04  # 0.036
dp_hot = 0.06  # dp_h/pin
# dp_cold = 0.02  # 0.022
dp_cold = 0.04  # dpc/pcin

PR = 9.0
TIT = 1500
eta_poly_c = 0.88
eta_poly_t = 0.84

kg_dry_engine_per_kg_per_s_of_air = 23.0
# m_engine_no_HEx = kg_dry_engine_per_kg_per_s_of_air * mdot_air


# --- Helicopter parameters from xflow.py (ref: eps=0.6, dp_h/pin=0.06, dpc/pcin=0.04) ---
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_C_COLD_OVER_C_HOT = 1.0
# DEFAULT_D_R_hot = 0.257  # chosen to get pressure drops?
DEFAULT_D_R_hot = 0.25
DEFAULT_GAMMA = 1.4
# DEFAULT_MACH_IN = 0.1
DEFAULT_MACH_IN = 0.11  # Mh_in
# g2h = 0.5 * gamma * M^2 (used only for reference Mach; we compute g2h from M dynamically)
DEFAULT_ST_OVER_F = 0.4
# DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_F_C_OVER_F_H = 0.25
# DEFAULT_T = 898 / 588
DEFAULT_T = 907 / 588  # T_ratios from xflow 106-109
DEFAULT_T_DEAD_OVER_T_COLD_IN = 288 / 588
# DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.04
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 9.0 / 1.064
# DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.04
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.064
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
# TARGET_EPS = 0.65
TARGET_EPS = 0.6
# NTU_MATCH = 1.824  # 1.701
NTU_MATCH = 1.479
DEFAULT_NTU_MAX = 8.0
DEFAULT_DP_MAX = 0.2
DEFAULT_MOLAR_MASS_RATIO = 1.0
# DEFAULT_A_R_hot = 1.0
DEFAULT_A_R_hot = 0.92

# Baseline and power reference (from plan)
# Cycle model net_work is J/kg (c_p in J/kg/K, T in K => work in J/kg)
# P = mdot * w_net => W = (kg/s) * (J/kg) = J/s = W
# mdot_ref = 2.24  # kg/s
# w_net_ref = 312e3  # J/kg (303 kJ/kg)
P_shaft_ref = 700e3  # W (~700 kW)
# NTU_ref = 1.824
NTU_ref = 1.479

# Sweep parameters
AO_SWEEP = np.linspace(0.15, 5, 100)  # extend to allow small m_hex (down to 0.1 kg)
m_hex_ref = 13.3  # kg
A_R_REF_VALUES = np.linspace(0.1 / m_hex_ref, 5.0, 100)  # A/A_ref sweep; start at m_hex = 0.1 kg

# Mission parameters
mission_hours = 2
mission_seconds = mission_hours * 3600
LHV_MJ_per_kg = 12.0 * 3.6  # MJ/kg (kerosene) — single definition
LHV_J_per_kg = LHV_MJ_per_kg * 1e6  # J/kg; mdot_fuel = P_shaft/(LHV×η/100), factor_fuel = t_s/(LHV)×η_turb/η_ov×Q_max
eta_turb = 0.88
eta_ov = 0.4045  #

# --- Black-line variant toggles ---
# If True, (mdot - mdot_baseline)*kg_dry is added to ALL dQ^M-based objectives.
# Effect 1 (BC) uses uncoupled cycle and reference mdot, so delta_engine ≈ 0 regardless.
INCLUDE_DELTA_ENGINE = False
# If True, black lines and black star are hidden in the plot (printouts always shown).
HIDE_BLACK_LINES = True
FONT_SIZE = 8


def calculate_cycle(PR, TIT, eta_poly_c, eta_poly_t):
    gamma = 1.4  # Ratio of specific heats
    c_p = 1070.0  # J/(kg*K) — high-temp air value (compromise between 1004 and 1170)
    R = c_p * (gamma - 1) / gamma  # Gas constant from c_p and gamma
    # R_universal = 8.314e3; M_air = 28.97; R = R_universal / M_air; c_p = R * gamma / (gamma - 1)  # 1004.45 J/kg/K

    # Initial conditions
    p_d = 1e5  # Inlet pressure (Pa)
    T_d = 288  # Inlet temperature (K)

    # Lists to store states
    P = [p_d]  # Pressures
    T = [T_d]  # Temperatures
    s = [0]  # Entropies

    # Compressor calculations
    P.append(P[0] * PR)
    # T_2s = T[0] * (PR ** ((gamma - 1) / gamma))
    T_compressor_out = T[0] * (PR ** ((gamma - 1) / (gamma * eta_poly_c)))
    T.append(T_compressor_out)
    s.append(s[-1] + c_p * np.log(T[-1] / T[-2]) - R * np.log(P[-1] / P[-2]))

    # Heat addition (constant pressure)
    T.append(TIT)
    P.append(P[-1])
    s.append(s[-1] + c_p * np.log(T[-1] / T[-2]))

    # Turbine calculations
    PR_turbine = P[-1] / p_d
    # T_4s = T[-1] * (1 / PR_turbine) ** ((gamma - 1) / gamma)
    T_turbine_out = T[-1] * (1 / PR_turbine) ** ((gamma - 1) / gamma * eta_poly_t)
    T.append(T_turbine_out)
    P.append(p_d)
    s.append(s[-1] + c_p * np.log(T[-1] / T[-2]) - R * np.log(P[-1] / P[-2]))

    # Calculate efficiency
    work_compressor = c_p * (T[1] - T[0])
    heat_addition = c_p * (T[2] - T[1])
    work_turbine = c_p * (T[2] - T[3])
    net_work = work_turbine - work_compressor
    efficiency = net_work / heat_addition * 100

    return P, T, s, efficiency, net_work


def calculate_recuperated_cycle_dp_eps(PR, TIT, eta_poly_c, eta_poly_t, effectiveness, dp_hot, dp_cold):
    gamma = 1.4  # Ratio of specific heats
    c_p = 1070.0  # J/(kg*K) — high-temp air value (compromise between 1004 and 1170)
    R = c_p * (gamma - 1) / gamma  # Gas constant from c_p and gamma
    # R_universal = 8.314e3; M_air = 28.97; R = R_universal / M_air; c_p = R * gamma / (gamma - 1)

    # print(f"Constants: R = {R:.2f} J/kg/K, c_p = {c_p:.2f} J/kg/K")

    # Initial conditions
    p_d = 1e5  # Inlet pressure (Pa)
    T_d = 288  # Inlet temperature (K)

    # Lists to store states
    P = [p_d]  # Pressures
    T = [T_d]  # Temperatures
    s = [0]  # Entropies

    # Compressor exit calculations
    P.append(P[0] * PR)
    T_compressor_out = T[0] * (PR ** ((gamma - 1) / (gamma * eta_poly_c)))
    T.append(T_compressor_out)
    s.append(s[-1] + c_p * np.log(T[-1] / T[-2]) - R * np.log(P[-1] / P[-2]))

    # Recuperator cold side exit calculations
    P.append(P[-1] * (1 - dp_cold))  # Pressure drop in cold side

    # Turbine calculations
    P_turbine_in = P[-1]
    P_turbine_out = p_d / (1 - dp_hot)

    # Turbine exit temperature (without recuperator)
    T_turbine_out = TIT * (P_turbine_out / P_turbine_in) ** ((gamma - 1) / gamma * eta_poly_t)

    # Maximum possible temperature rise in recuperator
    max_temp_rise = max(T_turbine_out - T[1], 0)

    # Actual temperature rise based on effectiveness
    T_recuperator_cold_out = T[1] + effectiveness * max_temp_rise
    T.append(T_recuperator_cold_out)
    s.append(s[-1] + c_p * np.log(T[-1] / T[-2]) - R * np.log(P[-1] / P[-2]))

    # Heat addition (constant pressure): turbine inlet
    T.append(TIT)
    P.append(P[-1])
    s.append(s[-1] + c_p * np.log(T[-1] / T[-2]))

    # Turbine expansion
    T.append(T_turbine_out)
    P.append(p_d / (1 - dp_hot))
    s.append(s[-1] + c_p * np.log(T[-1] / T[-2]) - R * np.log(P[-1] / P[-2]))

    # Recuperator hot side
    T_recuperator_hot_out = T[-1] - effectiveness * max_temp_rise
    T.append(T_recuperator_hot_out)
    P.append(P[-1] * (1 - dp_hot))  # Pressure drop in hot side = pd
    s.append(s[-1] + c_p * np.log(T[-1] / T[-2]) - R * np.log(P[-1] / P[-2]))

    # Calculate efficiency
    work_compressor = c_p * (T[1] - T[0])
    heat_addition = c_p * (T[3] - T[2])
    work_turbine = c_p * (T[3] - T[4])
    net_work = work_turbine - work_compressor
    efficiency = net_work / heat_addition * 100

    if net_work < 0:  # or T[4] > 800 + 273.15:
        # print('net_work is negative')
        # set all variables to NaN
        P = [np.nan] * 6
        T = [np.nan] * 6
        s = [np.nan] * 6
        efficiency = np.nan
        net_work = np.nan

    return P, T, s, efficiency, net_work


def _a_r_ref(ao_r_ref, ntu):
    """A/A_ref = (NTU / NTU_MATCH) * (Ao/Ao_ref)**0.587"""
    return (ntu / NTU_MATCH) * (ao_r_ref**0.587)


def _get_eps_dp_from_g2h(g2_h, ntu, c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r_hot, pressure_drop_ratio):
    """Return (eps, dp_hot, dp_cold) for given g2_h and NTU. dp as fraction of inlet pressure."""
    C_min_over_C_hot, C_min_over_C_cold, _ = calculate_capacity_ratios(c_cold_over_c_hot)
    if c_cold_over_c_hot <= 1.0:
        C_ratio = c_cold_over_c_hot
    else:
        C_ratio = 1.0 / c_cold_over_c_hot
    eps = epsilon_ntu(
        np.array([ntu]),
        C_ratio,
        exchanger_type="aligned_flow",
        flow_type="counterflow",
        n_passes=1,
    )[0]
    st_over_f_h = st_over_f
    st_over_f_c = st_over_f
    dp_coeff = g2_h * (
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r_hot * C_min_over_C_cold
    )
    dp_hot = dp_coeff * ntu
    dp_cold = pressure_drop_ratio * dp_hot
    return eps, dp_hot, dp_cold


def _practical_at_ao_ntu_g2h(
    g2_h,
    ntu,
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
):
    """Practical unavailable creation (dQ_o^M/Qmax) given g2_h and NTU."""
    eps, dp_hot, dp_cold = _get_eps_dp_from_g2h(
        g2_h, ntu, c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r_hot, pressure_drop_ratio
    )
    if dp_hot >= dp_max or dp_cold >= dp_max:
        return np.nan, np.nan, np.nan, np.nan
    out = practical_unavailable_creation_hex(
        np.array([eps]),
        t,
        np.array([dp_hot]),
        np.array([dp_cold]),
        np.array([True]),
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
    )
    return float(out[0]) if len(out) > 0 else np.nan, eps, dp_hot, dp_cold


def _compute_mach_and_g2h(mdot, T_hot_in, ao_r_ref, mdot_ref, T_hot_in_ref, gamma):
    """M_in = M_ref * (mdot/mdot_ref) * sqrt(T_hot_in/T_ref) / ao_r_ref; g2h = 0.5*gamma*M^2."""
    M_in = DEFAULT_MACH_IN * (mdot / mdot_ref) * np.sqrt(T_hot_in / T_hot_in_ref) / ao_r_ref
    g2_h = 0.5 * gamma * M_in**2
    return M_in, g2_h


def solve_mdot_at_constant_power(
    ao_r_ref,
    ntu,
    pressure_drop_ratio,
    mdot_ref_val,
    T_hot_in_ref,
    P_ref,
):
    """
    Solve for mdot such that mdot * w_net = P_shaft_ref.
    Uses scipy root on residuals: [mdot - P_ref/w_net, T_hot_in - T[4]].
    Returns (mdot, w_net, T_hot_in, eps, dp_hot, dp_cold, dq_o_m_over_qmax, eta_cycle, Mach_in).
    """
    c_cold_over_c_hot = DEFAULT_C_COLD_OVER_C_HOT
    st_over_f = DEFAULT_ST_OVER_F
    f_c_over_f_h = DEFAULT_F_C_OVER_F_H
    d_r_hot = DEFAULT_D_R_hot
    t = DEFAULT_T
    p_cold_in_over_p_hot_in = DEFAULT_P_COLD_IN_OVER_P_HOT_IN
    p_dead_over_p_hot_in = DEFAULT_P_DEAD_OVER_P_HOT_IN
    gamma = DEFAULT_GAMMA
    dp_max = DEFAULT_DP_MAX

    def _residual(x):
        mdot, T_hot_in = x
        M_in, g2_h = _compute_mach_and_g2h(mdot, T_hot_in, ao_r_ref, mdot_ref_val, T_hot_in_ref, gamma)
        dq, eps, dp_hot, dp_cold = _practical_at_ao_ntu_g2h(
            g2_h,
            ntu,
            c_cold_over_c_hot,
            st_over_f,
            f_c_over_f_h,
            d_r_hot,
            pressure_drop_ratio,
            t,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            gamma,
            dp_max,
        )
        if not np.isfinite(dq):
            return [np.nan, np.nan]
        P, T, s, eff, w_net = calculate_recuperated_cycle_dp_eps(PR, TIT, eta_poly_c, eta_poly_t, eps, dp_hot, dp_cold)
        if not np.isfinite(w_net) or w_net <= 0:
            return [np.nan, np.nan]
        T_hot_out = T[4]
        mdot_eq = P_ref / w_net
        return [mdot - mdot_eq, T_hot_in - T_hot_out]

    sol = root(_residual, [mdot_ref_val, T_hot_in_ref], method="hybr")
    if not sol.success or not np.all(np.isfinite(sol.x)):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mdot, T_hot_in = sol.x
    M_in, g2_h = _compute_mach_and_g2h(mdot, T_hot_in, ao_r_ref, mdot_ref_val, T_hot_in_ref, gamma)
    dq, eps, dp_hot, dp_cold = _practical_at_ao_ntu_g2h(
        g2_h,
        ntu,
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r_hot,
        pressure_drop_ratio,
        t,
        p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in,
        gamma,
        dp_max,
    )
    P, T, s, eff, w_net = calculate_recuperated_cycle_dp_eps(PR, TIT, eta_poly_c, eta_poly_t, eps, dp_hot, dp_cold)
    return mdot, w_net, T_hot_in, eps, dp_hot, dp_cold, dq, eff, M_in


def _optimal_ao_r_ref_for_each_a_r_ref(
    pressure_drop_ratio,
    mdot_ref_val,
    T_hot_in_ref,
    P_ref,
    mdot_baseline,
    eff_baseline,
    factor_fuel,
    include_delta_engine,
):
    """
    For each A/A_ref, sweeps Ao/Ao_ref calling the coupled cycle model ONCE per point.
    Simultaneously tracks optima for three objectives:

      'red'        : cycle-efficiency fuel + m_hex + delta_engine  (always includes delta_engine)
      'mdot_dqom'  : dQ^M (fixed DEFAULT BCs, actual Mach via actual mdot/T_hot_in)
                     * factor_fuel + m_hex  [+ delta_engine if include_delta_engine]
      'bc_mdot_dqom': dQ^M (actual BCs: t=T_hot_in/T_cold_in, p_ratios from dp_hot,
                     actual Mach) * factor_fuel + m_hex  [+ delta_engine if include_delta_engine]

    'mdot_dqom' isolates Effect 2 (changing mdot → changing Mach).
    'bc_mdot_dqom' captures Effects 1+2 combined.
    The dQ^M for 'mdot_dqom' re-uses the dq value already returned by solve_mdot_at_constant_power,
    which internally uses DEFAULT_T and DEFAULT_P_... for the BCs but the actual g2_h from actual mdot.

    Returns a dict of numpy arrays keyed by variant name.
    """
    gamma = DEFAULT_GAMMA
    # Compressor exit temperature — constant for fixed PR/eta_poly_c
    T_cold_in = 288.0 * (PR ** ((gamma - 1) / (gamma * eta_poly_c)))

    mdot_fuel_baseline = P_ref / (LHV_J_per_kg * eff_baseline / 100)

    red = dict(
        a=[],
        ao=[],
        ntu=[],
        mdot=[],
        dq=[],
        m_hex=[],
        delta_engine=[],
        eps=[],
        dp_hot=[],
        dp_cold=[],
        eff=[],
        M_in=[],
        T_hot_in=[],
    )
    mdot_dqom = dict(a=[], ao=[], obj=[])
    bc_mdot_dqom = dict(a=[], ao=[], obj=[])

    for a_r_ref_target in A_R_REF_VALUES:
        m_hex = a_r_ref_target * m_hex_ref

        # Per-A/A_ref bests
        bz_red = np.inf
        bao_red = bNtu = bMdot = bDq = bEps = bDph = bDpc = bEff = bMin = bThi = None
        bz_mdot = np.inf
        bao_mdot = None
        bz_bc = np.inf
        bao_bc = None

        for ao_r_ref in AO_SWEEP:
            ntu = a_r_ref_target * NTU_MATCH / (ao_r_ref**0.587)
            if ntu < 0.02 or ntu > DEFAULT_NTU_MAX:
                continue

            mdot, w_net, T_hot_in, eps, dp_hot, dp_cold, dq, eff, M_in = solve_mdot_at_constant_power(
                ao_r_ref, ntu, pressure_drop_ratio, mdot_ref_val, T_hot_in_ref, P_ref
            )
            if not np.isfinite(dq) or not np.isfinite(mdot) or not np.isfinite(eff) or eff <= 0:
                continue
            if mdot > 50 or mdot < 0.5:
                continue

            delta_engine_val = (mdot - mdot_baseline) * kg_dry_engine_per_kg_per_s_of_air
            de = delta_engine_val if include_delta_engine else 0.0

            # --- Red line (always includes delta_engine) ---
            mdot_fuel = P_ref / (LHV_J_per_kg * eff / 100)
            delta_fuel_cycle = (mdot_fuel - mdot_fuel_baseline) * mission_seconds
            obj_red = delta_fuel_cycle + m_hex + delta_engine_val
            if obj_red < bz_red:
                bz_red = obj_red
                bao_red = ao_r_ref
                bNtu = ntu
                bMdot = mdot
                bDq = dq
                bEps = eps
                bDph = dp_hot
                bDpc = dp_cold
                bEff = eff
                bMin = M_in
                bThi = T_hot_in

            # --- mdot-dQ^M (Effect 2 only: actual Mach, fixed DEFAULT BCs) ---
            # dq from solve_mdot already uses actual g2_h (actual mdot, T_hot_in) but
            # fixed DEFAULT_T and DEFAULT_P_... for the BCs — isolates the mdot/Mach effect.
            obj_mdot = dq * factor_fuel + m_hex + de
            if obj_mdot < bz_mdot:
                bz_mdot = obj_mdot
                bao_mdot = ao_r_ref

            # --- bc-mdot-dQ^M (Effects 1+2: actual BCs + actual Mach) ---
            if np.isfinite(T_hot_in) and np.isfinite(dp_hot):
                t_act = T_hot_in / T_cold_in
                p_dead_act = 1.0 - dp_hot
                p_cold_act = PR * (1.0 - dp_hot)
                dq_bc_raw = practical_unavailable_creation_hex(
                    np.array([eps]),
                    t_act,
                    np.array([dp_hot]),
                    np.array([dp_cold]),
                    np.array([True]),
                    p_cold_in_over_p_hot_in=p_cold_act,
                    p_dead_over_p_hot_in=p_dead_act,
                    gamma=gamma,
                )
                dq_bc = float(dq_bc_raw[0]) if len(dq_bc_raw) > 0 else np.nan
                if np.isfinite(dq_bc):
                    obj_bc = dq_bc * factor_fuel + m_hex + de
                    if obj_bc < bz_bc:
                        bz_bc = obj_bc
                        bao_bc = ao_r_ref

        # Store red line result
        if bao_red is not None:
            red["a"].append(a_r_ref_target)
            red["ao"].append(bao_red)
            red["ntu"].append(bNtu)
            red["mdot"].append(bMdot)
            red["dq"].append(bDq)
            red["m_hex"].append(m_hex)
            red["delta_engine"].append((bMdot - mdot_baseline) * kg_dry_engine_per_kg_per_s_of_air)
            red["eps"].append(bEps)
            red["dp_hot"].append(bDph)
            red["dp_cold"].append(bDpc)
            red["eff"].append(bEff)
            red["M_in"].append(bMin)
            red["T_hot_in"].append(bThi)

        # Store dQ^M variant results
        if bao_mdot is not None:
            mdot_dqom["a"].append(a_r_ref_target)
            mdot_dqom["ao"].append(bao_mdot)
            mdot_dqom["obj"].append(bz_mdot)
        if bao_bc is not None:
            bc_mdot_dqom["a"].append(a_r_ref_target)
            bc_mdot_dqom["ao"].append(bao_bc)
            bc_mdot_dqom["obj"].append(bz_bc)

    # Convert all lists to numpy arrays
    for d in (red, mdot_dqom, bc_mdot_dqom):
        for k in d:
            d[k] = np.array(d[k])

    return {"red": red, "mdot_dqom": mdot_dqom, "bc_mdot_dqom": bc_mdot_dqom}


def _sweep_bc_only(pressure_drop_ratio, factor_fuel):
    """
    Effect 1 (BC) only variant: reference Mach for g2_h → eps/dp, then uncoupled cycle
    to obtain the actual T_hot_in (turbine exit) at those HEx conditions.  Uses actual
    BCs (t = T_hot_in/T_cold_in, p_ratios from dp_hot) in the dQ^M calculation.

    No coupled cycle solver call, so mdot = mdot_ref and delta_engine ≈ 0 by definition.
    Objective: dQ^M * factor_fuel + m_hex  (no delta_engine).

    Returns dict with 'a', 'ao', 'obj' numpy arrays.
    """
    gamma = DEFAULT_GAMMA
    T_cold_in = 288.0 * (PR ** ((gamma - 1) / (gamma * eta_poly_c)))
    g2h_ref = 0.5 * gamma * DEFAULT_MACH_IN**2

    a_out, ao_out, obj_out = [], [], []

    for a_r_ref_target in A_R_REF_VALUES:
        m_hex = a_r_ref_target * m_hex_ref
        best_z = np.inf
        best_ao = None

        for ao_r_ref in AO_SWEEP:
            ntu = a_r_ref_target * NTU_MATCH / (ao_r_ref**0.587)
            if ntu < 0.02 or ntu > DEFAULT_NTU_MAX:
                continue

            # Reference Mach scaled by ao_r_ref only (mdot = mdot_ref)
            g2_h = g2h_ref / ao_r_ref**2
            eps, dp_hot, dp_cold = _get_eps_dp_from_g2h(
                g2_h,
                ntu,
                DEFAULT_C_COLD_OVER_C_HOT,
                DEFAULT_ST_OVER_F,
                DEFAULT_F_C_OVER_F_H,
                DEFAULT_D_R_hot,
                pressure_drop_ratio,
            )
            if dp_hot >= DEFAULT_DP_MAX or dp_cold >= DEFAULT_DP_MAX:
                continue

            # Uncoupled cycle → actual T_hot_in at these eps/dp (no mdot solve)
            _, T_cyc, _, _, w_net_cyc = calculate_recuperated_cycle_dp_eps(
                PR, TIT, eta_poly_c, eta_poly_t, eps, dp_hot, dp_cold
            )
            if not np.isfinite(w_net_cyc) or w_net_cyc <= 0:
                continue
            T_hot_in = T_cyc[4]

            # Actual BCs from cycle at this (eps, dp)
            t_act = T_hot_in / T_cold_in
            p_dead_act = 1.0 - dp_hot
            p_cold_act = PR * (1.0 - dp_hot)

            dq_raw = practical_unavailable_creation_hex(
                np.array([eps]),
                t_act,
                np.array([dp_hot]),
                np.array([dp_cold]),
                np.array([True]),
                p_cold_in_over_p_hot_in=p_cold_act,
                p_dead_over_p_hot_in=p_dead_act,
                gamma=gamma,
            )
            dq = float(dq_raw[0]) if len(dq_raw) > 0 else np.nan
            if not np.isfinite(dq):
                continue

            obj = dq * factor_fuel + m_hex
            if obj < best_z:
                best_z = obj
                best_ao = ao_r_ref

        if best_ao is not None:
            a_out.append(a_r_ref_target)
            ao_out.append(best_ao)
            obj_out.append(best_z)

    return {"a": np.array(a_out), "ao": np.array(ao_out), "obj": np.array(obj_out)}


def run_plot(base_name="fig9_w_cycle_model"):
    """Run the cycle-coupled fig8 analysis and plot three weight-delta lines."""
    sigma_r_hot = DEFAULT_D_R_hot * DEFAULT_A_R_hot
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        DEFAULT_PRESSURE_DROP_ASSUMPTION,
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_T,
        DEFAULT_D_R_hot,
        DEFAULT_MOLAR_MASS_RATIO,
        sigma_r_hot,
        DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    )

    # Reference design: ao_r_ref=1, NTU_MATCH. Get T_hot_in_ref and verify eps, dp.
    g2h_ref = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2
    eps_ref, dp_hot_ref, dp_cold_ref = _get_eps_dp_from_g2h(
        g2h_ref,
        NTU_MATCH,
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R_hot,
        pressure_drop_ratio,
    )
    P_ref_cyc, T_ref_cyc, s, eff_ref, w_net_ref = calculate_recuperated_cycle_dp_eps(
        PR, TIT, eta_poly_c, eta_poly_t, eps_ref, dp_hot_ref, dp_cold_ref
    )
    T_hot_in_ref = T_ref_cyc[4]
    mdot_at_ref = P_shaft_ref / w_net_ref
    # Baseline unrecuperated (mdot chosen to produce P_shaft_ref)
    P_b, T_b, s_b, eff_b, w_net_baseline = calculate_cycle(PR, TIT, eta_poly_c, eta_poly_t)
    mdot_baseline = P_shaft_ref / w_net_baseline
    # P_baseline = mdot_baseline * w_net_baseline

    # Fixed fuel conversion factor (reference Q_max — same for all dQ^M variants)
    cp_hot = 1070.0  # J/(kg·K) — match cycle model
    Th_in_ref = 898
    Tc_in_ref = 588
    Q_max = mdot_at_ref * cp_hot * (Th_in_ref - Tc_in_ref)  # W
    factor_fuel = mission_seconds / LHV_J_per_kg * eta_turb / eta_ov * Q_max  # kg

    # Print active toggle
    de_label = "YES (delta_engine included)" if INCLUDE_DELTA_ENGINE else "NO  (delta_engine excluded)"
    print(f"\n{'=' * 65}")
    print(f"  INCLUDE_DELTA_ENGINE = {INCLUDE_DELTA_ENGINE}  ->  {de_label}")
    print(f"  (delta_engine = (mdot - mdot_baseline) * {kg_dry_engine_per_kg_per_s_of_air} kg/(kg/s))")
    print(f"  factor_fuel  = {factor_fuel:.2f} kg  (fixed reference Q_max conversion)")
    print(f"{'=' * 65}\n")

    # --- Main coupled sweep: red, mdot-dQ^M, bc-mdot-dQ^M variants ---
    print("Running coupled cycle sweep (red / mdot-dQ^M / bc-mdot-dQ^M)...")
    res = _optimal_ao_r_ref_for_each_a_r_ref(
        pressure_drop_ratio,
        mdot_at_ref,
        T_hot_in_ref,
        P_shaft_ref,
        mdot_baseline,
        eff_b,
        factor_fuel,
        INCLUDE_DELTA_ENGINE,
    )
    red_r = res["red"]

    if len(red_r["a"]) == 0:
        print("No valid optimum points found.")
        return

    # Unpack red line arrays for backward-compatible table/plot code
    a_r_ref_opt = red_r["a"]
    ao_r_ref_opt = red_r["ao"]
    ntu_opt = red_r["ntu"]
    mdot_opt = red_r["mdot"]
    dq_o_m_opt = red_r["dq"]
    m_hex_opt = red_r["m_hex"]
    delta_engine_opt = red_r["delta_engine"]
    eps_opt = red_r["eps"]
    dp_hot_opt = red_r["dp_hot"]
    dp_cold_opt = red_r["dp_cold"]
    eff_opt = red_r["eff"]
    M_in_opt = red_r["M_in"]
    T_hot_in_opt = red_r["T_hot_in"]

    # --- Effect 1 (BC only) sweep: uncoupled cycle, reference Mach ---
    print("Running BC-only sweep (uncoupled cycle, ref Mach)...")
    res_bc = _sweep_bc_only(pressure_drop_ratio, factor_fuel)

    if len(res["mdot_dqom"]["a"]) == 0:
        print("No valid optimum points found.")
        return

    # Reference ao=1 case: solve coupled model and verify eps, dp, mass deltas
    (
        mdot_ref_coupled,
        w_net_ref_c,
        T_hot_ref_c,
        eps_ref_c,
        dp_hot_ref_c,
        dp_cold_ref_c,
        dq_ref,
        eff_ref_c,
        M_in_ref_c,
    ) = solve_mdot_at_constant_power(1.0, NTU_MATCH, pressure_drop_ratio, mdot_at_ref, T_hot_in_ref, P_shaft_ref)
    a_r_ref_at_1 = _a_r_ref(1.0, NTU_MATCH) if np.isfinite(dq_ref) else np.nan
    m_hex_at_ref = a_r_ref_at_1 * m_hex_ref if np.isfinite(dq_ref) else np.nan
    if np.isfinite(dq_ref):
        mdot_fuel_ref = P_shaft_ref / (LHV_J_per_kg * eff_ref_c / 100)
        mdot_fuel_baseline_ref = P_shaft_ref / (LHV_J_per_kg * eff_b / 100)
        delta_fuel_ref = (mdot_fuel_ref - mdot_fuel_baseline_ref) * mission_seconds
        delta_hex_ref = m_hex_at_ref
        delta_engine_ref = (mdot_ref_coupled - mdot_baseline) * kg_dry_engine_per_kg_per_s_of_air
        line3_ref = delta_fuel_ref + delta_hex_ref + delta_engine_ref
    else:
        delta_fuel_ref = delta_hex_ref = delta_engine_ref = line3_ref = np.nan

    # Sweep point closest to A/A_ref=1 (optimizer may pick ao_r_ref != 1)
    id_close = np.argmin(np.abs(a_r_ref_opt - 1.0))

    # Weight deltas vs baseline unrecuperated — red line (cycle efficiency)
    mdot_fuel_baseline = P_shaft_ref / (LHV_J_per_kg * eff_b / 100)
    mdot_fuel_opt = P_shaft_ref / (LHV_J_per_kg * eff_opt / 100)
    delta_fuel = (mdot_fuel_opt - mdot_fuel_baseline) * mission_seconds
    # dQ^M at red-line's optimal Ao/Ao_ref (fixed DEFAULT BCs, actual Mach) — for table
    delta_fuel_dqom = dq_o_m_opt * factor_fuel
    delta_hex = m_hex_opt
    delta_engine = (mdot_opt - mdot_baseline) * kg_dry_engine_per_kg_per_s_of_air

    # Red line curves
    line1 = delta_fuel  # fuel savings only
    line2 = delta_fuel + delta_hex  # fuel + HEx
    line3 = delta_fuel + delta_hex + delta_engine  # fuel + HEx + engine
    id_min_red = int(np.argmin(line3))

    # dQ^M variant objective lines (pre-aligned to red line's A/A_ref grid via shared loop)
    # mdot-dQ^M: Effect 2 only (actual Mach, fixed DEFAULT BCs)
    line_mdot = res["mdot_dqom"]["obj"]  # optimal obj per A/A_ref
    id_min_mdot = int(np.nanargmin(line_mdot))

    # bc-mdot-dQ^M: Effects 1+2 (actual BCs + actual Mach)
    line_bc_mdot = res["bc_mdot_dqom"]["obj"]
    id_min_bc_mdot = int(np.nanargmin(line_bc_mdot))

    # --- Evaluate black star's (ao, A/A_ref) through the full red cycle model ---
    ao_black_star = res["bc_mdot_dqom"]["ao"][id_min_bc_mdot]
    a_r_black_star = res["bc_mdot_dqom"]["a"][id_min_bc_mdot]
    ntu_black_star = a_r_black_star * NTU_MATCH / (ao_black_star**0.587)
    (mdot_bs, w_net_bs, T_hi_bs, eps_bs, dph_bs, dpc_bs, dq_bs, eff_bs, Min_bs) = solve_mdot_at_constant_power(
        ao_black_star, ntu_black_star, pressure_drop_ratio, mdot_at_ref, T_hot_in_ref, P_shaft_ref
    )
    if np.isfinite(eff_bs) and eff_bs > 0:
        mf_bs = P_shaft_ref / (LHV_J_per_kg * eff_bs / 100)
        df_bs = (mf_bs - P_shaft_ref / (LHV_J_per_kg * eff_b / 100)) * mission_seconds
        de_bs = (mdot_bs - mdot_baseline) * kg_dry_engine_per_kg_per_s_of_air
        line3_bs = df_bs + a_r_black_star * m_hex_ref + de_bs  # on the red model scale
        # Red line value at same A/A_ref (interpolated)
        line3_red_at_bs = float(np.interp(a_r_black_star, a_r_ref_opt, line3))
        delta_subopt = line3_bs - line3_red_at_bs  # positive → black star is worse than red optimum
    else:
        line3_bs = line3_red_at_bs = delta_subopt = np.nan

    # BC-only (Effect 1): uncoupled cycle, ref Mach — may have different A/A_ref grid
    line_bc = res_bc["obj"]
    m_hex_bc = res_bc["a"] * m_hex_ref
    id_min_bc = int(np.nanargmin(line_bc)) if len(line_bc) > 0 else None

    # Optimal HEx mass for each variant (for printed comparison)
    def _opt_m_hex(idx, arr_a):
        return arr_a[idx] * m_hex_ref if idx is not None and len(arr_a) > idx else np.nan

    opt_mhex_red = m_hex_opt[id_min_red]
    opt_mhex_mdot = _opt_m_hex(id_min_mdot, res["mdot_dqom"]["a"])
    opt_mhex_bc_mdot = _opt_m_hex(id_min_bc_mdot, res["bc_mdot_dqom"]["a"])
    opt_mhex_bc = _opt_m_hex(id_min_bc, res_bc["a"])

    print(f"\n{'-' * 65}")
    print(f"  VARIANT COMPARISON  (INCLUDE_DELTA_ENGINE={INCLUDE_DELTA_ENGINE})")
    print(f"  {'Variant':<42}  {'Opt m_HEx(kg)':>13}  {'Min dm(kg)':>10}")
    print(f"  {'-' * 67}")
    de_note = "+dEng" if INCLUDE_DELTA_ENGINE else "     "
    bc_min_obj = line_bc[id_min_bc] if id_min_bc is not None else float("nan")
    print(f"  {'Fig 8 style (neither effect - see fig8)':42}  {'(see fig8)':>13}  {'(see fig8)':>10}")
    print(f"  {'Effect 1 only: BC (uncoupled cyc, ref mdot)':42}  {opt_mhex_bc:>13.2f}  {bc_min_obj:>10.2f}")
    print(f"  {f'Effect 2 only: mdot/Mach {de_note}':42}  {opt_mhex_mdot:>13.2f}  {line_mdot[id_min_mdot]:>10.2f}")
    print(
        f"  {f'Effects 1+2:   BC+mdot/Mach {de_note}':42}  {opt_mhex_bc_mdot:>13.2f}  {line_bc_mdot[id_min_bc_mdot]:>10.2f}"
    )
    print(f"  {'Red line (cycle efficiency, always +dEng)':42}  {opt_mhex_red:>13.2f}  {line3[id_min_red]:>10.2f}")
    print(f"  {'-' * 67}")
    print(f"\n  Black star (BC+mdot opt) plugged into RED cycle model:")
    print(f"    A/A_ref = {a_r_black_star:.3f},  Ao/Ao_ref = {ao_black_star:.3f},  NTU = {ntu_black_star:.3f}")
    if np.isfinite(line3_bs):
        print(f"    Red-model value at black-star design:    {line3_bs:>8.2f} kg")
        print(f"    Red-model optimum at same A/A_ref:       {line3_red_at_bs:>8.2f} kg")
        print(f"    Sub-optimality (black -> red cost delta): {delta_subopt:>+8.2f} kg")
    else:
        print(f"    Could not evaluate black star through red model (solver failed).")
    print(f"  {'-' * 67}\n")

    def _val(i, key):
        """Get value for point i (index into sweep arrays) or 'ref'/'unrecup' for special designs."""
        if i == "unrecup":
            # Baseline unrecuperated: cycle outputs only, rest N/A
            return {
                "eta_cycle": eff_b,
                "w_net": w_net_baseline / 1e3,
                "mdot": mdot_baseline,
            }
        if i == "ref":
            if not np.isfinite(dq_ref):
                return "—"
            return {
                "A/A_ref": a_r_ref_at_1,
                "m_hex": m_hex_at_ref,
                "ao/ao_ref": 1.0,
                "Mach_in": M_in_ref_c,
                "NTU": NTU_MATCH,
                "eps": eps_ref_c * 100,
                "eps_P": -dq_ref * 100,
                "dph": dp_hot_ref_c * 100,
                "dpc": dp_cold_ref_c * 100,
                "eta_cycle": eff_ref_c,
                "w_net": w_net_ref_c / 1e3,
                "mdot": mdot_ref_coupled,
                "delta_fuel": delta_fuel_ref,
                "delta_hex": delta_hex_ref,
                "delta_engine": delta_engine_ref,
                "cum_fuel": delta_fuel_ref,
                "cum_fuel_hex": delta_fuel_ref + delta_hex_ref,
                "cum_total": line3_ref,
                "cum_fuel_dqom": dq_ref * factor_fuel,
                "cum_fuel_hex_dqom": dq_ref * factor_fuel + delta_hex_ref,
            }
        if not isinstance(i, int) or i < -len(m_hex_opt) or i >= len(m_hex_opt):
            return "—"
        return {
            "A/A_ref": a_r_ref_opt[i],
            "m_hex": m_hex_opt[i],
            "ao/ao_ref": ao_r_ref_opt[i],
            "Mach_in": M_in_opt[i],
            "NTU": ntu_opt[i],
            "eps": eps_opt[i] * 100,
            "eps_P": -dq_o_m_opt[i] * 100,
            "dph": dp_hot_opt[i] * 100,
            "dpc": dp_cold_opt[i] * 100,
            "eta_cycle": eff_opt[i],
            "w_net": (P_shaft_ref / mdot_opt[i]) / 1e3,
            "mdot": mdot_opt[i],
            "delta_fuel": delta_fuel[i],
            "delta_hex": delta_hex[i],
            "delta_engine": delta_engine[i],
            "cum_fuel": line1[i],
            "cum_fuel_hex": line2[i],
            "cum_total": line3[i],
            "cum_fuel_dqom": delta_fuel_dqom[i],
            "cum_fuel_hex_dqom": delta_fuel_dqom[i] + delta_hex[i],
        }

    def _cell(i, key):
        if key == "ref":
            d = _val("ref", key)
        elif i == "unrecup":
            d = _val("unrecup", key)
        else:
            d = _val(i, key)
        if d == "—":
            return "—"
        if isinstance(d, dict) and key not in d:
            return "—"
        v = d[key] if isinstance(d, dict) else d
        if isinstance(v, float) and np.isnan(v):
            return "—"
        if key in ("dph", "dpc", "eps", "eps_P"):
            return f"{v:.2f}%"
        if key == "eta_cycle":
            return f"{v:.2f}%"
        if key in ("Mach_in", "NTU", "ao/ao_ref", "A/A_ref"):
            return f"{v:.4f}" if abs(v) < 1e-3 or abs(v) > 1e4 else f"{v:.3f}"
        if key in (
            "w_net",
            "m_hex",
            "mdot",
            "delta_fuel",
            "delta_hex",
            "delta_engine",
            "cum_fuel",
            "cum_fuel_hex",
            "cum_total",
            "cum_fuel_dqom",
            "cum_fuel_hex_dqom",
        ):
            return f"{v:.3f}"
        return str(v)

    # Build table: rows = quantities, columns = points
    points = [
        ("1st", 0 if len(m_hex_opt) >= 1 else None),
        ("ref", "ref"),
        ("ref_opt", int(id_close) if np.isfinite(dq_ref) and id_close < len(a_r_ref_opt) else None),
        ("square", id_min_red),  # red square: cycle-model optimum
        ("star", id_min_bc_mdot),  # black star: BC+mdot dQ^M optimum
        ("last", -1 if len(m_hex_opt) >= 2 else None),
        ("unrecup", "unrecup"),
    ]
    point_cols = [p[0] for p in points]

    def _build_row(label, key):
        row = [label]
        for _, pi in points:
            if pi is None:
                row.append("—")
            elif pi == "ref":
                if key == "ref":
                    row.append("ref")
                else:
                    row.append(_cell("ref", key))
            elif pi == "unrecup":
                row.append(_cell("unrecup", key))
            else:
                row.append(_cell(pi, key))
        return row

    table_rows = [
        ["--- INPUTS ---", "", "", "", "", "", "", ""],
        _build_row("A/A_ref", "A/A_ref"),
        _build_row("m_hex (kg)", "m_hex"),
        _build_row("ao/ao_ref", "ao/ao_ref"),
        ["--- HEx ---", "", "", "", "", "", "", ""],
        _build_row("Mach_in", "Mach_in"),
        _build_row("NTU", "NTU"),
        _build_row("eps (%)", "eps"),
        _build_row("eps^P (%)", "eps_P"),
        _build_row("dph (%)", "dph"),
        _build_row("dpc (%)", "dpc"),
        ["--- HEx model ---", "", "", "", "", "", "", ""],
        _build_row("cum_fuel (dQo^M)", "cum_fuel_dqom"),
        _build_row("cum_fuel+hex (dQo^M)", "cum_fuel_hex_dqom"),
        ["--- CYCLE OUTPUTS ---", "", "", "", "", "", "", ""],
        _build_row("eta_cycle (%)", "eta_cycle"),
        _build_row("w_net (kJ/kg)", "w_net"),
        _build_row("mdot (kg/s)", "mdot"),
        ["--- DELTAS (kg) ---", "", "", "", "", "", "", ""],
        _build_row("delta_fuel", "delta_fuel"),
        _build_row("delta_hex", "delta_hex"),
        _build_row("delta_engine", "delta_engine"),
        ["--- CUMULATIVE (kg) ---", "", "", "", "", "", "", ""],
        _build_row("cum_fuel", "cum_fuel"),
        _build_row("cum_fuel+hex", "cum_fuel_hex"),
        _build_row("cum_total", "cum_total"),
    ]
    print("\n" + tabulate(table_rows, headers=["", *point_cols], tablefmt="simple", stralign="right"))

    # Plot
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "mathtext.fontset": "stix",
        }
    )
    fig, ax = plt.subplots(figsize=(9 / 2.54, 7 / 2.54))

    # Red lines — full cycle model (always includes delta_engine)
    ax.plot(m_hex_opt, line1, "r--", linewidth=1.5, label="Fuel saving")
    ax.plot(m_hex_opt, line3, "r-", linewidth=1.5, label=r"$\Delta$Fuel + m$_{\mathrm{HEx}}$ + $\Delta$Engine")

    # Black lines — dQ^M variants (hidden when HIDE_BLACK_LINES=True)
    if not HIDE_BLACK_LINES:
        if len(line_bc) > 0:
            ax.plot(m_hex_bc, line_bc, "k--", linewidth=1.2, label=r"$\Delta\dot{W}^M$: BC effect only")
        ax.plot(
            m_hex_opt,
            line_bc_mdot,
            "k-",
            linewidth=1.5,
            label=r"$\Delta\dot{W}^M$: BC + $\dot{m}$" + (r" + $\Delta$Eng" if INCLUDE_DELTA_ENGINE else ""),
        )

    # Red square: red-line optimum
    ax.scatter(
        m_hex_opt[id_min_red],
        line3[id_min_red],
        color="red",
        s=25,
        zorder=6,
        marker="s",
        facecolor="red",
        edgecolor="white",
        linewidths=1,
    )
    # Black star: BC+mdot dQ^M optimum on the black line
    ax.scatter(
        a_r_black_star * m_hex_ref,
        line_bc_mdot[id_min_bc_mdot],
        color="black",
        s=90,
        zorder=5,
        marker="*",
        facecolor="black",
        edgecolor="white",
        linewidths=1,
    )
    # Circle on red line showing where the black-star design lands on the cycle model
    if np.isfinite(line3_bs):
        ax.scatter(
            a_r_black_star * m_hex_ref,
            line3_bs,
            color="red",
            s=90,
            zorder=4,
            marker="*",
            facecolor="red",
            edgecolor="white",
            linewidths=1,
        )
    if np.isfinite(dq_ref):
        line3_ref = delta_fuel_ref + delta_hex_ref + delta_engine_ref
        ax.scatter(
            m_hex_at_ref,
            line3_ref,
            color="red",
            s=25,
            zorder=5,
            marker="+",
            linewidths=0.7,
        )

    ax.set_xlabel(r"Heat Exchanger (HEx) Core Mass $m_{\mathrm{HEx}}$ (kg)")
    ax.set_ylabel(r"Change in Mass $\Delta m$ (kg)")
    legend_handles = [
        Line2D([], [], color="r", linestyle="-", linewidth=1.5, label=r"fuel + HEx + engine"),
        Line2D([], [], color="r", linestyle="--", linewidth=1.5, label="fuel only"),
    ]
    if not HIDE_BLACK_LINES:
        legend_handles += [
            Line2D([], [], color="k", linestyle="--", linewidth=1.2, label=r"$\Delta\dot{W}^M$: BC effect only"),
            Line2D(
                [],
                [],
                color="k",
                linestyle="-",
                linewidth=1.5,
                label=r"$\Delta\dot{W}^M$: BC + $\dot{m}$" + (" + dEng" if INCLUDE_DELTA_ENGINE else ""),
            ),
        ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=1,
        fontsize=FONT_SIZE,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.set_xlim(0, 60)
    ax.set_ylim(-100, 0)
    ax.set_yticks(np.arange(-100, 1, 20))

    # Annotations
    ax.annotate(
        "optimal design A for fuel\n burn from cycle $\eta$",
        xy=(m_hex_opt[id_min_red], line3[id_min_red]),
        xytext=(m_hex_opt[id_min_red] - 16, -78),
        fontsize=FONT_SIZE,
        zorder=6,
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )
    if np.isfinite(dq_ref):
        ax.annotate(
            "baseline design",
            xy=(m_hex_at_ref, line3_ref),
            xytext=(m_hex_at_ref - 10, line3_ref + 15),
            fontsize=FONT_SIZE,
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )
    # r"optimal: $\sum\Delta\dot{W}^{\mathrm{M}}_{\mathrm{A}}$ with\n $\Delta m_{\mathrm{f}} \propto \sum\Delta\dot{W}^{\mathrm{M}}_{\mathrm{A}}$",
    ax.annotate(
        "optimal design B for fuel\n" + r" burn from $\sum\Delta\dot{W}^{\mathrm{M}}_{\mathrm{A}}$",
        xy=(a_r_black_star * m_hex_ref, line_bc_mdot[id_min_bc_mdot]),
        xytext=(a_r_black_star * m_hex_ref + 4, line_bc_mdot[id_min_bc_mdot] - 12),
        fontsize=FONT_SIZE,
        zorder=6,
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )
    if np.isfinite(line3_bs):
        ax.annotate(
            "design B with fuel burn\n from cycle $\eta$",
            xy=(a_r_black_star * m_hex_ref, line3_bs),
            xytext=(a_r_black_star * m_hex_ref + 4, line3_bs + 8),
            fontsize=FONT_SIZE,
            zorder=6,
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )

    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png", "pdf"]:
        path = os.path.join(save_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    run_plot(base_name="fig9_w_cycle_model")
