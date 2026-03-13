"""
Fig8-style cycle-coupled HEx analysis.
Uses xflow helicopter parameters, g2h = 0.5*gamma*M^2, solves for mdot at constant shaft power.
Three weight-delta lines vs baseline unrecuperated engine: delta fuel, delta fuel+hex, delta fuel+hex+engine.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from xflow import (
    calculate_capacity_ratios,
    calculate_pressure_drop_ratio,
    practical_unavailable_creation_hex,
)

from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figs_current")


effectiveness = 0.65
dp_hot = 0.04  # 0.036
dp_cold = 0.02  # 0.022

PR = 9.0
TIT = 1500
eta_poly_c = 0.88
eta_poly_t = 0.84

kg_dry_engine_per_kg_per_s_of_air = 23.0
# m_engine_no_HEx = kg_dry_engine_per_kg_per_s_of_air * mdot_air


# --- Helicopter parameters from xflow.py (lines 79-92) ---
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_C_COLD_OVER_C_HOT = 1.0
DEFAULT_D_R = 0.257
DEFAULT_GAMMA = 1.4
DEFAULT_MACH_IN = 0.1
# g2h = 0.5 * gamma * M^2 (used only for reference Mach; we compute g2h from M dynamically)
DEFAULT_ST_OVER_F = 0.4
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_T = 898 / 588
DEFAULT_T_DEAD_OVER_T_COLD_IN = 288 / 588
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 8.82 / 1.04
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.04
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
TARGET_EPS = 0.65
NTU_MATCH = 1.824
DEFAULT_NTU_MAX = 8.0
DEFAULT_DP_MAX = 0.2
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 1.0

# Baseline and power reference (from plan)
# Cycle model net_work is J/kg (c_p in J/kg/K, T in K => work in J/kg)
# P = mdot * w_net => W = (kg/s) * (J/kg) = J/s = W
mdot_ref = 2.3  # kg/s
w_net_ref = 303e3  # J/kg (303 kJ/kg)
P_shaft_ref = 697e3  # W (~700 kW)

# Sweep parameters
AO_SWEEP = np.linspace(0.15, 5, 100)  # extend to allow small m_hex (down to 0.1 kg)
m_hex_ref = 13.3  # kg
A_OVER_A_REF_VALUES = np.linspace(0.1 / m_hex_ref, 5.0, 80)  # start at m_hex = 0.1 kg

# Mission parameters
mission_hours = 2
mission_seconds = mission_hours * 3600
LHV_J_per_kg = 43.2e6  # J/kg (43.2 MJ/kg)
eta_turb = 0.84
eta_ov = 0.434


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


def _a_over_a_ref(ao_over_ao_ref, ntu):
    """A/A_ref = (NTU / NTU_MATCH) * (ao_over_ao_ref)**0.587"""
    return (ntu / NTU_MATCH) * (ao_over_ao_ref**0.587)


def _get_eps_dp_from_g2h(g2_h, ntu, c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, pressure_drop_ratio):
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
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
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
    d_r,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
):
    """Practical unavailable creation (dQ_o^M/Qmax) given g2_h and NTU."""
    eps, dp_hot, dp_cold = _get_eps_dp_from_g2h(
        g2_h, ntu, c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, pressure_drop_ratio
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


def _compute_mach_and_g2h(mdot, T_hot_in, ao_over_ao_ref, mdot_ref, T_hot_in_ref, gamma):
    """M_in = M_ref * (mdot/mdot_ref) * sqrt(T_hot_in/T_ref) / (A_o/A_o_ref); g2h = 0.5*gamma*M^2."""
    M_in = DEFAULT_MACH_IN * (mdot / mdot_ref) * np.sqrt(T_hot_in / T_hot_in_ref) / ao_over_ao_ref
    g2_h = 0.5 * gamma * M_in**2
    return M_in, g2_h


def solve_mdot_at_constant_power(
    ao_over_ao_ref,
    ntu,
    pressure_drop_ratio,
    mdot_ref_val,
    T_hot_in_ref,
    P_ref,
    max_iter=15,
    tol=1e-6,
):
    """
    Solve for mdot such that mdot * w_net = P_shaft_ref.
    Returns (mdot, w_net, T_hot_in, eps, dp_hot, dp_cold, dq_o_m_over_qmax, eta_cycle, Mach_in).
    """
    c_cold_over_c_hot = DEFAULT_C_COLD_OVER_C_HOT
    st_over_f = DEFAULT_ST_OVER_F
    f_c_over_f_h = DEFAULT_F_C_OVER_F_H
    d_r = DEFAULT_D_R
    t = DEFAULT_T
    p_cold_in_over_p_hot_in = DEFAULT_P_COLD_IN_OVER_P_HOT_IN
    p_dead_over_p_hot_in = DEFAULT_P_DEAD_OVER_P_HOT_IN
    gamma = DEFAULT_GAMMA
    dp_max = DEFAULT_DP_MAX

    mdot = mdot_ref_val
    T_hot_in = T_hot_in_ref

    for _ in range(max_iter):
        M_in, g2_h = _compute_mach_and_g2h(mdot, T_hot_in, ao_over_ao_ref, mdot_ref_val, T_hot_in_ref, gamma)
        dq, eps, dp_hot, dp_cold = _practical_at_ao_ntu_g2h(
            g2_h,
            ntu,
            c_cold_over_c_hot,
            st_over_f,
            f_c_over_f_h,
            d_r,
            pressure_drop_ratio,
            t,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            gamma,
            dp_max,
        )
        if not np.isfinite(dq):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        P, T, s, eff, w_net = calculate_recuperated_cycle_dp_eps(PR, TIT, eta_poly_c, eta_poly_t, eps, dp_hot, dp_cold)
        if not np.isfinite(w_net) or w_net <= 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        # Cycle model returns w_net in J/kg; guard against kJ/kg from alternate codebases
        if 0 < w_net < 1000:
            w_net = w_net * 1000
        T_hot_in_new = T[4]  # turbine exit = hot side inlet to recuperator
        mdot_new = P_ref / w_net  # W / (J/kg) = kg/s

        if abs(mdot_new - mdot) < tol and abs(T_hot_in_new - T_hot_in) < 0.1:
            break
        mdot = mdot_new
        T_hot_in = T_hot_in_new

    M_in, _ = _compute_mach_and_g2h(mdot, T_hot_in, ao_over_ao_ref, mdot_ref_val, T_hot_in_ref, gamma)
    return mdot, w_net, T_hot_in, eps, dp_hot, dp_cold, dq, eff, M_in


def _optimal_ao_for_each_a_over_a_ref(
    pressure_drop_ratio,
    mdot_ref_val,
    T_hot_in_ref,
    P_ref,
    mdot_baseline,
    eff_baseline,
):
    """
    For each A/A_ref, find best ao (and NTU) minimizing delta_fuel_cycle + delta_hex + delta_engine.
    Delta fuel from cycle: (P_shaft/(LHV*eta) - P_shaft/(LHV*eta_baseline)) * mission_seconds.
    Returns (a_over_a_ref, ao_opt, ntu_opt, mdot, dq, ...).
    """
    a_out, ao_out, ntu_out = [], [], []
    mdot_out, dq_out, m_hex_out, delta_engine_out = [], [], [], []
    eps_out, dp_hot_out, dp_cold_out, eff_out, M_in_out = [], [], [], [], []
    mdot_fuel_baseline = P_ref / (LHV_J_per_kg * eff_baseline / 100)

    for a_target in A_OVER_A_REF_VALUES:
        best_z = np.inf
        best_ao = best_ntu = best_mdot = best_dq = None
        best_eps = best_dp_hot = best_dp_cold = best_eff = best_M_in = None

        for ao in AO_SWEEP:
            ntu = a_target * NTU_MATCH / (ao**0.587)
            if ntu < 0.02 or ntu > DEFAULT_NTU_MAX:  # allow small m_hex (down to 0.1 kg)
                continue

            mdot, w_net, T_hot_in, eps, dp_hot, dp_cold, dq, eff, M_in = solve_mdot_at_constant_power(
                ao, ntu, pressure_drop_ratio, mdot_ref_val, T_hot_in_ref, P_ref
            )
            if not np.isfinite(dq) or not np.isfinite(mdot) or not np.isfinite(eff) or eff <= 0:
                continue

            mdot_fuel = P_ref / (LHV_J_per_kg * eff / 100)
            delta_fuel_cycle = (mdot_fuel - mdot_fuel_baseline) * mission_seconds
            m_hex = a_target * m_hex_ref
            delta_engine = (mdot - mdot_baseline) * kg_dry_engine_per_kg_per_s_of_air
            total = delta_fuel_cycle + m_hex + delta_engine

            if total < best_z:
                best_z = total
                best_ao = ao
                best_ntu = ntu
                best_mdot = mdot
                best_dq = dq
                best_eps = eps
                best_dp_hot = dp_hot
                best_dp_cold = dp_cold
                best_eff = eff
                best_M_in = M_in

        if best_ao is not None:
            # delta_engine vs baseline (computed in run_plot); here use mdot_ref_val for iteration
            # Sanity: mdot should be ~2-5 kg/s for this engine class
            if best_mdot > 50 or best_mdot < 0.5:
                continue  # skip unphysical mdot (unit error elsewhere)
            a_out.append(a_target)
            ao_out.append(best_ao)
            ntu_out.append(best_ntu)
            mdot_out.append(best_mdot)
            dq_out.append(best_dq)
            m_hex_out.append(a_target * m_hex_ref)
            delta_engine_out.append((best_mdot - mdot_baseline) * kg_dry_engine_per_kg_per_s_of_air)
            eps_out.append(best_eps)
            dp_hot_out.append(best_dp_hot)
            dp_cold_out.append(best_dp_cold)
            eff_out.append(best_eff)
            M_in_out.append(best_M_in)

    return (
        np.array(a_out),
        np.array(ao_out),
        np.array(ntu_out),
        np.array(mdot_out),
        np.array(dq_out),
        np.array(m_hex_out),
        np.array(delta_engine_out),
        np.array(eps_out),
        np.array(dp_hot_out),
        np.array(dp_cold_out),
        np.array(eff_out),
        np.array(M_in_out),
    )


def run_plot(base_name="fig9_w_cycle_model"):
    """Run the cycle-coupled fig8 analysis and plot three weight-delta lines."""
    sigma_r = DEFAULT_D_R * DEFAULT_A_R
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        DEFAULT_PRESSURE_DROP_ASSUMPTION,
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_T,
        DEFAULT_D_R,
        DEFAULT_MOLAR_MASS_RATIO,
        sigma_r,
        DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    )

    # Reference design: ao=1, NTU_MATCH. Get T_hot_in_ref and verify eps, dp.
    g2h_ref = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2
    eps_ref, dp_hot_ref, dp_cold_ref = _get_eps_dp_from_g2h(
        g2h_ref,
        NTU_MATCH,
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        pressure_drop_ratio,
    )
    P_ref_cyc, T_ref_cyc, s, eff_ref, w_net_ref = calculate_recuperated_cycle_dp_eps(
        PR, TIT, eta_poly_c, eta_poly_t, eps_ref, dp_hot_ref, dp_cold_ref
    )
    if 0 < w_net_ref < 1000:
        w_net_ref = w_net_ref * 1000  # guard: assume kJ/kg from alternate codebase
    T_hot_in_ref = T_ref_cyc[4]
    mdot_at_ref = P_shaft_ref / w_net_ref
    # Baseline unrecuperated (mdot chosen to produce P_shaft_ref)
    P_b, T_b, s_b, eff_b, w_net_baseline = calculate_cycle(PR, TIT, eta_poly_c, eta_poly_t)
    if 0 < w_net_baseline < 1000:
        w_net_baseline = w_net_baseline * 1000  # guard: assume kJ/kg from alternate codebase
    mdot_baseline = P_shaft_ref / w_net_baseline
    # P_baseline = mdot_baseline * w_net_baseline

    # For red line (dQ_o^M based): factor to convert dQ_o^M/Qmax to fuel mass delta
    lhv_kwh_per_kg = 43.2 / 3.6
    mdot_hot_ref = mdot_at_ref
    cp_hot = 1.07  # kJ/(kg*K) — high-temp air value (compromise between 1004 and 1170)
    Th_in_ref = 898
    Tc_in_ref = 588
    Q_max = mdot_hot_ref * cp_hot * (Th_in_ref - Tc_in_ref)  # kW
    factor_fuel = mission_hours / lhv_kwh_per_kg * eta_turb / eta_ov * Q_max  # kg (for dQ_o^M scaling)

    # Sweep and optimize
    (
        a_over_a_ref_opt,
        ao_opt,
        ntu_opt,
        mdot_opt,
        dq_o_m_opt,
        m_hex_opt,
        delta_engine_opt,
        eps_opt,
        dp_hot_opt,
        dp_cold_opt,
        eff_opt,
        M_in_opt,
    ) = _optimal_ao_for_each_a_over_a_ref(
        pressure_drop_ratio,
        mdot_at_ref,
        T_hot_in_ref,
        P_shaft_ref,
        mdot_baseline,
        eff_b,
    )

    if len(a_over_a_ref_opt) == 0:
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
    a_over_a_ref_at_1 = _a_over_a_ref(1.0, NTU_MATCH) if np.isfinite(dq_ref) else np.nan
    m_hex_at_ref = a_over_a_ref_at_1 * m_hex_ref if np.isfinite(dq_ref) else np.nan
    if np.isfinite(dq_ref):
        mdot_fuel_ref = P_shaft_ref / (LHV_J_per_kg * eff_ref_c / 100)
        mdot_fuel_baseline_ref = P_shaft_ref / (LHV_J_per_kg * eff_b / 100)
        delta_fuel_ref = (mdot_fuel_ref - mdot_fuel_baseline_ref) * mission_seconds
        delta_hex_ref = m_hex_at_ref
        delta_engine_ref = (mdot_ref_coupled - mdot_baseline) * kg_dry_engine_per_kg_per_s_of_air
        line3_ref = delta_fuel_ref + delta_hex_ref + delta_engine_ref
    else:
        delta_fuel_ref = delta_hex_ref = delta_engine_ref = line3_ref = np.nan

    # Sweep point closest to A/A_ref=1 (optimizer may pick ao != 1)
    id_close = np.argmin(np.abs(a_over_a_ref_opt - 1.0))

    # Weight deltas vs baseline unrecuperated
    # Delta fuel from cycle efficiency: mdot_fuel = P_shaft/(LHV*eta), delta = (mdot_fuel - mdot_fuel_baseline)*mission_seconds
    mdot_fuel_baseline = P_shaft_ref / (LHV_J_per_kg * eff_b / 100)
    mdot_fuel_opt = P_shaft_ref / (LHV_J_per_kg * eff_opt / 100)
    delta_fuel = (mdot_fuel_opt - mdot_fuel_baseline) * mission_seconds
    # Red line: old dQ_o^M based delta fuel (for comparison only)
    # Offset so dashed red = dashed black at reference mass (m_hex_ref_design)
    delta_fuel_dqom = dq_o_m_opt * factor_fuel
    if np.isfinite(dq_ref) and False:
        delta_fuel_dqom_ref = dq_ref * factor_fuel
        shift_kg = delta_fuel_ref - delta_fuel_dqom_ref
        delta_fuel_dqom = delta_fuel_dqom + shift_kg
        print(f"\nRed line offset at reference m_hex={m_hex_at_ref:.2f} kg: shift = {shift_kg:.2f} kg")
    else:
        shift_kg = 0.0
    delta_hex = m_hex_opt  # HEx mass added
    # delta_engine: (mdot - mdot_baseline) * 23
    delta_engine = (mdot_opt - mdot_baseline) * kg_dry_engine_per_kg_per_s_of_air

    # Three lines
    line1 = delta_fuel  # fuel savings only
    line2 = delta_fuel + delta_hex  # fuel + HEx
    line3 = delta_fuel + delta_hex + delta_engine  # fuel + HEx + engine
    line3_dqom = delta_fuel_dqom + delta_hex + delta_engine
    id_min_black = int(np.argmin(line3))
    id_min_red = int(np.argmin(line3_dqom))

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
                "A/A_ref": a_over_a_ref_at_1,
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
            "A/A_ref": a_over_a_ref_opt[i],
            "m_hex": m_hex_opt[i],
            "ao/ao_ref": ao_opt[i],
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
        ("ref_opt", int(id_close) if np.isfinite(dq_ref) and id_close < len(a_over_a_ref_opt) else None),
        ("glob", id_min_black),
        ("square", id_min_red),
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

    ax.plot(m_hex_opt, line1, "r--", linewidth=1.5, label=r"delta fuel (cycle $\eta$)")
    ax.plot(m_hex_opt, delta_fuel_dqom, "k--", linewidth=1.5, label="_nolegend_")
    # ax.plot(m_hex_opt, line2, "k-.", linewidth=1.5, label="delta fuel + HEx")
    ax.plot(m_hex_opt, line3, "r-", linewidth=1.5, label="delta fuel + HEx + engine")
    ax.plot(m_hex_opt, line3_dqom, "k-", linewidth=1.5, label="_nolegend_")

    ax.scatter(
        m_hex_opt[id_min_black],
        line3[id_min_black],
        color="red",
        s=80,
        zorder=5,
        marker="s",
        facecolor="red",
        edgecolor="white",
        linewidths=1,
    )
    ax.scatter(
        m_hex_opt[id_min_red],
        line3[id_min_red],
        color="red",
        s=200,
        zorder=5,
        marker="*",
        facecolor="red",
        edgecolor="white",
        linewidths=1,
    )
    ax.scatter(
        m_hex_opt[id_min_red],
        line3_dqom[id_min_red],
        color="black",
        s=200,
        zorder=5,
        marker="*",
        facecolor="black",
        edgecolor="white",
        linewidths=1,
    )
    if np.isfinite(dq_ref):
        line3_ref = delta_fuel_ref + delta_hex_ref + delta_engine_ref
        ax.scatter(
            m_hex_at_ref,
            line3_ref,
            color="red",
            s=80,
            zorder=5,
            marker="+",
            linewidths=1.5,
        )

    ax.set_xlabel(r"Heat Exchanger (HEx) Core Mass $m_{\mathrm{HEx}}$ (kg)")
    ax.set_ylabel(r"Change in Mass $\Delta m$ (kg) vs baseline unrecuperated")
    ax.legend(
        loc="upper center",
        fontsize=6,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.set_xlim(0, 60)
    ax.set_ylim(-150, 0)
    ax.set_yticks(np.arange(-150, 1, 25))

    # Annotations with arrows (xytext further from markers so arrowheads are visible)
    ax.annotate(
        "global optimal design\nwith cycle model",
        xy=(m_hex_opt[id_min_black] + 1, line3[id_min_black] - 1),
        xytext=(m_hex_opt[id_min_black] + 8, -68),
        fontsize=6,
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )
    if np.isfinite(dq_ref):
        ax.annotate(
            "reference design",
            xy=(m_hex_at_ref, line3_ref),
            xytext=(m_hex_at_ref + 3, line3_ref + 15),
            fontsize=6,
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
