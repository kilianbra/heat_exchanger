"""
Plot of dQ_o^M/Qmax vs A/A_ref along the optimum NTU line.
Uses explicit defaults matching fig8_full_cycle.py for direct comparison.
X-axis: A/A_ref (m_hex)
Y-axis: Change in mass (dQ_o^M-based fuel + HEx)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xflow
from xflow import (
    calculate_capacity_ratios,
    calculate_pressure_drop_ratio,
    practical_unavailable_creation_hex,
)

from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figs_current")

# --- Defaults for fig8_full_cycle and newfig6.py ---
# Group parameters that are THE SAME for both scripts first:
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_GAMMA = 1.4
DEFAULT_MACH_IN = 0.1
DEFAULT_ST_OVER_F = 0.4
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_DP_MAX = 0.2
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 1.0

# Define a switch for input set: 'fig8' (default, matches fig8_full_cycle.py) or 'newfig6'
NEW_INPUT_SET = True  # Options: "fig8", "newfig6"

if NEW_INPUT_SET:
    # Parameters UNIQUE TO THIS SCRIPT / MATCHING fig8_full_cycle.py
    DEFAULT_C_COLD_OVER_C_HOT = 1.0
    DEFAULT_D_R = 0.257
    DEFAULT_T = 898 / 588  # = 1.527
    DEFAULT_T_DEAD_OVER_T_COLD_IN = 288 / 588
    DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 8.82 / 1.04
    DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.04
    NTU_MATCH = 1.824
    DEFAULT_NTU_MAX = 8.0
    # g2h = 0.5 * gamma * M^2 at reference (M=M_ref for ao=1)
    DEFAULT_G2_H = 0.5 * DEFAULT_GAMMA * DEFAULT_MACH_IN**2
else:
    # Parameters UNIQUE TO newfig6.py
    DEFAULT_C_COLD_OVER_C_HOT = 0.95
    DEFAULT_D_R = 0.44
    DEFAULT_T = 1.7
    DEFAULT_T_DEAD_OVER_T_COLD_IN = 0.52
    DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 7.2
    DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.03
    NTU_MATCH = 1.479
    DEFAULT_NTU_MAX = 15.0
    # Use fixed value for g2h
    DEFAULT_G2_H = 2e-2

DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD

# Sweep parameters (matching fig8_full_cycle.py)
# newfig6: AO_SWEEP was linspace(0.4, 2.5, 400), A_OVER_A_REF linspace(0.01, 5.0, 800)
m_hex_ref = 13.3  # kg
AO_SWEEP = np.linspace(0.15, 2.5, 100)
A_OVER_A_REF_VALUES = np.linspace(0.1 / m_hex_ref, 5.0, 80)


def _a_over_a_ref(ao_over_ao_ref, ntu):
    """A/A_ref = (NTU / NTU_MATCH) * (ao_over_ao_ref)**0.587"""
    return (ntu / NTU_MATCH) * (ao_over_ao_ref**0.587)


def _practical_at_ao_ntu(
    ao_over_ao_ref,
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
    """Practical unavailable creation at single (Ao, NTU). g2_h = DEFAULT_G2_H / (ao/ao_ref)**2"""
    g2_h = DEFAULT_G2_H / (ao_over_ao_ref**2)
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
    if dp_hot >= dp_max or dp_cold >= dp_max:
        return np.nan
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
    return float(out[0]) if len(out) > 0 else np.nan


def _get_eps_dp_at_ao_ntu(ao_over_ao_ref, ntu, c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, pressure_drop_ratio):
    """Return (eps, dp_hot_pct, dp_cold_pct) for a given (ao, NTU). dp values as % of inlet pressure."""
    g2_h = DEFAULT_G2_H / (ao_over_ao_ref**2)
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
    dp_hot = dp_coeff * ntu  # fraction of inlet pressure
    dp_cold = pressure_drop_ratio * dp_hot
    return eps, dp_hot * 100, dp_cold * 100


def _optimal_ao_for_each_a_over_a_ref(
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
    """Vertical-slice optimization: for each target A/A_ref, find the best ao (and NTU).

    For a given A/A_ref, the constraint is:
        NTU = A/A_ref * NTU_MATCH / ao^0.587
    We sweep ao to find the one that minimizes dQ_o^M at the constrained NTU.

    Returns (a_over_a_ref_values, ao_opt_values, ntu_opt_values, dq_o_m_over_qmax_values).
    """
    a_over_a_ref_out = []
    ao_opt_out = []
    ntu_opt_out = []
    dq_out = []

    for a_target in A_OVER_A_REF_VALUES:
        best_z = np.inf
        best_ao = None
        best_ntu = None

        for ao in AO_SWEEP:
            # Constraint: A/A_ref = (NTU / NTU_MATCH) * ao^0.587
            ntu = a_target * NTU_MATCH / (ao**0.587)

            # Skip if NTU is out of physical range (0.02 allows small m_hex down to 0.1 kg)
            if ntu < 0.02 or ntu > DEFAULT_NTU_MAX:
                continue

            z = _practical_at_ao_ntu(
                ao,
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

            if np.isfinite(z) and z < best_z:
                best_z = z
                best_ao = ao
                best_ntu = ntu

        if best_ao is not None:
            a_over_a_ref_out.append(a_target)
            ao_opt_out.append(best_ao)
            ntu_opt_out.append(best_ntu)
            dq_out.append(best_z)

    return np.array(a_over_a_ref_out), np.array(ao_opt_out), np.array(ntu_opt_out), np.array(dq_out)


def _practical_at_ao_ntu_constant_g2h(
    ao_over_ao_ref,
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
    g2_h_ref,
):
    """Practical unavailable creation at single (Ao, NTU) using constant reference g2_h and linear relations.
    This uses g2_h = g2_h_ref (constant) instead of g2_h = DEFAULT_G2_H / (Ao/Ao_ref)**2.
    """
    # Use constant reference g2_h
    g2_h = g2_h_ref

    # Calculate capacity ratios
    C_min_over_C_hot, C_min_over_C_cold, _ = calculate_capacity_ratios(c_cold_over_c_hot)

    # C_ratio for epsilon-NTU calculation
    if c_cold_over_c_hot <= 1.0:
        C_ratio = c_cold_over_c_hot
    else:
        C_ratio = 1.0 / c_cold_over_c_hot

    # Calculate effectiveness
    eps = epsilon_ntu(
        np.array([ntu]),
        C_ratio,
        exchanger_type="aligned_flow",
        flow_type="counterflow",
        n_passes=1,
    )[0]

    # Calculate pressure drop using normal linear formula (not cubic)
    st_over_f_h = st_over_f
    st_over_f_c = st_over_f
    dp_coeff_normal = g2_h * (
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
    )
    dp_hot = dp_coeff_normal * ntu
    dp_cold = pressure_drop_ratio * dp_hot

    validity = (dp_hot < dp_max) & (dp_cold < dp_max)
    if not validity:
        return np.nan

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
    return float(out[0]) if len(out) > 0 else np.nan


def _optimal_ntu_line_constant_g2h(
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
    g2_h_ref,
):
    """Calculate dQ_o^M/Qmax for constant g2_h case by sweeping NTU.
    For constant g2_h, A/A_ref = NTU/NTU_MATCH (with ao = 1.0 at reference).
    Returns (a_over_a_ref_values, dq_o_m_over_qmax_values) for the constant g2_h case.
    """
    # Ensure SHOW_CUBIC is False for linear relations
    xflow.SHOW_CUBIC = False

    a_over_a_ref_vals = []
    dq_o_m_over_qmax_vals = []

    # Sweep NTU from NTU_MATCH up to DEFAULT_NTU_MAX
    ntu_sweep = np.linspace(0.1, DEFAULT_NTU_MAX, 200)
    ao_ref = 1.0  # Use reference ao value when g2_h is constant

    for ntu in ntu_sweep:
        # For constant g2_h, A/A_ref = NTU/NTU_MATCH directly
        a_over_a_ref = ntu / NTU_MATCH

        # Calculate dQ_o^M/Qmax for this NTU with constant g2_h
        z = _practical_at_ao_ntu_constant_g2h(
            ao_ref,
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
            g2_h_ref,
        )

        if np.isfinite(z):
            a_over_a_ref_vals.append(a_over_a_ref)
            dq_o_m_over_qmax_vals.append(z)

    return np.array(a_over_a_ref_vals), np.array(dq_o_m_over_qmax_vals)


def run_plot(
    c_cold_over_c_hot=DEFAULT_C_COLD_OVER_C_HOT,
    st_over_f=DEFAULT_ST_OVER_F,
    f_c_over_f_h=DEFAULT_F_C_OVER_F_H,
    d_r=DEFAULT_D_R,
    t=DEFAULT_T,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig8_no_cycle_model",
):
    """Calculate and plot dQ_o^M vs A/A_ref using optimum NTU line (defaults match fig8_full_cycle)."""
    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption,
        c_cold_over_c_hot,
        t,
        d_r,
        molar_mass_ratio,
        sigma_r,
        p_cold_in_over_p_hot_in,
    )

    # Vertical-slice optimization: for each A/A_ref, find the best ao (and NTU).
    # This ensures the black line shows the best possible dQ_o^M for each metal volume,
    # and must pass through (or below) the reference point at A/A_ref=1.
    a_over_a_ref_opt_line, ao_opt_line, ntu_opt_line, dq_o_m_over_qmax_opt_line = _optimal_ao_for_each_a_over_a_ref(
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

    if len(a_over_a_ref_opt_line) == 0:
        print("No valid optimum points found.")
        return

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
    # Mission and Q_max (matching fig8_full_cycle.py)
    # Previously fig8 used: mdot_hot=1.91, Th_in=980, Tc_in=576
    mission_hours = 2
    lhv_kwh_per_kg = 12.0  # 43.2 / 3.6
    eta_turb = 0.88
    eta_ov = 0.434
    mdot_hot_ref = 2.3  # kg/s (matches fig8_full_cycle mdot_ref)
    cp_hot = 1.07  # kJ/(kg*K) — high-temp air value (compromise between 1004 and 1170)
    Th_in_ref = 898  # K
    Tc_in_ref = 588  # K
    Q_max = mdot_hot_ref * cp_hot * (Th_in_ref - Tc_in_ref)
    factor_fuel = mission_hours / lhv_kwh_per_kg * eta_turb / eta_ov * Q_max

    m_hex = a_over_a_ref_opt_line * m_hex_ref

    dm_fuel_and_hex = dq_o_m_over_qmax_opt_line * factor_fuel + m_hex_ref * a_over_a_ref_opt_line
    id_min = np.argmin(dm_fuel_and_hex)

    ax.plot(
        m_hex,
        dm_fuel_and_hex,
        "k-",
        linewidth=1.5,
        label="fuel saving + HEx mass",
    )

    # Global optimum: star marker (was circle)
    ao_min = ao_opt_line[id_min]
    ntu_min = ntu_opt_line[id_min]
    ax.scatter(
        m_hex[id_min],
        dm_fuel_and_hex[id_min],
        color="black",
        s=150,
        zorder=5,
        marker="*",
        facecolor="black",
        edgecolor="white",
    )

    # Reference design: compute exact (ao=1, NTU=NTU_MATCH) values first (needed for x-position)
    a_over_a_ref_ref = _a_over_a_ref(1.0, NTU_MATCH)
    m_hex_ref_design = a_over_a_ref_ref * m_hex_ref  # exact x-position, same as +

    # Fixed mass optimal: interpolate at exact A/A_ref=1 (m_hex_ref_design) so circle aligns with +
    dm_fixed_interp = np.interp(m_hex_ref_design, m_hex, dm_fuel_and_hex)
    ao_fixed_interp = np.interp(m_hex_ref_design, m_hex, ao_opt_line)
    ntu_fixed_interp = np.interp(m_hex_ref_design, m_hex, ntu_opt_line)
    dq_fixed_interp = np.interp(m_hex_ref_design, m_hex, dq_o_m_over_qmax_opt_line)

    ax.scatter(
        m_hex_ref_design,
        dm_fixed_interp,
        facecolor="black",
        edgecolor="white",
        s=50,
        zorder=5,
        marker="o",
    )

    # Reference design: + at (ao=1, NTU=NTU_MATCH)
    dq_ref = _practical_at_ao_ntu(
        1.0,
        NTU_MATCH,
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
    dm_ref_design = dq_ref * factor_fuel + m_hex_ref * a_over_a_ref_ref
    ax.scatter(
        m_hex_ref_design,
        dm_ref_design,
        color="black",
        s=80,
        zorder=5,
        marker="+",
        linewidths=1.5,
    )

    # One-liner summaries for each design point
    eps_ref, dp_h_ref, dp_c_ref = _get_eps_dp_at_ao_ntu(
        1.0, NTU_MATCH, c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, pressure_drop_ratio
    )
    eps_fixed, dp_h_fixed, dp_c_fixed = _get_eps_dp_at_ao_ntu(
        ao_fixed_interp, ntu_fixed_interp, c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, pressure_drop_ratio
    )
    eps_global, dp_h_global, dp_c_global = _get_eps_dp_at_ao_ntu(
        ao_min, ntu_min, c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, pressure_drop_ratio
    )
    dq_ref_over_qmax = dq_ref  # dQ_o^M / Q_max for reference
    dq_fixed_over_qmax = dq_fixed_interp
    dq_global_over_qmax = dq_o_m_over_qmax_opt_line[id_min]
    print(
        f"Reference (+): NTU={NTU_MATCH:.3f}, m_HEx={m_hex_ref_design:.2f} kg, eps={eps_ref:.4f}, "
        f"Q={eps_ref * Q_max:.1f} kW, dQ_o^M={dq_ref_over_qmax * Q_max:.1f} kW, "
        f"dp/p_in hot={dp_h_ref:.2f}%, cold={dp_c_ref:.2f}%"
    )
    print(
        f"Fixed mass optimal (circle): NTU={ntu_fixed_interp:.3f}, m_HEx={m_hex_ref_design:.2f} kg, eps={eps_fixed:.4f}, "
        f"Q={eps_fixed * Q_max:.1f} kW, dQ_o^M={dq_fixed_over_qmax * Q_max:.1f} kW, "
        f"dp/p_in hot={dp_h_fixed:.2f}%, cold={dp_c_fixed:.2f}%"
    )
    print(
        f"Global mass optimal (star): NTU={ntu_min:.3f}, m_HEx={m_hex[id_min]:.2f} kg, eps={eps_global:.4f}, "
        f"Q={eps_global * Q_max:.1f} kW, dQ_o^M={dq_global_over_qmax * Q_max:.1f} kW, "
        f"dp/p_in hot={dp_h_global:.2f}%, cold={dp_c_global:.2f}%"
    )

    ax.plot(
        m_hex,
        dq_o_m_over_qmax_opt_line * factor_fuel,
        "k--",
        linewidth=1.5,
        label="fuel saving",
    )

    # # Add red lines for constant reference g2^h case (using linear relations)
    # # Calculate dQ_o^M/Qmax for constant g2_h = DEFAULT_G2_H
    # a_over_a_ref_constant_g2h, dq_o_m_over_qmax_constant_g2h = _optimal_ntu_line_constant_g2h(
    #     c_cold_over_c_hot,
    #     st_over_f,
    #     f_c_over_f_h,
    #     d_r,
    #     pressure_drop_ratio,
    #     t,
    #     p_cold_in_over_p_hot_in,
    #     p_dead_over_p_hot_in,
    #     gamma,
    #     dp_max,
    #     DEFAULT_G2_H,
    # )
    #
    # if len(a_over_a_ref_constant_g2h) > 0:
    #     # HEx mass scales linearly with A/A_ref when g2_h is constant
    #     m_hex_constant_g2h = a_over_a_ref_constant_g2h * m_hex_ref
    #
    #     # Red dashed: fuel component only (matching pattern of black dashed line)
    #     ax.plot(
    #         m_hex_constant_g2h,
    #         dq_o_m_over_qmax_constant_g2h * factor_fuel,
    #         "r--",
    #         linewidth=1.5,
    #         label="fuel (const. $g^2_h$)",
    #     )
    #
    #     # Red solid: fuel + HEx (using constant g2_h calculations, matching pattern)
    #     dm_fuel_and_hex_constant_g2h = (
    #         dq_o_m_over_qmax_constant_g2h * factor_fuel + m_hex_ref * a_over_a_ref_constant_g2h
    #     )
    #     ax.plot(
    #         m_hex_constant_g2h,
    #         dm_fuel_and_hex_constant_g2h,
    #         "r-",
    #         linewidth=1.5,
    #         label="fuel + HEx (const. $g^2_h$)",
    #     )

    # Annotations with arrows: reference design (+), fixed mass optimal (circle), global optimal (star)
    font_size = 8
    arrow_kw = dict(arrowstyle="->", color="black", lw=1, shrinkB=12)
    ax.annotate(
        "reference design",
        xy=(m_hex_ref_design, dm_ref_design),
        xytext=(6, -32),
        fontsize=font_size,
        ha="left",
        arrowprops=arrow_kw,
    )
    ax.annotate(
        "fixed mass optimal design",
        xy=(m_hex_ref_design, dm_fixed_interp),
        xytext=(15, -55),
        fontsize=font_size,
        ha="left",
        arrowprops=arrow_kw,
    )
    ax.annotate(
        "global optimal design",
        xy=(m_hex[id_min], dm_fuel_and_hex[id_min]),
        xytext=(35, -70),
        fontsize=font_size,
        ha="left",
        arrowprops=arrow_kw,
    )

    ax.set_xlabel(r"Heat Exchanger (HEx) Core Mass $m_{\mathrm{HEx}}$ (kg)")
    # ax.set_ylabel(r"$\Delta Q_0^M / Q_{\mathrm{max}}$")
    ax.set_ylabel(r"Change in Mass $\Delta m$ (kg)")
    # ax.set_title(r"HEx $\Delta Q_0^M / Q_{\mathrm{max}}$ vs $A/A_{\mathrm{ref}}$")
    # ax.set_title(r"Practical Design Example")
    ax.legend(loc="upper right", frameon=True, edgecolor="black", facecolor="white", framealpha=1.0, fancybox=False)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(0, 60)
    ax.set_ylim(-100, 0)
    ax.set_yticks(np.arange(-100, 1, 20))

    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png", "pdf"]:
        path = os.path.join(save_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    run_plot(base_name="fig8_no_cycle_model")  # Uses explicit defaults matching fig8_full_cycle.py
