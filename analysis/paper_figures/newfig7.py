"""
Plot of dQ_o^M/Qmax vs A/A_ref along the optimum NTU line from newfig6.
Uses same sweep as newfig6.py.
X-axis: A/A_ref
Y-axis: dQ_o^M/Qmax (practical unavailable creation normalized by Q_max)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from newfig6 import (
    DEFAULT_A_R,
    DEFAULT_C_COLD_OVER_C_HOT,
    DEFAULT_D_R,
    DEFAULT_DP_MAX,
    DEFAULT_F_C_OVER_F_H,
    DEFAULT_G2_H,
    DEFAULT_GAMMA,
    DEFAULT_MOLAR_MASS_RATIO,
    DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    DEFAULT_P_DEAD_OVER_P_HOT_IN,
    DEFAULT_PRESSURE_DROP_ASSUMPTION,
    DEFAULT_ST_OVER_F,
    DEFAULT_T,
    NTU_MATCH,
    NTU_MAX,
    _a_over_a_ref,
    _practical_at_ao_ntu,
)
from xflow import (
    calculate_capacity_ratios,
    calculate_pressure_drop_ratio,
    practical_unavailable_creation_hex,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu
import xflow

AO_SWEEP = np.linspace(0.4, 2.5, 400)
A_OVER_A_REF_VALUES = np.linspace(0.3, 5.0, 800)
save_dir = os.path.dirname(os.path.abspath(__file__))


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

            # Skip if NTU is out of physical range
            if ntu < 0.1 or ntu > NTU_MAX:
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

    Note: ao_over_ao_ref is accepted for API compatibility with _practical_at_ao_ntu but is NOT used
    in the calculation — all frontal-area information is encoded in the constant g2_h_ref.
    """
    # Use constant reference g2_h (ao_over_ao_ref is unused; g2_h encodes the frontal area)
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

    # Sweep NTU from NTU_MATCH up to NTU_MAX
    ntu_sweep = np.linspace(NTU_MATCH / 10, NTU_MAX, 200)
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
    base_name="newfig7",
):
    """Calculate and plot dQ_o^M/Qmax vs A/A_ref using optimum NTU line from newfig6."""
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
    mission_hours = 2
    lhv_kwh_per_kg = 43.2 / 3.6
    eta_turb = 0.8
    eta_ov = 0.2
    mdot_hot = 1.91  # kg/s
    cp_hot = 1.17  # kJ/(kg*K)
    Th_in = 980.0  # K
    Tc_in = 576.0  # K
    Q_max = mdot_hot * cp_hot * (Th_in - Tc_in)
    factor_fuel = mission_hours / lhv_kwh_per_kg * eta_turb / eta_ov * Q_max

    m_hex_ref = 13.3  # kg of tubes

    m_hex = a_over_a_ref_opt_line * m_hex_ref

    dm_fuel_and_hex = dq_o_m_over_qmax_opt_line * factor_fuel + m_hex_ref * a_over_a_ref_opt_line
    id_min = np.argmin(dm_fuel_and_hex)

    ax.plot(
        m_hex,
        dm_fuel_and_hex,
        "k-",
        linewidth=1.5,
        label="fuel + HEx",
    )

    ax.scatter(
        m_hex[id_min],
        dm_fuel_and_hex[id_min],
        color="white",
        s=50,
        zorder=5,
        marker="o",
        facecolor="black",
    )

    id_ref = np.argmin(np.abs(a_over_a_ref_opt_line - 1))

    # Compute the TRUE reference value at (ao=1, NTU=NTU_MATCH) — this lies on the red line at A/A_ref=1.
    # Note: on the black (optimal) line, A/A_ref=1 corresponds to ao~0.83 (not ao=1) because
    # NTU_MATCH is not the optimal NTU for the linear dp formula. The optimal NTU at ao=1 is ~2.3,
    # placing the ao=1 point at A/A_ref~1.55.
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
    dm_ref = dq_ref * factor_fuel + m_hex_ref  # fuel + HEx at reference
    ax.scatter(m_hex_ref, dm_ref, color="white", s=50, zorder=5, marker="D", facecolor="black")

    ax.plot(
        m_hex,
        dq_o_m_over_qmax_opt_line * factor_fuel,
        "k--",
        linewidth=1.5,
        label="fuel",
    )

    # Add red lines for constant reference g2^h case (using linear relations)
    # Calculate dQ_o^M/Qmax for constant g2_h = DEFAULT_G2_H
    a_over_a_ref_constant_g2h, dq_o_m_over_qmax_constant_g2h = _optimal_ntu_line_constant_g2h(
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
        DEFAULT_G2_H,
    )

    if len(a_over_a_ref_constant_g2h) > 0:
        # HEx mass scales linearly with A/A_ref when g2_h is constant
        m_hex_constant_g2h = a_over_a_ref_constant_g2h * m_hex_ref

        # Red dashed: fuel component only (matching pattern of black dashed line)
        ax.plot(
            m_hex_constant_g2h,
            dq_o_m_over_qmax_constant_g2h * factor_fuel,
            "r--",
            linewidth=1.5,
            label="fuel (const. $g^2_h$)",
        )

        # Red solid: fuel + HEx (using constant g2_h calculations, matching pattern)
        dm_fuel_and_hex_constant_g2h = (
            dq_o_m_over_qmax_constant_g2h * factor_fuel + m_hex_ref * a_over_a_ref_constant_g2h
        )
        ax.plot(
            m_hex_constant_g2h,
            dm_fuel_and_hex_constant_g2h,
            "r-",
            linewidth=1.5,
            label="fuel + HEx (const. $g^2_h$)",
        )

    ax.set_xlabel(r"Heat Exchanger (HEx) Core Mass $m_{\mathrm{HEx}}$ (kg)")
    # ax.set_ylabel(r"$\Delta Q_0^M / Q_{\mathrm{max}}$")
    ax.set_ylabel(r"Change in Mass $\Delta m$ (kg)")
    # ax.set_title(r"HEx $\Delta Q_0^M / Q_{\mathrm{max}}$ vs $A/A_{\mathrm{ref}}$")
    # ax.set_title(r"Practical Design Example")
    ax.legend(loc="best", frameon=True, edgecolor="black", facecolor="white", framealpha=1.0, fancybox=False)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(0, 60)
    ax.set_ylim(-200, 0)

    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png", "pdf"]:
        path = os.path.join(save_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    run_plot(
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
        base_name="newfig7",
    )
