"""
Plot of dQ_o^M/Qmax vs A/A_ref along the optimum NTU line from newfig6.
Uses same sweep as newfig6.py.
X-axis: A/A_ref
Y-axis: dQ_o^M/Qmax (practical unavailable creation normalized by Q_max)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from xflow import (
    calculate_capacity_ratios,
    calculate_pressure_drop_ratio,
)

from newfig6 import (
    _optimal_ntu_line_ao,
    DEFAULT_PRESSURE_DROP_ASSUMPTION,
    DEFAULT_C_COLD_OVER_C_HOT,
    DEFAULT_D_R,
    DEFAULT_ST_OVER_F,
    DEFAULT_F_C_OVER_F_H,
    DEFAULT_T,
    DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    DEFAULT_P_DEAD_OVER_P_HOT_IN,
    DEFAULT_GAMMA,
    DEFAULT_MOLAR_MASS_RATIO,
    DEFAULT_A_R,
    DEFAULT_DP_MAX,
)

save_dir = os.path.dirname(os.path.abspath(__file__))


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

    # Get optimum NTU line from newfig6 (includes dQ_o^M/Qmax values)
    ao_opt_line, ntu_opt_line, a_over_a_ref_opt_line, dq_o_m_over_qmax_opt_line = _optimal_ntu_line_ao(
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
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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

    ax.plot(
        a_over_a_ref_opt_line,
        dq_o_m_over_qmax_opt_line * factor_fuel + m_hex_ref * a_over_a_ref_opt_line,
        "k-",
        linewidth=1.5,
        label="optimum",
    )

    ax.set_xlabel(r"$A / A_{\mathrm{ref}}$")
    # ax.set_ylabel(r"$\Delta Q_0^M / Q_{\mathrm{max}}$")
    ax.set_ylabel(r"$\Delta m_{TO}$ (kg)")
    # ax.set_title(r"HEx $\Delta Q_0^M / Q_{\mathrm{max}}$ vs $A/A_{\mathrm{ref}}$")
    ax.set_title(r"HEx $\Delta m_{TO}$ vs $A/A_{\mathrm{ref}}$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png"]:
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
