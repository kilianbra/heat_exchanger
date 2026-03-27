"""
Paper fig8 (combined): black and red lines from fig9_w_cycle_model + black lines from
fig8_no_cycle_model, on one plot with no markers. Saves as fig8_combined.* in Figs_current.
"""

import os
import sys
from pathlib import Path

try:
    import fig8_no_cycle_model
    import fig9_w_cycle_model
except ImportError:
    _old_scripts = Path(__file__).resolve().parent / "Old_figs" / "old_scripts"
    if _old_scripts.is_dir():
        sys.path.insert(0, str(_old_scripts))
    import fig8_no_cycle_model
    import fig9_w_cycle_model

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figs_current")

# fig8/fig9 shared
m_hex_ref = 13.3
NTU_MATCH = 1.479


def _print_design_comparison(data8, data9):
    """Print four-column comparison: reference, fixed mass opt, practical optimum, cycle optimum."""
    # --- Reference design: ao=1, NTU=NTU_MATCH ---
    ao_ref = 1.0
    ntu_ref = NTU_MATCH
    a_r_ref = fig8_no_cycle_model._a_over_a_ref(ao_ref, ntu_ref)
    m_hex_ref_design = a_r_ref * m_hex_ref

    # --- Fixed mass optimum (fig8 black circle): A/A_ref=1, optimal ao at that mass ---
    a_r_fixed = data8["a_r_fixed"]
    ao_fixed = data8["ao_fixed"]
    ntu_fixed = data8["ntu_fixed"]
    m_hex_fixed = data8["m_hex_fixed"]

    # --- Practical optimum (fig8 black star) ---
    id_prac = data8["id_min"]
    a_r_prac = data8["a_over_a_ref"][id_prac]
    ao_prac = data8["ao"][id_prac]
    ntu_prac = data8["ntu"][id_prac]
    m_hex_prac = data8["m_hex"][id_prac]

    # --- Cycle optimum (fig9 red square) ---
    id_cyc = data9["id_min_red"]
    a_r_cyc = data9["red"]["a"][id_cyc]
    ao_cyc = data9["red"]["ao"][id_cyc]
    ntu_cyc = data9["red"]["ntu"][id_cyc]
    m_hex_cyc = data9["red"]["m_hex"][id_cyc]

    # Build table: inputs
    rows = [
        ["--- INPUTS ---", "", "", "", ""],
        ["A/A_ref", f"{a_r_ref:.4f}", f"{a_r_fixed:.4f}", f"{a_r_prac:.4f}", f"{a_r_cyc:.4f}"],
        ["Ao/Ao_ref", f"{ao_ref:.4f}", f"{ao_fixed:.4f}", f"{ao_prac:.4f}", f"{ao_cyc:.4f}"],
        ["m_hex (kg)", f"{m_hex_ref_design:.3f}", f"{m_hex_fixed:.3f}", f"{m_hex_prac:.3f}", f"{m_hex_cyc:.3f}"],
    ]

    # Practical model outputs for each design (dQ^M/Qmax, eps, dp_h, dp_c, M_in, delta_fuel_approx)
    factor_fuel_fig9 = data9["factor_fuel"]
    designs = [("ref", ao_ref, ntu_ref), ("fixed", ao_fixed, ntu_fixed), ("prac", ao_prac, ntu_prac), ("cyc", ao_cyc, ntu_cyc)]
    for name, ao, ntu in designs:
        dq_qmax = fig8_no_cycle_model._practical_at_ao_ntu(
            ao,
            ntu,
            fig8_no_cycle_model.DEFAULT_C_COLD_OVER_C_HOT,
            fig8_no_cycle_model.DEFAULT_ST_OVER_F,
            fig8_no_cycle_model.DEFAULT_F_C_OVER_F_H,
            fig8_no_cycle_model.DEFAULT_D_R,
            data8["pressure_drop_ratio"],
            fig8_no_cycle_model.DEFAULT_T,
            fig8_no_cycle_model.DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
            fig8_no_cycle_model.DEFAULT_P_DEAD_OVER_P_HOT_IN,
            fig8_no_cycle_model.DEFAULT_GAMMA,
            fig8_no_cycle_model.DEFAULT_DP_MAX,
        )
        eps, dp_h, dp_c = fig8_no_cycle_model._get_eps_dp_at_ao_ntu(
            ao,
            ntu,
            fig8_no_cycle_model.DEFAULT_C_COLD_OVER_C_HOT,
            fig8_no_cycle_model.DEFAULT_ST_OVER_F,
            fig8_no_cycle_model.DEFAULT_F_C_OVER_F_H,
            fig8_no_cycle_model.DEFAULT_D_R,
            data8["pressure_drop_ratio"],
        )
        M_in_prac = fig8_no_cycle_model.DEFAULT_MACH_IN / ao
        delta_fuel_prac = dq_qmax * factor_fuel_fig9 if np.isfinite(dq_qmax) else np.nan
        if name == "ref":
            dq_ref, eps_ref, dp_h_ref, dp_c_ref = dq_qmax, eps, dp_h, dp_c
            M_in_ref, delta_fuel_ref = M_in_prac, delta_fuel_prac
        elif name == "fixed":
            dq_fixed, eps_fixed, dp_h_fixed, dp_c_fixed = dq_qmax, eps, dp_h, dp_c
            M_in_fixed, delta_fuel_fixed = M_in_prac, delta_fuel_prac
        elif name == "prac":
            dq_prac, eps_prac, dp_h_prac, dp_c_prac = dq_qmax, eps, dp_h, dp_c
            M_in_prac_val, delta_fuel_prac_val = M_in_prac, delta_fuel_prac
        else:
            dq_cyc, eps_cyc, dp_h_cyc, dp_c_cyc = dq_qmax, eps, dp_h, dp_c
            M_in_cyc_prac, delta_fuel_cyc_prac = M_in_prac, delta_fuel_prac

    rows += [
        ["--- PRACTICAL MODEL (dQ^M) ---", "", "", "", ""],
        [r"Sum_i dW_Ai/Qmax", f"{dq_ref:.4f}", f"{dq_fixed:.4f}", f"{dq_prac:.4f}", f"{dq_cyc:.4f}"],
        ["eps", f"{eps_ref:.4f}", f"{eps_fixed:.4f}", f"{eps_prac:.4f}", f"{eps_cyc:.4f}"],
        ["dp_h (%)", f"{dp_h_ref:.2f}", f"{dp_h_fixed:.2f}", f"{dp_h_prac:.2f}", f"{dp_h_cyc:.2f}"],
        ["dp_c (%)", f"{dp_c_ref:.2f}", f"{dp_c_fixed:.2f}", f"{dp_c_prac:.2f}", f"{dp_c_cyc:.2f}"],
        ["M_inh", f"{M_in_ref:.4f}", f"{M_in_fixed:.4f}", f"{M_in_prac_val:.4f}", f"{M_in_cyc_prac:.4f}"],
        [r"delta_m_f approx (kg)", f"{delta_fuel_ref:.2f}", f"{delta_fuel_fixed:.2f}", f"{delta_fuel_prac_val:.2f}", f"{delta_fuel_cyc_prac:.2f}"],
    ]

    # Cycle model outputs for each design
    mdot_ref, mdot_fixed, mdot_prac, mdot_cyc = np.nan, np.nan, np.nan, np.nan
    T_hot_in_ref, T_hot_in_fixed, T_hot_in_prac, T_hot_in_cyc = np.nan, np.nan, np.nan, np.nan
    Qmax_ref, Qmax_fixed, Qmax_prac, Qmax_cyc = np.nan, np.nan, np.nan, np.nan
    eps_cyc_ref = eps_cyc_fixed = eps_cyc_prac = eps_cyc_cyc = np.nan
    dp_h_cyc_ref = dp_h_cyc_fixed = dp_h_cyc_prac = dp_h_cyc_cyc = np.nan
    dp_c_cyc_ref = dp_c_cyc_fixed = dp_c_cyc_prac = dp_c_cyc_cyc = np.nan
    M_in_cyc_ref = M_in_cyc_fixed = M_in_cyc_prac = M_in_cyc_cyc = np.nan
    delta_fuel_cyc_ref = delta_fuel_cyc_fixed = delta_fuel_cyc_prac = delta_fuel_cyc_cyc = np.nan
    delta_engine_ref = delta_engine_fixed = delta_engine_prac = delta_engine_cyc = np.nan
    total_ref = total_fixed = total_prac = total_cyc = np.nan

    mdot_baseline = data9["mdot_baseline"]
    eff_b = data9["eff_b"]
    P_shaft = data9["P_shaft_ref"]
    mdot_at_ref = data9["mdot_at_ref"]
    T_hot_in_ref_val = data9["T_hot_in_ref"]
    pressure_drop_ratio_fig9 = data9["pressure_drop_ratio"]
    mission_seconds = fig9_w_cycle_model.mission_seconds
    LHV_J_per_kg = fig9_w_cycle_model.LHV_J_per_kg
    kg_dry = fig9_w_cycle_model.kg_dry_engine_per_kg_per_s_of_air

    for label, ao, ntu in [("ref", ao_ref, ntu_ref), ("fixed", ao_fixed, ntu_fixed), ("prac", ao_prac, ntu_prac), ("cyc", ao_cyc, ntu_cyc)]:
        out = fig9_w_cycle_model.solve_mdot_at_constant_power(
            ao, ntu, pressure_drop_ratio_fig9, mdot_at_ref, T_hot_in_ref_val, P_shaft
        )
        mdot, w_net, T_hot_in, eps_cyc, dp_h_cyc, dp_c_cyc, dq, eff, M_in_cyc = out
        if np.isfinite(mdot) and np.isfinite(eff) and eff > 0:
            mdot_fuel = P_shaft / (LHV_J_per_kg * eff / 100)
            mdot_fuel_baseline = P_shaft / (LHV_J_per_kg * eff_b / 100)
            delta_fuel_cyc = (mdot_fuel - mdot_fuel_baseline) * mission_seconds
            delta_engine = (mdot - mdot_baseline) * kg_dry
            a_r_d = a_r_ref if label == "ref" else (a_r_fixed if label == "fixed" else (a_r_prac if label == "prac" else a_r_cyc))
            m_hex_d = a_r_d * m_hex_ref
            total = delta_fuel_cyc + m_hex_d + delta_engine

            cp_hot = 1070.0
            gamma = 1.4
            eta_poly_c = fig9_w_cycle_model.eta_poly_c
            T_cold_in = 288.0 * (fig9_w_cycle_model.PR ** ((gamma - 1) / (gamma * eta_poly_c)))
            Qmax = mdot * cp_hot * (T_hot_in - T_cold_in)

            if label == "ref":
                mdot_ref, T_hot_in_ref, Qmax_ref = mdot, T_hot_in, Qmax
                eps_cyc_ref, dp_h_cyc_ref, dp_c_cyc_ref, M_in_cyc_ref = eps_cyc, dp_h_cyc, dp_c_cyc, M_in_cyc
                delta_fuel_cyc_ref, delta_engine_ref, total_ref = delta_fuel_cyc, delta_engine, total
            elif label == "fixed":
                mdot_fixed, T_hot_in_fixed, Qmax_fixed = mdot, T_hot_in, Qmax
                eps_cyc_fixed, dp_h_cyc_fixed, dp_c_cyc_fixed, M_in_cyc_fixed = eps_cyc, dp_h_cyc, dp_c_cyc, M_in_cyc
                delta_fuel_cyc_fixed, delta_engine_fixed, total_fixed = delta_fuel_cyc, delta_engine, total
            elif label == "prac":
                mdot_prac, T_hot_in_prac, Qmax_prac = mdot, T_hot_in, Qmax
                eps_cyc_prac, dp_h_cyc_prac, dp_c_cyc_prac, M_in_cyc_prac = eps_cyc, dp_h_cyc, dp_c_cyc, M_in_cyc
                delta_fuel_cyc_prac, delta_engine_prac, total_prac = delta_fuel_cyc, delta_engine, total
            else:
                mdot_cyc, T_hot_in_cyc, Qmax_cyc = mdot, T_hot_in, Qmax
                eps_cyc_cyc, dp_h_cyc_cyc, dp_c_cyc_cyc, M_in_cyc_cyc = eps_cyc, dp_h_cyc, dp_c_cyc, M_in_cyc
                delta_fuel_cyc_cyc, delta_engine_cyc, total_cyc = delta_fuel_cyc, delta_engine, total

    def _fmt(x, fmt_str=".2f"):
        return f"{x:{fmt_str}}" if np.isfinite(x) else "—"

    # dp_h, dp_c from cycle model are fractions; convert to % for display
    rows += [
        ["--- CYCLE MODEL ---", "", "", "", ""],
        ["mdot (kg/s)", _fmt(mdot_ref, ".4f"), _fmt(mdot_fixed, ".4f"), _fmt(mdot_prac, ".4f"), _fmt(mdot_cyc, ".4f")],
        ["T_hot_in (K)", _fmt(T_hot_in_ref, ".1f"), _fmt(T_hot_in_fixed, ".1f"), _fmt(T_hot_in_prac, ".1f"), _fmt(T_hot_in_cyc, ".1f")],
        ["Q_max (kW)", _fmt(Qmax_ref / 1e3, ".1f"), _fmt(Qmax_fixed / 1e3, ".1f"), _fmt(Qmax_prac / 1e3, ".1f"), _fmt(Qmax_cyc / 1e3, ".1f")],
        ["eps", _fmt(eps_cyc_ref, ".4f"), _fmt(eps_cyc_fixed, ".4f"), _fmt(eps_cyc_prac, ".4f"), _fmt(eps_cyc_cyc, ".4f")],
        ["dp_h (%)", _fmt(dp_h_cyc_ref * 100), _fmt(dp_h_cyc_fixed * 100), _fmt(dp_h_cyc_prac * 100), _fmt(dp_h_cyc_cyc * 100)],
        ["dp_c (%)", _fmt(dp_c_cyc_ref * 100), _fmt(dp_c_cyc_fixed * 100), _fmt(dp_c_cyc_prac * 100), _fmt(dp_c_cyc_cyc * 100)],
        ["M_inh", _fmt(M_in_cyc_ref, ".4f"), _fmt(M_in_cyc_fixed, ".4f"), _fmt(M_in_cyc_prac, ".4f"), _fmt(M_in_cyc_cyc, ".4f")],
        [r"delta_m_f from cycle (kg)", _fmt(delta_fuel_cyc_ref), _fmt(delta_fuel_cyc_fixed), _fmt(delta_fuel_cyc_prac), _fmt(delta_fuel_cyc_cyc)],
        ["delta_m_engine (kg)", _fmt(delta_engine_ref), _fmt(delta_engine_fixed), _fmt(delta_engine_prac), _fmt(delta_engine_cyc)],
        ["TOTAL (solid line) (kg)", _fmt(total_ref), _fmt(total_fixed), _fmt(total_prac), _fmt(total_cyc)],
    ]

    print("\n" + "=" * 100)
    print("  FIG8 (combined) DESIGN COMPARISON: Reference | Fixed mass opt | Practical optimum | Cycle optimum")
    print("=" * 100)
    print(tabulate(rows, headers=["", "Reference", "Fixed mass opt", "Practical opt", "Cycle opt"], tablefmt="simple", stralign="right"))
    print("=" * 100 + "\n")


def run_plot(base_name="fig8_combined"):
    """Plot all lines from fig8 and fig9 together, no markers."""
    # Get fig8 data (no cycle model: fuel only dashed, fuel+HEx solid)
    data8 = fig8_no_cycle_model.get_line_data()
    if data8 is None:
        print("Fig8: No valid data.")
        return

    # Get fig9 data (cycle model: red dashed/solid)
    data9 = fig9_w_cycle_model.get_line_data()
    if data9 is None:
        print("Fig9: No valid data.")
        return

    # Print design comparison
    _print_design_comparison(data8, data9)

    m_hex_8 = data8["m_hex"]
    line_fuel_only_8 = data8["line_fuel_only"]
    line_fuel_hex_8 = data8["line_fuel_hex"]
    m_hex_opt = data9["m_hex_opt"]
    line1 = data9["line1"]
    line3 = data9["line3"]

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

    # Fig8 lines (no cycle model)
    ax.plot(m_hex_8, line_fuel_only_8, "k--", linewidth=1.5, label="fuel only practical")
    ax.plot(m_hex_8, line_fuel_hex_8, "k-", linewidth=1.5, label="fuel + HEx practical")

    # Fig9 red lines (cycle model)
    ax.plot(m_hex_opt, line1, "r--", linewidth=1.5, label=r"fuel only cycle")
    ax.plot(m_hex_opt, line3, "r-", linewidth=1.5, label=r"fuel + HEx cycle")

    # Red square at minimum of solid red line (line3)
    id_min_red = data9["id_min_red"]
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
    # Black star at minimum of fig8 fuel + HEx line
    id_min_fig8 = data8["id_min"]
    ax.scatter(
        m_hex_8[id_min_fig8],
        line_fuel_hex_8[id_min_fig8],
        color="black",
        s=90,
        zorder=5,
        marker="*",
        facecolor="black",
        edgecolor="white",
        linewidths=1,
    )

    ax.set_xlabel(r"Heat Exchanger (HEx) Core Mass $m_{\mathrm{HEx}}$ (kg)")
    ax.set_ylabel(r"Change in Take-off Mass $\Delta m$ (kg)")
    ax.legend(
        loc="upper right",
        ncol=1,
        fontsize=8,
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

    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png", "pdf"]:
        path = os.path.join(save_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    run_plot(base_name="fig8_combined")
