"""
Figure 7b: Practical unavailable energy breakdown along optimal practical curve.
Similar to fig6b but with V_core/V_ref (d/d_ref) as x-axis instead of NTU.
Reads optimal practical data from newfig5_practical_data.parquet.
"""

import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xflow
from matplotlib.ticker import MultipleLocator
from xflow import calculate_capacity_ratios, calculate_pressure_drop_ratio, practical_unavailable_creation_hex

from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = Path(save_dir) / "newfig5_practical_data.parquet"

# Reference design point (d/d_ref = 1)
D_REF = 1.0

# Helicopter defaults (for calculations)
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_C_COLD_OVER_C_HOT = 0.95
DEFAULT_D_R = 0.44
DEFAULT_G2_H = 2e-2
DEFAULT_ST_OVER_F = 0.4
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_T = 1.7
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 7.2
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.03
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_GAMMA = 1.4
NTU_REF = 1.479
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 1.0
DEFAULT_DP_MAX = 0.3
DP_REF = 0.106
NTU_MIN = 0.4
NTU_NUM = 100


def _compute_dp_at_ntu_ref(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    pressure_drop_ratio,
):
    """Compute dp/p_in_hot at NTU=NTU_REF and d=d_ref using xflow."""
    xflow.SHOW_CUBIC = True
    xflow.NTU_MATCH = NTU_REF
    ntu_max = 5.0

    C_min_over_C_hot, C_min_over_C_cold, _ = calculate_capacity_ratios(c_cold_over_c_hot)
    st_over_f_h = st_over_f
    st_over_f_c = st_over_f
    dp_coeff_normal = g2_h * (
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
    )

    ntu_arr = np.linspace(0.1, ntu_max, 200)
    closest_idx = np.argmin(np.abs(ntu_arr - NTU_REF))
    ntu_at_anchor = ntu_arr[closest_idx]

    dp_linear_at_anchor = dp_coeff_normal * ntu_at_anchor
    dp_at_ntu_ref = dp_linear_at_anchor * (NTU_REF / ntu_at_anchor) ** 4.407
    return float(dp_at_ntu_ref)


def _get_eps_dp_at_d_ntu(
    d_over_d_ref,
    ntu,
    dp_at_ntu_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    dp_max,
):
    """Get epsilon and pressure drops at single (d, NTU). Returns (eps, dp_hot, dp_cold, validity)."""
    dp_hot = dp_at_ntu_ref * (ntu / NTU_REF) ** 4.407 * (d_over_d_ref ** (-1.407))
    dp_cold = pressure_drop_ratio * dp_hot

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

    validity = (dp_hot < dp_max) & (dp_cold < dp_max)
    return eps, dp_hot, dp_cold, validity


def _practical_thermal_at_d_ntu(
    d_over_d_ref,
    ntu,
    dp_at_ntu_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
):
    """Thermal creation term only (no pressure drop) at single (d, NTU)."""
    eps, dp_hot, dp_cold, validity = _get_eps_dp_at_d_ntu(
        d_over_d_ref, ntu, dp_at_ntu_ref, c_cold_over_c_hot, pressure_drop_ratio, dp_max
    )

    if not validity:
        return np.nan

    # Set pressure drop to zero for thermal-only calculation
    dp_hot = 0.0
    dp_cold = 0.0

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


def _practical_at_d_ntu(
    d_over_d_ref,
    ntu,
    dp_at_ntu_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
):
    """Practical unavailable creation at single (d, NTU)."""
    eps, dp_hot, dp_cold, validity = _get_eps_dp_at_d_ntu(
        d_over_d_ref, ntu, dp_at_ntu_ref, c_cold_over_c_hot, pressure_drop_ratio, dp_max
    )

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


def _ntu_max_for_d(d_over_d_ref, dp_max, dp_ref=DP_REF, ntu_max_at_d_ref=4.0):
    """NTU max for a given d/d_ref and dp_max."""
    return ntu_max_at_d_ref * (dp_max / dp_ref) ** (1 / 4.407) * (d_over_d_ref) ** (-1.407 / 4.407)


def _optimal_ntu_for_d(
    d_over_d_ref,
    dp_at_ntu_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
    ntu_max_global,
):
    """Find optimal NTU for a given d/d_ref that minimizes practical unavailable energy."""
    ntu_fine = np.linspace(NTU_MIN, ntu_max_global, NTU_NUM)
    practical_vals = []

    for ntu in ntu_fine:
        z_practical = _practical_at_d_ntu(
            d_over_d_ref,
            ntu,
            dp_at_ntu_ref,
            c_cold_over_c_hot,
            pressure_drop_ratio,
            t,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            gamma,
            dp_max,
        )
        practical_vals.append(z_practical)

    practical_vals = np.array(practical_vals)
    valid = np.isfinite(practical_vals)

    if np.any(valid):
        idx = np.nanargmin(practical_vals[valid])
        idx_original = np.where(valid)[0][idx]
        ntu_opt = float(ntu_fine[idx_original])
        return ntu_opt
    return None


def save_figures(base_name="fig7b"):
    """
    Save figures as SVG, TIFF, and HD PNG showing practical unavailable energy breakdown
    along optimal practical curve vs V_core/V_ref (d/d_ref).
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file {DATA_FILE} not found. Please run newfig5_practical.py first to generate the data."
        )

    font_size = 8
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": font_size,
            "mathtext.fontset": "stix",
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
        }
    )

    # Load data from parquet
    df = pd.read_parquet(DATA_FILE)

    # Extract optimal line data (along practical optimum)
    opt_mask = df["d_opt"].notna()
    d_opt = df[opt_mask]["d_opt"].values
    
    # Try to get ntu_opt from parquet, otherwise compute optimal NTU
    if "ntu_opt" in df.columns and df[opt_mask]["ntu_opt"].notna().any():
        ntu_opt = df[opt_mask]["ntu_opt"].values
    else:
        # Compute optimal NTU for each d/d_ref
        print("Computing optimal NTU for each d/d_ref...")
        sigma_r = DEFAULT_D_R * DEFAULT_A_R if DEFAULT_A_R is not None else None
        pressure_drop_ratio = calculate_pressure_drop_ratio(
            DEFAULT_PRESSURE_DROP_ASSUMPTION,
            DEFAULT_C_COLD_OVER_C_HOT,
            DEFAULT_T,
            DEFAULT_D_R,
            DEFAULT_MOLAR_MASS_RATIO,
            sigma_r,
            DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        )
        dp_at_ntu_ref = _compute_dp_at_ntu_ref(
            DEFAULT_C_COLD_OVER_C_HOT,
            DEFAULT_ST_OVER_F,
            DEFAULT_F_C_OVER_F_H,
            DEFAULT_D_R,
            DEFAULT_G2_H,
            pressure_drop_ratio,
        )
        
        # Find max NTU needed
        ntu_max_global = max([_ntu_max_for_d(d, DEFAULT_DP_MAX, DP_REF, 4.0) for d in d_opt]) if len(d_opt) > 0 else 4.0
        
        ntu_opt = []
        for d in d_opt:
            ntu = _optimal_ntu_for_d(
                d,
                dp_at_ntu_ref,
                DEFAULT_C_COLD_OVER_C_HOT,
                pressure_drop_ratio,
                DEFAULT_T,
                DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
                DEFAULT_P_DEAD_OVER_P_HOT_IN,
                DEFAULT_GAMMA,
                DEFAULT_DP_MAX,
                ntu_max_global,
            )
            ntu_opt.append(ntu if ntu is not None else np.nan)
        ntu_opt = np.array(ntu_opt)
    
    # Compute thermal and viscous terms along optimal line
    print("Computing thermal and viscous terms along optimal line...")
    sigma_r = DEFAULT_D_R * DEFAULT_A_R if DEFAULT_A_R is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        DEFAULT_PRESSURE_DROP_ASSUMPTION,
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_T,
        DEFAULT_D_R,
        DEFAULT_MOLAR_MASS_RATIO,
        sigma_r,
        DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    )
    dp_at_ntu_ref = _compute_dp_at_ntu_ref(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        pressure_drop_ratio,
    )
    
    z_practical_thermal_only = []
    z_practical_combined = []
    
    for d, ntu in zip(d_opt, ntu_opt, strict=False):
        if np.isnan(ntu):
            z_practical_thermal_only.append(np.nan)
            z_practical_combined.append(np.nan)
        else:
            thermal = _practical_thermal_at_d_ntu(
                d,
                ntu,
                dp_at_ntu_ref,
                DEFAULT_C_COLD_OVER_C_HOT,
                pressure_drop_ratio,
                DEFAULT_T,
                DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
                DEFAULT_P_DEAD_OVER_P_HOT_IN,
                DEFAULT_GAMMA,
                DEFAULT_DP_MAX,
            )
            total = _practical_at_d_ntu(
                d,
                ntu,
                dp_at_ntu_ref,
                DEFAULT_C_COLD_OVER_C_HOT,
                pressure_drop_ratio,
                DEFAULT_T,
                DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
                DEFAULT_P_DEAD_OVER_P_HOT_IN,
                DEFAULT_GAMMA,
                DEFAULT_DP_MAX,
            )
            z_practical_thermal_only.append(thermal)
            z_practical_combined.append(total)
    
    z_practical_thermal_only = np.array(z_practical_thermal_only)
    z_practical_combined = np.array(z_practical_combined)

    if len(d_opt) == 0:
        raise ValueError("No optimal line data found in parquet file.")

    # Sort by d_opt for proper plotting
    sort_idx = np.argsort(d_opt)
    d_opt = d_opt[sort_idx]
    z_practical_thermal_only = z_practical_thermal_only[sort_idx]
    z_practical_combined = z_practical_combined[sort_idx]

    # Filter out NaN values
    valid_mask = np.isfinite(z_practical_thermal_only) & np.isfinite(z_practical_combined)
    d_opt = d_opt[valid_mask]
    z_practical_thermal_only = z_practical_thermal_only[valid_mask]
    z_practical_combined = z_practical_combined[valid_mask]

    if len(d_opt) == 0:
        raise ValueError("No valid data points found.")

    # Match fig6b figure size (triple column 6 cm)
    fig = plt.figure(figsize=(9 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)

    # Plot thermal-only line (no pressure drop)
    _ = ax.plot(  # line_thermal (unused; change back if needed for legend etc.)
        d_opt,
        z_practical_thermal_only,
        "k-",
        label="_nolegend_",
        zorder=2,
    )[0]

    # Plot combined line (with pressure drop)
    _ = ax.plot(  # line_combined (unused; change back if needed for legend etc.)
        d_opt,
        z_practical_combined,
        "k-",
        label="_nolegend_",
        zorder=3,
    )[0]

    # Fill between thermal and 0 (thermal creation) - note: practical can be negative
    ax.fill_between(d_opt, z_practical_thermal_only, 0, alpha=0.3, color="gray", zorder=1)

    # Fill between combined and thermal (viscous dissipation)
    ax.fill_between(
        d_opt,
        z_practical_combined,
        z_practical_thermal_only,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
        hatch="///",
        zorder=1,
        label="_nolegend_",
    )

    # Optimal marker: circle, black, s=50, white edge (match fig6b)
    # Find minimum of combined (optimal practical point)
    # idx_optimum = np.nanargmin(z_practical_combined)
    # x_optimum = d_opt[idx_optimum]
    # y_optimum = z_practical_combined[idx_optimum]
    # ax.scatter(x_optimum, y_optimum, marker="o", facecolor="black", edgecolor="white", zorder=5, s=50)

    # print(f"Optimum d/d_ref: {x_optimum:.4f}")

    # Reference marker: diamond (D), black face, white edge, s=50 (match fig6b)
    idx_ref = np.argmin(np.abs(d_opt - D_REF))
    x_ref = d_opt[idx_ref]
    y_ref = z_practical_combined[idx_ref]
    ax.scatter(x_ref, y_ref, marker="D", facecolor="black", edgecolor="white", zorder=5, s=50)

    # Annotations with arrows (match fig6a style)
    # arrow_kw_opt = dict(arrowstyle="->", color="black", lw=1, shrinkB=10)
    # arrow_kw_ref = dict(arrowstyle="->", color="black", lw=1, shrinkB=10)
    # ax.annotate(
    #     "reference design",
    #     xy=(x_ref, y_ref),
    #     xytext=(x_ref + 0.1, y_ref + 0.01),
    #     fontsize=font_size,
    #     ha="left",
    #     arrowprops=arrow_kw_ref,
    # )
    # ax.annotate(
    #     "optimal design",
    #     xy=(x_optimum, y_optimum),
    #     xytext=(x_optimum - 0.1, y_optimum + 0.01),
    #     fontsize=font_size,
    #     ha="right",
    #     arrowprops=arrow_kw_opt,
    # )

    ax.set_title("")
    ax.set_xlim(d_opt.min() * 0.95, 3.0)
    y_min = min(np.nanmin(z_practical_combined), np.nanmin(z_practical_thermal_only))
    y_max = max(np.nanmax(z_practical_combined), np.nanmax(z_practical_thermal_only))
    ax.set_ylim(y_min * 1.1, max(0, y_max * 1.1))
    ax.set_xlabel(r"Core Volume Ratio ($V_\mathrm{core}/V_\mathrm{ref}$ [-])")
    ax.set_ylabel(r"Change in Unavailable Energy ($\Delta \dot{Q}_0^\mathrm{M}/\dot{Q}_{\mathrm{max}}$ [-])")

    ax.set_ylim(-0.4,0)
    ax.set_xlim(0.5,2.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

    # Legend: thermal creation and viscous dissipation (match fig6b box style)
    patch_thermal = mpatches.Patch(facecolor="gray", alpha=0.3, edgecolor="black", label="Thermal creation")
    patch_viscous = mpatches.Patch(
        facecolor="none", edgecolor="black", hatch="///", linewidth=1.5, label="Viscous dissipation"
    )
    ax.legend(
        handles=[patch_thermal, patch_viscous],
        loc="upper left",
        labelspacing=0.05,
        edgecolor="black",
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
    )

    plt.tight_layout(pad=0.5)

    fig.savefig(
        os.path.join(save_dir, f"{base_name}.svg"),
        dpi=300,
        facecolor="white",
        format="svg",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.tiff"),
        dpi=300,
        facecolor="white",
        format="tiff",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.png"),
        dpi=300,
        facecolor="white",
        format="png",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.pdf"),
        dpi=300,
        facecolor="white",
        format="pdf",
        bbox_inches=None,
        pad_inches=0,
    )

    plt.close(fig)
    print(f"Saved figures: {base_name}.svg, {base_name}.tiff, {base_name}.png, {base_name}.pdf")


if __name__ == "__main__":
    save_figures()
