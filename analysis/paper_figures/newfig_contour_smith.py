"""
Contour plot of eps_M (practical unavailable energy change) vs Mach_inlet and NTU.

Uses Mach numbers as input: g² = 0.5 * gamma * Mach².
Sweeps Mach from 0.02 to 0.2, NTU with g²-dependent upper limit (pressure drop constraint).
Caches results in parquet for fast re-runs with same parameters.
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure xflow doesn't use cubic formula
import xflow
from scipy.interpolate import griddata
from xflow import (
    calculate_capacity_ratios,
    calculate_pressure_drop_ratio,
    practical_unavailable_creation_hex,
)

from heat_exchanger.epsilon_ntu import epsilon_ntu

xflow.SHOW_CUBIC = False

save_dir = Path(__file__).resolve().parent
CACHE_DIR = save_dir / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# --- Default parameters (match fig2c) ---
DEFAULT_C_COLD_OVER_C_HOT = 1.0
DEFAULT_ST_OVER_F = 0.4
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_D_R = 1.0
DEFAULT_DP_MAX = 0.3
DEFAULT_T = 2.0
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 10.0
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.1
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_GAMMA = 1.4
NTU_GLOBAL_MAX = 15.0  # Never care about NTU > 15

# Pressure drop assumptions (same as fig2c)
PRESSURE_DROP_OPTIONS = ["dp_c=dp_h", "dp_c<<dp_h", "inlet_density"]
DEFAULT_PRESSURE_DROP_ASSUMPTION = "dp_c<<dp_h"
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 0.1

# Mach sweep bounds
MACH_MIN = 0.02
MACH_MAX = 0.2


def _sanitize_pressure_drop_name(assumption: str) -> str:
    """Sanitize pressure drop assumption for use in filenames (Windows disallows <, =, etc.)."""
    return assumption.replace("=", "_eq_").replace("<<", "_ll_")


def mach_to_g2(mach: float, gamma: float = DEFAULT_GAMMA) -> float:
    """Convert Mach number to g²: g² = 0.5 * gamma * Mach²."""
    return 0.5 * gamma * mach**2


def ntu_max_valid(
    g2_h: float,
    c_cold_over_c_hot: float,
    st_over_f: float,
    f_c_over_f_h: float,
    d_r: float,
    dp_max: float,
    pressure_drop_ratio: float,
) -> float:
    """
    Maximum NTU before pressure drop limit is hit.
    dp_hot = dp_coeff * NTU, dp_cold = pressure_drop_ratio * dp_hot.
    Both must be < dp_max.
    """
    C_min_over_C_hot, C_min_over_C_cold, _ = calculate_capacity_ratios(c_cold_over_c_hot)
    st_over_f_h = st_over_f
    st_over_f_c = st_over_f
    dp_coeff = g2_h * (
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
    )
    if dp_coeff <= 0:
        return NTU_GLOBAL_MAX
    # Effective dp_max for hot: cold limits if ratio > 1
    dp_max_eff = dp_max if pressure_drop_ratio <= 1.0 else dp_max / pressure_drop_ratio
    return min(dp_max_eff / dp_coeff, NTU_GLOBAL_MAX)


def compute_eps_m_grid(
    n_mach: int = 50,
    n_ntu: int = 50,
    pressure_drop_assumption: str = DEFAULT_PRESSURE_DROP_ASSUMPTION,
    c_cold_over_c_hot: float = DEFAULT_C_COLD_OVER_C_HOT,
    st_over_f: float = DEFAULT_ST_OVER_F,
    f_c_over_f_h: float = DEFAULT_F_C_OVER_F_H,
    d_r: float = DEFAULT_D_R,
    dp_max: float = DEFAULT_DP_MAX,
    t: float = DEFAULT_T,
    p_cold_in_over_p_hot_in: float = DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in: float = DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma: float = DEFAULT_GAMMA,
    molar_mass_ratio: float | None = DEFAULT_MOLAR_MASS_RATIO,
    a_r: float | None = DEFAULT_A_R,
    use_log_ntu: bool = True,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Compute eps_M on a (Mach, NTU) grid with g²-dependent NTU limits.

    Uses log-spaced NTU per row for better resolution at low NTU (high g²).
    Caches to parquet when pressure_drop_assumption and inputs match.
    """
    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption,
        c_cold_over_c_hot,
        t,
        d_r,
        molar_mass_ratio or 1.0,
        sigma_r,
        p_cold_in_over_p_hot_in,
    )

    # Cache key from all inputs
    cache_config = {
        "pressure_drop_assumption": pressure_drop_assumption,
        "c_cold_over_c_hot": c_cold_over_c_hot,
        "st_over_f": st_over_f,
        "f_c_over_f_h": f_c_over_f_h,
        "d_r": d_r,
        "dp_max": dp_max,
        "t": t,
        "p_cold_in_over_p_hot_in": p_cold_in_over_p_hot_in,
        "p_dead_over_p_hot_in": p_dead_over_p_hot_in,
        "gamma": gamma,
        "molar_mass_ratio": molar_mass_ratio,
        "a_r": a_r,
        "mach_min": MACH_MIN,
        "mach_max": MACH_MAX,
        "n_mach": n_mach,
        "n_ntu": n_ntu,
        "use_log_ntu": use_log_ntu,
    }
    config_str = json.dumps(cache_config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    # Sanitize assumption for filename (avoid = and < on some filesystems)
    assumption_safe = pressure_drop_assumption.replace("=", "_eq_").replace("<", "_lt_")
    cache_path = CACHE_DIR / f"contour_smith_{assumption_safe}_{config_hash}.parquet"

    if not force_recompute and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"Loaded cached data from {cache_path.name}")
        return df

    # Capacity ratios and C_ratio (epsilon depends only on NTU and C_ratio)
    C_min_over_C_hot, C_min_over_C_cold, _ = calculate_capacity_ratios(c_cold_over_c_hot)
    if c_cold_over_c_hot <= 1.0:
        C_ratio = c_cold_over_c_hot
    else:
        C_ratio = 1.0 / c_cold_over_c_hot

    mach_values = np.linspace(MACH_MIN, MACH_MAX, n_mach)
    rows = []

    for mach in mach_values:
        g2_h = mach_to_g2(mach, gamma)
        ntu_max = ntu_max_valid(g2_h, c_cold_over_c_hot, st_over_f, f_c_over_f_h, d_r, dp_max, pressure_drop_ratio)
        ntu_max = max(ntu_max, 0.15)  # ensure at least a small range

        if use_log_ntu:
            ntu_array = np.logspace(np.log10(0.1), np.log10(ntu_max), n_ntu)
        else:
            ntu_array = np.linspace(0.1, ntu_max, n_ntu)

        epsilon = epsilon_ntu(ntu_array, C_ratio, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)
        st_over_f_h = st_over_f
        st_over_f_c = st_over_f
        dp_coeff = g2_h * (
            1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
        )
        dp_hot = dp_coeff * ntu_array
        dp_cold = pressure_drop_ratio * dp_hot
        validity_mask = (dp_hot < dp_max) & (dp_cold < dp_max)

        eps_M_raw = practical_unavailable_creation_hex(
            epsilon,
            t,
            dp_hot,
            dp_cold,
            validity_mask,
            p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in=p_dead_over_p_hot_in,
            gamma=gamma,
        )
        # practical_unavailable_creation_hex returns only valid points; map back to full length
        eps_M_full = np.full(len(ntu_array), np.nan)
        eps_M_full[validity_mask] = eps_M_raw

        for i, ntu in enumerate(ntu_array):
            valid = bool(validity_mask[i])
            eps_m_val = float(eps_M_full[i]) if valid else np.nan
            rows.append(
                {
                    "mach": mach,
                    "g2": g2_h,
                    "ntu": ntu,
                    "eps_M": eps_m_val,
                    "valid": valid,
                    "dp_hot": dp_hot[i],
                    "dp_cold": dp_cold[i],
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(cache_path, index=False)
    print(f"Cached data to {cache_path.name}")
    return df


def plot_contour(
    df: pd.DataFrame,
    n_grid: int = 100,
    pressure_drop_assumption: str = DEFAULT_PRESSURE_DROP_ASSUMPTION,
    base_name: str = "newfig_contour_smith",
    c_cold_over_c_hot: float = DEFAULT_C_COLD_OVER_C_HOT,
    st_over_f: float = DEFAULT_ST_OVER_F,
    f_c_over_f_h: float = DEFAULT_F_C_OVER_F_H,
    d_r: float = DEFAULT_D_R,
    dp_max: float = DEFAULT_DP_MAX,
    t: float = DEFAULT_T,
    p_cold_in_over_p_hot_in: float = DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in: float = DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma: float = DEFAULT_GAMMA,
    molar_mass_ratio: float | None = DEFAULT_MOLAR_MASS_RATIO,
    a_r: float | None = DEFAULT_A_R,
) -> None:
    """
    Plot contour of eps_M vs Mach (y) and NTU (x).
    Interpolates irregular (mach, ntu) points onto a regular grid for contouring.
    Masks regions where dp > dp_max (white).
    """
    # Use only valid points for interpolation
    valid_df = df[df["valid"]].copy()
    if valid_df.empty:
        print("No valid points to plot.")
        return

    mach_pts = valid_df["mach"].values
    ntu_pts = valid_df["ntu"].values
    eps_M_pts = valid_df["eps_M"].values

    # Regular grid for contour (x=NTU, y=Mach)
    mach_min, mach_max = mach_pts.min(), mach_pts.max()
    ntu_min, ntu_max = ntu_pts.min(), ntu_pts.max()
    mach_grid = np.linspace(mach_min, mach_max, n_grid)
    ntu_grid = np.linspace(ntu_min, ntu_max, n_grid)
    NTU_GRID, MACH_GRID = np.meshgrid(ntu_grid, mach_grid)

    # Interpolate (linear; use nan for points outside convex hull)
    eps_M_grid = griddata((mach_pts, ntu_pts), eps_M_pts, (MACH_GRID, NTU_GRID), method="linear", fill_value=np.nan)

    # Mask grid points where dp > dp_max (pressure drop limit exceeded)
    sigma_r = d_r * a_r if a_r is not None else None
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption,
        c_cold_over_c_hot,
        t,
        d_r,
        molar_mass_ratio or 1.0,
        sigma_r,
        p_cold_in_over_p_hot_in,
    )
    C_min_over_C_hot, C_min_over_C_cold, _ = calculate_capacity_ratios(c_cold_over_c_hot)
    st_over_f_h = st_over_f
    st_over_f_c = st_over_f
    g2_grid = mach_to_g2(MACH_GRID, gamma)
    dp_coeff_grid = g2_grid * (
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
    )
    dp_hot_grid = dp_coeff_grid * NTU_GRID
    dp_cold_grid = pressure_drop_ratio * dp_hot_grid
    over_limit = (dp_hot_grid >= dp_max) | (dp_cold_grid >= dp_max)
    eps_M_grid[over_limit] = np.nan

    font_size = 8
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": font_size,
            "mathtext.fontset": "stix",
        }
    )

    fig, ax = plt.subplots(figsize=(9 / 2.54, 9 / 2.54))
    valid_vals = eps_M_grid[~np.isnan(eps_M_grid)]
    if len(valid_vals) == 0:
        print("No valid interpolated values for contour.")
        plt.close(fig)
        return
    levels = [-0.4, -0.35, -0.3, -0.25, -0.2]  # must be increasing for contour
    cs = ax.contour(NTU_GRID, MACH_GRID, eps_M_grid, levels=levels, colors="k", linewidths=0.5)
    manual_locations = [(10, 0.04), (8, 0.08), (5, 0.11), (1.5, 0.135), (1.0, 0.16)]
    ax.clabel(cs, cs.levels, inline=True, fmt=lambda x: f"{-x * 100:.0f}%", fontsize=font_size, manual=manual_locations)
    ax.set_xlabel(r"Number of Heat Transfer Units ($N_\mathrm{tu}$ [-])")
    ax.set_ylabel(r"Hot Inlet Mach number ($M_\mathrm{in}$ [-])")
    ax.set_title(f"Pressure drop: {pressure_drop_assumption}")
    ax.set_xlim(0, ntu_max)
    ax.set_ylim(mach_min, mach_max)
    ax.set_yticks(np.arange(0.02, 0.21, 0.03))
    ax.set_xticks(np.arange(0, ntu_max + 1, 5))
    plt.tight_layout()

    safe_name = _sanitize_pressure_drop_name(pressure_drop_assumption)
    for fmt in ["svg", "png", "pdf"]:
        out_path = save_dir / f"{base_name}_{safe_name}.{fmt}"
        fig.savefig(out_path, dpi=300, facecolor="white", format=fmt, bbox_inches="tight")
        print(f"Saved {out_path.name}")
    plt.close(fig)


def main(
    n_mach: int = 5,
    n_ntu: int = 5,
    pressure_drop_assumption: str = DEFAULT_PRESSURE_DROP_ASSUMPTION,
    plot: bool = True,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Run contour computation and optionally plot.

    For quick testing: n_mach=5, n_ntu=5.
    For publication: n_mach=50, n_ntu=80 or similar.
    """
    df = compute_eps_m_grid(
        n_mach=n_mach,
        n_ntu=n_ntu,
        pressure_drop_assumption=pressure_drop_assumption,
        force_recompute=force_recompute,
    )
    if plot:
        plot_contour(df, pressure_drop_assumption=pressure_drop_assumption)
    return df


if __name__ == "__main__":
    # Edit these and run: uv run python analysis/paper_figures/newfig_contour_smith.py
    df = main(
        n_mach=2000,
        n_ntu=2000,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        plot=True,
        force_recompute=False,
    )
    print(f"Computed {len(df)} points, {df['valid'].sum()} valid")
