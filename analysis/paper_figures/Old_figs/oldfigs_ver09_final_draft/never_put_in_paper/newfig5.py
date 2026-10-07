"""
2D contour plot of practical availability vs d_over_d_ref and scaled NTU.
Uses same inputs as newfig4 (Helicopter defaults). Pressure drop includes
multiplier d_over_d_ref**1.407. Y-axis: (NTU/NTU_match)**(-1.704) * (d_over_d_ref)**(-0.704).
"""

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import xflow
from scipy.interpolate import griddata
from xflow import (
    calculate_epsilon_ntu_curve,
    calculate_pressure_drop_ratio,
    practical_unavailable_creation_hex,
)

from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = Path(save_dir) / "newfig5_data.parquet"

# Helicopter defaults (same as newfig4)
DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
DEFAULT_C_COLD_OVER_C_HOT = 0.95
DEFAULT_D_R = 0.44
DEFAULT_G2_H = 2e-2
DEFAULT_ST_OVER_F = 0.4
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_T = 1.7
DEFAULT_T_DEAD_OVER_T_COLD_IN = 0.52
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 7.2
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.03
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_GAMMA = 1.4
NTU_REF = 1.479
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 1.0
DEFAULT_DP_MAX = 0.3
DP_REF = 0.115

# Sweep: d_over_d_ref from 0.5 to 1.2, 8 values step 0.1
D_OVER_D_REF_VALUES = np.linspace(1.0, 3, 1000)  # 0.5, 0.6, ..., 1.2
NTU_MAX_AT_D_REF = 4.0  # NTU max at d/d_ref = 1; for smaller d, NTU max increases (2/d)
NTU_NUM = 1000  # number of NTU points per d (sweep resolution)
CONTOUR_GRID_N = 2000  # grid size for interpolation; contour smoothness is set by this, not NTU_NUM
NTU_MIN = 0.4
AO_MAX = 4


def _ntu_max_for_d(d_over_d_ref, dp_max, dp_ref=DP_REF):
    """NTU max for a given d/d_ref and dp_max.

    Formula: NTU_max = NTU_max_at_d_ref * (DP_max/DP_ref)^(1/4.407) * (d/d_ref)^{-1.407/4.407}
    """
    return NTU_MAX_AT_D_REF * (dp_max / dp_ref) ** (1 / 4.407) * (d_over_d_ref) ** (-1.407 / 4.407)


def _y_axis(d_over_d_ref, ntu):
    """Y-axis coordinate: (NTU/NTU_match)**(-1.704) * (d_over_d_ref)**(-0.704)."""
    return (ntu / NTU_REF) ** (-1.704) * (d_over_d_ref) ** (-0.704)


def _compute_reference_dp_curve(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    pressure_drop_ratio,
    dp_max,
):
    """Reference hot pressure drop vs NTU at d_over_d_ref=1 (same formula as xflow 4.407).
    ntu_max set so the curve covers the full NTU range needed for all d (max at smallest d).
    Use a much higher dp_max for the reference curve so it covers full NTU range;
    actual dp_max check happens after scaling by d**1.407.
    """
    xflow.SHOW_CUBIC = True
    xflow.NTU_REF = NTU_REF
    ntu_max_ref = _ntu_max_for_d(D_OVER_D_REF_VALUES.min(), dp_max)
    # Add safety margin (1.2x) to ensure we cover all NTU values needed
    ntu_max_ref = ntu_max_ref * 1.2
    # Use much higher dp_max for reference curve so it covers full NTU range
    # The actual dp_max limit is applied after scaling by d**1.407
    dp_max_ref = dp_max * 3.0  # Allow reference curve to go much higher
    ntu_arr, _, dp_hot_ref, _, _ = calculate_epsilon_ntu_curve(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=ntu_max_ref,
        dp_max=dp_max_ref,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )
    return ntu_arr, dp_hot_ref


def _practical_at_d_ntu(
    d_over_d_ref,
    ntu,
    ntu_ref,
    dp_hot_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
):
    """Practical unavailable creation at single (d, NTU). Pressure drop = ref_dp(NTU) * d**1.407."""
    # Interpolate reference dp at this NTU, then scale by d**1.407
    # np.interp will extrapolate if needed, but we'll check validity after
    dp_hot = np.interp(ntu, ntu_ref, dp_hot_ref) * (d_over_d_ref**1.407)
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


def _fig4_optimum_ntu(
    ntu_ref,
    dp_hot_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
):
    """At d_over_d_ref=1, find NTU that minimizes practical_unavailable_creation (fig4 optimum)."""
    ntu_fine = np.linspace(0.35, NTU_MAX_AT_D_REF, 150)
    vals = []
    for ntu in ntu_fine:
        z = _practical_at_d_ntu(
            1.0,
            ntu,
            ntu_ref,
            dp_hot_ref,
            c_cold_over_c_hot,
            pressure_drop_ratio,
            t,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            gamma,
            dp_max,
        )
        vals.append(z)
    vals = np.array(vals)
    valid = np.isfinite(vals)
    if not np.any(valid):
        return None
    idx = np.nanargmin(vals)
    return float(ntu_fine[idx])


def _optimal_ntu_line(
    ntu_ref,
    dp_hot_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
):
    """Find optimal NTU for each d/d_ref, return (d_values, ntu_opt_values, y_opt_values, z_opt_values)."""
    d_vals = []
    ntu_opt_vals = []
    y_opt_vals = []
    z_opt_vals = []

    for d in D_OVER_D_REF_VALUES:
        ntu_max_d = _ntu_max_for_d(d, dp_max)
        ntu_fine = np.linspace(NTU_MIN, ntu_max_d, NTU_NUM)
        vals = []
        for ntu in ntu_fine:
            z = _practical_at_d_ntu(
                d,
                ntu,
                ntu_ref,
                dp_hot_ref,
                c_cold_over_c_hot,
                pressure_drop_ratio,
                t,
                p_cold_in_over_p_hot_in,
                p_dead_over_p_hot_in,
                gamma,
                dp_max,
            )
            vals.append(z)
        vals = np.array(vals)
        valid = np.isfinite(vals)
        if np.any(valid):
            idx = np.nanargmin(vals)
            ntu_opt = float(ntu_fine[idx])
            y_opt = _y_axis(d, ntu_opt)
            z_opt = float(vals[idx])
            d_vals.append(d)
            ntu_opt_vals.append(ntu_opt)
            y_opt_vals.append(y_opt)
            z_opt_vals.append(z_opt)

    return np.array(d_vals), np.array(ntu_opt_vals), np.array(y_opt_vals), np.array(z_opt_vals)


def _compute_input_hash(**kwargs):
    """Compute hash of input parameters to check if cache is valid."""
    # Create a dictionary of all relevant inputs
    inputs = {
        "c_cold_over_c_hot": kwargs.get("c_cold_over_c_hot"),
        "st_over_f": kwargs.get("st_over_f"),
        "f_c_over_f_h": kwargs.get("f_c_over_f_h"),
        "d_r": kwargs.get("d_r"),
        "g2_h": kwargs.get("g2_h"),
        "t": kwargs.get("t"),
        "p_cold_in_over_p_hot_in": kwargs.get("p_cold_in_over_p_hot_in"),
        "p_dead_over_p_hot_in": kwargs.get("p_dead_over_p_hot_in"),
        "gamma": kwargs.get("gamma"),
        "pressure_drop_assumption": kwargs.get("pressure_drop_assumption"),
        "molar_mass_ratio": kwargs.get("molar_mass_ratio"),
        "a_r": kwargs.get("a_r"),
        "dp_max": kwargs.get("dp_max"),
        "d_over_d_ref_values": D_OVER_D_REF_VALUES.tolist(),
        "ntu_max_at_d_ref": NTU_MAX_AT_D_REF,
        "ntu_num": NTU_NUM,
        "ntu_min": NTU_MIN,
        "ntu_ref": NTU_REF,
        "dp_ref": DP_REF,
    }
    # Convert to JSON string and hash
    input_str = json.dumps(inputs, sort_keys=True)
    return hashlib.md5(input_str.encode()).hexdigest()


def run_sweep_and_plot(
    c_cold_over_c_hot=DEFAULT_C_COLD_OVER_C_HOT,
    st_over_f=DEFAULT_ST_OVER_F,
    f_c_over_f_h=DEFAULT_F_C_OVER_F_H,
    d_r=DEFAULT_D_R,
    g2_h=DEFAULT_G2_H,
    t=DEFAULT_T,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
    dp_max=DEFAULT_DP_MAX,
    base_name="newfig5",
):
    """Build 2D sweep (d_over_d_ref, NTU) and plot contour of practical availability."""
    # Check if we can load from cache
    input_hash = _compute_input_hash(
        c_cold_over_c_hot=c_cold_over_c_hot,
        st_over_f=st_over_f,
        f_c_over_f_h=f_c_over_f_h,
        d_r=d_r,
        g2_h=g2_h,
        t=t,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=gamma,
        pressure_drop_assumption=pressure_drop_assumption,
        molar_mass_ratio=molar_mass_ratio,
        a_r=a_r,
        dp_max=dp_max,
    )

    load_from_cache = False
    if DATA_FILE.exists():
        try:
            df = pd.read_parquet(DATA_FILE)
            if "input_hash" in df.columns and len(df) > 0:
                cached_hash = df["input_hash"].iloc[0]
                if cached_hash == input_hash:
                    load_from_cache = True
                    print(f"Loading cached data from {DATA_FILE}")
        except Exception as e:
            print(f"Error reading cache: {e}, will recalculate")

    if load_from_cache:
        # Load data from parquet
        df = pd.read_parquet(DATA_FILE)
        # Extract sweep data (where d_over_d_ref is not NaN)
        sweep_mask = df["d_over_d_ref"].notna()
        xx = df[sweep_mask]["d_over_d_ref"].values
        yy = df[sweep_mask]["ao_over_ao_ref"].values
        zz = df[sweep_mask]["dqom_over_qmax"].values
        # Extract optimal line data
        opt_mask = df["d_opt"].notna()
        d_opt_line = df[opt_mask]["d_opt"].values
        y_opt_line = df[opt_mask]["ao_opt"].values
        z_opt_line = df[opt_mask]["dqom_opt"].values
        # Get fig4 optimum
        fig4_row = df[df["is_fig4_opt"]]
        if len(fig4_row) > 0:
            y_opt = float(fig4_row["ao_over_ao_ref"].iloc[0])
        else:
            y_opt = None
    else:
        # Calculate data
        print("Computing sweep data...")
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

        # Reference dp curve at d=1 (used for all d via multiplier d**1.407)
        ntu_ref, dp_hot_ref = _compute_reference_dp_curve(
            c_cold_over_c_hot,
            st_over_f,
            f_c_over_f_h,
            d_r,
            g2_h,
            pressure_drop_ratio,
            dp_max,
        )

        # Fig4 optimum: at d=1, NTU that minimizes practical
        ntu_opt = _fig4_optimum_ntu(
            ntu_ref,
            dp_hot_ref,
            c_cold_over_c_hot,
            pressure_drop_ratio,
            t,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            gamma,
            dp_max,
        )
        if ntu_opt is not None:
            y_opt = _y_axis(1.0, ntu_opt)
        else:
            y_opt = None

        # Optimal NTU line: for each d, find optimal NTU
        d_opt_line, ntu_opt_line, y_opt_line, z_opt_line = _optimal_ntu_line(
            ntu_ref,
            dp_hot_ref,
            c_cold_over_c_hot,
            pressure_drop_ratio,
            t,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            gamma,
            dp_max,
        )

        # 2D sweep: for each d, sweep NTU (NTU max increases for smaller d)
        # For smaller d/d_ref, pressure drop is lower (scaled by d**1.407), so we can go to higher NTU
        xx, yy, zz = [], [], []
        for d in D_OVER_D_REF_VALUES:
            ntu_max_d = _ntu_max_for_d(d, dp_max)  # NTU max depends on d and dp_max
            ntu_sweep = np.linspace(NTU_MIN, ntu_max_d, NTU_NUM)
            for ntu in ntu_sweep:
                z = _practical_at_d_ntu(
                    d,
                    ntu,
                    ntu_ref,
                    dp_hot_ref,
                    c_cold_over_c_hot,
                    pressure_drop_ratio,
                    t,
                    p_cold_in_over_p_hot_in,
                    p_dead_over_p_hot_in,
                    gamma,
                    dp_max,
                )
                y = _y_axis(d, ntu)
                xx.append(d)
                yy.append(y)
                zz.append(z)

        xx = np.array(xx)
        yy = np.array(yy)
        zz = np.array(zz)
        valid = np.isfinite(zz)
        if not np.any(valid):
            print("No valid practical values in sweep.")
            print(f"Total points: {len(zz)}, Valid points: {np.sum(valid)}")
            return

        print(f"Valid points: {np.sum(valid)}/{len(zz)}")
        print(f"x range: [{xx[valid].min():.3f}, {xx[valid].max():.3f}]")
        print(f"y range: [{yy[valid].min():.3f}, {yy[valid].max():.3f}]")
        print(f"z range: [{zz[valid].min():.3f}, {zz[valid].max():.3f}]")

        # Save data to parquet
        # Create separate DataFrames and concatenate
        sweep_df = pd.DataFrame(
            {
                "input_hash": [input_hash] * len(xx),
                "d_over_d_ref": xx,
                "ao_over_ao_ref": yy,
                "dqom_over_qmax": zz,
                "d_opt": [np.nan] * len(xx),
                "ao_opt": [np.nan] * len(xx),
                "dqom_opt": [np.nan] * len(xx),
                "is_fig4_opt": [False] * len(xx),
            }
        )

        opt_df = pd.DataFrame(
            {
                "input_hash": [input_hash] * len(d_opt_line),
                "d_over_d_ref": [np.nan] * len(d_opt_line),
                "ao_over_ao_ref": [np.nan] * len(d_opt_line),
                "dqom_over_qmax": [np.nan] * len(d_opt_line),
                "d_opt": d_opt_line,
                "ao_opt": y_opt_line,
                "dqom_opt": z_opt_line,
                "is_fig4_opt": [False] * len(d_opt_line),
            }
        )

        dfs = [sweep_df, opt_df]

        # Add fig4 optimum point if available
        if y_opt is not None:
            # Calculate dQo^M/Qmax for fig4 optimum
            z_fig4 = _practical_at_d_ntu(
                1.0,
                ntu_opt,
                ntu_ref,
                dp_hot_ref,
                c_cold_over_c_hot,
                pressure_drop_ratio,
                t,
                p_cold_in_over_p_hot_in,
                p_dead_over_p_hot_in,
                gamma,
                dp_max,
            )
            fig4_df = pd.DataFrame(
                {
                    "input_hash": [input_hash],
                    "d_over_d_ref": [1.0],
                    "ao_over_ao_ref": [y_opt],
                    "dqom_over_qmax": [z_fig4],
                    "d_opt": [np.nan],
                    "ao_opt": [np.nan],
                    "dqom_opt": [np.nan],
                    "is_fig4_opt": [True],
                }
            )
            dfs.append(fig4_df)

        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(DATA_FILE)
        print(f"Data saved to {DATA_FILE}")

    # Interpolate onto regular grid for contour (smoothness = CONTOUR_GRID_N, not NTU_NUM)
    # The irregular (x, y, z) points from the sweep are interpolated onto a regular grid
    # for smooth contour plotting. CONTOUR_GRID_N controls smoothness, not NTU_NUM.
    valid = np.isfinite(zz)
    x_min, x_max = D_OVER_D_REF_VALUES.min(), D_OVER_D_REF_VALUES.max()
    y_min, y_max = yy[valid].min(), yy[valid].max()
    # Slightly extend for nicer contours
    y_min = max(y_min * 0.95, 1e-5)
    y_max = min(y_max * 1.05, AO_MAX)
    grid_x = np.linspace(x_min, x_max, CONTOUR_GRID_N)
    grid_y = np.linspace(y_min, y_max, CONTOUR_GRID_N)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = griddata(
        (xx[valid], yy[valid]),
        zz[valid],
        (X, Y),
        method="cubic",
        fill_value=np.nan,
    )

    # Debug: check interpolation result
    # If cubic interpolation fails (e.g., insufficient points), fall back to linear
    z_valid_count = np.sum(np.isfinite(Z))
    print(f"Interpolated Z: {z_valid_count}/{Z.size} valid values")
    if z_valid_count == 0:
        print("Trying linear interpolation instead of cubic...")
        Z = griddata(
            (xx[valid], yy[valid]),
            zz[valid],
            (X, Y),
            method="linear",
            fill_value=np.nan,
        )
        z_valid_count = np.sum(np.isfinite(Z))
        print(f"After linear: {z_valid_count}/{Z.size} valid values")

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
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$d / d_{\mathrm{ref}}$")
    ax.set_ylabel(r"$A_o/A_{o,\mathrm{ref}}$")

    # Contour plot (practical availability)
    # Check for valid Z values before creating levels
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    if not (np.isfinite(z_min) and np.isfinite(z_max)):
        print("Warning: No valid Z values for contour plot.")
        return

    levels = np.linspace(z_min, min(0, z_max), 15)
    # Ensure levels don't contain NaN or inf
    levels = levels[np.isfinite(levels)]
    if len(levels) < 2:
        print("Warning: Not enough valid levels for contour plot.")
        return

    cs = ax.contourf(X, Y, Z, levels=levels, cmap="gray_r", extend="both")
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3, alpha=0.5)

    # Optimal NTU line: plot line of optimal NTU for each d/d_ref
    if len(d_opt_line) > 0:
        # Filter points within plot bounds
        mask = (d_opt_line >= x_min) & (d_opt_line <= x_max) & (y_opt_line >= y_min) & (y_opt_line <= y_max)
        if np.any(mask):
            ax.plot(
                d_opt_line[mask],
                y_opt_line[mask],
                "k-",
                linewidth=1.5,
                zorder=6,
                label="optimal NTU",
            )

    # Ref at (1, 1): grey cross
    ax.scatter([1.0], [1.0], marker="x", s=80, color="grey", linewidths=2, zorder=5, label="ref")
    # Fig4 optimum: black circle (at d=1, y = y_opt)
    if y_opt is not None and y_min <= y_opt <= y_max:
        ax.scatter(
            [1.0],
            [y_opt],
            s=80,
            facecolors="black",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label="fig4 opt",
        )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(r"HEx $\Delta Q_0^M / Q_{\mathrm{max}}$")
    plt.colorbar(cs, ax=ax, format=mtick.FormatStrFormatter("%.2f"))
    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png"]:
        path = os.path.join(save_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    run_sweep_and_plot(
        c_cold_over_c_hot=DEFAULT_C_COLD_OVER_C_HOT,
        st_over_f=DEFAULT_ST_OVER_F,
        f_c_over_f_h=DEFAULT_F_C_OVER_F_H,
        d_r=DEFAULT_D_R,
        g2_h=DEFAULT_G2_H,
        t=DEFAULT_T,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
        dp_max=DEFAULT_DP_MAX,
        base_name="newfig5",
    )
