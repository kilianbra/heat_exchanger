"""
2D contour plot of practical availability vs d_over_d_ref and scaled NTU.
Uses same inputs as newfig4 (Helicopter defaults). Pressure drop includes
multiplier d_over_d_ref**1.407. Y-axis: (NTU/NTU_match)**(-1.704) * (d_over_d_ref)**(-0.704).
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import xflow_ver07
from scipy.interpolate import griddata
from xflow_ver07 import (
    calculate_epsilon_ntu_curve,
    calculate_pressure_drop_ratio,
    practical_unavailable_creation_hex,
)

from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.dirname(os.path.abspath(__file__))

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
NTU_MATCH = 1.479
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 1.0
DEFAULT_DP_MAX = 0.2

# Sweep: d_over_d_ref from 0.5 to 1.2, 8 values step 0.1
D_OVER_D_REF_VALUES = np.linspace(0.5, 1.2, 10)  # 0.5, 0.6, ..., 1.2
NTU_MAX_AT_D_REF = 15.0  # NTU max at d/d_ref = 1; for smaller d, NTU max increases (2/d)
NTU_NUM = 10  # number of NTU points per d (sweep resolution)
CONTOUR_GRID_N = 20  # grid size for interpolation; contour smoothness is set by this, not NTU_NUM


def _ntu_max_for_d(d_over_d_ref):
    """NTU max for a given d/d_ref; increases as d decreases (e.g. 2/d)."""
    return NTU_MAX_AT_D_REF / d_over_d_ref


def _y_axis(d_over_d_ref, ntu):
    """Y-axis coordinate: (NTU/NTU_match)**(-1.704) * (d_over_d_ref)**(-0.704)."""
    return (ntu / NTU_MATCH) ** (-1.704) * (d_over_d_ref) ** (-0.704)


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
    xflow_ver07.SHOW_CUBIC = True
    xflow_ver07.NTU_MATCH = NTU_MATCH
    ntu_max_ref = _ntu_max_for_d(D_OVER_D_REF_VALUES.min())
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
    """Find optimal NTU for each d/d_ref, return (d_values, ntu_opt_values, y_opt_values)."""
    d_vals = []
    ntu_opt_vals = []
    y_opt_vals = []

    for d in D_OVER_D_REF_VALUES:
        ntu_max_d = _ntu_max_for_d(d)
        ntu_fine = np.linspace(0.35, ntu_max_d, NTU_NUM)
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
            d_vals.append(d)
            ntu_opt_vals.append(ntu_opt)
            y_opt_vals.append(y_opt)

    return np.array(d_vals), np.array(ntu_opt_vals), np.array(y_opt_vals)


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
    d_opt_line, ntu_opt_line, y_opt_line = _optimal_ntu_line(
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
        ntu_max_d = _ntu_max_for_d(d)  # NTU max increases as d decreases (2/d formula)
        ntu_sweep = np.linspace(0.2, ntu_max_d, NTU_NUM)
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

    # Interpolate onto regular grid for contour (smoothness = CONTOUR_GRID_N, not NTU_NUM)
    # The irregular (x, y, z) points from the sweep are interpolated onto a regular grid
    # for smooth contour plotting. CONTOUR_GRID_N controls smoothness, not NTU_NUM.
    x_min, x_max = D_OVER_D_REF_VALUES.min(), D_OVER_D_REF_VALUES.max()
    y_min, y_max = yy[valid].min(), yy[valid].max()
    # Slightly extend for nicer contours
    y_min = max(y_min * 0.95, 1e-5)
    y_max = min(y_max * 1.05, 3.0)
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
    ax.legend(loc="upper right", fontsize=9)
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
        base_name="newfig5_salvaged",
    )
