"""
Bare bones version: 2D contour plot of practical availability vs d_over_d_ref and scaled NTU.
Focuses on the y-axis function (scaling laws) and practical unavailable creation only.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.interpolate import griddata

import xflow
from xflow import (
    calculate_epsilon_ntu_curve,
    calculate_pressure_drop_ratio,
    practical_unavailable_creation_hex,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.dirname(os.path.abspath(__file__))

# Helicopter defaults
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
DP_REF = 0.106

# Y-axis type: "ao_over_ao_ref", "dp_over_p_in", "g2h", or "ntu"
Y_AXIS_TYPE = "dp_over_p_in"  # Options: "ao_over_ao_ref", "dp_over_p_in", "g2h", "ntu"
UNNORMALISED_Y_AXIS_IF_POSS = True  # If True, plot unnormalized values

# Sweep parameters
D_OVER_D_REF_VALUES = np.linspace(0.5, 4, 500)  # d/d_ref values
NTU_MAX_AT_D_REF = 4.0  # NTU max at d/d_ref = 1
NTU_NUM = 500  # number of NTU points per d
CONTOUR_GRID_N = 500  # grid size for interpolation
NTU_MIN = 0.4

match Y_AXIS_TYPE:
    case "dp_over_p_in":
        Y_MAX = 0.11
    case "ao_over_ao_ref":
        Y_MAX = 4
    case "g2h":
        Y_MAX = 0.03
    case "ntu":
        Y_MAX = 2.0


def _ntu_max_for_d(d_over_d_ref, dp_max, dp_ref=DP_REF):
    """NTU max for a given d/d_ref and dp_max.

    Formula: NTU_max = NTU_max_at_d_ref * (DP_max/DP_ref)^(1/4.407) * (d/d_ref)^{-1.407/4.407}
    """
    return NTU_MAX_AT_D_REF * (dp_max / dp_ref) ** (1 / 4.407) * (d_over_d_ref) ** (-1.407 / 4.407)


def _y_axis(d_over_d_ref, ntu, unnormalised=False):
    """Y-axis coordinate based on Y_AXIS_TYPE.

    This is the fundamental scaling law: for fixed d_over_d_ref and fixed mass/heat transfer area,
    these are the scaling laws for certain outputs as you vary NTU.

    Options:
    - "ao_over_ao_ref": Ao/Ao_ref = (ntu/NTU_REF)^(-1.704) * (d/d_ref)^(-0.704)
    - "dp_over_p_in": dp/p_in = dp_ref * (d/d_ref)^1.407 * (ntu/NTU_REF)^4.407
    - "g2h": g2h = g2h_ref * ((ntu/NTU_REF)^(-1.704) * (d/d_ref)^(-0.704))^(-2)
    - "ntu": NTU (normalized or unnormalized)

    If unnormalised=True, multiply by reference value to get absolute units.
    """
    match Y_AXIS_TYPE:
        case "ao_over_ao_ref":
            y_norm = (ntu / NTU_REF) ** (-1.704) * (d_over_d_ref) ** (-0.704)
            return y_norm
        case "dp_over_p_in":
            y_norm = (d_over_d_ref) ** (-1.407) * (ntu / NTU_REF) ** (4.407)
            if unnormalised:
                return y_norm * DP_REF
            return y_norm
        case "g2h":
            ao_over_ao_ref = (ntu / NTU_REF) ** (-1.704) * (d_over_d_ref) ** (-0.704)
            g2h_abs = DEFAULT_G2_H * (ao_over_ao_ref) ** (-2)
            if unnormalised:
                return g2h_abs
            return g2h_abs / DEFAULT_G2_H
        case "ntu":
            if unnormalised:
                return ntu
            return ntu / NTU_REF
        case _:
            raise ValueError(f"Unknown Y_AXIS_TYPE: {Y_AXIS_TYPE}")


def _get_y_axis_label():
    """Get the y-axis label based on Y_AXIS_TYPE."""
    match Y_AXIS_TYPE:
        case "ao_over_ao_ref":
            return r"$A_o/A_{o,\mathrm{ref}}$"
        case "dp_over_p_in":
            if UNNORMALISED_Y_AXIS_IF_POSS:
                return r"$\Delta p / p_{\mathrm{in}}$"
            return r"$\Delta p / p_{\mathrm{in}} / (\Delta p / p_{\mathrm{in}})_{\mathrm{ref}}$"
        case "g2h":
            if UNNORMALISED_Y_AXIS_IF_POSS:
                return r"$g_2$"
            return r"$g_2 / g_{2,\mathrm{ref}}$"
        case "ntu":
            if UNNORMALISED_Y_AXIS_IF_POSS:
                return r"$\mathrm{NTU}$"
            return r"$\mathrm{NTU} / \mathrm{NTU}_{\mathrm{ref}}$"
        case _:
            raise ValueError(f"Unknown Y_AXIS_TYPE: {Y_AXIS_TYPE}")


def _compute_dp_at_ntu_ref(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    pressure_drop_ratio,
):
    """Compute dp/p_in_hot at NTU=NTU_REF and d=d_ref using xflow.

    Returns the scalar dp_hot value at the reference point. All other dp values
    can be computed analytically:
        dp(NTU, d) = dp_at_ntu_ref * (NTU / NTU_REF)^4.407 * (d / d_ref)^1.407

    Uses the same ntu_max as xflow.py (DEFAULT_NTU_MAX = 5.0 for helicopter) to ensure
    the same NTU array discretization and anchor point for the cubic formula.
    """
    from xflow import calculate_capacity_ratios

    xflow.SHOW_CUBIC = True
    xflow.NTU_MATCH = NTU_REF
    # Use the same ntu_max as xflow.py uses for helicopter defaults (5.0)
    # This ensures the same NTU array discretization and anchor point
    ntu_max = 5.0  # DEFAULT_NTU_MAX for helicopter defaults

    # Calculate the linear pressure drop coefficient first (before cubic transformation)
    C_min_over_C_hot, C_min_over_C_cold, _ = calculate_capacity_ratios(c_cold_over_c_hot)
    st_over_f_h = st_over_f
    st_over_f_c = st_over_f
    dp_coeff_normal = g2_h * (
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
    )

    # Generate NTU array to find anchor point
    ntu_arr = np.linspace(0.1, ntu_max, 200)
    closest_idx = np.argmin(np.abs(ntu_arr - NTU_REF))
    ntu_at_anchor = ntu_arr[closest_idx]

    # Calculate linear dp at anchor point
    dp_linear_at_anchor = dp_coeff_normal * ntu_at_anchor

    # Apply cubic formula to get exact value at NTU_REF
    # dp(NTU) = dp_linear(anchor) * (NTU / anchor)^4.407
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
    """Get epsilon and pressure drops at single (d, NTU). Returns (eps, dp_hot, dp_cold, validity).

    Uses the analytical formula:
        dp_hot = dp_at_ntu_ref * (NTU / NTU_REF)^4.407 * (d / d_ref)^1.407
    """
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
    """Thermal creation term only (no pressure drop) at single (d, NTU).

    This calculates the practical unavailable creation with dp_hot=0 and dp_cold=0,
    which represents the thermal creation term only.
    """
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


def _practical_viscous_at_d_ntu(
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
    """Viscous dissipation term at single (d, NTU).

    This is the difference between total practical unavailable creation and thermal creation.
    """
    total = _practical_at_d_ntu(
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
    thermal = _practical_thermal_at_d_ntu(
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

    if np.isnan(total) or np.isnan(thermal):
        return np.nan

    return total - thermal


def _optimal_ntu_line(
    dp_at_ntu_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
):
    """Find optimal NTU for each d/d_ref that minimizes practical unavailable energy.

    Returns (d_values, ntu_opt_values, y_opt_values, z_practical_opt_values).
    """
    d_vals = []
    ntu_opt_vals = []
    y_opt_vals = []
    z_practical_opt_vals = []

    # Use a fixed NTU grid covering the largest range needed
    ntu_max_global = _ntu_max_for_d(D_OVER_D_REF_VALUES.min(), dp_max)
    ntu_fine = np.linspace(NTU_MIN, ntu_max_global, NTU_NUM)

    print("Finding optimal NTU line...")
    for d in D_OVER_D_REF_VALUES:
        practical_vals = []

        for ntu in ntu_fine:
            z_practical = _practical_at_d_ntu(
                d,
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
            y_opt = _y_axis(d, ntu_opt, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
            z_practical_opt = float(practical_vals[idx_original])

            d_vals.append(d)
            ntu_opt_vals.append(ntu_opt)
            y_opt_vals.append(y_opt)
            z_practical_opt_vals.append(z_practical_opt)

    return (
        np.array(d_vals),
        np.array(ntu_opt_vals),
        np.array(y_opt_vals),
        np.array(z_practical_opt_vals),
    )


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
    base_name="fig9",
):
    """Build 2D sweep (d_over_d_ref, NTU) and plot contour of practical availability."""
    print("Computing reference pressure drop...")
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

    # Compute reference dp at NTU_REF
    dp_at_ntu_ref = _compute_dp_at_ntu_ref(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        pressure_drop_ratio,
    )
    print(
        f"  dp/p_in at NTU_REF (input vs from ntu_ref): {DP_REF * 100:.2f}% ({dp_at_ntu_ref * 100:.2f}%) and dp/pin _ratio c/h = {pressure_drop_ratio:.3f}"
    )

    # Find optimal NTU line
    (
        d_opt_line,
        ntu_opt_line,
        y_opt_line,
        z_practical_opt_line,
    ) = _optimal_ntu_line(
        dp_at_ntu_ref,
        c_cold_over_c_hot,
        pressure_drop_ratio,
        t,
        p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in,
        gamma,
        dp_max,
    )

    # 2D sweep: for each d, sweep NTU
    print("Computing sweep...")
    xx, yy, zz = [], [], []
    for d in D_OVER_D_REF_VALUES:
        ntu_max_d = _ntu_max_for_d(d, dp_max)
        ntu_sweep = np.linspace(NTU_MIN, ntu_max_d, NTU_NUM)
        for ntu in ntu_sweep:
            z = _practical_at_d_ntu(
                d,
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
            y = _y_axis(d, ntu, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
            xx.append(d)
            yy.append(y)
            zz.append(z)

    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    valid = np.isfinite(zz)
    if not np.any(valid):
        print("No valid practical values in sweep.")
        return

    print(f"Valid points: {np.sum(valid)}/{len(zz)}")
    print(f"x range: [{xx[valid].min():.3f}, {xx[valid].max():.3f}]")
    print(f"y range: [{yy[valid].min():.3f}, {yy[valid].max():.3f}]")
    print(f"z range: [{zz[valid].min():.3f}, {zz[valid].max():.3f}]")

    # Interpolate onto regular grid for contour
    print("Interpolating onto regular grid...")
    x_min, x_max = D_OVER_D_REF_VALUES.min(), D_OVER_D_REF_VALUES.max()
    y_min, y_max = yy[valid].min(), yy[valid].max()
    y_min = max(y_min * 0.95, 1e-5)
    y_max = min(y_max * 1.05, Y_MAX)
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

    # Fallback to linear if cubic fails
    z_valid_count = np.sum(np.isfinite(Z))
    if z_valid_count == 0:
        print("Trying linear interpolation instead...")
        Z = griddata(
            (xx[valid], yy[valid]),
            zz[valid],
            (X, Y),
            method="linear",
            fill_value=np.nan,
        )

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
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$d / d_{\mathrm{ref}}$")
    ax.set_ylabel(_get_y_axis_label())

    # Contour plot
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    if not (np.isfinite(z_min) and np.isfinite(z_max)):
        print("Warning: No valid Z values for contour plot.")
        return

    levels = np.linspace(z_min, min(0, z_max), 15)
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
                label="optimal NTU (practical)",
            )

            # Find point on optimal line nearest to d/d_ref = 1
            idx_d1 = np.argmin(np.abs(d_opt_line - 1.0))
            if idx_d1 < len(d_opt_line):
                d_d1 = d_opt_line[idx_d1]
                y_d1 = y_opt_line[idx_d1]
                if x_min <= d_d1 <= x_max and y_min <= y_d1 <= y_max:
                    ax.scatter(
                        [d_d1],
                        [y_d1],
                        s=80,
                        facecolors="black",
                        edgecolors="black",
                        linewidths=2,
                        zorder=5,
                        label="opt at d/d_ref=1",
                    )

    # Reference point
    y_ref = _y_axis(1.0, NTU_REF, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
    ax.scatter([1.0], [y_ref], marker="x", s=80, color="grey", linewidths=2, zorder=5, label="ref")

    ax.set_title(r"HEx $\Delta Q_0^M / Q_{\mathrm{max}}$")
    plt.colorbar(cs, ax=ax, format=mtick.FormatStrFormatter("%.2f"))
    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png"]:
        path = os.path.join(save_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)

    # Print 10 evenly spaced d/d_ref and dQ0^M at optimum
    if len(d_opt_line) > 0:
        d_sample = np.linspace(d_opt_line.min(), d_opt_line.max(), 8)
        z_sample = np.interp(d_sample, d_opt_line, z_practical_opt_line)
        print("\n10 evenly spaced d/d_ref and ΔQ₀^M/Q_max at optimum:")
        print(f"   {'d/d_ref':>10} {'ΔQ₀^M/Q_max':>12}")
        for d, z in zip(d_sample, z_sample, strict=True):
            print(f"   {d:10.4f} {z:12.6f}")


def plot_thermal_viscous_vs_core_volume_ratio(
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
    base_name="fig6_fast_thermal_viscous",
    use_optimal_ntu=True,
    fixed_ntu=None,
):
    """Plot thermal creation and viscous dissipation terms vs core volume ratio (d/d_ref).

    Parameters:
    -----------
    use_optimal_ntu : bool
        If True, use optimal NTU for each d/d_ref (minimizes total practical unavailable creation).
        If False, use fixed_ntu for all d/d_ref values.
    fixed_ntu : float, optional
        Fixed NTU value to use if use_optimal_ntu=False. If None and use_optimal_ntu=False,
        uses NTU_REF.
    """
    print("Computing reference pressure drop...")
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

    # Compute reference dp at NTU_REF
    dp_at_ntu_ref = _compute_dp_at_ntu_ref(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        pressure_drop_ratio,
    )
    print(
        f"  dp/p_in at NTU_REF (input vs from ntu_ref): {DP_REF * 100:.2f}% ({dp_at_ntu_ref * 100:.2f}%) and dp/pin _ratio c/h = {pressure_drop_ratio:.3f}"
    )

    # Sweep over d/d_ref
    print("Computing thermal and viscous terms vs core volume ratio...")
    d_vals = []
    thermal_vals = []
    viscous_vals = []
    ntu_vals = []

    if use_optimal_ntu:
        # Find optimal NTU for each d/d_ref
        print("Finding optimal NTU for each d/d_ref...")
        ntu_max_global = _ntu_max_for_d(D_OVER_D_REF_VALUES.min(), dp_max)
        ntu_fine = np.linspace(NTU_MIN, ntu_max_global, NTU_NUM)

        for d in D_OVER_D_REF_VALUES:
            practical_vals = []
            for ntu in ntu_fine:
                z_practical = _practical_at_d_ntu(
                    d,
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

                # Calculate thermal and viscous at optimal NTU
                thermal = _practical_thermal_at_d_ntu(
                    d,
                    ntu_opt,
                    dp_at_ntu_ref,
                    c_cold_over_c_hot,
                    pressure_drop_ratio,
                    t,
                    p_cold_in_over_p_hot_in,
                    p_dead_over_p_hot_in,
                    gamma,
                    dp_max,
                )
                viscous = _practical_viscous_at_d_ntu(
                    d,
                    ntu_opt,
                    dp_at_ntu_ref,
                    c_cold_over_c_hot,
                    pressure_drop_ratio,
                    t,
                    p_cold_in_over_p_hot_in,
                    p_dead_over_p_hot_in,
                    gamma,
                    dp_max,
                )

                if not (np.isnan(thermal) or np.isnan(viscous)):
                    d_vals.append(d)
                    thermal_vals.append(thermal)
                    viscous_vals.append(viscous)
                    ntu_vals.append(ntu_opt)
    else:
        # Use fixed NTU
        ntu_to_use = fixed_ntu if fixed_ntu is not None else NTU_REF
        print(f"Using fixed NTU = {ntu_to_use}")

        for d in D_OVER_D_REF_VALUES:
            thermal = _practical_thermal_at_d_ntu(
                d,
                ntu_to_use,
                dp_at_ntu_ref,
                c_cold_over_c_hot,
                pressure_drop_ratio,
                t,
                p_cold_in_over_p_hot_in,
                p_dead_over_p_hot_in,
                gamma,
                dp_max,
            )
            viscous = _practical_viscous_at_d_ntu(
                d,
                ntu_to_use,
                dp_at_ntu_ref,
                c_cold_over_c_hot,
                pressure_drop_ratio,
                t,
                p_cold_in_over_p_hot_in,
                p_dead_over_p_hot_in,
                gamma,
                dp_max,
            )

            if not (np.isnan(thermal) or np.isnan(viscous)):
                d_vals.append(d)
                thermal_vals.append(thermal)
                viscous_vals.append(viscous)
                ntu_vals.append(ntu_to_use)

    d_vals = np.array(d_vals)
    thermal_vals = np.array(thermal_vals)
    viscous_vals = np.array(viscous_vals)

    if len(d_vals) == 0:
        print("No valid points found.")
        return

    print(f"Valid points: {len(d_vals)}")
    print(f"Thermal range: [{thermal_vals.min():.4f}, {thermal_vals.max():.4f}]")
    print(f"Viscous range: [{viscous_vals.min():.4f}, {viscous_vals.max():.4f}]")

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

    ax.plot(d_vals, thermal_vals, "k-", linewidth=1.5, label="Thermal creation")
    ax.plot(d_vals, viscous_vals, "k--", linewidth=1.5, label="Viscous dissipation")

    ax.set_xlabel(r"$d / d_{\mathrm{ref}}$")
    ax.set_ylabel(r"$\Delta \dot{Q}_0^\mathrm{M} / \dot{Q}_{\mathrm{max}}$")
    ax.set_title("Thermal creation and viscous dissipation vs core volume ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

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
        base_name="fig9",
    )
