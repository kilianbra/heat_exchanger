"""
2D contour plot of practical availability vs A/A_ref and Ao/Ao_ref.
Uses same inputs as newfig4 (Helicopter defaults).
For each Ao, g2_h = DEFAULT_G2_H / (Ao/Ao_ref)**2.
Uses normal linear pressure drop formula (not cubic).
X-axis: A/A_ref = NTU/NTU_MATCH * (Ao/Ao_ref)**(0.587)
Y-axis: Ao/Ao_ref
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.interpolate import griddata
from xflow import (
    calculate_capacity_ratios,
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

# Sweep: Ao/Ao_ref from 0.5 to 1.2
AO_OVER_AO_REF_VALUES = np.linspace(0.5, 2.5, 500)
NTU_MAX = 15.0
NTU_NUM = 1000  # number of NTU points per Ao
CONTOUR_GRID_N = 300  # grid size for interpolation


def _a_over_a_ref(ao_over_ao_ref, ntu):
    """X-axis coordinate: NTU/NTU_MATCH * (Ao/Ao_ref)**(0.587)."""
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
    """Practical unavailable creation at single (Ao, NTU) using linear pressure drop formula.
    g2_h = DEFAULT_G2_H / (Ao/Ao_ref)**2
    """
    # Calculate modified g2_h based on Ao
    g2_h = DEFAULT_G2_H / (ao_over_ao_ref**2)

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


def _optimal_ntu_line_ao(
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
    """Find optimal NTU for each Ao/Ao_ref, return (ao_values, ntu_opt_values, a_over_a_ref_values, ao_values)."""
    ao_vals = []
    ntu_opt_vals = []
    a_over_a_ref_vals = []

    for ao in AO_OVER_AO_REF_VALUES:
        ntu_fine = np.linspace(0.15, NTU_MAX, 200)
        vals = []
        for ntu in ntu_fine:
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
            vals.append(z)
        vals = np.array(vals)
        valid = np.isfinite(vals)
        if np.any(valid):
            idx = np.nanargmin(vals)
            ntu_opt = float(ntu_fine[idx])
            a_over_a_ref_opt = _a_over_a_ref(ao, ntu_opt)
            ao_vals.append(ao)
            ntu_opt_vals.append(ntu_opt)
            a_over_a_ref_vals.append(a_over_a_ref_opt)

    return np.array(ao_vals), np.array(ntu_opt_vals), np.array(a_over_a_ref_vals)


def run_sweep_and_plot(
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
    base_name="newfig6",
):
    """Build 2D sweep (Ao/Ao_ref, NTU) and plot contour of practical availability."""
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

    # Optimal NTU line: for each Ao, find optimal NTU
    ao_opt_line, ntu_opt_line, a_over_a_ref_opt_line = _optimal_ntu_line_ao(
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

    # 2D sweep: for each Ao, sweep NTU
    # Outer loop over Ao/Ao_ref, inner loop over NTU
    # For each Ao, g2_h is modified: g2_h = DEFAULT_G2_H / (Ao/Ao_ref)**2
    # This accounts for the change in heat transfer area
    xx, yy, zz = [], [], []
    for ao in AO_OVER_AO_REF_VALUES:
        ntu_sweep = np.linspace(0.1, NTU_MAX, NTU_NUM)
        for ntu in ntu_sweep:
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
            x = _a_over_a_ref(ao, ntu)
            y = ao  # Y-axis is Ao/Ao_ref
            xx.append(x)
            yy.append(y)
            zz.append(z)

    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    valid = np.isfinite(zz)
    if not np.any(valid):
        print("No valid practical values in sweep.")
        return

    # Interpolate onto regular grid for contour
    x_min, x_max = xx[valid].min(), xx[valid].max()
    y_min, y_max = AO_OVER_AO_REF_VALUES.min(), AO_OVER_AO_REF_VALUES.max()
    # Slightly extend for nicer contours
    x_min = max(x_min * 0.95, 1e-5)
    x_max = min(x_max * 1.05, 15.0)
    y_min = max(y_min * 0.95, 0.4)
    y_max = min(y_max * 1.05, 2.5)
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
    ax.set_xlabel(r"$A / A_{\mathrm{ref}}$")
    ax.set_ylabel(r"$A_o/A_{o,\mathrm{ref}}$")

    # Contour plot (practical availability)
    levels = np.linspace(np.nanmin(Z), min(0, np.nanmax(Z)), 15)
    cs = ax.contourf(X, Y, Z, levels=levels, cmap="gray_r", extend="both")
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3, alpha=0.5)

    # Optimal NTU line: plot line of optimal NTU for each Ao/Ao_ref
    if len(ao_opt_line) > 0:
        # Filter points within plot bounds
        mask = (
            (a_over_a_ref_opt_line >= x_min)
            & (a_over_a_ref_opt_line <= x_max)
            & (ao_opt_line >= y_min)
            & (ao_opt_line <= y_max)
        )
        if np.any(mask):
            ax.plot(
                a_over_a_ref_opt_line[mask],
                ao_opt_line[mask],
                "k-",
                linewidth=1.5,
                zorder=6,
                label="optimal Ao/Ao_ref",
            )

    # Ref at (1, 1): grey cross (A/A_ref=1, Ao/Ao_ref=1)
    ax.scatter([1.0], [1.0], marker="x", s=80, color="grey", linewidths=2, zorder=5, label="ref")
    ax.legend(loc="lower right", fontsize=9)
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
        t=DEFAULT_T,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
        dp_max=DEFAULT_DP_MAX,
        base_name="newfig6",
    )
