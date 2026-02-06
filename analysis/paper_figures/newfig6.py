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

# Y-axis option: True = Ao/Ao_ref, False = g^2_h
# g^2_h = DEFAULT_G2_H / (Ao/Ao_ref)^2
y_axis_Ao_rather_than_g2h = False


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


def _find_optimal_line_from_sweep(ao_values, ntu_values, z_values):
    """Find optimal NTU for each Ao/Ao_ref from sweep results.
    Returns (ao_opt_values, a_over_a_ref_opt_values) for plotting the optimal line.
    """
    ao_opt_vals = []
    a_over_a_ref_opt_vals = []

    # Group by unique Ao values (using tolerance to handle floating point issues)
    unique_ao = np.unique(ao_values)

    for ao in unique_ao:
        # Find all points with this Ao value (within tolerance)
        mask = np.abs(ao_values - ao) < 1e-10
        if not np.any(mask):
            continue

        # Get z values for this Ao
        z_at_ao = z_values[mask]
        ntu_at_ao = ntu_values[mask]

        # Find valid (finite) values
        valid = np.isfinite(z_at_ao)
        if not np.any(valid):
            continue

        # Find minimum z value
        idx_min = np.nanargmin(z_at_ao[valid])
        ntu_opt = ntu_at_ao[valid][idx_min]
        a_over_a_ref_opt = _a_over_a_ref(ao, ntu_opt)

        ao_opt_vals.append(ao)
        a_over_a_ref_opt_vals.append(a_over_a_ref_opt)

    return np.array(ao_opt_vals), np.array(a_over_a_ref_opt_vals)


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
    # Is cold pressure ratio equal, much smaller or to be calculated from hot based on fluid properties?
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        pressure_drop_assumption,
        c_cold_over_c_hot,
        t,
        d_r,
        molar_mass_ratio,
        sigma_r,
        p_cold_in_over_p_hot_in,
    )

    # 2D sweep: for each Ao, sweep NTU
    # Outer loop over Ao/Ao_ref, inner loop over NTU
    # For each Ao, g2_h is modified: g2_h = DEFAULT_G2_H / (Ao/Ao_ref)**2
    # This accounts for the change in heat transfer area
    xx, yy, zz, ntu_sweep_vals, ao_sweep_vals = [], [], [], [], []
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
            # Y-axis: either Ao/Ao_ref or g^2_h
            if y_axis_Ao_rather_than_g2h:
                y = ao  # Y-axis is Ao/Ao_ref
            else:
                # g^2_h = DEFAULT_G2_H / (Ao/Ao_ref)^2
                y = DEFAULT_G2_H / (ao**2) if ao > 0 else np.nan
            xx.append(x)
            yy.append(y)
            zz.append(z)
            ntu_sweep_vals.append(ntu)
            ao_sweep_vals.append(ao)

    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    ntu_sweep_vals = np.array(ntu_sweep_vals)
    ao_sweep_vals = np.array(ao_sweep_vals)

    # Find optimal NTU line from sweep results (group by ao values, not y values)
    ao_opt_line, a_over_a_ref_opt_line = _find_optimal_line_from_sweep(ao_sweep_vals, ntu_sweep_vals, zz)

    # Calculate y-coordinates for optimal line based on chosen axis
    if y_axis_Ao_rather_than_g2h:
        y_opt_line = ao_opt_line
    else:
        # Calculate g^2_h for optimal points: DEFAULT_G2_H / (Ao/Ao_ref)^2
        y_opt_line = DEFAULT_G2_H / (ao_opt_line**2)
        # Filter out invalid points
        valid_opt = np.isfinite(y_opt_line) & (ao_opt_line > 0)
        ao_opt_line = ao_opt_line[valid_opt]
        a_over_a_ref_opt_line = a_over_a_ref_opt_line[valid_opt]
        y_opt_line = y_opt_line[valid_opt]
    valid = np.isfinite(zz)
    if not np.any(valid):
        print("No valid practical values in sweep.")
        return

    # Interpolate onto regular grid for contour
    # Convert irregular (x, y, z) points from sweep to regular grid for contour plotting
    x_min, x_max = xx[valid].min(), xx[valid].max()
    if y_axis_Ao_rather_than_g2h:
        y_min, y_max = AO_OVER_AO_REF_VALUES.min(), AO_OVER_AO_REF_VALUES.max()
    else:
        y_min, y_max = yy[valid].min(), yy[valid].max()
    # Slightly extend for nicer contours
    x_min = max(x_min * 0.95, 1e-5)
    x_max = min(x_max * 1.05, 10.0)
    if y_axis_Ao_rather_than_g2h:
        y_min = max(y_min * 0.95, 0.4)
        y_max = min(y_max * 1.05, 2.5)
    else:
        # For log scale, ensure positive values
        y_min = max(y_min * 0.95, 1e-6)
        y_max = min(y_max * 1.05, 1.0)
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
    # X-axis label options
    # ax.set_xlabel(r"$A / A_{\mathrm{ref}}$")
    ax.set_xlabel(r"$V_{\mathrm{metal}} / V_{\mathrm{metal,ref}}$")
    # Y-axis label and scale based on chosen option
    if y_axis_Ao_rather_than_g2h:
        ax.set_ylabel(r"$A_o/A_{o,\mathrm{ref}}$")
    else:
        ax.set_ylabel(r"$g^2_h$")
        ax.set_yscale('log')

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
            & (y_opt_line >= y_min)
            & (y_opt_line <= y_max)
        )
        if np.any(mask):
            ax.plot(
                a_over_a_ref_opt_line[mask],
                y_opt_line[mask],
                "k-",
                linewidth=1.5,
                zorder=6,
                label="optimal Ao/Ao_ref" if y_axis_Ao_rather_than_g2h else "optimal",
            )

    # Ref point: grey cross
    if y_axis_Ao_rather_than_g2h:
        # At (A/A_ref=1, Ao/Ao_ref=1)
        ref_x, ref_y = 1.0, 1.0
    else:
        # At (A/A_ref=1, Ao/Ao_ref=1) gives g^2_h = DEFAULT_G2_H / (1^2) = DEFAULT_G2_H
        ref_x, ref_y = 1.0, DEFAULT_G2_H
    ax.scatter([ref_x], [ref_y], marker="x", s=80, color="grey", linewidths=2, zorder=5, label="ref")
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
