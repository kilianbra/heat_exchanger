"""
Analysis and visualization for general counterflow heat exchanger.

Case B: Air and combustion products heat exchanger
Generates publication-quality figures with optimization curves.

Based on the models in heat_exchanger.geometries.general_counterflow
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from heat_exchanger.correlations import general_hex_friction_factor, general_hex_j_factor
from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.general_counterflow import (
    calculate_pressure_ratio,
    rate_hex_simple,
    rate_hex_compressible_two_stream,
    xflow_guess_0d,
)

# ============================================================================
# CASE B PARAMETERS: Air and Combustion Products Heat Exchanger
# ============================================================================

# Fluid models (global)
FLUID_HOT = PerfectGasFluid.from_name("kerocomb_helicopter")
FLUID_COLD = PerfectGasFluid.from_name("air")

# Fluid inputs (global) - all other values derived from this
F_IN = FluidInputs(
    hot=FLUID_HOT,
    cold=FLUID_COLD,
    m_dot_hot=1.6,  # kg/s
    m_dot_cold=1.6,  # kg/s
    Th_in=980,  # K
    Ph_in=1.06e5,  # Pa (1.06 bar)
    Tc_in=576,  # K
    Pc_in=7.2e5,  # Pa (7.2 bar)
)

# Reference conditions (for non-dimensionalization)
TD = 300  # K
PD = 1e5  # Pa

# Geometry
T_OVER_DHC = 0.02  # t/d_h_c = 0.02 (85 micron t over 4 mm walls)
SIGMA_R = 2.0  # Ratio of free flow areas (Ao_h/Ao_c)
SIGMA_W = 1.0  # Ratio of heat transfer areas (Ah/Ac)
A_FR_OVER_AO_C = (1 + SIGMA_R) + 2 * T_OVER_DHC * (1 + SIGMA_W)  # Ratio of frontal area to cold side free flow area
D_H_C = 4e-3  # m, cold side hydraulic diameter

AQ_BASELINE = 43.0  # m², baseline total heat transfer area (Ah + Ac)

# Geometry parameters
LS_OVER_DH = 5.0  # Strip length to hydraulic diameter ratio


# Plotting parameters
TICK_LABEL_SIZE = 12
AXIS_LABEL_SIZE = 14
LEGEND_SIZE = 12
FLIP_Y_AXIS = True
MAX_DP = 0.10  # Maximum pressure drop (6%)


# Helper functions to calculate values from F_IN
def _get_capacity_ratio():
    """Calculate capacity ratio from F_IN."""
    state_hot = F_IN.hot.state(T=F_IN.Th_in, P=F_IN.Ph_in if F_IN.Ph_in else F_IN.Ph_out)
    state_cold = F_IN.cold.state(T=F_IN.Tc_in, P=F_IN.Pc_in)
    c_hot = F_IN.m_dot_hot * state_hot.cp
    c_cold = F_IN.m_dot_cold * state_cold.cp
    c_min = min(c_hot, c_cold)
    c_max = max(c_hot, c_cold)
    return c_min / c_max, c_hot / c_cold


def print_case_parameters():
    """Print case B parameters."""
    state_hot_in = F_IN.hot.state(T=F_IN.Th_in, P=F_IN.Ph_in if F_IN.Ph_in else F_IN.Ph_out)
    state_cold_in = F_IN.cold.state(T=F_IN.Tc_in, P=F_IN.Pc_in)
    cr, c_h_c = _get_capacity_ratio()

    print("=" * 80)
    print("CASE B: Air and Combustion Products Heat Exchanger")
    print("=" * 80)
    print("Hot fluid (combustion products):")
    print(f"  T_in = {F_IN.Th_in} K, P_in = {(F_IN.Ph_in if F_IN.Ph_in else F_IN.Ph_out) / 1e5:.2f} bar")
    print(f"  rho = {state_hot_in.rho:.3f} kg/m³, mu = {state_hot_in.mu:.2e} Pa·s, cp = {state_hot_in.cp:.0f} J/(kg·K)")
    print("Cold fluid (air):")
    print(f"  T_in = {F_IN.Tc_in} K, P_in = {F_IN.Pc_in / 1e5:.2f} bar")
    print(
        f"  rho = {state_cold_in.rho:.3f} kg/m³, mu = {state_cold_in.mu:.2e} Pa·s, cp = {state_cold_in.cp:.0f} J/(kg·K)"
    )
    print(f"Mass flow rate: {F_IN.m_dot_hot} kg/s (hot), {F_IN.m_dot_cold} kg/s (cold)")
    print(
        f"Geometry: d_h_c = {D_H_C * 1000:.1f} mm, sigma_r = {SIGMA_R} (Ao_h/Ao_c), "
        f"sigma_w = {SIGMA_W} (Ah/Ac), A_fr/Ao_c = {A_FR_OVER_AO_C}, Aq_baseline = {AQ_BASELINE} m²"
    )
    print(f"Capacity ratio: Cr = C_min/C_max = {cr:.4f} (cold side is C_min)")
    print("=" * 80)


def frontal_area_sweep():
    """
    Perform frontal area sweep to find baseline configuration.

    Returns baseline result and all valid results.
    """
    p_hot_in = F_IN.Ph_in if F_IN.Ph_in else F_IN.Ph_out
    state_hot_in = F_IN.hot.state(T=F_IN.Th_in, P=p_hot_in)
    state_cold_in = F_IN.cold.state(T=F_IN.Tc_in, P=F_IN.Pc_in)

    rho_hot_in = state_hot_in.rho
    rho_cold_in = state_cold_in.rho
    mu_hot_in = state_hot_in.mu
    mu_cold_in = state_cold_in.mu

    # Get Prandtl numbers from fluid models
    Pr_hot = (
        F_IN.hot.Pr if isinstance(F_IN.hot, PerfectGasFluid) else state_hot_in.cp * state_hot_in.mu / state_hot_in.k
    )
    Pr_cold = (
        F_IN.cold.Pr
        if isinstance(F_IN.cold, PerfectGasFluid)
        else state_cold_in.cp * state_cold_in.mu / state_cold_in.k
    )
    Pr = 0.5 * (Pr_hot + Pr_cold)  # Average for correlations

    # Calculate capacity ratio
    cr, c_h_c = _get_capacity_ratio()

    # Sweep frontal area from 0.03 to 0.9 m²
    a_fr_sweep = np.linspace(0.04, 0.20, 10)

    print("\nFrontal area sweep (filtering out dp_hot < 0% or dp_hot > 20%):")
    header = (
        f"{'A_fr (m²)':<12} {'eps':<8} {'dp_hot (%)':<12} {'dp_cold (%)':<12} "
        f"{'Re_hot':<12} {'Re_cold':<12} {'g²_hot':<12} {'g²_cold':<12}"
    )
    print(header)
    print("-" * 100)

    results = []

    # Calculate heat transfer areas from total A_q and sigma_w
    a_c = 2 * AQ_BASELINE / (1 + SIGMA_W)
    a_h = a_c * SIGMA_W

    # Calculate hydraulic diameters: dh_h/dh_c = sigma_r / sigma_w
    d_h_h = D_H_C * SIGMA_R / SIGMA_W

    for a_fr in a_fr_sweep:
        r_s, r_c, r_xf = run_one_model_example(a_fr, verbose=False)

        if r_c["cold"]["mach_out"] > 0.3 or r_c["hot"]["mach_out"] > 0.3:
            dp_hot = r_c["dp_hot"]
            dp_cold = r_c["dp_cold"]
            re_hot = r_c["re_hot"]
            re_cold = r_c["re_cold"]
            g2_hot = r_c["g2_hot"]
            g2_cold = r_c["g2_cold"]
            eps = r_c["eps"]
            A_fr_over_Ao_c = (1 + SIGMA_R) + 2 * T_OVER_DHC * (1 + SIGMA_W)
            Ao_c = a_fr / A_fr_over_Ao_c
            ao_h = Ao_c * SIGMA_R
        else:  # use r_s results
            dp_hot = r_s["dp_hot"]
            dp_cold = r_s["dp_cold"]
            re_hot = r_s["re_hot"]
            re_cold = r_s["re_cold"]
            g2_hot = r_s["g2_hot"]
            g2_cold = r_s["g2_cold"]
            eps = r_s["eps"]
            A_fr_over_Ao_c = (1 + SIGMA_R) + 2 * T_OVER_DHC * (1 + SIGMA_W)
            Ao_c = a_fr / A_fr_over_Ao_c
            ao_h = Ao_c * SIGMA_R

        # Skip this A_fr if dp_hot is negative or exceeds 20%
        if dp_hot < 0 or dp_hot > MAX_DP:
            continue

        results.append(
            {
                "a_fr": a_fr,
                "ao_h": ao_h,
                "eps": eps,
                "dp_hot": dp_hot,
                "dp_cold": dp_cold,
                "re_hot": re_hot,
                "re_cold": re_cold,
                "g2_hot": g2_hot,
                "g2_cold": g2_cold,
            }
        )

    # Check if any valid results were found
    if len(results) == 0:
        print("\n" + "=" * 80)
        print("WARNING: No valid A_fr values found!")
        print(f"All frontal areas in the sweep had dp_hot < 0% or dp_hot > {MAX_DP}%")
        print("Consider adjusting the sweep range or geometry parameters.")
        print("=" * 80)
        raise ValueError("No valid results found - all A_fr values had invalid pressure drops")

    # Select 10 evenly spaced values from the valid results
    n_values = min(10, len(results))
    indices = np.linspace(0, len(results) - 1, n_values, dtype=int)
    selected_results = [results[i] for i in indices]

    for r in selected_results:
        print(
            f"{r['a_fr']:<12.6f} {r['eps']:<8.4f} {r['dp_hot']:<12.4f} {r['dp_cold']:<12.4f} "
            f"{r['re_hot']:<12.1f} {r['re_cold']:<12.1f} {r['g2_hot']:<12.2e} {r['g2_cold']:<12.2e}"
        )

    # Find the one closest to 60% effectiveness
    target_eps = 0.60
    closest_idx = np.argmin([abs(r["eps"] - target_eps) for r in results])
    baseline_result = results[closest_idx]

    print("\n" + "=" * 80)
    print(f"Baseline selected (closest to {target_eps * 100:.0f}% effectiveness):")
    print(f"  A_fr_baseline = {baseline_result['a_fr']:.6f} m²")
    print(f"  Ao_h_baseline = {baseline_result['ao_h']:.6f} m²")
    print(f"  Effectiveness = {baseline_result['eps']:.4f} ({baseline_result['eps'] * 100:.2f}%)")
    print(f"  Pressure drop (hot) = {baseline_result['dp_hot']:.4f}%")
    print(f"  Pressure drop (cold) = {baseline_result['dp_cold']:.4f}%")
    print(f"  Re_hot = {baseline_result['re_hot']:.1f}")
    print(f"  Re_cold = {baseline_result['re_cold']:.1f}")
    print(f"  g²_hot = {baseline_result['g2_hot']:.2e}")
    print(f"  g²_cold = {baseline_result['g2_cold']:.2e}")
    print("=" * 80)

    return baseline_result, results


def calculate_optimization_curves(baseline_result):
    """
    Calculate optimization curves for Exergy and Euergy.

    Parameters
    ----------
    baseline_result : dict
        Baseline configuration from frontal_area_sweep

    Returns
    -------
    dict
        Dictionary containing all arrays for plotting
    """
    # Set baseline values
    a_fr_baseline = baseline_result["a_fr"]
    ao_h_baseline = baseline_result["ao_h"]
    ao_c_baseline = ao_h_baseline / SIGMA_R
    re_h_baseline = baseline_result["re_hot"]
    re_c_baseline = baseline_result["re_cold"]
    g2_hot_baseline = baseline_result["g2_hot"]
    g2_cold_baseline = baseline_result["g2_cold"]

    # Calculate heat transfer areas from total A_q and sigma_w
    a_c_baseline = 2 * AQ_BASELINE / (1 + SIGMA_W)
    a_h_baseline = a_c_baseline * SIGMA_W

    # Calculate baseline A_q/A_o for each side (for reference, not used in calculations)
    # aq_over_ao_h_baseline = a_h_baseline / ao_h_baseline
    # aq_over_ao_c_baseline = a_c_baseline / ao_c_baseline

    a_fr_min_ratio = 0.2  # Minimum A_fr/A_fr_baseline
    a_fr_max_ratio = 2.0  # Maximum A_fr/A_fr_baseline

    # Create array of A_fr_over_A_fr_baseline values
    a_fr_over_a_fr_baseline = np.linspace(a_fr_min_ratio, a_fr_max_ratio, 1000)

    # Calculate actual A_fr values
    a_fr = a_fr_over_a_fr_baseline * a_fr_baseline

    # Calculate free flow areas (scales with A_fr)
    ao_h = ao_h_baseline * a_fr_over_a_fr_baseline
    ao_c = ao_c_baseline * a_fr_over_a_fr_baseline

    # Heat transfer areas remain constant (A_q is fixed)
    a_h = np.full_like(a_fr, a_h_baseline)
    a_c = np.full_like(a_fr, a_c_baseline)

    # Calculate A_q/A_o for each side (varies with A_fr)
    aq_over_ao_h = a_h / ao_h
    aq_over_ao_c = a_c / ao_c

    # Calculate Reynolds numbers and g² values
    re_h = re_h_baseline / a_fr_over_a_fr_baseline
    re_c = re_c_baseline / a_fr_over_a_fr_baseline
    g2_hot = g2_hot_baseline / a_fr_over_a_fr_baseline**2
    g2_cold = g2_cold_baseline / a_fr_over_a_fr_baseline**2

    # Calculate friction factors and Stanton numbers
    f_hot = general_hex_friction_factor(re_h, LS_OVER_DH)
    f_cold = general_hex_friction_factor(re_c, LS_OVER_DH)

    # Get Prandtl numbers from fluid models
    p_hot_in = F_IN.Ph_in if F_IN.Ph_in else F_IN.Ph_out
    state_hot_ref = F_IN.hot.state(T=F_IN.Th_in, P=p_hot_in)
    state_cold_ref = F_IN.cold.state(T=F_IN.Tc_in, P=F_IN.Pc_in)
    if isinstance(F_IN.hot, PerfectGasFluid):
        Pr_hot = F_IN.hot.Pr
    else:
        Pr_hot = state_hot_ref.cp * state_hot_ref.mu / state_hot_ref.k
    if isinstance(F_IN.cold, PerfectGasFluid):
        Pr_cold = F_IN.cold.Pr
    else:
        Pr_cold = state_cold_ref.cp * state_cold_ref.mu / state_cold_ref.k
    Pr = 0.5 * (Pr_hot + Pr_cold)  # Average for correlations

    st_hot = general_hex_j_factor(re_h, LS_OVER_DH) * Pr ** (-2 / 3)
    st_cold = general_hex_j_factor(re_c, LS_OVER_DH) * Pr ** (-2 / 3)

    # Calculate NTU and effectiveness
    # NTU = 1 / ((1/St_hot + 1/St_cold * sigma_w/sigma_r) / (A_h/Ao_h))
    ntu = 1 / ((1 / st_hot + 1 / st_cold * SIGMA_W / SIGMA_R) / aq_over_ao_h)
    eps = ntu / (1 + ntu)

    # Calculate capacity ratio
    cr, c_h_c = _get_capacity_ratio()

    # Temperature and pressure ratios
    p_hot_in = F_IN.Ph_in if F_IN.Ph_in else F_IN.Ph_out
    th_in_td = F_IN.Th_in / TD
    tc_in_td = F_IN.Tc_in / TD
    t = th_in_td / tc_in_td
    ph_in_pd = p_hot_in / PD
    pc_in_pd = F_IN.Pc_in / PD

    # Calculate temperature ratios
    if c_h_c > 1:  # cold side is C_min
        t_hot_out_t_hot_in = 1 - eps / c_h_c * (1 - 1 / t)
        t_cold_out_t_cold_in = 1 + eps * (t - 1)
    else:  # hot side is C_min
        t_hot_out_t_hot_in = 1 - eps * (1 - 1 / t)
        t_cold_out_t_cold_in = 1 + eps * c_h_c * (t - 1)

    # Calculate pressure ratios using A_q/A_o
    # Note: Physical flow length L is the same for both sides
    p_hot_out_p_hot_in = calculate_pressure_ratio(
        aq_over_ao_h, f_hot, g2_hot, t_i_td=th_in_td, p_i_pd=ph_in_pd, eps=eps, t=t, hot_fluid=True, c_h_c=c_h_c
    )

    p_cold_out_p_cold_in = calculate_pressure_ratio(
        aq_over_ao_c, f_cold, g2_cold, t_i_td=tc_in_td, p_i_pd=pc_in_pd, eps=eps, t=t, hot_fluid=False, c_h_c=c_h_c
    )

    # Mask for pressure drop criteria: p_out/p_in < 1, and p_out/p_in > (1 - MAX_DP)
    valid_pressure_mask = (
        (p_hot_out_p_hot_in < 1)
        & (p_hot_out_p_hot_in > (1 - MAX_DP))
        & (p_cold_out_p_cold_in < 1)
        & (p_cold_out_p_cold_in > (1 - MAX_DP))
    )

    # Initialize arrays with NaNs
    dw_pot_ex = np.full_like(a_fr, np.nan)
    dw_pot_eu = np.full_like(a_fr, np.nan)

    # Get gamma values from fluid models
    state_hot_ref = F_IN.hot.state(T=F_IN.Th_in, P=p_hot_in)
    state_cold_ref = F_IN.cold.state(T=F_IN.Tc_in, P=F_IN.Pc_in)
    gamma_hot = state_hot_ref.gamma
    gamma_cold = state_cold_ref.gamma
    gm1og_hot = (gamma_hot - 1) / gamma_hot
    gm1og_cold = (gamma_cold - 1) / gamma_cold
    gm1og_avg = 0.5 * (gm1og_hot + gm1og_cold)

    # Calculate work potentials with different gamma values for hot and cold
    log_p_hot = np.log(p_hot_out_p_hot_in[valid_pressure_mask])
    log_p_cold = np.log(p_cold_out_p_cold_in[valid_pressure_mask])
    dw_pot_ex[valid_pressure_mask] = (
        np.log(t_hot_out_t_hot_in[valid_pressure_mask])
        + np.log(t_cold_out_t_cold_in[valid_pressure_mask])
        - gm1og_avg * (log_p_hot + log_p_cold)
    )

    # Calculate Euergy change with different gamma for hot and cold
    dw_pot_eu_hot = (
        (1 / ph_in_pd) ** gm1og_hot * th_in_td * (t_hot_out_t_hot_in * (1 / p_hot_out_p_hot_in) ** gm1og_hot - 1)
    )
    dw_pot_eu_cold = (
        (1 / pc_in_pd) ** gm1og_cold * tc_in_td * (t_cold_out_t_cold_in * (1 / p_cold_out_p_cold_in) ** gm1og_cold - 1)
    )
    dw_pot_eu[valid_pressure_mask] = dw_pot_eu_hot[valid_pressure_mask] + dw_pot_eu_cold[valid_pressure_mask]

    # Normalize by Q_max
    norm = 1 / (th_in_td - tc_in_td)
    dw_pot_ex_norm = dw_pot_ex * norm
    dw_pot_eu_norm = dw_pot_eu * norm

    return {
        "a_fr": a_fr,
        "dw_pot_ex_norm": dw_pot_ex_norm,
        "dw_pot_eu_norm": dw_pot_eu_norm,
        "eps": eps,
        "dp_hot": 1 - p_hot_out_p_hot_in,
        "dp_cold": 1 - p_cold_out_p_cold_in,
    }


def plot_optimization_curves(curves_data, save_svg=False):
    """
    Create publication-quality plot of optimization curves.

    Parameters
    ----------
    curves_data : dict
        Output from calculate_optimization_curves
    save_svg : bool
        Whether to save as SVG file
    """
    a_fr = curves_data["a_fr"]
    dw_pot_ex_norm = curves_data["dw_pot_ex_norm"]
    dw_pot_eu_norm = curves_data["dw_pot_eu_norm"]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot Exergy (dashed, black)
    ax.plot(a_fr, dw_pot_ex_norm, "k--", label="Exergy", linewidth=1.5)

    # Plot Euergy (solid, black)
    ax.plot(a_fr, dw_pot_eu_norm, "k-", label="Euergy", linewidth=1.5)

    # Find and mark optimal points
    # Euergy: minimum of dW_pot_Eu
    valid_mask_eu = ~np.isnan(dw_pot_eu_norm)
    if np.any(valid_mask_eu):
        eu_min_idx = np.nanargmin(dw_pot_eu_norm[valid_mask_eu])
        eu_min_x = a_fr[valid_mask_eu][eu_min_idx]
        eu_min_y = dw_pot_eu_norm[valid_mask_eu][eu_min_idx]
        ax.scatter(eu_min_x, eu_min_y, color="black", s=100, zorder=5, marker="o")
        ax.plot([eu_min_x, eu_min_x], [0, eu_min_y], "k--", alpha=0.5, linewidth=1)

    # Exergy: minimum of dW_pot_Ex
    valid_mask_ex = ~np.isnan(dw_pot_ex_norm)
    if np.any(valid_mask_ex):
        ex_min_idx = np.nanargmin(dw_pot_ex_norm[valid_mask_ex])
        ex_min_x = a_fr[valid_mask_ex][ex_min_idx]
        ex_min_y = dw_pot_ex_norm[valid_mask_ex][ex_min_idx]
        ax.scatter(ex_min_x, ex_min_y, color="black", s=100, zorder=5, marker="s")
        ax.plot([ex_min_x, ex_min_x], [0, ex_min_y], "k:", alpha=0.5, linewidth=1)

    # Set labels
    ax.set_xlabel("$A_{fr}$ [m²]", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(r"$\Delta Q_0/Q_{\mathrm{max}}$", fontsize=AXIS_LABEL_SIZE)

    # Set tick label sizes
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    # Use PercentFormatter for y-axis
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

    # Set x-axis to linear scale (changed from log)
    # ax.set_xscale("log")  # Removed log scale

    # Flip y-axis if requested
    if FLIP_Y_AXIS:
        ax.invert_yaxis()

    # Set x-axis limits to only show valid range
    valid_mask = ~np.isnan(dw_pot_ex_norm)
    if np.any(valid_mask):
        x_min = a_fr[valid_mask][0]
        x_max = a_fr[valid_mask][-1]
        ax.set_xlim(x_min, x_max)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc="best", fontsize=LEGEND_SIZE, frameon=True)

    # Tight layout
    plt.tight_layout()

    # Save as SVG if requested
    if save_svg:
        output_filename = "hex_Afr_isolation_figure.svg"
        plt.savefig(output_filename, format="svg", bbox_inches="tight")
        print(f"Figure saved as {output_filename}")

    return fig, ax


def run_one_model_example(a_fr, verbose=True):
    """Run example using the simple rate_hex_simple function."""
    if verbose:
        print("\n" + "=" * 80)
        print("Running simple model example with rate_hex_simple()")
        print("=" * 80)

    # Use baseline frontal area

    geom = {
        "A_fr": a_fr,
        "A_q": AQ_BASELINE,
        "d_h_c": D_H_C,
        "sigma_r": SIGMA_R,
        "sigma_w": SIGMA_W,
        "t_over_dhc": T_OVER_DHC,
        "ls_over_dh": LS_OVER_DH,
    }

    r_xf = xflow_guess_0d(geom, F_IN)

    r_s = rate_hex_simple(
        A_fr=a_fr,
        A_q=AQ_BASELINE,
        f_in=F_IN,
        d_h_c=D_H_C,
        sigma_r=SIGMA_R,
        sigma_w=SIGMA_W,
        t_over_dhc=T_OVER_DHC,
        ls_over_dh=LS_OVER_DH,
    )

    r_c = rate_hex_compressible_two_stream(
        A_fr=a_fr,
        A_q=AQ_BASELINE,
        f_in=F_IN,
        d_h_c=D_H_C,
        sigma_r=SIGMA_R,
        sigma_w=SIGMA_W,
        t_over_dhc=T_OVER_DHC,
        ls_over_dh=LS_OVER_DH,
        inlet_is_stagnation=True,
    )

    if verbose:
        print(f"Xflow guess Th_out, Tc_out: {r_xf[0]:.1f} K, {r_xf[1]:.1f} K")
        print(
            f"Xflow guess Δp_h, Δp_c: {(1 - r_xf[2] / F_IN.Ph_in) * 100:.1f}%, {(1 - r_xf[3] / F_IN.Pc_in) * 100:.1f}%"
        )

        print(f"\nSimple model results for A_fr = {a_fr:.4f} m², A_q = {AQ_BASELINE} m²")
        print(f"  Aq/Ao h,c: {r_s['Aq_over_Ao_h']:.4f}, {r_s['Aq_over_Ao_c']:.4f}")
        print(f"  Effectiveness: {r_s['eps']:.4f} ({r_s['eps'] * 100:.2f}%)")
        print(f"  Hot side pressure drop: {r_s['dp_hot'] * 100:.4f}%")
        print(f"  Cold side pressure drop: {r_s['dp_cold'] * 100:.4f}%")
        print(f"  Hot outlet temperature: {r_s['t_hot_out']:.1f} K")
        print(f"  Cold outlet temperature: {r_s['t_cold_out']:.1f} K")
        print(f"  Re_hot: {r_s['re_hot']:.0f}")
        print(f"  Re_cold: {r_s['re_cold']:.0f}")
        print(f"  NTU: {r_s['ntu']:.2f}")

        print(f"\nCompressible model results for A_fr = {a_fr:.4f} m², A_q = {AQ_BASELINE} m²:")
        print(f"  Aq/Ao h,c: {r_c['Aq_over_Ao_h']:.4f}, {r_c['Aq_over_Ao_c']:.4f}")
        print(f"  Effectiveness: {r_c['eps']:.4f} ({r_c['eps'] * 100:.2f}%)")
        print("  Stagnation properties: ")
        print(f"  Δp_h: {r_c['dp_hot_stag'] * 100:.4f}% (k and ksi: {r_c['hot']['k']:.2f}, {r_c['hot']['ksi']:.4f})")
        print(f"  Δp_c: {r_c['dp_cold_stag'] * 100:.4f}% (k and ksi: {r_c['cold']['k']:.2f}, {r_c['cold']['ksi']:.4f})")
        print(f"  T hot stag out: {r_c['t_stag_hot_out']:.1f} K")
        print(f"  T cold stag out: {r_c['t_stag_cold_out']:.1f} K")
        print(f"  Static dp: hot {r_c['dp_hot'] * 100:.4f}%  cold {r_c['dp_cold'] * 100:.4f}%")
        print(f"  Static T: hot {r_c['t_hot_out']:.1f} K  cold {r_c['t_cold_out']:.1f} K")
        print(f"  Re_hot: {r_c['re_hot']:.0f}, M_in, M_out: {r_c['hot']['mach_in']:.2f}, {r_c['hot']['mach_out']:.2f}")
        print(
            f"  Re_cold: {r_c['re_cold']:.0f}, M_in, M_out: {r_c['cold']['mach_in']:.2f}, {r_c['cold']['mach_out']:.2f}"
        )
        print(f"  NTU: {r_c['ntu']:.2f}")
    else:
        print(
            f"Simple model: Δp_h={r_s['dp_hot'] * 100:.2f}%, Δp_c={r_s['dp_cold'] * 100:.2f}%, eps={r_s['eps']:.3f}; "
            f"Compressible: Δp_h={r_c['dp_hot'] * 100:.2f}%, Δp_c={r_c['dp_cold'] * 100:.2f}%, eps={r_c['eps']:.3f}, dp_h_f = {r_c['dp_hot_friction'] * 100:.2f}%, dp_h_heat = {r_c['dp_hot_heat'] * 100:.2f}%; "
            f"M_hot: {r_c['hot']['mach_in']:.3f}->{r_c['hot']['mach_out']:.3f}, ksi_hot={r_c['hot']['ksi']:.4f}; "
            f"M_cold: {r_c['cold']['mach_in']:.3f}->{r_c['cold']['mach_out']:.3f}, ksi_cold={r_c['cold']['ksi']:.4f}"
        )

    return r_s, r_c, r_xf


def main():
    """Main function to run the analysis."""
    # Print case parameters
    print_case_parameters()

    # Perform frontal area sweep
    baseline_result, all_results = frontal_area_sweep()

    # Calculate optimization curves
    curves_data = calculate_optimization_curves(baseline_result)

    # Create plot
    fig, ax = plot_optimization_curves(curves_data, save_svg=False)

    # Run simple model example
    run_one_model_example(baseline_result["a_fr"], verbose=True)

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
