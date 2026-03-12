import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.widgets import RadioButtons, Slider
from scipy.signal import find_peaks

from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.dirname(os.path.abspath(__file__))

# Default modeling assumptions
DEFAULT_C_COLD_OVER_C_HOT = 1.0  # C_cold / C_hot
DEFAULT_ST_OVER_F = 0.4  # Same for both fluids
DEFAULT_F_C_OVER_F_H = 1.0  # f_c/f_h
DEFAULT_D_R = 1.0  # d_r = sigma_r/A_r (cold/hot ratio)
DEFAULT_G2_H = 1e-5  # g2_h

# Default NTU max fixed at 15
DEFAULT_NTU_MAX = 15.0
DEFAULT_DP_MAX = 0.2

# Parameter ranges for sliders
C_COLD_OVER_C_HOT_RANGE = (0.1, 10.0)  # Will use exponential slider
ST_OVER_F_RANGE = (0.2, 0.5)
F_C_OVER_F_H_RANGE = (0.1, 10.0)  # Will use exponential slider
D_R_RANGE = (0.1, 10.0)  # Will use exponential slider
G2_H_RANGE = (1e-5, 8e-2)  # Will use exponential slider


# Pressure drop assumption options
PRESSURE_DROP_OPTIONS = ["dp_c=dp_h", "dp_c<<dp_h", "inlet_density"]
DEFAULT_PRESSURE_DROP_ASSUMPTION = "dp_c<<dp_h"  # Default to option 2

# Default values for inlet density assumption (option 3)
DEFAULT_MOLAR_MASS_RATIO = 1.0  # M_cold / M_hot (cold/hot)
DEFAULT_A_R = 1.0  # A_r (cold/hot) - sigma_r is calculated as d_r * A_r
DEFAULT_A_R_MIN = 0.5

# Default parameters for framework calculations
DEFAULT_T = 2.0  # T_hot_in / T_cold_in
DEFAULT_T_DEAD_OVER_T_COLD_IN = 1.1
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 10.0  # p_cold_in / p_hot_in
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.1  # p_hot_in / p_dead (slider value)
DEFAULT_GAMMA = 1.4
TARGET_EPS = 0.8

SHOW_CUBIC = False
NTU_MATCH = None  # Will be set in match case if SHOW_CUBIC is True


defaults = "Helicopter"
match defaults:
    case "Brewer":
        DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
        DEFAULT_C_COLD_OVER_C_HOT = 0.11  # C_cold / C_hot
        DEFAULT_ST_OVER_F = 0.4  # Same for both fluids

        DEFAULT_G2_H = 7e-2  # g2_h

        # These values are fudged to achieve the right eps and dp values (could also fudge further to get right St/f)
        DEFAULT_F_C_OVER_F_H = 10  # f_c/f_h
        DEFAULT_D_R = 0.01  # d_r = sigma_r/A_r (cold/hot ratio)
        D_R_RANGE = (0.0035, 10.0)  # Will use exponential slider
        F_C_OVER_F_H_RANGE = (0.1, 100)  # Will use exponential slider

        DEFAULT_MOLAR_MASS_RATIO = 0.07  # M_cold / M_hot (cold/hot)
        DEFAULT_A_R = 0.4  # A_r (cold/hot) - sigma_r is calculated as d_r * A_r
        DEFAULT_A_R_MIN = 0.1

        DEFAULT_T = 2.95  # 778/264 T static in ratio
        DEFAULT_T_DEAD_OVER_T_COLD_IN = 219 / 288

        DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 17.3 / 0.4  # 43.25

        DEFAULT_P_HOT_IN_OVER_P_DEAD = 0.4 / 0.24  # 0.4/0.24 abt 1.7
        TARGET_EPS = 0.8043
        NTU_MATCH = 1.747
    case "Helicopter":
        DEFAULT_PRESSURE_DROP_ASSUMPTION = "inlet_density"
        DEFAULT_C_COLD_OVER_C_HOT = 1  # 0.95  # C_cold / C_hot
        DEFAULT_D_R = 0.257  # 0.44
        # g2h = 2e-2
        DEFAULT_G2_H = 0.5 * 1.4 * 0.1**2  # 2e-2
        DEFAULT_T = 898 / 588  # 1.7  # 980/576
        DEFAULT_T_DEAD_OVER_T_COLD_IN = 288 / 588  # 0.52  # 300/576
        DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 8.82 / 1.04  # 7.2
        DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.04  # 1.03
        TARGET_EPS = 0.65  # 0.6
        NTU_MATCH = 1.824
        DEFAULT_NTU_MAX = 5.0
        SHOW_CUBIC = True
    case "g2lim":
        DEFAULT_PRESSURE_DROP_ASSUMPTION = "dp_c<<dp_h"
        DEFAULT_C_COLD_OVER_C_HOT = 1.0
        DEFAULT_ST_OVER_F = 0.4
        DEFAULT_F_C_OVER_F_H = 1.0
        DEFAULT_D_R = 1.0
        DEFAULT_G2_H = 1e-2
        DEFAULT_T = 2.0
        DEFAULT_T_DEAD_OVER_T_COLD_IN = 1.1
        DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 10.0
        DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.1

# NTU sweep range
NTU_SWEEP = np.linspace(0.1, DEFAULT_NTU_MAX, 200)


class ExpSlider(Slider):
    """Custom slider class that handles exponential steps.
    Input bounds should be the actual desired values (must be positive).
    The slider will internally work in log space to create exponential steps."""

    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valstep=None, valfmt=None, **kwargs):
        # Validate inputs
        if valmin <= 0 or valmax <= 0 or valinit <= 0:
            raise ValueError("All values must be positive for exponential slider")

        # Store original bounds
        self.valmin = valmin
        self.valmax = valmax

        # Convert to log space for internal use
        self.log_valmin = np.log(valmin)
        self.log_valmax = np.log(valmax)
        self.log_valinit = np.log(valinit)

        # Convert valstep to log space if provided
        if valstep is not None:
            self.log_valstep = np.log(1 + valstep)  # This creates multiplicative steps
        else:
            self.log_valstep = None

        # Create the slider in log space
        super().__init__(
            ax,
            label,
            self.log_valmin,
            self.log_valmax,
            valinit=self.log_valinit,
            valstep=self.log_valstep,
            valfmt=valfmt,
            **kwargs,
        )

    def _format(self, val):
        """Override the format method to display the exponential value in scientific notation"""
        if self.valfmt:
            return self.valfmt % np.exp(val)
        return f"{np.exp(val):.1e}"

    @property
    def val(self):
        """Override val property to return exponential value"""
        return np.exp(self._val)

    @val.setter
    def val(self, val):
        self._val = val  # val is already in log space from the slider


def calculate_capacity_ratios(c_cold_over_c_hot):
    """
    Calculate C_min/C_hot and C_min/C_cold based on C_cold/C_hot ratio.

    Returns:
        C_min_over_C_hot: C_min/C_hot
        C_min_over_C_cold: C_min/C_cold
        C_hot_over_C_cold: C_hot/C_cold
    """
    C_hot_over_C_cold = 1.0 / c_cold_over_c_hot

    if c_cold_over_c_hot <= 1.0:
        # C_cold <= C_hot, so C_min = C_cold
        C_min_over_C_hot = c_cold_over_c_hot  # C_cold/C_hot
        C_min_over_C_cold = 1.0  # C_cold/C_cold
    else:
        # C_cold > C_hot, so C_min = C_hot
        C_min_over_C_hot = 1.0  # C_hot/C_hot
        C_min_over_C_cold = C_hot_over_C_cold  # C_hot/C_cold

    return C_min_over_C_hot, C_min_over_C_cold, C_hot_over_C_cold


def calculate_temperature_ratio(eps, t, hot_fluid=True):
    """
    Calculate the outlet/inlet temperature ratio for a given effectiveness and temperature ratio.

    Parameters:
    -----------
    eps : float
        Heat exchanger effectiveness
    t : float
        Temperature ratio T_h_in/T_c_in
    hot_fluid : bool
        Whether this is for the hot fluid (True) or cold fluid (False)

    Returns:
    --------
    float
        Temperature ratio T_out/T_in
    """
    if hot_fluid:
        return 1 - eps * (1 - 1 / t)
    else:
        return 1 + eps * (t - 1)


def classical_unavailable_creation_hex(
    epsilon, t, dp_over_p_in_hot, dp_over_p_in_cold, validity_mask, t_dead_over_t_cold_in=1.0, gamma=1.4
):
    """
    Calculate entropy generation normalized by mass flow rate and specific heat capacity.

    This function implements the classical thermodynamic framework for evaluating heat exchanger
    performance based on entropy generation. It assumes ideal gas behavior, equal mass flow rates
    and specific heat capacities for both fluids, and no change in kinetic energy.

    Parameters:
    -----------
    epsilon : array_like
        Heat exchanger effectiveness values
    t : float
        Temperature ratio T_hot_in / T_cold_in
    dp_over_p_in_hot : array_like
        Hot side pressure drop as fraction of inlet pressure
    dp_over_p_in_cold : array_like
        Cold side pressure drop as fraction of inlet pressure
    validity_mask : array_like, bool
        Boolean mask indicating valid pressure drop points (dp < dp_max)
    t_dead_over_t_cold_in : float, optional
        Dead state temperature normalized by cold inlet temperature. Default is 1.0.
        This is a significant assumption in the classical framework.
    gamma : float, optional
        Specific heat ratio (cp/cv). Default is 1.4 for air.

    Returns:
    --------
    array_like
        Classical unavailable energy creation normalized by Q_max.
        This represents the entropy generation times dead state temperature,
        normalized by the maximum possible heat transfer rate.
    """
    gm1og = (gamma - 1) / gamma

    # Calculate temperature ratios
    t_hot_out_over_t_hot_in = calculate_temperature_ratio(epsilon, t, hot_fluid=True)
    t_cold_out_over_t_cold_in = calculate_temperature_ratio(epsilon, t, hot_fluid=False)

    # Calculate pressure ratios
    p_hot_out_over_p_hot_in = 1 - dp_over_p_in_hot
    p_cold_out_over_p_cold_in = 1 - dp_over_p_in_cold

    # Entropy generation per unit mass flow and specific heat
    # For ideal gas: s_gen/(mdot*cp) = ln(T_out/T_in) - (gamma-1)/gamma * ln(p_out/p_in)
    s_gen_over_mdot_cp = (
        np.log(t_hot_out_over_t_hot_in[validity_mask])
        + np.log(t_cold_out_over_t_cold_in[validity_mask])
        - gm1og * (np.log(p_hot_out_over_p_hot_in[validity_mask]) + np.log(p_cold_out_over_p_cold_in[validity_mask]))
    )

    # Classical unavailable energy creation normalized by Q_max
    # Q_max = C_min * (T_hot_in - T_cold_in) = mdot*cp * (T_hot_in - T_cold_in) for equal capacities
    # Normalization factor: 1/(t-1) = T_cold_in / (T_hot_in - T_cold_in)
    classical_unavailable_creation = s_gen_over_mdot_cp * t_dead_over_t_cold_in / (t - 1)

    return classical_unavailable_creation


def practical_unavailable_creation_hex(
    epsilon,
    t,
    dp_over_p_in_hot,
    dp_over_p_in_cold,
    validity_mask,
    p_cold_in_over_p_hot_in=1.0,
    p_dead_over_p_hot_in=1.0,
    gamma=1.4,
):
    """
    Calculate work potential change normalized by maximum heat transfer rate.

    This function implements the practical framework for evaluating heat exchanger performance
    based on work potential relative to a dead state. It assumes perfect gas behavior (constant cp)
    and that performance depends only on inlet and outlet temperatures and pressures. No change in
    kinetic energy is assumed when calculating static temperatures from effectiveness.

    Parameters:
    -----------
    epsilon : array_like
        Heat exchanger effectiveness values
    t : float
        Temperature ratio T_hot_in / T_cold_in
    dp_over_p_in_hot : array_like
        Hot side pressure drop as fraction of inlet pressure
    dp_over_p_in_cold : array_like
        Cold side pressure drop as fraction of inlet pressure
    validity_mask : array_like, bool
        Boolean mask indicating valid pressure drop points (dp < dp_max)
    p_cold_in_over_p_hot_in : float, optional
        Cold inlet pressure normalized by hot inlet pressure. Default is 1.0.
    p_dead_over_p_hot_in : float, optional
        Dead state pressure normalized by hot inlet pressure. Default is 1.0.
    gamma : float, optional
        Specific heat ratio (cp/cv). Default is 1.4 for air.

    Returns:
    --------
    array_like
        Practical work potential change normalized by Q_max.
        Negative values indicate work potential destruction (irreversibility).
        Positive values indicate work potential creation (as in recuperation).
    """
    gm1og = (gamma - 1) / gamma

    # Calculate temperature ratios
    t_hot_out_over_t_hot_in = calculate_temperature_ratio(epsilon, t, hot_fluid=True)
    t_cold_out_over_t_cold_in = calculate_temperature_ratio(epsilon, t, hot_fluid=False)

    # Calculate pressure ratios
    p_hot_out_over_p_hot_in = 1 - dp_over_p_in_hot
    p_cold_out_over_p_cold_in = 1 - dp_over_p_in_cold

    # Apply validity mask BEFORE power operations to avoid invalid values
    # Only calculate for valid pressure drop points
    p_hot_out_over_p_hot_in_valid = p_hot_out_over_p_hot_in[validity_mask]
    p_cold_out_over_p_cold_in_valid = p_cold_out_over_p_cold_in[validity_mask]
    t_hot_out_over_t_hot_in_valid = t_hot_out_over_t_hot_in[validity_mask]
    t_cold_out_over_t_cold_in_valid = t_cold_out_over_t_cold_in[validity_mask]

    # Pressure ratio for cold side relative to dead state
    p_cold_in_over_p_dead = p_cold_in_over_p_hot_in / p_dead_over_p_hot_in

    # Hot side work potential contribution (only for valid points)
    # Dimensionalized by mdot*cp*T_dead, normalized by Q_max
    # Q_max = C_min * (T_hot_in - T_cold_in) = mdot*cp * (T_hot_in - T_cold_in) for equal capacities
    # Normalization: 1/(1-1/t) = T_hot_in / (T_hot_in - T_cold_in) for hot side
    work_pot_hot = (
        (p_dead_over_p_hot_in) ** gm1og
        * 1
        / (1 - 1 / t)
        * (t_hot_out_over_t_hot_in_valid * (1 / p_hot_out_over_p_hot_in_valid) ** gm1og - 1)
    )

    # Cold side work potential contribution (only for valid points)
    # Normalization: 1/(t-1) = T_cold_in / (T_hot_in - T_cold_in) for cold side
    work_pot_cold = (
        (1 / p_cold_in_over_p_dead) ** gm1og
        * 1
        / (t - 1)
        * (t_cold_out_over_t_cold_in_valid * (1 / p_cold_out_over_p_cold_in_valid) ** gm1og - 1)
    )

    # Total work potential change (already masked)
    practical_unavailable_creation = work_pot_hot + work_pot_cold

    return practical_unavailable_creation


def calculate_epsilon_ntu_curve(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    ntu_array=NTU_SWEEP,
    ntu_max=DEFAULT_NTU_MAX,
    dp_max=0.2,
    pressure_drop_percent_ratio_cold_over_hot=0.0,
):
    """
    Calculate epsilon-NTU curve and pressure drop for given parameters.

    Parameters:
        c_cold_over_c_hot: C_cold / C_hot ratio
        st_over_f: St/f ratio (same for both fluids)
        f_c_over_f_h: f_c / f_h ratio
        d_r: d_r = sigma_r/A_r (cold/hot ratio) - note: d_r = sigma_r/A_r where it's always cold/hot in the ratios
        g2_h: g2_h parameter
        ntu_array: Array of NTU values to evaluate
        ntu_max: Maximum NTU to plot (None for no limit)
        dp_max: Maximum pressure drop fraction (default 0.2 = 20%)
        pressure_drop_percent_ratio_cold_over_hot: Ratio of (dp_cold/p_cold_in) / (dp_hot/p_hot_in).
            Options:
            - 1.0: dp_c/pcin = dp_h/phin (equal pressure drop percentages)
            - 0.0: dp_c/pcin << dp_h/phin (negligible cold pressure drop)
            - Calculated value: Based on inlet density assumption (requires additional parameters)

    Returns:
        ntu: NTU values
        epsilon: Effectiveness values
        dp_over_p_in_hot: Hot side pressure drop (as fraction)
        dp_over_p_in_cold: Cold side pressure drop (as fraction)
        validity_mask: Boolean mask for valid points (dp < dp_max and ntu <= ntu_max if specified)
    """
    # Generate NTU array up to ntu_max if specified, otherwise use full range
    if ntu_max is not None:
        ntu_array = np.linspace(0.1, ntu_max, 200)
    else:
        ntu_array = NTU_SWEEP

    # Calculate capacity ratios
    C_min_over_C_hot, C_min_over_C_cold, C_hot_over_C_cold = calculate_capacity_ratios(c_cold_over_c_hot)

    # C_ratio for epsilon-NTU calculation (C_min/C_max)
    if c_cold_over_c_hot <= 1.0:
        C_ratio = c_cold_over_c_hot  # C_min/C_max = C_cold/C_hot
    else:
        C_ratio = 1.0 / c_cold_over_c_hot  # C_min/C_max = C_hot/C_cold

    # Calculate effectiveness using epsilon-NTU method for counterflow
    epsilon = epsilon_ntu(ntu_array, C_ratio, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)

    # Calculate hot side pressure drop
    # dp_over_p_in_hot = g2_h * NTU * (1/St_over_f_h * Cmin/C_h + 1/f_c_over_f_h * 1/St_over_f_c * d_r* C_min/C_c)
    # Note: St_over_f_h = St_over_f_c = st_over_f (same for both fluids)
    # d_r = sigma_r/A_r where it's always cold/hot in the ratios

    st_over_f_h = st_over_f
    st_over_f_c = st_over_f

    # Pressure drop coefficient (constant part)
    dp_coeff_normal = g2_h * (
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
    )
    # Pressure drop varies linearly with NTU
    dp_over_p_in_hot_array = dp_coeff_normal * ntu_array

    # Now we are doing for the g2_h that contains the heat transfer area that is fixed rather than free flow area
    if SHOW_CUBIC and NTU_MATCH is not None:
        closest_idx = np.argmin(np.abs(ntu_array - NTU_MATCH))

        dp_over_p_in_hot_array = dp_over_p_in_hot_array[closest_idx] * np.power(
            (ntu_array / ntu_array[closest_idx]), 4.407
        )

    # Calculate cold side pressure drop based on pressure_drop_percent_ratio_cold_over_hot
    # dp_cold_over_p_cold_in = pressure_drop_percent_ratio_cold_over_hot * dp_hot_over_p_hot_in
    # Note: This is the ratio of (dp_cold/p_cold_in) / (dp_hot/p_hot_in)
    dp_over_p_in_cold_array = pressure_drop_percent_ratio_cold_over_hot * dp_over_p_in_hot_array

    # Create validity mask: stop when dp/p_in >= dp_max for either side
    validity_mask = (dp_over_p_in_hot_array < dp_max) & (dp_over_p_in_cold_array < dp_max)

    return ntu_array, epsilon, dp_over_p_in_hot_array, dp_over_p_in_cold_array, validity_mask


def create_plot(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    ntu_max=None,
    dp_max=0.2,
    ax=None,
    ax_twin=None,
    plot_triple_g2=None,
    framework="conventional",
    t=2.0,
    t_dead_over_t_cold_in=1.0,
    p_cold_in_over_p_hot_in=1.0,
    p_dead_over_p_hot_in=1.0,
    gamma=1.4,
    pressure_drop_percent_ratio_cold_over_hot=0.0,
):
    """
    Create or update the plot with given parameters.

    This function plots directly on the provided axes (ax and ax_twin). If ax is None,
    it creates a new figure and axes. The function clears existing plots on the axes
    before plotting new data.

    Parameters:
        c_cold_over_c_hot: C_cold / C_hot ratio
        st_over_f: St/f ratio (same for both fluids)
        f_c_over_f_h: f_c / f_h ratio
        d_r: d_r = sigma_r/A_r (cold/hot ratio)
        g2_h: g2_h parameter
        ntu_max: Maximum NTU to plot (None for default)
        dp_max: Maximum pressure drop fraction (default 0.2 = 20%)
        ax: Matplotlib axes object to plot on (creates new figure if None)
        ax_twin: Matplotlib twin axes object for right y-axis (created if None and ax provided)
        plot_triple_g2: If False/None, use single g2_h value. If a list/array, use those
                        g^2 values for plotting multiple lines. Note: True is treated as a
                        truthy value and will cause an error - pass a list/array instead.
        framework: Framework to use for right axis. Options: "conventional" (pressure drop),
                   "classical" availability (exergy), "practical" availability (euergy).
        t: Temperature ratio T_hot_in / T_cold_in. Default 2.0.
        t_dead_over_t_cold_in: Dead state temperature normalized by cold inlet temperature. Default 1.0.
        p_cold_in_over_p_hot_in: Cold inlet pressure normalized by hot inlet pressure. Default 1.0.
        p_dead_over_p_hot_in: Dead state pressure normalized by hot inlet pressure. Default 1.0.
        gamma: Specific heat ratio. Default 1.4.
        pressure_drop_percent_ratio_cold_over_hot: Ratio of (dp_cold/p_cold_in) / (dp_hot/p_hot_in).
            Options:
            - 1.0: dp_c/pcin = dp_h/phin (equal pressure drop percentages)
            - 0.0: dp_c/pcin << dp_h/phin (negligible cold pressure drop)
            - Calculated value: Based on inlet density assumption (requires additional parameters)

    Returns:
        For "conventional" framework:
            - If multiple g2 values: (line_eps, line_dp_list, ax, ax_twin)
            - If single g2 value: (line_eps, line_dp, ax, ax_twin)
        For "classical" or "practical" frameworks:
            - (line_dp_list[0], line_dp_list, ax, ax_twin)
        where:
            - line_eps: Line object for epsilon curve (only for conventional framework)
            - line_dp or line_dp_list: Line object(s) for right axis curve(s)
            - ax: The axes object used for plotting
            - ax_twin: The twin axes object used for right y-axis (may be hidden for classical/practical)
    """
    if plot_triple_g2 is None or plot_triple_g2 is False:
        # Use single g^2 value
        g2_values = [g2_h]
    else:
        # plot_triple_g2 is a list/array of g^2 values
        g2_values = list(plot_triple_g2)

    # Calculate epsilon and pressure drops (same for all g^2 values)
    ntu, epsilon, dp_over_p_in_hot, dp_over_p_in_cold, validity_mask = calculate_epsilon_ntu_curve(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_values[0],
        ntu_max=ntu_max,
        dp_max=dp_max,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_percent_ratio_cold_over_hot,
    )

    if ax is None:
        _ = plt.figure(figsize=(9 / 2.54, 7 / 2.54))  # fig (unused; change back if needed)
        ax = plt.subplot(111)
        ax_twin = ax.twinx()

    # Clear existing lines
    ax.clear()
    ax_twin.clear()

    # Check if we're plotting multiple g2 values
    is_multiple_g2 = len(g2_values) > 1

    # Calculate right axis metric based on framework
    line_dp_list = []
    dark_blue = "b"  # Dark blue color from Fig 3

    if framework == "conventional":
        # Make sure twin axis is visible for conventional framework
        ax_twin.set_visible(True)
        # Plot epsilon on left axis (only valid points)
        eps_label = r"$\varepsilon$"
        if is_multiple_g2:
            left_axis_color = "r"
        else:
            left_axis_color = "k"
        line_eps = ax.plot(
            ntu[validity_mask], epsilon[validity_mask], "-", label=eps_label, color=left_axis_color, zorder=3
        )[0]
        ax.set_ylim(0, 1)
        ax.set_xlabel("NTU [-]")
        ax.set_ylabel(r"$\varepsilon$ [%]" + " (all)" if is_multiple_g2 else "")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        # Set NTU max from parameter
        ax.set_xlim(0, ntu_max if ntu_max is not None else 15)

        # Find point closest to TARGET_EPS and add title with values

        # Use only valid points for finding closest
        eps_valid = epsilon[validity_mask]
        ntu_valid = ntu[validity_mask]
        dp_hot_valid = dp_over_p_in_hot[validity_mask]
        dp_cold_valid = dp_over_p_in_cold[validity_mask]
        if len(eps_valid) > 0:
            closest_idx = np.argmin(np.abs(eps_valid - TARGET_EPS))
            eps_closest = eps_valid[closest_idx]
            ntu_closest = ntu_valid[closest_idx]
            dp_hot_closest = dp_hot_valid[closest_idx]
            dp_cold_closest = dp_cold_valid[closest_idx]
            # Format title with epsilon, NTU, and dp_hot/p_hot_in
            ax.set_title(
                rf"Hot $\Delta p/p_{{in}}$ at $\varepsilon$ = {eps_closest:.4f}: "
                rf"NTU = {ntu_closest:.3f}, $\Delta p/p_{{in}}$ = {dp_hot_closest * 100:.1f}%(hot) + {dp_cold_closest * 100:.1f}%(cold) = {(dp_hot_closest + dp_cold_closest) * 100:.1f}%(total)",
                fontsize=10,
            )

        # Determine pressure drop plotting based on ratio
        plot_both_sides = pressure_drop_percent_ratio_cold_over_hot > 0.0
        use_inlet_density = (
            pressure_drop_percent_ratio_cold_over_hot > 0.0 and pressure_drop_percent_ratio_cold_over_hot < 1.0
        )

        # Case 3: Can't use with multiple g^2 lines (too messy)
        if use_inlet_density and is_multiple_g2:
            raise ValueError(
                "Cannot use inlet density assumption (option 3) with multiple g^2 values. Use single g^2 value."
            )

        # Plot pressure drop
        if is_multiple_g2:
            # Multiple g^2 values - only plot hot side (cases 1 and 2 can't have multiple g^2 with both sides)
            linestyles = ["-", "--", ":", "-", (0, (3, 1, 1, 1)), (0, (5, 5))]
            for i, g2_val in enumerate(reversed(g2_values)):
                ntu_dp, _, dp_hot_dp, dp_cold_dp, validity_mask_dp = calculate_epsilon_ntu_curve(
                    c_cold_over_c_hot,
                    st_over_f,
                    f_c_over_f_h,
                    d_r,
                    g2_val,
                    ntu_max=ntu_max,
                    dp_max=dp_max,
                    pressure_drop_percent_ratio_cold_over_hot=pressure_drop_percent_ratio_cold_over_hot,
                )
                linestyle_idx = (len(g2_values) - 1 - i) % len(linestyles)
                label = rf"$g^2$ = {g2_val:.0e}"
                line_dp = ax_twin.plot(
                    ntu_dp[validity_mask_dp],
                    dp_hot_dp[validity_mask_dp],  # Plot as fraction (0-0.2), PercentFormatter converts to %
                    color="k",
                    linestyle=linestyles[linestyle_idx],
                    label=label,
                    zorder=1,
                )[0]
                line_dp_list.append(line_dp)
            if pressure_drop_percent_ratio_cold_over_hot == 1.0:
                ylabel = r"hot & cold $\Delta p/p_{\mathrm{in}}$ (%)"
            else:
                ylabel = r"hot $\Delta p/p_{\mathrm{in}}$ (%)"
            ylim_max = dp_max  # Keep as fraction (0.2), PercentFormatter will show as 20%
            axis_color = "k"
        else:
            # Single g^2 value
            if plot_both_sides:
                # Plot both hot and cold sides
                if use_inlet_density:
                    # Case 3: Different colors for hot and cold
                    line_hot = ax_twin.plot(
                        ntu[validity_mask],
                        dp_over_p_in_hot[validity_mask],
                        "r--",
                        label=r"$\Delta p/p_{\mathrm{in}}$ $_h$",
                        zorder=1,
                    )[0]
                    line_cold = ax_twin.plot(
                        ntu[validity_mask],
                        dp_over_p_in_cold[validity_mask],
                        "b--",
                        label=r"$\Delta p/p_{\mathrm{in}}$ $_c$",
                        zorder=1,
                    )[0]
                    line_dp_list = [line_hot, line_cold]
                    ylabel = r"$\Delta p/p_{\mathrm{in}}$ (%)"
                    axis_color = "k"  # Use black for axis when showing both
                else:
                    # Case 1: Equal pressure drops, plot one line
                    line_dp = ax_twin.plot(
                        ntu[validity_mask],
                        dp_over_p_in_hot[validity_mask],  # Same as cold when ratio = 1
                        "r--",
                        label=r"$\Delta p/p_{\mathrm{in}}$",
                        zorder=1,
                    )[0]
                    line_dp_list = [line_dp]
                    ylabel = r"hot & cold $\Delta p/p_{\mathrm{in}}$ (%)"
                    axis_color = "r"
            else:
                # Case 2: Only plot hot side
                line_dp = ax_twin.plot(
                    ntu[validity_mask],
                    dp_over_p_in_hot[validity_mask],  # Plot as fraction (0-0.2), PercentFormatter converts to %
                    "r--",
                    label=r"$\Delta p/p_{\mathrm{in}}$",
                    zorder=1,
                )[0]
                line_dp_list = [line_dp]
                ylabel = r"hot $\Delta p/p_{\mathrm{in}}$ (%)"
                axis_color = "r"
            ylim_max = dp_max  # Keep as fraction (0.2), PercentFormatter will show as 20%

        ax_twin.set_ylim(0, ylim_max)
        ax_twin.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

        # Add grey vertical line at NTU_MATCH when SHOW_CUBIC is True to indicate where cubic and linear plots match
        if SHOW_CUBIC and NTU_MATCH is not None:
            ax.axvline(x=NTU_MATCH, color="grey", linestyle="--", linewidth=1, zorder=2)

    elif framework == "classical":
        # Plot classical metric for each g^2 value
        linestyles = ["-", "--", ":", "-", (0, (3, 1, 1, 1)), (0, (5, 5))]
        dark_blue = "k"  # Dark blue color from Fig 3
        line_list = []
        all_classical_metrics = []

        for i, g2_val in enumerate(reversed(g2_values)):
            # Calculate for this g^2 value
            ntu_g2, eps_g2, dp_hot_g2, dp_cold_g2, validity_mask_g2 = calculate_epsilon_ntu_curve(
                c_cold_over_c_hot,
                st_over_f,
                f_c_over_f_h,
                d_r,
                g2_val,
                ntu_max=ntu_max,
                dp_max=dp_max,
                pressure_drop_percent_ratio_cold_over_hot=pressure_drop_percent_ratio_cold_over_hot,
            )
            classical_metric_g2 = classical_unavailable_creation_hex(
                eps_g2, t, dp_hot_g2, dp_cold_g2, validity_mask_g2, t_dead_over_t_cold_in, gamma
            )
            all_classical_metrics.append(classical_metric_g2)

            # Plot with different linestyle for each g^2 (all dark blue)
            linestyle_idx = (len(g2_values) - 1 - i) % len(linestyles)
            label = rf"$g^2$ = {g2_val:.0e}"
            line = ax.plot(
                ntu_g2[validity_mask_g2],
                classical_metric_g2,
                color=dark_blue,
                linestyle=linestyles[linestyle_idx],
                label=label,
                zorder=3,
            )[0]
            line_list.append(line)

        line_eps = line_list[0] if len(line_list) > 0 else None
        ax.set_xlabel("NTU [-]")
        ax.set_ylabel("Net classical unavailable energy creation [-]")
        # Set NTU max from parameter
        ax.set_xlim(0, ntu_max if ntu_max is not None else 15)
        # Auto-scale y-axis for classical metric
        if len(all_classical_metrics) > 0:
            all_values = np.concatenate([m for m in all_classical_metrics if len(m) > 0])
            if len(all_values) > 0:
                ylim_max = np.max(all_values) * 1.1
                ax.set_ylim(0, max(ylim_max, 0.01))
            else:
                ax.set_ylim(0, 0.01)
        else:
            ax.set_ylim(0, 0.01)
        # Hide twin axis
        ax_twin.set_visible(False)
        line_dp_list = line_list
        axis_color = "k"
        ylabel = "Net classical unavailable energy creation [-]"

        # Add grey vertical line at NTU_MATCH when SHOW_CUBIC is True to indicate where cubic and linear plots match
        if SHOW_CUBIC and NTU_MATCH is not None:
            ax.axvline(x=NTU_MATCH, color="grey", linestyle="--", linewidth=1, zorder=2)

        # Add optimum point markers (minimum after first peak) for each line
        # Also find optimum for title (use first line if multiple g^2 values)
        optimum_found = False
        optimum_energy = None
        optimum_ntu = None
        for line_idx, line in enumerate(line_list):
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            # Filter out invalid/masked data
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if np.any(valid_mask):
                x_plot = x_data[valid_mask]
                y_plot = y_data[valid_mask]
                # Find peaks in the data
                peaks, _ = find_peaks(y_plot)
                if len(peaks) > 0:
                    first_peak = peaks[0]
                    # Find minimum after first peak
                    y_local_min_idx = first_peak + np.argmin(y_plot[first_peak:])
                    ax.scatter(
                        x_plot[y_local_min_idx],
                        y_plot[y_local_min_idx],
                        color="black",
                        marker="o",
                        zorder=5,
                    )
                    # Store optimum values for title (use first line)
                    if line_idx == 0:
                        optimum_found = True
                        optimum_energy = y_plot[y_local_min_idx]
                        optimum_ntu = x_plot[y_local_min_idx]
                        opt_eps = epsilon[y_local_min_idx]
                        opt_dp_hot = dp_over_p_in_hot[y_local_min_idx]
                        opt_dp_cold = dp_over_p_in_cold[y_local_min_idx]

        # Add title with optimum point values if found
        if optimum_found:
            ax.set_title(
                rf"Optimum: {optimum_energy:.3f}, "
                rf"NTU = {optimum_ntu:.3f}, "
                rf"$\varepsilon$ = {opt_eps:.3f},"
                rf"$\Delta p/p_{{in}}$ = {opt_dp_hot * 100:.1f}% (hot) + "
                rf"{opt_dp_cold * 100:.1f}% (cold) = "
                rf"{(opt_dp_hot + opt_dp_cold) * 100:.1f}% (total)",
                fontsize=10,
            )

    elif framework == "practical":
        # Plot practical metric for each g^2 value
        linestyles = ["-", "--", ":", "-", (0, (3, 1, 1, 1)), (0, (5, 5))]
        dark_blue = "k"  # black
        line_list = []
        all_metrics = []

        for i, g2_val in enumerate(reversed(g2_values)):
            # Calculate for this g^2 value
            ntu_g2, eps_g2, dp_hot_g2, dp_cold_g2, validity_mask_g2 = calculate_epsilon_ntu_curve(
                c_cold_over_c_hot,
                st_over_f,
                f_c_over_f_h,
                d_r,
                g2_val,
                ntu_max=ntu_max,
                dp_max=dp_max,
                pressure_drop_percent_ratio_cold_over_hot=pressure_drop_percent_ratio_cold_over_hot,
            )
            practical_metric_g2 = practical_unavailable_creation_hex(
                eps_g2,
                t,
                dp_hot_g2,
                dp_cold_g2,
                validity_mask_g2,
                p_cold_in_over_p_hot_in,
                p_dead_over_p_hot_in,
                gamma,
            )
            all_metrics.append(practical_metric_g2)

            # Plot with different linestyle for each g^2 (all dark blue)
            linestyle_idx = (len(g2_values) - 1 - i) % len(linestyles)
            label = rf"$g^2$ = {g2_val:.0e}"
            line = ax.plot(
                ntu_g2[validity_mask_g2],
                practical_metric_g2,
                color=dark_blue,
                linestyle=linestyles[linestyle_idx],
                label=label,
                zorder=3,
            )[0]
            line_list.append(line)

        line_eps = line_list[0] if len(line_list) > 0 else None
        ax.set_xlabel("NTU [-]")
        ax.set_ylabel("Net practical unavailable energy creation [-]")
        # Set NTU max from parameter
        ax.set_xlim(0, ntu_max if ntu_max is not None else 15)
        # Auto-scale y-axis for practical metric (only negative values, y_max = 0)
        if len(all_metrics) > 0:
            all_values = np.concatenate([m for m in all_metrics if len(m) > 0])
            if len(all_values) > 0:
                ylim_min = np.min(all_values) * 1.1
                ax.set_ylim(ylim_min, 0)
            else:
                ax.set_ylim(-0.01, 0)
        else:
            ax.set_ylim(-0.01, 0)
        # Hide twin axis
        ax_twin.set_visible(False)
        line_dp_list = line_list
        axis_color = "k"
        ylabel = "Net practical unavailable energy creation [-]"

        # Add grey vertical line at NTU_MATCH when SHOW_CUBIC is True to indicate where cubic and linear plots match
        if SHOW_CUBIC and NTU_MATCH is not None:
            ax.axvline(x=NTU_MATCH, color="grey", linestyle="--", linewidth=1, zorder=2)

        # Add optimum point markers (minimum) for each line
        # Also find optimum for title (use first line if multiple g^2 values)
        optimum_found = False
        optimum_energy = None
        optimum_ntu = None
        for line_idx, line in enumerate(line_list):
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            # Filter out invalid/masked data
            valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
            if np.any(valid_mask):
                x_plot = x_data[valid_mask]
                y_plot = y_data[valid_mask]
                # Find minimum
                arg_y_min = np.argmin(y_plot)
                ax.scatter(
                    x_plot[arg_y_min],
                    y_plot[arg_y_min],
                    color="black",
                    marker="o",
                    zorder=5,
                )
                # Store optimum values for title (use first line)
                if line_idx == 0:
                    optimum_found = True
                    optimum_energy = y_plot[arg_y_min]
                    optimum_ntu = x_plot[arg_y_min]
                    opt_eps = epsilon[arg_y_min]
                    opt_dp_hot = dp_over_p_in_hot[arg_y_min]
                    opt_dp_cold = dp_over_p_in_cold[arg_y_min]
                    opt_g2 = DEFAULT_G2_H * (optimum_ntu / NTU_MATCH) ** (3.407)

        # Add title with optimum point values if found
        if optimum_found:
            ax.set_title(
                rf"Optimum: {optimum_energy:.3f}, "
                rf"NTU = {optimum_ntu:.3f}, "
                rf"$\varepsilon$ = {opt_eps:.3f},"
                rf"$\Delta p/p_{{in}}$ = {opt_dp_hot * 100:.1f}% (hot) + "
                rf"{opt_dp_cold * 100:.1f}% (cold) = "
                rf"{(opt_dp_hot + opt_dp_cold) * 100:.1f}% (total), "
                rf"$g^2$ = {opt_g2:.2e}",
                fontsize=10,
            )

    # Set up axis labels and colors based on framework
    if framework == "conventional":
        # Set ylabel and ensure it's on the right side for twin axis
        ax_twin.set_ylabel(ylabel)
        ax_twin.yaxis.set_label_position("right")
        # Make axis labels, ticks, and tick labels same color
        ax_twin.spines["right"].set_color(axis_color)
        ax_twin.yaxis.label.set_color(axis_color)
        ax_twin.tick_params(axis="y", colors=axis_color)

        ax.spines["left"].set_color(left_axis_color)
        ax.yaxis.label.set_color(left_axis_color)
        ax.tick_params(axis="y", colors=left_axis_color)

        # Also ensure ax_twin does not overpaint the left spine:
        ax_twin.spines["left"].set_visible(False)

        # Combine legend handles and labels
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax_twin.get_legend_handles_labels()
        legend = ax.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="lower right",
            labelspacing=0.05,
            edgecolor="black",
            frameon=True,
            facecolor="white",
            framealpha=1.0,
            fancybox=True,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_edgecolor("black")
    else:
        # For classical and practical, show legend with g^2 labels
        if len(line_dp_list) > 0:
            legend = ax.legend(
                loc="upper right",
                labelspacing=0.05,
                edgecolor="black",
                frameon=True,
                facecolor="white",
                framealpha=1.0,
                fancybox=True,
            )
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(1.0)
            legend.get_frame().set_edgecolor("black")

    if framework == "conventional":
        if is_multiple_g2:
            return line_eps, line_dp_list, ax, ax_twin
        else:
            return line_eps, line_dp_list[0], ax, ax_twin
    else:
        # For classical and practical, return list of lines
        if len(line_dp_list) > 0:
            return line_dp_list[0], line_dp_list, ax, ax_twin
        else:
            return None, None, ax, ax_twin


def plot_unavailable_energy_breakdown(
    c_cold_over_c_hot,
    st_over_f,
    f_c_over_f_h,
    d_r,
    g2_h,
    ntu_max=None,
    dp_max=0.2,
    ax=None,
    framework="classical",
    t=2.0,
    t_dead_over_t_cold_in=1.0,
    p_cold_in_over_p_hot_in=1.0,
    p_dead_over_p_hot_in=1.0,
    gamma=1.4,
    pressure_drop_percent_ratio_cold_over_hot=0.0,
):
    """
    Plot unavailable energy creation breakdown: with and without pressure drop.

    This function plots two lines:
    1. Unavailable energy creation assuming no pressure drop (p_out/p_in = 1 for both streams)
    2. Unavailable energy creation with actual pressure drop

    This function plots directly on the provided axes. If ax is None, it creates a new figure
    and axes. The function clears existing plots on the axes before plotting new data.

    Parameters:
        c_cold_over_c_hot: C_cold / C_hot ratio
        st_over_f: St/f ratio (same for both fluids)
        f_c_over_f_h: f_c / f_h ratio
        d_r: d_r = sigma_r/A_r (cold/hot ratio)
        g2_h: g2_h parameter
        ntu_max: Maximum NTU to plot (None for default)
        dp_max: Maximum pressure drop fraction (default 0.2 = 20%)
        ax: Matplotlib axes object to plot on (creates new figure if None)
        framework: Framework to use. Options: "classical" or "practical". Default "classical".
        t: Temperature ratio T_hot_in / T_cold_in. Default 2.0.
        t_dead_over_t_cold_in: Dead state temperature normalized by cold inlet temperature. Default 1.0.
        p_cold_in_over_p_hot_in: Cold inlet pressure normalized by hot inlet pressure. Default 1.0.
        p_dead_over_p_hot_in: Dead state pressure normalized by hot inlet pressure. Default 1.0.
        gamma: Specific heat ratio. Default 1.4.
        pressure_drop_percent_ratio_cold_over_hot: Ratio of (dp_cold/p_cold_in) / (dp_hot/p_hot_in).

    Returns:
        (line_no_dp, line_with_dp, ax): Tuple containing line objects and axes
    """
    # Calculate epsilon-NTU curve with actual pressure drop
    ntu, epsilon, dp_over_p_in_hot, dp_over_p_in_cold, validity_mask = calculate_epsilon_ntu_curve(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=ntu_max,
        dp_max=dp_max,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_percent_ratio_cold_over_hot,
    )

    # Create axes if not provided
    if ax is None:
        plt.figure(figsize=(9 / 2.54, 7 / 2.54))
        ax = plt.subplot(111)

    # Clear existing plots
    ax.clear()

    # Calculate unavailable energy creation with no pressure drop
    # Set dp_over_p_in = 0 for both streams (p_out/p_in = 1)
    dp_over_p_in_hot_no_dp = np.zeros_like(dp_over_p_in_hot)
    dp_over_p_in_cold_no_dp = np.zeros_like(dp_over_p_in_cold)
    # All points are valid when there's no pressure drop
    validity_mask_no_dp = np.ones_like(validity_mask, dtype=bool)

    if framework == "classical":
        unavailable_no_dp = classical_unavailable_creation_hex(
            epsilon,
            t,
            dp_over_p_in_hot_no_dp,
            dp_over_p_in_cold_no_dp,
            validity_mask_no_dp,
            t_dead_over_t_cold_in,
            gamma,
        )
        unavailable_with_dp = classical_unavailable_creation_hex(
            epsilon, t, dp_over_p_in_hot, dp_over_p_in_cold, validity_mask, t_dead_over_t_cold_in, gamma
        )
        ylabel = r"HEx $\Delta Q_0/Q_{\mathrm{max}}$"
    elif framework == "practical":
        unavailable_no_dp = practical_unavailable_creation_hex(
            epsilon,
            t,
            dp_over_p_in_hot_no_dp,
            dp_over_p_in_cold_no_dp,
            validity_mask_no_dp,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            gamma,
        )
        unavailable_with_dp = practical_unavailable_creation_hex(
            epsilon,
            t,
            dp_over_p_in_hot,
            dp_over_p_in_cold,
            validity_mask,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            gamma,
        )
        ylabel = r"HEx $\Delta Q_0^M/Q_{\mathrm{max}}$"
    else:
        raise ValueError(f"Unknown framework: {framework}. Must be 'classical' or 'practical'.")

    # Plot both lines
    # For no pressure drop, use all valid points (which is all points)
    line_no_dp = ax.plot(
        ntu[validity_mask_no_dp],
        unavailable_no_dp,
        "k--",
        label="No pressure drop",
        zorder=2,
    )[0]

    # For with pressure drop, use only points where pressure drop is valid
    line_with_dp = ax.plot(
        ntu[validity_mask],
        unavailable_with_dp,
        "k-",
        label="With pressure drop",
        zorder=3,
    )[0]

    # Set axis labels and limits
    ax.set_xlabel("NTU [-]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, ntu_max if ntu_max is not None else 15)

    # Auto-scale y-axis
    all_values = np.concatenate([unavailable_no_dp, unavailable_with_dp])
    if len(all_values) > 0:
        ylim_min = np.min(all_values) * 1.1
        ylim_max = np.max(all_values) * 1.1
        if framework == "practical":
            # For practical, typically negative values, set y_max to 0
            ax.set_ylim(ylim_min, 0)
        else:
            # For classical, typically positive values, set y_min to 0
            ax.set_ylim(0, max(ylim_max, 0.01))
    else:
        if framework == "practical":
            ax.set_ylim(-0.01, 0)
        else:
            ax.set_ylim(0, 0.01)

    # Add legend
    ax.legend(
        loc="upper right",
        labelspacing=0.05,
        edgecolor="black",
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        fancybox=True,
    )

    return line_no_dp, line_with_dp, ax


def calculate_pressure_drop_ratio(
    assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
):
    """
    Calculate pressure_drop_percent_ratio_cold_over_hot based on assumption.

    Parameters:
        assumption: One of "dp_c=dp_h", "dp_c<<dp_h", or "inlet_density"
        c_cold_over_c_hot: C_cold / C_hot ratio
        t: T_hot_in / T_cold_in ratio
        d_r: d_r = sigma_r/A_r (cold/hot ratio)
        molar_mass_ratio: M_cold / M_hot (cold/hot)
        sigma_r: sigma_r (cold/hot)
        p_cold_in_over_p_hot_in: p_cold_in / p_hot_in

        assumes equal gammas for both fluids

    Returns:
        pressure_drop_percent_ratio_cold_over_hot: Ratio of (dp_cold/p_cold_in) / (dp_hot/p_hot_in)
    """
    if assumption == "dp_c=dp_h":
        return 1.0
    elif assumption == "dp_c<<dp_h":
        return 0.0
    elif assumption == "inlet_density":
        # dp_cold = dp_hot / (p_c_in_over_p_h_in / c_cold_over_c_hot * t_h_in_over_t_c_in * d_ratio * sigma_r / molar_mass_ratio)
        # pressure_drop_percent_ratio_cold_over_hot = (dp_cold/p_cold_in) / (dp_hot/p_hot_in)
        # = c_cold_over_c_hot^2 * molar_mass_ratio / (p_c_in_over_p_h_in^2 * t * d_r * sigma_r)
        return c_cold_over_c_hot**2 * molar_mass_ratio / (p_cold_in_over_p_hot_in**2 * t * d_r * sigma_r**2)
    else:
        return 0.0  # Default to option 2


if __name__ == "__main__":
    # Boolean to control triple g^2 mode (True = three lines, False = single line)
    PLOT_TRIPLE_G2 = None

    # Framework selection: "conventional", "classical", "practical"
    FRAMEWORK = "conventional"
    FRAMEWORKS = ["conventional", "classical", "practical"]

    # Create figure with space for sliders on the right
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    ax_twin = ax.twinx()
    plt.subplots_adjust(right=0.7)  # Make room for sliders on the right

    # Set font sizes
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 10,
        }
    )

    # Store framework in a mutable container to allow modification in nested function
    framework_state = {"value": FRAMEWORK}
    pressure_drop_assumption_state = {"value": DEFAULT_PRESSURE_DROP_ASSUMPTION}

    # Calculate initial pressure drop ratio
    # Calculate sigma_r from d_r * A_r
    initial_sigma_r = DEFAULT_D_R * DEFAULT_A_R
    initial_pressure_drop_ratio = calculate_pressure_drop_ratio(
        DEFAULT_PRESSURE_DROP_ASSUMPTION,
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_T,
        DEFAULT_D_R,
        DEFAULT_MOLAR_MASS_RATIO,
        initial_sigma_r,
        DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    )

    # Initial plot
    create_plot(
        DEFAULT_C_COLD_OVER_C_HOT,
        DEFAULT_ST_OVER_F,
        DEFAULT_F_C_OVER_F_H,
        DEFAULT_D_R,
        DEFAULT_G2_H,
        ntu_max=DEFAULT_NTU_MAX,
        dp_max=DEFAULT_DP_MAX,
        ax=ax,
        ax_twin=ax_twin,
        plot_triple_g2=PLOT_TRIPLE_G2,
        framework=framework_state["value"],
        t=DEFAULT_T,
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        p_dead_over_p_hot_in=1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD,
        gamma=DEFAULT_GAMMA,
        pressure_drop_percent_ratio_cold_over_hot=initial_pressure_drop_ratio,
    )

    # Create sliders on the right side
    slider_height = 0.03
    slider_spacing = 0.04
    start_y = 0.8
    slider_left = 0.8
    slider_width = 0.15
    button_height = 0.04

    y_pos = start_y

    # Framework radio buttons at the top
    radio_framework = RadioButtons(
        plt.axes([slider_left, y_pos, slider_width, button_height * 3]),
        FRAMEWORKS,
        active=FRAMEWORKS.index(FRAMEWORK),
    )
    y_pos -= button_height * 3 + slider_spacing

    # Pressure drop assumption radio buttons
    radio_pressure_drop = RadioButtons(
        plt.axes([slider_left, y_pos, slider_width, button_height * 3]),
        PRESSURE_DROP_OPTIONS,
        active=PRESSURE_DROP_OPTIONS.index(DEFAULT_PRESSURE_DROP_ASSUMPTION),
    )
    y_pos -= button_height * 1 + slider_spacing

    # C_cold/C_hot slider (exponential)
    slider_c_cold_over_c_hot = ExpSlider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        r"$C_{c}$",
        C_COLD_OVER_C_HOT_RANGE[0],
        C_COLD_OVER_C_HOT_RANGE[1],
        valinit=DEFAULT_C_COLD_OVER_C_HOT,
        valstep=0.1,
        valfmt="%.1f" + r"$C_{h}$",
    )
    y_pos -= slider_spacing

    # St_over_f slider (linear)
    slider_st_over_f = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "St",
        ST_OVER_F_RANGE[0],
        ST_OVER_F_RANGE[1],
        valinit=DEFAULT_ST_OVER_F,
        valfmt="%.2f" + r"$f$",
    )
    y_pos -= slider_spacing

    # f_c/f_h slider (exponential)
    slider_f_c_over_f_h = ExpSlider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "f_c",
        F_C_OVER_F_H_RANGE[0],
        F_C_OVER_F_H_RANGE[1],
        valinit=DEFAULT_F_C_OVER_F_H,
        valstep=0.01,
        valfmt="%.2f" + r"$f_{h}$",
    )
    y_pos -= slider_spacing

    # d_r slider (exponential)
    slider_d_r = ExpSlider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "d_r",
        D_R_RANGE[0],
        D_R_RANGE[1],
        valinit=DEFAULT_D_R,
        valstep=0.01,
        valfmt="%.1e",
    )
    y_pos -= slider_spacing

    # g2_h slider (exponential)
    slider_g2_h = ExpSlider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "g2_h",
        G2_H_RANGE[0],
        G2_H_RANGE[1],
        valinit=DEFAULT_G2_H,
        valstep=1e-5,
        valfmt="%.1e",
    )
    y_pos -= slider_spacing

    # Conditional sliders for framework-specific parameters
    # Temperature ratio slider (for classical and practical)
    # Note: matplotlib Slider valfmt only accepts format strings, so we format via label
    slider_t_hot_over_t_cold = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        r"$T_h$",
        1.01,
        3.0,
        valinit=DEFAULT_T,
        valstep=0.01,
        valfmt="%.2f" + r"$T_{c}$",
    )
    slider_t_hot_over_t_cold.ax.set_visible(False)  # Hidden by default
    y_pos -= slider_spacing

    # Dead state temperature slider (for classical and practical)
    slider_t_dead_over_t_cold = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        r"$T_{dead}$",
        0.7,
        1.3,
        valinit=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        valstep=0.01,
        valfmt="%.2f" + r"$T_{c}$",
    )
    slider_t_dead_over_t_cold.ax.set_visible(False)  # Hidden by default
    y_pos -= slider_spacing

    # Cold/hot pressure ratio slider (for practical only)
    slider_p_cold_over_p_hot = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        r"$p_c$",
        5.0,
        50.0,
        valinit=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        valstep=0.1,
        valfmt="%.1f" + r"$p_{h}$",
    )
    slider_p_cold_over_p_hot.ax.set_visible(False)  # Hidden by default
    y_pos -= slider_spacing

    # Hot/dead pressure ratio slider (for practical only)
    # Note: Slider represents p_h_in / p_dead, but function needs p_dead / p_h_in
    # So we'll invert the value when using it
    slider_p_hot_over_p_dead = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        r"$p_h$",
        1.01,
        1.8,
        valinit=DEFAULT_P_HOT_IN_OVER_P_DEAD,
        valstep=0.01,
        valfmt="%.2f" + r"$p_{d}$",
    )
    slider_p_hot_over_p_dead.ax.set_visible(False)  # Hidden by default

    y_pos -= slider_spacing
    # Molar mass ratio slider (for inlet density assumption, option 3)
    slider_molar_mass_ratio = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        r"$M_{c}$",
        0.07,
        1.0,
        valinit=DEFAULT_MOLAR_MASS_RATIO,
        valstep=0.93,
        valfmt="%.2f" + r"$M_{h}$",
    )
    slider_molar_mass_ratio.ax.set_visible(False)  # Hidden by default
    y_pos -= slider_spacing

    # A_r slider (for inlet density assumption, option 3)
    slider_a_r = ExpSlider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        r"$A_r$",
        DEFAULT_A_R_MIN,
        2.0,
        valinit=DEFAULT_A_R,
        valstep=0.01,
        valfmt="%.2f" + r"$A_{h}$",
    )
    slider_a_r.ax.set_visible(False)  # Hidden by default
    y_pos -= slider_spacing

    def update_pressure_drop_slider_visibility():
        """Update visibility of pressure drop assumption sliders"""
        assumption = pressure_drop_assumption_state["value"]
        if assumption == "inlet_density":
            slider_molar_mass_ratio.ax.set_visible(True)
            slider_a_r.ax.set_visible(True)
        else:
            slider_molar_mass_ratio.ax.set_visible(False)
            slider_a_r.ax.set_visible(False)

    def select_pressure_drop_assumption(label):
        """Handle pressure drop assumption selection"""
        pressure_drop_assumption_state["value"] = label
        update_slider_visibility()  # This will also call update_pressure_drop_slider_visibility()
        update_plot()

    def update_plot(val=None):
        """Update the plot when any slider changes"""
        # ExpSlider.val already returns the actual value (no conversion needed)
        c_cold_over_c_hot = slider_c_cold_over_c_hot.val
        st_over_f = slider_st_over_f.val
        f_c_over_f_h = slider_f_c_over_f_h.val
        d_r = slider_d_r.val
        print(d_r)
        g2_h = slider_g2_h.val
        # NTU max from default (can be overridden by defaults case)
        ntu_max = DEFAULT_NTU_MAX

        # Get framework-specific parameters from sliders if they exist
        current_framework = framework_state["value"]
        assumption = pressure_drop_assumption_state["value"]
        t = DEFAULT_T
        t_dead_over_t_cold_in = DEFAULT_T_DEAD_OVER_T_COLD_IN
        p_cold_in_over_p_hot_in = DEFAULT_P_COLD_IN_OVER_P_HOT_IN
        p_dead_over_p_hot_in = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD  # p_dead / p_hot_in (for calculations)

        # Temperature ratio needed for classical, practical, or conventional with inlet_density
        if current_framework in ["classical", "practical"] or (
            current_framework == "conventional" and assumption == "inlet_density"
        ):
            if slider_t_hot_over_t_cold is not None and slider_t_hot_over_t_cold.ax.get_visible():
                t = slider_t_hot_over_t_cold.val
            if slider_t_dead_over_t_cold is not None and slider_t_dead_over_t_cold.ax.get_visible():
                t_dead_over_t_cold_in = slider_t_dead_over_t_cold.val

        # Pressure ratio needed for practical, or classical/conventional with inlet_density
        if current_framework == "practical" or (
            current_framework in ["classical", "conventional"] and assumption == "inlet_density"
        ):
            if slider_p_cold_over_p_hot is not None and slider_p_cold_over_p_hot.ax.get_visible():
                p_cold_in_over_p_hot_in = slider_p_cold_over_p_hot.val
            if slider_p_hot_over_p_dead is not None and slider_p_hot_over_p_dead.ax.get_visible():
                # Slider is p_h_in / p_dead, but function needs p_dead / p_h_in
                p_dead_over_p_hot_in = 1.0 / slider_p_hot_over_p_dead.val

        # Calculate pressure drop ratio based on assumption
        molar_mass_ratio = DEFAULT_MOLAR_MASS_RATIO
        a_r = DEFAULT_A_R
        if assumption == "inlet_density":
            # Get values from sliders
            if slider_molar_mass_ratio is not None:
                molar_mass_ratio = slider_molar_mass_ratio.val
            if slider_a_r is not None:
                a_r = slider_a_r.val

        # Calculate sigma_r from d_r * A_r
        sigma_r = d_r * a_r

        pressure_drop_ratio = calculate_pressure_drop_ratio(
            assumption, c_cold_over_c_hot, t, d_r, molar_mass_ratio, sigma_r, p_cold_in_over_p_hot_in
        )

        # Clear and recreate plot with current settings
        ax.clear()
        ax_twin.clear()

        create_plot(
            c_cold_over_c_hot,
            st_over_f,
            f_c_over_f_h,
            d_r,
            g2_h,
            ntu_max=ntu_max,
            dp_max=DEFAULT_DP_MAX,
            ax=ax,
            ax_twin=ax_twin,
            plot_triple_g2=PLOT_TRIPLE_G2,
            framework=current_framework,
            t=t,
            t_dead_over_t_cold_in=t_dead_over_t_cold_in,
            p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in=p_dead_over_p_hot_in,
            gamma=DEFAULT_GAMMA,
            pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
        )

        fig.canvas.draw_idle()

    def select_framework(label):
        """Handle framework selection from radio buttons"""
        framework_state["value"] = label
        update_slider_visibility()
        update_plot()

    def update_slider_visibility():
        """Show/hide sliders based on current framework"""
        current_framework = framework_state["value"]
        assumption = pressure_drop_assumption_state["value"]

        # Temperature sliders (for classical, practical, or conventional with inlet_density)
        show_temp_sliders = current_framework in ["classical", "practical"] or (
            current_framework == "conventional" and assumption == "inlet_density"
        )
        if slider_t_hot_over_t_cold is not None:
            slider_t_hot_over_t_cold.ax.set_visible(show_temp_sliders)
        if slider_t_dead_over_t_cold is not None:
            # T_dead slider only for classical/practical, not for conventional even with inlet_density
            slider_t_dead_over_t_cold.ax.set_visible(current_framework in ["classical", "practical"])

        # Pressure sliders (for practical, or classical/conventional with inlet_density)
        show_pressure_sliders = current_framework == "practical" or (
            current_framework in ["classical", "conventional"] and assumption == "inlet_density"
        )
        if slider_p_cold_over_p_hot is not None:
            slider_p_cold_over_p_hot.ax.set_visible(show_pressure_sliders)
        if slider_p_hot_over_p_dead is not None:
            # p_hot/p_dead slider only for practical, not for classical/conventional even with inlet_density
            slider_p_hot_over_p_dead.ax.set_visible(current_framework == "practical")

        # Update pressure drop assumption slider visibility
        update_pressure_drop_slider_visibility()

    # Connect sliders to update function
    slider_c_cold_over_c_hot.on_changed(update_plot)
    slider_st_over_f.on_changed(update_plot)
    slider_f_c_over_f_h.on_changed(update_plot)
    slider_d_r.on_changed(update_plot)
    slider_g2_h.on_changed(update_plot)

    # Connect conditional sliders
    slider_t_hot_over_t_cold.on_changed(update_plot)
    slider_t_dead_over_t_cold.on_changed(update_plot)
    slider_p_cold_over_p_hot.on_changed(update_plot)
    slider_p_hot_over_p_dead.on_changed(update_plot)

    # Connect pressure drop assumption sliders
    slider_a_r.on_changed(update_plot)
    slider_molar_mass_ratio.on_changed(update_plot)

    # Connect framework radio buttons
    radio_framework.on_clicked(select_framework)

    # Connect pressure drop assumption radio buttons
    radio_pressure_drop.on_clicked(select_pressure_drop_assumption)

    # Initialize slider visibility
    update_slider_visibility()

    # Initialize slider visibility
    update_slider_visibility()

    plt.show()
