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
    classical_unavailable_creation_hex,
    practical_unavailable_creation_hex,
)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    class _TqdmFallback:
        def __init__(self, iterable=None, desc=None, total=None, unit=None, **kwargs):
            self.iterable = iterable
            self.desc = desc or ""
            self.total = total
            self.unit = unit or "it"
            self.n = 0
            if iterable is not None:
                self._iter = iter(iterable)
            else:
                self._iter = None

        def __iter__(self):
            if self._iter is None:
                return self
            return self._iter

        def __next__(self):
            if self._iter is None:
                raise StopIteration
            self.n += 1
            if self.total and self.n <= self.total:
                print(f"\r{self.desc}: {self.n}/{self.total} {self.unit}", end="", flush=True)
            return next(self._iter)

        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"\r{self.desc}: {self.n}/{self.total} {self.unit}", end="", flush=True)
            if self.total and self.n >= self.total:
                print()  # New line when complete

        def close(self):
            if self.total and self.n < self.total:
                print()  # New line if not already printed

    def tqdm(iterable=None, desc=None, total=None, unit=None, **kwargs):
        return _TqdmFallback(iterable, desc, total, unit, **kwargs)


from heat_exchanger.epsilon_ntu import epsilon_ntu

save_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = Path(save_dir) / "newfig5_practical_data.parquet"

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

# Y-axis type: "ao_over_ao_ref", "dp_over_p_in", "g2h", or "ntu"
Y_AXIS_TYPE = "g2h"  # Options: "ao_over_ao_ref", "dp_over_p_in", "g2h", "ntu"
UNNORMALISED_Y_AXIS_IF_POSS = True  # If True, plot unnormalized values (multiply by reference)

# Sweep: d_over_d_ref from 0.5 to 1.2, 8 values step 0.1
D_OVER_D_REF_VALUES = np.linspace(0.5, 4, 100)  # 0.5, 0.6, ..., 1.2
NTU_MAX_AT_D_REF = 4.0  # NTU max at d/d_ref = 1; for smaller d, NTU max increases (2/d)
NTU_NUM = 100  # number of NTU points per d (sweep resolution)
CONTOUR_GRID_N = 200  # grid size for interpolation; contour smoothness is set by this, not NTU_NUM
NTU_MIN = 0.4
Y_MAX = 0.03


def _ntu_max_for_d(d_over_d_ref, dp_max, dp_ref=DP_REF):
    """NTU max for a given d/d_ref and dp_max.

    Formula: NTU_max = NTU_max_at_d_ref * (DP_max/DP_ref)^(1/4.407) * (d/d_ref)^{-1.407/4.407}
    """
    return NTU_MAX_AT_D_REF * (dp_max / dp_ref) ** (1 / 4.407) * (d_over_d_ref) ** (-1.407 / 4.407)


def _y_axis(d_over_d_ref, ntu, unnormalised=False):
    """Y-axis coordinate based on Y_AXIS_TYPE global variable.

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
            if unnormalised:
                # Ao_ref is not directly available, but we can compute from g2h_ref
                # Actually, Ao/Ao_ref is dimensionless, so unnormalized doesn't make sense
                # Return as-is (it's already a ratio)
                return y_norm
            return y_norm
        case "dp_over_p_in":
            # Formula: dp/p_in = dp_ref * (d/d_ref)^1.407 * (NTU/NTU_REF)^4.407
            y_norm = (d_over_d_ref) ** 1.407 * (ntu / NTU_REF) ** (4.407)
            if unnormalised:
                # Unnormalized: absolute dp/p_in value
                return y_norm * DP_REF
            # Normalized: dp/dp_ref (DP_REF cancels out)
            return y_norm
        case "g2h":
            ao_over_ao_ref = (ntu / NTU_REF) ** (-1.704) * (d_over_d_ref) ** (-0.704)
            g2h_abs = DEFAULT_G2_H * (ao_over_ao_ref) ** (-2)
            if unnormalised:
                return g2h_abs  # Absolute units
            return g2h_abs / DEFAULT_G2_H  # Normalized
        case "ntu":
            if unnormalised:
                return ntu  # Absolute NTU
            return ntu / NTU_REF  # Normalized NTU
        case _:
            raise ValueError(f"Unknown Y_AXIS_TYPE: {Y_AXIS_TYPE}")


def _get_y_axis_label():
    """Get the y-axis label based on Y_AXIS_TYPE and UNNORMALISED_Y_AXIS_IF_POSS."""
    match Y_AXIS_TYPE:
        case "ao_over_ao_ref":
            if UNNORMALISED_Y_AXIS_IF_POSS:
                return r"$A_o$"  # Unnormalized doesn't really make sense for ratios
            return r"$A_o/A_{o,\mathrm{ref}}$"
        case "dp_over_p_in":
            if UNNORMALISED_Y_AXIS_IF_POSS:
                return r"$\Delta p / p_{\mathrm{in}}$"  # Already dimensionless
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


def _get_reference_y_value():
    """Get the y-axis value at the reference point (d=1, NTU=NTU_REF).

    At the reference point:
    - ao_over_ao_ref = 1.0
    - dp_over_p_in = DP_REF (or actual computed value)
    - g2h = DEFAULT_G2_H (normalized: 1.0)
    - ntu = NTU_REF (normalized: 1.0)
    """
    return _y_axis(1.0, NTU_REF, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)


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

    This avoids interpolating a discretized reference curve, eliminating numerical
    artifacts (zigzags) in the optimum line.
    """
    xflow.SHOW_CUBIC = True
    xflow.NTU_MATCH = NTU_REF
    # Generate a small curve around NTU_REF to extract dp at the reference point
    # The cubic formula in xflow anchors at NTU_MATCH, so dp at NTU_MATCH is exact
    ntu_arr, _, dp_hot_arr, _, _ = calculate_epsilon_ntu_curve(
        c_cold_over_c_hot,
        st_over_f,
        f_c_over_f_h,
        d_r,
        g2_h,
        ntu_max=NTU_REF * 1.5,
        dp_max=1.0,  # Large enough to not clip at NTU_REF
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )
    dp_at_ntu_ref = float(np.interp(NTU_REF, ntu_arr, dp_hot_arr))
    return dp_at_ntu_ref


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
    dp_hot = dp_at_ntu_ref * (ntu / NTU_REF) ** 4.407 * (d_over_d_ref**1.407)
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
    """Practical unavailable creation at single (d, NTU). Pressure drop = analytical formula."""
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


def _classical_at_d_ntu(
    d_over_d_ref,
    ntu,
    dp_at_ntu_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    t_dead_over_t_cold_in,
    gamma,
    dp_max,
    thermal_only=False,
):
    """Classical unavailable creation at single (d, NTU). Pressure drop = analytical formula.

    Parameters:
    -----------
    thermal_only : bool
        If True, calculate with no pressure drop (dp=0). If False, use actual pressure drop.
    """
    eps, dp_hot, dp_cold, validity = _get_eps_dp_at_d_ntu(
        d_over_d_ref, ntu, dp_at_ntu_ref, c_cold_over_c_hot, pressure_drop_ratio, dp_max
    )

    if not validity:
        return np.nan

    if thermal_only:
        # Set pressure drop to zero for thermal-only calculation
        dp_hot = 0.0
        dp_cold = 0.0

    out = classical_unavailable_creation_hex(
        np.array([eps]),
        t,
        np.array([dp_hot]),
        np.array([dp_cold]),
        np.array([True]),
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        gamma=gamma,
    )
    return float(out[0]) if len(out) > 0 else np.nan


def _practical_at_d_ntu_thermal_only(
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
    """Practical unavailable creation at single (d, NTU) with no pressure drop (thermal only)."""
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


def _fig4_optimum_ntu(
    dp_at_ntu_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    gamma,
    dp_max,
):
    """At d_over_d_ref=1, find NTU that minimizes practical_unavailable_creation (fig4 optimum)."""
    ntu_fine = np.linspace(0.35, NTU_MAX_AT_D_REF, 500)
    vals = []
    for ntu in ntu_fine:
        z = _practical_at_d_ntu(
            1.0,
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
        vals.append(z)
    vals = np.array(vals)
    valid = np.isfinite(vals)
    if not np.any(valid):
        return None
    idx = np.nanargmin(vals)
    return float(ntu_fine[idx])


def _optimal_ntu_line(
    dp_at_ntu_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    p_cold_in_over_p_hot_in,
    p_dead_over_p_hot_in,
    t_dead_over_t_cold_in,
    gamma,
    dp_max,
    optimize_practical=True,
    ntu_min=NTU_MIN,
):
    """Find optimal NTU for each d/d_ref.

    If optimize_practical=True, minimizes practical unavailable energy and also calculates classical.
    If optimize_practical=False, minimizes classical unavailable energy and also calculates practical.

    Parameters:
    -----------
    ntu_min : float
        Minimum NTU value to use for the optimization sweep. Should match the minimum NTU used in the main sweep.

    Returns (d_values, ntu_opt_values, y_opt_values, z_practical_opt_values, z_classical_opt_values,
             z_practical_thermal_only_opt_values, z_classical_thermal_only_opt_values).
    """
    d_vals = []
    ntu_opt_vals = []
    y_opt_vals = []
    z_practical_opt_vals = []
    z_classical_opt_vals = []
    z_practical_thermal_only_opt_vals = []
    z_classical_thermal_only_opt_vals = []

    # Use a FIXED NTU grid to prevent zigzag artifacts from shifting grid points.
    # Covers the largest range needed (smallest d has highest NTU_max).
    # Invalid points (dp > dp_max) are automatically excluded by validity check.
    ntu_max_global = _ntu_max_for_d(D_OVER_D_REF_VALUES.min(), dp_max)
    ntu_fine = np.linspace(ntu_min, ntu_max_global, NTU_NUM)

    total_d_values = len(D_OVER_D_REF_VALUES)
    for d in tqdm(D_OVER_D_REF_VALUES, desc="Finding optimal NTU", total=total_d_values):
        practical_vals = []
        classical_vals = []

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
            z_classical = _classical_at_d_ntu(
                d,
                ntu,
                dp_at_ntu_ref,
                c_cold_over_c_hot,
                pressure_drop_ratio,
                t,
                t_dead_over_t_cold_in,
                gamma,
                dp_max,
            )
            practical_vals.append(z_practical)
            classical_vals.append(z_classical)

        practical_vals = np.array(practical_vals)
        classical_vals = np.array(classical_vals)

        if optimize_practical:
            valid = np.isfinite(practical_vals)
            if np.any(valid):
                idx = np.nanargmin(practical_vals[valid])
                idx_original = np.where(valid)[0][idx]
                ntu_opt = float(ntu_fine[idx_original])
                y_opt = _y_axis(d, ntu_opt, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
                z_practical_opt = float(practical_vals[idx_original])
                z_classical_opt = float(classical_vals[idx_original])
                # Calculate thermal_only values at the optimum point
                z_practical_thermal_only = _practical_at_d_ntu_thermal_only(
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
                z_classical_thermal_only = _classical_at_d_ntu(
                    d,
                    ntu_opt,
                    dp_at_ntu_ref,
                    c_cold_over_c_hot,
                    pressure_drop_ratio,
                    t,
                    t_dead_over_t_cold_in,
                    gamma,
                    dp_max,
                    thermal_only=True,
                )
        else:
            valid = np.isfinite(classical_vals)
            if np.any(valid):
                idx = np.nanargmin(classical_vals[valid])
                idx_original = np.where(valid)[0][idx]
                ntu_opt = float(ntu_fine[idx_original])
                y_opt = _y_axis(d, ntu_opt, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
                z_practical_opt = float(practical_vals[idx_original])
                z_classical_opt = float(classical_vals[idx_original])
                # Calculate thermal_only values at the optimum point
                z_practical_thermal_only = _practical_at_d_ntu_thermal_only(
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
                z_classical_thermal_only = _classical_at_d_ntu(
                    d,
                    ntu_opt,
                    dp_at_ntu_ref,
                    c_cold_over_c_hot,
                    pressure_drop_ratio,
                    t,
                    t_dead_over_t_cold_in,
                    gamma,
                    dp_max,
                    thermal_only=True,
                )

        if np.any(valid):
            d_vals.append(d)
            ntu_opt_vals.append(ntu_opt)
            y_opt_vals.append(y_opt)
            z_practical_opt_vals.append(z_practical_opt)
            z_classical_opt_vals.append(z_classical_opt)
            z_practical_thermal_only_opt_vals.append(z_practical_thermal_only)
            z_classical_thermal_only_opt_vals.append(z_classical_thermal_only)

    return (
        np.array(d_vals),
        np.array(ntu_opt_vals),
        np.array(y_opt_vals),
        np.array(z_practical_opt_vals),
        np.array(z_classical_opt_vals),
        np.array(z_practical_thermal_only_opt_vals),
        np.array(z_classical_thermal_only_opt_vals),
    )


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
        "y_axis_type": Y_AXIS_TYPE,
        "unnormalised_y_axis_if_poss": UNNORMALISED_Y_AXIS_IF_POSS,
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
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
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
        zz = df[sweep_mask]["dqom_practical_over_qmax"].values
        # Extract optimal line data
        opt_mask = df["d_opt"].notna()
        d_opt_line = df[opt_mask]["d_opt"].values
        y_opt_line = df[opt_mask]["ao_opt"].values
        z_practical_opt_line = df[opt_mask]["dqom_practical_opt"].values
        z_classical_opt_line = df[opt_mask]["dqom_classical_opt"].values
        # Load thermal_only values if they exist, otherwise create NaN arrays
        if "dqom_practical_thermal_only_opt" in df.columns:
            z_practical_thermal_only_opt_line = df[opt_mask]["dqom_practical_thermal_only_opt"].values
        else:
            z_practical_thermal_only_opt_line = np.full(len(d_opt_line), np.nan)
        if "dqom_classical_thermal_only_opt" in df.columns:
            z_classical_thermal_only_opt_line = df[opt_mask]["dqom_classical_thermal_only_opt"].values
        else:
            z_classical_thermal_only_opt_line = np.full(len(d_opt_line), np.nan)
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

        # Compute dp/p_in at NTU_REF (reference point); all other dp values are analytical
        dp_at_ntu_ref = _compute_dp_at_ntu_ref(
            c_cold_over_c_hot,
            st_over_f,
            f_c_over_f_h,
            d_r,
            g2_h,
            pressure_drop_ratio,
        )
        print(f"  dp/p_in at NTU_REF (from xflow): {dp_at_ntu_ref:.6f} ({dp_at_ntu_ref * 100:.2f}%)")

        # Fig4 optimum: at d=1, NTU that minimizes practical
        ntu_opt = _fig4_optimum_ntu(
            dp_at_ntu_ref,
            c_cold_over_c_hot,
            pressure_drop_ratio,
            t,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            gamma,
            dp_max,
        )
        # Compute all derived values at the optimum (d=1)
        if ntu_opt is not None:
            # Compute dp analytically at the optimum
            dp_hot_at_opt = dp_at_ntu_ref * (ntu_opt / NTU_REF) ** 4.407
            dp_hot_at_ref = dp_at_ntu_ref  # By definition

            # Compute all y-axis quantities at optimum
            ao_over_ao_ref_opt = (ntu_opt / NTU_REF) ** (-1.704) * (1.0) ** (-0.704)
            g2h_opt = DEFAULT_G2_H * (ao_over_ao_ref_opt) ** (-2)
            dp_over_p_in_opt = DP_REF * (1.0) ** 1.407 * (ntu_opt / NTU_REF) ** (4.407)
            ntu_norm_opt = ntu_opt / NTU_REF

            print("\n  === Optimum at d/d_ref = 1.0 ===")
            print(f"  NTU: {ntu_opt:.4f} (normalized: {ntu_norm_opt:.4f})")
            print(f"  dp/p_in hot: {dp_hot_at_opt:.6f} ({dp_hot_at_opt * 100:.2f}%)")
            print(f"  dp_opt / dp_ref ratio: {dp_hot_at_opt / dp_hot_at_ref:.4f}")
            print("\n  === All derived values at optimum ===")
            print(f"  Ao/Ao_ref: {ao_over_ao_ref_opt:.6f}")
            print(f"  g2h: {g2h_opt:.6f}")
            print(f"  dp/p_in (from formula): {dp_over_p_in_opt:.6f} ({dp_over_p_in_opt * 100:.2f}%)")
            print(f"  _y_axis(1, ntu_opt) = {_y_axis(1.0, ntu_opt, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS):.6f}")
            print("\n  === Reference point (d=1, NTU=NTU_REF) ===")
            print(f"  NTU: {NTU_REF:.4f}")
            print(f"  dp/p_in hot: {dp_hot_at_ref:.6f} ({dp_hot_at_ref * 100:.2f}%)")
            print("  Ao/Ao_ref: 1.0")
            print(f"  g2h: {DEFAULT_G2_H:.6f}")
            print(f"  _y_axis(1, NTU_REF) = {_get_reference_y_value():.6f}")

            y_opt = _y_axis(1.0, ntu_opt, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
        else:
            print("\n  === No optimum found at d/d_ref = 1.0 ===")
            y_opt = None

        # Optimal NTU line: for each d, find optimal NTU (minimizing practical, also calculate classical)
        (
            d_opt_line,
            ntu_opt_line,
            y_opt_line,
            z_practical_opt_line,
            z_classical_opt_line,
            z_practical_thermal_only_opt_line,
            z_classical_thermal_only_opt_line,
        ) = _optimal_ntu_line(
            dp_at_ntu_ref,
            c_cold_over_c_hot,
            pressure_drop_ratio,
            t,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            t_dead_over_t_cold_in,
            gamma,
            dp_max,
            optimize_practical=True,
            ntu_min=NTU_MIN,
        )

        # 2D sweep: for each d, sweep NTU (NTU max increases for smaller d)
        # For smaller d/d_ref, pressure drop is lower (scaled by d**1.407), so we can go to higher NTU
        xx, yy, zz = [], [], []
        total_d_values = len(D_OVER_D_REF_VALUES)
        total_points = total_d_values * NTU_NUM
        pbar = tqdm(total=total_points, desc="Computing sweep", unit="points")
        for d in D_OVER_D_REF_VALUES:
            ntu_max_d = _ntu_max_for_d(d, dp_max)  # NTU max depends on d and dp_max
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
                pbar.update(1)
        pbar.close()

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
                "dqom_practical_over_qmax": zz,
                "dqom_classical_over_qmax": [np.nan] * len(xx),  # Not calculated for sweep
                "d_opt": [np.nan] * len(xx),
                "ao_opt": [np.nan] * len(xx),
                "dqom_practical_opt": [np.nan] * len(xx),
                "dqom_classical_opt": [np.nan] * len(xx),
                "dqom_practical_thermal_only_opt": [np.nan] * len(xx),
                "dqom_classical_thermal_only_opt": [np.nan] * len(xx),
                "is_fig4_opt": [False] * len(xx),
            }
        )

        opt_df = pd.DataFrame(
            {
                "input_hash": [input_hash] * len(d_opt_line),
                "d_over_d_ref": [np.nan] * len(d_opt_line),
                "ao_over_ao_ref": [np.nan] * len(d_opt_line),
                "dqom_practical_over_qmax": [np.nan] * len(d_opt_line),
                "dqom_classical_over_qmax": [np.nan] * len(d_opt_line),
                "d_opt": d_opt_line,
                "ao_opt": y_opt_line,
                "dqom_practical_opt": z_practical_opt_line,
                "dqom_classical_opt": z_classical_opt_line,
                "dqom_practical_thermal_only_opt": z_practical_thermal_only_opt_line,
                "dqom_classical_thermal_only_opt": z_classical_thermal_only_opt_line,
                "is_fig4_opt": [False] * len(d_opt_line),
            }
        )

        dfs = [sweep_df, opt_df]

        # Add fig4 optimum point if available
        if y_opt is not None:
            # Calculate dQo^M/Qmax (practical) and dQ0/Qmax (classical) for fig4 optimum
            z_fig4_practical = _practical_at_d_ntu(
                1.0,
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
            z_fig4_classical = _classical_at_d_ntu(
                1.0,
                ntu_opt,
                dp_at_ntu_ref,
                c_cold_over_c_hot,
                pressure_drop_ratio,
                t,
                t_dead_over_t_cold_in,
                gamma,
                dp_max,
            )
            z_fig4_practical_thermal_only = _practical_at_d_ntu_thermal_only(
                1.0,
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
            z_fig4_classical_thermal_only = _classical_at_d_ntu(
                1.0,
                ntu_opt,
                dp_at_ntu_ref,
                c_cold_over_c_hot,
                pressure_drop_ratio,
                t,
                t_dead_over_t_cold_in,
                gamma,
                dp_max,
                thermal_only=True,
            )
            fig4_df = pd.DataFrame(
                {
                    "input_hash": [input_hash],
                    "d_over_d_ref": [1.0],
                    "ao_over_ao_ref": [y_opt],
                    "dqom_practical_over_qmax": [z_fig4_practical],
                    "dqom_classical_over_qmax": [z_fig4_classical],
                    "d_opt": [np.nan],
                    "ao_opt": [np.nan],
                    "dqom_practical_opt": [np.nan],
                    "dqom_classical_opt": [np.nan],
                    "dqom_practical_thermal_only_opt": [z_fig4_practical_thermal_only],
                    "dqom_classical_thermal_only_opt": [z_fig4_classical_thermal_only],
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
    print("Interpolating onto regular grid...")
    valid = np.isfinite(zz)
    x_min, x_max = D_OVER_D_REF_VALUES.min(), D_OVER_D_REF_VALUES.max()
    y_min, y_max = yy[valid].min(), yy[valid].max()
    # Slightly extend for nicer contours
    y_min = max(y_min * 0.95, 1e-5)
    y_max = min(y_max * 1.05, Y_MAX)
    grid_x = np.linspace(x_min, x_max, CONTOUR_GRID_N)
    grid_y = np.linspace(y_min, y_max, CONTOUR_GRID_N)
    X, Y = np.meshgrid(grid_x, grid_y)
    # griddata doesn't support progress bars directly, but we can show a message
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
    ax.set_ylabel(_get_y_axis_label())

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
                label="optimal NTU (practical)",
            )

    # Ref at (1, y_ref): grey cross (y_ref depends on Y_AXIS_TYPE)
    y_ref = _get_reference_y_value()
    ax.scatter([1.0], [y_ref], marker="x", s=80, color="grey", linewidths=2, zorder=5, label="ref")
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
    # ax.legend(loc="upper left", fontsize=9)
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
        t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
        gamma=DEFAULT_GAMMA,
        pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
        molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
        a_r=DEFAULT_A_R,
        dp_max=DEFAULT_DP_MAX,
        base_name="newfig5_practical",
    )
