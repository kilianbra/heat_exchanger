"""
2D contour plot of practical availability vs d_over_d_ref and scaled NTU.
Uses same inputs as newfig4 (Helicopter defaults). Pressure drop includes
multiplier d_over_d_ref**1.407. Y-axis: configurable via Y_AXIS_TYPE.
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
DATA_FILE = Path(save_dir) / "fig6_data.parquet"

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
DP_REF = 0.106

# Y-axis type: "ao_over_ao_ref", "dp_over_p_in", "g2h", or "ntu"
Y_AXIS_TYPE = "g2h"  # Options: "ao_over_ao_ref", "dp_over_p_in", "g2h", "ntu"
UNNORMALISED_Y_AXIS_IF_POSS = True  # If True, plot unnormalized values

# Sweep: d_over_d_ref from 0.5 to 2.5
D_OVER_D_REF_VALUES = np.linspace(0.5, 2.5, 2000)
NTU_MAX_AT_D_REF = 4.0  # NTU max at d/d_ref = 1
NTU_NUM = 1000  # number of NTU points per d (sweep resolution)
CONTOUR_GRID_N = 2000  # grid size for interpolation; contour smoothness is set by this, not NTU_NUM
NTU_MIN = 0.4

# Y-axis max based on type
match Y_AXIS_TYPE:
    case "dp_over_p_in":
        Y_MAX = 0.11
    case "ao_over_ao_ref":
        Y_MAX = 4
    case "g2h":
        Y_MAX = 0.03
    case "ntu":
        Y_MAX = 2.0
    case _:
        Y_MAX = 4  # Default


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
            # return r"Dimensionless Mass Velocity ($(\dot{m}/A_o)^2 / (2p_{\mathrm{in}} \rho)$ [-])"
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
        dp_hot = dp_at_ntu_ref * (NTU / NTU_REF)^4.407 * (d / d_ref)^(-1.407)
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
    ntu_fine = np.linspace(0.35, NTU_MAX_AT_D_REF, 150)
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
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
    dp_max=DEFAULT_DP_MAX,
    base_name="fig6",
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
    skip_interpolation = False
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
        yy = df[sweep_mask]["y_axis_value"].values  # Changed from ao_over_ao_ref
        zz = df[sweep_mask]["dqom_practical_over_qmax"].values
        # Extract optimal line data (stored as d, ntu pairs - y computed on the fly)
        opt_mask = df["d_opt"].notna()
        if opt_mask.any():
            d_opt_line = df[opt_mask]["d_opt"].values
            # Check if ntu_opt exists (new format)
            if "ntu_opt" in df.columns and df[opt_mask]["ntu_opt"].notna().any():
                ntu_opt_line = df[opt_mask]["ntu_opt"].values
                z_practical_opt_line = df[opt_mask]["dqom_practical_opt"].values
                # Compute y_opt_line from ntu_opt_line based on current Y_AXIS_TYPE
                y_opt_line = np.array(
                    [
                        _y_axis(d, ntu, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
                        for d, ntu in zip(d_opt_line, ntu_opt_line)
                    ]
                )
            else:
                # Old format or missing - will need to recompute optimal line
                print("Warning: Optimal line not in cache or old format. Will recompute.")
                d_opt_line = np.array([])
                y_opt_line = np.array([])
                z_practical_opt_line = np.array([])
                ntu_opt_line = None
        else:
            d_opt_line = np.array([])
            y_opt_line = np.array([])
            z_practical_opt_line = np.array([])
            ntu_opt_line = None
        # Get fig4 optimum (stored as ntu_opt, compute y on the fly)
        fig4_row = df[df["is_fig4_opt"]]
        if len(fig4_row) > 0:
            if "ntu_opt" in fig4_row.columns and pd.notna(fig4_row["ntu_opt"].iloc[0]):
                ntu_opt = float(fig4_row["ntu_opt"].iloc[0])
                y_opt = _y_axis(1.0, ntu_opt, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
            elif "y_axis_value" in fig4_row.columns:
                # Old format - use stored y_axis_value (approximate)
                y_opt = float(fig4_row["y_axis_value"].iloc[0])
                ntu_opt = None  # Will need to recompute if needed
            else:
                y_opt = None
                ntu_opt = None
        else:
            y_opt = None
            ntu_opt = None

        # Try to load interpolated grid if available
        if "is_interpolation_grid" in df.columns:
            grid_rows = df[df["is_interpolation_grid"]]
            if len(grid_rows) > 0:
                grid_row = grid_rows.iloc[0]
                if "grid_x" in grid_row.index and "grid_y" in grid_row.index and "grid_z" in grid_row.index:
                    try:
                        grid_x = grid_row["grid_x"]
                        grid_y = grid_row["grid_y"]
                        grid_z = grid_row["grid_z"]
                        x_min = float(grid_row["x_min"])
                        x_max = float(grid_row["x_max"])
                        y_min = float(grid_row["y_min"])
                        y_max = float(grid_row["y_max"])
                        grid_n = int(grid_row["grid_n"])

                        # Reconstruct meshgrid
                        X = np.array(grid_x).reshape((grid_n, grid_n))
                        Y = np.array(grid_y).reshape((grid_n, grid_n))
                        Z = np.array(grid_z).reshape((grid_n, grid_n))
                        print(f"Loaded interpolated grid from cache ({grid_n}x{grid_n})")
                        skip_interpolation = True
                    except Exception as e:
                        print(f"Error loading interpolated grid: {e}, will recompute")
                        skip_interpolation = False

        # If optimal line is missing or in old format, recompute it
        if len(d_opt_line) == 0 or ntu_opt_line is None:
            print("Recomputing optimal NTU line...")
            # Need to compute dp_at_ntu_ref and other parameters for optimal line calculation
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
            dp_at_ntu_ref = _compute_dp_at_ntu_ref(
                c_cold_over_c_hot,
                st_over_f,
                f_c_over_f_h,
                d_r,
                g2_h,
                pressure_drop_ratio,
            )
            # Recompute optimal line
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
            # Also recompute fig4 optimum if missing
            if ntu_opt is None:
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
                if ntu_opt is not None:
                    y_opt = _y_axis(1.0, ntu_opt, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
                else:
                    y_opt = None

            # Save recomputed optimal line back to cache
            print("Saving recomputed optimal line to cache...")
            # Remove old optimal line data from cache
            df = pd.read_parquet(DATA_FILE)
            # Keep only sweep data (d_opt is NaN) and grid data, remove old optimal line
            df = df[(df["d_opt"].isna()) | (df.get("is_interpolation_grid", pd.Series([False] * len(df))) == True)]

            # Add new optimal line data
            opt_df = pd.DataFrame(
                {
                    "input_hash": [input_hash] * len(d_opt_line),
                    "d_over_d_ref": [np.nan] * len(d_opt_line),
                    "y_axis_value": [np.nan] * len(d_opt_line),
                    "dqom_practical_over_qmax": [np.nan] * len(d_opt_line),
                    "d_opt": d_opt_line,
                    "ntu_opt": ntu_opt_line,
                    "dqom_practical_opt": z_practical_opt_line,
                    "is_fig4_opt": [False] * len(d_opt_line),
                }
            )
            dfs_to_save = [df, opt_df]

            # Add fig4 optimum if available
            if ntu_opt is not None:
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
                fig4_df = pd.DataFrame(
                    {
                        "input_hash": [input_hash],
                        "d_over_d_ref": [1.0],
                        "y_axis_value": [y_opt],  # Keep for backward compatibility
                        "dqom_practical_over_qmax": [z_fig4_practical],
                        "d_opt": [np.nan],
                        "ntu_opt": [ntu_opt],
                        "dqom_practical_opt": [np.nan],
                        "is_fig4_opt": [True],
                    }
                )
                dfs_to_save.append(fig4_df)

            # Re-add grid data if it exists
            if skip_interpolation and "is_interpolation_grid" in df.columns:
                grid_rows = df[df["is_interpolation_grid"]]
                if len(grid_rows) > 0:
                    dfs_to_save.append(grid_rows)

            df_updated = pd.concat(dfs_to_save, ignore_index=True)
            df_updated.to_parquet(DATA_FILE)
            print("Updated cache with recomputed optimal line")
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
        print(
            f" Optimum NTU and y-axis value at d=1: {ntu_opt}, {_y_axis(1.0, ntu_opt, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)}"
        )
        if ntu_opt is not None:
            y_opt = _y_axis(1.0, ntu_opt, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
        else:
            y_opt = None

        # Optimal NTU line: for each d, find optimal NTU (minimizing practical)
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
            print(f"Total points: {len(zz)}, Valid points: {np.sum(valid)}")
            return

        print(f"Valid points: {np.sum(valid)}/{len(zz)}")
        print(f"x range: [{xx[valid].min():.3f}, {xx[valid].max():.3f}]")
        print(f"y range: [{yy[valid].min():.3f}, {yy[valid].max():.3f}]")
        print(f"z range: [{zz[valid].min():.3f}, {zz[valid].max():.3f}]")

        # Save data to parquet
        sweep_df = pd.DataFrame(
            {
                "input_hash": [input_hash] * len(xx),
                "d_over_d_ref": xx,
                "y_axis_value": yy,  # Generic name for y-axis value
                "dqom_practical_over_qmax": zz,
                "d_opt": [np.nan] * len(xx),
                "ntu_opt": [np.nan] * len(xx),  # Changed from y_opt
                "dqom_practical_opt": [np.nan] * len(xx),
                "is_fig4_opt": [False] * len(xx),
            }
        )

        opt_df = pd.DataFrame(
            {
                "input_hash": [input_hash] * len(d_opt_line),
                "d_over_d_ref": [np.nan] * len(d_opt_line),
                "y_axis_value": [np.nan] * len(d_opt_line),
                "dqom_practical_over_qmax": [np.nan] * len(d_opt_line),
                "d_opt": d_opt_line,
                "ntu_opt": ntu_opt_line,  # Store NTU instead of y_opt
                "dqom_practical_opt": z_practical_opt_line,
                "is_fig4_opt": [False] * len(d_opt_line),
            }
        )

        dfs = [sweep_df, opt_df]

        # Add fig4 optimum point if available
        if y_opt is not None:
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
            fig4_df = pd.DataFrame(
                {
                    "input_hash": [input_hash],
                    "d_over_d_ref": [1.0],
                    "y_axis_value": [y_opt],  # Keep for backward compatibility
                    "dqom_practical_over_qmax": [z_fig4_practical],
                    "d_opt": [np.nan],
                    "ntu_opt": [ntu_opt],  # Store NTU instead of y_opt
                    "dqom_practical_opt": [np.nan],
                    "is_fig4_opt": [True],
                }
            )
            dfs.append(fig4_df)

        # Interpolate onto regular grid for contour
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
        Z = griddata(
            (xx[valid], yy[valid]),
            zz[valid],
            (X, Y),
            method="cubic",
            fill_value=np.nan,
        )

        # Fallback to linear if cubic fails
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

        # Store interpolated grid in parquet
        grid_df = pd.DataFrame(
            {
                "input_hash": [input_hash],
                "d_over_d_ref": [np.nan],
                "y_axis_value": [np.nan],
                "dqom_practical_over_qmax": [np.nan],
                "d_opt": [np.nan],
                "ntu_opt": [np.nan],
                "dqom_practical_opt": [np.nan],
                "is_fig4_opt": [False],
                "is_interpolation_grid": [True],
                "grid_x": [X.flatten()],
                "grid_y": [Y.flatten()],
                "grid_z": [Z.flatten()],
                "x_min": [x_min],
                "x_max": [x_max],
                "y_min": [y_min],
                "y_max": [y_max],
                "grid_n": [CONTOUR_GRID_N],
            }
        )
        dfs.append(grid_df)

        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(DATA_FILE)
        print(f"Data saved to {DATA_FILE}")

    # Interpolate onto regular grid for contour (if not loaded from cache)
    if not skip_interpolation:
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
        Z = griddata(
            (xx[valid], yy[valid]),
            zz[valid],
            (X, Y),
            method="cubic",
            fill_value=np.nan,
        )

        # Fallback to linear if cubic fails
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

        # Save interpolated grid to cache if it wasn't already there
        if load_from_cache:
            # Load existing dataframe
            df = pd.read_parquet(DATA_FILE)
            # Remove old grid if it exists
            if "is_interpolation_grid" in df.columns:
                df = df[~df["is_interpolation_grid"]]
            # Add new grid
            grid_df = pd.DataFrame(
                {
                    "input_hash": [input_hash],
                    "d_over_d_ref": [np.nan],
                    "y_axis_value": [np.nan],
                    "dqom_practical_over_qmax": [np.nan],
                    "d_opt": [np.nan],
                    "ntu_opt": [np.nan],
                    "dqom_practical_opt": [np.nan],
                    "is_fig4_opt": [False],
                    "is_interpolation_grid": [True],
                    "grid_x": [X.flatten()],
                    "grid_y": [Y.flatten()],
                    "grid_z": [Z.flatten()],
                    "x_min": [x_min],
                    "x_max": [x_max],
                    "y_min": [y_min],
                    "y_max": [y_max],
                    "grid_n": [CONTOUR_GRID_N],
                }
            )
            df = pd.concat([df, grid_df], ignore_index=True)
            df.to_parquet(DATA_FILE)
            print(f"Saved interpolated grid to cache")

    # Plot with fig6.py formatting
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "mathtext.fontset": "stix",
        }
    )
    fig, ax = plt.subplots(figsize=(9 / 2.54, 7 / 2.54))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 0.03)
    # ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Core Volume Ratio ($V_\mathrm{core}/V_\mathrm{ref}$ [-])")
    # ax.set_ylabel(_get_y_axis_label())
    ax.set_ylabel(r"Dimensionless Mass Velocity ($(\dot{m}/A_o)^2 / (2p_{\mathrm{in}} \rho)$ [-])")

    # Contour plot
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    if not (np.isfinite(z_min) and np.isfinite(z_max)):
        print("Warning: No valid Z values for contour plot.")
        return

    # Round colorbar limits to nearest tenth
    vmin_r = np.round(z_min, 2)
    vmax_r = 0
    levels = np.linspace(vmin_r, vmax_r, 15)
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

    # Ref at (1, y_ref): diamond (match fig5/6/8 style); markers on top
    y_ref = _y_axis(1.0, NTU_REF, unnormalised=UNNORMALISED_Y_AXIS_IF_POSS)
    ax.scatter(
        [1.0],
        [y_ref],
        marker="D",
        facecolor="black",
        edgecolor="white",
        s=50,
        zorder=10,
        label="ref",
    )
    # Fig4 optimum: circle (match other plots: black face, white edge, s=50)
    if y_opt is not None and y_min <= y_opt <= y_max:
        ax.scatter(
            [1.0],
            [y_opt],
            marker="o",
            facecolor="black",
            edgecolor="white",
            s=50,
            zorder=10,
            label="fig4 opt",
        )
    # ax.legend(loc="upper left", fontsize=9)
    cbar = plt.colorbar(cs, ax=ax, format=mtick.FormatStrFormatter("%.2f"))
    cbar.set_label(r"Change in Unavailable Energy ($\Delta \dot{Q}_0^\mathrm{M}/\dot{Q}_{\mathrm{max}}$ [-])")
    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png", "pdf"]:
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
        base_name="fig6",
    )
