"""
2D contour plot of classical unavailable energy vs d_over_d_ref and scaled NTU.
Uses same inputs as newfig5_practical (Helicopter defaults). Pressure drop includes
multiplier d_over_d_ref**1.407. Y-axis: (NTU/NTU_match)**(-1.704) * (d_over_d_ref)**(-0.704).
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

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


# Import shared constants and functions from practical version
try:
    import newfig5_practical as nf5p
except ImportError:
    # If running as a script, try relative import
    from . import newfig5_practical as nf5p

save_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = Path(save_dir) / "newfig5_classical_data.parquet"

# Import constants
AO_MAX = nf5p.AO_MAX  # 4
CONTOUR_GRID_N = nf5p.CONTOUR_GRID_N
D_OVER_D_REF_VALUES = nf5p.D_OVER_D_REF_VALUES
DEFAULT_A_R = nf5p.DEFAULT_A_R
DEFAULT_C_COLD_OVER_C_HOT = nf5p.DEFAULT_C_COLD_OVER_C_HOT
DEFAULT_DP_MAX = nf5p.DEFAULT_DP_MAX
DEFAULT_D_R = nf5p.DEFAULT_D_R
DEFAULT_F_C_OVER_F_H = nf5p.DEFAULT_F_C_OVER_F_H
DEFAULT_G2_H = nf5p.DEFAULT_G2_H
DEFAULT_GAMMA = nf5p.DEFAULT_GAMMA
DEFAULT_MOLAR_MASS_RATIO = nf5p.DEFAULT_MOLAR_MASS_RATIO
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = nf5p.DEFAULT_P_COLD_IN_OVER_P_HOT_IN
DEFAULT_P_DEAD_OVER_P_HOT_IN = nf5p.DEFAULT_P_DEAD_OVER_P_HOT_IN
DEFAULT_PRESSURE_DROP_ASSUMPTION = nf5p.DEFAULT_PRESSURE_DROP_ASSUMPTION
DEFAULT_ST_OVER_F = nf5p.DEFAULT_ST_OVER_F
DEFAULT_T = nf5p.DEFAULT_T
DEFAULT_T_DEAD_OVER_T_COLD_IN = nf5p.DEFAULT_T_DEAD_OVER_T_COLD_IN
DP_REF = nf5p.DP_REF
NTU_MAX_AT_D_REF = nf5p.NTU_MAX_AT_D_REF
NTU_MIN = nf5p.NTU_MIN  # 0.4
NTU_NUM = nf5p.NTU_NUM
NTU_REF = nf5p.NTU_REF


def _classical_fig4_optimum_ntu(
    ntu_ref,
    dp_hot_ref,
    c_cold_over_c_hot,
    pressure_drop_ratio,
    t,
    t_dead_over_t_cold_in,
    gamma,
    dp_max,
):
    """At d_over_d_ref=1, find NTU that minimizes classical_unavailable_creation."""
    ntu_fine = np.linspace(NTU_MIN, NTU_MAX_AT_D_REF, 150)
    vals = []
    for ntu in ntu_fine:
        z = nf5p._classical_at_d_ntu(
            1.0,
            ntu,
            ntu_ref,
            dp_hot_ref,
            c_cold_over_c_hot,
            pressure_drop_ratio,
            t,
            t_dead_over_t_cold_in,
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
    base_name="newfig5_classical",
):
    """Build 2D sweep (d_over_d_ref, NTU) and plot contour of classical unavailable energy."""
    # Check if we can load from cache
    input_hash = nf5p._compute_input_hash(
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
        zz = df[sweep_mask]["dqom_classical_over_qmax"].values
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
        pressure_drop_ratio = nf5p.calculate_pressure_drop_ratio(
            pressure_drop_assumption,
            c_cold_over_c_hot,
            t,
            d_r,
            molar_mass_ratio,
            sigma_r,
            p_cold_in_over_p_hot_in,
        )

        # Reference dp curve at d=1 (used for all d via multiplier d**1.407)
        ntu_ref, dp_hot_ref = nf5p._compute_reference_dp_curve(
            c_cold_over_c_hot,
            st_over_f,
            f_c_over_f_h,
            d_r,
            g2_h,
            pressure_drop_ratio,
            dp_max,
        )

        # Fig4 optimum: at d=1, NTU that minimizes classical
        ntu_opt = _classical_fig4_optimum_ntu(
            ntu_ref,
            dp_hot_ref,
            c_cold_over_c_hot,
            pressure_drop_ratio,
            t,
            t_dead_over_t_cold_in,
            gamma,
            dp_max,
        )
        if ntu_opt is not None:
            y_opt = nf5p._y_axis(1.0, ntu_opt)
        else:
            y_opt = None

        # Optimal NTU line: for each d, find optimal NTU (minimizing classical, also calculate practical)
        # NOTE: Classical unavailable energy may not have a true optimum - it might always increase with NTU
        # If so, the "optimum" will always be at NTU_MIN (the lowest NTU value)
        (
            d_opt_line,
            ntu_opt_line,
            y_opt_line,
            z_practical_opt_line,
            z_classical_opt_line,
            z_practical_thermal_only_opt_line,
            z_classical_thermal_only_opt_line,
        ) = nf5p._optimal_ntu_line(
            ntu_ref,
            dp_hot_ref,
            c_cold_over_c_hot,
            pressure_drop_ratio,
            t,
            p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in,
            t_dead_over_t_cold_in,
            gamma,
            dp_max,
            optimize_practical=False,  # Minimize classical instead
            ntu_min=NTU_MIN,
        )

        # Debug: Check if optimum is always at minimum NTU (indicating no true optimum)
        if len(ntu_opt_line) > 0:
            min_ntu_ratio = np.min(ntu_opt_line) / NTU_MIN
            max_ntu_ratio = np.max(ntu_opt_line) / NTU_MIN
            print(f"Classical optimum NTU range: {np.min(ntu_opt_line):.3f} to {np.max(ntu_opt_line):.3f}")
            print(f"NTU_MIN = {NTU_MIN:.3f}")
            print(f"Optimum NTU is {min_ntu_ratio:.2f}x to {max_ntu_ratio:.2f}x the minimum NTU")
            if max_ntu_ratio < 1.1:  # If all optima are very close to NTU_MIN
                print(
                    "WARNING: Classical optimum appears to always be at minimum NTU - classical unavailable energy may be monotonic!"
                )

        # 2D sweep: for each d, sweep NTU (NTU max increases for smaller d)
        # For smaller d/d_ref, pressure drop is lower (scaled by d**1.407), so we can go to higher NTU
        xx, yy, zz = [], [], []
        total_d_values = len(D_OVER_D_REF_VALUES)
        total_points = total_d_values * NTU_NUM
        pbar = tqdm(total=total_points, desc="Computing sweep", unit="points")
        for d in D_OVER_D_REF_VALUES:
            ntu_max_d = nf5p._ntu_max_for_d(d, dp_max)  # NTU max depends on d and dp_max
            ntu_sweep = np.linspace(NTU_MIN, ntu_max_d, NTU_NUM)
            for ntu in ntu_sweep:
                z = nf5p._classical_at_d_ntu(
                    d,
                    ntu,
                    ntu_ref,
                    dp_hot_ref,
                    c_cold_over_c_hot,
                    pressure_drop_ratio,
                    t,
                    t_dead_over_t_cold_in,
                    gamma,
                    dp_max,
                )
                y = nf5p._y_axis(d, ntu)
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
            print("No valid classical values in sweep.")
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
                "dqom_practical_over_qmax": [np.nan] * len(xx),  # Not calculated for sweep
                "dqom_classical_over_qmax": zz,
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
            z_fig4_practical = nf5p._practical_at_d_ntu(
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
            z_fig4_classical = nf5p._classical_at_d_ntu(
                1.0,
                ntu_opt,
                ntu_ref,
                dp_hot_ref,
                c_cold_over_c_hot,
                pressure_drop_ratio,
                t,
                t_dead_over_t_cold_in,
                gamma,
                dp_max,
            )
            z_fig4_practical_thermal_only = nf5p._practical_at_d_ntu_thermal_only(
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
            z_fig4_classical_thermal_only = nf5p._classical_at_d_ntu(
                1.0,
                ntu_opt,
                ntu_ref,
                dp_hot_ref,
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
    print("Interpolating onto regular grid...")
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

    # Contour plot (classical unavailable energy)
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    if not (np.isfinite(z_min) and np.isfinite(z_max)):
        print("Warning: No valid Z values for contour plot.")
        return

    # Classical unavailable energy is always positive, so use full range
    levels = np.linspace(min(0, z_min), z_max, 15)
    levels = levels[np.isfinite(levels)]
    # Ensure levels are sorted and increasing
    levels = np.sort(levels)
    levels = np.unique(levels)  # Remove duplicates
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
                label="optimal NTU (classical)",
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
    ax.set_title(r"HEx $\Delta Q_0 / Q_{\mathrm{max}}$")
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
        base_name="newfig5_classical",
    )
