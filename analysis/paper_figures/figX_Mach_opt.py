"""
Fig X: Mach sweep comparing practical (euergy) optimum NTU vs classical (exergy)
interior local optimum NTU, with error metric and performance traces.

Defaults match fig6_smith_chart (dp_c<<dp_h, etc.). Classical availability uses
t_dead_over_t_cold_in = 1.1 for consistency with simple exergy normalization
when T_hot/T_cold = 2 (fig6 does not set dead state; override via constant below).

eps_M in outputs = practical availability increase = -practical_unavailable_creation_hex
(normalized by Q_max), same sign convention as fig4_bar_chart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xflow
from matplotlib.transforms import blended_transform_factory
from scipy.optimize import minimize_scalar
from xflow import (
    calculate_capacity_ratios,
    calculate_pressure_drop_ratio,
    classical_unavailable_creation_hex,
    practical_unavailable_creation_hex,
)

from heat_exchanger.epsilon_ntu import epsilon_ntu

xflow.SHOW_CUBIC = False

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_OUTPUT_DIR = SCRIPT_DIR / "Figs_current"
CACHE_DIR = SCRIPT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Match fig6_smith_chart defaults ---
DEFAULT_C_COLD_OVER_C_HOT = 1.0
DEFAULT_ST_OVER_F = 0.4
DEFAULT_F_C_OVER_F_H = 1.0
DEFAULT_D_R = 1.0
DEFAULT_DP_MAX = 0.3
DEFAULT_T = 2.0
DEFAULT_P_COLD_IN_OVER_P_HOT_IN = 10.0
DEFAULT_P_HOT_IN_OVER_P_DEAD = 1.1
DEFAULT_P_DEAD_OVER_P_HOT_IN = 1.0 / DEFAULT_P_HOT_IN_OVER_P_DEAD
DEFAULT_GAMMA = 1.4
NTU_GLOBAL_MAX = 15.0
DEFAULT_PRESSURE_DROP_ASSUMPTION = "dp_c<<dp_h"
DEFAULT_MOLAR_MASS_RATIO = 1.0
DEFAULT_A_R = 0.1

# Classical dead-state ratio (not in fig6); 1.1 matches xflow nominal g2lim docs.
DEFAULT_T_DEAD_OVER_T_COLD_IN = 1.1

MACH_MIN_DEFAULT = 0.02
MACH_MAX_DEFAULT = 0.2

# Lower end of NTU search domain (>0; excludes NTU=0 global artefact for classical).
NTU_DOMAIN_LO = 0.1


def _sanitize_pressure_drop_name(assumption: str) -> str:
    return assumption.replace("=", "_eq_").replace("<<", "_ll_")


def mach_to_g2(mach: float, gamma: float = DEFAULT_GAMMA) -> float:
    return 0.5 * gamma * mach**2


def ntu_max_valid(
    g2_h: float,
    c_cold_over_c_hot: float,
    st_over_f: float,
    f_c_over_f_h: float,
    d_r: float,
    dp_max: float,
    pressure_drop_ratio: float,
) -> float:
    C_min_over_C_hot, C_min_over_C_cold, _ = calculate_capacity_ratios(c_cold_over_c_hot)
    st_over_f_h = st_over_f
    st_over_f_c = st_over_f
    dp_coeff = g2_h * (
        1.0 / st_over_f_h * C_min_over_C_hot + 1.0 / f_c_over_f_h * 1.0 / st_over_f_c * d_r * C_min_over_C_cold
    )
    if dp_coeff <= 0:
        return NTU_GLOBAL_MAX
    dp_max_eff = dp_max if pressure_drop_ratio <= 1.0 else dp_max / pressure_drop_ratio
    return min(dp_max_eff / dp_coeff, NTU_GLOBAL_MAX)


def _ntu_coarse_grid(ntu_max: float, n_points: int) -> np.ndarray:
    """Adaptive NTU samples: log spacing when range is large, linear when ntu_max is small."""
    lo = min(NTU_DOMAIN_LO, ntu_max * 0.5)
    lo = max(lo, ntu_max * 1e-4)
    hi = max(ntu_max, lo * 1.01)
    if hi / lo > 5.0:
        return np.logspace(np.log10(lo), np.log10(hi), n_points)
    return np.linspace(lo, hi, n_points)


def _interior_local_max_ntu(ntu: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """
    Strict interior local maxima: y[i] >= neighbors for 0 < i < n-1.
    If several, return the one with largest y.
    """
    mask = np.isfinite(ntu) & np.isfinite(y)
    nv = ntu[mask]
    yv = y[mask]
    if len(yv) < 3:
        return None
    best_ntu, best_y = None, -np.inf
    for i in range(1, len(yv) - 1):
        if yv[i] >= yv[i - 1] and yv[i] >= yv[i + 1] and yv[i] > best_y:
            best_y = float(yv[i])
            best_ntu = float(nv[i])
    if best_ntu is None:
        return None
    return best_ntu, best_y


class MachOptContext:
    """Fixed thermo/geometry; evaluates metrics at scalar NTU."""

    def __init__(
        self,
        mach: float,
        *,
        pressure_drop_assumption: str,
        c_cold_over_c_hot: float,
        st_over_f: float,
        f_c_over_f_h: float,
        d_r: float,
        dp_max: float,
        t: float,
        t_dead_over_t_cold_in: float,
        p_cold_in_over_p_hot_in: float,
        p_dead_over_p_hot_in: float,
        gamma: float,
        molar_mass_ratio: float,
        a_r: float,
    ) -> None:
        self.mach = float(mach)
        self.g2_h = mach_to_g2(mach, gamma)
        self.c_cold_over_c_hot = c_cold_over_c_hot
        self.st_over_f = st_over_f
        self.f_c_over_f_h = f_c_over_f_h
        self.d_r = d_r
        self.dp_max = dp_max
        self.t = t
        self.t_dead_over_t_cold_in = t_dead_over_t_cold_in
        self.p_cold_in_over_p_hot_in = p_cold_in_over_p_hot_in
        self.p_dead_over_p_hot_in = p_dead_over_p_hot_in
        self.gamma = gamma

        sigma_r = d_r * a_r
        self.pressure_drop_ratio = calculate_pressure_drop_ratio(
            pressure_drop_assumption,
            c_cold_over_c_hot,
            t,
            d_r,
            molar_mass_ratio,
            sigma_r,
            p_cold_in_over_p_hot_in,
        )
        self.C_min_over_C_hot, self.C_min_over_C_cold, _ = calculate_capacity_ratios(c_cold_over_c_hot)
        if c_cold_over_c_hot <= 1.0:
            self.C_ratio = c_cold_over_c_hot
        else:
            self.C_ratio = 1.0 / c_cold_over_c_hot

        self.ntu_max = ntu_max_valid(
            self.g2_h,
            c_cold_over_c_hot,
            st_over_f,
            f_c_over_f_h,
            d_r,
            dp_max,
            self.pressure_drop_ratio,
        )
        self.ntu_max = max(self.ntu_max, 0.15)  # fig6-style minimum span

        self.dp_coeff = self.g2_h * (
            1.0 / st_over_f * self.C_min_over_C_hot
            + 1.0 / f_c_over_f_h * 1.0 / st_over_f * d_r * self.C_min_over_C_cold
        )

    def bounds(self) -> tuple[float, float]:
        hi = self.ntu_max
        lo = min(NTU_DOMAIN_LO, hi * 0.5)
        lo = max(lo, hi * 1e-5)
        return lo, hi

    def evaluate(self, ntu: float) -> dict:
        """Return dict with practical/classical availability, epsilon, dp fractions; NaNs if invalid."""
        ntu = float(np.clip(ntu, self.bounds()[0], self.bounds()[1]))
        eps = float(
            epsilon_ntu(
                np.array([ntu]),
                self.C_ratio,
                exchanger_type="aligned_flow",
                flow_type="counterflow",
                n_passes=1,
            )[0]
        )
        dp_hot = self.dp_coeff * ntu
        dp_cold = self.pressure_drop_ratio * dp_hot
        valid = (dp_hot < self.dp_max) & (dp_cold < self.dp_max)
        if not valid:
            return {
                "ntu": ntu,
                "epsilon": eps,
                "dp_hot": dp_hot,
                "dp_cold": dp_cold,
                "practical_avail": np.nan,
                "classical_avail": np.nan,
                "valid": False,
            }

        mask = np.array([True])
        pu = practical_unavailable_creation_hex(
            np.array([eps]),
            self.t,
            np.array([dp_hot]),
            np.array([dp_cold]),
            mask,
            p_cold_in_over_p_hot_in=self.p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in=self.p_dead_over_p_hot_in,
            gamma=self.gamma,
        )
        cu = classical_unavailable_creation_hex(
            np.array([eps]),
            self.t,
            np.array([dp_hot]),
            np.array([dp_cold]),
            mask,
            t_dead_over_t_cold_in=self.t_dead_over_t_cold_in,
            gamma=self.gamma,
        )
        pa = float(-np.asarray(pu).flat[0])
        ca = float(-np.asarray(cu).flat[0])
        return {
            "ntu": ntu,
            "epsilon": eps,
            "dp_hot": dp_hot,
            "dp_cold": dp_cold,
            "practical_avail": pa,
            "classical_avail": ca,
            "valid": True,
        }


def _vector_metrics(ctx: MachOptContext, ntu_arr: np.ndarray) -> dict[str, np.ndarray]:
    """Batch evaluate for coarse search."""
    epsilon = epsilon_ntu(ntu_arr, ctx.C_ratio, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)
    dp_hot = ctx.dp_coeff * ntu_arr
    dp_cold = ctx.pressure_drop_ratio * dp_hot
    valid = (dp_hot < ctx.dp_max) & (dp_cold < ctx.dp_max)

    n = len(ntu_arr)
    practical_avail = np.full(n, np.nan)
    classical_avail = np.full(n, np.nan)
    if np.any(valid):
        pu = practical_unavailable_creation_hex(
            epsilon,
            ctx.t,
            dp_hot,
            dp_cold,
            valid,
            p_cold_in_over_p_hot_in=ctx.p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in=ctx.p_dead_over_p_hot_in,
            gamma=ctx.gamma,
        )
        cu = classical_unavailable_creation_hex(
            epsilon,
            ctx.t,
            dp_hot,
            dp_cold,
            valid,
            t_dead_over_t_cold_in=ctx.t_dead_over_t_cold_in,
            gamma=ctx.gamma,
        )
        practical_avail[valid] = -np.asarray(pu, dtype=float)
        classical_avail[valid] = -np.asarray(cu, dtype=float)
    return {
        "ntu": ntu_arr,
        "epsilon": epsilon,
        "dp_hot": dp_hot,
        "dp_cold": dp_cold,
        "practical_avail": practical_avail,
        "classical_avail": classical_avail,
        "valid": valid,
    }


def optimize_practical_ntu(ctx: MachOptContext) -> tuple[float, dict]:
    lo, hi = ctx.bounds()

    def neg_pa(n):
        d = ctx.evaluate(float(n))
        if not d["valid"] or not np.isfinite(d["practical_avail"]):
            return 1e300
        return -d["practical_avail"]

    res = minimize_scalar(neg_pa, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    ntu_p = float(res.x)
    return ntu_p, ctx.evaluate(ntu_p)


def classical_interior_local_optimum_ntu(ctx: MachOptContext, n_coarse: int) -> tuple[float | None, dict | None]:
    """
    Find an interior local maximum of classical availability on (0, ntu_max].
    Returns (ntu_c, metrics_dict) or (None, None) if none exists.
    """
    lo, hi = ctx.bounds()
    grid = _ntu_coarse_grid(hi, n_coarse)
    grid = grid[(grid >= lo) & (grid <= hi)]
    if len(grid) < 5:
        grid = np.linspace(lo, hi, max(5, n_coarse))

    vm = _vector_metrics(ctx, grid)
    peak = _interior_local_max_ntu(vm["ntu"], vm["classical_avail"])
    if peak is None:
        return None, None

    ntu0, _y0 = peak
    # Refine with bounded scalar optimization on classical availability
    idx = int(np.argmin(np.abs(vm["ntu"] - ntu0)))
    i0 = max(0, idx - 2)
    i1 = min(len(vm["ntu"]) - 1, idx + 2)
    lo_r = float(max(lo, vm["ntu"][i0] * 0.98))
    hi_r = float(min(hi, vm["ntu"][i1] * 1.02))
    if hi_r <= lo_r:
        lo_r, hi_r = lo, hi

    def neg_ca(n):
        d = ctx.evaluate(float(n))
        if not d["valid"] or not np.isfinite(d["classical_avail"]):
            return 1e300
        return -d["classical_avail"]

    res = minimize_scalar(neg_ca, bounds=(lo_r, hi_r), method="bounded", options={"xatol": 1e-5})
    ntu_c = float(res.x)
    out = ctx.evaluate(ntu_c)
    if not out["valid"]:
        return None, None
    # Confirm not a spurious plateau: should beat at least one side on a tight bracket
    span = max((hi - lo) * 1e-4, 1e-6)
    dminus = ctx.evaluate(max(lo, ntu_c - span))
    dplus = ctx.evaluate(min(hi, ntu_c + span))
    if (
        dminus["valid"]
        and dplus["valid"]
        and out["classical_avail"] + 1e-12 < max(dminus["classical_avail"], dplus["classical_avail"])
    ):
        return None, None
    return ntu_c, out


def sweep_mach(
    mach_values: np.ndarray,
    *,
    n_coarse_classical: int = 120,
    pressure_drop_assumption: str = DEFAULT_PRESSURE_DROP_ASSUMPTION,
    c_cold_over_c_hot: float = DEFAULT_C_COLD_OVER_C_HOT,
    st_over_f: float = DEFAULT_ST_OVER_F,
    f_c_over_f_h: float = DEFAULT_F_C_OVER_F_H,
    d_r: float = DEFAULT_D_R,
    dp_max: float = DEFAULT_DP_MAX,
    t: float = DEFAULT_T,
    t_dead_over_t_cold_in: float = DEFAULT_T_DEAD_OVER_T_COLD_IN,
    p_cold_in_over_p_hot_in: float = DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in: float = DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma: float = DEFAULT_GAMMA,
    molar_mass_ratio: float = DEFAULT_MOLAR_MASS_RATIO,
    a_r: float = DEFAULT_A_R,
) -> pd.DataFrame:
    rows = []

    for mach in mach_values:
        ctx = MachOptContext(
            float(mach),
            pressure_drop_assumption=pressure_drop_assumption,
            c_cold_over_c_hot=c_cold_over_c_hot,
            st_over_f=st_over_f,
            f_c_over_f_h=f_c_over_f_h,
            d_r=d_r,
            dp_max=dp_max,
            t=t,
            t_dead_over_t_cold_in=t_dead_over_t_cold_in,
            p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
            p_dead_over_p_hot_in=p_dead_over_p_hot_in,
            gamma=gamma,
            molar_mass_ratio=molar_mass_ratio,
            a_r=a_r,
        )

        ntu_p, met_p = optimize_practical_ntu(ctx)
        ntu_c, met_c = classical_interior_local_optimum_ntu(ctx, n_coarse=n_coarse_classical)

        has_classical = met_c is not None

        if has_classical:
            met_at_c = ctx.evaluate(float(ntu_c))
            eps_m_c = met_at_c["practical_avail"]
            eps_m_p = met_p["practical_avail"]
            if np.isfinite(eps_m_c) and np.isfinite(eps_m_p) and abs(eps_m_p) > 1e-15:
                err = eps_m_c / eps_m_p - 1.0
            else:
                err = np.nan
        else:
            met_at_c = {
                "ntu": 0.0,
                "epsilon": 0.0,
                "dp_hot": 0.0,
                "dp_cold": 0.0,
                "practical_avail": np.nan,
                "classical_avail": np.nan,
                "valid": True,
            }
            err = np.nan

        rows.append(
            {
                "mach": float(mach),
                "g2_h": ctx.g2_h,
                "ntu_max_valid": ctx.ntu_max,
                "ntu_practical_opt": ntu_p,
                "ntu_classical_local_opt": met_at_c["ntu"],
                "has_classical_interior_opt": has_classical,
                "eps_m_practical_opt": met_p["practical_avail"],
                "eps_m_at_classical_opt": met_at_c["practical_avail"],
                "epsilon_practical_opt": met_p["epsilon"],
                "epsilon_classical_opt": met_at_c["epsilon"],
                "dp_hot_practical_opt": met_p["dp_hot"],
                "dp_hot_classical_opt": met_at_c["dp_hot"],
                "dp_cold_practical_opt": met_p["dp_cold"],
                "dp_cold_classical_opt": met_at_c["dp_cold"],
                "classical_avail_at_classical_opt": met_c["classical_avail"] if has_classical else np.nan,
                "error_eps_m_ratio_minus_1": err,
            }
        )

    df = pd.DataFrame(rows)
    pair = _last_transition_pair_from_df(df)
    if pair is not None:
        m_last, m_first_bad = pair
        print(
            "Classical interior local optimum lost (increasing Mach): "
            f"last Mach where found = {m_last}, first Mach where not found = {m_first_bad}"
        )
    else:
        print(
            "No Mach transition from classical interior opt present to absent in this sweep "
            "(see column has_classical_interior_opt)."
        )
    return df


def _cache_path(config: dict) -> Path:
    s = json.dumps(config, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()[:12]
    safe = _sanitize_pressure_drop_name(str(config.get("pressure_drop_assumption", "")))
    return CACHE_DIR / f"figX_mach_opt_{safe}_{h}.parquet"


def run_sweep_cached(
    mach_min: float,
    mach_max: float,
    n_mach: int,
    *,
    force_recompute: bool = False,
    **sweep_kw,
) -> pd.DataFrame:
    mach_values = np.linspace(mach_min, mach_max, n_mach)
    config = {
        "mach_min": mach_min,
        "mach_max": mach_max,
        "n_mach": n_mach,
        **{k: sweep_kw[k] for k in sorted(sweep_kw)},
    }
    path = _cache_path(config)
    if not force_recompute and path.exists():
        print(f"Loaded cached sweep from {path.name}")
        return pd.read_parquet(path)
    t0 = time.perf_counter()
    df = sweep_mach(mach_values, **sweep_kw)
    df.to_parquet(path, index=False)
    print(f"Saved sweep cache {path.name} ({time.perf_counter() - t0:.2f}s)")
    return df


def _last_transition_pair_from_df(df: pd.DataFrame) -> tuple[float, float] | None:
    """Last (increasing Mach) transition from has_classical True to False; else None."""
    m = df["mach"].values
    h = df["has_classical_interior_opt"].values
    last_true: float | None = None
    pair: tuple[float, float] | None = None
    for i in range(len(h)):
        if h[i]:
            last_true = float(m[i])
        elif last_true is not None:
            pair = (last_true, float(m[i]))
            last_true = None
    return pair


def _m_lim_from_df(df: pd.DataFrame) -> float | None:
    """M_lim = Mach of first False after last contiguous True region (last transition)."""
    p = _last_transition_pair_from_df(df)
    return None if p is None else p[1]


def _apply_plot_style():
    font_size = 8
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": font_size,
            "mathtext.fontset": "stix",
        }
    )


def plot_figures(df: pd.DataFrame, base_name: str = "figX_Mach_opt") -> None:
    _apply_plot_style()
    m_lim = _m_lim_from_df(df)
    mach = df["mach"].values

    # --- 1) Error ---
    fig1, ax1 = plt.subplots(figsize=(9 / 2.54, 7 / 2.54))
    mask_err = df["has_classical_interior_opt"] & np.isfinite(df["error_eps_m_ratio_minus_1"])
    ax1.plot(mach[mask_err], df.loc[mask_err, "error_eps_m_ratio_minus_1"], color="k", lw=1.0)
    ax1.set_xlabel(r"Hot inlet Mach number $M_\mathrm{in}$ [-]")
    ax1.set_ylabel(r"$\varepsilon_{M,\mathrm{cl}} / \varepsilon_{M,\mathrm{pr}} - 1$")
    ax1.axhline(0.0, color="0.7", lw=0.5, ls=":")
    if m_lim is not None:
        ax1.axvline(m_lim, color="0.5", ls="--", lw=0.6)
        trans = blended_transform_factory(ax1.transData, ax1.transAxes)
        ax1.text(
            m_lim,
            -0.22,
            f"$M_\\mathrm{{lim}}={m_lim:.2f}$",
            transform=trans,
            va="top",
            ha="center",
            fontsize=7,
            color="0.3",
        )
    plt.tight_layout()
    for fmt in ("svg", "pdf", "png"):
        p = FIG_OUTPUT_DIR / f"{base_name}_error.{fmt}"
        fig1.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p.name}")
    plt.close(fig1)

    # --- 2) eps_M practical metric at both NTUs ---
    fig2, ax2 = plt.subplots(figsize=(9 / 2.54, 7 / 2.54))
    ax2.plot(mach, df["eps_m_practical_opt"], color="k", ls="-", lw=1.0, label=r"$\varepsilon_M$ at practical opt.")
    mc = df["has_classical_interior_opt"]
    ax2.plot(
        mach[mc],
        df.loc[mc, "eps_m_at_classical_opt"],
        color="k",
        ls="--",
        lw=1.0,
        label=r"$\varepsilon_M$ at classical local opt.",
    )
    ax2.set_xlabel(r"$M_\mathrm{in}$ [-]")
    ax2.set_ylabel(r"Practical availability increase $\varepsilon_M$ [-]")
    if m_lim is not None:
        ax2.axvline(m_lim, color="0.5", ls="--", lw=0.6)
        trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)
        ax2.text(
            m_lim,
            -0.22,
            f"$M_\\mathrm{{lim}}={m_lim:.2f}$",
            transform=trans2,
            va="top",
            ha="center",
            fontsize=7,
            color="0.3",
        )
    ax2.legend(frameon=False, loc="best", fontsize=7)
    plt.tight_layout()
    for fmt in ("svg", "pdf", "png"):
        p = FIG_OUTPUT_DIR / f"{base_name}_epsM.{fmt}"
        fig2.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p.name}")
    plt.close(fig2)

    # --- 3) epsilon + dp (practical solid, classical dashed) ---
    fig3, ax3 = plt.subplots(figsize=(9 / 2.54, 7 / 2.54))
    c_eps = (0.85, 0.2, 0.2)
    c_dph = (0.2, 0.65, 0.35)
    c_dpc = (0.25, 0.45, 0.85)

    ax3.plot(mach, df["epsilon_practical_opt"], color=c_eps, ls="-", lw=1.0, label=r"$\varepsilon$ practical")
    ax3.plot(mach, df["epsilon_classical_opt"], color=c_eps, ls="--", lw=1.0, label=r"$\varepsilon$ classical")
    ax3.plot(
        mach,
        df["dp_hot_practical_opt"],
        color=c_dph,
        ls="-",
        lw=1.0,
        label=r"$(\Delta p/p_\mathrm{in})_\mathrm{hot}$ practical",
    )
    ax3.plot(
        mach,
        df["dp_hot_classical_opt"],
        color=c_dph,
        ls="--",
        lw=1.0,
        label=r"$(\Delta p/p_\mathrm{in})_\mathrm{hot}$ classical",
    )
    ax3.plot(
        mach,
        df["dp_cold_practical_opt"],
        color=c_dpc,
        ls="-",
        lw=1.0,
        label=r"$(\Delta p/p_\mathrm{in})_\mathrm{cold}$ practical",
    )
    ax3.plot(
        mach,
        df["dp_cold_classical_opt"],
        color=c_dpc,
        ls="--",
        lw=1.0,
        label=r"$(\Delta p/p_\mathrm{in})_\mathrm{cold}$ classical",
    )

    ax3.set_xlabel(r"$M_\mathrm{in}$ [-]")
    ax3.set_ylabel(r"$\varepsilon$, $\Delta p / p_\mathrm{in}$ [-]")
    if m_lim is not None:
        ax3.axvline(m_lim, color="0.5", ls="--", lw=0.6)
        trans3 = blended_transform_factory(ax3.transData, ax3.transAxes)
        ax3.text(
            m_lim,
            -0.32,
            f"$M_\\mathrm{{lim}}={m_lim:.2f}$",
            transform=trans3,
            va="top",
            ha="center",
            fontsize=7,
            color="0.3",
        )
    ax3.legend(frameon=False, loc="best", fontsize=6, ncol=1)
    plt.tight_layout()
    for fmt in ("svg", "pdf", "png"):
        p = FIG_OUTPUT_DIR / f"{base_name}_perf.{fmt}"
        fig3.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved {p.name}")
    plt.close(fig3)


def main():
    parser = argparse.ArgumentParser(description="Mach sweep: practical vs classical NTU optima (fig X).")
    parser.add_argument("--mach-min", type=float, default=MACH_MIN_DEFAULT)
    parser.add_argument("--mach-max", type=float, default=MACH_MAX_DEFAULT)
    parser.add_argument("--n-mach", type=int, default=40)
    parser.add_argument("--n-coarse", type=int, default=120, help="Coarse NTU points per Mach for classical local peak.")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    sweep_kw = {
        "n_coarse_classical": args.n_coarse,
        "pressure_drop_assumption": DEFAULT_PRESSURE_DROP_ASSUMPTION,
        "c_cold_over_c_hot": DEFAULT_C_COLD_OVER_C_HOT,
        "st_over_f": DEFAULT_ST_OVER_F,
        "f_c_over_f_h": DEFAULT_F_C_OVER_F_H,
        "d_r": DEFAULT_D_R,
        "dp_max": DEFAULT_DP_MAX,
        "t": DEFAULT_T,
        "t_dead_over_t_cold_in": DEFAULT_T_DEAD_OVER_T_COLD_IN,
        "p_cold_in_over_p_hot_in": DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        "p_dead_over_p_hot_in": DEFAULT_P_DEAD_OVER_P_HOT_IN,
        "gamma": DEFAULT_GAMMA,
        "molar_mass_ratio": DEFAULT_MOLAR_MASS_RATIO,
        "a_r": DEFAULT_A_R,
    }

    df = run_sweep_cached(
        args.mach_min,
        args.mach_max,
        args.n_mach,
        force_recompute=args.force_recompute,
        **sweep_kw,
    )

    mlim = _m_lim_from_df(df)
    frac_ok = df["has_classical_interior_opt"].mean()
    print(f"Fraction of Mach with classical interior opt: {frac_ok:.2%}")
    if mlim is not None:
        print(f"M_lim (first Mach without classical interior local opt): {mlim:.4f}")

    if not args.no_plot:
        plot_figures(df)

    return df


if __name__ == "__main__":
    main()
