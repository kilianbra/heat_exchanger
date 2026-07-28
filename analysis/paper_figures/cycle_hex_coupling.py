"""
Bridge between geometric HEx sweeps (fig6/fig7) and the simple cycle model.

Lengthening (fig6): Ao/Ao_ref = 1, sweep NTU with A/A_ref = NTU/NTU_MATCH.

Coupling modes
--------------
cycle_fxd_mdot  — industrial Mach (constant g2_h), fixed mdot_ref, (eps, dp) from xflow.
cycle_fxd_power — mdot adjusted so P_shaft is constant; Mach and g2_h update via fig9.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xflow import calculate_pressure_drop_ratio

from cycle_assumptions import CycleAssumptions, RecuperatorInputs
from cycle_model import CycleSolution, solve_recuperated_cycle
from cycle_waterfall import CycleWaterfallBreakdown, waterfall_from_solution

# Match fig6c / fig9 geometric defaults
NTU_MATCH = 1.479
MACH_INDUSTRIAL = 0.11
GAMMA = 1.4
C_COLD_OVER_C_HOT = 1.0
ST_OVER_F = 0.4
F_C_OVER_F_H = 0.25
D_R = 0.25
A_R = 0.92
DP_MAX = 0.35
NTU_MAX_LENGTH = 30.0
NTU_SWEEP_N = 200
AO_OVER_AO_REF = 1.0
MDOT_MIN = 0.5
MDOT_MAX = 50.0


@dataclass(frozen=True)
class HexSweepPoint:
    """One point from a length or diffusion sweep."""

    ntu: float
    eps: float
    dp_hot_frac: float
    dp_cold_frac: float
    ao_over_ao_ref: float | None = None
    a_over_a_ref: float | None = None


@dataclass(frozen=True)
class CoupledCyclePoint:
    """Recuperated cycle + waterfall at one HEx sweep point."""

    hex_point: HexSweepPoint
    cycle: CycleSolution
    waterfall: CycleWaterfallBreakdown


@dataclass(frozen=True)
class LengtheningBootstrap:
    """Industrial / baseline design anchor for lengthening sweeps."""

    ntu_match: float
    mach_industrial: float
    g2_h_industrial: float
    pressure_drop_ratio: float
    mdot_ref_kg_per_s: float
    T_hot_in_ref_K: float
    eta_ref: float
    P_shaft_W: float


@dataclass(frozen=True)
class LengtheningCoupledPoint:
    """One point on a lengthening sweep with cycle coupling metadata."""

    ntu: float
    a_over_a_ref: float
    eps: float
    dp_hot_frac: float
    dp_cold_frac: float
    mdot_kg_per_s: float
    mach_in: float
    g2_h: float
    eta_cycle: float
    w_net_J_per_kg: float
    T_hot_in_K: float
    coupling: str


@dataclass(frozen=True)
class LengtheningEtaOptimum:
    """Result of cycle_fxd_power lengthening sweep and eta maximum search."""

    bootstrap: LengtheningBootstrap
    sweep: tuple[LengtheningCoupledPoint, ...]
    idx_industrial: int
    idx_eta_opt: int
    on_boundary: bool


def _import_fig9():
    try:
        import fig9_w_cycle_model as fig9
    except ImportError:
        old_scripts = Path(__file__).resolve().parent / "Old_figs" / "old_scripts"
        if old_scripts.is_dir():
            sys.path.insert(0, str(old_scripts))
        import fig9_w_cycle_model as fig9

    return fig9


def _fig9_pressure_drop_ratio(fig9) -> float:
    sigma_r = fig9.DEFAULT_D_R_hot * fig9.DEFAULT_A_R_hot
    return calculate_pressure_drop_ratio(
        fig9.DEFAULT_PRESSURE_DROP_ASSUMPTION,
        fig9.DEFAULT_C_COLD_OVER_C_HOT,
        fig9.DEFAULT_T,
        fig9.DEFAULT_D_R_hot,
        fig9.DEFAULT_MOLAR_MASS_RATIO,
        sigma_r,
        fig9.DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        f_c_over_f_h=fig9.DEFAULT_F_C_OVER_F_H,
    )


def bootstrap_industrial(fig9=None) -> LengtheningBootstrap:
    """Industrial design: NTU_MATCH, Ao=1, reference Mach; coupled mdot at P_shaft."""
    fig9 = fig9 or _import_fig9()
    pressure_drop_ratio = _fig9_pressure_drop_ratio(fig9)
    g2_h_ind = 0.5 * fig9.DEFAULT_GAMMA * fig9.DEFAULT_MACH_IN**2

    eps_ref, dp_h_ref, dp_c_ref = fig9._get_eps_dp_from_g2h(
        g2_h_ind,
        fig9.NTU_MATCH,
        fig9.DEFAULT_C_COLD_OVER_C_HOT,
        fig9.DEFAULT_ST_OVER_F,
        fig9.DEFAULT_F_C_OVER_F_H,
        fig9.DEFAULT_D_R_hot,
        pressure_drop_ratio,
    )
    _P, T_cyc, _s, _eff, w_net_ref = fig9.calculate_recuperated_cycle_dp_eps(
        fig9.PR,
        fig9.TIT,
        fig9.eta_poly_c,
        fig9.eta_poly_t,
        eps_ref,
        dp_h_ref,
        dp_c_ref,
    )
    T_hot_in_ref = T_cyc[4]
    mdot_ref = fig9.P_shaft_ref / w_net_ref

    mdot, w_net, T_hot_in, eps, dp_h, dp_c, _dq, eff_pct, M_in = fig9.solve_mdot_at_constant_power(
        AO_OVER_AO_REF,
        fig9.NTU_MATCH,
        pressure_drop_ratio,
        mdot_ref,
        T_hot_in_ref,
        fig9.P_shaft_ref,
    )
    if not np.isfinite(mdot) or not np.isfinite(eff_pct):
        raise RuntimeError("Industrial design coupled solve failed")

    return LengtheningBootstrap(
        ntu_match=float(fig9.NTU_MATCH),
        mach_industrial=float(fig9.DEFAULT_MACH_IN),
        g2_h_industrial=float(g2_h_ind),
        pressure_drop_ratio=float(pressure_drop_ratio),
        mdot_ref_kg_per_s=float(mdot_ref),
        T_hot_in_ref_K=float(T_hot_in_ref),
        eta_ref=float(eff_pct) / 100.0,
        P_shaft_W=float(fig9.P_shaft_ref),
    )


def _hex_at_ntu(
    fig9,
    ntu: float,
    g2_h: float,
    pressure_drop_ratio: float,
    dp_max: float = DP_MAX,
) -> tuple[float, float, float] | None:
    eps, dp_h, dp_c = fig9._get_eps_dp_from_g2h(
        g2_h,
        ntu,
        fig9.DEFAULT_C_COLD_OVER_C_HOT,
        fig9.DEFAULT_ST_OVER_F,
        fig9.DEFAULT_F_C_OVER_F_H,
        fig9.DEFAULT_D_R_hot,
        pressure_drop_ratio,
    )
    if not np.isfinite(eps) or dp_h >= dp_max or dp_c >= dp_max:
        return None
    return float(eps), float(dp_h), float(dp_c)


def evaluate_lengthening_fxd_mdot(
    ntu: float,
    bootstrap: LengtheningBootstrap,
    fig9=None,
    assumptions: CycleAssumptions | None = None,
) -> LengtheningCoupledPoint | None:
    """cycle_fxd_mdot: constant industrial Mach, fixed mdot_ref, lengthening geometry."""
    fig9 = fig9 or _import_fig9()
    hex_state = _hex_at_ntu(fig9, ntu, bootstrap.g2_h_industrial, bootstrap.pressure_drop_ratio)
    if hex_state is None:
        return None
    eps, dp_h, dp_c = hex_state

    recup = RecuperatorInputs(eps=eps, dp_hot_frac=dp_h, dp_cold_frac=dp_c)
    sol = solve_recuperated_cycle(
        recup,
        assumptions,
        mdot_kg_per_s=bootstrap.mdot_ref_kg_per_s,
    )
    if sol is None:
        return None

    return LengtheningCoupledPoint(
        ntu=float(ntu),
        a_over_a_ref=float(ntu / bootstrap.ntu_match),
        eps=eps,
        dp_hot_frac=dp_h,
        dp_cold_frac=dp_c,
        mdot_kg_per_s=bootstrap.mdot_ref_kg_per_s,
        mach_in=bootstrap.mach_industrial,
        g2_h=bootstrap.g2_h_industrial,
        eta_cycle=sol.eta_cycle,
        w_net_J_per_kg=sol.w_net_J_per_kg,
        T_hot_in_K=sol.hex_hot_inlet().T_K,
        coupling="fxd_mdot",
    )


def evaluate_lengthening_fxd_power(
    ntu: float,
    bootstrap: LengtheningBootstrap,
    fig9=None,
) -> LengtheningCoupledPoint | None:
    """cycle_fxd_power: solve mdot at P_shaft; Mach and dp follow from coupled state."""
    fig9 = fig9 or _import_fig9()
    mdot, w_net, T_hot_in, eps, dp_h, dp_c, _dq, eff_pct, M_in = fig9.solve_mdot_at_constant_power(
        AO_OVER_AO_REF,
        ntu,
        bootstrap.pressure_drop_ratio,
        bootstrap.mdot_ref_kg_per_s,
        bootstrap.T_hot_in_ref_K,
        bootstrap.P_shaft_W,
    )
    if not np.isfinite(mdot) or not np.isfinite(eff_pct) or eff_pct <= 0:
        return None
    if mdot < MDOT_MIN or mdot > MDOT_MAX:
        return None

    g2_h = 0.5 * fig9.DEFAULT_GAMMA * float(M_in) ** 2
    return LengtheningCoupledPoint(
        ntu=float(ntu),
        a_over_a_ref=float(ntu / bootstrap.ntu_match),
        eps=float(eps),
        dp_hot_frac=float(dp_h),
        dp_cold_frac=float(dp_c),
        mdot_kg_per_s=float(mdot),
        mach_in=float(M_in),
        g2_h=float(g2_h),
        eta_cycle=float(eff_pct) / 100.0,
        w_net_J_per_kg=float(w_net),
        T_hot_in_K=float(T_hot_in),
        coupling="fxd_power",
    )


def sweep_lengthening_fxd_power(
    bootstrap: LengtheningBootstrap,
    fig9=None,
    *,
    ntu_max: float = NTU_MAX_LENGTH,
    ntu_n: int = NTU_SWEEP_N,
) -> tuple[LengtheningCoupledPoint, ...]:
    """Sweep NTU at Ao=1 with cycle_fxd_power coupling."""
    fig9 = fig9 or _import_fig9()
    ntu_values = np.linspace(0.1, ntu_max, ntu_n)
    out: list[LengtheningCoupledPoint] = []
    for ntu in ntu_values:
        pt = evaluate_lengthening_fxd_power(float(ntu), bootstrap, fig9=fig9)
        if pt is not None:
            out.append(pt)
    return tuple(out)


def find_eta_optimum(
    sweep: tuple[LengtheningCoupledPoint, ...],
    bootstrap: LengtheningBootstrap,
) -> LengtheningEtaOptimum:
    """Maximum cycle efficiency on a cycle_fxd_power lengthening sweep."""
    if not sweep:
        raise RuntimeError("Empty lengthening sweep")

    eta_v = np.array([p.eta_cycle for p in sweep])
    idx_eta_opt = int(np.argmax(eta_v))

    ntu_v = np.array([p.ntu for p in sweep])
    idx_industrial = int(np.argmin(np.abs(ntu_v - bootstrap.ntu_match)))

    on_boundary = idx_eta_opt == 0 or idx_eta_opt == len(sweep) - 1
    return LengtheningEtaOptimum(
        bootstrap=bootstrap,
        sweep=sweep,
        idx_industrial=idx_industrial,
        idx_eta_opt=idx_eta_opt,
        on_boundary=on_boundary,
    )


def coupled_point_to_cycle_solution(
    point: LengtheningCoupledPoint,
    assumptions: CycleAssumptions | None = None,
) -> CycleSolution | None:
    """Build CycleSolution for waterfall from a lengthening coupled point."""
    recup = RecuperatorInputs(
        eps=point.eps,
        dp_hot_frac=point.dp_hot_frac,
        dp_cold_frac=point.dp_cold_frac,
    )
    if point.coupling == "fxd_mdot":
        return solve_recuperated_cycle(recup, assumptions, mdot_kg_per_s=point.mdot_kg_per_s)
    return solve_recuperated_cycle(recup, assumptions)


def coupled_point_to_waterfall(
    point: LengtheningCoupledPoint,
    assumptions: CycleAssumptions | None = None,
) -> CoupledCyclePoint | None:
    """Recuperated cycle + dimensional waterfall at one lengthening point."""
    sol = coupled_point_to_cycle_solution(point, assumptions)
    if sol is None:
        return None
    hex_pt = HexSweepPoint(
        ntu=point.ntu,
        eps=point.eps,
        dp_hot_frac=point.dp_hot_frac,
        dp_cold_frac=point.dp_cold_frac,
        ao_over_ao_ref=AO_OVER_AO_REF,
        a_over_a_ref=point.a_over_a_ref,
    )
    return CoupledCyclePoint(
        hex_point=hex_pt,
        cycle=sol,
        waterfall=waterfall_from_solution(sol),
    )


def evaluate_at_hex_point(
    point: HexSweepPoint,
    assumptions: CycleAssumptions | None = None,
) -> CoupledCyclePoint | None:
    """Solve recuperated cycle and waterfall for one (eps, dp) point."""
    recup = RecuperatorInputs(
        eps=point.eps,
        dp_hot_frac=point.dp_hot_frac,
        dp_cold_frac=point.dp_cold_frac,
    )
    sol = solve_recuperated_cycle(recup, assumptions)
    if sol is None:
        return None
    return CoupledCyclePoint(
        hex_point=point,
        cycle=sol,
        waterfall=waterfall_from_solution(sol),
    )


def evaluate_hex_sweep(
    points: list[HexSweepPoint],
    assumptions: CycleAssumptions | None = None,
) -> list[CoupledCyclePoint]:
    """Evaluate a list of sweep points; skip invalid cycle solutions."""
    out: list[CoupledCyclePoint] = []
    for pt in points:
        coupled = evaluate_at_hex_point(pt, assumptions)
        if coupled is not None:
            out.append(coupled)
    return out


def hex_point_from_arrays(
    ntu_v: np.ndarray,
    eps_v: np.ndarray,
    dp_h_v: np.ndarray,
    dp_c_v: np.ndarray,
    index: int,
    *,
    ao_over_ao_ref: float | None = None,
    a_over_a_ref: float | None = None,
) -> HexSweepPoint:
    """Build HexSweepPoint from fig6/fig7 sweep arrays at one index."""
    return HexSweepPoint(
        ntu=float(ntu_v[index]),
        eps=float(eps_v[index]),
        dp_hot_frac=float(dp_h_v[index]),
        dp_cold_frac=float(dp_c_v[index]),
        ao_over_ao_ref=ao_over_ao_ref,
        a_over_a_ref=a_over_a_ref,
    )


def print_lengthening_point(label: str, pt: LengtheningCoupledPoint, bootstrap: LengtheningBootstrap) -> None:
    """Print one lengthening coupled point vs industrial reference."""
    print(f"--- {label} ({pt.coupling}) ---")
    print(f"  NTU = {pt.ntu:.4f},  A/A_ref = {pt.a_over_a_ref:.4f}")
    print(f"  epsilon = {pt.eps:.4f}")
    print(f"  dp_h/p_hi = {pt.dp_hot_frac * 100:.2f} %,  dp_c/p_ci = {pt.dp_cold_frac * 100:.2f} %")
    print(f"  M_in = {pt.mach_in:.4f}  (industrial {bootstrap.mach_industrial:.4f},  delta = {pt.mach_in - bootstrap.mach_industrial:+.4f})")
    print(f"  mdot = {pt.mdot_kg_per_s:.4f} kg/s  (ref {bootstrap.mdot_ref_kg_per_s:.4f},  ratio = {pt.mdot_kg_per_s / bootstrap.mdot_ref_kg_per_s:.4f})")
    print(f"  eta_cycle = {pt.eta_cycle * 100:.2f} %  (ref {bootstrap.eta_ref * 100:.2f} %,  delta = {(pt.eta_cycle - bootstrap.eta_ref) * 100:+.2f} pp)")
    print(f"  T_hot,in = {pt.T_hot_in_K:.1f} K,  w_net = {pt.w_net_J_per_kg / 1e3:.2f} kJ/kg")


def print_local_pract_opt(bootstrap: LengtheningBootstrap) -> None:
    """Fig6 fixed-Mach practical optimum (fixed BC) for comparison — reuses ppt_values sweep."""
    import xflow
    from ppt_values import _compute_av_curves

    fig9 = _import_fig9()
    xflow.SHOW_CUBIC = False
    xflow.NTU_MATCH = None

    ntu_v, eps_v, dp_h_v, dp_c_v, av_v = _compute_av_curves(
        fig9.DEFAULT_T,
        fig9.DEFAULT_T_DEAD_OVER_T_COLD_IN,
        fig9.DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
        fig9.DEFAULT_P_DEAD_OVER_P_HOT_IN,
        bootstrap.g2_h_industrial,
        bootstrap.pressure_drop_ratio,
        ntu_max=NTU_MAX_LENGTH,
        dp_max=DP_MAX,
    )
    if len(ntu_v) < 1:
        print("--- local_pract_opt (fig6 fixed Mach) ---  no valid sweep points")
        return

    idx = int(np.argmax(av_v))
    on_boundary = idx == 0 or idx == len(ntu_v) - 1
    print("--- local_pract_opt (fig6 fixed Mach, max practical availability / fixed BC) ---")
    print(
        f"  NTU_opt = {ntu_v[idx]:.4f},  A/A_ref_opt = {ntu_v[idx] / bootstrap.ntu_match:.4f},  "
        f"epsilon_opt = {eps_v[idx]:.4f}"
    )
    print(f"  dp_h/p_h,in = {dp_h_v[idx] * 100:.2f} %,  dp_c/p_c,in = {dp_c_v[idx] * 100:.2f} %")
    print(f"  av_prac_total_opt / Q_max = {av_v[idx] * 100:.4f} %")
    if on_boundary:
        print("  Warning: local_pract_opt on sweep boundary.")


def print_eta_optimum_summary(result: LengtheningEtaOptimum) -> None:
    """Print industrial vs eta-optimum comparison for cycle_fxd_power lengthening."""
    b = result.bootstrap
    ind = evaluate_lengthening_fxd_power(b.ntu_match, b)
    if ind is None:
        raise RuntimeError("cycle_fxd_power failed at industrial NTU_MATCH")
    opt = result.sweep[result.idx_eta_opt]

    print("=" * 72)
    print("Lengthening sweep: cycle_fxd_power — cycle efficiency optimum")
    print("=" * 72)
    print(f"Industrial NTU_MATCH = {b.ntu_match:.4f},  P_shaft = {b.P_shaft_W / 1e3:.0f} kW")
    print()
    print_lengthening_point("Industrial design", ind, b)
    print()
    print_lengthening_point("Eta optimum", opt, b)
    if result.on_boundary:
        print()
        print("  Warning: eta optimum lies on sweep boundary — consider raising NTU_MAX or DP_MAX.")
    print("=" * 72)
