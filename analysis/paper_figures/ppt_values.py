"""
Presentation helper: single-point availability (Fig 4-style) + two sweeps.

- Baseline point: fixed epsilon; hot dp/p_in = DP_HOT_OF_INLET; cold dp from xflow assumption.
- Length sweep (Fig 6c): constant Mach, linear dp vs NTU, A/A_ref = NTU/NTU_ref.
- Diffusion sweep (Fig 7c): plot_unavailable_energy_breakdown, SHOW_CUBIC, fixed NTU_MATCH,
  A_o,ref/A_o = (NTU/NTU_MATCH)^0.704; matched designs on both sides of optimum.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xflow
from fig4_bar_chart import static_pressure_from_stagnation, static_temperature_from_stagnation
from xflow import (
    calculate_epsilon_ntu_curve,
    calculate_pressure_drop_ratio,
    classical_unavailable_creation_hex,
    plot_unavailable_energy_breakdown,
    practical_unavailable_creation_hex,
)

# --- Match fig6c / fig7c / fig8 geometric defaults ---
C_COLD_OVER_C_HOT = 1.0
ST_OVER_F = 0.4
F_C_OVER_F_H = 0.25
D_R = 0.25
A_R = 0.92
MOLAR_MASS_RATIO = 1.0
PRESSURE_DROP_ASSUMPTION = "inlet_density"  # (dp_c/p_c,in)/(dp_h/p_h,in) ~ 0.56–0.60
# PRESSURE_DROP_ASSUMPTION = "dp_c=dp_h"

# Fig 7: A_o propto NTU^-0.704 => A_o,ref/A_o = (NTU/NTU_ref)^AO_REF_OVER_AO_EXP
AO_REF_OVER_AO_EXP = 0.704

# --- New-case inlets (stagnation) ---
GAMMA = 1.4
T_CIN_STAG = 600.0  # K
T_HIN_STAG = 900.0  # K
T0_STAG = 288.0  # K
P0 = 1.0  # bar (dead)
P_CIN = 10.0  # bar stagnation at cold inlet
P_HIN = 1.1  # bar stagnation at hot inlet

MACH_H = 0.11
MACH_C = 0.05

# --- Point calculation ---
EPSILON_POINT = 0.6
DP_HOT_OF_INLET = 0.06

# --- Length sweep (fig 6c-style) ---
NTU_MAX_LENGTH = 40.0
DP_MAX_LENGTH = 0.35

# --- Diffusion sweep (fig 7c-style, same as fig7c_aspect_ratio.py) ---
FIG7_NTU_MATCH = 1.479  # reference design NTU; baseline + at Ao_ref/Ao = 1
NTU_MAX_DIFFUSION = 8.0
DP_MAX_DIFFUSION = 0.2

# fig9 cycle at A/A_ref = 1 (heat transfer area fixed); vary Ao/Ao_ref only
FIG9_AO_EXP = 0.587  # A/A_ref = (NTU / NTU_MATCH) * (Ao/Ao_ref)^FIG9_AO_EXP
AO_REF_OVER_AO_MATCHED = 0.7582
# Diffusion sweep (fig 7c) practical optimum at A/A_ref = 1 (from NTU_opt on sweep)
AO_REF_OVER_AO_DIFF_OPT = 0.8889
CP_HOT_CYCLE = 1070.0  # J/(kg·K), match fig9


@dataclass(frozen=True)
class Fig9Bootstrap:
    pressure_drop_ratio: float
    mdot_at_ref: float
    T_hot_in_ref: float
    mdot_fuel_baseline: float


@dataclass(frozen=True)
class MatchPoint:
    ntu: float
    x: float
    eps: float
    dp_h: float
    dp_c: float
    av: float


@dataclass(frozen=True)
class SweepResult:
    sweep_label: str
    sweep_kind: str  # "length" | "diffusion"
    ntu_v: np.ndarray
    eps_v: np.ndarray
    dp_h_v: np.ndarray
    dp_c_v: np.ndarray
    av_prac_v: np.ndarray
    ntu_ref: float
    x_v: np.ndarray
    idx_opt: int
    ntu_opt: float
    x_opt: float
    match_low: MatchPoint | None  # NTU < NTU_opt (lower A_o,ref/A_o on diffusion sweep)
    match_high: MatchPoint | None  # NTU > NTU_opt


@dataclass(frozen=True)
class Fig9CyclePoint:
    label: str
    ao_over_ao_ref: float
    ao_ref_over_ao: float
    ntu: float
    a_over_a_ref: float
    mdot: float
    eta_cycle: float
    eps: float
    dp_h: float
    dp_c: float
    M_in: float
    delta_fuel_kg: float
    T_hot_in: float
    T_cold_in: float
    Q_max_kW: float
    dq_over_qmax: float
    dq_kW: float
    T_cold_in_hex: float
    P_cold_in_hex_bar: float
    T_cold_out_hex: float
    P_cold_out_hex_bar: float
    T_hot_in_hex: float
    P_hot_in_hex_bar: float
    T_hot_out_hex: float
    P_hot_out_hex_bar: float


@dataclass(frozen=True)
class FixedBcHexPoint:
    """Fig 7c fixed-BC point at A/A_ref = 1 (no cycle coupling)."""

    ao_ref_over_ao: float
    ntu: float
    eps: float
    dp_h: float
    dp_c: float
    dq_over_qmax: float  # av_prac = -pu (positive = destruction fraction)
    thermal_hex_book: float  # ((p_d/p_h,out)^k - (p_d/p_c,o)^k) * epsilon


@dataclass(frozen=True)
class CycleWaterfallBreakdown:
    """Dimensionless shaft-work breakdown: thermal(fuel) - visc(rest) - visc(HEx) + thermal(HEx) = eta."""

    ao_ref_over_ao: float
    thermal_fuel: float
    visc_rest: float  # magnitude (printed with minus)
    visc_hex: float  # |visc_hex_signed|
    visc_hex_signed: float
    thermal_hex: float
    total: float  # P_shaft / Q_fuel = eta_cycle (fraction)


@dataclass(frozen=True)
class CycleHexOverPshaft:
    """HEx-only availability destruction, normalized by constant P_shaft (varying BC)."""

    ao_ref_over_ao: float
    thermal_hex: float  # ((p_d/p_h,out)^k - (p_d/p_c,o)^k) * Q_HEX / P_shaft
    visc_hex: float  # (dQ_o^M - pressure_term * Q_HEX) / P_shaft, signed
    dq_over_pshaft: float  # dQ_o^M / P_shaft (positive destruction)


def _import_fig9():
    try:
        import fig9_w_cycle_model as fig9
    except ImportError:
        _old_scripts = Path(__file__).resolve().parent / "Old_figs" / "old_scripts"
        if _old_scripts.is_dir():
            sys.path.insert(0, str(_old_scripts))
        import fig9_w_cycle_model as fig9

    return fig9


def evaluate_fixed_bc_diffusion_at_aoref(
    ao_ref_over_ao: float,
    ntu_v: np.ndarray,
    eps_v: np.ndarray,
    dp_h_v: np.ndarray,
    dp_c_v: np.ndarray,
    av_prac_v: np.ndarray,
    *,
    p_hin_static_bar: float,
    p_cin_static_bar: float,
) -> FixedBcHexPoint:
    """Fixed BC (fig 7c sweep) at A/A_ref = 1 for given A_o,ref/A_o."""
    ao_over_ao_ref = 1.0 / ao_ref_over_ao
    ntu = FIG7_NTU_MATCH / (ao_over_ao_ref**FIG9_AO_EXP)
    eps = float(np.interp(ntu, ntu_v, eps_v))
    dp_h = float(np.interp(ntu, ntu_v, dp_h_v))
    dp_c = float(np.interp(ntu, ntu_v, dp_c_v))
    dq_over_qmax = float(np.interp(ntu, ntu_v, av_prac_v))

    k = (GAMMA - 1.0) / GAMMA
    p_d_pa = P0 * 1e5
    p_hi_pa = p_hin_static_bar * 1e5
    p_ci_pa = p_cin_static_bar * 1e5
    p_h_out = p_hi_pa * (1.0 - dp_h)
    p_c_out = p_ci_pa * (1.0 - dp_c)
    thermal_hex_book = ((p_d_pa / p_h_out) ** k - (p_d_pa / p_c_out) ** k) * eps

    return FixedBcHexPoint(
        ao_ref_over_ao=ao_ref_over_ao,
        ntu=ntu,
        eps=eps,
        dp_h=dp_h,
        dp_c=dp_c,
        dq_over_qmax=dq_over_qmax,
        thermal_hex_book=thermal_hex_book,
    )


def _fig9_ntu_at_a_over_a_ref_one(ao_over_ao_ref: float) -> float:
    """NTU when A/A_ref = 1: NTU = NTU_MATCH / (Ao/Ao_ref)^FIG9_AO_EXP."""
    return FIG7_NTU_MATCH / (ao_over_ao_ref**FIG9_AO_EXP)


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


def _evaluate_fig9_cycle_point(
    fig9,
    *,
    label: str,
    ao_over_ao_ref: float,
    ao_ref_over_ao: float,
    pressure_drop_ratio: float,
    mdot_ref_val: float,
    T_hot_in_ref: float,
    mdot_fuel_baseline: float,
) -> Fig9CyclePoint | None:
    """Fully coupled fig9 point at A/A_ref = 1 for given Ao/Ao_ref."""
    ntu = _fig9_ntu_at_a_over_a_ref_one(ao_over_ao_ref)
    a_over_a_ref = (ntu / fig9.NTU_MATCH) * (ao_over_ao_ref**FIG9_AO_EXP)

    mdot, _w_net, T_hot_in, eps, dp_h, dp_c, dq_raw, eff, M_in = fig9.solve_mdot_at_constant_power(
        ao_over_ao_ref,
        ntu,
        pressure_drop_ratio,
        mdot_ref_val,
        T_hot_in_ref,
        fig9.P_shaft_ref,
    )
    if not np.isfinite(mdot) or not np.isfinite(eff) or eff <= 0:
        return None

    mdot_fuel = fig9.P_shaft_ref / (fig9.LHV_J_per_kg * eff / 100)
    delta_fuel_kg = (mdot_fuel - mdot_fuel_baseline) * fig9.mission_seconds

    gamma = fig9.DEFAULT_GAMMA
    T_cold_in = 288.0 * (fig9.PR ** ((gamma - 1) / (gamma * fig9.eta_poly_c)))
    Q_max_kW = mdot * CP_HOT_CYCLE * (T_hot_in - T_cold_in) / 1000.0

    t_act = T_hot_in / T_cold_in
    p_dead_act = 1.0 - dp_h
    p_cold_act = fig9.PR * (1.0 - dp_h)
    pu = practical_unavailable_creation_hex(
        np.array([eps]),
        t_act,
        np.array([dp_h]),
        np.array([dp_c]),
        np.array([True]),
        p_cold_in_over_p_hot_in=p_cold_act,
        p_dead_over_p_hot_in=p_dead_act,
        gamma=gamma,
    )
    dq_over_qmax = -float(np.asarray(pu, dtype=float).flat[0])
    dq_kW = dq_over_qmax * Q_max_kW

    P_cyc, T, _s, _eff_cyc, _w = fig9.calculate_recuperated_cycle_dp_eps(
        fig9.PR,
        fig9.TIT,
        fig9.eta_poly_c,
        fig9.eta_poly_t,
        eps,
        dp_h,
        dp_c,
    )
    pa_to_bar = 1e-5

    return Fig9CyclePoint(
        label=label,
        ao_over_ao_ref=ao_over_ao_ref,
        ao_ref_over_ao=ao_ref_over_ao,
        ntu=ntu,
        a_over_a_ref=a_over_a_ref,
        mdot=mdot,
        eta_cycle=eff,
        eps=eps,
        dp_h=dp_h,
        dp_c=dp_c,
        M_in=M_in,
        delta_fuel_kg=delta_fuel_kg,
        T_hot_in=T_hot_in,
        T_cold_in=T_cold_in,
        Q_max_kW=Q_max_kW,
        dq_over_qmax=dq_over_qmax,
        dq_kW=dq_kW,
        T_cold_in_hex=T[1],
        P_cold_in_hex_bar=P_cyc[1] * pa_to_bar,
        T_cold_out_hex=T[2],
        P_cold_out_hex_bar=P_cyc[2] * pa_to_bar,
        T_hot_in_hex=T[4],
        P_hot_in_hex_bar=P_cyc[4] * pa_to_bar,
        T_hot_out_hex=T[5],
        P_hot_out_hex_bar=P_cyc[5] * pa_to_bar,
    )


def _print_fig9_cycle_comparison(ref_pt: Fig9CyclePoint, matched_pt: Fig9CyclePoint) -> None:
    def _pt_block(pt: Fig9CyclePoint) -> None:
        print(f"--- {pt.label} ---")
        print(f"  A/A_ref = {pt.a_over_a_ref:.4f}, Ao/Ao_ref = {pt.ao_over_ao_ref:.4f}, A_o,ref/A_o = {pt.ao_ref_over_ao:.4f}")
        print(f"  NTU = {pt.ntu:.4f}, epsilon = {pt.eps:.4f}")
        print(f"  dp_h/p_h,in = {pt.dp_h * 100:.2f} %, dp_c/p_c,in = {pt.dp_c * 100:.2f} %")
        print(f"  eta_cycle = {pt.eta_cycle:.2f} %, mdot = {pt.mdot:.4f} kg/s, M_in,h = {pt.M_in:.4f}")
        print(f"  delta_m_f fuel only (cycle) = {pt.delta_fuel_kg:.2f} kg")
        print(
            f"  Hot in:  {pt.P_hot_in_hex_bar:.4f} bar, {pt.T_hot_in_hex:.2f} K; "
            f"Hot out: {pt.P_hot_out_hex_bar:.4f} bar, {pt.T_hot_out_hex:.2f} K"
        )
        print(
            f"  Cold in: {pt.P_cold_in_hex_bar:.4f} bar, {pt.T_cold_in_hex:.2f} K; "
            f"Cold out:{pt.P_cold_out_hex_bar:.4f} bar, {pt.T_cold_out_hex:.2f} K"
        )
        print(f"  Q_max = {pt.Q_max_kW:.2f} kW")
        print(f"  dQ_o^M/Q_max = {pt.dq_over_qmax * 100:.4f} %, dQ_o^M = {pt.dq_kW:.2f} kW")

    print()
    print("=" * 72)
    print("Fig9 coupled cycle at A/A_ref = 1 (two Ao/Ao_ref values)")
    print("=" * 72)
    _pt_block(ref_pt)
    print()
    _pt_block(matched_pt)
    print()
    print("--- Delta (matched - reference) ---")
    print(f"  delta_m_f fuel only = {matched_pt.delta_fuel_kg - ref_pt.delta_fuel_kg:.2f} kg")
    print(f"  Hot in:  dT = {matched_pt.T_hot_in_hex - ref_pt.T_hot_in_hex:.2f} K, dP = {(matched_pt.P_hot_in_hex_bar - ref_pt.P_hot_in_hex_bar) * 1e3:.1f} mbar")
    print(f"  Cold in: dT = {matched_pt.T_cold_in_hex - ref_pt.T_cold_in_hex:.2f} K, dP = {(matched_pt.P_cold_in_hex_bar - ref_pt.P_cold_in_hex_bar) * 1e3:.1f} mbar")
    print(f"  dQ_o^M/Q_max = {(matched_pt.dq_over_qmax - ref_pt.dq_over_qmax) * 100:.4f} %")
    print(f"  dQ_o^M = {matched_pt.dq_kW - ref_pt.dq_kW:.2f} kW")


def _cycle_hex_fluxes_w(pt: Fig9CyclePoint, fig9) -> tuple[float, float, float, float]:
    """Return (p_shaft_w, q_hex_w, pressure_term, dq_o_m_w) for HEx book-keeping."""
    gamma = fig9.DEFAULT_GAMMA
    k = (gamma - 1.0) / gamma
    p_d = 1e5  # Pa (cycle dead / ambient)
    p_cold_out = pt.P_cold_out_hex_bar * 1e5
    p_hot_out = pt.P_hot_out_hex_bar * 1e5
    q_hex_w = pt.mdot * CP_HOT_CYCLE * (pt.T_hot_in_hex - pt.T_hot_out_hex)
    pressure_term = (p_d / p_hot_out) ** k - (p_d / p_cold_out) ** k
    dq_o_m_w = pt.dq_kW * 1e3
    return fig9.P_shaft_ref, q_hex_w, pressure_term, dq_o_m_w


def compute_cycle_hex_over_pshaft(pt: Fig9CyclePoint, fig9) -> CycleHexOverPshaft:
    """HEx thermal + viscous destruction vs P_shaft at coupled cycle state."""
    p_shaft_w, q_hex_w, pressure_term, dq_o_m_w = _cycle_hex_fluxes_w(pt, fig9)
    thermal_hex = pressure_term * q_hex_w / p_shaft_w
    visc_hex = (dq_o_m_w - pressure_term * q_hex_w) / p_shaft_w
    return CycleHexOverPshaft(
        ao_ref_over_ao=pt.ao_ref_over_ao,
        thermal_hex=thermal_hex,
        visc_hex=visc_hex,
        dq_over_pshaft=dq_o_m_w / p_shaft_w,
    )


def _compute_cycle_waterfall(pt: Fig9CyclePoint, fig9) -> CycleWaterfallBreakdown:
    """
    Ideal work-potential breakdown (dimensionless vs Q_fuel).

    thermal_fuel = 1 - (p_dead/p_cold,out)^k
    thermal_hex = ((p_d/p_hot,out)^k - (p_d/p_cold,out)^k) * Q_HEX / Q_fuel
    visc_hex = (dQ_o^M - pressure_term * Q_HEX) / Q_fuel  (dQ_o^M = positive destruction, pt.dq_kW)
    visc_rest from closure: thermal_fuel + thermal_hex + visc_hex + visc_rest = eta_cycle
    """
    gamma = fig9.DEFAULT_GAMMA
    k = (gamma - 1.0) / gamma
    p_d = 1e5  # Pa (cycle dead / ambient)
    p_dead = p_d
    p_cold_out = pt.P_cold_out_hex_bar * 1e5

    thermal_fuel = 1.0 - (p_dead / p_cold_out) ** k

    q_fuel_w = pt.mdot * CP_HOT_CYCLE * (fig9.TIT - pt.T_cold_out_hex)
    _, q_hex_w, pressure_term, dq_o_m_w = _cycle_hex_fluxes_w(pt, fig9)
    thermal_hex = pressure_term * q_hex_w / q_fuel_w
    visc_hex_signed = (dq_o_m_w - pressure_term * q_hex_w) / q_fuel_w
    visc_hex = abs(visc_hex_signed)

    total = pt.eta_cycle / 100.0
    visc_rest_signed = total - thermal_fuel - thermal_hex - visc_hex_signed
    visc_rest = abs(visc_rest_signed)

    return CycleWaterfallBreakdown(
        ao_ref_over_ao=pt.ao_ref_over_ao,
        thermal_fuel=thermal_fuel,
        visc_rest=visc_rest,
        visc_hex=visc_hex,
        visc_hex_signed=visc_hex_signed,
        thermal_hex=thermal_hex,
        total=total,
    )


def _pct_pp(fraction: float) -> str:
    """Format dimensionless fraction as percentage points, e.g. 41.56%."""
    return f"{fraction * 100:.2f}%"


def _fig9_bootstrap(fig9) -> Fig9Bootstrap:
    pressure_drop_ratio = _fig9_pressure_drop_ratio(fig9)
    g2h_ref = 0.5 * fig9.DEFAULT_GAMMA * fig9.DEFAULT_MACH_IN**2
    eps_ref, dp_h_ref, dp_c_ref = fig9._get_eps_dp_from_g2h(
        g2h_ref,
        fig9.NTU_MATCH,
        fig9.DEFAULT_C_COLD_OVER_C_HOT,
        fig9.DEFAULT_ST_OVER_F,
        fig9.DEFAULT_F_C_OVER_F_H,
        fig9.DEFAULT_D_R_hot,
        pressure_drop_ratio,
    )
    _P, T_ref_cyc, _s, _eff_ref, w_net_ref = fig9.calculate_recuperated_cycle_dp_eps(
        fig9.PR,
        fig9.TIT,
        fig9.eta_poly_c,
        fig9.eta_poly_t,
        eps_ref,
        dp_h_ref,
        dp_c_ref,
    )
    T_hot_in_ref = T_ref_cyc[4]
    mdot_at_ref = fig9.P_shaft_ref / w_net_ref
    _P_b, _T_b, _s_b, eff_b, _w_b = fig9.calculate_cycle(
        fig9.PR, fig9.TIT, fig9.eta_poly_c, fig9.eta_poly_t
    )
    mdot_fuel_baseline = fig9.P_shaft_ref / (fig9.LHV_J_per_kg * eff_b / 100)
    return Fig9Bootstrap(pressure_drop_ratio, mdot_at_ref, T_hot_in_ref, mdot_fuel_baseline)


def evaluate_fig9_at_ao_ref_over_ao(
    ao_ref_over_ao: float,
    fig9,
    bootstrap: Fig9Bootstrap,
    *,
    label: str,
) -> Fig9CyclePoint | None:
    """Fully coupled fig9 point at A/A_ref = 1 for given A_o,ref/A_o."""
    ao_over_ao_ref = 1.0 / ao_ref_over_ao
    return _evaluate_fig9_cycle_point(
        fig9,
        label=label,
        ao_over_ao_ref=ao_over_ao_ref,
        ao_ref_over_ao=ao_ref_over_ao,
        pressure_drop_ratio=bootstrap.pressure_drop_ratio,
        mdot_ref_val=bootstrap.mdot_at_ref,
        T_hot_in_ref=bootstrap.T_hot_in_ref,
        mdot_fuel_baseline=bootstrap.mdot_fuel_baseline,
    )


def fetch_fig9_waterfall_at_ao_ref_over_ao(
    ao_ref_over_ao: float,
    fig9=None,
    bootstrap: Fig9Bootstrap | None = None,
    *,
    label: str | None = None,
) -> tuple[Fig9CyclePoint, CycleWaterfallBreakdown]:
    """Plug-in: cycle point + waterfall breakdown at A/A_ref=1 for one A_o,ref/A_o."""
    fig9 = fig9 or _import_fig9()
    bootstrap = bootstrap or _fig9_bootstrap(fig9)
    lbl = label or f"A_o,ref/A_o = {ao_ref_over_ao:.4f}"
    pt = evaluate_fig9_at_ao_ref_over_ao(ao_ref_over_ao, fig9, bootstrap, label=lbl)
    if pt is None:
        raise RuntimeError(f"fig9 cycle solve failed for A_o,ref/A_o = {ao_ref_over_ao}")
    return pt, _compute_cycle_waterfall(pt, fig9)


def _print_cycle_waterfall(
    bd: CycleWaterfallBreakdown,
    pt: Fig9CyclePoint,
    fig9,
    *,
    heading: str | None = None,
) -> None:
    p_shaft_kw = fig9.P_shaft_ref / 1e3
    dq_over_pshaft = pt.dq_kW / p_shaft_kw
    hex_net_alt = bd.thermal_hex + bd.visc_hex_signed
    dq_over_pshaft_eta = dq_over_pshaft * bd.total
    dq_over_qfuel = pt.dq_kW * 1e3 / (pt.mdot * CP_HOT_CYCLE * (fig9.TIT - pt.T_cold_out_hex))

    print(heading or f"--- A_o,ref/A_o = {bd.ao_ref_over_ao:.4f} ---")
    print(f"  thermal (fuel): + {_pct_pp(bd.thermal_fuel)}")
    print(f"  Visc (rest): - {_pct_pp(bd.visc_rest)}")
    visc_hex_sign = "-" if bd.visc_hex_signed < 0 else "+"
    print(f"  Visc (HEx): {visc_hex_sign} {_pct_pp(bd.visc_hex)}")
    print(f"  Thermal (HEx): + {_pct_pp(bd.thermal_hex)}")
    print(f"  Total: {_pct_pp(bd.total)}")
    print(
        f"  [check] dQ_o^M/P_shaft = {_pct_pp(dq_over_pshaft)},  "
        f"dQ_o^M/P_shaft*eta = {_pct_pp(dq_over_pshaft_eta)}"
    )
    print(
        f"  [check] Thermal(HEx)+Visc(HEx) signed = {_pct_pp(hex_net_alt)},  "
        f"dQ_o^M/Q_fuel = {_pct_pp(dq_over_qfuel)}"
    )


def _print_cycle_waterfall_section(
    ref_pt: Fig9CyclePoint,
    matched_pt: Fig9CyclePoint,
    fig9,
    *,
    diffusion_opt_pt: Fig9CyclePoint | None = None,
) -> None:
    print()
    print("=" * 72)
    print("Cycle shaft-work waterfall (% of Q_fuel, percentage points)")
    print("=" * 72)
    if diffusion_opt_pt is not None:
        print(
            f"  Optimal from diffusion sweep (fixed BC), "
            f"A_o,ref/A_o = {diffusion_opt_pt.ao_ref_over_ao:.4f}, A/A_ref = 1"
        )
        bd_opt = _compute_cycle_waterfall(diffusion_opt_pt, fig9)
        _print_cycle_waterfall(
            bd_opt,
            diffusion_opt_pt,
            fig9,
            heading=f"--- Diffusion optimum (A_o,ref/A_o = {diffusion_opt_pt.ao_ref_over_ao:.4f}) ---",
        )
        print()
    for pt in (ref_pt, matched_pt):
        bd = _compute_cycle_waterfall(pt, fig9)
        _print_cycle_waterfall(bd, pt, fig9)
        print()


def fetch_fig9_waterfall_pair() -> tuple[CycleWaterfallBreakdown, CycleWaterfallBreakdown, object]:
    """
    Reference (Ao/Ao_ref=1) and matched (A_o,ref/A_o=0.7582) waterfall breakdowns at A/A_ref=1.
    Returns (ref_breakdown, matched_breakdown, fig9_module).
    """
    fig9 = _import_fig9()
    bootstrap = _fig9_bootstrap(fig9)
    ref_pt = evaluate_fig9_at_ao_ref_over_ao(1.0, fig9, bootstrap, label="Reference")
    matched_pt = evaluate_fig9_at_ao_ref_over_ao(AO_REF_OVER_AO_MATCHED, fig9, bootstrap, label="Matched")
    if ref_pt is None or matched_pt is None:
        raise RuntimeError("fig9 cycle solve failed for reference or matched Ao point")

    return (
        _compute_cycle_waterfall(ref_pt, fig9),
        _compute_cycle_waterfall(matched_pt, fig9),
        fig9,
    )


def fetch_open_cycle_waterfall(fig9=None) -> CycleWaterfallBreakdown:
    """No-recuperator cycle: thermal(fuel) and visc(rest) only (HEx terms zero)."""
    fig9 = fig9 or _import_fig9()
    gamma = fig9.DEFAULT_GAMMA
    k = (gamma - 1.0) / gamma
    p_d = 1e5
    P_b, _T_b, _s_b, eff_b, _w_b = fig9.calculate_cycle(
        fig9.PR, fig9.TIT, fig9.eta_poly_c, fig9.eta_poly_t
    )
    p_cold_in = P_b[1]
    thermal_fuel = 1.0 - (p_d / p_cold_in) ** k
    total = eff_b / 100.0
    visc_hex_signed = 0.0
    visc_rest_signed = total - thermal_fuel
    return CycleWaterfallBreakdown(
        ao_ref_over_ao=0.0,
        thermal_fuel=thermal_fuel,
        visc_rest=abs(visc_rest_signed),
        visc_hex=0.0,
        visc_hex_signed=0.0,
        thermal_hex=0.0,
        total=total,
    )


def _run_fig9_two_ao_points() -> tuple[Fig9CyclePoint, Fig9CyclePoint] | None:
    fig9 = _import_fig9()
    bootstrap = _fig9_bootstrap(fig9)

    diffusion_opt_pt = evaluate_fig9_at_ao_ref_over_ao(
        AO_REF_OVER_AO_DIFF_OPT,
        fig9,
        bootstrap,
        label=f"Diffusion optimum (A_o,ref/A_o = {AO_REF_OVER_AO_DIFF_OPT})",
    )
    ref_pt = evaluate_fig9_at_ao_ref_over_ao(
        1.0,
        fig9,
        bootstrap,
        label="Reference (Ao/Ao_ref = 1, A_o,ref/A_o = 1)",
    )
    matched_pt = evaluate_fig9_at_ao_ref_over_ao(
        AO_REF_OVER_AO_MATCHED,
        fig9,
        bootstrap,
        label=f"Matched Ao (A_o,ref/A_o = {AO_REF_OVER_AO_MATCHED}, A/A_ref = 1)",
    )
    if ref_pt is None or matched_pt is None:
        print("Warning: fig9 cycle solve failed for one or both Ao points.")
        return None
    _print_fig9_cycle_comparison(ref_pt, matched_pt)
    _print_cycle_waterfall_section(ref_pt, matched_pt, fig9, diffusion_opt_pt=diffusion_opt_pt)
    return ref_pt, matched_pt


def baseline_point():
    """Fig 4-style totals at EPSILON_POINT; cold dp matches sweep assumption (ratio * dp_hot)."""
    p_hin = P_HIN
    t0_static = static_temperature_from_stagnation(T0_STAG, 0.0, GAMMA)
    t_cin_static = static_temperature_from_stagnation(T_CIN_STAG, MACH_C, GAMMA)
    p_hin_static = static_pressure_from_stagnation(p_hin, MACH_H, GAMMA)
    p_cin_static = static_pressure_from_stagnation(P_CIN, MACH_C, GAMMA)

    t_ratio_stag = T_HIN_STAG / T_CIN_STAG
    t_dead_over_t_cold_in = t0_static / t_cin_static
    p_cold_in_over_p_hot_in = p_cin_static / p_hin_static
    p_dead_over_p_hot_in = P0 / p_hin_static

    sigma_r = D_R * A_R
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        PRESSURE_DROP_ASSUMPTION,
        C_COLD_OVER_C_HOT,
        t_ratio_stag,
        D_R,
        MOLAR_MASS_RATIO,
        sigma_r,
        p_cold_in_over_p_hot_in,
        f_c_over_f_h=F_C_OVER_F_H,
    )
    dp_hot = float(DP_HOT_OF_INLET)
    dp_cold = float(pressure_drop_ratio * dp_hot)
    if dp_cold < 0.0 or dp_cold >= 1.0:
        raise ValueError(
            f"Derived cold dp/p_in = {dp_cold:.4g} invalid; check DP_HOT_OF_INLET and PRESSURE_DROP_ASSUMPTION"
        )

    eps_a = np.atleast_1d(EPSILON_POINT)
    dp_h = np.atleast_1d(dp_hot)
    dp_c = np.atleast_1d(dp_cold)
    mask = np.ones_like(eps_a, dtype=bool)

    av_class_total = float(
        np.atleast_1d(
            -classical_unavailable_creation_hex(
                eps_a,
                t_ratio_stag,
                dp_h,
                dp_c,
                mask,
                t_dead_over_t_cold_in=t_dead_over_t_cold_in,
                gamma=GAMMA,
            )
        ).flat[0]
    )
    av_prac_total = float(
        np.atleast_1d(
            -practical_unavailable_creation_hex(
                eps_a,
                t_ratio_stag,
                dp_h,
                dp_c,
                mask,
                p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
                p_dead_over_p_hot_in=p_dead_over_p_hot_in,
                gamma=GAMMA,
            )
        ).flat[0]
    )

    return {
        "t_ratio_stag": t_ratio_stag,
        "t_dead_over_t_cold_in": t_dead_over_t_cold_in,
        "p_cold_in_over_p_hot_in": p_cold_in_over_p_hot_in,
        "p_dead_over_p_hot_in": p_dead_over_p_hot_in,
        "T0_static": t0_static,
        "T_cin_static": t_cin_static,
        "P_hin_static": p_hin_static,
        "P_cin_static": p_cin_static,
        "P_hin_stag": p_hin,
        "pressure_drop_ratio": pressure_drop_ratio,
        "dp_hot_point": dp_hot,
        "dp_cold_point": dp_cold,
        "av_class_total": av_class_total,
        "av_prac_total": av_prac_total,
    }


def _interp_x_at_y(x: np.ndarray, y: np.ndarray, y_target: float) -> float:
    """Linear interpolation: find x where y(x) = y_target."""
    if len(x) < 2:
        raise ValueError("Need at least two points for interpolation")
    dy = y - y_target
    for i in range(len(dy) - 1):
        if dy[i] == 0:
            return float(x[i])
        if dy[i] * dy[i + 1] < 0:
            t = -dy[i] / (dy[i + 1] - dy[i])
            return float(x[i] + t * (x[i + 1] - x[i]))
    raise ValueError(f"No bracket for y_target={y_target} in y range [{y.min()}, {y.max()}]")


def _compute_av_curves(
    t_ratio_stag: float,
    t_dead_over_t_cold_in: float,
    p_cold_in_over_p_hot_in: float,
    p_dead_over_p_hot_in: float,
    g2_h: float,
    pressure_drop_ratio: float,
    ntu_max: float,
    dp_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ntu_v, eps_v, dp_h_v, dp_c_v, av_prac_v) on valid mask."""
    ntu, epsilon, dp_hot, dp_cold, validity_mask = calculate_epsilon_ntu_curve(
        C_COLD_OVER_C_HOT,
        ST_OVER_F,
        F_C_OVER_F_H,
        D_R,
        g2_h,
        ntu_max=ntu_max,
        dp_max=dp_max,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )
    ntu_v = ntu[validity_mask]
    eps_v = epsilon[validity_mask]
    dp_h_v = dp_hot[validity_mask]
    dp_c_v = dp_cold[validity_mask]

    pu = practical_unavailable_creation_hex(
        epsilon,
        t_ratio_stag,
        dp_hot,
        dp_cold,
        validity_mask,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=GAMMA,
    )
    av_prac_v = -np.asarray(pu, dtype=float)
    return ntu_v, eps_v, dp_h_v, dp_c_v, av_prac_v


def _sweep_length(
    t_ratio_stag: float,
    t_dead_over_t_cold_in: float,
    p_cold_in_over_p_hot_in: float,
    p_dead_over_p_hot_in: float,
    g2_h: float,
    pressure_drop_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    xflow.SHOW_CUBIC = False
    xflow.NTU_MATCH = None
    ntu_v, eps_v, dp_h_v, dp_c_v, av_prac_v = _compute_av_curves(
        t_ratio_stag,
        t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in,
        g2_h,
        pressure_drop_ratio,
        ntu_max=NTU_MAX_LENGTH,
        dp_max=DP_MAX_LENGTH,
    )
    ntu_ref = _interp_x_at_y(ntu_v, eps_v, EPSILON_POINT)
    return ntu_v, eps_v, dp_h_v, dp_c_v, av_prac_v, ntu_ref


def _sweep_diffusion_fig7c(
    t_ratio_stag: float,
    t_dead_over_t_cold_in: float,
    p_cold_in_over_p_hot_in: float,
    p_dead_over_p_hot_in: float,
    g2_h: float,
    pressure_drop_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Same data path as fig7c_aspect_ratio.save_figures (practical breakdown line).
    """
    xflow.SHOW_CUBIC = True
    xflow.NTU_MATCH = FIG7_NTU_MATCH

    fig, ax = plt.subplots()
    _, line_with_dp, _ax = plot_unavailable_energy_breakdown(
        C_COLD_OVER_C_HOT,
        ST_OVER_F,
        F_C_OVER_F_H,
        D_R,
        g2_h,
        ntu_max=NTU_MAX_DIFFUSION,
        dp_max=DP_MAX_DIFFUSION,
        ax=ax,
        framework="practical",
        t=t_ratio_stag,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=GAMMA,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )
    plt.close(fig)

    ntu_v = np.asarray(line_with_dp.get_xdata(), dtype=float)
    av_prac_v = -np.asarray(line_with_dp.get_ydata(), dtype=float)

    ntu_all, eps_all, dp_h_all, dp_c_all, validity_mask = calculate_epsilon_ntu_curve(
        C_COLD_OVER_C_HOT,
        ST_OVER_F,
        F_C_OVER_F_H,
        D_R,
        g2_h,
        ntu_max=NTU_MAX_DIFFUSION,
        dp_max=DP_MAX_DIFFUSION,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )
    ntu_curve = ntu_all[validity_mask]
    eps_v = np.interp(ntu_v, ntu_curve, eps_all[validity_mask])
    dp_h_v = np.interp(ntu_v, ntu_curve, dp_h_all[validity_mask])
    dp_c_v = np.interp(ntu_v, ntu_curve, dp_c_all[validity_mask])

    if len(ntu_v) < 3:
        raise RuntimeError("Diffusion sweep produced too few valid NTU points")
    return ntu_v, eps_v, dp_h_v, dp_c_v, av_prac_v, FIG7_NTU_MATCH


def _x_coord_length(ntu: np.ndarray, ntu_ref: float) -> np.ndarray:
    return ntu / ntu_ref


def _x_coord_diffusion(ntu: np.ndarray, ntu_ref: float) -> np.ndarray:
    return (ntu / ntu_ref) ** AO_REF_OVER_AO_EXP


def _find_match_on_branch(
    ntu_v: np.ndarray,
    av_prac_v: np.ndarray,
    ntu_opt: float,
    target_av: float,
    *,
    branch: str,
) -> float | None:
    """Interpolate NTU where av_prac crosses target on NTU < or > NTU_opt."""
    if branch == "low":
        sel = ntu_v < ntu_opt
    elif branch == "high":
        sel = ntu_v > ntu_opt
    else:
        raise ValueError(f"branch must be 'low' or 'high', got {branch!r}")
    branch_ntu = ntu_v[sel]
    branch_av = av_prac_v[sel]
    if len(branch_ntu) < 2:
        return None
    diff = branch_av - target_av
    for i in range(len(diff) - 1):
        if diff[i] == 0:
            return float(branch_ntu[i])
        if diff[i] * diff[i + 1] < 0:
            t = -diff[i] / (diff[i + 1] - diff[i])
            return float(branch_ntu[i] + t * (branch_ntu[i + 1] - branch_ntu[i]))
    return None


def _match_point_from_ntu(
    ntu_match: float,
    ntu_v: np.ndarray,
    eps_v: np.ndarray,
    dp_h_v: np.ndarray,
    dp_c_v: np.ndarray,
    av_prac_v: np.ndarray,
    x_v: np.ndarray,
) -> MatchPoint:
    return MatchPoint(
        ntu=ntu_match,
        x=float(np.interp(ntu_match, ntu_v, x_v)),
        eps=float(np.interp(ntu_match, ntu_v, eps_v)),
        dp_h=float(np.interp(ntu_match, ntu_v, dp_h_v)),
        dp_c=float(np.interp(ntu_match, ntu_v, dp_c_v)),
        av=float(np.interp(ntu_match, ntu_v, av_prac_v)),
    )


def analyze_sweep(
    sweep_label: str,
    sweep_kind: str,
    ntu_v: np.ndarray,
    eps_v: np.ndarray,
    dp_h_v: np.ndarray,
    dp_c_v: np.ndarray,
    av_prac_v: np.ndarray,
    ntu_ref: float,
    target_av: float,
) -> SweepResult:
    if sweep_kind == "length":
        if eps_v.min() > EPSILON_POINT or eps_v.max() < EPSILON_POINT:
            raise RuntimeError(
                f"{sweep_label}: EPSILON_POINT={EPSILON_POINT} outside sweep "
                f"[{eps_v.min():.4f}, {eps_v.max():.4f}]"
            )
        x_v = _x_coord_length(ntu_v, ntu_ref)
    elif sweep_kind == "diffusion":
        x_v = _x_coord_diffusion(ntu_v, ntu_ref)
    else:
        raise ValueError(f"Unknown sweep_kind: {sweep_kind}")

    idx_opt = int(np.nanargmax(av_prac_v))
    if idx_opt == 0 or idx_opt == len(av_prac_v) - 1:
        print(f"Warning [{sweep_label}]: practical optimum on sweep boundary")

    ntu_opt = float(ntu_v[idx_opt])
    x_opt = float(x_v[idx_opt])

    ntu_low = _find_match_on_branch(ntu_v, av_prac_v, ntu_opt, target_av, branch="low")
    ntu_high = _find_match_on_branch(ntu_v, av_prac_v, ntu_opt, target_av, branch="high")
    match_low = (
        _match_point_from_ntu(ntu_low, ntu_v, eps_v, dp_h_v, dp_c_v, av_prac_v, x_v) if ntu_low is not None else None
    )
    match_high = (
        _match_point_from_ntu(ntu_high, ntu_v, eps_v, dp_h_v, dp_c_v, av_prac_v, x_v)
        if ntu_high is not None
        else None
    )

    return SweepResult(
        sweep_label=sweep_label,
        sweep_kind=sweep_kind,
        ntu_v=ntu_v,
        eps_v=eps_v,
        dp_h_v=dp_h_v,
        dp_c_v=dp_c_v,
        av_prac_v=av_prac_v,
        ntu_ref=ntu_ref,
        x_v=x_v,
        idx_opt=idx_opt,
        ntu_opt=ntu_opt,
        x_opt=x_opt,
        match_low=match_low,
        match_high=match_high,
    )


def _print_sweep_header(result: SweepResult, g2_h: float, pressure_drop_ratio: float, ntu_max: float):
    print(f"--- {result.sweep_label} ---")
    if result.sweep_kind == "length":
        print("  Model: constant A_o (Mach), linear dp vs NTU (SHOW_CUBIC off)")
        print(f"  NTU in [0.1, {ntu_max:.1f}]; coordinate A/A_ref = NTU/NTU_ref")
    else:
        print("  Model: fig7c_aspect_ratio (plot_unavailable_energy_breakdown, practical)")
        print("  SHOW_CUBIC on; NTU_MATCH fixed (fig7 reference design)")
        print(
            f"  NTU in [0.1, {ntu_max:.1f}]; A_o,ref/A_o = (NTU/NTU_MATCH)^{AO_REF_OVER_AO_EXP}; "
            f"NTU_MATCH = {FIG7_NTU_MATCH}"
        )
        eps_at_ref = float(np.interp(FIG7_NTU_MATCH, result.ntu_v, result.eps_v))
        print(f"  At NTU_MATCH (+ marker): epsilon = {eps_at_ref:.4f}, A_o,ref/A_o = 1")
    print(f"  (dp_c/p_c,in)/(dp_h/p_h,in) = {pressure_drop_ratio:.6f}; g2_h = {g2_h:.4e}")
    if result.sweep_kind == "length":
        print(f"  NTU_ref (epsilon = {EPSILON_POINT}) = {result.ntu_ref:.4f}")


def _print_sweep_optimum(result: SweepResult):
    i = result.idx_opt
    x_label = "A/A_ref" if result.sweep_kind == "length" else "A_o,ref/A_o"
    print(f"--- Optimum [{result.sweep_label}] (max practical availability) ---")
    print(
        f"  NTU_opt = {result.ntu_opt:.4f}, {x_label}_opt = {result.x_opt:.4f}, "
        f"epsilon_opt = {result.eps_v[i]:.4f}"
    )
    print(
        f"  dp_h/p_h,in = {result.dp_h_v[i] * 100:.2f} %, dp_c/p_c,in = {result.dp_c_v[i] * 100:.2f} %"
    )
    print(f"  av_prac_total_opt / Q_max = {result.av_prac_v[i] * 100:.4f} %")


def _print_match_point(
    title: str,
    mp: MatchPoint | None,
    x_label: str,
    target_av: float,
):
    if mp is None:
        print(f"--- {title}: NOT FOUND ---")
        print(f"  Target av_prac / Q_max = {target_av * 100:.4f} %")
        return
    print(f"--- {title} ---")
    print(f"  NTU = {mp.ntu:.4f}, {x_label} = {mp.x:.4f}, epsilon = {mp.eps:.4f}")
    print(f"  dp_h/p_h,in = {mp.dp_h * 100:.2f} %, dp_c/p_c,in = {mp.dp_c * 100:.2f} %")
    print(f"  av_prac_total / Q_max = {mp.av * 100:.4f} % (target {target_av * 100:.4f} %)")


def _print_sweep_matched(result: SweepResult, target_av: float, ntu_max: float, dp_max: float):
    x_label = "A/A_ref" if result.sweep_kind == "length" else "A_o,ref/A_o"
    if result.sweep_kind == "length":
        _print_match_point(
            f"Matched design [{result.sweep_label}] (baseline av_prac, NTU > NTU_opt / higher {x_label})",
            result.match_high,
            x_label,
            target_av,
        )
        if result.match_high is None:
            print(f"  Try raising NTU_MAX or DP_MAX (currently {ntu_max}, {dp_max}).")
        return
    _print_match_point(
        f"Matched design [{result.sweep_label}] (baseline av_prac, lower {x_label}, NTU < NTU_opt)",
        result.match_low,
        x_label,
        target_av,
    )
    print()
    _print_match_point(
        f"Matched design [{result.sweep_label}] (baseline av_prac, higher {x_label}, NTU > NTU_opt)",
        result.match_high,
        x_label,
        target_av,
    )
    if result.match_low is None and result.match_high is None:
        print(f"  Try raising NTU_MAX or DP_MAX (currently {ntu_max}, {dp_max}).")


def main():
    base = baseline_point()
    target_av = base["av_prac_total"]
    g2_h = 0.5 * GAMMA * MACH_H**2
    pressure_drop_ratio = base["pressure_drop_ratio"]

    ntu_l, eps_l, dp_h_l, dp_c_l, av_l, ntu_ref_l = _sweep_length(
        base["t_ratio_stag"],
        base["t_dead_over_t_cold_in"],
        base["p_cold_in_over_p_hot_in"],
        base["p_dead_over_p_hot_in"],
        g2_h,
        pressure_drop_ratio,
    )
    length = analyze_sweep(
        "Length sweep (fig 6c)",
        "length",
        ntu_l,
        eps_l,
        dp_h_l,
        dp_c_l,
        av_l,
        ntu_ref_l,
        target_av,
    )

    ntu_d, eps_d, dp_h_d, dp_c_d, av_d, ntu_ref_d = _sweep_diffusion_fig7c(
        base["t_ratio_stag"],
        base["t_dead_over_t_cold_in"],
        base["p_cold_in_over_p_hot_in"],
        base["p_dead_over_p_hot_in"],
        g2_h,
        pressure_drop_ratio,
    )
    diffusion = analyze_sweep(
        "Diffusion sweep (fig 7c)",
        "diffusion",
        ntu_d,
        eps_d,
        dp_h_d,
        dp_c_d,
        av_d,
        ntu_ref_d,
        target_av,
    )

    print("=" * 72)
    print("ppt_values: presentation numbers")
    print("=" * 72)
    print(f"Inlets (stagnation): T_h,in = {T_HIN_STAG:.0f} K, T_c,in = {T_CIN_STAG:.0f} K")
    print(f"Mach: M_h = {MACH_H:.3f}, M_c = {MACH_C:.3f}; gamma = {GAMMA}")
    print(f"P_c,in (stagnation) = {P_CIN:.1f} bar; P_h,in (stagnation) = {P_HIN:.1f} bar")
    print(f"  baseline dp_h/p_h,in = {DP_HOT_OF_INLET:.3f}")
    print(f"Dead state: T0 = {T0_STAG:.0f} K, P0 = {P0:.1f} bar")
    print(f"Static cold inlet: T = {base['T_cin_static']:.2f} K, P = {base['P_cin_static']:.4f} bar")
    print(
        f"Static hot inlet:  T = {static_temperature_from_stagnation(T_HIN_STAG, MACH_H, GAMMA):.2f} K, "
        f"P = {base['P_hin_static']:.4f} bar"
    )
    print(
        f"t = T_h,stag/T_c,stag = {base['t_ratio_stag']:.4f}; "
        f"T0_static/T_c,static = {base['t_dead_over_t_cold_in']:.4f}"
    )
    print(
        f"p_c,in/p_h,in (static) = {base['p_cold_in_over_p_hot_in']:.4f}; "
        f"p_dead/p_h,in = {base['p_dead_over_p_hot_in']:.4f}"
    )
    print()
    print(
        f"--- Baseline point (fixed epsilon; dp split = {PRESSURE_DROP_ASSUMPTION!r}, "
        f"(dp_c/p_c,in)/(dp_h/p_h,in) = {pressure_drop_ratio:.6f}) ---"
    )
    print(
        f"epsilon = {EPSILON_POINT}; dp_h/p_h,in = {base['dp_hot_point']:.4f}; "
        f"dp_c/p_c,in = {base['dp_cold_point']:.4f}"
    )
    print(f"av_class_total / Q_max = {base['av_class_total'] * 100:.4f} %")
    print(f"av_prac_total / Q_max  = {base['av_prac_total'] * 100:.4f} %")
    print()

    _print_sweep_header(length, g2_h, pressure_drop_ratio, NTU_MAX_LENGTH)
    print()
    _print_sweep_optimum(length)
    print()
    _print_sweep_matched(length, target_av, NTU_MAX_LENGTH, DP_MAX_LENGTH)
    print()

    _print_sweep_header(diffusion, g2_h, pressure_drop_ratio, NTU_MAX_DIFFUSION)
    print()
    _print_sweep_optimum(diffusion)
    print()
    _print_sweep_matched(diffusion, target_av, NTU_MAX_DIFFUSION, DP_MAX_DIFFUSION)

    _run_fig9_two_ao_points()


if __name__ == "__main__":
    main()
