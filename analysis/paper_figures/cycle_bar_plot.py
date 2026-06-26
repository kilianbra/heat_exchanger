"""Waterfall bar-chart plotting for dimensional cycle availability breakdowns [kW]."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from cycle_assumptions import RecuperatorInputs, RecupHexGeometry, mach_ratio_first_order
from cycle_waterfall import CycleWaterfallBreakdown
from plot_colors import COLOR_THERMAL, COLOR_VISC_HOT, color_with_alpha

GREEN_HEX = "#57724A"
BLACK_BAR = "black"
CYCLE_ALPHA_LIGHT = 0.1
COLOR_THERMAL_FUEL = color_with_alpha(COLOR_THERMAL, CYCLE_ALPHA_LIGHT)
COLOR_VISC_COMP = color_with_alpha(COLOR_VISC_HOT, CYCLE_ALPHA_LIGHT)
COLOR_VISC_TURB = COLOR_VISC_HOT
THERMAL_HATCH = "///"
EDGE = "black"
EDGE_LW = 0.6
ARROW_LW = 0.8

# Single horizontal scale: bars, gaps, x margins, and panel figure width all scale together.
LAYOUT_X_SCALE = 0.6
_BAR_WIDTH_REF = 0.38
_BAR_GAP_REF = 0.55
_X_FIRST_REF = 0.35
_X_MAX_BUFFER_FRAC = 0.6
_FIG_WIDTH_RECUP_REF = 3.8

BAR_WIDTH = _BAR_WIDTH_REF * LAYOUT_X_SCALE
BAR_GAP = _BAR_GAP_REF * LAYOUT_X_SCALE
X_FIRST = _X_FIRST_REF * LAYOUT_X_SCALE
X_MAX_BUFFER = BAR_GAP * _X_MAX_BUFFER_FRAC
FIG_WIDTH_RECUP = _FIG_WIDTH_RECUP_REF * LAYOUT_X_SCALE
FIG_HEIGHT = 2.8
PANEL_WSPACE = 0.12  # gap between left/right subplots (lower = closer panels)
PANEL_WIDTH_PAD_IN = 0.2 * LAYOUT_X_SCALE  # extra figure width [in] beyond the two panels
VALUE_FS = 7  # font size for bar value labels
XLABEL_FS = 7  # font size for x-axis component labels
TITLE_FS = 8
VALUE_PAD = 0.012
Y_LIMIT_BUFFER_FRAC = 0.10
SUBPLOT_ADJ = dict(left=0.18, right=0.95, bottom=0.20, top=0.82)
SUBPLOT_ADJ_HEAT_WORK = dict(left=0.07, right=0.98, bottom=0.18, top=0.74)
SUBPLOT_ADJ_NRG_PRAC = dict(left=0.08, right=0.98, bottom=0.18, top=0.82)
LEGEND_FS = 6
LEGEND_HANDLELENGTH = 2.2
LEGEND_HANDLEHEIGHT = 1.35
YLABEL = "Practical availability [kW]"

LABELS_NOREC = [
    r"$\dot{Q}_{in}$",
    r"$\dot{\Phi}_{comp}$",
    r"$\dot{\Phi}_{turb}$",
    r"$\dot{W}_{net}$",
]
LABELS_RECUP = [
    r"$\dot{Q}_{in}$",
    r"$\dot{Q}_{rec}$",
    r"$\dot{\Phi}_{rec}$",
    r"$\dot{\Phi}_{comp}$",
    r"$\dot{\Phi}_{turb}$",
    r"$\dot{W}_{net}$",
]
LABELS_WORK_WATERFALL_OPEN = [
    r"$\dot{Q}_{in}$",
    r"$\dot{\Phi}_{\mathrm{turbom}}$",
    r"$\dot{W}_{net}$",
]
LABELS_WORK_WATERFALL_RECUP = [
    r"$\dot{Q}_{in}$",
    r"$\dot{Q}_{rec}$",
    r"$\dot{\Phi}_{rec}$",
    r"$\dot{\Phi}_{\mathrm{turbom}}$",
    r"$\dot{W}_{net}$",
]
REFERENCE_N_SLOTS = len(LABELS_RECUP)
REF_PRAC_STEPS = len(LABELS_WORK_WATERFALL_RECUP) - 1
REF_ENERGY_SLOTS = 2
REF_HEAT_INPUT_SLOTS = 1
WORK_WATERFALL_SLOTS = len(LABELS_WORK_WATERFALL_RECUP)
PLACEHOLDER_BAR_FRAC = 0.018
HEAT_INPUT_YLABEL = "Heat [kW]"
HEAT_WORK_YLABEL = "Work [kW]"
HEAT_WORK_OUT_STEM = "bar_heat_work"
NRG_PRAC_AV_OUT_STEM = "bar_nrg_prac_av"
ENERGY_YLABEL = "Energy [kW]"
LABEL_ENERGY_WX = r"$\dot{W}_{x}$"
QIN_ONE_DECIMAL_SUFFIXES = frozenset({"rec_length_opt_fix_Pnet", "rec_fix_mass_opt"})


def cycle_plot_title(
    bd: CycleWaterfallBreakdown,
    recup: RecuperatorInputs | None = None,
    *,
    geom: RecupHexGeometry | None = None,
    mdot_ref_kg_per_s: float | None = None,
    case_label: str | None = None,
) -> str:
    """Multi-line LaTeX title: case label; optional geometry; HEx and cycle parameters."""
    eta_pct = bd.eta_cycle * 100
    mdot = bd.mdot_kg_per_s
    hex_line = (
        (
            rf"$\varepsilon = {recup.eps * 100:.2f}\,\%$, "
            rf"$\Delta p/p_{{hi}} = {recup.dp_hot_frac * 100:.2f}\,\%$, "
            rf"$\Delta p/p_{{ci}} = {recup.dp_cold_frac * 100:.2f}\,\%$, "
            rf"$\eta_{{cycle}} = {eta_pct:.2f}\,\%$"
        )
        if recup is not None
        else (
            rf"$\eta_{{cycle}} = {eta_pct:.2f}\,\%$, "
            rf"$\dot{{m}} = {mdot:.3f}\,\mathrm{{kg/s}}$, "
            rf"$\dot{{W}}_{{net}} = {bd.w_net_J_per_kg / 1e3:.1f}\,\mathrm{{kJ/kg}}$"
        )
    )

    if recup is None:
        line1 = case_label if case_label else "Open cycle"
        return f"{line1}\n{hex_line}"

    line1 = case_label if case_label else "Recuperated cycle"
    lines = [line1]

    if geom is not None:
        ao_ref_over_ao = geom.ao_ref_over_ao
        geo_parts: list[str] = []
        if abs(geom.a_over_a_ref - 1.0) > 1e-4:
            geo_parts.extend(
                [
                    rf"$A/A_{{\mathrm{{ref}}}} = {geom.a_over_a_ref:.3f}$",
                    rf"$A_{{o,\mathrm{{ref}}}}/A_o = {ao_ref_over_ao:.3f}$",
                ]
            )
            if mdot_ref_kg_per_s is not None:
                m_ratio = mach_ratio_first_order(mdot, mdot_ref_kg_per_s, ao_ref_over_ao)
                geo_parts.append(rf"$M/M_{{\mathrm{{ref}}}} = {m_ratio:.3f}$")
        elif abs(geom.ao_over_ao_ref - 1.0) > 1e-4:
            geo_parts.append(rf"$A_{{o,\mathrm{{ref}}}}/A_o = {ao_ref_over_ao:.3f}$")
        if geo_parts:
            lines.append(", ".join(geo_parts))

    lines.append(hex_line)
    return "\n".join(lines)


def lengthening_waterfall_title(case_label: str, pt, recup: RecuperatorInputs, bd: CycleWaterfallBreakdown) -> str:
    """Title for lengthening coupled waterfall: geometry, Mach, HEx, and eta (two lines)."""
    eta_pct = bd.eta_cycle * 100
    line1 = (
        f"{case_label}  "
        rf"$A/A_{{\mathrm{{ref}}}}={pt.a_over_a_ref:.3f}$, "
        rf"$\mathrm{{NTU}}={pt.ntu:.3f}$, "
        rf"$M={pt.mach_in:.4f}$"
    )
    line2 = (
        rf"$\varepsilon={recup.eps * 100:.2f}\,\%$, "
        rf"$\Delta p/p_{{hi}}={recup.dp_hot_frac * 100:.2f}\,\%$, "
        rf"$\Delta p/p_{{ci}}={recup.dp_cold_frac * 100:.2f}\,\%$, "
        rf"$\eta_{{cycle}}={eta_pct:.2f}\,\%$"
    )
    return f"{line1}\n{line2}"


def figure_title_two_lines(title: str) -> str:
    """Collapse a multi-line case title to exactly two lines for figure suptitles."""
    parts = [p.strip() for p in title.split("\n") if p.strip()]
    if len(parts) <= 2:
        return "\n".join(parts)
    return f"{parts[0]}  {parts[1]}\n{parts[2]}"


def qin_one_decimal_for_suffix(stem_suffix: str | None) -> bool:
    return stem_suffix in QIN_ONE_DECIMAL_SUFFIXES


def _format_step_value_label(delta_kw: float, *, one_decimal: bool = False) -> str:
    if one_decimal:
        return f"+{delta_kw:.1f}" if delta_kw >= 0 else f"{delta_kw:.1f}"
    return f"+{delta_kw:.0f}" if delta_kw >= 0 else f"{delta_kw:.0f}"


def _set_heat_work_figure_title(fig: plt.Figure, title: str) -> None:
    fig.suptitle(figure_title_two_lines(title), fontsize=TITLE_FS, y=0.97)


def _x_layout(n_slots: int) -> tuple[list[float], float]:
    """Bar centres and xmax (with trailing buffer) for n_slots bars."""
    x_slot = [X_FIRST + i * BAR_GAP for i in range(n_slots)]
    x_max = x_slot[-1] + X_MAX_BUFFER
    return x_slot, x_max


@lru_cache(maxsize=1)
def _reference_axis_x_max() -> tuple[float, float, float]:
    """Fixed x-limits per panel type (widest case in each chart class)."""
    _, work_x_max = _x_layout(WORK_WATERFALL_SLOTS)
    _, prac_x_max = _x_layout(REF_PRAC_STEPS)
    _, energy_x_max = _x_layout(REF_ENERGY_SLOTS)
    return work_x_max, prac_x_max, energy_x_max


def _reference_x_slot(n_slots: int) -> list[float]:
    x_slot, _ = _x_layout(n_slots)
    return x_slot


def _heat_input_axis_limits() -> tuple[float, float]:
    """x centre and x_max for the single-bar heat-input panel (compact, like nrg/prac energy)."""
    x_slot, x_max = _x_layout(REF_HEAT_INPUT_SLOTS)
    return x_slot[0], x_max


@dataclass(frozen=True)
class _WaterfallDrawSlot:
    step: WorkWaterfallStep | None
    label: str
    x: float
    is_total: bool = False
    qin_one_decimal: bool = False


def _work_waterfall_draw_slots(
    bd: CycleWaterfallBreakdown,
    steps: list[WorkWaterfallStep],
    bar_labels: list[str],
    *,
    qin_one_decimal: bool = False,
) -> list[_WaterfallDrawSlot]:
    """Fixed recuperator slot layout; open cycles get empty placeholders at Q_rec and Phi_rec."""
    x_slot = _reference_x_slot(WORK_WATERFALL_SLOTS)
    if bd.recuperated:
        slots = [
            _WaterfallDrawSlot(steps[i], bar_labels[i], x_slot[i], qin_one_decimal=qin_one_decimal and i == 0)
            for i in range(len(steps))
        ]
        slots.append(_WaterfallDrawSlot(None, bar_labels[-1], x_slot[-1], is_total=True))
        return slots

    return [
        _WaterfallDrawSlot(steps[0], bar_labels[0], x_slot[0], qin_one_decimal=qin_one_decimal),
        _WaterfallDrawSlot(None, LABELS_WORK_WATERFALL_RECUP[1], x_slot[1]),
        _WaterfallDrawSlot(None, LABELS_WORK_WATERFALL_RECUP[2], x_slot[2]),
        _WaterfallDrawSlot(steps[1], bar_labels[1], x_slot[3]),
        _WaterfallDrawSlot(None, bar_labels[-1], x_slot[4], is_total=True),
    ]


def _prac_av_draw_slots(
    bd: CycleWaterfallBreakdown,
    steps: list[WorkWaterfallStep],
    step_labels: list[str],
    *,
    qin_one_decimal: bool = False,
) -> list[_WaterfallDrawSlot]:
    """Mirrored prac-av slots (turbom left, Q_in right); open cycles get empty recup placeholders."""
    x_slot = _reference_x_slot(REF_PRAC_STEPS)
    mirrored_labels = list(reversed(LABELS_WORK_WATERFALL_RECUP[:-1]))
    if bd.recuperated:
        return [
            _WaterfallDrawSlot(
                steps[REF_PRAC_STEPS - 1 - i],
                mirrored_labels[i],
                x_slot[i],
                qin_one_decimal=qin_one_decimal and (REF_PRAC_STEPS - 1 - i) == 0,
            )
            for i in range(REF_PRAC_STEPS)
        ]

    return [
        _WaterfallDrawSlot(steps[1], mirrored_labels[0], x_slot[0]),
        _WaterfallDrawSlot(None, mirrored_labels[1], x_slot[1]),
        _WaterfallDrawSlot(None, mirrored_labels[2], x_slot[2]),
        _WaterfallDrawSlot(steps[0], mirrored_labels[3], x_slot[3], qin_one_decimal=qin_one_decimal),
    ]


def _heat_work_figure_widths() -> tuple[float, float, float]:
    """Return (heat panel, work panel, total) figure widths [in] for the heat/work class."""
    work_x_max, _, _ = _reference_axis_x_max()
    _, heat_x_max = _x_layout(REF_HEAT_INPUT_SLOTS)
    work_w = FIG_WIDTH_RECUP
    heat_w = FIG_WIDTH_RECUP * heat_x_max / work_x_max
    total = heat_w + work_w + PANEL_WIDTH_PAD_IN
    return heat_w, work_w, total


def _nrg_prac_figure_widths() -> tuple[float, float, float]:
    """Return (energy panel, prac-av panel, total) figure widths [in] for the nrg/prac class."""
    _, prac_x_max, energy_x_max = _reference_axis_x_max()
    prac_w = FIG_WIDTH_RECUP
    energy_w = FIG_WIDTH_RECUP * energy_x_max / prac_x_max
    total = energy_w + prac_w + PANEL_WIDTH_PAD_IN
    return energy_w, prac_w, total


def _figsize_for_x_max(x_max: float) -> tuple[float, float]:
    """Scale figure width with xmax so on-screen bar width matches recuperated chart."""
    _, x_max_ref = _x_layout(REFERENCE_N_SLOTS)
    width = FIG_WIDTH_RECUP * x_max / x_max_ref
    return width, FIG_HEIGHT


def _peak_kw(bd: CycleWaterfallBreakdown, steps: list[WaterfallStep]) -> float:
    cumul = 0.0
    y_max = max(bd.dwm_qin_kw, bd.dwm_qrec_kw + bd.dwm_qin_kw if bd.recuperated else bd.dwm_qin_kw, bd.p_shaft_kw)
    for step in steps:
        cumul += step.delta_kw
        y_max = max(y_max, cumul)
        if step.delta_kw < 0:
            y_max = max(y_max, cumul + step.delta_kw)
    return y_max


@dataclass(frozen=True)
class _BarSpec:
    x: float
    bottom: float
    height: float
    width: float
    color: str | tuple[float, float, float, float]
    hatch: str = ""


@dataclass(frozen=True)
class WorkWaterfallStep:
    """One step on the work availability waterfall."""

    delta_kw: float
    kind: Literal["thermal", "viscous"]


@dataclass(frozen=True)
class WaterfallStep:
    delta_kw: float
    color: str | tuple[float, float, float, float]


def open_cycle_steps(bd: CycleWaterfallBreakdown) -> list[WaterfallStep]:
    """dWM_qin up; dWM_comp and dWM_turb down (= -dQ0M_v each)."""
    return [
        WaterfallStep(bd.dwm_qin_kw, COLOR_THERMAL_FUEL),
        WaterfallStep(bd.dwm_comp_kw, COLOR_VISC_COMP),
        WaterfallStep(bd.dwm_turb_kw, COLOR_VISC_TURB),
    ]


def recuperated_cycle_steps(bd: CycleWaterfallBreakdown) -> list[WaterfallStep]:
    """dWM_qin and dWM_qrec up; dWM_v_rec, dWM_comp, dWM_turb down (= -dQ0M_v each)."""
    return [
        WaterfallStep(bd.dwm_qin_kw, COLOR_THERMAL_FUEL),
        WaterfallStep(bd.dwm_qrec_kw, COLOR_THERMAL),
        WaterfallStep(bd.dwm_v_rec_kw, COLOR_VISC_HOT),
        WaterfallStep(bd.dwm_comp_kw, COLOR_VISC_COMP),
        WaterfallStep(bd.dwm_turb_kw, COLOR_VISC_TURB),
    ]


def open_cycle_work_waterfall(bd: CycleWaterfallBreakdown) -> list[WorkWaterfallStep]:
    """Work-axis waterfall: dWM_qin up; combined turbomachinery viscous down."""
    return [
        WorkWaterfallStep(bd.dwm_qin_kw, "thermal"),
        WorkWaterfallStep(bd.dwm_comp_kw + bd.dwm_turb_kw, "viscous"),
    ]


def recuperated_cycle_work_waterfall(bd: CycleWaterfallBreakdown) -> list[WorkWaterfallStep]:
    """Work-axis waterfall: thermal up; viscous down (Phi_rec, Phi_turbom)."""
    return [
        WorkWaterfallStep(bd.dwm_qin_kw, "thermal"),
        WorkWaterfallStep(bd.dwm_qrec_kw, "thermal"),
        WorkWaterfallStep(bd.dwm_v_rec_kw, "viscous"),
        WorkWaterfallStep(bd.dwm_comp_kw + bd.dwm_turb_kw, "viscous"),
    ]


def work_waterfall_for_breakdown(bd: CycleWaterfallBreakdown) -> tuple[list[WorkWaterfallStep], list[str]]:
    if bd.recuperated:
        return recuperated_cycle_work_waterfall(bd), LABELS_WORK_WATERFALL_RECUP
    return open_cycle_work_waterfall(bd), LABELS_WORK_WATERFALL_OPEN


def _work_waterfall_peak_kw(bd: CycleWaterfallBreakdown, steps: list[WorkWaterfallStep]) -> float:
    cumul = 0.0
    y_max = bd.p_shaft_kw
    for step in steps:
        cumul += step.delta_kw
        y_max = max(y_max, cumul)
        if step.delta_kw < 0:
            y_max = max(y_max, cumul + step.delta_kw)
    return y_max


def _collect_heat_work_breakdowns():
    """Yield all breakdowns used for shared heat/work axis limits."""
    from cycle_assumptions import DEFAULT_CYCLE, RECUP_PPT_CASES
    from cycle_hex_coupling import (
        bootstrap_industrial,
        coupled_point_to_waterfall,
        evaluate_lengthening_fxd_mdot,
        evaluate_lengthening_fxd_power,
        find_eta_optimum,
        sweep_lengthening_fxd_power,
    )
    from cycle_model import solve_open_cycle, solve_recuperated_cycle
    from cycle_waterfall import waterfall_from_solution

    yield waterfall_from_solution(solve_open_cycle(DEFAULT_CYCLE))

    for case in RECUP_PPT_CASES:
        rec_sol = solve_recuperated_cycle(case.recup, DEFAULT_CYCLE)
        if rec_sol is not None:
            yield waterfall_from_solution(rec_sol)

    bootstrap = bootstrap_industrial()
    sweep = sweep_lengthening_fxd_power(bootstrap)
    if len(sweep) >= 3:
        result = find_eta_optimum(sweep, bootstrap)
        opt_fxp = result.sweep[result.idx_eta_opt]
        opt_fxd = evaluate_lengthening_fxd_mdot(opt_fxp.ntu, bootstrap)
        for pt in (opt_fxp, opt_fxd):
            coupled = coupled_point_to_waterfall(pt, DEFAULT_CYCLE)
            if coupled is not None:
                yield coupled.waterfall


@lru_cache(maxsize=1)
def _shared_heat_work_panel_y_limits() -> tuple[float, float]:
    """Single y span for heat-input, work, and placeholder axes across all bar_heat_work charts."""
    peaks: list[float] = []
    for bd in _collect_heat_work_breakdowns():
        peaks.append(bd.Q_in_kw)
        steps, _ = work_waterfall_for_breakdown(bd)
        peaks.append(_work_waterfall_peak_kw(bd, steps))
    y_max = max(peaks) * (1.0 + Y_LIMIT_BUFFER_FRAC)
    y_min = -y_max * Y_LIMIT_BUFFER_FRAC
    return y_min, y_max


@lru_cache(maxsize=1)
def _shared_y_limits() -> tuple[float, float]:
    """Identical y-axis span for open and recuperated charts."""
    from cycle_assumptions import DEFAULT_CYCLE, RECUP_PPT_CASES
    from cycle_model import solve_open_cycle, solve_recuperated_cycle
    from cycle_waterfall import waterfall_from_solution

    peaks: list[float] = []
    open_bd = waterfall_from_solution(solve_open_cycle(DEFAULT_CYCLE))
    peaks.append(_peak_kw(open_bd, open_cycle_steps(open_bd)))

    for case in RECUP_PPT_CASES:
        rec_sol = solve_recuperated_cycle(case.recup, DEFAULT_CYCLE)
        if rec_sol is not None:
            rec_bd = waterfall_from_solution(rec_sol)
            peaks.append(_peak_kw(rec_bd, recuperated_cycle_steps(rec_bd)))

    y_max = max(peaks) * (1.0 + Y_LIMIT_BUFFER_FRAC)
    y_min = -y_max * Y_LIMIT_BUFFER_FRAC
    return y_min, y_max


def _delta_bar_geom(cumul: float, delta: float) -> tuple[float, float]:
    if delta >= 0:
        return cumul, delta
    return cumul + delta, -delta


def _draw_bar(ax: plt.Axes, spec: _BarSpec, *, zorder: int = 3) -> None:
    ax.bar(
        spec.x,
        spec.height,
        width=spec.width,
        bottom=spec.bottom,
        color=spec.color,
        hatch=spec.hatch,
        edgecolor=EDGE,
        linewidth=EDGE_LW,
        zorder=zorder,
    )


def _thermal_bar_spec(x: float, bottom: float, height: float) -> _BarSpec:
    return _BarSpec(
        x=x,
        bottom=bottom,
        height=height,
        width=BAR_WIDTH,
        color=COLOR_THERMAL,
        hatch=THERMAL_HATCH,
    )


def _viscous_bar_spec(x: float, bottom: float, height: float) -> _BarSpec:
    return _BarSpec(
        x=x,
        bottom=bottom,
        height=height,
        width=BAR_WIDTH,
        color=COLOR_VISC_HOT,
    )


def _black_bar_spec(x: float, bottom: float, height: float) -> _BarSpec:
    return _BarSpec(
        x=x,
        bottom=bottom,
        height=height,
        width=BAR_WIDTH,
        color=BLACK_BAR,
    )


def _placeholder_bar_spec(x: float, y_lo: float, y_hi: float) -> _BarSpec:
    """Empty white bar occupying a recuperator slot in open-cycle charts."""
    return _BarSpec(
        x=x,
        bottom=y_lo,
        height=(y_hi - y_lo) * PLACEHOLDER_BAR_FRAC,
        width=BAR_WIDTH,
        color="white",
    )


def _draw_placeholder_bar(ax: plt.Axes, slot: _WaterfallDrawSlot, *, y_lo: float, y_hi: float) -> _BarSpec:
    spec = _placeholder_bar_spec(slot.x, y_lo, y_hi)
    _draw_bar(ax, spec)
    _draw_xlabel(ax, slot.x, slot.label, y_lo)
    return spec


def _bar_top(spec: _BarSpec) -> float:
    return spec.bottom + spec.height


def _draw_value_label(ax: plt.Axes, spec: _BarSpec, text: str, y_lo: float) -> None:
    """Draw numerical value just above the bar top, or below for very small bars."""
    ax.text(
        spec.x,
        _bar_top(spec) + VALUE_PAD,
        text,
        ha="center",
        va="bottom",
        fontsize=VALUE_FS,
        color="black",
        zorder=5,
    )


def _draw_xlabel(ax: plt.Axes, x: float, label: str, y_lo: float) -> None:
    """Draw component label below the x-axis."""
    ax.text(
        x,
        y_lo - VALUE_PAD * 6,
        label,
        ha="center",
        va="top",
        fontsize=XLABEL_FS,
        color="black",
        zorder=5,
    )


def _draw_connector(ax: plt.Axes, left: _BarSpec, right: _BarSpec, y: float, *, zorder: int = 4) -> None:
    ax.plot(
        [left.x + left.width / 2, right.x - right.width / 2],
        [y, y],
        "k:",
        linewidth=EDGE_LW,
        zorder=zorder,
    )


def _style_axes_arrows(
    ax: plt.Axes,
    *,
    y_lo: float,
    y_hi: float,
    x_max: float,
    ylabel: str | None = None,
) -> None:
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelleft=False, length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["left"].set_position(("data", 0.0))
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(EDGE_LW)
        ax.spines[spine].set_color("black")

    dy = y_hi - y_lo
    ax.annotate(
        "",
        xy=(0.0, y_hi + 0.04 * dy),
        xytext=(0.0, y_hi - 0.08 * dy),
        xycoords="data",
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", lw=ARROW_LW),
        annotation_clip=False,
    )
    dx = x_max
    ax.annotate(
        "",
        xy=(x_max + 0.02 * dx, 0.0),
        xytext=(x_max - 0.08 * dx, 0.0),
        xycoords="data",
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", lw=ARROW_LW),
        annotation_clip=False,
    )


def _work_step_bar_spec(x: float, bottom: float, height: float, step: WorkWaterfallStep) -> _BarSpec:
    if step.kind == "thermal":
        return _thermal_bar_spec(x, bottom, height)
    return _viscous_bar_spec(x, bottom, height)


def _add_waterfall_legend(ax: plt.Axes) -> None:
    handles = [
        mpatches.Patch(
            facecolor=COLOR_THERMAL,
            edgecolor=EDGE,
            hatch=THERMAL_HATCH,
            label="Thermal",
        ),
        mpatches.Patch(
            facecolor=COLOR_VISC_HOT,
            edgecolor=EDGE,
            label="Viscous",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=LEGEND_FS,
        frameon=False,
        handlelength=LEGEND_HANDLELENGTH,
        handleheight=LEGEND_HANDLEHEIGHT,
        borderpad=0.4,
        labelspacing=0.35,
    )


def _draw_heat_input_axis(
    ax: plt.Axes,
    bd: CycleWaterfallBreakdown,
    *,
    y_lo: float,
    y_hi: float,
) -> None:
    """Combustor heat rate Q_in [kW] on a compact single-bar axis."""
    x, heat_x_max = _heat_input_axis_limits()
    spec = _black_bar_spec(x, 0.0, bd.Q_in_kw)
    _draw_bar(ax, spec)
    _draw_value_label(ax, spec, f"{bd.Q_in_kw:.0f}", y_lo)
    _draw_xlabel(ax, x, r"$\dot{Q}_{in}$", y_lo)
    _style_axes_arrows(
        ax,
        y_lo=y_lo,
        y_hi=y_hi,
        x_max=heat_x_max,
        ylabel=HEAT_INPUT_YLABEL,
    )


def _draw_work_waterfall(
    ax: plt.Axes,
    bd: CycleWaterfallBreakdown,
    *,
    steps: list[WorkWaterfallStep],
    bar_labels: list[str],
    y_lo: float,
    y_hi: float,
    qin_one_decimal: bool = False,
) -> None:
    """Original availability waterfall on the work axis."""
    work_x_max, _, _ = _reference_axis_x_max()
    draw_slots = _work_waterfall_draw_slots(bd, steps, bar_labels, qin_one_decimal=qin_one_decimal)

    cumul = 0.0
    prev: _BarSpec | None = None

    for slot in draw_slots:
        if slot.is_total:
            total_spec = _black_bar_spec(slot.x, 0.0, bd.p_shaft_kw)
            if prev is not None:
                _draw_connector(ax, prev, total_spec, y=cumul)
            _draw_bar(ax, total_spec)
            _draw_value_label(ax, total_spec, f"{bd.p_shaft_kw:.0f}", y_lo)
            _draw_xlabel(ax, slot.x, slot.label, y_lo)
            continue

        if slot.step is None:
            _draw_placeholder_bar(ax, slot, y_lo=y_lo, y_hi=y_hi)
            continue

        bottom, height = _delta_bar_geom(cumul, slot.step.delta_kw)
        spec = _work_step_bar_spec(slot.x, bottom, height, slot.step)
        if prev is not None:
            _draw_connector(ax, prev, spec, y=cumul)
        _draw_bar(ax, spec)
        lbl = _format_step_value_label(
            slot.step.delta_kw,
            one_decimal=slot.qin_one_decimal,
        )
        _draw_value_label(ax, spec, lbl, y_lo)
        _draw_xlabel(ax, slot.x, slot.label, y_lo)
        cumul += slot.step.delta_kw
        prev = spec

    _style_axes_arrows(ax, y_lo=y_lo, y_hi=y_hi, x_max=work_x_max, ylabel=HEAT_WORK_YLABEL)
    _add_waterfall_legend(ax)


def _draw_prac_av_waterfall(
    ax: plt.Axes,
    bd: CycleWaterfallBreakdown,
    *,
    steps: list[WorkWaterfallStep],
    bar_labels: list[str],
    y_lo: float,
    y_hi: float,
    qin_one_decimal: bool = False,
) -> None:
    """Mirrored availability waterfall: turbom loss on the left, Q_in on the right."""
    _, prac_x_max, _ = _reference_axis_x_max()
    draw_slots = _prac_av_draw_slots(bd, steps, bar_labels, qin_one_decimal=qin_one_decimal)

    real_forward = [slot for slot in reversed(draw_slots) if slot.step is not None]
    geoms: dict[float, tuple[float, float]] = {}
    conn_levels: dict[float, float] = {}
    cumul = 0.0
    for slot in real_forward:
        bottom, height = _delta_bar_geom(cumul, slot.step.delta_kw)
        geoms[slot.x] = (bottom, height)
        conn_levels[slot.x] = cumul
        cumul += slot.step.delta_kw

    prev: _BarSpec | None = None
    for slot in real_forward:
        bottom, height = geoms[slot.x]
        spec = _work_step_bar_spec(slot.x, bottom, height, slot.step)
        if prev is not None:
            _draw_connector(ax, prev, spec, y=conn_levels[slot.x])
        _draw_bar(ax, spec)
        lbl = _format_step_value_label(slot.step.delta_kw, one_decimal=slot.qin_one_decimal)
        _draw_value_label(ax, spec, lbl, y_lo)
        _draw_xlabel(ax, slot.x, slot.label, y_lo)
        prev = spec

    for slot in draw_slots:
        if slot.step is None:
            _draw_placeholder_bar(ax, slot, y_lo=y_lo, y_hi=y_hi)

    _style_axes_arrows(ax, y_lo=y_lo, y_hi=y_hi, x_max=prac_x_max, ylabel=YLABEL)
    _add_waterfall_legend(ax)


def _draw_energy_axis(
    ax: plt.Axes,
    bd: CycleWaterfallBreakdown,
    *,
    y_lo: float,
    y_hi: float,
) -> None:
    """Combustor heat input and net shaft work on the energy axis [kW]."""
    _, _, energy_x_max = _reference_axis_x_max()
    x_positions = _reference_x_slot(REF_ENERGY_SLOTS)

    q_spec = _black_bar_spec(x_positions[0], 0.0, bd.Q_in_kw)
    _draw_bar(ax, q_spec)
    _draw_value_label(ax, q_spec, f"{bd.Q_in_kw:.0f}", y_lo)
    _draw_xlabel(ax, x_positions[0], r"$\dot{Q}_{in}$", y_lo)

    w_spec = _black_bar_spec(x_positions[1], 0.0, bd.p_shaft_kw)
    _draw_bar(ax, w_spec)
    _draw_value_label(ax, w_spec, f"{bd.p_shaft_kw:.0f}", y_lo)
    _draw_xlabel(ax, x_positions[1], LABEL_ENERGY_WX, y_lo)

    _style_axes_arrows(ax, y_lo=y_lo, y_hi=y_hi, x_max=energy_x_max, ylabel=ENERGY_YLABEL)


def fig_cycle_waterfall(
    bd: CycleWaterfallBreakdown,
    *,
    steps: list[WaterfallStep],
    bar_labels: list[str],
    title: str | None = None,
) -> plt.Figure:
    """Draw the waterfall.  bar_labels must have len(steps)+1 entries (last = total)."""
    n_slots = len(steps) + 1
    x_slot, x_max = _x_layout(n_slots)
    y_lo, y_hi = _shared_y_limits()

    fig, ax = plt.subplots(figsize=_figsize_for_x_max(x_max), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cumul = 0.0
    prev: _BarSpec | None = None

    for i, step in enumerate(steps):
        x = x_slot[i]
        bottom, height = _delta_bar_geom(cumul, step.delta_kw)
        spec = _BarSpec(x=x, bottom=bottom, height=height, width=BAR_WIDTH, color=step.color)
        if prev is not None:
            _draw_connector(ax, prev, spec, y=cumul)
        _draw_bar(ax, spec)
        lbl = f"+{step.delta_kw:.0f}" if step.delta_kw >= 0 else f"{step.delta_kw:.0f}"
        _draw_value_label(ax, spec, lbl, y_lo)
        _draw_xlabel(ax, x, bar_labels[i], y_lo)
        cumul += step.delta_kw
        prev = spec

    total_x = x_slot[-1]
    total_spec = _BarSpec(x=total_x, bottom=0.0, height=bd.p_shaft_kw, width=BAR_WIDTH, color=GREEN_HEX)
    if prev is not None:
        _draw_connector(ax, prev, total_spec, y=cumul)
    _draw_bar(ax, total_spec)
    _draw_value_label(ax, total_spec, f"{bd.p_shaft_kw:.0f}", y_lo)
    _draw_xlabel(ax, total_x, bar_labels[-1], y_lo)

    if title is not None:
        title_pad = 10 if "\n" in title else 6
        ax.set_title(title, fontsize=TITLE_FS, pad=title_pad)

    _style_axes_arrows(ax, y_lo=y_lo, y_hi=y_hi, x_max=x_max, ylabel=YLABEL)
    _add_waterfall_legend(ax)
    ax.set_aspect("auto")
    subplot_adj = dict(SUBPLOT_ADJ)
    if title is not None and "\n" in title:
        subplot_adj["top"] = 0.82
    fig.subplots_adjust(**subplot_adj)
    return fig


def _style_placeholder_axes(ax: plt.Axes, *, y_lo: float, y_hi: float) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


def _draw_nrg_prac_av_panel(
    ax_energy: plt.Axes,
    ax_prac: plt.Axes,
    bd: CycleWaterfallBreakdown,
    *,
    steps: list[WorkWaterfallStep],
    step_labels: list[str],
    y_lo: float,
    y_hi: float,
    qin_one_decimal: bool = False,
) -> None:
    _draw_energy_axis(ax_energy, bd, y_lo=y_lo, y_hi=y_hi)
    _draw_prac_av_waterfall(
        ax_prac,
        bd,
        steps=steps,
        bar_labels=step_labels,
        y_lo=y_lo,
        y_hi=y_hi,
        qin_one_decimal=qin_one_decimal,
    )


def fig_cycle_nrg_prac_av(
    bd: CycleWaterfallBreakdown,
    *,
    title: str | None = None,
    qin_one_decimal: bool = False,
) -> plt.Figure:
    """Energy Q_in + W_x (left) and mirrored practical-availability waterfall (right)."""
    steps, bar_labels = work_waterfall_for_breakdown(bd)
    step_labels = bar_labels[:-1]
    y_lo, y_hi = _shared_heat_work_panel_y_limits()

    energy_w, prac_w, panel_w = _nrg_prac_figure_widths()

    fig = plt.figure(figsize=(panel_w, FIG_HEIGHT), dpi=150)
    fig.patch.set_facecolor("white")
    plt.rcParams["hatch.linewidth"] = 0.5

    inner = fig.add_gridspec(1, 2, width_ratios=[energy_w, prac_w], wspace=PANEL_WSPACE)
    ax_energy = fig.add_subplot(inner[0])
    ax_prac = fig.add_subplot(inner[1])
    ax_energy.set_facecolor("white")
    ax_prac.set_facecolor("white")

    _draw_nrg_prac_av_panel(
        ax_energy,
        ax_prac,
        bd,
        steps=steps,
        step_labels=step_labels,
        y_lo=y_lo,
        y_hi=y_hi,
        qin_one_decimal=qin_one_decimal,
    )

    if title is not None:
        title_pad = 10 if "\n" in title else 6
        ax_prac.set_title(title, fontsize=TITLE_FS, pad=title_pad)

    subplot_adj = dict(SUBPLOT_ADJ_NRG_PRAC)
    if title is not None and "\n" in title:
        subplot_adj["top"] = 0.82
    fig.subplots_adjust(**subplot_adj)
    return fig


def fig_cycle_heat_work(
    bd: CycleWaterfallBreakdown,
    *,
    title: str | None = None,
    qin_one_decimal: bool = False,
) -> plt.Figure:
    """Heat input Q_in [kW] (left) + availability waterfall (right)."""
    steps, bar_labels = work_waterfall_for_breakdown(bd)
    y_lo, y_hi = _shared_heat_work_panel_y_limits()

    heat_w, work_w, panel_w = _heat_work_figure_widths()

    fig = plt.figure(figsize=(panel_w, FIG_HEIGHT), dpi=150)
    fig.patch.set_facecolor("white")
    plt.rcParams["hatch.linewidth"] = 0.5

    inner = fig.add_gridspec(1, 2, width_ratios=[heat_w, work_w], wspace=PANEL_WSPACE)

    ax_heat = fig.add_subplot(inner[0])
    ax_work = fig.add_subplot(inner[1])

    ax_heat.set_facecolor("white")
    ax_work.set_facecolor("white")

    _draw_heat_input_axis(ax_heat, bd, y_lo=y_lo, y_hi=y_hi)
    _draw_work_waterfall(
        ax_work,
        bd,
        steps=steps,
        bar_labels=bar_labels,
        y_lo=y_lo,
        y_hi=y_hi,
        qin_one_decimal=qin_one_decimal,
    )

    if title is not None:
        _set_heat_work_figure_title(fig, title)

    subplot_adj = dict(SUBPLOT_ADJ_HEAT_WORK)
    if title is not None:
        subplot_adj["top"] = 0.74
    fig.subplots_adjust(**subplot_adj)
    return fig


def save_heat_work_bar(fig: plt.Figure, out_dir: Path, stem_suffix: str) -> Path:
    from datetime import datetime

    path = (out_dir / f"{HEAT_WORK_OUT_STEM}_{stem_suffix}.png").resolve()
    fig.savefig(path, dpi=300, facecolor="white")
    print(f"Wrote {path}  ({datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M:%S})")
    return path


def save_nrg_prac_av_bar(fig: plt.Figure, out_dir: Path, stem_suffix: str) -> Path:
    from datetime import datetime

    path = (out_dir / f"{NRG_PRAC_AV_OUT_STEM}_{stem_suffix}.png").resolve()
    fig.savefig(path, dpi=300, facecolor="white")
    print(f"Wrote {path}  ({datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M:%S})")
    return path


def save_cycle_bar(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    print(f"Wrote {path}")
    return path
