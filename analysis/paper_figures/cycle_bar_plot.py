"""Waterfall bar-chart plotting for dimensional cycle availability breakdowns [kW]."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt

from cycle_assumptions import RecuperatorInputs
from cycle_waterfall import CycleWaterfallBreakdown
from plot_colors import COLOR_THERMAL, COLOR_VISC_HOT, color_with_alpha

GREEN_HEX = "#57724A"
CYCLE_ALPHA_LIGHT = 0.1
COLOR_THERMAL_FUEL = color_with_alpha(COLOR_THERMAL, CYCLE_ALPHA_LIGHT)
COLOR_VISC_COMP = color_with_alpha(COLOR_VISC_HOT, CYCLE_ALPHA_LIGHT)
COLOR_VISC_TURB = COLOR_VISC_HOT
EDGE = "black"
EDGE_LW = 0.6
ARROW_LW = 0.8
BAR_WIDTH = 0.38
BAR_GAP = 0.55
X_FIRST = 0.5
X_MAX_BUFFER = BAR_GAP * 0.6
VALUE_FS = 7   # font size for bar value labels
XLABEL_FS = 7  # font size for x-axis component labels
TITLE_FS = 8
VALUE_PAD = 0.012
FIG_HEIGHT = 2.8
FIG_WIDTH_RECUP = 3.8
Y_LIMIT_BUFFER_FRAC = 0.10
SUBPLOT_ADJ = dict(left=0.18, right=0.95, bottom=0.20, top=0.86)
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
REFERENCE_N_SLOTS = len(LABELS_RECUP)


def cycle_plot_title(bd: CycleWaterfallBreakdown, recup: RecuperatorInputs | None = None) -> str:
    """LaTeX title with recuperator parameters (if any) and cycle efficiency."""
    eta_pct = bd.eta_cycle * 100
    if recup is None:
        return rf"$\eta_{{cycle}} = {eta_pct:.2f}\,\%$"
    return (
        rf"$\varepsilon = {recup.eps * 100:.2f}\,\%$, "
        rf"$\Delta p/p_{{hi}} = {recup.dp_hot_frac * 100:.2f}\,\%$, "
        rf"$\Delta p/p_{{ci}} = {recup.dp_cold_frac * 100:.2f}\,\%$, "
        rf"$\eta_{{cycle}} = {eta_pct:.2f}\,\%$"
    )


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


def _x_layout(n_slots: int) -> tuple[list[float], float]:
    """Bar centres and xmax (with trailing buffer) for n_slots bars."""
    x_slot = [X_FIRST + i * BAR_GAP for i in range(n_slots)]
    x_max = x_slot[-1] + X_MAX_BUFFER
    return x_slot, x_max


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


@lru_cache(maxsize=1)
def _shared_y_limits() -> tuple[float, float]:
    """Identical y-axis span for open and recuperated charts."""
    from cycle_assumptions import DEFAULT_CYCLE, RECUP_PPT_CASES
    from cycle_model import solve_open_cycle, solve_recuperated_cycle
    from cycle_waterfall import waterfall_from_solution

    peaks: list[float] = []
    open_bd = waterfall_from_solution(solve_open_cycle(DEFAULT_CYCLE))
    peaks.append(_peak_kw(open_bd, open_cycle_steps(open_bd)))

    for _, _, recup in RECUP_PPT_CASES:
        rec_sol = solve_recuperated_cycle(recup, DEFAULT_CYCLE)
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
        edgecolor=EDGE,
        linewidth=EDGE_LW,
        zorder=zorder,
    )


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


def _style_axes_arrows(ax: plt.Axes, *, y_lo: float, y_hi: float, x_max: float) -> None:
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(YLABEL, fontsize=8)
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

    _style_axes_arrows(ax, y_lo=y_lo, y_hi=y_hi, x_max=x_max)
    ax.set_aspect("auto")
    subplot_adj = dict(SUBPLOT_ADJ)
    if title is not None and "\n" in title:
        subplot_adj["top"] = 0.82
    fig.subplots_adjust(**subplot_adj)
    return fig


def save_cycle_bar(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    print(f"Wrote {path}")
    return path
