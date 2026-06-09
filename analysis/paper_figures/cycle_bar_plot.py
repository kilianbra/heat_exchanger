"""Waterfall bar-chart plotting for cycle practical-availability breakdowns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from cycle_waterfall import CycleWaterfallBreakdown
from plot_colors import COLOR_THERMAL, COLOR_VISC_HOT, color_with_alpha

GREEN_HEX = "#57724A"
GRAY = "lightgray"
CYCLE_ALPHA_LIGHT = 0.1
COLOR_THERMAL_FUEL = color_with_alpha(COLOR_THERMAL, CYCLE_ALPHA_LIGHT)
COLOR_VISC_REST = color_with_alpha(COLOR_VISC_HOT, CYCLE_ALPHA_LIGHT)
EDGE = "black"
EDGE_LW = 0.6
ARROW_LW = 0.8
BAR_WIDTH = 0.38
BAR_GAP = 0.55
X_FIRST = 0.5
N_SLOTS = 5
X_SLOT = [X_FIRST + i * BAR_GAP for i in range(N_SLOTS)]
X_MAX = 3.3
Y_LO = 0.0
Y_HI = 0.5
Y_TICK_STEP = 0.1
YLABEL = r"$\Sigma \Delta W_A^M / Q_{\mathrm{fuel}}$"
LABEL_FS = 7
LABEL_PAD = 0.012
FIGSIZE = (3.4, 2.8)
SUBPLOT_ADJ = dict(left=0.22, right=0.95, bottom=0.12, top=0.95)


@dataclass(frozen=True)
class _BarSpec:
    x: float
    bottom: float
    height: float
    width: float
    color: str | tuple[float, float, float, float]


@dataclass(frozen=True)
class WaterfallStep:
    delta: float
    color: str | tuple[float, float, float, float]


def open_cycle_steps(bd: CycleWaterfallBreakdown) -> list[WaterfallStep]:
    """Two-step open cycle: fuel thermal + closure term (closes to eta)."""
    return [
        WaterfallStep(bd.thermal_fuel, COLOR_THERMAL_FUEL),
        WaterfallStep(bd.visc_rest_signed, COLOR_VISC_REST),
    ]


def recuperated_cycle_steps(bd: CycleWaterfallBreakdown) -> list[WaterfallStep]:
    """Four-step recuperated cycle (matches legacy ppt_bar_concl order)."""
    return [
        WaterfallStep(bd.thermal_fuel, COLOR_THERMAL_FUEL),
        WaterfallStep(bd.visc_rest_signed, COLOR_VISC_REST),
        WaterfallStep(bd.visc_hex_signed, COLOR_VISC_HOT),
        WaterfallStep(bd.thermal_hex, COLOR_THERMAL),
    ]


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


def _draw_bar_label(ax: plt.Axes, spec: _BarSpec, text: str) -> None:
    ax.text(
        spec.x,
        _bar_top(spec) + LABEL_PAD,
        text,
        ha="center",
        va="bottom",
        fontsize=LABEL_FS,
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


def _style_axes_arrows(ax: plt.Axes, x_max: float) -> None:
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel(YLABEL)
    ax.tick_params(axis="y", labelsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["left"].set_position(("data", 0.0))
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(EDGE_LW)
        ax.spines[spine].set_color("black")

    dy = Y_HI - Y_LO
    ax.annotate(
        "",
        xy=(0.0, Y_HI + 0.04 * dy),
        xytext=(0.0, Y_HI - 0.08 * dy),
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
    total_color: str,
    total_slot: int,
) -> plt.Figure:
    if total_slot < len(steps):
        raise ValueError("total_slot must be after all step bars")
    if total_slot >= N_SLOTS:
        raise ValueError(f"total_slot must be < {N_SLOTS}")

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cumul = 0.0
    prev: _BarSpec | None = None

    for i, step in enumerate(steps):
        x = X_SLOT[i]
        bottom, height = _delta_bar_geom(cumul, step.delta)
        spec = _BarSpec(x=x, bottom=bottom, height=height, width=BAR_WIDTH, color=step.color)
        if prev is not None:
            _draw_connector(ax, prev, spec, y=cumul)
        _draw_bar(ax, spec)
        step_lbl = f"+{step.delta:.3f}" if step.delta >= 0 else f"{step.delta:.3f}"
        _draw_bar_label(ax, spec, step_lbl)
        cumul += step.delta
        prev = spec

    total_x = X_SLOT[total_slot]
    total_spec = _BarSpec(x=total_x, bottom=0.0, height=bd.total, width=BAR_WIDTH, color=total_color)
    if prev is not None:
        _draw_connector(ax, prev, total_spec, y=cumul)
    _draw_bar(ax, total_spec)
    _draw_bar_label(ax, total_spec, rf"$\eta = {bd.total:.3f}$")

    _style_axes_arrows(ax, x_max=X_MAX)
    ax.set_aspect("auto")
    fig.subplots_adjust(**SUBPLOT_ADJ)
    return fig


def save_cycle_bar(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    print(f"Wrote {path}")
    return path
