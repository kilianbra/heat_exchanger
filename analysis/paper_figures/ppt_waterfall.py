"""
Shared PPT-style waterfall bar charts: floating steps, connectors, arrow axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

EDGE = "black"
EDGE_LW = 0.6
ARROW_LW = 0.8
BAR_WIDTH = 0.38
BAR_GAP = 0.55
X_FIRST = 0.5
LABEL_FS = 7
LABEL_PAD_FRAC = 0.03
SUBPLOT_ADJ = dict(left=0.22, right=0.95, bottom=0.12, top=0.95)


@dataclass(frozen=True)
class BarSpec:
    x: float
    bottom: float
    height: float
    width: float
    color: str | tuple[float, float, float, float]


def delta_bar_geom(cumul: float, delta: float) -> tuple[float, float]:
    if delta >= 0:
        return cumul, delta
    return cumul + delta, -delta


def format_step_label_fraction(delta: float) -> str:
    if delta >= 0:
        return f"+{delta:.3f}"
    return f"{delta:.3f}"


def format_net_label_fraction(net_sum: float) -> str:
    return f"{net_sum:.3f}"


def format_step_label_percent_points(delta_frac: float) -> str:
    pp = delta_frac * 100.0
    if pp >= 0:
        return f"+{pp:.2f}%"
    return f"{pp:.2f}%"


def format_net_label_percent_points(net_frac: float) -> str:
    return f"{net_frac * 100.0:.2f}%"


def _draw_bar(ax: plt.Axes, spec: BarSpec, *, zorder: int = 3) -> None:
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


def _draw_bar_label(ax: plt.Axes, spec: BarSpec, text: str, y_offset: float) -> None:
    ax.text(
        spec.x,
        _bar_top(spec) + y_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=LABEL_FS,
        color="black",
        zorder=5,
    )


def _draw_connector(ax: plt.Axes, left: BarSpec, right: BarSpec, y: float, *, zorder: int = 4) -> None:
    ax.plot(
        [left.x + left.width / 2, right.x - right.width / 2],
        [y, y],
        "k:",
        linewidth=EDGE_LW,
        zorder=zorder,
    )


def _bar_top(spec: BarSpec) -> float:
    return spec.bottom + spec.height


def _bar_bottom(spec: BarSpec) -> float:
    return spec.bottom


def net_sum_bar_geom(net_sum: float) -> tuple[float, float]:
    """Anchor net total on the x-axis (bottom=0), not stacked on waterfall end."""
    if net_sum >= 0:
        return 0.0, net_sum
    return net_sum, -net_sum


def auto_y_limits(y_lo: float, y_hi: float, *, pad_frac: float = 0.12) -> tuple[float, float]:
    span = y_hi - y_lo
    if span <= 0:
        span = max(abs(y_hi), abs(y_lo), 1e-3)
    pad = span * pad_frac
    y0 = min(y_lo, 0.0) - pad
    y1 = max(y_hi, 0.0) + pad
    if y1 <= y0:
        y1 = y0 + 0.05
    return y0, y1


def style_axes_arrows_yticks(
    ax: plt.Axes,
    x_max: float,
    y_lo: float,
    y_hi: float,
    *,
    ylabel: str,
    y_tick_step: float | None = None,
    show_y_axis: bool = True,
    show_arrows: bool = True,
) -> None:
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([])
    if show_y_axis:
        if y_tick_step is not None:
            ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="y", labelsize=8)
    else:
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", length=0, labelsize=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["left"].set_position(("data", 0.0))
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(EDGE_LW)
        ax.spines[spine].set_color("black")

    if show_arrows:
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


def fig_delta_waterfall(
    waterfall_steps: list[tuple[float, str | tuple[float, float, float, float]]],
    net_sum: float,
    net_color: str | tuple[float, float, float, float],
    *,
    ylabel: str,
    figsize: tuple[float, float],
    y_tick_step: float | None = None,
    show_bar_labels: bool = True,
    show_y_axis: bool = True,
    show_arrows: bool = True,
    y_limits: tuple[float, float] | None = None,
    format_step_label: Callable[[float], str] = format_step_label_fraction,
    format_net_label: Callable[[float], str] = format_net_label_fraction,
    step_label_texts: list[str] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    n_bars = len(waterfall_steps) + 1
    xs = [X_FIRST + i * BAR_GAP for i in range(n_bars)]
    cumul = 0.0
    prev: BarSpec | None = None
    y_lo = 0.0
    y_hi = 0.0

    step_labels: list[tuple[BarSpec, str]] = []
    if step_label_texts is not None and len(step_label_texts) != len(waterfall_steps):
        raise ValueError(
            f"step_label_texts length {len(step_label_texts)} != waterfall steps {len(waterfall_steps)}"
        )

    for i, ((delta, color), x) in enumerate(zip(waterfall_steps, xs[:-1], strict=True)):
        bottom, height = delta_bar_geom(cumul, delta)
        spec = BarSpec(x=x, bottom=bottom, height=height, width=BAR_WIDTH, color=color)
        if prev is not None:
            _draw_connector(ax, prev, spec, y=cumul)
        _draw_bar(ax, spec)
        if step_label_texts is not None:
            label = step_label_texts[i]
        else:
            label = format_step_label(delta)
        step_labels.append((spec, label))
        y_lo = min(y_lo, _bar_bottom(spec))
        y_hi = max(y_hi, _bar_top(spec))
        cumul += delta
        prev = spec

    if not math.isclose(cumul, net_sum, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"waterfall cumulative {cumul:.6g} != net_sum {net_sum:.6g}")

    net_bottom, net_height = net_sum_bar_geom(net_sum)
    net_spec = BarSpec(
        x=xs[-1],
        bottom=net_bottom,
        height=net_height,
        width=BAR_WIDTH,
        color=net_color,
    )
    if prev is not None:
        _draw_connector(ax, prev, net_spec, y=cumul)
    _draw_bar(ax, net_spec)
    y_lo = min(y_lo, _bar_bottom(net_spec))
    y_hi = max(y_hi, _bar_top(net_spec))

    x_max = xs[-1] + BAR_WIDTH / 2 + 0.35
    if y_limits is not None:
        y0, y1 = y_limits
    else:
        y0, y1 = auto_y_limits(y_lo, y_hi)
    if show_bar_labels:
        label_off = LABEL_PAD_FRAC * (y1 - y0)
        if y_limits is None:
            y1 += label_off
        for spec, text in step_labels:
            if text:
                _draw_bar_label(ax, spec, text, label_off)
        _draw_bar_label(ax, net_spec, format_net_label(net_sum), label_off)

    style_axes_arrows_yticks(
        ax,
        x_max=x_max,
        y_lo=y0,
        y_hi=y1,
        ylabel=ylabel,
        y_tick_step=y_tick_step,
        show_y_axis=show_y_axis,
        show_arrows=show_arrows,
    )
    ax.set_aspect("auto")
    fig.subplots_adjust(**SUBPLOT_ADJ)
    return fig
