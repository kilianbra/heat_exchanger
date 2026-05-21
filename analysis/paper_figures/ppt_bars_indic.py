"""
Minimal PPT-style bar charts: L-shaped axes with arrows, no ticks or labels.

- ppt_bar_1gray.png: one gray bar of height 1.
- ppt_bar_2gray_green.png: gray baseline + three mini waterfall deltas + green total bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from plot_colors import COLOR_THERMAL, COLOR_VISC_HOT

GREEN_HEX = "#57724A"
# Bar width and axis limits shared across both figs for consistent pixel scale.
BAR_WIDTH = 0.38
MINI_BAR_WIDTH = 0.12
X_MAX = 2.75
Y_MAX = 1.32
FIGSIZE = (3.4, 2.8)
SUBPLOT_ADJ = dict(left=0.12, right=0.95, bottom=0.12, top=0.95)
GRAY = "lightgray"
BLACK = "black"
EDGE = "black"
EDGE_LW = 0.6
ARROW_LW = 0.8

# Placeholder waterfall data (easy to swap later).
BASELINE = 1.0
BASELINE_X = 1.0
TOTAL_X = 2.0
DELTAS: list[tuple[float, str | tuple[float, float, float, float]]] = [
    (0.05, COLOR_THERMAL),
    (-0.03, COLOR_VISC_HOT),
    (0.08, BLACK),
]


@dataclass(frozen=True)
class _BarSpec:
    x: float
    bottom: float
    height: float
    width: float
    color: str | tuple[float, float, float, float]


def _delta_bar_geom(cumul: float, delta: float) -> tuple[float, float, float]:
    """Return (bottom, height, top) for a floating delta bar."""
    if delta >= 0:
        return cumul, delta, cumul + delta
    return cumul + delta, -delta, cumul


def _waterfall_mini_xs(
    baseline_x: float,
    total_x: float,
    n_mini: int,
    main_width: float = BAR_WIDTH,
    mini_width: float = MINI_BAR_WIDTH,
) -> list[float]:
    """Mini-bar centers with equal edge gaps: gray|gap|mini|gap|...|gap|green."""
    gray_right = baseline_x + main_width / 2
    green_left = total_x - main_width / 2
    gap = (green_left - gray_right - n_mini * mini_width) / (n_mini + 1)

    xs: list[float] = []
    cursor = gray_right + gap + mini_width / 2
    for _ in range(n_mini):
        xs.append(cursor)
        cursor += mini_width + gap
    return xs


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


def _draw_connector(
    ax: plt.Axes,
    left: _BarSpec,
    right: _BarSpec,
    y: float,
    *,
    zorder: int = 4,
) -> None:
    ax.plot(
        [left.x + left.width / 2, right.x - right.width / 2],
        [y, y],
        "k:",
        linewidth=EDGE_LW,
        zorder=zorder,
    )


def _style_axes_arrows(ax: plt.Axes, x_max: float, y_max: float) -> None:
    """Origin at (0,0); only bottom + left spines; no ticks/labels; arrows at axis tips."""
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0, labelsize=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["left"].set_position(("data", 0.0))
    for s in ("bottom", "left"):
        ax.spines[s].set_linewidth(EDGE_LW)
        ax.spines[s].set_color("black")

    # Y arrow (along x = 0)
    y0, y1 = 0.0, y_max
    dy = y1 - y0
    ax.annotate(
        "",
        xy=(0.0, y1 + 0.04 * dy),
        xytext=(0.0, y1 - 0.08 * dy),
        xycoords="data",
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", lw=ARROW_LW),
        annotation_clip=False,
    )

    # X arrow (along y = 0)
    x0, x1 = 0.0, x_max
    dx = x1 - x0
    ax.annotate(
        "",
        xy=(x1 + 0.02 * dx, 0.0),
        xytext=(x1 - 0.08 * dx, 0.0),
        xycoords="data",
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", lw=ARROW_LW),
        annotation_clip=False,
    )


def _fig_one_gray_bar() -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _draw_bar(ax, _BarSpec(x=1.0, bottom=0.0, height=BASELINE, width=BAR_WIDTH, color=GRAY))

    _style_axes_arrows(ax, x_max=X_MAX, y_max=Y_MAX)
    ax.set_aspect("auto")
    fig.subplots_adjust(**SUBPLOT_ADJ)
    return fig


def _fig_waterfall_bars() -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    baseline = _BarSpec(x=BASELINE_X, bottom=0.0, height=BASELINE, width=BAR_WIDTH, color=GRAY)
    _draw_bar(ax, baseline)

    mini_xs = _waterfall_mini_xs(BASELINE_X, TOTAL_X, len(DELTAS))
    cumul = BASELINE
    prev = baseline

    for (delta, color), x in zip(DELTAS, mini_xs, strict=True):
        bottom, height, _top = _delta_bar_geom(cumul, delta)
        spec = _BarSpec(x=x, bottom=bottom, height=height, width=MINI_BAR_WIDTH, color=color)
        _draw_connector(ax, prev, spec, y=cumul)
        _draw_bar(ax, spec)
        cumul += delta
        prev = spec

    final_total = cumul
    total = _BarSpec(x=TOTAL_X, bottom=0.0, height=final_total, width=BAR_WIDTH, color=GREEN_HEX)
    _draw_connector(ax, prev, total, y=cumul)
    _draw_bar(ax, total)

    _style_axes_arrows(ax, x_max=X_MAX, y_max=Y_MAX)
    ax.set_aspect("auto")
    fig.subplots_adjust(**SUBPLOT_ADJ)
    return fig


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "Figs_current"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1 = _fig_one_gray_bar()
    p1 = out_dir / "ppt_bar_1gray.png"
    fig1.savefig(p1, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig1)
    print(f"Wrote {p1}")

    fig2 = _fig_waterfall_bars()
    p2 = out_dir / "ppt_bar_2gray_green.png"
    fig2.savefig(p2, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
