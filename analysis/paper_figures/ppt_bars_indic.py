"""
Minimal PPT-style bar charts: L-shaped axes with arrows, no ticks or labels.

- ppt_bar_1gray.png: one gray bar of height 1.
- ppt_bar_2gray_green.png: gray bar 1 + green (#57724A) bar 1.1 side by side (waterfall connectors later).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GREEN_HEX = "#57724A"
# Bar width and axis limits match the two-bar plot so both figs share the same pixel scale.
BAR_WIDTH = 0.38
X_MAX = 2.75
Y_MAX = 1.32
FIGSIZE = (3.4, 2.8)
SUBPLOT_ADJ = dict(left=0.12, right=0.95, bottom=0.12, top=0.95)
GRAY = "lightgray"
EDGE = "black"
EDGE_LW = 0.6
ARROW_LW = 0.8


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

    x = np.array([1.0])
    h = np.array([1.0])
    ax.bar(x, h, width=BAR_WIDTH, color=GRAY, edgecolor=EDGE, linewidth=EDGE_LW, zorder=3)

    _style_axes_arrows(ax, x_max=X_MAX, y_max=Y_MAX)
    ax.set_aspect("auto")
    fig.subplots_adjust(**SUBPLOT_ADJ)
    return fig


def _fig_two_bars() -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs = np.array([1.0, 2.0])
    heights = np.array([1.0, 1.1])
    colors = [GRAY, GREEN_HEX]
    for xi, hi, c in zip(xs, heights, colors, strict=True):
        ax.bar(xi, hi, width=BAR_WIDTH, color=c, edgecolor=EDGE, linewidth=EDGE_LW, zorder=3)

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

    fig2 = _fig_two_bars()
    p2 = out_dir / "ppt_bar_2gray_green.png"
    fig2.savefig(p2, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
