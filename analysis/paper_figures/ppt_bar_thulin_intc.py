"""
Thulin intercooling comparison: literature waterfall deltas (nacelle vs secondary cycle).

Writes comp_nacelle.png and comp_second_cycle.png to Figs_current/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from plot_colors import COLOR_THERMAL
from ppt_waterfall import (
    format_net_label_percent_points,
    format_step_label_percent_points,
    fig_delta_waterfall,
)

GREEN_HEX = "#57724A"
GRAY = "lightgray"
BENEFIT_COLOR = GREEN_HEX
PENALTY_COLOR = COLOR_THERMAL
NET_COLOR = GRAY

YLABEL = "Classical availability destruction\n/ Fuel Exergy"
FIGSIZE_10BAR = (7.6, 2.8)
Y_TICK_STEP = 0.005  # fraction; 0.5 pp on axis

OUT_DIR_NAME = "Figs_current"
STEM_NACELLE = "comp_nacelle"
STEM_SECOND_CYCLE = "comp_second_cycle"


def _pp(value_percent_points: float) -> float:
    """Convert literature value in percent points to axis fraction."""
    return value_percent_points / 100.0


Y_LIMITS = (_pp(-0.5), _pp(2.0))


@dataclass(frozen=True)
class ThulinWaterfall:
    benefits_pp: tuple[float, float, float]
    penalties_pp: tuple[float, float, float, float, float, float]
    net_pp: float

    def steps(self) -> list[tuple[float, str | tuple[float, float, float, float]]]:
        steps: list[tuple[float, str | tuple[float, float, float, float]]] = [
            (_pp(d), BENEFIT_COLOR) for d in self.benefits_pp
        ]
        steps.extend((_pp(d), PENALTY_COLOR) for d in self.penalties_pp)
        return steps

    def net_frac(self) -> float:
        return _pp(self.net_pp)

    def check_net(self) -> None:
        parts = sum(self.benefits_pp) + sum(self.penalties_pp)
        if abs(parts - self.net_pp) > 1e-6:
            raise ValueError(f"benefits+penalties = {parts:.4f} % != net {self.net_pp:.4f} %")


NACELLE = ThulinWaterfall(
    benefits_pp=(1.25, 0.57, 0.06),
    penalties_pp=(-1.28, -0.42, -0.20, -0.15, -0.06, -0.07),
    net_pp=-0.30,
)

SECOND_CYCLE = ThulinWaterfall(
    benefits_pp=(1.25, 0.59, 0.02),
    penalties_pp=(-0.64, -0.35, -0.22, -0.14, -0.07, -0.09),
    net_pp=0.35,
)


def _fig_thulin_waterfall(data: ThulinWaterfall) -> plt.Figure:
    data.check_net()
    return fig_delta_waterfall(
        data.steps(),
        data.net_frac(),
        NET_COLOR,
        ylabel=YLABEL,
        figsize=FIGSIZE_10BAR,
        y_tick_step=Y_TICK_STEP,
        y_limits=Y_LIMITS,
        show_arrows=False,
        format_step_label=format_step_label_percent_points,
        format_net_label=format_net_label_percent_points,
    )


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    print(f"Wrote {path}")
    return path


def main() -> None:
    out_dir = Path(__file__).resolve().parent / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    for data, stem in ((NACELLE, STEM_NACELLE), (SECOND_CYCLE, STEM_SECOND_CYCLE)):
        fig = _fig_thulin_waterfall(data)
        _save_figure(fig, out_dir, stem)
        plt.close(fig)


if __name__ == "__main__":
    main()
