"""
Thulin intercooling comparison: literature waterfall deltas.

comp_second_cycle.png — secondary cycle vs bypass-cooling baseline.
comp_nacelle.png — nacelle cycle relative to secondary cycle (component-wise delta).

Writes PNGs to Figs_current/.
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
NACELLE_Y_LIMITS = (-0.01, 0.015)  # tighter view for nacelle vs secondary deltas


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


@dataclass(frozen=True)
class ThulinDiffStep:
    delta_pp: float
    show_label: bool = False


@dataclass(frozen=True)
class ThulinDiffWaterfall:
    """Ordered component deltas (nacelle minus secondary), omitting exact zeros."""

    steps: tuple[ThulinDiffStep, ...]
    net_pp: float

    def waterfall_steps(self) -> list[tuple[float, str | tuple[float, float, float, float]]]:
        return [
            (_pp(s.delta_pp), BENEFIT_COLOR if s.delta_pp >= 0 else PENALTY_COLOR) for s in self.steps
        ]

    def step_label_texts(self) -> list[str]:
        return [
            format_step_label_percent_points(_pp(s.delta_pp)) if s.show_label else "" for s in self.steps
        ]

    def net_frac(self) -> float:
        return _pp(self.net_pp)

    def check_net(self) -> None:
        parts = sum(s.delta_pp for s in self.steps)
        if abs(parts - self.net_pp) > 1e-6:
            raise ValueError(f"step sum = {parts:.4f} % != net {self.net_pp:.4f} %")


# Secondary cycle vs bypass-cooling baseline (Thulin literature).
SECOND_CYCLE = ThulinWaterfall(
    benefits_pp=(1.25, 0.59, 0.02),
    penalties_pp=(-0.64, -0.35, -0.22, -0.14, -0.07, -0.09),
    net_pp=0.35,
)

# Nacelle vs bypass-cooling baseline (reference tuples for subtraction).
_NACELLE_BASE = ThulinWaterfall(
    benefits_pp=(1.25, 0.57, 0.06),
    penalties_pp=(-1.28, -0.42, -0.20, -0.15, -0.06, -0.07),
    net_pp=-0.30,
)

# Penalty labels (same order as penalties_pp): IC, BP therm, BP jet mix, Core Exh therm,
# BP Nozzle, turbomachinery (all). IC Noz & Exh is 0.00 pp and omitted.
_PENALTY_NAMES = (
    "Intercooler",
    "BP exhaust thermal",
    "BP jet mix",
    "Core Exh therm",
    "BP Nozzle",
    "Turbomachinery",
)
_BENEFIT_NAMES = ("IC integ.", "Core jet mix", "Burner etc")


def _component_deltas_pp(
    nacelle: ThulinWaterfall,
    secondary: ThulinWaterfall,
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name, n, s in zip(_BENEFIT_NAMES, nacelle.benefits_pp, secondary.benefits_pp, strict=True):
        deltas[name] = n - s
    for name, n, s in zip(_PENALTY_NAMES, nacelle.penalties_pp, secondary.penalties_pp, strict=True):
        deltas[name] = n - s
    deltas["IC Noz & Exh"] = 0.0
    return deltas


def _nacelle_vs_secondary_waterfall() -> ThulinDiffWaterfall:
    deltas = _component_deltas_pp(_NACELLE_BASE, SECOND_CYCLE)
    net_pp = _NACELLE_BASE.net_pp - SECOND_CYCLE.net_pp

    # Drop exact zeros (IC integ. benefit and IC Noz & Exh are both 0.00 pp).
    nonzero = {k: v for k, v in deltas.items() if abs(v) > 1e-9}

    # Intercooler first, then remaining steps by decreasing |delta|.
    ordered_names = ["Intercooler"]
    rest = sorted(
        (name for name in nonzero if name != "Intercooler"),
        key=lambda name: abs(nonzero[name]),
        reverse=True,
    )
    ordered_names.extend(rest)

    steps = tuple(
        ThulinDiffStep(
            delta_pp=nonzero[name],
            show_label=name in ("Intercooler", "BP exhaust thermal", "Burner etc"),
        )
        for name in ordered_names
    )
    return ThulinDiffWaterfall(steps=steps, net_pp=net_pp)


NACELLE_VS_SECOND = _nacelle_vs_secondary_waterfall()


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


def _fig_nacelle_vs_secondary(data: ThulinDiffWaterfall) -> plt.Figure:
    data.check_net()
    return fig_delta_waterfall(
        data.waterfall_steps(),
        data.net_frac(),
        NET_COLOR,
        ylabel=YLABEL,
        figsize=FIGSIZE_10BAR,
        y_tick_step=Y_TICK_STEP,
        y_limits=NACELLE_Y_LIMITS,
        show_arrows=False,
        format_net_label=format_net_label_percent_points,
        step_label_texts=data.step_label_texts(),
    )


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    print(f"Wrote {path}")
    return path


def main() -> None:
    out_dir = Path(__file__).resolve().parent / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_nacelle = _fig_nacelle_vs_secondary(NACELLE_VS_SECOND)
    _save_figure(fig_nacelle, out_dir, STEM_NACELLE)
    plt.close(fig_nacelle)

    fig_second = _fig_thulin_waterfall(SECOND_CYCLE)
    _save_figure(fig_second, out_dir, STEM_SECOND_CYCLE)
    plt.close(fig_second)


if __name__ == "__main__":
    main()
