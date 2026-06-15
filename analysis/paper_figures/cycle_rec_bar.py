"""Shared runner for recuperated-cycle conf PPT waterfall bar charts."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cycle_assumptions import CycleAssumptions, RecuperatorInputs
from cycle_bar_plot import LABELS_RECUP, fig_cycle_waterfall, recuperated_cycle_steps, save_cycle_bar
from cycle_model import solve_recuperated_cycle
from cycle_waterfall import print_waterfall, waterfall_from_solution
from fig_paths import CONF_PPT_PLOTS


def recup_title(case_label: str, recup: RecuperatorInputs) -> str:
    return (
        f"{case_label}  eps={recup.eps * 100:.2f}%  "
        f"dp_h={recup.dp_hot_frac * 100:.2f}%  dp_c={recup.dp_cold_frac * 100:.2f}%"
    )


def plot_rec_bar(
    recup: RecuperatorInputs,
    stem: str,
    *,
    case_label: str,
    cycle: CycleAssumptions | None = None,
) -> None:
    """Solve, print breakdown, and save one recuperated waterfall chart."""
    sol = solve_recuperated_cycle(recup, cycle)
    if sol is None:
        raise RuntimeError(f"recuperated cycle solve failed for {case_label}")

    bd = waterfall_from_solution(sol)
    print_waterfall(bd, title=recup_title(case_label, recup))

    fig = fig_cycle_waterfall(bd, steps=recuperated_cycle_steps(bd), bar_labels=LABELS_RECUP)
    save_cycle_bar(fig, CONF_PPT_PLOTS, stem)
    plt.close(fig)
