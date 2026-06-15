"""
Recuperated cycle practical-availability waterfall bar charts.

Outputs (Figs_current/conf_ppt_plots/):
  cycle_rec_ref_bar.png   — reference recuperator
  cycle_rec_fix_bar.png   — fixed mass optimum
  cycle_rec_glob_bar.png  — global (aircraft mass) optimum
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cycle_assumptions import DEFAULT_CYCLE, RECUP_PPT_CASES, RecuperatorInputs
from cycle_bar_plot import LABELS_RECUP, cycle_plot_title, fig_cycle_waterfall, recuperated_cycle_steps, save_cycle_bar
from cycle_model import solve_recuperated_cycle
from cycle_waterfall import print_waterfall, waterfall_from_solution
from fig_paths import CONF_PPT_PLOTS, ensure_fig_dirs


def _recup_title(label: str, recup: RecuperatorInputs) -> str:
    return (
        f"{label}  "
        f"eps={recup.eps:.2%}  "
        f"dp_h={recup.dp_hot_frac:.2%}  "
        f"dp_c={recup.dp_cold_frac:.2%}"
    )


def plot_rec_bar(stem: str, label: str, recup: RecuperatorInputs) -> None:
    sol = solve_recuperated_cycle(recup, DEFAULT_CYCLE)
    if sol is None:
        raise RuntimeError(f"recuperated cycle solve failed for {stem}")

    bd = waterfall_from_solution(sol)
    print_waterfall(bd, title=_recup_title(label, recup))

    fig = fig_cycle_waterfall(
        bd,
        steps=recuperated_cycle_steps(bd),
        bar_labels=LABELS_RECUP,
        title=cycle_plot_title(bd, recup),
    )
    save_cycle_bar(fig, CONF_PPT_PLOTS, f"cycle_{stem}_bar")
    plt.close(fig)


def main() -> None:
    ensure_fig_dirs()
    for stem, label, recup in RECUP_PPT_CASES:
        plot_rec_bar(stem, label, recup)


if __name__ == "__main__":
    main()
