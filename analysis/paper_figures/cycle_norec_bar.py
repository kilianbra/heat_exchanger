"""
Open (no-recuperator) cycle practical-availability waterfall bar chart.

Output: Figs_current/conf_ppt_plots/cycle_norec_bar.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cycle_assumptions import DEFAULT_CYCLE
from cycle_bar_plot import LABELS_NOREC, cycle_plot_title, fig_cycle_waterfall, open_cycle_steps, save_cycle_bar
from cycle_model import solve_open_cycle
from cycle_waterfall import print_waterfall, waterfall_from_solution
from fig_paths import CONF_PPT_PLOTS, ensure_fig_dirs


def main() -> None:
    ensure_fig_dirs()
    sol = solve_open_cycle(DEFAULT_CYCLE)
    bd = waterfall_from_solution(sol)
    print_waterfall(bd, title="Open cycle (no recuperator)")

    fig = fig_cycle_waterfall(
        bd,
        steps=open_cycle_steps(bd),
        bar_labels=LABELS_NOREC,
        title=cycle_plot_title(bd),
    )
    save_cycle_bar(fig, CONF_PPT_PLOTS, "cycle_norec_bar")
    plt.close(fig)


if __name__ == "__main__":
    main()
