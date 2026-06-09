"""
Open (no-recuperator) cycle waterfall bar chart for conference PPT.

Output: Figs_current/conf_ppt_plots/cycle_bar_norecup.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cycle_assumptions import DEFAULT_CYCLE
from cycle_bar_plot import GRAY, fig_cycle_waterfall, open_cycle_steps, save_cycle_bar
from cycle_model import solve_open_cycle
from cycle_waterfall import open_cycle_waterfall
from fig_paths import CONF_PPT_PLOTS, ensure_fig_dirs


def main() -> None:
    ensure_fig_dirs()
    sol = solve_open_cycle(DEFAULT_CYCLE)
    bd = open_cycle_waterfall(sol)

    print("Open cycle (no recuperator)")
    print(f"  mdot = {sol.mdot_kg_per_s:.4f} kg/s")
    print(f"  eta  = {bd.total:.4f}")
    print(f"  thermal (fuel): + {bd.thermal_fuel:.4f}")
    print(f"  visc (rest):    {bd.visc_rest_signed:.4f}")
    print(f"  [detail] comp:  {bd.visc_comp_signed:.4f}, turb: {bd.visc_turb_signed:.4f}")

    fig = fig_cycle_waterfall(
        bd,
        steps=open_cycle_steps(bd),
        total_color=GRAY,
        total_slot=2,
    )
    save_cycle_bar(fig, CONF_PPT_PLOTS, "cycle_bar_norecup")
    plt.close(fig)


if __name__ == "__main__":
    main()
