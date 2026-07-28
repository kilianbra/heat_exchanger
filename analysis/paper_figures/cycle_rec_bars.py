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

from cycle_assumptions import DEFAULT_CYCLE, RECUP_PPT_CASES, RecupPptCase, mach_ratio_first_order
from cycle_bar_plot import LABELS_RECUP, cycle_plot_title, fig_cycle_waterfall, recuperated_cycle_steps, save_cycle_bar
from cycle_model import solve_recuperated_cycle
from cycle_waterfall import CycleWaterfallBreakdown, print_waterfall, waterfall_from_solution
from fig_paths import CONF_PPT_PLOTS, ensure_fig_dirs


def _print_geometry(bd: CycleWaterfallBreakdown, case: RecupPptCase, mdot_ref_kg_per_s: float) -> None:
    geom = case.geom
    ao_ref_over_ao = geom.ao_ref_over_ao
    if abs(geom.a_over_a_ref - 1.0) > 1e-4:
        print(f"  A/A_ref:                {geom.a_over_a_ref:8.4f}")
    if abs(geom.ao_over_ao_ref - 1.0) > 1e-4 or abs(geom.a_over_a_ref - 1.0) > 1e-4:
        print(f"  A_o,ref/A_o:            {ao_ref_over_ao:8.4f}")
    if abs(geom.a_over_a_ref - 1.0) > 1e-4 and mdot_ref_kg_per_s > 0:
        m_ratio = mach_ratio_first_order(bd.mdot_kg_per_s, mdot_ref_kg_per_s, ao_ref_over_ao)
        print(f"  M/M_ref (1st order):    {m_ratio:8.4f}  (= mdot/mdot_ref * A_o,ref/A_o)")


def plot_rec_bar(case: RecupPptCase, mdot_ref_kg_per_s: float) -> None:
    sol = solve_recuperated_cycle(case.recup, DEFAULT_CYCLE)
    if sol is None:
        raise RuntimeError(f"recuperated cycle solve failed for {case.stem}")

    bd = waterfall_from_solution(sol)
    print_waterfall(
        bd,
        title=(
            f"{case.label}  "
            f"eps={case.recup.eps:.2%}  "
            f"dp_h={case.recup.dp_hot_frac:.2%}  "
            f"dp_c={case.recup.dp_cold_frac:.2%}"
        ),
    )
    hi = sol.hex_hot_inlet()
    ci = sol.hex_cold_inlet()
    print(
        f"  HEx hot in (turb out):  T = {hi.T_K:.2f} K,  p = {hi.p_Pa / 1e5:.4f} bar"
    )
    print(
        f"  HEx cold in (comp out): T = {ci.T_K:.2f} K,  p = {ci.p_Pa / 1e5:.4f} bar"
    )
    print(f"  t = T_h,in/T_c,in = {hi.T_K / ci.T_K:.4f}")
    _print_geometry(bd, case, mdot_ref_kg_per_s)

    fig = fig_cycle_waterfall(
        bd,
        steps=recuperated_cycle_steps(bd),
        bar_labels=LABELS_RECUP,
        title=cycle_plot_title(
            bd,
            case.recup,
            geom=case.geom,
            mdot_ref_kg_per_s=mdot_ref_kg_per_s,
            case_label=case.label,
        ),
    )
    save_cycle_bar(fig, CONF_PPT_PLOTS, f"cycle_{case.stem}_bar")
    plt.close(fig)


def main() -> None:
    ensure_fig_dirs()
    ref_sol = solve_recuperated_cycle(RECUP_PPT_CASES[0].recup, DEFAULT_CYCLE)
    if ref_sol is None:
        raise RuntimeError("reference recuperated cycle solve failed")
    mdot_ref = ref_sol.mdot_kg_per_s

    for case in RECUP_PPT_CASES:
        plot_rec_bar(case, mdot_ref)


if __name__ == "__main__":
    main()
