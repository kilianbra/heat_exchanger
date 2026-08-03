"""
Journal heat/work cycle bar charts (heat input left, work breakdown right).

Outputs (Figs_current/final_journal_paper/):
  fig9a_baseline.*
  fig9b_fix_mass_opt.*
  fig9c_min_ac_mass.*
  bar_heat_work_no_rec.*  (open-cycle companion; not renumbered)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycle_assumptions import DEFAULT_CYCLE, RECUP_PPT_CASES, RecupPptCase
from cycle_bar_plot import (
    fig_cycle_heat_work,
    qin_one_decimal_for_suffix,
    save_heat_work_bar,
)
from cycle_model import solve_open_cycle, solve_recuperated_cycle
from cycle_waterfall import print_waterfall, waterfall_from_solution
from fig_paths import JOURNAL_PLOTS, ensure_fig_dirs

JOURNAL_FORMATS = ("svg", "png", "tiff", "pdf", "eps")

# Recuperated journal figure stems (suffix used for formatting flags → output name)
RECUP_JOURNAL_STEMS = {
    "rec_baseline": "fig9a_baseline",
    "rec_fix_mass_opt": "fig9b_fix_mass_opt",
    "rec_min_ac_mass": "fig9c_min_ac_mass",
}


def _save_heat_work_chart(bd, out_suffix: str, *, stem: str | None = None) -> None:
    qin_dec = qin_one_decimal_for_suffix(out_suffix)
    if out_suffix == "rec_fix_mass_opt":
        qin_dec = False
    fig_hw = fig_cycle_heat_work(
        bd,
        show_title=False,
        qin_one_decimal=qin_dec,
        label_style="journal",
    )
    save_heat_work_bar(
        fig_hw,
        JOURNAL_PLOTS,
        out_suffix,
        formats=JOURNAL_FORMATS,
        stem=stem,
    )
    plt.close(fig_hw)


def _plot_open(out_suffix: str) -> None:
    sol = solve_open_cycle(DEFAULT_CYCLE)
    bd = waterfall_from_solution(sol)
    print_waterfall(bd, title=f"Open cycle ({out_suffix})")
    _save_heat_work_chart(bd, out_suffix)


def _plot_recup_case(case: RecupPptCase, out_suffix: str) -> None:
    sol = solve_recuperated_cycle(case.recup, DEFAULT_CYCLE)
    if sol is None:
        raise RuntimeError(f"recuperated cycle solve failed for {case.stem}")

    bd = waterfall_from_solution(sol)
    print_waterfall(bd, title=f"{case.label} ({out_suffix})")
    _save_heat_work_chart(bd, out_suffix, stem=RECUP_JOURNAL_STEMS[out_suffix])


def main() -> None:
    ensure_fig_dirs()

    _plot_open("no_rec")

    case_by_stem = {case.stem: case for case in RECUP_PPT_CASES}
    _plot_recup_case(case_by_stem["rec_ref"], "rec_baseline")
    _plot_recup_case(case_by_stem["rec_fix"], "rec_fix_mass_opt")
    _plot_recup_case(case_by_stem["rec_glob"], "rec_min_ac_mass")


if __name__ == "__main__":
    main()
