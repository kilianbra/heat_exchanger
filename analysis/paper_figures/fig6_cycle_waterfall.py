"""
Fig6 lengthening + cycle coupling: eta optimum on cycle_fxd_power sweep and waterfall bars.

Outputs (Figs_current/conf_ppt_plots/):
  cycle_len_ind_fxp_bar.png   — industrial design, cycle_fxd_power
  cycle_len_etopt_fxp_bar.png — eta-cycle maximum, cycle_fxd_power
  cycle_len_ind_fxd_bar.png   — industrial design, cycle_fxd_mdot
  cycle_len_etontu_fxd_bar.png — same NTU as eta opt, cycle_fxd_mdot (comparison)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cycle_assumptions import DEFAULT_CYCLE, RecuperatorInputs
from cycle_bar_plot import (
    LABELS_RECUP,
    fig_cycle_waterfall,
    lengthening_waterfall_title,
    recuperated_cycle_steps,
    save_cycle_bar,
)
from cycle_hex_coupling import (
    bootstrap_industrial,
    coupled_point_to_waterfall,
    evaluate_lengthening_fxd_mdot,
    evaluate_lengthening_fxd_power,
    find_eta_optimum,
    print_eta_optimum_summary,
    print_local_pract_opt,
    print_lengthening_point,
    sweep_lengthening_fxd_power,
)
from cycle_waterfall import print_waterfall
from fig_paths import CONF_PPT_PLOTS, ensure_fig_dirs


def _recup_from_point(pt) -> RecuperatorInputs:
    return RecuperatorInputs(
        eps=pt.eps,
        dp_hot_frac=pt.dp_hot_frac,
        dp_cold_frac=pt.dp_cold_frac,
    )


def _plot_lengthening_waterfall(stem: str, case_label: str, pt) -> None:
    coupled = coupled_point_to_waterfall(pt, DEFAULT_CYCLE)
    if coupled is None:
        raise RuntimeError(f"waterfall failed for {stem}")

    recup = _recup_from_point(pt)
    bd = coupled.waterfall
    print_waterfall(bd, title=lengthening_waterfall_title(case_label, pt, recup, bd))

    fig = fig_cycle_waterfall(
        bd,
        steps=recuperated_cycle_steps(bd),
        bar_labels=LABELS_RECUP,
        title=lengthening_waterfall_title(case_label, pt, recup, bd),
    )
    save_cycle_bar(fig, CONF_PPT_PLOTS, stem)
    plt.close(fig)


def main() -> None:
    ensure_fig_dirs()
    bootstrap = bootstrap_industrial()
    sweep = sweep_lengthening_fxd_power(bootstrap)
    if len(sweep) < 3:
        raise RuntimeError(f"Too few valid sweep points ({len(sweep)})")

    result = find_eta_optimum(sweep, bootstrap)
    print_eta_optimum_summary(result)
    print()
    print_local_pract_opt(bootstrap)

    ind_fxp = evaluate_lengthening_fxd_power(bootstrap.ntu_match, bootstrap)
    if ind_fxp is None:
        raise RuntimeError("cycle_fxd_power failed at industrial NTU_MATCH")
    opt_fxp = result.sweep[result.idx_eta_opt]

    ind_fxd = evaluate_lengthening_fxd_mdot(bootstrap.ntu_match, bootstrap)
    if ind_fxd is None:
        raise RuntimeError("cycle_fxd_mdot failed at industrial design")
    opt_fxd = evaluate_lengthening_fxd_mdot(opt_fxp.ntu, bootstrap)
    if opt_fxd is None:
        raise RuntimeError("cycle_fxd_mdot failed at eta-optimum NTU")

    print()
    print_lengthening_point("Industrial (cycle_fxd_mdot)", ind_fxd, bootstrap)
    print()
    print_lengthening_point("Eta-opt NTU (cycle_fxd_mdot)", opt_fxd, bootstrap)
    print()

    _plot_lengthening_waterfall("cycle_len_ind_fxp", "Industrial", ind_fxp)
    _plot_lengthening_waterfall("cycle_len_etopt_fxp", r"$\eta$-opt", opt_fxp)
    _plot_lengthening_waterfall("cycle_len_ind_fxd", "Industrial (fxd mdot)", ind_fxd)
    _plot_lengthening_waterfall("cycle_len_etontu_fxd", r"$\eta$-opt NTU (fxd mdot)", opt_fxd)


if __name__ == "__main__":
    main()
