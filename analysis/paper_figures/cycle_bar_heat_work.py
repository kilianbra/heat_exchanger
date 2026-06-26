"""
Heat/work and energy/practical-availability cycle bar charts.

Outputs (Figs_current/conf_ppt_plots/):
  bar_heat_work_*.png          — method 1: Q_in heat axis + work waterfall
  bar_nrg_prac_av_*.png        — method 2: energy (Q_in, W_x) + mirrored prac-av waterfall

Optional cases (not in default run — restore in main() if needed):
  cycle_len_ind_fxp   — industrial NTU, fixed shaft power (cycle_fxd_power)
  cycle_len_ind_fxd   — industrial NTU, fixed mdot (cycle_fxd_mdot)
  cycle_rec_eps_dp    — legacy eps/dp comparison chart (no generator in repo)
  cycle_bar_norecup / cycle_bar_recup — older naming aliases
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cycle_assumptions import DEFAULT_CYCLE, RECUP_PPT_CASES, RecupPptCase
from cycle_bar_plot import (
    cycle_plot_title,
    fig_cycle_heat_work,
    fig_cycle_nrg_prac_av,
    lengthening_waterfall_title,
    qin_one_decimal_for_suffix,
    save_heat_work_bar,
    save_nrg_prac_av_bar,
)
from cycle_hex_coupling import (
    bootstrap_industrial,
    coupled_point_to_waterfall,
    evaluate_lengthening_fxd_mdot,
    evaluate_lengthening_fxd_power,
    find_eta_optimum,
    sweep_lengthening_fxd_power,
)
from cycle_model import solve_open_cycle, solve_recuperated_cycle
from cycle_waterfall import print_waterfall, waterfall_from_solution
from fig_paths import CONF_PPT_PLOTS, ensure_fig_dirs

# Optional lengthening cases — enable by adding to _OPTIONAL_LEN_STEMS in main().
_OPTIONAL_LEN_STEMS: dict[str, str] = {
    "rec_length_ind_fix_Pnet": "cycle_len_ind_fxp",
    "rec_length_ind_fix_mdot": "cycle_len_ind_fxd",
}


def _save_case_charts(bd, out_suffix: str, title: str) -> None:
    qin_dec = qin_one_decimal_for_suffix(out_suffix)
    fig_hw = fig_cycle_heat_work(bd, title=title, qin_one_decimal=qin_dec)
    save_heat_work_bar(fig_hw, CONF_PPT_PLOTS, out_suffix)
    plt.close(fig_hw)

    fig_npa = fig_cycle_nrg_prac_av(bd, title=title, qin_one_decimal=qin_dec)
    save_nrg_prac_av_bar(fig_npa, CONF_PPT_PLOTS, out_suffix)
    plt.close(fig_npa)


def _plot_open(out_suffix: str) -> None:
    sol = solve_open_cycle(DEFAULT_CYCLE)
    bd = waterfall_from_solution(sol)
    print_waterfall(bd, title=f"Open cycle ({out_suffix})")
    _save_case_charts(
        bd,
        out_suffix,
        cycle_plot_title(bd, case_label="Open cycle (no recuperator)"),
    )


def _plot_recup_case(case: RecupPptCase, out_suffix: str, mdot_ref_kg_per_s: float) -> None:
    sol = solve_recuperated_cycle(case.recup, DEFAULT_CYCLE)
    if sol is None:
        raise RuntimeError(f"recuperated cycle solve failed for {case.stem}")

    bd = waterfall_from_solution(sol)
    print_waterfall(bd, title=f"{case.label} ({out_suffix})")
    _save_case_charts(
        bd,
        out_suffix,
        cycle_plot_title(
            bd,
            case.recup,
            geom=case.geom,
            mdot_ref_kg_per_s=mdot_ref_kg_per_s,
            case_label=case.label,
        ),
    )


def _plot_lengthening(out_suffix: str, case_label: str, pt) -> None:
    coupled = coupled_point_to_waterfall(pt, DEFAULT_CYCLE)
    if coupled is None:
        raise RuntimeError(f"waterfall failed for {out_suffix}")

    from cycle_assumptions import RecuperatorInputs

    recup = RecuperatorInputs(eps=pt.eps, dp_hot_frac=pt.dp_hot_frac, dp_cold_frac=pt.dp_cold_frac)
    bd = coupled.waterfall
    print_waterfall(bd, title=lengthening_waterfall_title(case_label, pt, recup, bd))
    _save_case_charts(bd, out_suffix, lengthening_waterfall_title(case_label, pt, recup, bd))


def main() -> None:
    ensure_fig_dirs()

    _plot_open("no_rec")

    case_by_stem = {case.stem: case for case in RECUP_PPT_CASES}
    ref_sol = solve_recuperated_cycle(case_by_stem["rec_ref"].recup, DEFAULT_CYCLE)
    if ref_sol is None:
        raise RuntimeError("reference recuperated cycle solve failed")
    mdot_ref = ref_sol.mdot_kg_per_s

    _plot_recup_case(case_by_stem["rec_ref"], "rec_baseline", mdot_ref)
    _plot_recup_case(case_by_stem["rec_fix"], "rec_fix_mass_opt", mdot_ref)
    _plot_recup_case(case_by_stem["rec_glob"], "rec_min_ac_mass", mdot_ref)

    bootstrap = bootstrap_industrial()
    sweep = sweep_lengthening_fxd_power(bootstrap)
    if len(sweep) < 3:
        raise RuntimeError(f"Too few valid sweep points ({len(sweep)})")
    result = find_eta_optimum(sweep, bootstrap)
    opt_fxp = result.sweep[result.idx_eta_opt]
    opt_fxd = evaluate_lengthening_fxd_mdot(opt_fxp.ntu, bootstrap)
    if opt_fxd is None:
        raise RuntimeError("cycle_fxd_mdot failed at eta-optimum NTU")

    _plot_lengthening("rec_length_opt_fix_Pnet", r"$\eta$-opt (fxd $P_{net}$)", opt_fxp)
    _plot_lengthening("rec_length_opt_fix_mdot", r"$\eta$-opt NTU (fxd $\dot{m}$)", opt_fxd)

    # Uncomment to regenerate optional industrial lengthening comparisons:
    # ind_fxp = evaluate_lengthening_fxd_power(bootstrap.ntu_match, bootstrap)
    # ind_fxd = evaluate_lengthening_fxd_mdot(bootstrap.ntu_match, bootstrap)
    # if ind_fxp is not None:
    #     _plot_lengthening("rec_length_ind_fix_Pnet", "Industrial (fxd $P_{net}$)", ind_fxp)
    # if ind_fxd is not None:
    #     _plot_lengthening("rec_length_ind_fix_mdot", "Industrial (fxd $\dot{m}$)", ind_fxd)


if __name__ == "__main__":
    main()
