"""
Delta bar charts: design comparisons (diff opt or matched vs reference baseline).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from plot_colors import COLOR_THERMAL, COLOR_VISC_HOT, color_with_alpha
from ppt_waterfall import fig_delta_waterfall
from ppt_values import (
    AO_REF_OVER_AO_DIFF_OPT,
    AO_REF_OVER_AO_MATCHED,
    CycleWaterfallBreakdown,
    _fig9_bootstrap,
    _import_fig9,
    _sweep_diffusion_fig7c,
    baseline_point,
    compute_cycle_hex_over_pshaft,
    evaluate_fig9_at_ao_ref_over_ao,
    evaluate_fixed_bc_diffusion_at_aoref,
    fetch_fig9_waterfall_at_ao_ref_over_ao,
)

GRAY = "lightgray"
GREEN_HEX = "#57724A"  # baseline reference total (ppt_bar_concl green)
FIGSIZE_3BAR = (3.6, 2.8)
FIGSIZE_6BAR = (5.2, 2.8)


@dataclass(frozen=True)
class OptVsRefDeltas:
    delta_thermal: float
    delta_total: float
    delta_visc: float


@dataclass(frozen=True)
class FullCycleOptVsRefDeltas:
    """Waterfall step deltas (opt - ref), same order as ppt_bar_concl breakdown vs Q_fuel."""

    delta_visc_hex: float
    delta_thermal_hex: float
    delta_thermal_fuel: float
    delta_visc_rest: float  # increment of -visc_rest step
    delta_eta: float  # opt.total - ref.total


def _fixed_bc_sweep_arrays() -> tuple[dict, object, object, object, object, object, float, float]:
    base = baseline_point()
    g2_h = 0.5 * 1.4 * 0.11**2
    ntu_v, eps_v, dp_h_v, dp_c_v, av_v, _ = _sweep_diffusion_fig7c(
        base["t_ratio_stag"],
        base["t_dead_over_t_cold_in"],
        base["p_cold_in_over_p_hot_in"],
        base["p_dead_over_p_hot_in"],
        g2_h,
        base["pressure_drop_ratio"],
    )
    return base, ntu_v, eps_v, dp_h_v, dp_c_v, av_v, base["P_hin_static"], base["P_cin_static"]


def _hex_deltas_from_pair(ref, other) -> OptVsRefDeltas:
    delta_thermal = other.thermal_hex_book - ref.thermal_hex_book
    delta_total = other.dq_over_qmax - ref.dq_over_qmax
    delta_visc = delta_total - delta_thermal
    return OptVsRefDeltas(
        delta_thermal=delta_thermal,
        delta_total=delta_total,
        delta_visc=delta_visc,
    )


def _compute_fixed_bc_deltas(
    ao_ref_to: float,
    *,
    ao_ref_from: float = 1.0,
) -> tuple[OptVsRefDeltas, object, object]:
    _, ntu_v, eps_v, dp_h_v, dp_c_v, av_v, p_hi, p_ci = _fixed_bc_sweep_arrays()
    ref = evaluate_fixed_bc_diffusion_at_aoref(
        ao_ref_from, ntu_v, eps_v, dp_h_v, dp_c_v, av_v, p_hin_static_bar=p_hi, p_cin_static_bar=p_ci
    )
    other = evaluate_fixed_bc_diffusion_at_aoref(
        ao_ref_to, ntu_v, eps_v, dp_h_v, dp_c_v, av_v, p_hin_static_bar=p_hi, p_cin_static_bar=p_ci
    )
    return _hex_deltas_from_pair(ref, other), ref, other


def _compute_cycle_hex_deltas(
    ao_ref_to: float,
    *,
    ao_ref_from: float = 1.0,
    label_to: str = "Matched",
) -> tuple[OptVsRefDeltas, object, object, object]:
    fig9 = _import_fig9()
    bootstrap = _fig9_bootstrap(fig9)
    ref_pt = evaluate_fig9_at_ao_ref_over_ao(ao_ref_from, fig9, bootstrap, label="Reference (green)")
    to_pt = evaluate_fig9_at_ao_ref_over_ao(ao_ref_to, fig9, bootstrap, label=label_to)
    if ref_pt is None or to_pt is None:
        raise RuntimeError(f"fig9 cycle solve failed for A_o,ref/A_o = {ao_ref_from} or {ao_ref_to}")

    ref = compute_cycle_hex_over_pshaft(ref_pt, fig9)
    other = compute_cycle_hex_over_pshaft(to_pt, fig9)
    delta_thermal = other.thermal_hex - ref.thermal_hex
    delta_total = other.dq_over_pshaft - ref.dq_over_pshaft
    delta_visc = delta_total - delta_thermal
    deltas = OptVsRefDeltas(
        delta_thermal=delta_thermal,
        delta_total=delta_total,
        delta_visc=delta_visc,
    )
    return deltas, ref, other, fig9


def _compute_opt_vs_ref_deltas() -> tuple[OptVsRefDeltas, object, object]:
    return _compute_fixed_bc_deltas(AO_REF_OVER_AO_DIFF_OPT)


def _compute_cycle_opt_vs_ref_deltas() -> tuple[OptVsRefDeltas, object, object, object]:
    return _compute_cycle_hex_deltas(AO_REF_OVER_AO_DIFF_OPT, label_to="Diffusion optimum")


def _waterfall_step_deltas(ref: CycleWaterfallBreakdown, opt: CycleWaterfallBreakdown) -> FullCycleOptVsRefDeltas:
    return FullCycleOptVsRefDeltas(
        delta_visc_hex=opt.visc_hex_signed - ref.visc_hex_signed,
        delta_thermal_hex=opt.thermal_hex - ref.thermal_hex,
        delta_thermal_fuel=opt.thermal_fuel - ref.thermal_fuel,
        delta_visc_rest=-opt.visc_rest - (-ref.visc_rest),
        delta_eta=opt.total - ref.total,
    )


def _compute_full_cycle_green_vs_opt_deltas() -> tuple[FullCycleOptVsRefDeltas, CycleWaterfallBreakdown, CycleWaterfallBreakdown]:
    fig9 = _import_fig9()
    bootstrap = _fig9_bootstrap(fig9)
    _, ref_bd = fetch_fig9_waterfall_at_ao_ref_over_ao(1.0, fig9, bootstrap, label="Reference (green)")
    _, opt_bd = fetch_fig9_waterfall_at_ao_ref_over_ao(
        AO_REF_OVER_AO_DIFF_OPT,
        fig9,
        bootstrap,
        label="Diffusion optimum (opt_diff_fix_bc)",
    )
    deltas = _waterfall_step_deltas(ref_bd, opt_bd)
    return deltas, ref_bd, opt_bd


def main() -> None:
    from fig_paths import WHITTLE_ASME_PRACTICE, ensure_fig_dirs

    ensure_fig_dirs()
    out_dir = WHITTLE_ASME_PRACTICE

    deltas, ref, opt = _compute_opt_vs_ref_deltas()

    print("Opt vs reference (fixed BC, fig 7c), delta = opt - ref")
    print(f"  reference: A_o,ref/A_o = {ref.ao_ref_over_ao:.4f}, NTU = {ref.ntu:.4f}, eps = {ref.eps:.4f}")
    print(f"  optimum:   A_o,ref/A_o = {opt.ao_ref_over_ao:.4f}, NTU = {opt.ntu:.4f}, eps = {opt.eps:.4f}")
    print(f"  thermal_hex_book: ref = {ref.thermal_hex_book * 100:.2f} %, opt = {opt.thermal_hex_book * 100:.2f} %")
    print(f"  delta thermal (book-kept): {deltas.delta_thermal * 100:.2f} %")
    print(f"  dQ_o^M/Q_max: ref = {ref.dq_over_qmax * 100:.2f} %, opt = {opt.dq_over_qmax * 100:.2f} %")
    print(f"  delta dQ_o^M/Q_max: {deltas.delta_total * 100:.2f} %")
    print(f"  delta visc (residual): {deltas.delta_visc * 100:.2f} %")

    ylabel_qfuel = r"$\Sigma \Delta W_A^M / Q_{\mathrm{fuel}}$"

    fig = fig_delta_waterfall(
        [
            (deltas.delta_visc, COLOR_VISC_HOT),
            (deltas.delta_thermal, COLOR_THERMAL),
        ],
        deltas.delta_total,
        GRAY,
        ylabel=r"$\Sigma \Delta W_A^M / Q_{\mathrm{max}}$",
        figsize=FIGSIZE_3BAR,
        y_tick_step=0.005,
    )
    path = out_dir / "ppt_bar_concl_doptvbase.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")

    cycle_deltas, cycle_ref, cycle_opt, _fig9 = _compute_cycle_opt_vs_ref_deltas()
    print()
    print("Opt vs reference (coupled cycle, varying BC, HEx only), delta = opt - ref")
    print(f"  reference: A_o,ref/A_o = {cycle_ref.ao_ref_over_ao:.4f}")
    print(f"  optimum:   A_o,ref/A_o = {cycle_opt.ao_ref_over_ao:.4f}")
    print(f"  thermal (HEx)/P_shaft: ref = {cycle_ref.thermal_hex * 100:.2f} %, opt = {cycle_opt.thermal_hex * 100:.2f} %")
    print(f"  delta thermal (HEx): {cycle_deltas.delta_thermal * 100:.2f} %")
    print(f"  dQ_o^M/P_shaft: ref = {cycle_ref.dq_over_pshaft * 100:.2f} %, opt = {cycle_opt.dq_over_pshaft * 100:.2f} %")
    print(f"  delta dQ_o^M/P_shaft: {cycle_deltas.delta_total * 100:.2f} %")
    print(f"  delta visc (HEx residual): {cycle_deltas.delta_visc * 100:.2f} %")

    fig_cycle = fig_delta_waterfall(
        [
            (cycle_deltas.delta_visc, COLOR_VISC_HOT),
            (cycle_deltas.delta_thermal, COLOR_THERMAL),
        ],
        cycle_deltas.delta_total,
        GRAY,
        ylabel=r"$\Sigma \Delta W_A^M / P_{\mathrm{shaft}}$",
        figsize=FIGSIZE_3BAR,
    )
    path_cycle = out_dir / "ppt_bar_concl_doptvbase_vBC_hex_only.png"
    fig_cycle.savefig(path_cycle, dpi=300, facecolor="white")
    plt.close(fig_cycle)
    print(f"Wrote {path_cycle}")

    full_deltas, ref_bd, opt_bd = _compute_full_cycle_green_vs_opt_deltas()
    parts_sum = (
        full_deltas.delta_visc_hex
        + full_deltas.delta_thermal_hex
        + full_deltas.delta_thermal_fuel
        + full_deltas.delta_visc_rest
    )
    print()
    print("Opt vs reference (green vs opt_diff_fix_bc), delta = opt - ref, vs Q_fuel")
    print(f"  eta_cycle: ref = {ref_bd.total * 100:.2f} %, opt = {opt_bd.total * 100:.2f} %")
    print(f"  delta Visc (HEx): {full_deltas.delta_visc_hex * 100:.2f} %")
    print(f"  delta Thermal (HEx): {full_deltas.delta_thermal_hex * 100:.2f} %")
    print(f"  delta Thermal (fuel): {full_deltas.delta_thermal_fuel * 100:.2f} %")
    print(f"  delta Visc (rest): {full_deltas.delta_visc_rest * 100:.2f} %")
    print(f"  sum of steps: {parts_sum * 100:.2f} %")
    print(f"  delta eta (total): {full_deltas.delta_eta * 100:.2f} %")

    full_steps = [
        (full_deltas.delta_visc_hex, COLOR_VISC_HOT),
        (full_deltas.delta_thermal_hex, COLOR_THERMAL),
        (full_deltas.delta_visc_rest, COLOR_VISC_REST),
        (full_deltas.delta_thermal_fuel, COLOR_THERMAL_FUEL),
    ]

    fig_full = fig_delta_waterfall(
        full_steps,
        full_deltas.delta_eta,
        GRAY,
        ylabel=ylabel_qfuel,
        figsize=FIGSIZE_6BAR,
    )
    path_full = out_dir / "ppt_bar_concl_doptvbase_full.png"
    fig_full.savefig(path_full, dpi=300, facecolor="white")
    plt.close(fig_full)
    print(f"Wrote {path_full}")

    fig_full_bare = fig_delta_waterfall(
        full_steps,
        full_deltas.delta_eta,
        GREEN_HEX,
        ylabel=ylabel_qfuel,
        figsize=FIGSIZE_6BAR,
        show_bar_labels=False,
        show_y_axis=False,
    )
    path_full_bare = out_dir / "ppt_bar_concl_doptvbase_full_bare.png"
    fig_full_bare.savefig(path_full_bare, dpi=300, facecolor="white")
    plt.close(fig_full_bare)
    print(f"Wrote {path_full_bare}")

    match_bc, match_bc_ref, match_bc_to = _compute_fixed_bc_deltas(AO_REF_OVER_AO_MATCHED)
    print()
    print("Matched vs reference (fixed BC, fig 7c), delta = matched - ref (green -> gray)")
    print(f"  reference: A_o,ref/A_o = {match_bc_ref.ao_ref_over_ao:.4f}")
    print(f"  matched:   A_o,ref/A_o = {match_bc_to.ao_ref_over_ao:.4f}")
    print(f"  delta thermal (book-kept): {match_bc.delta_thermal * 100:.2f} %")
    print(f"  delta visc (residual): {match_bc.delta_visc * 100:.2f} %")
    print(f"  delta dQ_o^M/Q_max: {match_bc.delta_total * 100:.2f} %")

    fig_match_bc = fig_delta_waterfall(
        [
            (match_bc.delta_visc, COLOR_VISC_HOT),
            (match_bc.delta_thermal, COLOR_THERMAL),
        ],
        match_bc.delta_total,
        GRAY,
        ylabel=r"$\Sigma \Delta W_A^M / Q_{\mathrm{max}}$",
        figsize=FIGSIZE_3BAR,
        y_tick_step=0.005,
    )
    path_match_bc = out_dir / "ppt_bar_app_dgreengray_fixBC.png"
    fig_match_bc.savefig(path_match_bc, dpi=300, facecolor="white")
    plt.close(fig_match_bc)
    print(f"Wrote {path_match_bc}")

    match_cycle, match_cyc_ref, match_cyc_to, _ = _compute_cycle_hex_deltas(
        AO_REF_OVER_AO_MATCHED, label_to="Matched (gray)"
    )
    print()
    print("Matched vs reference (coupled cycle, HEx only), delta = matched - ref (green -> gray)")
    print(f"  reference: A_o,ref/A_o = {match_cyc_ref.ao_ref_over_ao:.4f}")
    print(f"  matched:   A_o,ref/A_o = {match_cyc_to.ao_ref_over_ao:.4f}")
    print(f"  delta thermal (HEx): {match_cycle.delta_thermal * 100:.2f} %")
    print(f"  delta visc (residual): {match_cycle.delta_visc * 100:.2f} %")
    print(f"  delta dQ_o^M/P_shaft: {match_cycle.delta_total * 100:.2f} %")

    fig_match_cycle = fig_delta_waterfall(
        [
            (match_cycle.delta_visc, COLOR_VISC_HOT),
            (match_cycle.delta_thermal, COLOR_THERMAL),
        ],
        match_cycle.delta_total,
        GRAY,
        ylabel=r"$\Sigma \Delta W_A^M / P_{\mathrm{shaft}}$",
        figsize=FIGSIZE_3BAR,
    )
    path_match_cycle = out_dir / "ppt_bar_app_dgreengray_vBC_hex_only.png"
    fig_match_cycle.savefig(path_match_cycle, dpi=300, facecolor="white")
    plt.close(fig_match_cycle)
    print(f"Wrote {path_match_cycle}")


if __name__ == "__main__":
    main()
