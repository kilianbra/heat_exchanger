"""
Heat exchanger effectiveness decomposition (NTU form).

Reference curve (balanced counterflow, C_r = 1)
------------------------------------------------
    eps_cf,bal(NTU) = NTU / (1 + NTU).

Dimensionless F-factors (all relative to eps_cf,bal):
    F_Cr     = eps_cf(NTU, Cr)    / eps_cf,bal      (unbalance penalty; same for every HX geometry)
    F_xf     = eps_arr(NTU, Cr=1) / eps_cf,bal      (arrangement penalty at balanced C_r; independent of operating C_r)
    F_tot    = eps_arr(NTU, Cr)   / eps_cf,bal       (combined ratio)
    F_couple = F_tot / (F_Cr * F_xf)                (coupling between C_r and arrangement; = 1 if independent)

with eps_arr the chosen geometry/flow and eps_cf the aligned counterflow formula.

Primary heat-transfer grouping
-------------------------------
    G = eps_arr / NTU,    Q = G * UA * dT_inlet    (NTU = UA / C_min, dT_inlet = T_h_in - T_c_in).

Because eps_cf,bal = NTU / (1 + NTU), the reference is proportional to NTU, which gives the
exceptionally clean product rule:

    G = F_Cr * F_xf * F_couple / (1 + NTU).

Each factor is < 1 for all physical operating points, so G < 1 always holds.
The factor F_xf / (1 + NTU) = eps_arr(NTU, 1) / NTU is exactly G evaluated at C_r = 1 (balanced,
arrangement-only); it is plotted in ``plot_fxf_over_one_plus_ntu_vs_ntu`` as the arrangement-only
G baseline.  The analogous quantity for a C_r = 0 (isothermal) reference is eps_arr(NTU, 0) / NTU,
which is arrangement-independent (all geometries collapse to (1 - exp(-NTU)) / NTU at C_r = 0) —
see ``eps_ntu_bkdwn_isoth_fluid.py`` for details.

Replace the epsilon_ntu import with your own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from heat_exchanger.epsilon_ntu import epsilon_ntu

RefKind = Literal["balanced_cr1", "cr0_isothermal"]

REF_BALANCED_CR1: RefKind = "balanced_cr1"
REF_CR0_ISOTHERMAL: RefKind = "cr0_isothermal"

_EPS_REF_TEX: dict[RefKind, str] = {
    REF_BALANCED_CR1: r"\varepsilon_{\mathrm{cf,bal}}",
    REF_CR0_ISOTHERMAL: r"\varepsilon_{\mathrm{cf},\,C_r=0}",
}


def figures_dir() -> Path:
    """Directory ``analysis/standalone_tools/figs`` (created if missing)."""
    d = Path(__file__).resolve().parent / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_figure(fig, stem: str, *, dpi: int = 150) -> Path:
    """Save ``fig`` to ``figs/<stem>.png`` beside this module."""
    path = figures_dir() / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def eps_cf_balanced(NTU):
    """Balanced counterflow (``C_r = 1``): ``NTU / (1 + NTU)``."""
    return NTU / (1.0 + NTU)


def eps_cf_cr0_counterflow(NTU):
    r"""Counterflow in the ``C_r \to 0`` limit (one stream isothermal / phase-change): ``1 - e^{-\mathrm{NTU}}``."""
    return 1.0 - np.exp(-np.asarray(NTU, dtype=float))


def compute_factors(
    epsilon_ntu,
    NTU,
    Cr,
    exchanger_type,
    flow_type,
    n_passes=1,
    *,
    ref: RefKind = REF_BALANCED_CR1,
):
    NTU_g, Cr_g = np.meshgrid(NTU, Cr, indexing="ij")
    if ref == REF_BALANCED_CR1:
        eps_base = eps_cf_balanced(NTU_g)
        cr_for_fxf = 1.0
    elif ref == REF_CR0_ISOTHERMAL:
        eps_base = eps_cf_cr0_counterflow(NTU_g)
        cr_for_fxf = 0.0
    else:
        raise ValueError(f"Unknown ref {ref!r}; use {REF_BALANCED_CR1!r} or {REF_CR0_ISOTHERMAL!r}")

    eps_arr = np.vectorize(
        lambda n, c: epsilon_ntu(n, c, exchanger_type=exchanger_type, flow_type=flow_type, n_passes=n_passes)
    )(NTU_g, Cr_g)

    eps_cf_Cr = np.vectorize(
        lambda n, c: epsilon_ntu(n, c, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)
    )(NTU_g, Cr_g)

    eps_arr_bal = np.vectorize(
        lambda n: epsilon_ntu(n, cr_for_fxf, exchanger_type=exchanger_type, flow_type=flow_type, n_passes=n_passes)
    )(NTU_g)

    F_tot = eps_arr / eps_base
    F_Cr = eps_cf_Cr / eps_base
    F_xf = eps_arr_bal / eps_base
    F_couple = F_tot / (F_Cr * F_xf)
    G = eps_arr / NTU_g

    return dict(
        NTU=NTU_g,
        Cr=Cr_g,
        G=G,
        F_tot=F_tot,
        F_Cr=F_Cr,
        F_xf=F_xf,
        F_couple=F_couple,
        eps_arr=eps_arr,
        ref=ref,
        eps_ref_tex=_EPS_REF_TEX[ref],
    )


def _contour_panel(
    ax,
    res,
    key,
    *,
    white_level=1.0,
    unity_floor=False,
    cr_ylim: tuple[float, float] | None = (0.0, 1.0),
):
    """Filled contours on ``(NTU, Cr)``; optional ``unity_floor`` maps ``Z < 1`` to white and pins the scale at 1.

    If ``cr_ylim`` is ``None``, the ``C_r`` axis limits are left to autoscale (useful when factors
    go below 1 with an isothermal reference so the colormap is not empty).
    """
    Z = np.asarray(res[key], dtype=float)
    X, Y = res["NTU"], res["Cr"]

    if unity_floor:
        zmax = float(np.nanmax(Z))
        hi = max(zmax, white_level + 1e-9)
        levels = np.linspace(white_level, hi, 22)
        cmap = mpl.colormaps["viridis"].copy()
        cmap.set_under("white")
        cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend="min")
    else:
        cs = ax.contourf(X, Y, Z, levels=20, cmap="viridis")

    ax.contour(X, Y, Z, levels=10, colors="k", linewidths=0.4, alpha=0.5)
    ax.contour(
        X,
        Y,
        Z,
        levels=[white_level],
        colors="white",
        linewidths=1.8,
        linestyles="solid",
        zorder=10,
    )
    ax.set_xlabel("NTU")
    ax.set_ylabel(r"$C_r$")
    if cr_ylim is not None:
        ax.set_ylim(cr_ylim[0], cr_ylim[1])
    return cs


def plot_fcr_contour(
    res,
    *,
    title_prefix="",
    unity_floor=True,
    cr_ylim: tuple[float, float] | None = (0.0, 1.0),
):
    """Single contour map of ``F_Cr`` (identical for all exchanger geometries)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    cs = _contour_panel(ax, res, "F_Cr", unity_floor=unity_floor, cr_ylim=cr_ylim)
    fig.colorbar(cs, ax=ax, extend="min")
    ax.set_title(rf"$F_{{\mathrm{{Cr}}}}$ (counterflow, same $C_r$) / ${res['eps_ref_tex']}$")
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12)
    return fig


def plot_ftot_fcouple_contours(
    res_top,
    res_bottom,
    *,
    top_label="both unmixed",
    bottom_label="Cmax mixed",
    title_prefix="",
    unity_floor_couple=True,
    cr_ylim: tuple[float, float] | None = (0.0, 1.0),
):
    """2×2: rows unmixed / Cmax mixed; columns ``F_tot`` and ``F_couple``."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    rows = [
        (0, res_top, top_label),
        (1, res_bottom, bottom_label),
    ]
    for row, res, row_tag in rows:
        ert = res["eps_ref_tex"]
        for col, (key, base_label) in enumerate(
            [
                ("F_tot", rf"$F_{{\mathrm{{tot}}}}=\varepsilon/{ert}$"),
                ("F_couple", r"$F_{\mathrm{couple}}$"),
            ]
        ):
            ax = axes[row, col]
            unity_floor = key == "F_couple" and unity_floor_couple
            cs = _contour_panel(ax, res, key, unity_floor=unity_floor, cr_ylim=cr_ylim)
            if unity_floor:
                fig.colorbar(cs, ax=ax, extend="min")
            else:
                fig.colorbar(cs, ax=ax)
            ax.set_title(f"{base_label}\n{row_tag}")
    fig.suptitle(title_prefix, fontsize=13)
    return fig


def plot_fxf_vs_ntu_arrangements(
    epsilon_ntu,
    NTU,
    Cr,
    fxf_arrangements,
    *,
    ref: RefKind = REF_BALANCED_CR1,
    xf_state_note=r"$F_{xf}$ uses $\varepsilon_{\mathrm{arr}}(\mathrm{NTU},\,C_r=1)$",
    title_prefix="",
):
    """``F_xf`` vs NTU for listed arrangements; ``F_xf`` is independent of the operating ``C_r`` grid column."""
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for label, kw in fxf_arrangements:
        res = compute_factors(epsilon_ntu, NTU, Cr, **kw, ref=ref)
        ntu = res["NTU"][:, 0]
        fxf = res["F_xf"][:, 0]
        ax.plot(ntu, fxf, label=label)
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("NTU")
    ax.set_ylabel(r"$F_{xf}$")
    t_core = rf"$F_{{xf}}$ vs NTU — {xf_state_note}"
    t = f"{title_prefix} — {t_core}" if title_prefix else t_core
    ax.set_title(t, fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_fxf_vs_ntu_cr1(epsilon_ntu, NTU, Cr, fxf_arrangements, *, title_prefix=""):
    """``F_xf(NTU)`` with balanced-``C_r`` reference (see ``plot_fxf_vs_ntu_arrangements``).

    For crossflow with one stream mixed, Cmax- and Cmin-mixed coincide at ``C_r = 1``; use a single
    curve labelled e.g. "one fluid mixed".
    """
    return plot_fxf_vs_ntu_arrangements(
        epsilon_ntu,
        NTU,
        Cr,
        fxf_arrangements,
        ref=REF_BALANCED_CR1,
        xf_state_note=r"$F_{xf}=\varepsilon_{\mathrm{arr}}(\mathrm{NTU},\,1)/\varepsilon_{\mathrm{cf,bal}}$",
        title_prefix=title_prefix,
    )


def plot_g_vs_ntu_arrangements(
    epsilon_ntu,
    NTU,
    Cr,
    arrangements,
    *,
    ref: RefKind = REF_BALANCED_CR1,
    cr_solid=0.33,
    cr_dash=0.66,
    title_prefix="",
):
    r"""``G = \varepsilon/\mathrm{NTU} = F_{\mathrm{tot}}/(1+\mathrm{NTU})`` for each arrangement.

    One color per arrangement; solid line at ``cr_solid``, dashed at ``cr_dash`` (nearest grid columns).
    """
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = [entry["color"] for entry in plt.rcParams["axes.prop_cycle"]]

    res0 = compute_factors(epsilon_ntu, NTU, Cr, **arrangements[0][1], ref=ref)
    cr_vec = res0["Cr"][0, :]
    j_s = int(np.argmin(np.abs(cr_vec - cr_solid)))
    j_d = int(np.argmin(np.abs(cr_vec - cr_dash)))
    cr_s = float(cr_vec[j_s])
    cr_d = float(cr_vec[j_d])

    for i, (label, kw) in enumerate(arrangements):
        res = compute_factors(epsilon_ntu, NTU, Cr, **kw, ref=ref)
        ntu = res["NTU"][:, 0]
        c = colors[i % len(colors)]
        ax.plot(ntu, res["G"][:, j_s], color=c, ls="-", lw=1.8, label=rf"{label}, $C_r={cr_s:.2f}$")
        ax.plot(
            ntu,
            res["G"][:, j_d],
            color=c,
            ls="--",
            lw=1.8,
            label=rf"{label}, $C_r={cr_d:.2f}$",
        )

    ax.set_xlabel("NTU")
    ax.set_ylabel(r"$G=\varepsilon/\mathrm{NTU}$")
    t_core = (
        r"$G = F_{\mathrm{tot}}/(1+\mathrm{NTU}) = "
        r"F_{Cr}F_{xf}F_{\mathrm{couple}}/(1+\mathrm{NTU})$" + rf" — solid $C_r={cr_s:.2f}$, dashed $C_r={cr_d:.2f}$"
    )
    t = f"{title_prefix} — {t_core}" if title_prefix else t_core
    ax.set_title(t, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    return fig


def plot_fxf_over_one_plus_ntu_vs_ntu(
    epsilon_ntu,
    NTU,
    Cr,
    arrangements,
    *,
    ref: RefKind = REF_BALANCED_CR1,
    title_prefix="",
):
    r"""``F_{xf}/(1+\mathrm{NTU})`` vs NTU for each arrangement (independent of ``C_r``).

    Since ``G = F_{Cr} F_{xf} F_{\mathrm{couple}}/(1+\mathrm{NTU})``, this curve is ``G`` when
    ``F_{Cr} F_{\mathrm{couple}} = 1`` (e.g. ``C_r = 1`` and ``F_{\mathrm{couple}} = 1``).
    """
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = [entry["color"] for entry in plt.rcParams["axes.prop_cycle"]]

    for i, (label, kw) in enumerate(arrangements):
        res = compute_factors(epsilon_ntu, NTU, Cr, **kw, ref=ref)
        ntu = res["NTU"][:, 0]
        fxf = res["F_xf"][:, 0]
        y = fxf / (1.0 + ntu)
        c = colors[i % len(colors)]
        ax.plot(ntu, y, color=c, lw=1.8, label=label)

    ax.set_xlabel("NTU")
    ax.set_ylabel(r"$F_{xf}/(1+\mathrm{NTU})$")
    t_core = (
        r"$F_{xf}/(1+\mathrm{NTU})$ — same colors as $G$ plot; "
        r"$G$ when $F_{Cr}F_{\mathrm{couple}}=1$"
        + (r"; with $C_r=0$ ref., $F_{xf}=1$ for all geometries" if ref == REF_CR0_ISOTHERMAL else "")
    )
    t = f"{title_prefix} — {t_core}" if title_prefix else t_core
    ax.set_title(t, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    return fig


def plot_couple_slices(res, Cr_slices=(0.2, 0.5, 0.8, 1.0), title_prefix=""):
    NTU_vec = res["NTU"][:, 0]
    Cr_vec = res["Cr"][0, :]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for cr in Cr_slices:
        j = int(np.argmin(np.abs(Cr_vec - cr)))
        ax.plot(NTU_vec, res["F_couple"][:, j], label=f"Cr={Cr_vec[j]:.2f}")
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("NTU")
    ax.set_ylabel("F_couple")
    ax.set_title(f"{title_prefix}  coupling term")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_factor_vs_ntu(res, Cr_slice=0.5, title_prefix=""):
    NTU_vec = res["NTU"][:, 0]
    Cr_vec = res["Cr"][0, :]
    j = int(np.argmin(np.abs(Cr_vec - Cr_slice)))
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(NTU_vec, res["G"][:, j], label=r"$G=\varepsilon/\mathrm{NTU}$", lw=2)
    ax.plot(NTU_vec, res["F_Cr"][:, j], label="F_Cr", ls="--")
    ax.plot(NTU_vec, res["F_xf"][:, j], label="F_xf", ls="--")
    ax.plot(NTU_vec, res["F_couple"][:, j], label="F_couple", ls=":")
    ax.plot(
        NTU_vec,
        res["F_Cr"][:, j] * res["F_xf"][:, j] / (1.0 + NTU_vec),
        label=r"$F_{Cr}F_{xf}/(1+\mathrm{NTU})$ ($F_{\mathrm{couple}}=1$)",
        ls="-.",
        color="grey",
    )
    ax.set_xlabel("NTU")
    ax.set_ylabel("factor")
    ax.set_title(f"{title_prefix}  Cr={Cr_vec[j]:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def compare_arrangements_couple(epsilon_ntu, NTU, Cr, arrangements, Cr_slice=0.5, *, ref: RefKind = REF_BALANCED_CR1):
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for label, kw in arrangements:
        res = compute_factors(epsilon_ntu, NTU, Cr, **kw, ref=ref)
        Cr_vec = res["Cr"][0, :]
        j = int(np.argmin(np.abs(Cr_vec - Cr_slice)))
        ax.plot(res["NTU"][:, 0], res["F_couple"][:, j], label=label)
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("NTU")
    ax.set_ylabel("F_couple")
    ax.set_title(f"Coupling across arrangements, Cr={Cr_slice}")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


if __name__ == "__main__":
    NTU = np.linspace(0.1, 8.0, 60)
    Cr = np.linspace(0.01, 1.0, 40)

    arrangements = [
        ("xf both unmixed", dict(exchanger_type="cross_flow", flow_type="unmixed")),
        ("xf Cmax mixed", dict(exchanger_type="cross_flow", flow_type="Cmax_mixed")),
        ("xf Cmin mixed", dict(exchanger_type="cross_flow", flow_type="Cmin_mixed")),
        ("coflow", dict(exchanger_type="aligned_flow", flow_type="coflow")),
    ]

    # At C_r = 1, crossflow Cmax-mixed and Cmin-mixed are the same; one curve, one label.
    fxf_arrangements_cr1 = [
        ("Crossflow, both unmixed", dict(exchanger_type="cross_flow", flow_type="unmixed")),
        ("Crossflow, one fluid mixed", dict(exchanger_type="cross_flow", flow_type="Cmax_mixed")),
        ("Coflow", dict(exchanger_type="aligned_flow", flow_type="coflow")),
    ]

    res_unmixed = compute_factors(epsilon_ntu, NTU, Cr, exchanger_type="cross_flow", flow_type="unmixed")
    res_cmax = compute_factors(epsilon_ntu, NTU, Cr, exchanger_type="cross_flow", flow_type="Cmax_mixed")

    fig1 = plot_fcr_contour(res_unmixed, title_prefix=r"$F_{Cr}$ · counterflow reference (any configuration)")
    save_figure(fig1, "eps_ntu_breakdown_fcr")

    fig2 = plot_ftot_fcouple_contours(
        res_unmixed,
        res_cmax,
        top_label="crossflow, both unmixed",
        bottom_label="crossflow, Cmax mixed",
        title_prefix=r"$F_{\mathrm{tot}}$ and $F_{\mathrm{couple}}$ · crossflow",
    )
    save_figure(fig2, "eps_ntu_breakdown_ftot_fcouple")

    fig3 = plot_fxf_vs_ntu_cr1(
        epsilon_ntu,
        NTU,
        Cr,
        fxf_arrangements_cr1,
        title_prefix="Arrangement factor",
    )
    save_figure(fig3, "eps_ntu_breakdown_fxf_cr1")

    fig4 = plot_g_vs_ntu_arrangements(
        epsilon_ntu,
        NTU,
        Cr,
        arrangements,
        cr_solid=0.33,
        cr_dash=0.66,
        title_prefix="Heat-transfer grouping",
    )
    save_figure(fig4, "eps_ntu_breakdown_g_arrangements")

    fig5 = plot_fxf_over_one_plus_ntu_vs_ntu(
        epsilon_ntu,
        NTU,
        Cr,
        arrangements,
        title_prefix="Arrangement-only scaling",
    )
    save_figure(fig5, "eps_ntu_breakdown_fxf_over_1pntu")

    # # Earlier one-slice / comparison figures (kept for reference).
    # plot_factor_vs_ntu(res_unmixed, Cr_slice=0.5, title_prefix="Crossflow unmixed")
    # plot_couple_slices(res_unmixed, title_prefix="Crossflow unmixed")
    # compare_arrangements_couple(epsilon_ntu, NTU, Cr, arrangements, Cr_slice=0.5)
    plt.show()
