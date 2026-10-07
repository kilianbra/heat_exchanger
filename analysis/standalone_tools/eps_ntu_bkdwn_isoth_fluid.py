r"""
NTU-eps breakdown with a **C_r = 0 (isothermal / phase-change)** counterflow reference.

This script reuses all plotting helpers from ``eps_ntu_breakdown`` but normalises the
F-factors by eps_cf(NTU, C_r=0) = 1 - exp(-NTU) instead of eps_cf,bal = NTU/(1+NTU).

Why two references?
-------------------
The **balanced (C_r = 1)** reference (default in ``eps_ntu_breakdown``) has the special property
that eps_cf,bal is *proportional to NTU*::

    eps_cf,bal(NTU) = NTU / (1 + NTU).

This makes the product rule for G exceptionally clean::

    G = eps_arr / NTU = F_Cr * F_xf * F_couple / (1 + NTU),       [balanced ref.]

where every factor is <= 1 and G < 1 always. The arrangement-only G baseline::

    F_xf / (1 + NTU) = eps_arr(NTU, C_r=1) / NTU                   [G at C_r=1]

varies between arrangements and is the key "cost of arrangement" curve.

The **isothermal (C_r -> 0)** reference asks instead: how does this exchanger compare to the
best possible single-stream (phase-change / condensing / evaporating) counterflow limit::

    eps_cf,0(NTU) = 1 - exp(-NTU).

F-factors with the isothermal reference::

    F_Cr     = eps_cf(NTU, C_r) / eps_cf,0       (counterflow unbalance vs isothermal limit; <= 1)
    F_xf     = eps_arr(NTU, 0)  / eps_cf,0        (arrangement at C_r=0 vs isothermal cf.)
    F_tot    = eps_arr(NTU, C_r) / eps_cf,0        (full ratio vs isothermal limit; <= 1)
    F_couple = F_tot / (F_Cr * F_xf)

The universal G = eps_arr / NTU is unchanged and always < 1::

    G = eps_arr / NTU,   Q = G * UA * dT_inlet   (NTU = UA / C_min).

Because eps_cf,0 = 1 - exp(-NTU) is *not* proportional to NTU, the link from F-factors to G is
less clean than with the balanced reference::

    G = F_tot * (1 - exp(-NTU)) / NTU.             [C_r=0 ref.; compare G = F_tot/(1+NTU) for balanced]

F_tot alone therefore does not directly give G here; you need the extra factor (1-exp(-NTU))/NTU.

Degeneracy of F_xf at C_r = 0
-------------------------------
``epsilon_ntu`` short-circuits to 1 - exp(-NTU) for *every* geometry when C_r = 0
(Kays & London 2-13a), so::

    F_xf = eps_arr(NTU, 0) / eps_cf,0 = 1    for every arrangement.

The F_xf plot is a flat line at 1 -- shown explicitly to demonstrate the degeneracy.

The "arrangement-only G baseline" analogue for the isothermal reference is::

    eps_arr(NTU, 0) / NTU = (1 - exp(-NTU)) / NTU.              [G at C_r=0]

This is also *arrangement-independent* (all geometries collapse to the same curve at C_r=0)
and is plotted as a single reference curve in place of the per-arrangement F_xf/(1+NTU) from
the balanced script. Contrast with the balanced reference where eps_arr(NTU,1)/NTU = F_xf/(1+NTU)
*does* vary with arrangement -- that is precisely what makes F_xf informative there.

**Contour plots (F_Cr, F_couple):** with this reference, factors often fall **below 1** everywhere,
so the "unity floor" colormap used in the balanced script would leave the field blank. This
script passes ``unity_floor=False`` and ``cr_ylim=None`` so colour limits and axes autoscale.

Consequence: with the C_r=0 reference the **arrangement effect lives entirely in F_couple**
(and in G at finite C_r). F_Cr and G are the most informative plots here.

Thermodynamic caveat
--------------------
Using C_r = 0 as a single global reference compares every operating point to a phase-change
limit that is independent of the actual C_r. It is useful for understanding the maximum possible
Q for a given UA and inlet temperature difference but does not replace a full temperature-corrected
analysis when both streams have finite capacity.

Run (from repo root)::

    uv run analysis/standalone_tools/eps_ntu_bkdwn_isoth_fluid.py

Figures are written to ``analysis/standalone_tools/figs/`` as ``eps_ntu_isoth_*.png``.
"""

from heat_exchanger.epsilon_ntu import epsilon_ntu

if __name__ == "__main__":
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    _tools = Path(__file__).resolve().parent
    if str(_tools) not in sys.path:
        sys.path.insert(0, str(_tools))

    from eps_ntu_breakdown import (
        REF_CR0_ISOTHERMAL,
        compute_factors,
        plot_fcr_contour,
        plot_ftot_fcouple_contours,
        plot_fxf_vs_ntu_arrangements,
        plot_g_vs_ntu_arrangements,
        save_figure,
    )

    NTU = np.linspace(0.1, 8.0, 60)
    Cr = np.linspace(0.01, 1.0, 40)
    ref = REF_CR0_ISOTHERMAL

    arrangements = [
        ("xf both unmixed", dict(exchanger_type="cross_flow", flow_type="unmixed")),
        ("xf Cmax mixed", dict(exchanger_type="cross_flow", flow_type="Cmax_mixed")),
        ("xf Cmin mixed", dict(exchanger_type="cross_flow", flow_type="Cmin_mixed")),
        ("coflow", dict(exchanger_type="aligned_flow", flow_type="coflow")),
    ]

    fxf_arrangements_cr0 = [
        ("Crossflow, both unmixed", dict(exchanger_type="cross_flow", flow_type="unmixed")),
        ("Crossflow, one fluid mixed", dict(exchanger_type="cross_flow", flow_type="Cmax_mixed")),
        ("Coflow", dict(exchanger_type="aligned_flow", flow_type="coflow")),
    ]

    res_unmixed = compute_factors(
        epsilon_ntu, NTU, Cr, exchanger_type="cross_flow", flow_type="unmixed", ref=ref
    )
    res_cmax = compute_factors(
        epsilon_ntu, NTU, Cr, exchanger_type="cross_flow", flow_type="Cmax_mixed", ref=ref
    )

    # --- Figure 1: F_Cr contour (float colour bounds; factors often < 1 vs isothermal ref.) ---
    fig1 = plot_fcr_contour(
        res_unmixed,
        title_prefix=r"$F_{Cr}$ · counterflow vs $\varepsilon_{\mathrm{cf},\,C_r\to 0}$ (any configuration)",
        unity_floor=False,
        cr_ylim=None,
    )
    save_figure(fig1, "eps_ntu_isoth_fcr")

    # --- Figure 2: F_tot and F_couple (float bounds for F_couple) ---
    fig2 = plot_ftot_fcouple_contours(
        res_unmixed,
        res_cmax,
        top_label="crossflow, both unmixed",
        bottom_label="crossflow, Cmax mixed",
        title_prefix=r"$F_{\mathrm{tot}}$ and $F_{\mathrm{couple}}$ · crossflow ($C_r\to 0$ ref.)",
        unity_floor_couple=False,
        cr_ylim=None,
    )
    save_figure(fig2, "eps_ntu_isoth_ftot_fcouple")

    # --- Figure 3: F_xf vs NTU (flat at 1 — degenerate with C_r=0 ref.) ---
    fig3 = plot_fxf_vs_ntu_arrangements(
        epsilon_ntu,
        NTU,
        Cr,
        fxf_arrangements_cr0,
        ref=ref,
        xf_state_note=(
            r"$F_{xf}=\varepsilon_{\mathrm{arr}}(\mathrm{NTU},0)/\varepsilon_{\mathrm{cf},0}\equiv 1$"
            r" (all geometries degenerate at $C_r=0$)"
        ),
        title_prefix="Arrangement factor",
    )
    save_figure(fig3, "eps_ntu_isoth_fxf")

    # --- Figure 4: eps_arr(NTU,0)/NTU = (1-exp(-NTU))/NTU — the C_r=0 G baseline ---
    fig4, ax4 = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ntu_vec = np.array(NTU)
    g_cr0 = (1.0 - np.exp(-ntu_vec)) / ntu_vec
    ax4.plot(ntu_vec, g_cr0, lw=2, color="C0", label=r"$(1-e^{-\mathrm{NTU}})/\mathrm{NTU}$")
    ax4.set_xlabel("NTU")
    ax4.set_ylabel(r"$\varepsilon_{\mathrm{arr}}(\mathrm{NTU},0)/\mathrm{NTU}$")
    ax4.set_title(
        r"$G|_{C_r=0}=(1-e^{-\mathrm{NTU}})/\mathrm{NTU}$ — arrangement-independent baseline"
        "\n"
        r"(analogue of $F_{xf}/(1+\mathrm{NTU})=\varepsilon_{\mathrm{arr}}(\mathrm{NTU},1)/\mathrm{NTU}$"
        r" from balanced ref., which varies by arrangement)"
    )
    ax4.legend()
    ax4.grid(alpha=0.3)
    save_figure(fig4, "eps_ntu_isoth_g_cr0_baseline")

    # --- Figure 5: G = eps/NTU for all arrangements at two C_r slices ---
    fig5 = plot_g_vs_ntu_arrangements(
        epsilon_ntu,
        NTU,
        Cr,
        arrangements,
        ref=ref,
        cr_solid=0.33,
        cr_dash=0.66,
        title_prefix=r"$G = \varepsilon/\mathrm{NTU}$ (universal, < 1 always)",
    )
    save_figure(fig5, "eps_ntu_isoth_g_arrangements")

    plt.show()
