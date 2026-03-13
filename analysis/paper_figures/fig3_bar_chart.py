"""
Final Figure 3: Classical (exergy) and Practical (euergy) availability breakdown.

Computes change in available energy at a single operating point.
Terminal output optional; waterfall bar charts (no savefig yet).
"""

# import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow import of xflow when run from project root or paper_figures dir
# _script_dir = Path(__file__).resolve().parent
# if str(_script_dir) not in sys.path:
#     sys.path.insert(0, str(_script_dir))
from xflow import classical_unavailable_creation_hex, practical_unavailable_creation_hex

# Single-column figure size from newfig7 (9 cm x 7 cm)
FIG_SINGLE_COL = (9 / 2.54 / 2, 7 / 2.54)


# ---------------------------------------------------------------------------
# Input parameters (stagnation values, Mach=0 → static = stagnation)
# ---------------------------------------------------------------------------
T_HIN_STAG = 885.0  # K
P_HIN = 1.0  # bar
T_CIN_STAG = 588.0  # K
P_CIN = 9.0  # bar
T0_STAG = 288.0  # K
P0 = 1.0  # bar

MACH_H = 0.0
MACH_C = 0.0

GAMMA_H = 1.4
GAMMA_C = 1.4
CP_H = 1070.0  # J/(kg·K)
CP_C = 1070.0  # J/(kg·K)
MDOT_H = 2.3  # kg/s
MDOT_C = 2.3  # kg/s

EPSILON = 0.65
DP_HOT_PCT = 0.03  # 36  # 3.6%
DP_COLD_PCT = 0.03  # 22  # 2.2%

# Waterfall chart: bigger = thicker bars, smaller gaps (try 1.2, 1.5, etc.)
BAR_WIDTH_SCALE = 2.0


def static_temperature_from_stagnation(T_stag, Mach, gamma):
    """T_static = T_stag / (1 + (gamma-1)/2 * M^2)"""
    return T_stag / (1.0 + (gamma - 1.0) / 2.0 * Mach**2)


def static_pressure_from_stagnation(P_stag, Mach, gamma):
    """P_static = P_stag / (1 + (gamma-1)/2 * M^2)"""
    return P_stag / (1.0 + (gamma - 1.0) / 2.0 * Mach**2) ** (gamma / (gamma - 1.0))


def run(do_print=False):
    # Static temperatures (t_dead_over_t_cold_in must use static, not stagnation)
    T0_static = static_temperature_from_stagnation(T0_STAG, 0.0, GAMMA_H)
    T_cin_static = static_temperature_from_stagnation(T_CIN_STAG, MACH_C, GAMMA_C)
    P_hin_static = static_pressure_from_stagnation(P_HIN, MACH_H, GAMMA_H)
    P_cin_static = static_pressure_from_stagnation(P_CIN, MACH_C, GAMMA_C)

    t = T_HIN_STAG / T_CIN_STAG
    t_dead_over_t_cold_in = T0_static / T_cin_static
    p_cold_in_over_p_hot_in = P_cin_static / P_hin_static
    p_dead_over_p_hot_in = P0 / P_hin_static

    dp_hot = DP_HOT_PCT
    dp_cold = DP_COLD_PCT

    eps_a = np.atleast_1d(EPSILON)
    dp_h_a = np.atleast_1d(dp_hot)
    dp_c_a = np.atleast_1d(dp_cold)
    mask = np.ones_like(eps_a, dtype=bool)

    # --- Total (with pressure drop) ---
    av_class_total = -classical_unavailable_creation_hex(
        eps_a,
        t,
        dp_h_a,
        dp_c_a,
        mask,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        gamma=GAMMA_H,
    )
    av_prac_total = -practical_unavailable_creation_hex(
        eps_a,
        t,
        dp_h_a,
        dp_c_a,
        mask,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=GAMMA_H,
    )
    av_class_total = float(np.atleast_1d(av_class_total).flat[0])
    av_prac_total = float(np.atleast_1d(av_prac_total).flat[0])

    # --- Thermal only (no pressure drop) ---
    dp_h_0 = np.zeros_like(dp_h_a)
    dp_c_0 = np.zeros_like(dp_c_a)
    av_class_thermal = -classical_unavailable_creation_hex(
        eps_a, t, dp_h_0, dp_c_0, mask, t_dead_over_t_cold_in=t_dead_over_t_cold_in, gamma=GAMMA_H
    )
    av_prac_thermal = -practical_unavailable_creation_hex(
        eps_a,
        t,
        dp_h_0,
        dp_c_0,
        mask,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=GAMMA_H,
    )
    av_class_thermal = float(np.atleast_1d(av_class_thermal).flat[0])
    av_prac_thermal = float(np.atleast_1d(av_prac_thermal).flat[0])

    # --- Pure viscous hot (eps=0, dp_hot only) ---
    eps_0 = np.atleast_1d(0.0)
    dp_h_only = np.atleast_1d(dp_hot)
    dp_c_zero = np.zeros_like(dp_h_only)
    av_class_visc_h = -classical_unavailable_creation_hex(
        eps_0, t, dp_h_only, dp_c_zero, mask, t_dead_over_t_cold_in=t_dead_over_t_cold_in, gamma=GAMMA_H
    )
    av_prac_visc_h = -practical_unavailable_creation_hex(
        eps_0,
        t,
        dp_h_only,
        dp_c_zero,
        mask,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=GAMMA_H,
    )
    av_class_visc_h = float(np.atleast_1d(av_class_visc_h).flat[0])
    av_prac_visc_h = float(np.atleast_1d(av_prac_visc_h).flat[0])

    # --- Pure viscous cold (eps=0, dp_cold only) ---
    dp_h_zero = np.zeros_like(np.atleast_1d(dp_cold))
    dp_c_only = np.atleast_1d(dp_cold)
    av_class_visc_c = -classical_unavailable_creation_hex(
        eps_0, t, dp_h_zero, dp_c_only, mask, t_dead_over_t_cold_in=t_dead_over_t_cold_in, gamma=GAMMA_H
    )
    av_prac_visc_c = -practical_unavailable_creation_hex(
        eps_0,
        t,
        dp_h_zero,
        dp_c_only,
        mask,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=GAMMA_H,
    )
    av_class_visc_c = float(np.atleast_1d(av_class_visc_c).flat[0])
    av_prac_visc_c = float(np.atleast_1d(av_prac_visc_c).flat[0])

    # --- Breakdown 1: Thermal + combined viscous ---
    delta_av_viscous_class = av_class_total - av_class_thermal
    delta_av_viscous_prac = av_prac_total - av_prac_thermal

    # --- Breakdown 2: interaction term ---
    delta_rest_av_class = av_class_total - av_class_thermal - av_class_visc_h - av_class_visc_c
    delta_rest_av_prac = av_prac_total - av_prac_thermal - av_prac_visc_h - av_prac_visc_c

    if do_print:
        _print_breakdown(
            av_class_total,
            av_class_thermal,
            delta_av_viscous_class,
            av_prac_total,
            av_prac_thermal,
            delta_av_viscous_prac,
            av_class_visc_h,
            av_class_visc_c,
            delta_rest_av_class,
            av_prac_visc_h,
            av_prac_visc_c,
            delta_rest_av_prac,
            t,
            t_dead_over_t_cold_in,
            T0_static,
            T_cin_static,
        )

    return {
        "av_class_total": av_class_total,
        "av_class_thermal": av_class_thermal,
        "delta_av_viscous_class": delta_av_viscous_class,
        "av_prac_total": av_prac_total,
        "av_prac_thermal": av_prac_thermal,
        "delta_av_viscous_prac": delta_av_viscous_prac,
        "av_class_visc_h": av_class_visc_h,
        "av_class_visc_c": av_class_visc_c,
        "delta_rest_av_class": delta_rest_av_class,
        "av_prac_visc_h": av_prac_visc_h,
        "av_prac_visc_c": av_prac_visc_c,
        "delta_rest_av_prac": delta_rest_av_prac,
    }


def _print_breakdown(
    av_class_total,
    av_class_thermal,
    delta_av_viscous_class,
    av_prac_total,
    av_prac_thermal,
    delta_av_viscous_prac,
    av_class_visc_h,
    av_class_visc_c,
    delta_rest_av_class,
    av_prac_visc_h,
    av_prac_visc_c,
    delta_rest_av_prac,
    t,
    t_dead_over_t_cold_in,
    T0_static,
    T_cin_static,
):
    def fmt(x, width=10):
        return f"{x:+{width}.3f}"

    factor_pct = 100.0

    print("=" * 60)
    print("Final Fig 3: Availability breakdown (all / Q_max)")
    print("=" * 60)
    print(f"Inputs: eps={EPSILON}, dp_hot={DP_HOT_PCT * 100:.1f}%, dp_cold={DP_COLD_PCT * 100:.1f}%")
    print(f"T_h_in/T_c_in = {t:.4f}, T_c_in/T_0 = {1 / t_dead_over_t_cold_in:.4f}")
    print(f"T0_static = {T0_static:.1f} K, T_cin_static = {T_cin_static:.1f} K")
    print()

    print("--- Breakdown 1: Thermal + Viscous (combined) (percentage points) ---")
    print("Classical (exergy):")
    print(f"  {'Total (change in available):':34} {fmt(av_class_total * factor_pct)} %")
    print(f"  {'Thermal only:':34} {fmt(av_class_thermal * factor_pct)} %")
    print(f"  {'Viscous (combined):':34} {fmt(delta_av_viscous_class * factor_pct)} %  (unavailable creation)")
    print()
    print("Practical (euergy):")
    print(f"  {'Total (change in available):':34} {fmt(av_prac_total * factor_pct)} %")
    print(f"  {'Thermal only:':34} {fmt(av_prac_thermal * factor_pct)} %")
    print(f"  {'Viscous (combined):':34} {fmt(delta_av_viscous_prac * factor_pct)} %  (unavailable creation)")
    print()

    print("--- Breakdown 2: Thermal + Visc_hot + Visc_cold + Interaction (percentage points) ---")
    print("Classical (exergy):")
    print(f"  {'Thermal only:':34} {fmt(av_class_thermal * factor_pct)} %")
    print(f"  {'Pure viscous hot:':34} {fmt(av_class_visc_h * factor_pct)} %  (unavailable)")
    print(f"  {'Pure viscous cold:':34} {fmt(av_class_visc_c * factor_pct)} %  (unavailable)")
    print(f"  {'Interaction:':34} {fmt(delta_rest_av_class * factor_pct)} %")
    print(
        f"  {'Sum check:':34} {fmt((av_class_thermal + av_class_visc_h + av_class_visc_c + delta_rest_av_class) * factor_pct)} % (expect total={av_class_total * factor_pct:.3f} %)"
    )
    print()
    print("Practical (euergy):")
    print(f"  {'Thermal only:':34} {fmt(av_prac_thermal * factor_pct)} %")
    print(f"  {'Pure viscous hot:':34} {fmt(av_prac_visc_h * factor_pct)} %  (unavailable)")
    print(f"  {'Pure viscous cold:':34} {fmt(av_prac_visc_c * factor_pct)} %  (unavailable)")
    print(f"  {'Interaction:':34} {fmt(delta_rest_av_prac * factor_pct)} %")
    print(
        f"  {'Sum check:':34} {fmt((av_prac_thermal + av_prac_visc_h + av_prac_visc_c + delta_rest_av_prac) * factor_pct)} % (expect total={av_prac_total * factor_pct:.3f} %)"
    )
    print("=" * 60)


def _hide_top_ytick(ax):
    """Remove the highest y-tick so it doesn't overlap the arrow."""
    HIDE_TOP_YTICK = True
    if HIDE_TOP_YTICK:
        tol = 1e-9
    else:
        tol = -1e-9
    ticks = ax.get_yticks()
    y_max = ax.get_ylim()[1]
    ax.set_yticks([t for t in ticks if t < y_max - tol])


def _style_axes_minimal(ax):
    """Remove box, keep horizontal line at 0 and left spine."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))


def _add_y_arrow_at_tip(ax):
    """Add arrow at top of y-axis. Uses axes fraction so arrowhead reaches the very top."""
    ax.annotate(
        "",
        xy=(0, 1.02),
        xycoords="axes fraction",
        xytext=(0, 0.94),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
    )


def _style_y_axis_post(ax, framework="classical"):
    """Hide top ytick + add arrow (if SHOW_Y_ARROW), or keep top tick. Hide y labels for practical when empty."""
    ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(round(x * 100))}%"))
    if SHOW_Y_ARROW:
        _hide_top_ytick(ax)
        _add_y_arrow_at_tip(ax)
    if framework == "practical" and YLABEL_PRACTICAL == "":
        ax.tick_params(axis="y", labelleft=False)


YLABEL_CLASSICAL = r"$\varepsilon^{\mathrm{C}} = (\Delta \dot{W}^{\mathrm{C}}_{\mathrm{A,hot}} + \Delta \dot{W}^{\mathrm{C}}_{\mathrm{A,cold}}) / \dot{Q}_{\mathrm{max}}$"
YLABEL_PRACTICAL = r"$\varepsilon^{\mathrm{P}} = (\Delta \dot{W}^{\mathrm{P}}_{\mathrm{A,hot}} + \Delta \dot{W}^{\mathrm{P}}_{\mathrm{A,cold}}) / \dot{Q}_{\mathrm{max}}$"

# False: no arrow, show top ytick (0.3). True: arrow at top, hide top ytick.
SHOW_Y_ARROW = False


def _plot_waterfall_breakdown1(ax, total, viscous, thermal, labels=None, framework="classical"):
    """
    Waterfall: netsum (hashed) -> viscous (starts at 0) -> thermal (starts from viscous).
    """
    if labels is None:
        labels = ("Total", "Viscous", "Thermal")
    bar_w = min(0.95, 0.35 * BAR_WIDTH_SCALE)
    xs = [0, 1, 2]
    # Bar 0: 0 to total (netsum, hashed)
    # Bar 1: 0 to viscous (first viscous starts at 0)
    # Bar 2: viscous to total (thermal starts from last viscous)
    bottoms = [0, 0, viscous]
    heights = [total, viscous, total - viscous]
    colors = ["gray", "white", "lightgray"]
    hatches = ["///", "", ""]
    edge = "black"

    bar_edge_lw = 0.5
    for i, (x, bot, h) in enumerate(zip(xs, bottoms, heights, strict=True)):
        fc = colors[i]
        hatch = hatches[i]
        if hatch:
            ax.bar(x, h, bar_w, bottom=bot, facecolor="none", edgecolor=edge, hatch=hatch, linewidth=bar_edge_lw)
        else:
            ax.bar(x, h, bar_w, bottom=bot, facecolor=fc, edgecolor=edge, linewidth=bar_edge_lw)

    # Dotted connectors between bars (skip i=0 to avoid duplicate at y=total)
    for i in range(1, len(xs) - 1):
        y_conn = bottoms[i] + heights[i]
        ax.plot([xs[i] + bar_w / 2, xs[i + 1] + bar_w / 2], [y_conn, y_conn], "k:", linewidth=bar_edge_lw)
    # Big dashed line: end of net sum to start of thermal
    ax.plot([xs[0] + bar_w / 2, xs[-1] - bar_w / 2], [total, total], "k--", linewidth=bar_edge_lw, zorder=0)
    ax.set_xticks([])
    y_lowest = min(min(bottoms[i], bottoms[i] + heights[i]) for i in range(len(xs)))
    label_offset = 0.02
    for xi, label in zip(xs, labels, strict=True):
        ax.text(xi, y_lowest - label_offset, label, ha="center", va="top", fontsize=8)
    ax.set_ylabel(YLABEL_CLASSICAL if framework == "classical" else YLABEL_PRACTICAL)
    ax.axhline(0, color="k", linewidth=0.5)
    _style_axes_minimal(ax)


# Labels for saved figures: classical uses full "Viscous (hot/cold)", practical uses short form
LABELS_BREAKDOWN2_CLASSICAL = ("Total", "Viscous (hot)", "Viscous (cold)", "Coupling", "Thermal")
LABELS_BREAKDOWN2_PRACTICAL = ("Total", "Visc. (h)", "Visc. (c)", "Coupling", "Thermal")


def _plot_waterfall_breakdown2(
    ax,
    total,
    visc_h,
    visc_c,
    interact,
    thermal,
    labels=None,
    has_interaction=True,
    framework="classical",
    for_save=False,
):
    """
    Waterfall: netsum (hashed) -> visc_hot (starts 0) -> visc_cold -> interaction -> thermal.
    Thermal starts from last viscous/interaction term.
    for_save=True: rotated labels, compact layout (fig3a/3b). for_save=False: horizontal labels, normal bar width (plt.show).
    """
    if labels is None:
        labels = ("Total", "Visc. (hot)", "Visc. (cold)", "Interaction", "Thermal")
    if not has_interaction and len(labels) == 5:
        labels = [lb for i, lb in enumerate(labels) if i != 3]
    bar_scale = BAR_WIDTH_SCALE if for_save else 1.0
    bar_w = min(0.95, 0.35 * bar_scale)
    n_bars = 5 if has_interaction else 4
    xs = list(range(n_bars))
    # First viscous starts at 0; thermal starts from last viscous/interaction
    # Cumulative: 0, visc_h, visc_h+visc_c, visc_h+visc_c+interact (=thermal level), total
    if has_interaction:
        bottoms = [0, 0, visc_h, visc_h + visc_c, visc_h + visc_c + interact]
        heights = [total, visc_h, visc_c, interact, total - (visc_h + visc_c + interact)]
    else:
        bottoms = [0, 0, visc_h, visc_h + visc_c]
        heights = [total, visc_h, visc_c, total - (visc_h + visc_c)]

    colors = (
        ["gray", "white", "white", "white", "lightgray"] if has_interaction else ["gray", "white", "white", "lightgray"]
    )
    hatches = ["///", "", "", "", ""] if has_interaction else ["///", "", "", ""]

    bar_edge_lw = 0.5
    for i, (x, bot, h) in enumerate(zip(xs, bottoms, heights, strict=True)):
        if hatches[i]:
            ax.bar(
                x, h, bar_w, bottom=bot, facecolor="none", edgecolor="black", hatch=hatches[i], linewidth=bar_edge_lw
            )
        else:
            ax.bar(x, h, bar_w, bottom=bot, facecolor=colors[i], edgecolor="black", linewidth=bar_edge_lw)

    # Dotted connectors between bars (skip i=0 to avoid duplicate at y=total)
    for i in range(1, len(xs) - 1):
        y_conn = bottoms[i] + heights[i]
        ax.plot([xs[i] + bar_w / 2, xs[i + 1] + bar_w / 2], [y_conn, y_conn], "k:", linewidth=bar_edge_lw)
    # Big dashed line: end of net sum to start of thermal
    ax.plot([xs[0] + bar_w / 2, xs[-1] - bar_w / 2], [total, total], "k--", linewidth=bar_edge_lw, zorder=0)
    ax.set_xticks([])
    y_lowest = min(min(bottoms[i], bottoms[i] + heights[i]) for i in range(len(xs)))
    label_offset = 0.005
    if for_save:
        # Rotated labels; classical above y=0, practical below lowest
        if framework == "classical":
            y_label, va_label = 0.0 + label_offset, "bottom"
        else:
            y_label, va_label = y_lowest - label_offset, "top"
        rot = 90
    else:
        # Horizontal labels below for both (interactive)
        y_label, va_label = y_lowest - label_offset, "top"
        rot = 0
    for xi, label in zip(xs, labels, strict=True):
        ax.text(xi, y_label, label, ha="center", va=va_label, fontsize=8, rotation=rot)
    ax.set_ylabel(YLABEL_CLASSICAL if framework == "classical" else YLABEL_PRACTICAL)
    ax.axhline(0, color="k", linewidth=0.5)
    _style_axes_minimal(ax)


def plot_waterfalls(data=None, do_print=False):
    """Compute data (if not provided), optionally print, and show waterfall charts.
    Layout: 2 rows (breakdown 1 top, breakdown 2 bottom), 2 cols (classical | practical).
    """
    d = data if isinstance(data, dict) else run(do_print=do_print)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "mathtext.fontset": "stix",
        }
    )

    w, h = FIG_SINGLE_COL
    fig, axes = plt.subplots(2, 2, figsize=(2 * w, 2 * h))  # Normal width for interactive view
    (ax_c1, ax_p1), (ax_c2, ax_p2) = axes

    YLIM_PRACTICAL = (-0.1, 0.3)  # bottom, top
    YLIM_CLASSICAL = YLIM_PRACTICAL  # (-0.3, 0.1)  # bottom, top: 0.1 at top, -0.3 at bottom

    _plot_waterfall_breakdown1(
        ax_c1, d["av_class_total"], d["delta_av_viscous_class"], d["av_class_thermal"], framework="classical"
    )
    ax_c1.set_title("Classical")
    ax_c1.set_ylim(YLIM_CLASSICAL)
    _style_y_axis_post(ax_c1, "classical")
    _plot_waterfall_breakdown1(
        ax_p1, d["av_prac_total"], d["delta_av_viscous_prac"], d["av_prac_thermal"], framework="practical"
    )
    ax_p1.set_title("Practical")
    ax_p1.set_ylim(YLIM_PRACTICAL)
    _style_y_axis_post(ax_p1, "practical")

    _plot_waterfall_breakdown2(
        ax_c2,
        d["av_class_total"],
        d["av_class_visc_h"],
        d["av_class_visc_c"],
        d["delta_rest_av_class"],
        d["av_class_thermal"],
        has_interaction=abs(d["delta_rest_av_class"]) > 1e-10,
        framework="classical",
    )
    ax_c2.set_title("Classical")
    ax_c2.set_ylim(YLIM_CLASSICAL)
    _style_y_axis_post(ax_c2, "classical")
    _plot_waterfall_breakdown2(
        ax_p2,
        d["av_prac_total"],
        d["av_prac_visc_h"],
        d["av_prac_visc_c"],
        d["delta_rest_av_prac"],
        d["av_prac_thermal"],
        has_interaction=True,
        framework="practical",
    )
    ax_p2.set_title("Practical")
    ax_p2.set_ylim(YLIM_PRACTICAL)
    _style_y_axis_post(ax_p2, "practical")

    plt.tight_layout(pad=0.5)
    # Big row titles above top and bottom halves
    fig.text(0.5, 0.98, "Breakdown 1", ha="center", fontsize=10)
    fig.text(0.5, 0.5, "Breakdown 2", ha="center", fontsize=10)
    plt.show()


def save_breakdown2_figures(data=None, do_print=False, save_dir=None):
    """
    Save Breakdown 2 as two separate figures: fig3a_bar_c (classical), fig3b_bar_p (practical).
    No title; viscous labels written fully with (hot) / (cold) on second line.
    Formats: .svg, .tiff, .png, and .pdf
    """
    d = data if isinstance(data, dict) else run(do_print=do_print)
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "Figs_current"
    save_dir = Path(save_dir)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "mathtext.fontset": "stix",
        }
    )

    YLIM_PRACTICAL = (-0.1, 0.3)
    YLIM_CLASSICAL = YLIM_PRACTICAL

    has_inter_class = abs(d["delta_rest_av_class"]) > 1e-10
    labels_class = (
        [lb for i, lb in enumerate(LABELS_BREAKDOWN2_CLASSICAL) if i != 3]
        if not has_inter_class
        else list(LABELS_BREAKDOWN2_CLASSICAL)
    )
    labels_prac = list(LABELS_BREAKDOWN2_PRACTICAL)

    w, h = FIG_SINGLE_COL

    # Figure 3a: Classical
    fig_a, ax_a = plt.subplots(figsize=(w, h))
    _plot_waterfall_breakdown2(
        ax_a,
        d["av_class_total"],
        d["av_class_visc_h"],
        d["av_class_visc_c"],
        d["delta_rest_av_class"],
        d["av_class_thermal"],
        labels=labels_class,
        has_interaction=has_inter_class,
        framework="classical",
        for_save=True,
    )
    ax_a.set_title("")
    ax_a.set_ylim(YLIM_CLASSICAL)
    _style_y_axis_post(ax_a, "classical")
    fig_a.tight_layout(pad=0.5)
    for ext in ["svg", "tiff", "png", "pdf"]:
        fig_a.savefig(Path(save_dir) / f"fig3a_bar_c.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig_a)

    # Figure 3b: Practical
    fig_b, ax_b = plt.subplots(figsize=(w, h))
    _plot_waterfall_breakdown2(
        ax_b,
        d["av_prac_total"],
        d["av_prac_visc_h"],
        d["av_prac_visc_c"],
        d["delta_rest_av_prac"],
        d["av_prac_thermal"],
        labels=labels_prac,
        has_interaction=True,
        framework="practical",
        for_save=True,
    )
    ax_b.set_title("")
    ax_b.set_ylim(YLIM_PRACTICAL)
    _style_y_axis_post(ax_b, "practical")
    ax_b.tick_params(axis="y", labelleft=False)  # fig3b: no tick numbers, just y-axis title
    fig_b.tight_layout(pad=0.5)
    for ext in ["svg", "tiff", "png", "pdf"]:
        fig_b.savefig(Path(save_dir) / f"fig3b_bar_p.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig_b)

    print(f"Saved fig3a_bar_c and fig3b_bar_p (.svg, .tiff, .png, .pdf) to {save_dir}")


if __name__ == "__main__":
    DO_PRINT = False  # Set True to print availability breakdown to terminal
    SHOW_INTERACTIVE = False  # False to skip plt.show(), only save
    data = run(do_print=DO_PRINT)
    if SHOW_INTERACTIVE:
        plot_waterfalls(data)
    save_breakdown2_figures(data)
