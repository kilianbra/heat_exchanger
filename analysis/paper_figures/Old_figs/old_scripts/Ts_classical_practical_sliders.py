"""
Interactive T-s diagram for perfect gas (air) showing classical vs practical available energy.

- Isobars at p0 (1 bar) and p (stream pressure)
- Line T0 + T0*(s-s0)/cp through dead state (s0, T0)
- Vertical lines: classical (T0 → T, split at T0+T0(s-s0)/cp) and practical (T0 → T, split at Tse)
- Tse = T * (p0/p)^((gamma-1)/gamma)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# Air as perfect gas
R = 287.0  # J/(kg·K)
CP = 1005.0  # J/(kg·K)
GAMMA = 1.4

# Dead state
P0 = 1e5  # Pa (1 bar)
T0 = 300.0  # K

# Initial stream state
INIT_T = 500.0  # K
INIT_P = 2e5  # Pa (2 bar)


def entropy_change(T, p, T_ref, p_ref):
    """s - s_ref for perfect gas: cp*ln(T/T_ref) - R*ln(p/p_ref) [J/(kg·K)]"""
    return CP * np.log(T / T_ref) - R * np.log(p / p_ref)


def T_from_entropy_isobar(s, s0, p, p0):
    """T on isobar at p given s-s0. From s-s0 = cp*ln(T/T0) - R*ln(p/p0) => T = T0 * (p/p0)^(R/cp) * exp((s-s0)/cp)"""
    return T0 * (p / p0) ** (R / CP) * np.exp((s - s0) / CP)


def Tse_from_T_p(T, p, p0):
    """Isentropic-equivalent temperature at p0: Tse = T * (p0/p)^((gamma-1)/gamma)"""
    return T * (p0 / p) ** ((GAMMA - 1) / GAMMA)


def update_plot(val=None):
    T = slider_T.val
    p = slider_p.val * 1e5  # bar -> Pa

    # Reference entropy at (T0, p0)
    s0 = 0.0  # arbitrary reference
    s_stream = entropy_change(T, p, T0, P0)
    Tse = Tse_from_T_p(T, p, P0)

    # Classical: bar runs from T0 to T. Boundary at T0 + T0(s-s0)/cp.
    # Unavailable = T0 to T0+T0(s-s0)/cp, available = T0+T0(s-s0)/cp to T
    T_unavail_boundary = T0 + T0 * (s_stream - s0) / CP  # boundary between unavail and avail

    # Practical: unavailable = Tse - T0, available = T - Tse (used in vertical bar/fills)

    # s range for plotting (span dead state and stream)
    s_min = min(0, s_stream) - 50
    s_max = max(0, s_stream) + 50
    s_plot = np.linspace(s_min, s_max, 300)

    # Isobar at p0
    T_p0 = T_from_entropy_isobar(s_plot, s0, P0, P0)
    line_p0.set_data(s_plot, T_p0)

    # Isobar at p
    T_p = T_from_entropy_isobar(s_plot, s0, p, P0)
    line_p.set_data(s_plot, T_p)

    # Line T0 + T0*(s-s0)/cp (through dead state (s0, T0))
    T_unavail_line = T0 + T0 * (s_plot - s0) / CP
    # Clip to reasonable T range for display
    valid = (T_unavail_line > T0 * 0.5) & (T_unavail_line < 1.5 * T)
    line_unavail.set_data(s_plot[valid], T_unavail_line[valid])

    # Stream state point (black) and Tse (red square)
    pt_stream.set_data([s_stream], [T])
    pt_stream.set_visible(True)
    pt_Tse.set_data([s_stream], [Tse])
    pt_Tse.set_visible(True)

    # Breakdown segments: offset available to each side to show difference
    avail_offset = 4.0  # shift avail bars away from centre
    unavail_offset = 12.0  # unavailable further out
    s_class_avail = s_stream - avail_offset
    s_class_unavail = s_stream - unavail_offset
    s_prac_avail = s_stream + avail_offset
    s_prac_unavail = s_stream + unavail_offset

    # Classical: available left of centre; unavailable further left
    line_class_avail.set_data([s_class_avail, s_class_avail], [T_unavail_boundary, T])
    line_class_unavail.set_data([s_class_unavail, s_class_unavail], [T0, T_unavail_boundary])

    # Practical: available right of centre; unavailable further right
    line_prac_avail.set_data([s_prac_avail, s_prac_avail], [Tse, T])
    line_prac_unavail.set_data([s_prac_unavail, s_prac_unavail], [T0, Tse])

    # Y-axis labels T and T0 (at right edge)
    label_T.set_x(s_max)
    label_T.set_y(T)
    label_T0.set_x(s_max)
    label_T0.set_y(T0)

    # Isobar labels p, p0
    s_label = s_max - 15
    T_at_p0 = float(T_from_entropy_isobar(np.array([s_label]), s0, P0, P0))
    T_at_p = float(T_from_entropy_isobar(np.array([s_label]), s0, p, P0))
    label_p0.set_x(s_label)
    label_p0.set_y(T_at_p0)
    label_p.set_x(s_label)
    label_p.set_y(T_at_p)

    T_hi = T * 1.05
    ax.set_xlim(0, s_max)
    ax.set_ylim(T0, T_hi + 20)

    fig.canvas.draw_idle()


if __name__ == "__main__":
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
    fig, ax = plt.subplots(figsize=(9 / 2.54, 7 / 2.54))
    plt.subplots_adjust(right=0.82, bottom=0.25)

    # Initial compute
    s0 = 0.0
    s_stream = entropy_change(INIT_T, INIT_P, T0, P0)
    Tse = Tse_from_T_p(INIT_T, INIT_P, P0)
    T_unavail_boundary = T0 + T0 * (s_stream - s0) / CP

    s_min = min(0, s_stream) - 50
    s_max = max(0, s_stream) + 50
    s_plot = np.linspace(s_min, s_max, 300)

    # Isobars
    T_p0 = T_from_entropy_isobar(s_plot, s0, P0, P0)
    T_p = T_from_entropy_isobar(s_plot, s0, INIT_P, P0)
    T_unavail_line = T0 + T0 * (s_plot - s0) / CP
    valid = (T_unavail_line > T0 * 0.5) & (T_unavail_line < 1.5 * INIT_T)

    (line_p0,) = ax.plot(s_plot, T_p0, "b-", lw=1.5)
    (line_p,) = ax.plot(s_plot, T_p, "b--", lw=1.5)
    (line_unavail,) = ax.plot(
        s_plot[valid],
        T_unavail_line[valid],
        "gray",
        lw=1.2,
        linestyle=":",
    )

    # Stream and Tse points (on top of lines)
    (pt_stream,) = ax.plot([s_stream], [INIT_T], "ko", ms=6, zorder=5)
    (pt_Tse,) = ax.plot([s_stream], [Tse], "rs", ms=5, zorder=5)

    # Breakdown segments: classical left, practical right; avail and unavail offset
    avail_offset = 4.0
    unavail_offset = 12.0
    s_class_avail = s_stream - avail_offset
    s_class_unavail = s_stream - unavail_offset
    s_prac_avail = s_stream + avail_offset
    s_prac_unavail = s_stream + unavail_offset

    (line_class_avail,) = ax.plot(
        [s_class_avail, s_class_avail],
        [T_unavail_boundary, INIT_T],
        "g-",
        lw=2,
    )
    (line_class_unavail,) = ax.plot(
        [s_class_unavail, s_class_unavail],
        [T0, T_unavail_boundary],
        "g-",
        lw=2,
        alpha=0.6,
    )
    (line_prac_avail,) = ax.plot(
        [s_prac_avail, s_prac_avail],
        [Tse, INIT_T],
        "m-",
        lw=2,
    )
    (line_prac_unavail,) = ax.plot(
        [s_prac_unavail, s_prac_unavail],
        [T0, Tse],
        "m-",
        lw=2,
        alpha=0.6,
    )

    # Y-axis labels T and T0
    label_T = ax.text(s_max, INIT_T, r"$T$", va="center", ha="left")
    label_T0 = ax.text(s_max, T0, r"$T_0$", va="center", ha="left")

    # Isobar labels p, p0
    s_label = s_max - 15
    T_at_p0 = float(T_from_entropy_isobar(np.array([s_label]), s0, P0, P0))
    T_at_p = float(T_from_entropy_isobar(np.array([s_label]), s0, INIT_P, P0))
    label_p0 = ax.text(s_label, T_at_p0, r"$p_0$", va="center", ha="left")
    label_p = ax.text(s_label, T_at_p, r"$p$", va="center", ha="left")

    ax.set_xlabel(r"Entropy $s - s_0$ (J/(kg·K))")
    ax.set_ylabel(r"Temperature $T$ (K)")
    ax.set_xlim(s_min, s_max)
    ax.set_ylim(T0, INIT_T * 1.15)

    # Sliders
    ax_T = plt.axes([0.15, 0.12, 0.5, 0.02])
    ax_p = plt.axes([0.15, 0.07, 0.5, 0.02])
    slider_T = Slider(ax_T, r"$T$ (K)", 310, 1200, valinit=INIT_T, valfmt="%.0f")
    slider_p = Slider(ax_p, r"$p$ (bar)", 0.2, 10.0, valinit=INIT_P / 1e5, valfmt="%.2f")

    slider_T.on_changed(update_plot)
    slider_p.on_changed(update_plot)

    plt.show()
