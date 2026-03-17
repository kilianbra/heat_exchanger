"""
Combined plot: black and red lines from fig9 + two lines from fig8, all on one plot with no markers.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import fig8_no_cycle_model
import fig9_w_cycle_model

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figs_current")


def run_plot(base_name="fig10_combined"):
    """Plot all lines from fig8 and fig9 together, no markers."""
    # Get fig8 data (no cycle model: fuel only dashed, fuel+HEx solid)
    data8 = fig8_no_cycle_model.get_line_data()
    if data8 is None:
        print("Fig8: No valid data.")
        return

    m_hex_8, line_fuel_only_8, line_fuel_hex_8 = data8

    # Get fig9 data (cycle model: red dashed/solid, black dashed/solid)
    data9 = fig9_w_cycle_model.get_line_data()
    if data9 is None:
        print("Fig9: No valid data.")
        return

    m_hex_opt, line1, line3, m_hex_bc, line_bc, line_bc_mdot = data9

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

    # Fig8 lines (no cycle model)
    ax.plot(m_hex_8, line_fuel_only_8, "k--", linewidth=1.5, label="fuel only practical")
    ax.plot(m_hex_8, line_fuel_hex_8, "k-", linewidth=1.5, label="fuel + HEx practical")

    # Fig9 red lines (cycle model)
    ax.plot(m_hex_opt, line1, "r--", linewidth=1.5, label=r"fuel only cycle")
    ax.plot(m_hex_opt, line3, "r-", linewidth=1.5, label=r"fuel + HEx cycle")

    # Fig9 black lines (dQ^M variants)
    # if len(line_bc) > 0:
    #     ax.plot(m_hex_bc, line_bc, "k--", linewidth=1.2, label=r"$\Delta\dot{W}^M$: BC effect only (fig9)")
    # ax.plot(
    #     m_hex_opt,
    #     line_bc_mdot,
    #     "k-",
    #     linewidth=1.2,
    #     label=r"$\Delta\dot{W}^M$: BC + $\dot{m}$ (fig9)",
    # )

    # Red square at minimum of solid red line (line3)
    id_min_red = int(np.argmin(line3))
    ax.scatter(
        m_hex_opt[id_min_red],
        line3[id_min_red],
        color="red",
        s=25,
        zorder=6,
        marker="s",
        facecolor="red",
        edgecolor="white",
        linewidths=1,
    )
    # Black star at minimum of fig8 fuel + HEx line
    id_min_fig8 = int(np.argmin(line_fuel_hex_8))
    ax.scatter(
        m_hex_8[id_min_fig8],
        line_fuel_hex_8[id_min_fig8],
        color="black",
        s=90,
        zorder=5,
        marker="*",
        facecolor="black",
        edgecolor="white",
        linewidths=1,
    )

    ax.set_xlabel(r"Heat Exchanger (HEx) Core Mass $m_{\mathrm{HEx}}$ (kg)")
    ax.set_ylabel(r"Change in Take-off Mass $\Delta m$ (kg)")
    ax.legend(
        loc="upper right",
        ncol=1,
        fontsize=8,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.set_xlim(0, 60)
    ax.set_ylim(-100, 0)
    ax.set_yticks(np.arange(-100, 1, 20))

    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png", "pdf"]:
        path = os.path.join(save_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    run_plot(base_name="fig10_combined")
