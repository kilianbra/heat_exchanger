"""
Figure 7b: Practical unavailable energy breakdown along optimal practical curve.
Similar to fig6b but with V_core/V_ref (d/d_ref) as x-axis instead of NTU.
Reads optimal practical data from newfig5_practical_data.parquet.
"""

import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = Path(save_dir) / "newfig5_practical_data.parquet"

# Reference design point (d/d_ref = 1)
D_REF = 1.0


def save_figures(base_name="fig7b"):
    """
    Save figures as SVG, TIFF, and HD PNG showing practical unavailable energy breakdown
    along optimal practical curve vs V_core/V_ref (d/d_ref).
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file {DATA_FILE} not found. Please run newfig5_practical.py first to generate the data."
        )

    font_size = 8
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": font_size,
            "mathtext.fontset": "stix",
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
        }
    )

    # Load data from parquet
    df = pd.read_parquet(DATA_FILE)

    # Extract optimal line data (along practical optimum)
    opt_mask = df["d_opt"].notna()
    d_opt = df[opt_mask]["d_opt"].values
    z_practical_thermal_only = df[opt_mask]["dqom_practical_thermal_only_opt"].values
    z_practical_combined = df[opt_mask]["dqom_practical_opt"].values

    if len(d_opt) == 0:
        raise ValueError("No optimal line data found in parquet file.")

    # Sort by d_opt for proper plotting
    sort_idx = np.argsort(d_opt)
    d_opt = d_opt[sort_idx]
    z_practical_thermal_only = z_practical_thermal_only[sort_idx]
    z_practical_combined = z_practical_combined[sort_idx]

    # Filter out NaN values
    valid_mask = np.isfinite(z_practical_thermal_only) & np.isfinite(z_practical_combined)
    d_opt = d_opt[valid_mask]
    z_practical_thermal_only = z_practical_thermal_only[valid_mask]
    z_practical_combined = z_practical_combined[valid_mask]

    if len(d_opt) == 0:
        raise ValueError("No valid data points found.")

    # Match fig6b figure size (triple column 6 cm)
    fig = plt.figure(figsize=(6 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)

    # Plot thermal-only line (no pressure drop)
    line_thermal = ax.plot(
        d_opt,
        z_practical_thermal_only,
        "k-",
        label="_nolegend_",
        zorder=2,
    )[0]

    # Plot combined line (with pressure drop)
    line_combined = ax.plot(
        d_opt,
        z_practical_combined,
        "k-",
        label="_nolegend_",
        zorder=3,
    )[0]

    # Fill between thermal and 0 (thermal creation) - note: practical can be negative
    ax.fill_between(d_opt, z_practical_thermal_only, 0, alpha=0.3, color="gray", zorder=1)

    # Fill between combined and thermal (viscous dissipation)
    ax.fill_between(
        d_opt,
        z_practical_combined,
        z_practical_thermal_only,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
        hatch="///",
        zorder=1,
        label="_nolegend_",
    )

    # Optimal marker: circle, black, s=50, white edge (match fig6b)
    # Find minimum of combined (optimal practical point)
    # idx_optimum = np.nanargmin(z_practical_combined)
    # x_optimum = d_opt[idx_optimum]
    # y_optimum = z_practical_combined[idx_optimum]
    # ax.scatter(x_optimum, y_optimum, marker="o", facecolor="black", edgecolor="white", zorder=5, s=50)

    # print(f"Optimum d/d_ref: {x_optimum:.4f}")

    # Reference marker: diamond (D), black face, white edge, s=50 (match fig6b)
    idx_ref = np.argmin(np.abs(d_opt - D_REF))
    x_ref = d_opt[idx_ref]
    y_ref = z_practical_combined[idx_ref]
    ax.scatter(x_ref, y_ref, marker="D", facecolor="black", edgecolor="white", zorder=5, s=50)

    # Annotations with arrows (match fig6a style)
    # arrow_kw_opt = dict(arrowstyle="->", color="black", lw=1, shrinkB=10)
    # arrow_kw_ref = dict(arrowstyle="->", color="black", lw=1, shrinkB=10)
    # ax.annotate(
    #     "reference design",
    #     xy=(x_ref, y_ref),
    #     xytext=(x_ref + 0.1, y_ref + 0.01),
    #     fontsize=font_size,
    #     ha="left",
    #     arrowprops=arrow_kw_ref,
    # )
    # ax.annotate(
    #     "optimal design",
    #     xy=(x_optimum, y_optimum),
    #     xytext=(x_optimum - 0.1, y_optimum + 0.01),
    #     fontsize=font_size,
    #     ha="right",
    #     arrowprops=arrow_kw_opt,
    # )

    ax.set_title("")
    ax.set_xlim(d_opt.min() * 0.95, 3.0)
    y_min = min(np.nanmin(z_practical_combined), np.nanmin(z_practical_thermal_only))
    y_max = max(np.nanmax(z_practical_combined), np.nanmax(z_practical_thermal_only))
    ax.set_ylim(y_min * 1.1, max(0, y_max * 1.1))
    ax.set_xlabel(r"Core Volume Ratio ($V_\mathrm{core}/V_\mathrm{ref}$ [-])")
    ax.set_ylabel(r"Change in Unavailable Energy ($\Delta \dot{Q}_0^\mathrm{M}/\dot{Q}_{\mathrm{max}}$)")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

    # Legend: thermal creation and viscous dissipation (match fig6b box style)
    patch_thermal = mpatches.Patch(facecolor="gray", alpha=0.3, edgecolor="black", label="Thermal creation")
    patch_viscous = mpatches.Patch(
        facecolor="none", edgecolor="black", hatch="///", linewidth=1.5, label="Viscous dissipation"
    )
    ax.legend(
        handles=[patch_thermal, patch_viscous],
        loc="lower left",
        labelspacing=0.05,
        edgecolor="black",
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        fancybox=False,
    )

    plt.tight_layout(pad=0.5)

    fig.savefig(
        os.path.join(save_dir, f"{base_name}.svg"),
        dpi=300,
        facecolor="white",
        format="svg",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.tiff"),
        dpi=300,
        facecolor="white",
        format="tiff",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.png"),
        dpi=300,
        facecolor="white",
        format="png",
        bbox_inches=None,
        pad_inches=0,
    )
    fig.savefig(
        os.path.join(save_dir, f"{base_name}.pdf"),
        dpi=300,
        facecolor="white",
        format="pdf",
        bbox_inches=None,
        pad_inches=0,
    )

    plt.close(fig)
    print(f"Saved figures: {base_name}.svg, {base_name}.tiff, {base_name}.png, {base_name}.pdf")


if __name__ == "__main__":
    save_figures()
