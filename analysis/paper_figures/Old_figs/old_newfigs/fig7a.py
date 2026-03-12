"""
Figure 7a: Classical unavailable energy breakdown along optimal practical curve.
Similar to fig6a but with V_core/V_ref (d/d_ref) as x-axis instead of NTU.
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


def save_figures(base_name="fig7a"):
    """
    Save figures as SVG, TIFF, and HD PNG showing classical unavailable energy breakdown
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
    z_classical_thermal_only = df[opt_mask]["dqom_classical_thermal_only_opt"].values
    z_classical_combined = df[opt_mask]["dqom_classical_opt"].values

    if len(d_opt) == 0:
        raise ValueError("No optimal line data found in parquet file.")

    # Sort by d_opt for proper plotting
    sort_idx = np.argsort(d_opt)
    d_opt = d_opt[sort_idx]
    z_classical_thermal_only = z_classical_thermal_only[sort_idx]
    z_classical_combined = z_classical_combined[sort_idx]

    # Filter out NaN values
    valid_mask = np.isfinite(z_classical_thermal_only) & np.isfinite(z_classical_combined)
    d_opt = d_opt[valid_mask]
    z_classical_thermal_only = z_classical_thermal_only[valid_mask]
    z_classical_combined = z_classical_combined[valid_mask]

    if len(d_opt) == 0:
        raise ValueError("No valid data points found.")

    # Match fig6a figure size (triple column)
    fig = plt.figure(figsize=(6 / 2.54, 7 / 2.54))
    ax = plt.subplot(111)

    # Plot thermal-only line (no pressure drop)
    _ = ax.plot(  # line_thermal (unused; change back if needed for legend etc.)
        d_opt,
        z_classical_thermal_only,
        "k-",
        label="_nolegend_",
        zorder=2,
    )[0]

    # Plot combined line (with pressure drop)
    _ = ax.plot(  # line_combined (unused; change back if needed for legend etc.)
        d_opt,
        z_classical_combined,
        "k-",
        label="_nolegend_",
        zorder=3,
    )[0]

    # Fill between 0 and thermal line (thermal dissipation)
    ax.fill_between(d_opt, 0, z_classical_thermal_only, alpha=0.3, color="gray", zorder=1)

    # Fill between thermal and combined (viscous dissipation)
    ax.fill_between(
        d_opt,
        z_classical_thermal_only,
        z_classical_combined,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
        hatch="///",
        zorder=1,
        label="_nolegend_",
    )

    # Reference marker: diamond (D), black face, white edge, s=50 (match fig6a)
    idx_ref = np.argmin(np.abs(d_opt - D_REF))
    x_ref = d_opt[idx_ref]
    y_ref = z_classical_combined[idx_ref]
    ax.scatter(x_ref, y_ref, marker="D", facecolor="black", edgecolor="white", zorder=5, s=50)

    # Optimal marker: circle, black, s=50, white edge (match fig6a)
    # Find minimum of combined (optimal practical point)
    # idx_opt = np.nanargmin(z_classical_combined)
    # x_opt = d_opt[idx_opt]
    # y_opt = z_classical_combined[idx_opt]
    # ax.scatter(x_opt, y_opt, marker="o", facecolor="black", edgecolor="white", zorder=5, s=50)

    ax.set_title("")
    # ax.set_xlim(d_opt.min() * 0.95, d_opt.max() * 1.05)
    ax.set_xlim(d_opt.min(), 3)
    y_max = max(np.nanmax(z_classical_combined), np.nanmax(z_classical_thermal_only))
    ax.set_ylim(0, y_max * 1.1)
    ax.set_xlabel(r"Core Volume Ratio ($V_\mathrm{core}/V_\mathrm{ref}$ [-])")
    ax.set_ylabel(r"Change in Unavailable Energy ($\Delta \dot{Q}_0/\dot{Q}_{\mathrm{max}}$)")

    patch_thermal = mpatches.Patch(facecolor="gray", alpha=0.3, edgecolor="black", label="Thermal dissipation")
    patch_viscous = mpatches.Patch(
        facecolor="none", edgecolor="black", hatch="///", linewidth=1.5, label="Viscous dissipation"
    )

    # Legend box style to match fig6a
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
