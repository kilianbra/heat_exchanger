"""
Figure 7: Two subplots showing:
- Left: dQ0/Qmax (classical) along optimal practical unavailability curve vs d/d_ref
- Right: dQ0^M/Qmax (practical) along optimal practical unavailability curve vs d/d_ref
Reads optimal data from newfig5_practical_data.parquet.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = Path(save_dir) / "newfig5_practical_data.parquet"


def plot_fig7(base_name="fig7"):
    """Plot two subplots: classical and practical unavailable energy along optimal practical curve."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file {DATA_FILE} not found. Please run newfig5_practical.py first to generate the data."
        )

    # Load data from parquet
    df = pd.read_parquet(DATA_FILE)

    # Extract optimal line data (along practical optimum)
    opt_mask = df["d_opt"].notna()
    d_opt = df[opt_mask]["d_opt"].values
    z_practical_opt = df[opt_mask]["dqom_practical_opt"].values
    z_classical_opt = df[opt_mask]["dqom_classical_opt"].values

    if len(d_opt) == 0:
        raise ValueError("No optimal line data found in parquet file.")

    # Sort by d_opt for proper plotting
    sort_idx = np.argsort(d_opt)
    d_opt = d_opt[sort_idx]
    z_practical_opt = z_practical_opt[sort_idx]
    z_classical_opt = z_classical_opt[sort_idx]

    # Create figure with two subplots
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18 / 2.54, 7 / 2.54))

    # Left subplot: Classical unavailable energy along practical optimum
    ax1.plot(d_opt, z_classical_opt, "k-", linewidth=1.5, label="classical (along practical opt)")
    ax1.set_xlabel(r"$d / d_{\mathrm{ref}}$")
    ax1.set_ylabel(r"$\Delta Q_0 / Q_{\mathrm{max}}$")
    ax1.set_title(
        r"HEx $\Delta Q_0 / Q_{\mathrm{max}}$ vs $d / d_{\mathrm{ref}}$"
        + "\n"
        + r"(along optimal practical $A_o/A_{o,\mathrm{ref}}$)"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # Right subplot: Practical unavailable energy along practical optimum
    ax2.plot(d_opt, z_practical_opt, "k-", linewidth=1.5, label="practical (optimal)")
    ax2.set_xlabel(r"$d / d_{\mathrm{ref}}$")
    ax2.set_ylabel(r"$\Delta Q_0^M / Q_{\mathrm{max}}$")
    ax2.set_title(
        r"HEx $\Delta Q_0^M / Q_{\mathrm{max}}$ vs $d / d_{\mathrm{ref}}$"
        + "\n"
        + r"(optimal $A_o/A_{o,\mathrm{ref}}$)"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    plt.tight_layout(pad=0.5)

    for ext in ["svg", "tiff", "png"]:
        path = os.path.join(save_dir, f"{base_name}.{ext}")
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_fig7()
