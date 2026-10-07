"""
Script to examine all outputs from r_s (rate_hex_simple) for debugging high NTU behavior.

This script uses the same parameters as Q_Pdot_sliders.py but displays all outputs
from r_s in a comprehensive format, with special attention to high NTU values.
"""

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.general_counterflow import rate_hex_simple

# ============================================================================
# PARAMETERS (default values from Q_Pdot_sliders.py with Aq sweep)
# ============================================================================

case = "B"  # "Brewer"

# Fluid models
FLUID_COLD = PerfectGasFluid.from_name("para_h2")
FLUID_HOT = PerfectGasFluid(M=27.5, S=150.0, T_ref=350.0, mu_ref=1.12e-5, gamma=1.37, Pr=0.74, cp=1170.0)

# Default values from Q_Pdot_sliders.py (Brewer case)
T_OVER_DHC = 0.06
SIGMA_R = 11.0
SIGMA_W = 1.14
D_H_C = 4.7e-2  # m
LS_OVER_DH = 60.0
AQ_BASELINE = 16.0
A_FR_BASELINE = 1.0
M_DOT_HOT = 19.07  # kg/s
M_DOT_COLD = 0.166  # kg/s
TH_IN = 778  # K
PH_IN = 0.388e5  # Pa
TC_IN = 264  # K
PC_IN = 16.8e5  # Pa
DP_MAX = 0.15

# Sweep settings (from Q_Pdot_sliders.py defaults)
PLOT_AQ_SWEEP_NOT_AFR = False  # True means Aq sweep
PLOT_PDOT_NOT_DP = False  # False means dp (not Pdot/Q_max)

# ============================================================================
# SETUP
# ============================================================================

# Create fluid inputs
F_IN = FluidInputs(
    hot=FLUID_HOT,
    cold=FLUID_COLD,
    m_dot_hot=M_DOT_HOT,
    m_dot_cold=M_DOT_COLD,
    Th_in=TH_IN,
    Ph_in=PH_IN,
    Tc_in=TC_IN,
    Pc_in=PC_IN,
)

# Generate x values for sweep
if PLOT_AQ_SWEEP_NOT_AFR:
    if A_FR_BASELINE < 1:
        x = np.geomspace(0.5, 500 / A_FR_BASELINE, 200)
    else:
        x = np.geomspace(0.5, 500 * A_FR_BASELINE, 200)
    Aq = x
    A_fr = A_FR_BASELINE
    sweep_type = "Aq"
else:
    x = np.geomspace(A_FR_BASELINE, 0.001, 100)
    Aq = AQ_BASELINE
    A_fr = x
    sweep_type = "A_fr"

# ============================================================================
# CALCULATE RESULTS
# ============================================================================

print("=" * 80)
print("Calculating r_s outputs...")
print("=" * 80)
print(f"Sweep type: {sweep_type}")
print(f"Number of points: {len(x)}")
print()

# Calculate results
r_s = rate_hex_simple(
    A_fr=A_fr,
    A_q=Aq,
    f_in=F_IN,
    d_h_c=D_H_C,
    sigma_r=SIGMA_R,
    sigma_w=SIGMA_W,
    t_over_dhc=T_OVER_DHC,
    ls_over_dh=LS_OVER_DH,
)

# Extract all outputs and ensure they're numpy arrays
# First, determine the result length from ntu (which should always be an array)
ntu = np.atleast_1d(r_s["ntu"])
result_length = len(ntu)

# Extract all outputs and broadcast scalars to match result_length
# If an output is a scalar (same for all points), broadcast it to an array
eps = np.atleast_1d(r_s["eps"])
dp_hot = np.atleast_1d(r_s["dp_hot"])
dp_cold = np.atleast_1d(r_s["dp_cold"])
t_hot_out = np.atleast_1d(r_s["t_hot_out"])
t_cold_out = np.atleast_1d(r_s["t_cold_out"])
re_hot = np.atleast_1d(r_s["re_hot"])
re_cold = np.atleast_1d(r_s["re_cold"])
g2_hot = np.atleast_1d(r_s["g2_hot"])
g2_cold = np.atleast_1d(r_s["g2_cold"])
Aq_over_Ao_c = np.atleast_1d(r_s["Aq_over_Ao_c"])
Aq_over_Ao_h = np.atleast_1d(r_s["Aq_over_Ao_h"])
dW_pot_Ex_norm = np.atleast_1d(r_s["dW_pot_Ex_norm"])
dW_pot_Eu_norm = np.atleast_1d(r_s["dW_pot_Eu_norm"])
R_cold_over_R_tot = np.atleast_1d(r_s["R_cold_over_R_tot"])

# Broadcast any scalar outputs to match result_length
# (If they're scalars, it means they're the same for all points)
if len(eps) == 1 and result_length > 1:
    eps = np.full(result_length, eps[0])
if len(dp_hot) == 1 and result_length > 1:
    dp_hot = np.full(result_length, dp_hot[0])
if len(dp_cold) == 1 and result_length > 1:
    dp_cold = np.full(result_length, dp_cold[0])
if len(t_hot_out) == 1 and result_length > 1:
    t_hot_out = np.full(result_length, t_hot_out[0])
if len(t_cold_out) == 1 and result_length > 1:
    t_cold_out = np.full(result_length, t_cold_out[0])
if len(re_hot) == 1 and result_length > 1:
    re_hot = np.full(result_length, re_hot[0])
if len(re_cold) == 1 and result_length > 1:
    re_cold = np.full(result_length, re_cold[0])
if len(g2_hot) == 1 and result_length > 1:
    g2_hot = np.full(result_length, g2_hot[0])
if len(g2_cold) == 1 and result_length > 1:
    g2_cold = np.full(result_length, g2_cold[0])
if len(Aq_over_Ao_c) == 1 and result_length > 1:
    Aq_over_Ao_c = np.full(result_length, Aq_over_Ao_c[0])
if len(Aq_over_Ao_h) == 1 and result_length > 1:
    Aq_over_Ao_h = np.full(result_length, Aq_over_Ao_h[0])
if len(dW_pot_Ex_norm) == 1 and result_length > 1:
    dW_pot_Ex_norm = np.full(result_length, dW_pot_Ex_norm[0])
if len(dW_pot_Eu_norm) == 1 and result_length > 1:
    dW_pot_Eu_norm = np.full(result_length, dW_pot_Eu_norm[0])
if len(R_cold_over_R_tot) == 1 and result_length > 1:
    R_cold_over_R_tot = np.full(result_length, R_cold_over_R_tot[0])

# Ensure A_fr and Aq are arrays with the same length as the results
# If one is a scalar, broadcast it to match the length
A_fr = np.atleast_1d(A_fr)
Aq = np.atleast_1d(Aq)

# Broadcast scalars to match result length
if len(A_fr) == 1 and result_length > 1:
    A_fr = np.full(result_length, A_fr[0])
if len(Aq) == 1 and result_length > 1:
    Aq = np.full(result_length, Aq[0])

# Find validity mask
over_limit = (dp_hot >= DP_MAX) | (dp_cold >= DP_MAX) | (dp_hot < 0)
if np.any(over_limit):
    first_exceed = np.argmax(over_limit)
    validity_mask = np.zeros_like(dp_hot, dtype=bool)
    validity_mask[:first_exceed] = True
else:
    validity_mask = np.ones_like(dp_hot, dtype=bool)

# ============================================================================
# CREATE COMPREHENSIVE DATA TABLE
# ============================================================================

print("Creating comprehensive output table...")
print()

# Create list of dictionaries with all outputs for easy access
data = []
for i in range(len(ntu)):
    data.append(
        {
            "index": i,
            "A_fr": A_fr[i],
            "Aq": Aq[i],
            "NTU": ntu[i],
            "eps": eps[i],
            "dp_hot": dp_hot[i],
            "dp_cold": dp_cold[i],
            "t_hot_out": t_hot_out[i],
            "t_cold_out": t_cold_out[i],
            "re_hot": re_hot[i],
            "re_cold": re_cold[i],
            "g2_hot": g2_hot[i],
            "g2_cold": g2_cold[i],
            "Aq_over_Ao_c": Aq_over_Ao_c[i],
            "Aq_over_Ao_h": Aq_over_Ao_h[i],
            "dW_pot_Ex_norm": dW_pot_Ex_norm[i],
            "dW_pot_Eu_norm": dW_pot_Eu_norm[i],
            "R_cold_over_R_tot": R_cold_over_R_tot[i],
            "valid": validity_mask[i],
        }
    )


# Helper function to convert to arrays for statistics
def get_array(key):
    return np.array([d[key] for d in data])


# ============================================================================
# DISPLAY SUMMARY STATISTICS
# ============================================================================

print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()
print(f"Total points: {len(data)}")
print(f"Valid points: {validity_mask.sum()}")
print(f"Invalid points: {(~validity_mask).sum()}")
print()

print("NTU range:")
print(f"  Min: {ntu.min():.4f}")
print(f"  Max: {ntu.max():.4f}")
if validity_mask.any():
    print(f"  Max (valid): {ntu[validity_mask].max():.4f}")
else:
    print("  Max (valid): N/A")
print()

print("High NTU region (top 10% of valid NTU values):")
if validity_mask.any():
    ntu_valid = ntu[validity_mask]
    high_ntu_threshold = np.percentile(ntu_valid, 90)
    high_ntu_mask = validity_mask & (ntu >= high_ntu_threshold)
    print(f"  Threshold: {high_ntu_threshold:.4f}")
    print(f"  Number of points: {high_ntu_mask.sum()}")
    print()

    print("  Statistics for high NTU region:")
    high_ntu_data = [d for i, d in enumerate(data) if high_ntu_mask[i]]
    for col in ["NTU", "eps", "dp_hot", "dp_cold", "re_hot", "re_cold", "g2_hot", "g2_cold"]:
        arr = get_array(col)[high_ntu_mask]
        print(f"    {col:20s}: min={arr.min():12.6e}, max={arr.max():12.6e}, mean={arr.mean():12.6e}")
else:
    print("  No valid points found!")
print()

# ============================================================================
# DISPLAY DETAILED TABLE FOR HIGH NTU VALUES
# ============================================================================

print("=" * 80)
print("DETAILED OUTPUTS - ALL VALUES")
print("=" * 80)
print()

# Display table using tabulate
headers = [
    "idx",
    "A_fr",
    "Aq",
    "NTU",
    "eps",
    "dp_hot",
    "dp_cold",
    "t_hot_out",
    "t_cold_out",
    "re_hot",
    "re_cold",
    "g2_hot",
    "g2_cold",
    "Aq/Ao_c",
    "Aq/Ao_h",
    "dW_Ex",
    "dW_Eu",
    "R_c/R_t",
    "valid",
]
table_data = []
for d in data:
    table_data.append(
        [
            d["index"],
            f"{d['A_fr']:.6e}",
            f"{d['Aq']:.6e}",
            f"{d['NTU']:.6f}",
            f"{d['eps']:.6f}",
            f"{d['dp_hot']:.6e}",
            f"{d['dp_cold']:.6e}",
            f"{d['t_hot_out']:.2f}",
            f"{d['t_cold_out']:.2f}",
            f"{d['re_hot']:.2e}",
            f"{d['re_cold']:.2e}",
            f"{d['g2_hot']:.6e}",
            f"{d['g2_cold']:.6e}",
            f"{d['Aq_over_Ao_c']:.6f}",
            f"{d['Aq_over_Ao_h']:.6f}",
            f"{d['dW_pot_Ex_norm']:.6e}",
            f"{d['dW_pot_Eu_norm']:.6e}",
            f"{d['R_cold_over_R_tot']:.6f}",
            d["valid"],
        ]
    )

print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".6e"))
print()

# ============================================================================
# FOCUS ON HIGH NTU REGION
# ============================================================================

if validity_mask.any():
    print("=" * 80)
    print("FOCUS: HIGH NTU REGION (where weird things happen)")
    print("=" * 80)
    print()

    # Sort by NTU (descending) and show top 20 valid points
    valid_data = [(i, d) for i, d in enumerate(data) if validity_mask[i]]
    valid_data_sorted = sorted(valid_data, key=lambda x: x[1]["NTU"], reverse=True)[:20]

    print("Top 20 points by NTU (highest first):")
    print()
    high_ntu_table = []
    for _i, d in valid_data_sorted:
        high_ntu_table.append(
            [
                d["index"],
                f"{d['NTU']:.6f}",
                f"{d['eps']:.6f}",
                f"{d['dp_hot']:.6e}",
                f"{d['dp_cold']:.6e}",
                f"{d['re_hot']:.2e}",
                f"{d['re_cold']:.2e}",
                f"{d['g2_hot']:.6e}",
                f"{d['g2_cold']:.6e}",
                f"{d['Aq_over_Ao_c']:.6f}",
                f"{d['Aq_over_Ao_h']:.6f}",
                f"{d['t_hot_out']:.2f}",
                f"{d['t_cold_out']:.2f}",
            ]
        )

    high_ntu_headers = [
        "idx",
        "NTU",
        "eps",
        "dp_hot",
        "dp_cold",
        "re_hot",
        "re_cold",
        "g2_hot",
        "g2_cold",
        "Aq/Ao_c",
        "Aq/Ao_h",
        "T_hot_out",
        "T_cold_out",
    ]
    print(tabulate(high_ntu_table, headers=high_ntu_headers, tablefmt="grid"))
    print()

    # Check for anomalies
    print("Checking for anomalies in high NTU region:")
    print()

    # Check for NaN or Inf values
    for col in ["NTU", "eps", "dp_hot", "dp_cold", "re_hot", "re_cold", "g2_hot", "g2_cold"]:
        arr = get_array(col)[validity_mask]
        if np.any(np.isnan(arr)):
            print(f"  WARNING: NaN values found in {col}")
        if np.any(np.isinf(arr)):
            print(f"  WARNING: Inf values found in {col}")

    # Check for rapid changes
    print("  Checking for rapid changes (derivatives):")
    valid_indices = np.where(validity_mask)[0]
    for col in ["eps", "dp_hot", "dp_cold", "re_hot", "re_cold"]:
        arr = get_array(col)[validity_mask]
        ntu_arr = ntu[validity_mask]
        # Sort by NTU for derivative calculation
        sort_idx = np.argsort(ntu_arr)
        arr_sorted = arr[sort_idx]
        ntu_sorted = ntu_arr[sort_idx]
        diff = np.diff(arr_sorted) / (np.diff(ntu_sorted) + 1e-10)
        if len(diff) > 0:
            max_diff = np.abs(diff).max()
            print(f"    {col:20s}: max |d/dNTU| = {max_diff:.6e}")

    print()

# ============================================================================
# SAVE TO CSV
# ============================================================================

output_file = "r_s_outputs.csv"
print(f"Saving full results to {output_file}...")

# Write CSV manually
with open(output_file, "w") as f:
    # Write header
    f.write(",".join(headers) + "\n")
    # Write data
    for d in data:
        f.write(
            ",".join(
                [
                    str(d["index"]),
                    str(d["A_fr"]),
                    str(d["Aq"]),
                    str(d["NTU"]),
                    str(d["eps"]),
                    str(d["dp_hot"]),
                    str(d["dp_cold"]),
                    str(d["t_hot_out"]),
                    str(d["t_cold_out"]),
                    str(d["re_hot"]),
                    str(d["re_cold"]),
                    str(d["g2_hot"]),
                    str(d["g2_cold"]),
                    str(d["Aq_over_Ao_c"]),
                    str(d["Aq_over_Ao_h"]),
                    str(d["dW_pot_Ex_norm"]),
                    str(d["dW_pot_Eu_norm"]),
                    str(d["R_cold_over_R_tot"]),
                    str(d["valid"]),
                ]
            )
            + "\n"
        )

print("Saved!")
print()

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================

print("Creating visualization plots...")
print()

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle("r_s Outputs vs NTU", fontsize=16)

# Plot 1: eps vs NTU
ax = axes[0, 0]
ax.plot(ntu[validity_mask], eps[validity_mask], "b-", label="eps")
ax.set_xlabel("NTU")
ax.set_ylabel("eps")
ax.set_title("Effectiveness")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: dp vs NTU
ax = axes[0, 1]
ax.plot(ntu[validity_mask], dp_hot[validity_mask], "r--", label="dp_hot")
ax.plot(ntu[validity_mask], dp_cold[validity_mask], "g--", label="dp_cold")
ax.axhline(y=DP_MAX, color="k", linestyle=":", label=f"dp_max={DP_MAX}")
ax.set_xlabel("NTU")
ax.set_ylabel("dp")
ax.set_title("Pressure Drop")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Reynolds numbers vs NTU
ax = axes[1, 0]
ax.plot(ntu[validity_mask], re_hot[validity_mask], "r-", label="Re_hot")
ax.plot(ntu[validity_mask], re_cold[validity_mask], "g-", label="Re_cold")
ax.set_xlabel("NTU")
ax.set_ylabel("Re")
ax.set_title("Reynolds Numbers")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: g² vs NTU
ax = axes[1, 1]
ax.plot(ntu[validity_mask], g2_hot[validity_mask], "r-", label="g²_hot")
ax.plot(ntu[validity_mask], g2_cold[validity_mask], "g-", label="g²_cold")
ax.set_xlabel("NTU")
ax.set_ylabel("g²")
ax.set_title("Dimensionless Mass Flux Squared")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 5: Aq_over_Ao vs NTU
ax = axes[2, 0]
ax.plot(ntu[validity_mask], Aq_over_Ao_h[validity_mask], "r-", label="Aq/Ao_hot")
ax.plot(ntu[validity_mask], Aq_over_Ao_c[validity_mask], "g-", label="Aq/Ao_cold")
ax.set_xlabel("NTU")
ax.set_ylabel("Aq/Ao")
ax.set_title("Heat Transfer Area / Free Flow Area")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 6: Temperature outputs vs NTU
ax = axes[2, 1]
ax.plot(ntu[validity_mask], t_hot_out[validity_mask], "r-", label="T_hot_out")
ax.plot(ntu[validity_mask], t_cold_out[validity_mask], "g-", label="T_cold_out")
ax.set_xlabel("NTU")
ax.set_ylabel("Temperature (K)")
ax.set_title("Outlet Temperatures")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 7: Work potentials vs NTU
ax = axes[3, 0]
ax.plot(ntu[validity_mask], dW_pot_Ex_norm[validity_mask], "b-", label="dW_pot_Ex_norm")
ax.plot(ntu[validity_mask], dW_pot_Eu_norm[validity_mask], "m-", label="dW_pot_Eu_norm")
ax.set_xlabel("NTU")
ax.set_ylabel("Normalized Work Potential")
ax.set_title("Work Potentials")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 8: R_cold_over_R_tot vs NTU
ax = axes[3, 1]
ax.plot(ntu[validity_mask], R_cold_over_R_tot[validity_mask], "b-", label="R_cold/R_tot")
ax.set_xlabel("NTU")
ax.set_ylabel("R_cold / R_tot")
ax.set_title("Cold Side Resistance Ratio")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plot_file = "r_s_outputs_plots.png"
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
print(f"Saved plots to {plot_file}")
print()

# ============================================================================
# FOCUSED HIGH NTU PLOTS
# ============================================================================

if validity_mask.any():
    print("Creating focused plots for high NTU region...")
    print()

    ntu_valid = ntu[validity_mask]
    high_ntu_threshold = np.percentile(ntu_valid, 90)
    high_ntu_mask = validity_mask & (ntu >= high_ntu_threshold)

    if high_ntu_mask.sum() > 0:
        fig2, axes2 = plt.subplots(3, 2, figsize=(14, 10))
        fig2.suptitle(f"High NTU Region (NTU >= {high_ntu_threshold:.4f})", fontsize=16)

        # Plot 1: eps vs NTU (high NTU)
        ax = axes2[0, 0]
        ax.plot(ntu[high_ntu_mask], eps[high_ntu_mask], "b-o", markersize=4, label="eps")
        ax.set_xlabel("NTU")
        ax.set_ylabel("eps")
        ax.set_title("Effectiveness (High NTU)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: dp vs NTU (high NTU)
        ax = axes2[0, 1]
        ax.plot(ntu[high_ntu_mask], dp_hot[high_ntu_mask], "r--o", markersize=4, label="dp_hot")
        ax.plot(ntu[high_ntu_mask], dp_cold[high_ntu_mask], "g--o", markersize=4, label="dp_cold")
        ax.axhline(y=DP_MAX, color="k", linestyle=":", label=f"dp_max={DP_MAX}")
        ax.set_xlabel("NTU")
        ax.set_ylabel("dp")
        ax.set_title("Pressure Drop (High NTU)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Reynolds numbers vs NTU (high NTU)
        ax = axes2[1, 0]
        ax.plot(ntu[high_ntu_mask], re_hot[high_ntu_mask], "r-o", markersize=4, label="Re_hot")
        ax.plot(ntu[high_ntu_mask], re_cold[high_ntu_mask], "g-o", markersize=4, label="Re_cold")
        ax.set_xlabel("NTU")
        ax.set_ylabel("Re")
        ax.set_title("Reynolds Numbers (High NTU)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 4: g² vs NTU (high NTU)
        ax = axes2[1, 1]
        ax.plot(ntu[high_ntu_mask], g2_hot[high_ntu_mask], "r-o", markersize=4, label="g²_hot")
        ax.plot(ntu[high_ntu_mask], g2_cold[high_ntu_mask], "g-o", markersize=4, label="g²_cold")
        ax.set_xlabel("NTU")
        ax.set_ylabel("g²")
        ax.set_title("Dimensionless Mass Flux Squared (High NTU)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 5: Aq_over_Ao vs NTU (high NTU)
        ax = axes2[2, 0]
        ax.plot(ntu[high_ntu_mask], Aq_over_Ao_h[high_ntu_mask], "r-o", markersize=4, label="Aq/Ao_hot")
        ax.plot(ntu[high_ntu_mask], Aq_over_Ao_c[high_ntu_mask], "g-o", markersize=4, label="Aq/Ao_cold")
        ax.set_xlabel("NTU")
        ax.set_ylabel("Aq/Ao")
        ax.set_title("Heat Transfer Area / Free Flow Area (High NTU)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 6: All outputs vs NTU (high NTU) - normalized
        ax = axes2[2, 1]
        # Normalize each quantity to [0, 1] for comparison
        ntu_norm = (ntu[high_ntu_mask] - ntu[high_ntu_mask].min()) / (
            ntu[high_ntu_mask].max() - ntu[high_ntu_mask].min() + 1e-10
        )
        eps_norm = (eps[high_ntu_mask] - eps[high_ntu_mask].min()) / (
            eps[high_ntu_mask].max() - eps[high_ntu_mask].min() + 1e-10
        )
        dp_hot_norm = (dp_hot[high_ntu_mask] - dp_hot[high_ntu_mask].min()) / (
            dp_hot[high_ntu_mask].max() - dp_hot[high_ntu_mask].min() + 1e-10
        )
        dp_cold_norm = (dp_cold[high_ntu_mask] - dp_cold[high_ntu_mask].min()) / (
            dp_cold[high_ntu_mask].max() - dp_cold[high_ntu_mask].min() + 1e-10
        )

        ax.plot(ntu[high_ntu_mask], eps_norm, "b-o", markersize=4, label="eps (norm)")
        ax.plot(ntu[high_ntu_mask], dp_hot_norm, "r--o", markersize=4, label="dp_hot (norm)")
        ax.plot(ntu[high_ntu_mask], dp_cold_norm, "g--o", markersize=4, label="dp_cold (norm)")
        ax.set_xlabel("NTU")
        ax.set_ylabel("Normalized Value")
        ax.set_title("Normalized Comparison (High NTU)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plot_file2 = "r_s_outputs_high_ntu.png"
        plt.savefig(plot_file2, dpi=150, bbox_inches="tight")
        print(f"Saved high NTU plots to {plot_file2}")
        print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Output files:")
print(f"  - {output_file} (CSV with all data)")
print(f"  - {plot_file} (Plots of all outputs)")
if validity_mask.any():
    print(f"  - {plot_file2} (Focused plots for high NTU region)")
print()
