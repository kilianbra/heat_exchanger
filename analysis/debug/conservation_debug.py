"""
Debug script to reproduce the conservation debug message:
DEBUG:heat_exchanger.conservation:Fluid step is not within desired tolerances:
Individual residuals: |dh_t|=1.00e+12 (want < 1.46e+02), |d(p+G²/ρ)|=1.91e+12 (want < 2.79e+02) |
State: (T_a=275.0 K, p_b=2.79e+06 Pa) | Inputs: (G=664.2 kg/m²s, dh0=3.08e+06 J/kg, tau_dA_over_A_c=6.98e+06)
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from heat_exchanger.conservation import update_s_prop, update_static_properties
from heat_exchanger.fluids.compressible_flow_friction_heat import (
    find_ksi_lim_adaptive as max_friction,
)
from heat_exchanger.fluids.compressible_flow_friction_heat import (
    p_static_over_p_static_in,
    solve_M_from_ksi,
)
from heat_exchanger.fluids.protocols import PerfectGasFluid
from heat_exchanger.logging_utils import configure_logging

# Configure logging to DEBUG to see the conservation debug message
configure_logging(logging.INFO)

# Set the conservation logger to DEBUG level
logging.getLogger("heat_exchanger.conservation").setLevel(logging.INFO)

print("=" * 70)
print("Testing update_static_properties with parameters that cause")
print("non-convergence to reproduce the debug message")
print("=" * 70)
print()

# Parameters from the debug message (total values)
T_in = 275.0  # K
p_in = 2.79e6  # Pa (2.79 MPa)
G = 664.2  # kg/m²s
dh0_total = 3.08e6  # J/kg (total)
tau_dA_over_A_c_total = 6.98e6 / 7  # Pa (total)
# Create fluid model (likely Para_Hydrogen based on full_flow.py)
fluid = PerfectGasFluid.from_name("Para_Hydrogen")
# fluid = CoolPropFluid("ParaHydrogen")
fluid_in = fluid.state(T_in, p_in)

ksi = tau_dA_over_A_c_total / (0.5 * G**2 / fluid_in.rho)


expected_dT = dh0_total / fluid_in.cp
expected_T_final = T_in + expected_dT
f_init = fluid.state(T_in, p_in)
V_init = G / f_init.rho
Mach_init = V_init / f_init.a

h_stag_in = fluid_in.h + 0.5 * G**2 / fluid_in.rho
T_stag_in = T_in + 0.5 * G**2 / (fluid_in.rho * fluid_in.cp)
g = fluid_in.gamma
gm1og = (g - 1) / g
p_stag_in = p_in * (1 + (g - 1) / 2 * Mach_init**2) ** (1 / gm1og)
k = dh0_total / (h_stag_in) / ksi
ksi_lim, _ = max_friction(Mach_init, k, gamma=fluid_in.gamma)


print(
    f"For k={k:.2e}, M_in={Mach_init:.2f}, and gamma={fluid_in.gamma:.2f}, ksi_lim = {ksi_lim:.2e} vs. ksi = {ksi:.2e}"
)


def single_stepping(n_steps=1):
    # Divide by 100 for sequential steps

    dh0_step = dh0_total / n_steps  # J/kg per step
    tau_dA_over_A_c_step = tau_dA_over_A_c_total / n_steps  # Pa per step

    M_lim = 0.95

    print("Fluid: Para-Hydrogen")
    print("Input parameters:")
    print(f" Inlet state: T={T_in:.2f} K, p={p_in:.2e} Pa, M={Mach_init:.2f}")
    print(f" Stagnation state: T={T_stag_in:.2f} K, p={p_stag_in:.2e} Pa")
    print(f"  G = {G:.1f} kg/m²s (Mach = {Mach_init:.2f})")
    print(f"  dh0_total = {dh0_total:.2e} J/kg (i.e. dT0_total = {dh0_total / fluid_in.cp:.2f} K)")
    print(f"  tau_dA_over_A_c_total = {tau_dA_over_A_c_total:.2e} Pa ")
    print()
    print(f"Applying in {n_steps} sequential steps:")
    print(f"  dh0_step = {dh0_step:.2e} J/kg per step (i.e. dT0_step = {dh0_step / fluid_in.cp:.2f} K)")
    print(f"  tau_dA_over_A_c_step = {tau_dA_over_A_c_step:.2e} Pa per step")
    print()

    # Initialize state
    T_current = T_in
    p_current = p_in

    # Track convergence issues
    convergence_failures = 0
    n_final = None  # Will be set if loop breaks early

    try:
        for step in range(0, n_steps):
            # T_prev = T_current
            # p_prev = p_current

            T_current, p_current = update_static_properties(
                fluid,
                G,
                dh0_step,
                tau_dA_over_A_c_step,
                T_current,
                p_current,
                a_is_in=True,
                b_is_in=True,
                max_iter=2000,
                tol_T=1e-8,
                rel_tol_p=1e-3,
            )

            # Check if solution is reasonable (not NaN or negative)
            if not (T_current > 0 and p_current > 0):
                convergence_failures += 1
                print(f"Step {step:3d}: WARNING - Non-physical solution: T={T_current:.2f} K, p={p_current:.2e} Pa")

            # Print progress every 10 steps
            if step % 1 == 0 or step == n_steps:
                dT = T_current - T_in
                dp = p_current - p_in
                f_current = fluid.state(T_current, p_current)
                V_current = G / f_current.rho
                Mach_current = V_current / f_current.a
                print(
                    f"Step {step:3d}: M={Mach_current:.2f}, T = {T_current:.2f} K (ΔT = {dT:+7.2f} K), p = {p_current:.2e} Pa (Δp = {dp / p_in * 100:6.2f} % inlet)"
                )
                if T_current > expected_T_final:
                    print("Aborting due to T_current > expected_T_final")
                    n_final = step
                    break
                elif Mach_current > M_lim:
                    print(f"Aborting due to Mach_current > {M_lim:.1f}")
                    n_final = step
                    break
        # Make sure n_taken is always defined after the loop
        n_taken = n_final if n_final is not None else n_steps
        f_current = fluid.state(T_current, p_current)
        V_current = G / f_current.rho
        Mach_current = V_current / f_current.a
        print()
        print(f"Final state after {n_taken} of {n_steps} steps:")
        print(f"  T_final = {T_current:.2f} K (ΔT_total = {T_current - T_in:+.2f} K)")
        print(f"  p_final = {p_current:.2e} Pa (Δp_total/p_in = {100 * (p_current - p_in) / p_in:.2f} %)")
        print(f"  M_final = {Mach_current:.2f} ")
        T_stag_final = T_current + 0.5 * G**2 / (f_current.rho * f_current.cp)
        p_stag_final = p_current * (1 + (g - 1) / 2 * Mach_current**2) ** (1 / gm1og)
        print(f" Stagnation state: T={T_stag_final:.2f} K, p={p_stag_final:.2e} Pa")
        print(
            f" p_static_out/p_static_in = {p_current / p_in:.6f}, p_stag_out/p_stag_in = {p_stag_final / p_stag_in:.6f}"
        )

        # Calculate p_static/p_stag_in using equation 18 from Sturas 1971
        try:
            p_static_over_p_static_in_theory = p_static_over_p_static_in(
                Mach_init, Mach_current, k, ksi, gamma=fluid_in.gamma
            )
            print(f" p_static_out/p_stag_in (theory, eq. 18) = {p_static_over_p_static_in_theory:.6f}")
            print(f" Difference (actual - theory) = {(p_current / p_in) - p_static_over_p_static_in_theory:.6e}")
        except Exception as e:
            print(f" Could not calculate p_static/p_stag_in from theory: {e}")

        tau_out = T_stag_final / T_stag_in
        print(f" tau_out = {tau_out:.2f} vs k * ksi + 1 = {k * ksi + 1:.2f}")

        # Calculate expected final state (approximate, assuming constant cp)
        if n_final is None:
            print()
            print(f"Expected final T (assuming constant cp): {expected_T_final:.2f} K")
            print(f"Difference from 0D step: {T_current - expected_T_final:.2f} K")
        else:
            print(f"Transfered {n_taken / n_steps * 100:.2f}% of the total heat and pressure drop")

        if convergence_failures > 0:
            print(f"\nWARNING: {convergence_failures} steps had convergence issues")
        else:
            print("\nAll steps converged successfully!")

    except Exception as e:
        print(f"Error at step {step}: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("Test complete")
    print(f"{'=' * 70}")


# ============================================================================
# NEW SECTION: Compare compressible flow theory vs update_static_properties
# Plot Mach_out vs ksi for both models
# ============================================================================

print("\n" + "=" * 70)
print("Comparing compressible flow theory vs update_static_properties")
print("Plotting Mach_out vs ksi for constant heat flux per unit length")
print("=" * 70)
print()

Mach_in = 0.15
T_in = 275.0  # K
p_in = 2.79e6  # Pa (2.79 MPa)
G = 664.2  # kg/m²s


def calculate_mach_theory(ksi_val, M_0, k, gamma):
    """Calculate Mach number using compressible flow theory."""
    try:
        _, M = solve_M_from_ksi(ksi_val, M_0, k, gamma=gamma)
        return M, True
    except Exception:
        return np.nan, False


def calculate_mach_solver(ksi_val, fluid, T_in, p_in, G, dh0_total, ksi_ref, tau_dA_over_A_c_total, n_steps=1):
    """
    Calculate Mach number using update_static_properties.

    For a given ksi_val, scales dh0_total and tau_dA_over_A_c_total by ksi_val/ksi_ref.
    """
    # Scale heat addition and friction by ksi_val/ksi_ref
    dh0_scaled = dh0_total * ksi_val / ksi_ref
    tau_dA_over_A_c_scaled = tau_dA_over_A_c_total * ksi_val / ksi_ref

    # Divide into steps
    dh0_step = dh0_scaled / n_steps
    tau_dA_over_A_c_step = tau_dA_over_A_c_scaled / n_steps

    # Initialize state
    T_current = T_in
    p_current = p_in

    try:
        for _ in range(n_steps):
            T_current, p_current = update_static_properties(
                fluid,
                G,
                dh0_step,
                tau_dA_over_A_c_step,
                T_current,
                p_current,
                a_is_in=True,
                b_is_in=True,
                max_iter=2000,
                tol_T=1e-8,
                rel_tol_p=1e-3,
            )

            # Check if solution is reasonable
            if not (T_current > 0 and p_current > 0):
                return np.nan, False

        # Calculate final Mach number
        f_out = fluid.state(T_current, p_current)
        V_out = G / f_out.rho
        Mach_out = V_out / f_out.a

        return Mach_out, True

    except Exception:
        return np.nan, False


def calculate_mach_new_solver(ksi_val, fluid, T_in, p_in, G, dh0_total, ksi_ref, tau_dA_over_A_c_total, n_steps=1):
    """
    Calculate Mach number using update_static_properties.

    For a given ksi_val, scales dh0_total and tau_dA_over_A_c_total by ksi_val/ksi_ref.
    """
    # Scale heat addition and friction by ksi_val/ksi_ref
    dh0_scaled = dh0_total * ksi_val / ksi_ref

    # Divide into steps
    dh0_step = dh0_scaled / n_steps
    ksi_step = ksi_val / n_steps

    # Initialize state
    T_current = T_in
    p_current = p_in

    try:
        for _ in range(n_steps):
            T_current, p_current = update_s_prop(
                fluid,
                G,
                dh0_step,
                ksi_step,
                T_current,
                p_current,
                a_is_in=True,
                b_is_in=True,
                max_iter=2000,
                tol_T=1e-8,
                rel_tol_p=1e-3,
            )

            # Check if solution is reasonable
            if not (T_current > 0 and p_current > 0):
                return np.nan, False

        # Calculate final Mach number
        f_out = fluid.state(T_current, p_current)
        V_out = G / f_out.rho
        Mach_out = V_out / f_out.a

        return Mach_out, True

    except Exception:
        return np.nan, False


# Setup for plotting
ksi_start = 0.0
ksi_max = max(ksi_lim * 2.0, ksi * 2.0, 20.0) if not np.isnan(ksi_lim) else max(ksi * 2.0, 20.0)
n_points = 200
ksi_array = np.linspace(ksi_start, ksi_max, n_points)

# Arrays to store results
M_theory_array = []
M_solver_array = []
M_new_solver_array = []
ksi_theory_valid = []
ksi_solver_valid = []
ksi_new_solver_valid = []

# Track choking conditions
theory_choked = False
solver_choked = False
new_solver_choked = False
M_theory_prev = Mach_init
M_solver_prev = Mach_init
M_new_solver_prev = Mach_init

print(f"Calculating Mach_out for ksi from {ksi_start:.4e} to {ksi_max:.4e}")
print(f"Reference ksi = {ksi:.6e}, k = {k:.6e}")
print(f"Theoretical ksi_lim = {ksi_lim:.6e}")
print()

for i, ksi_val in enumerate(ksi_array):
    if i % 20 == 0:
        print(f"  Progress: {i}/{n_points} ({100 * i / n_points:.1f}%)")

    # Calculate using theory (if not choked yet)
    if not theory_choked:
        M_th, valid_th = calculate_mach_theory(ksi_val, Mach_init, k, fluid_in.gamma)
        if valid_th and np.isfinite(M_th):
            M_theory_array.append(M_th)
            ksi_theory_valid.append(ksi_val)

            # Check for choking: M >= 1.0 or M decreasing
            if M_th >= 1.0:
                theory_choked = True
                print(f"  Theory choked at ksi = {ksi_val:.6e} (M = {M_th:.6f})")
            elif M_th < M_theory_prev and M_theory_prev > 0.9:
                # M started decreasing (supersonic branch)
                theory_choked = True
                print(f"  Theory choked at ksi = {ksi_val:.6e} (M decreasing from {M_theory_prev:.6f} to {M_th:.6f})")
            M_theory_prev = M_th
        else:
            theory_choked = True
            print(f"  Theory failed at ksi = {ksi_val:.6e}")
    else:
        M_theory_array.append(np.nan)
        ksi_theory_valid.append(ksi_val)

    # Calculate using solver (if not choked yet)
    if not solver_choked:
        M_sol, valid_sol = calculate_mach_solver(
            ksi_val, fluid, T_in, p_in, G, dh0_total, ksi, tau_dA_over_A_c_total, n_steps=1
        )
        if valid_sol and np.isfinite(M_sol):
            M_solver_array.append(M_sol)
            ksi_solver_valid.append(ksi_val)

            # Check for choking: M >= 1.0 or M decreasing
            if M_sol >= 1.0:
                solver_choked = True
                print(f"  Solver choked at ksi = {ksi_val:.6e} (M = {M_sol:.6f})")
            elif M_sol < M_solver_prev and M_solver_prev > 0.9:
                # M started decreasing (supersonic branch)
                solver_choked = True
                print(f"  Solver choked at ksi = {ksi_val:.6e} (M decreasing from {M_solver_prev:.6f} to {M_sol:.6f})")
            M_solver_prev = M_sol
        else:
            solver_choked = True
            print(f"  Solver failed at ksi = {ksi_val:.6e}")
    else:
        M_solver_array.append(np.nan)
        ksi_solver_valid.append(ksi_val)

    if not new_solver_choked:
        M_new_sol, valid_new_sol = calculate_mach_new_solver(
            ksi_val, fluid, T_in, p_in, G, dh0_total, ksi, tau_dA_over_A_c_total, n_steps=1
        )
        if valid_new_sol and np.isfinite(M_new_sol):
            M_new_solver_array.append(M_new_sol)
            ksi_new_solver_valid.append(ksi_val)
        else:
            new_solver_choked = True
            print(f"  New solver failed at ksi = {ksi_val:.6e}")
    else:
        M_new_solver_array.append(np.nan)
        ksi_new_solver_valid.append(ksi_val)

    # Stop if both are choked
    if theory_choked and solver_choked and new_solver_choked:
        print(f"  All models choked, stopping at ksi = {ksi_val:.6e}")
        break

print(
    f"  Completed: Theory choked = {theory_choked}, Solver choked = {solver_choked}, New solver choked = {new_solver_choked}"
)
print()

# Convert to numpy arrays
M_theory_array = np.array(M_theory_array)
M_solver_array = np.array(M_solver_array)
M_new_solver_array = np.array(M_new_solver_array)
ksi_theory_valid = np.array(ksi_theory_valid)
ksi_solver_valid = np.array(ksi_solver_valid)
ksi_new_solver_valid = np.array(ksi_new_solver_valid)

# Create mask for valid points
valid_theory = np.isfinite(M_theory_array)
valid_solver = np.isfinite(M_solver_array)
valid_new_solver = np.isfinite(M_new_solver_array)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot theory curve
if np.any(valid_theory):
    ax.plot(
        ksi_theory_valid[valid_theory],
        M_theory_array[valid_theory],
        "r-",
        linewidth=2,
        label="Compressible flow theory (solve_M_from_ksi)",
    )

# Plot solver curve
if np.any(valid_solver):
    ax.plot(
        ksi_solver_valid[valid_solver],
        M_solver_array[valid_solver],
        "b-",
        linewidth=2,
        label="update_static_properties solver",
    )

# Plot new solver curve
if np.any(valid_new_solver):
    ax.plot(
        ksi_new_solver_valid[valid_new_solver],
        M_new_solver_array[valid_new_solver],
        "g-",
        linewidth=2,
        label="update_s_prop solver",
    )

# Mark reference ksi and ksi_lim
ax.axvline(x=ksi, color="orange", linestyle=":", linewidth=2, alpha=0.7, label=f"Reference ksi = {ksi:.4e}")
if not np.isnan(ksi_lim) and ksi_lim > 0:
    ax.axvline(
        x=ksi_lim, color="g", linestyle=":", linewidth=2, alpha=0.7, label=f"Theoretical ksi_lim = {ksi_lim:.4e}"
    )

# Mark Mach = 1.0
ax.axhline(y=1.0, color="k", linestyle="--", linewidth=1, alpha=0.5, label="Mach = 1.0")

ax.set_xlabel("ξ (ksi)", fontsize=12)
ax.set_ylabel("Mach number at exit", fontsize=12)
ax.set_title(
    f"Mach_out vs ksi comparison (M_in={Mach_init:.3f}, k={k:.4e}, γ={fluid_in.gamma:.3f})\n"
    f"Constant heat flux per unit length: dh0/ksi = {dh0_total / ksi:.2e} J/kg",
    fontsize=14,
)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=9)

# Set y-axis limits
if np.any(valid_theory) or np.any(valid_solver):
    M_max = max(
        np.nanmax(M_theory_array[valid_theory]) if np.any(valid_theory) else 0,
        np.nanmax(M_solver_array[valid_solver]) if np.any(valid_solver) else 0,
    )
    ax.set_ylim([0, max(1.1, M_max * 1.1)])
else:
    ax.set_ylim([0, 1.1])

plt.tight_layout()

# Save plot
# plot_dir = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets")
# os.makedirs(plot_dir, exist_ok=True)
# plot_filename = os.path.join(plot_dir, "mach_out_vs_ksi_comparison.png")
# plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
# print(f"Plot saved to: {plot_filename}")

# Show plot
plt.show()

# Print summary
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)

if np.any(valid_theory):
    idx_max_th = np.nanargmax(M_theory_array[valid_theory])
    M_max_th = M_theory_array[valid_theory][idx_max_th]
    ksi_at_max_th = ksi_theory_valid[valid_theory][idx_max_th]
    print(f"Theory: Max M = {M_max_th:.6f} at ksi = {ksi_at_max_th:.6e}")
if np.any(valid_solver):
    idx_max_sol = np.nanargmax(M_solver_array[valid_solver])
    M_max_sol = M_solver_array[valid_solver][idx_max_sol]
    ksi_at_max_sol = ksi_solver_valid[valid_solver][idx_max_sol]
    print(f"Solver: Max M = {M_max_sol:.6f} at ksi = {ksi_at_max_sol:.6e}")

print("=" * 70)
