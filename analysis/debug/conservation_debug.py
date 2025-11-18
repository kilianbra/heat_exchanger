"""
Debug script to reproduce the conservation debug message:
DEBUG:heat_exchanger.conservation:Fluid step is not within desired tolerances:
Individual residuals: |dh_t|=1.00e+12 (want < 1.46e+02), |d(p+G²/ρ)|=1.91e+12 (want < 2.79e+02) |
State: (T_a=275.0 K, p_b=2.79e+06 Pa) | Inputs: (G=664.2 kg/m²s, dh0=3.08e+06 J/kg, tau_dA_over_A_c=6.98e+06)
"""

import logging

from heat_exchanger.conservation import update_static_properties
from heat_exchanger.fluids.protocols import PerfectGasFluid, CoolPropFluid
from heat_exchanger.logging_utils import configure_logging

# Configure logging to DEBUG to see the conservation debug message
configure_logging(logging.DEBUG)

# Set the conservation logger to DEBUG level
logging.getLogger("heat_exchanger.conservation").setLevel(logging.DEBUG)

print("=" * 70)
print("Testing update_static_properties with parameters that cause")
print("non-convergence to reproduce the debug message")
print("=" * 70)
print()

# Parameters from the debug message (total values)
T_a_initial = 275.0  # K
p_b_initial = 2.79e6  # Pa (2.79 MPa)
G = 664.2  # kg/m²s
dh0_total = 3.08e6  # J/kg (total)
tau_dA_over_A_c_total = 6.98e6  # Pa (total)

# Divide by 100 for sequential steps
n_steps = 200
dh0_step = dh0_total / n_steps  # J/kg per step
tau_dA_over_A_c_step = tau_dA_over_A_c_total / n_steps  # Pa per step

# Create fluid model (likely Para_Hydrogen based on full_flow.py)
fluid = PerfectGasFluid.from_name("Para_Hydrogen")
fluid = CoolPropFluid("ParaHydrogen")
fluid_in = fluid.state(T_a_initial, p_b_initial)

expected_dT = dh0_total / fluid_in.cp
expected_T_final = T_a_initial + expected_dT

print("Fluid: Para-Hydrogen")
print("Input parameters:")
print(f"  T_a_initial = {T_a_initial:.1f} K")
print(f"  p_b_initial = {p_b_initial:.2e} Pa ({p_b_initial / 1e5:.2f} bar)")
print(f"  G = {G:.1f} kg/m²s")
print(f"  dh0_total = {dh0_total:.2e} J/kg (i.e. dT0_total = {dh0_total / fluid_in.cp:.2f} K)")
print(f"  tau_dA_over_A_c_total = {tau_dA_over_A_c_total:.2e} Pa ")
print()
print(f"Applying in {n_steps} sequential steps:")
print(f"  dh0_step = {dh0_step:.2e} J/kg per step (i.e. dT0_step = {dh0_step / fluid_in.cp:.2f} K)")
print(f"  tau_dA_over_A_c_step = {tau_dA_over_A_c_step:.2e} Pa per step")
print()


# Initialize state
T_current = T_a_initial
p_current = p_b_initial

# Track convergence issues
convergence_failures = 0
n_final = None  # Will be set if loop breaks early

try:
    for step in range(0, n_steps):
        T_prev = T_current
        p_prev = p_current

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
        if step % 5 == 0 or step == n_steps:
            dT = T_current - T_a_initial
            dp = p_current - p_b_initial
            f_current = fluid.state(T_current, p_current)
            V_current = G / f_current.rho
            Mach_current = V_current / f_current.a
            print(
                f"Step {step:3d}: M={Mach_current:.2f}, T = {T_current:.2f} K (ΔT = {dT:+6.2f} K), p = {p_current:.2e} Pa (Δp = {dp / p_b_initial * 100:6.2f} % inlet)"
            )
            if T_current > expected_T_final:
                print("Aborting due to T_current > expected_T_final")
                n_final = step
                break
            elif Mach_current > 0.6:
                print("Aborting due to Mach_current > 0.6")
                n_final = step
                break
    # Make sure n_taken is always defined after the loop
    n_taken = n_final if n_final is not None else n_steps
    print()
    print(f"Final state after {n_taken} of {n_steps} steps:")
    print(f"  T_final = {T_current:.2f} K (ΔT_total = {T_current - T_a_initial:+.2f} K)")
    print(f"  p_final = {p_current:.2e} Pa (Δp_total/p_in = {100 * (p_current - p_b_initial) / p_b_initial:.2f} %)")

    # Calculate expected final state (approximate, assuming constant cp)
    if n_final is None:
        print()
        print(f"Expected final T (assuming constant cp): {expected_T_final:.2f} K")
        print(f"Difference from expected: {T_current - expected_T_final:.2f} K")
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
