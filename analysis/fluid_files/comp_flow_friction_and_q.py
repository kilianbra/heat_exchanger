"""
Plot compressible flow relations for constant area flow with both friction and heat transfer.

Analysis and plotting functions for compressible flow with friction and heat transfer.
Source code functions are imported from heat_exchanger.fluids.compressible_flow_friction_heat
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from heat_exchanger.fluids.compressible_flow_friction_heat import (
    M_squared_from_V_tau,
    calculate_V0_from_M0,
    find_ksi_lim_adaptive,
    ksi_from_V_V0,
    solve_M_from_ksi,
    solve_V_from_ksi,
)


def plot_mach_vs_ksi():
    """
    Plot Mach number as a function of ksi for different k values.
    For k < 0: Interpolates to find ksi where M = 1.0 and plots up to that point.
    For k > 0: Detects supersonic branch (M decreasing) and stops at maximum M.
    """
    # Fixed parameters
    M_in = 0.15
    gamma = 1.4
    k_values = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # ksi range: 0 to 16
    ksi_max = 16.0
    n_points = 500  # Fine resolution for smooth curves
    ksi_array = np.linspace(0, ksi_max, n_points)

    # Mach number threshold for choking (for positive k)
    M_choke = 0.999

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for each k value
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

    for k, color in zip(k_values, colors, strict=True):
        ksi_valid = []
        M_valid = []

        # Handle k=0 as special case (equation 13 has division by k)
        if abs(k) < 1e-10:
            # For k=0, tau = 1 (constant), so M should remain constant
            # We can't use equation 13 directly, so we'll skip k=0 for now
            # or calculate M directly from tau=1
            print("Note: k=0 case not implemented (equation 13 requires k != 0)")
            continue

        # Calculate V_0 once for this k value
        try:
            V_0 = calculate_V0_from_M0(M_in, k, ksi_0=0.0, gamma=gamma)
        except Exception as e:
            print(f"Warning: Could not calculate V_0 for k={k}: {e}")
            continue

        # Use previous V as guess for next iteration (helps convergence)
        V_guess = V_0

        # Track if flow choked
        is_choked = False

        # For k > 0: track maximum M and detect supersonic branch
        max_M = M_in
        ksi_at_max_M = 0.0
        last_M = M_in

        # For k < 0: track points around M = 1.0 for interpolation
        ksi_before_M1 = None
        M_before_M1 = None

        for ksi in ksi_array:
            try:
                # Solve for V and M from ksi
                V, M = solve_M_from_ksi(ksi, M_in, k, gamma=gamma, V_guess=V_guess)

                # Update guess for next iteration
                V_guess = V

                # Handle k < 0: find first point where M > 1.0
                if k < 0:
                    if M < 1.0:
                        # Still subsonic, add point
                        ksi_valid.append(ksi)
                        M_valid.append(M)
                        ksi_before_M1 = ksi
                        M_before_M1 = M
                    elif M >= 1.0:
                        # First point above M = 1.0
                        if ksi_before_M1 is not None:
                            # Interpolate to find ksi where M = 1.0
                            # Linear interpolation: M = M_before + (M_after - M_before) * (ksi - ksi_before) / (ksi_after - ksi_before)
                            # Solve for ksi when M = 1.0
                            ksi_interp = ksi_before_M1 + (1.0 - M_before_M1) * (ksi - ksi_before_M1) / (M - M_before_M1)

                            # Calculate actual M at interpolated ksi
                            try:
                                V_interp, M_interp = solve_M_from_ksi(ksi_interp, M_in, k, gamma=gamma, V_guess=V_guess)
                                ksi_valid.append(ksi_interp)
                                M_valid.append(M_interp)
                                is_choked = True
                                print(
                                    f"k={k:.2f}: Interpolated to M=1.0 at ksi={ksi_interp:.4f}, actual M={M_interp:.6f}"
                                )
                            except Exception:
                                # If interpolation fails, use last valid point
                                print(
                                    f"k={k:.2f}: Interpolation failed, using last point at ksi={ksi_before_M1:.4f}, M={M_before_M1:.4f}"
                                )
                        else:
                            # No valid point before M=1.0, just stop
                            print(f"k={k:.2f}: M exceeded 1.0 immediately, stopping")
                        break

                # Handle k > 0: detect supersonic branch (M decreasing)
                elif k > 0:
                    # Update maximum M tracking before checking for decrease
                    if max_M < M:
                        max_M = M
                        ksi_at_max_M = ksi

                    # Check if M is decreasing (supersonic branch)
                    if last_M > M and last_M > 0.9:
                        # M started decreasing, we've jumped to supersonic branch
                        # Find index of point closest to ksi_at_max_M (with tolerance)
                        idx_max = len(ksi_valid) - 1  # Default to last point
                        tol = 1e-6
                        for i, ksi_val in enumerate(ksi_valid):
                            if abs(ksi_val - ksi_at_max_M) < tol:
                                idx_max = i
                                break
                        # Truncate arrays to max_M point
                        ksi_valid = ksi_valid[: idx_max + 1]
                        M_valid = M_valid[: idx_max + 1]
                        is_choked = True
                        print(
                            f"k={k:.2f}: Detected supersonic branch (M decreasing from {last_M:.4f} to {M:.4f}), stopping at ksi={ksi_at_max_M:.4f}, M={max_M:.4f}"
                        )
                        break

                    # Check if choked (use M_choke threshold)
                    if M_choke <= M:
                        ksi_valid.append(ksi)
                        M_valid.append(M)
                        is_choked = True
                        print(f"k={k:.2f}: Flow choked at ksi={ksi:.4f}, M={M:.4f}")
                        break

                    # Add valid point
                    ksi_valid.append(ksi)
                    M_valid.append(M)
                    last_M = M

            except (ValueError, RuntimeError) as e:
                # If solution fails, stop this curve
                if ksi_valid:  # Only print if we had some valid points
                    print(f"Warning: k={k:.2f} failed at ksi={ksi:.4f}: {e}")
                break
            except Exception as e:
                # Other errors - skip this point but continue
                print(f"Warning: k={k:.2f} error at ksi={ksi:.4f}: {e}")
                continue

        # Plot this k value's curve
        if ksi_valid:
            # Use dashed line for negative k values
            linestyle = "--" if k < 0 else "-"
            ax.plot(ksi_valid, M_valid, label=f"k = {k:.2f}", color=color, linewidth=2, linestyle=linestyle)

            # Add label at the end of the curve
            if ksi_valid and M_valid:
                last_ksi = ksi_valid[-1]
                last_M = M_valid[-1]

                if is_choked:
                    # Label above if choked (vertical)
                    ax.text(
                        last_ksi,
                        last_M,
                        f" {k:.2f}",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                        fontsize=9,
                        color=color,
                        rotation=90,
                    )
                else:
                    # Label to the right if not choked (vertical)
                    ax.text(
                        last_ksi,
                        last_M,
                        f" {k:.2f}",
                        verticalalignment="center",
                        horizontalalignment="left",
                        fontsize=9,
                        color=color,
                        rotation=0,
                    )

    # Formatting
    ax.set_xlabel("ξ (ksi) = 4fx/d_h = f A_w / A_o", fontsize=12)
    ax.set_ylabel("Mach number M", fontsize=12)
    # Use raw LaTeX for xi and Gamma, and print their values in the title
    title = (
        rf"Mach Number vs $\xi$ ($M_{{\rm in}} = {M_in}$, $\gamma = {gamma}$) for different $k$ values"
        + r"   $\dot{Q} = k\cdot  \dot{m} c_p T_{\rm stag,in}$"
    )
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    # ax.legend(loc="best", fontsize=10)

    # Add a horizontal line at M = 0.99 to indicate choking
    ax.axhline(y=M_choke, color="r", linestyle="--", alpha=0.5, label=f"Choking (M={M_choke})")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 16.0)

    plt.tight_layout()
    return fig, ax


def test_equations_6_7_13():
    """
    Test function demonstrating the use of equations 6, 7, and 13.
    """
    # Test parameters
    M_0 = 0.15
    gamma = 1.4
    k = 0.1

    print("Testing equations 6, 7, and 13:")
    print(f"M_0 = {M_0}, gamma = {gamma}, k = {k}")
    print()

    # Step 1: Calculate V_0 from inlet Mach number (equation 7)
    V_0 = calculate_V0_from_M0(M_0, k, ksi_0=0.0, gamma=gamma)
    print(f"Step 1: V_0 = {V_0:.6f} (from equation 7)")

    # Step 2: Test equation 13 - calculate ksi for a given V
    V_test = V_0 * 1.2  # Test with V slightly less than V_0
    ksi_calc = ksi_from_V_V0(V_test, V_0, k, gamma=gamma)
    print(f"Step 2: For V = {V_test:.6f}, ksi = {ksi_calc:.6f} (from equation 13)")

    # Step 3: Solve inverse problem - find V from ksi
    ksi_target = 0.1
    V_solved = solve_V_from_ksi(ksi_target, V_0, k, gamma=gamma, V_guess=V_0)
    print(f"Step 3: For ksi = {ksi_target:.6f}, solved V = {V_solved:.6f} (inverse of equation 13)")

    # Step 4: Calculate M from V and tau (equation 6)
    tau = 1 + k * ksi_target
    M_squared = M_squared_from_V_tau(V_solved, tau, gamma=gamma)
    M = np.sqrt(M_squared)
    print(f"Step 4: For V = {V_solved:.6f}, tau = {tau:.6f}, M = {M:.6f} (from equation 6)")

    # Step 5: Complete solution using solve_M_from_ksi
    V_complete, M_complete = solve_M_from_ksi(ksi_target, M_0, k, gamma=gamma)
    print(f"Step 5: Complete solution for ksi = {ksi_target:.6f}: V = {V_complete:.6f}, M = {M_complete:.6f}")
    print()

    # Verify: Check that we can recover ksi from the solved V
    ksi_verify = ksi_from_V_V0(V_complete, V_0, k, gamma=gamma)
    print(f"Verification: ksi from solved V = {ksi_verify:.6f} (target was {ksi_target:.6f})")
    print(f"Error: {abs(ksi_verify - ksi_target):.2e}")


if __name__ == "__main__":
    # Test the new equations
    # test_equations_6_7_13()

    # Test find_ksi_lim_adaptive function
    print("Testing find_ksi_lim_adaptive function")
    print("=" * 60)
    gamma = 1.4

    # Easy to add test cases: just (M_in, k) pairs
    test_cases = [(0.15, 0.1), (0.15, -0.3), (0.6, 0.1), (0.9, 0.5), (0.02, 1.5), (0.22, 0.75)]

    # Suppress warnings during tests
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for M_in, k in test_cases:
            # Format: M_in always 2 decimals, k always shows sign
            k_str = f"{k:+.2f}"  # + sign for positive, - for negative
            print(f"M_in = {M_in:.2f}, k = {k_str} (γ = {gamma}):", end=" ")
            try:
                ksi_lim, M_at_ksi_lim = find_ksi_lim_adaptive(M_in, k, gamma=gamma)
                if np.isnan(ksi_lim):
                    print("Failed to find ksi_lim")
                else:
                    # Verify by calculating M at ksi_lim
                    try:
                        _, M_verify = solve_M_from_ksi(ksi_lim, M_in, k, gamma=gamma)
                        diff = abs(M_verify - M_at_ksi_lim)

                        if diff > 1e-3:
                            print(
                                f"ksi_lim = {ksi_lim:>6.3f}, M = {M_at_ksi_lim:.6f} "
                                f"(WARNING: verification M = {M_verify:.6f}, diff = {diff:.2e})"
                            )
                        else:
                            print(f"ksi_lim = {ksi_lim:>6.3f}, M = {M_at_ksi_lim:.6f} (verified: diff = {diff:.2e})")
                    except Exception as e:
                        print(f"ksi_lim = {ksi_lim:>6.3f}, M = {M_at_ksi_lim:.6f} (verification failed: {e})")
            except Exception as e:
                print(f"Error: {e}")

    print("=" * 60)

    # Create the plots
    # fig1, ax1 = plot_mach_vs_ksi()
    # fig2, ax2 = plot_ksi_lim_vs_M_in()

    # Show the plots
    # plt.show()

    # Optionally save the figures
    # fig1.savefig('mach_vs_ksi.png', dpi=300, bbox_inches='tight')
    # fig2.savefig('ksi_lim_vs_M_in.png', dpi=300, bbox_inches='tight')
