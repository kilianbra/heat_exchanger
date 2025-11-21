"""
Compressible flow relations for constant area flow with both friction and heat transfer.

Based on Sturas 1971 equations for compressible flow with friction and heat addition.
https://ntrs.nasa.gov/api/citations/19720004565/downloads/19720004565.pdf

ksi = 4fx/d_h = f A_w / A_o is the friction-distance or friction-area parameter and assumes average friction factor
f = tau_w / (0.5 * rho * V^2) is the friction factor
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, fsolve


def p_static_over_p_static_in(M_0, M, k, ksi, gamma=1.4):
    """
    Calculate pressure ratio p/p_0 (static pressure at exit / stagnation pressure at inlet).
    Sturas 1971 equation 18.

    Parameters
    ----------
    M_0 : float
        Inlet Mach number
    M : float
        Exit Mach number
    k : float
        Parameter k
    ksi : float or array
        Parameter ξ (ksi)
    gamma : float, optional
        Ratio of specific heats (default: 1.4)

    Returns
    -------
    float or array
        Pressure ratio p/p_0 (static pressure at exit / static pressure at inlet)
    """
    # Check that (1 + k*ksi) > 0 for valid solution
    term = 1 + k * ksi
    if np.any(term <= 0):
        raise ValueError(f"Invalid: (1 + k*ksi) must be > 0. Got k={k}, ksi={ksi}")

    # Check that M > 0
    if np.any(M <= 0):
        raise ValueError(f"Invalid: M must be > 0. Got M={M}")

    # Calculate the ratio term: (1 + (gamma - 1)/2 * M_0^2) / (1 + (gamma - 1)/2 * M^2)
    numerator = 1 + (gamma - 1) / 2 * M_0**2
    denominator = 1 + (gamma - 1) / 2 * M**2

    # Calculate the expression inside the square bracket
    bracket_term = (numerator / denominator) * term

    # Calculate pressure ratio: p/p_0 = (M_0 / M) * sqrt(bracket_term)
    p_over_p0 = (M_0 / M) * np.sqrt(bracket_term)

    return p_over_p0


def p_static_limiting_over_p_stag_in(M_0, k, ksi, gamma=1.4):
    """
    Calculate pressure ratio p_l/p_0.
    Sturas 1971 equation 24.

    Parameters
    ----------
    M_0 : float
        Inlet Mach number
    k : float
        Parameter k
    ksi : float or array
        Parameter ξ (ksi)
    gamma : float, optional
        Ratio of specific heats (default: 1.4)

    Returns
    -------
    float or array
        Pressure p_static_limiting_over_p_stag_in p_lim/P_0
    """
    # Check that (1 + k*ksi) > 0 for valid solution
    term = 1 + k * ksi
    if np.any(term <= 0):
        raise ValueError(f"Invalid: (1 + k*ksi) must be > 0. Got k={k}, ksi={ksi}")

    # Calculate the fraction term
    numerator = 1 + (gamma - 1) / 2 * M_0**2
    denominator = 1 + (gamma - 1) / 2

    # Calculate the expression inside the square bracket
    bracket_term = (numerator / denominator) * term

    # Calculate pressure ratio
    p_static_limiting_over_p_stag_in = M_0 * np.sqrt(bracket_term)

    return p_static_limiting_over_p_stag_in


def V_squared_from_M_tau(M, tau, gamma=1.4):
    """
    Calculate V^2 from Mach number and tau using equation 7.
    Sturas 1971 equation 7.

    Parameters
    ----------
    M : float
        Mach number
    tau : float
        Parameter τ = 1 + k*ksi
    gamma : float, optional
        Ratio of specific heats (default: 1.4)

    Returns
    -------
    float
        V^2 (dimensionless flow parameter squared)
    """
    numerator = gamma * M**2 * tau
    denominator = 1 + ((gamma - 1) / 2) * M**2
    V_squared = numerator / denominator
    return V_squared


def M_squared_from_V_tau(V, tau, gamma=1.4):
    """
    Calculate M^2 from V and tau using equation 6.
    Sturas 1971 equation 6.

    Parameters
    ----------
    V : float
        Dimensionless flow parameter
    tau : float
        Parameter τ = 1 + k*ksi
    gamma : float, optional
        Ratio of specific heats (default: 1.4)

    Returns
    -------
    float
        M^2 (Mach number squared)
    """
    numerator = V**2
    denominator = gamma * tau - ((gamma - 1) / 2) * V**2

    if denominator <= 0:
        raise ValueError(f"Invalid: denominator must be > 0. Got gamma={gamma}, tau={tau}, V={V}")

    M_squared = numerator / denominator
    return M_squared


def ksi_from_V_V0(V, V_0, k, gamma=1.4):
    """
    Calculate ksi (ξ) from V and V_0 using equation 13.
    Sturas 1971 equation 13.

    Parameters
    ----------
    V : float
        Dimensionless flow parameter at distance x
    V_0 : float
        Dimensionless flow parameter at inlet
    k : float
        Parameter k
    gamma : float, optional
        Ratio of specific heats (default: 1.4)

    Returns
    -------
    float
        ksi (ξ) value
    """
    # Term 1: V / V_0
    term1 = V / V_0

    # Term 2: sqrt((V_0^2 + 2k) / (V^2 + 2k))
    term2_num = V_0**2 + 2 * k
    term2_den = V**2 + 2 * k

    if term2_den <= 0:
        raise ValueError(f"Invalid: (V^2 + 2k) must be > 0. Got V={V}, k={k}")

    term2 = np.sqrt(term2_num / term2_den)

    # Term 3: (V/2) * ((γ+1)/γ) / sqrt(V^2 + 2k) * ln(...)
    sqrt_V2_2k = np.sqrt(V**2 + 2 * k)
    sqrt_V0_2_2k = np.sqrt(V_0**2 + 2 * k)

    # Numerator of ln argument
    ln_num = V_0**2 + k + V_0 * sqrt_V0_2_2k

    # Denominator of ln argument
    ln_den = V**2 + k + V * sqrt_V2_2k

    if ln_den <= 0 or ln_num <= 0:
        raise ValueError(f"Invalid: ln arguments must be > 0. Got V={V}, V_0={V_0}, k={k}")

    ln_term = np.log(ln_num / ln_den)

    term3 = (V / 2) * ((gamma + 1) / gamma) / sqrt_V2_2k * ln_term

    # Combine all terms
    ksi = term1 * term2 / k - 1 / k + term3

    return ksi


def solve_V_from_ksi(ksi, V_0, k, gamma=1.4, V_guess=None):
    """
    Solve for V given ksi using equation 13.
    This is an inverse problem: given ksi, find V.

    Parameters
    ----------
    ksi : float
        Parameter ξ (ksi)
    V_0 : float
        Dimensionless flow parameter at inlet
    k : float
        Parameter k
    gamma : float, optional
        Ratio of specific heats (default: 1.4)
    V_guess : float, optional
        Initial guess for V (default: V_0)

    Returns
    -------
    float
        V (dimensionless flow parameter at distance x)
    """
    if V_guess is None:
        V_guess = V_0

    def residual(V_val):
        """Residual function: ksi_calculated - ksi_target"""
        try:
            ksi_calc = ksi_from_V_V0(V_val, V_0, k, gamma)
            return ksi_calc - ksi
        except (ValueError, ZeroDivisionError):
            # Return a large residual if V is invalid
            return 1e6

    # Solve for V
    V_solution = fsolve(residual, V_guess, xtol=1e-10)[0]

    return V_solution


def calculate_V0_from_M0(M_0, k, ksi_0=0.0, gamma=1.4):
    """
    Calculate V_0 from inlet Mach number M_0.
    At inlet, ksi_0 = 0, so tau_0 = 1 + k*ksi_0 = 1.

    Parameters
    ----------
    M_0 : float
        Inlet Mach number
    k : float
        Parameter k (not used for inlet calculation but kept for consistency)
    ksi_0 : float, optional
        ksi at inlet (default: 0.0)
    gamma : float, optional
        Ratio of specific heats (default: 1.4)

    Returns
    -------
    float
        V_0 (dimensionless flow parameter at inlet)
    """
    tau_0 = 1 + k * ksi_0  # At inlet, ksi_0 = 0, so tau_0 = 1
    V_0_squared = V_squared_from_M_tau(M_0, tau_0, gamma)
    V_0 = np.sqrt(V_0_squared)
    return V_0


def solve_M_from_ksi(ksi, M_0, k, gamma=1.4, V_guess=None):
    """
    Complete solution: given ksi and inlet Mach number, solve for V and then M.

    Parameters
    ----------
    ksi : float
        Parameter ξ (ksi) at distance x
    M_0 : float
        Inlet Mach number
    k : float
        Parameter k
    gamma : float, optional
        Ratio of specific heats (default: 1.4)
    V_guess : float, optional
        Initial guess for V (default: None, will use V_0)

    Returns
    -------
    tuple
        (V, M) where V is dimensionless flow parameter and M is Mach number
    """
    # Step 1: Calculate V_0 from inlet conditions
    V_0 = calculate_V0_from_M0(M_0, k, ksi_0=0.0, gamma=gamma)

    # Step 2: Solve for V from ksi using equation 13
    if V_guess is None:
        V_guess = V_0

    V = solve_V_from_ksi(ksi, V_0, k, gamma=gamma, V_guess=V_guess)

    # Step 3: Calculate tau at distance x
    tau = 1 + k * ksi

    # Step 4: Calculate M from V and tau using equation 6
    M_squared = M_squared_from_V_tau(V, tau, gamma)
    M = np.sqrt(M_squared)

    return V, M


def calculate_ksi_lim(M_0, k, max_p_ratio=10.0, gamma=1.4):
    """
    Calculate ksi_lim such that the pressure ratio doesn't exceed max_p_ratio.

    Parameters
    ----------
    M_0 : float
        Inlet Mach number
    k : float
        Parameter k
    max_p_ratio : float, optional
        Maximum allowed pressure ratio (default: 10.0)
    gamma : float, optional
        Ratio of specific heats (default: 1.4)

    Returns
    -------
    float
        ksi_lim value
    """
    if k == 0:
        # If k=0, the equation simplifies and ksi doesn't affect it
        # Return a reasonable default
        return 1.0

    # Solve for ksi such that p_ratio = max_p_ratio
    # p_ratio = M_0 * sqrt( (numerator/denominator) * (1 + k*ksi) )
    # max_p_ratio^2 = M_0^2 * (numerator/denominator) * (1 + k*ksi)
    # (max_p_ratio^2) / (M_0^2 * numerator/denominator) = 1 + k*ksi
    # ksi = ((max_p_ratio^2) / (M_0^2 * numerator/denominator) - 1) / k

    numerator = 1 + (gamma - 1) / 2 * M_0**2
    denominator = 1 + (gamma - 1) / 2

    ksi_lim = ((max_p_ratio**2) / (M_0**2 * numerator / denominator) - 1) / k

    # Ensure ksi_lim is positive
    return max(ksi_lim, 0.0)


def find_ksi_lim_adaptive(M_in, k, ksi_start=0.01, ksi_max=50.0, dM_threshold=1e-6, gamma=1.4):
    """
    Find ksi_lim (where M reaches maximum < 1.0) using adaptive stepping.
    Steps gradually, slowing down as M approaches 1.0, and stops when M decreases
    (supersonic branch) or exceeds 1.0.

    Parameters
    ----------
    M_in : float
        Inlet Mach number
    k : float
        Parameter k
    ksi_start : float, optional
        Starting ksi value (default: 0.01)
    ksi_max : float, optional
        Maximum ksi to search (default: 50.0)
    dM_threshold : float, optional
        Threshold for detecting M decrease (default: 1e-6)
    gamma : float, optional
        Ratio of specific heats (default: 1.4)

    Returns
    -------
    tuple
        (ksi_lim, M_at_ksi_lim) where ksi_lim is the ksi at maximum M < 1.0
    """
    # Handle k=0 case
    if abs(k) < 1e-10:
        return np.nan, np.nan

    # Calculate V_0 once
    try:
        V_0 = calculate_V0_from_M0(M_in, k, ksi_0=0.0, gamma=gamma)
    except Exception:
        return np.nan, np.nan

    # Initial step size
    dksi_initial = 0.1
    dksi_min = 1e-6
    dksi_max = 1.0

    # Track maximum M and corresponding ksi
    ksi_max_M = ksi_start
    M_max = M_in
    V_guess = V_0

    # Track previous values for gradient calculation
    ksi_prev = 0.0
    M_prev = M_in
    dksi = dksi_initial

    ksi = ksi_start

    max_iter = 10000
    iter_count = 0

    while ksi < ksi_max and iter_count < max_iter:
        iter_count += 1

        try:
            # Solve for M at current ksi
            V, M = solve_M_from_ksi(ksi, M_in, k, gamma=gamma, V_guess=V_guess)
            V_guess = V

            # Check if M exceeded 1.0
            if M >= 1.0:
                # Interpolate between previous point and current point to find ksi where M = 1.0
                if ksi_prev < ksi and abs(M - M_prev) > 1e-10:
                    ksi_interp = ksi_prev + (1.0 - M_prev) * (ksi - ksi_prev) / (M - M_prev)
                    # Calculate actual M at interpolated ksi
                    try:
                        V_interp, M_interp = solve_M_from_ksi(ksi_interp, M_in, k, gamma=gamma, V_guess=V_guess)
                        if M_interp < 1.0:
                            return ksi_interp, M_interp
                    except Exception:
                        pass
                # If interpolation fails, return previous point
                return ksi_prev, M_prev

            # Check if M started decreasing (supersonic branch detected)
            if M_prev - dM_threshold > M and M_prev > 0.9:
                # M is decreasing, we've hit supersonic branch
                # Return the point with maximum M
                return ksi_max_M, M_max

            # Update maximum M tracking
            if M_max < M:
                M_max = M
                ksi_max_M = ksi

            # Calculate gradient (dM/dksi) for adaptive step sizing
            if ksi > ksi_prev:
                dM_dksi = (M - M_prev) / (ksi - ksi_prev)
            else:
                dM_dksi = 1.0  # Default gradient

            # Adaptive step sizing: slow down as M approaches 1.0 and as gradient decreases
            # Step size is inversely proportional to how close we are to M=1.0
            # and inversely proportional to gradient magnitude
            if M > 0.9:
                # Close to choking, use very small steps
                step_factor = max(0.01, (1.0 - M) / 0.1)
            else:
                # Further from choking, use larger steps
                step_factor = 1.0

            # Adjust step size based on gradient
            if abs(dM_dksi) > 1e-10:
                gradient_factor = min(1.0, max(0.01, 0.1 / abs(dM_dksi)))
            else:
                gradient_factor = 0.01

            # Update step size
            dksi = dksi_initial * step_factor * gradient_factor
            dksi = max(dksi_min, min(dksi_max, dksi))

            # Update previous values
            ksi_prev = ksi
            M_prev = M

            # Step forward
            ksi += dksi

        except (ValueError, RuntimeError):
            # If solution fails, try smaller step
            dksi *= 0.5
            if dksi < dksi_min:
                # Can't proceed further, return best found so far
                return ksi_max_M, M_max
            continue
        except Exception:
            # Other errors, try smaller step
            dksi *= 0.5
            if dksi < dksi_min:
                return ksi_max_M, M_max
            continue

    # Reached ksi_max or max_iter, return best found
    return ksi_max_M, M_max


def solve_ksi_lim_from_M_in(M_in, k, ksi_bracket=None, gamma=1.4):
    """
    Solve for ksi_lim where Mach number reaches 1.0 (choked flow).
    Uses brentq since M(ksi) is monotonically increasing.

    Parameters
    ----------
    M_in : float
        Inlet Mach number
    k : float
        Parameter k
    ksi_bracket : tuple, optional
        Bracket (ksi_low, ksi_high) for brentq (default: None, will be found automatically)
    gamma : float, optional
        Ratio of specific heats (default: 1.4)

    Returns
    -------
    float
        ksi_lim value where M = 1.0
    """

    def residual(ksi_val):
        """Residual function: M(ksi) - 1.0"""
        try:
            _, M = solve_M_from_ksi(ksi_val, M_in, k, gamma=gamma)
            return M - 1.0
        except (ValueError, RuntimeError, ZeroDivisionError):
            # Return NaN to signal invalid ksi
            return np.nan

    # Find bracket if not provided
    if ksi_bracket is None:
        # Start with a reasonable bracket
        ksi_low = 0.0
        ksi_high = 20.0

        # Check if we can find a valid bracket
        try:
            M_low, _ = solve_M_from_ksi(ksi_low, M_in, k, gamma=gamma)
            if M_low >= 1.0:
                # Already choked at ksi=0, return 0
                return 0.0
        except Exception:
            return np.nan

        # Find upper bound where M > 1.0
        max_iter = 50
        for _ in range(max_iter):
            try:
                M_high, _ = solve_M_from_ksi(ksi_high, M_in, k, gamma=gamma)
                if M_high > 1.0:
                    break
                ksi_high *= 2.0
            except Exception:
                ksi_high *= 2.0
                if ksi_high > 1000.0:
                    return np.nan

        ksi_bracket = (ksi_low, ksi_high)

    # Use brentq for robust root finding
    try:
        ksi_lim = brentq(residual, ksi_bracket[0], ksi_bracket[1], xtol=1e-10, maxiter=100)
        # Verify the solution
        _, M_check = solve_M_from_ksi(ksi_lim, M_in, k, gamma=gamma)
        if abs(M_check - 1.0) > 1e-5:
            # Solution didn't converge properly
            return np.nan
        return ksi_lim
    except (ValueError, RuntimeError):
        return np.nan
