# ruff: noqa: I001
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import least_squares

from heat_exchanger.fluids.protocols import FluidModel


logger = logging.getLogger(__name__)


def update_static_properties(
    fluid: FluidModel,
    G,
    dh0,
    tau_dA_over_A_c,
    T_a,
    p_b,
    a_is_in=True,
    b_is_in=True,
    max_iter=50,
    tol_T=1e-4,  # K tolerance for temperature change
    rel_tol_p=1e-3,  # % tolerance for pressure drop (strictly speaking of p + G^2/rho)
):
    r"""
    Solve simultaneously for static temperature T_not_a and static pressure p_not_b so that:
      1) Energy/stagnation enthalpy: (h_out + 0.5*(G^2/rho_out^2)) - (h_in + 0.5*(G^2/rho_in^2)) = dh0
      2) Momentum/impulse:           (p_out + G^2/rho_out) - (p_in + G^2/rho_in) = - tau * dA / A_c

    a can either be in (if a_is_in is True) or out (if a_is_in is False) of the heat exchanger.
    b can either be in (if b_is_in is True) or out (if b_is_in is False) of the heat exchanger.

    Tolerances and finite-difference steps:
      - tol_T: Absolute convergence tolerance on the energy residual R1 (units of J/kg).
               When |R1| < cp_in * tol_T, the energy equation is considered converged.
      - rel_tol_p: Relative convergence tolerance on the momentum residual R2, scaled by p_b.
                   Converged when |R2| < rel_tol_p / 100 * p_b (units of Pa).

    Note:  tau_eff dA_friction / A_cross_section > 0.

    Returns:
        (T_not_a, p_not_b)
    """

    # ------------------------------------------------------------
    # 1) Initial Guesses for T_non_a assumes no pressure drop for c_p
    # ------------------------------------------------------------
    # Get cp at reference state (T_a, p_b) for scaling
    state_ref = fluid.state(T_a, p_b)
    cp_ref = state_ref.cp

    # Could improve guess by then using c_p(T_avg) to get T_guess
    if a_is_in:
        T_in = T_a
        T_initial_guess = T_in + dh0 / cp_ref if cp_ref != 0 else T_in
        tol_dh0 = cp_ref * tol_T
    else:
        T_out = T_a
        T_initial_guess = T_out - dh0 / cp_ref if cp_ref != 0 else T_out
        tol_dh0 = cp_ref * tol_T

    # For p_guess, a naive shift by dFA is typical (neglect density change)
    p_initial_guess = p_b - tau_dA_over_A_c if b_is_in else p_b + tau_dA_over_A_c

    tol_dFA = rel_tol_p / 100 * p_b

    # ------------------------------------------------------------
    # 2) Variable scaling for better numerical behavior
    # ------------------------------------------------------------
    # Scale variables to O(1) to help the solver
    # Temperature scale: use a typical temperature change scale
    T_scale = max(abs(dh0 / cp_ref) if cp_ref != 0 else 100.0, 100.0)  # K
    # Pressure scale: use a typical pressure change scale (e.g., 10% of p_b)
    p_scale = max(abs(tau_dA_over_A_c), p_b * 0.1)  # Pa

    # Reference values for scaling
    T_ref = T_a
    p_ref = p_b

    # Scaled initial guess
    x0_scaled = np.array(
        [
            (T_initial_guess - T_ref) / T_scale,
            (p_initial_guess - p_ref) / p_scale,
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # 3) Helper function: compute scaled residuals
    # ------------------------------------------------------------
    def fluid_residuals_scaled(x_scaled: np.ndarray) -> np.ndarray:
        """
        Returns dimensionless scaled residuals [F1, F2] given scaled variables.
        Both residuals are scaled to be O(1) at convergence.
        """
        # Unscale variables
        T_guess = x_scaled[0] * T_scale + T_ref
        p_guess = x_scaled[1] * p_scale + p_ref

        # Guard against non-physical states that would break the fluid model.
        if T_guess <= 0 or p_guess <= 0:
            large_residual = 1e6  # Large but not extreme for least_squares
            return np.array([large_residual, large_residual], dtype=float)

        # Build local variables for both sides to avoid scoping issues
        # Pressures
        if b_is_in:
            p_in_loc = p_b
            p_out_loc = p_guess
        else:
            p_in_loc = p_guess
            p_out_loc = p_b

        if a_is_in:
            T_in_loc = T_a
            T_out_loc = T_guess
        else:
            T_in_loc = T_guess
            T_out_loc = T_a

        state_in = fluid.state(T_in_loc, p_in_loc)
        state_out = fluid.state(T_out_loc, p_out_loc)

        rho_in_loc = state_in.rho
        rho_out_loc = state_out.rho

        h_in = state_in.h
        h_out = state_out.h

        # Stagnation enthalpies (per unit mass)
        h0_in = h_in + 0.5 * (G / rho_in_loc) ** 2
        h0_out = h_out + 0.5 * (G / rho_out_loc) ** 2

        # Physical residuals
        R1 = (h0_out - h0_in) - dh0  # J/kg
        R2 = (p_out_loc + G**2 / rho_out_loc) - (p_in_loc + G**2 / rho_in_loc) + tau_dA_over_A_c  # Pa

        # Scale residuals to be dimensionless and O(1) at convergence
        F1 = R1 / tol_dh0  # Dimensionless, should be < 1 at convergence
        F2 = R2 / tol_dFA  # Dimensionless, should be < 1 at convergence

        return np.array([F1, F2], dtype=float)

    # ------------------------------------------------------------
    # 4) Solve using least_squares with proper scaling
    # ------------------------------------------------------------
    # Use x_scale to help the solver understand variable scales
    x_scale = np.array([1.0, 1.0], dtype=float)  # Variables are already scaled to O(1)

    # Tolerance for scaled residuals (both should be < 1.0 for convergence)
    ftol = 1e-6  # Function tolerance: stop when max(|F_i|) < ftol
    xtol = 1e-8  # Variable tolerance: stop when relative change in x < xtol

    sol = least_squares(
        fluid_residuals_scaled,
        x0_scaled,
        method="lm",  # Levenberg-Marquardt is robust for small problems
        ftol=ftol,
        xtol=xtol,
        max_nfev=max_iter,
        x_scale=x_scale,
    )

    # Unscale solution
    x_sol_scaled = sol.x
    T_solution = x_sol_scaled[0] * T_scale + T_ref
    p_solution = x_sol_scaled[1] * p_scale + p_ref

    # Recompute residuals at solution to verify convergence
    F1_final, F2_final = fluid_residuals_scaled(x_sol_scaled)
    R1_final = F1_final * tol_dh0
    R2_final = F2_final * tol_dFA

    # Check convergence: both scaled residuals should be < 1.0
    converged = sol.success and abs(F1_final) < 1.0 and abs(F2_final) < 1.0

    if not converged:
        logger.debug(
            (
                "Fluid not conv after %d it: "
                "Residuals: |dh_t|=%.2e (want < %.2e), |d(p+G²/ρ)|=%.2e (want < %.2e) | "
                "State: (T_a=%.1f K, p_b=%.2e Pa) | Inputs: (G=%.1f kg/m²s, dh0=%.2e J/kg, tau_dA_over_A_c=%.2e)"
            ),
            sol.nfev,
            abs(R1_final),
            tol_dh0,
            abs(R2_final),
            tol_dFA,
            T_a,
            p_b,
            G,
            dh0,
            tau_dA_over_A_c,
        )

    return T_solution, p_solution


def update_s_prop(
    fluid: FluidModel,
    G,
    dh0,
    f_dA_over_A_c,
    T_a,
    p_b,
    a_is_in=True,
    b_is_in=True,
    max_iter=50,
    tol_T=1e-4,  # K tolerance for temperature change
    rel_tol_p=1e-3,  # % tolerance for pressure drop (strictly speaking of p + G^2/rho)
):
    r"""
    Solve simultaneously for static temperature T_not_a and static pressure p_not_b so that:
      1) Energy/stagnation enthalpy: (h_out + 0.5*(G^2/rho_out^2)) - (h_in + 0.5*(G^2/rho_in^2)) = dh0
      2) Momentum/impulse:           (p_out + G^2/rho_out) - (p_in + G^2/rho_in) = - 0.5 G^2/rho_avg * f_dA_over_A_c

    a can either be in (if a_is_in is True) or out (if a_is_in is False) of the heat exchanger.
    b can either be in (if b_is_in is True) or out (if b_is_in is False) of the heat exchanger.

    Tolerances and finite-difference steps:
      - tol_T: Absolute convergence tolerance on the energy residual R1 (units of J/kg).
               When |R1| < cp_in * tol_T, the energy equation is considered converged.
      - rel_tol_p: Relative convergence tolerance on the momentum residual R2, scaled by p_b.
                   Converged when |R2| < rel_tol_p / 100 * p_b (units of Pa).

    Note:  f_dA_over_A_c > 0.

    Returns:
        (T_not_a, p_not_b)
    """

    # ------------------------------------------------------------
    # 1) Initial Guesses for T_non_a assumes no pressure drop for c_p
    # ------------------------------------------------------------
    # Get cp at reference state (T_a, p_b) for scaling
    state_ref = fluid.state(T_a, p_b)
    cp_ref = state_ref.cp

    # Could improve guess by then using c_p(T_avg) to get T_guess
    if a_is_in:
        T_in = T_a
        T_initial_guess = T_in + dh0 / cp_ref if cp_ref != 0 else T_in
        tol_dh0 = cp_ref * tol_T
    else:
        T_out = T_a
        T_initial_guess = T_out - dh0 / cp_ref if cp_ref != 0 else T_out
        tol_dh0 = cp_ref * tol_T

    rho_guess = state_ref.rho
    tau_dA_over_A_c_guess = 0.5 * G**2 / rho_guess * f_dA_over_A_c

    # For p_guess, a naive shift by dFA is typical (neglect density change)
    p_initial_guess = p_b - tau_dA_over_A_c_guess if b_is_in else p_b + tau_dA_over_A_c_guess

    tol_dFA = rel_tol_p / 100 * p_b

    # ------------------------------------------------------------
    # 2) Variable scaling for better numerical behavior
    # ------------------------------------------------------------
    # Scale variables to O(1) to help the solver
    # Temperature scale: use a typical temperature change scale
    T_scale = max(abs(dh0 / cp_ref) if cp_ref != 0 else 100.0, 100.0)  # K
    # Pressure scale: use a typical pressure change scale (e.g., 10% of p_b)
    p_scale = max(abs(tau_dA_over_A_c_guess), p_b * 0.1)  # Pa

    # Reference values for scaling
    T_ref = T_a
    p_ref = p_b

    # Scaled initial guess
    x0_scaled = np.array(
        [
            (T_initial_guess - T_ref) / T_scale,
            (p_initial_guess - p_ref) / p_scale,
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # 3) Helper function: compute scaled residuals
    # ------------------------------------------------------------
    def fluid_residuals_scaled(x_scaled: np.ndarray) -> np.ndarray:
        """
        Returns dimensionless scaled residuals [F1, F2] given scaled variables.
        Both residuals are scaled to be O(1) at convergence.
        """
        # Unscale variables
        T_guess = x_scaled[0] * T_scale + T_ref
        p_guess = x_scaled[1] * p_scale + p_ref

        # Guard against non-physical states that would break the fluid model.
        if T_guess <= 0 or p_guess <= 0:
            large_residual = 1e6  # Large but not extreme for least_squares
            return np.array([large_residual, large_residual], dtype=float)

        # Build local variables for both sides to avoid scoping issues
        # Pressures
        if b_is_in:
            p_in_loc = p_b
            p_out_loc = p_guess
        else:
            p_in_loc = p_guess
            p_out_loc = p_b

        if a_is_in:
            T_in_loc = T_a
            T_out_loc = T_guess
        else:
            T_in_loc = T_guess
            T_out_loc = T_a

        state_in = fluid.state(T_in_loc, p_in_loc)
        state_out = fluid.state(T_out_loc, p_out_loc)

        rho_in_loc = state_in.rho
        rho_out_loc = state_out.rho
        inv_rho_mean = 0.5 * (1 / rho_in_loc + 1 / rho_out_loc)

        h_in = state_in.h
        h_out = state_out.h

        # Stagnation enthalpies (per unit mass)
        h0_in = h_in + 0.5 * (G / rho_in_loc) ** 2
        h0_out = h_out + 0.5 * (G / rho_out_loc) ** 2

        # Physical residuals
        R1 = (h0_out - h0_in) - dh0  # J/kg
        R2 = (
            (p_out_loc + G**2 / rho_out_loc)
            - (p_in_loc + G**2 / rho_in_loc)
            + 0.5 * G**2 * inv_rho_mean * f_dA_over_A_c
        )  # Pa

        # Scale residuals to be dimensionless and O(1) at convergence
        F1 = R1 / tol_dh0  # Dimensionless, should be < 1 at convergence
        F2 = R2 / tol_dFA  # Dimensionless, should be < 1 at convergence

        return np.array([F1, F2], dtype=float)

    # ------------------------------------------------------------
    # 4) Solve using least_squares with proper scaling
    # ------------------------------------------------------------
    # Use x_scale to help the solver understand variable scales
    x_scale = np.array([1.0, 1.0], dtype=float)  # Variables are already scaled to O(1)

    # Tolerance for scaled residuals (both should be < 1.0 for convergence)
    ftol = 1e-6  # Function tolerance: stop when max(|F_i|) < ftol
    xtol = 1e-8  # Variable tolerance: stop when relative change in x < xtol

    sol = least_squares(
        fluid_residuals_scaled,
        x0_scaled,
        method="lm",  # Levenberg-Marquardt is robust for small problems
        ftol=ftol,
        xtol=xtol,
        max_nfev=max_iter,
        x_scale=x_scale,
    )

    # Unscale solution
    x_sol_scaled = sol.x
    T_solution = x_sol_scaled[0] * T_scale + T_ref
    p_solution = x_sol_scaled[1] * p_scale + p_ref

    # Recompute residuals at solution to verify convergence
    F1_final, F2_final = fluid_residuals_scaled(x_sol_scaled)
    R1_final = F1_final * tol_dh0
    R2_final = F2_final * tol_dFA

    # Check convergence: both scaled residuals should be < 1.0
    converged = sol.success and abs(F1_final) < 1.0 and abs(F2_final) < 1.0

    if not converged:
        logger.debug(
            (
                "Fluid not conv after %d it: "
                "Residuals: |dh_t|=%.2e (want < %.2e), |d(p+G²/ρ)|=%.2e (want < %.2e) | "
                "State: (T_a=%.1f K, p_b=%.2e Pa) | Inputs: (G=%.1f kg/m²s, dh0=%.2e J/kg, tau_dA_over_A_c=%.2e)"
            ),
            sol.nfev,
            abs(R1_final),
            tol_dh0,
            abs(R2_final),
            tol_dFA,
            T_a,
            p_b,
            G,
            dh0,
            tau_dA_over_A_c,
        )

    return T_solution, p_solution
