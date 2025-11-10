# ruff: noqa: I001
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import root

from heat_exchanger.fluids.protocols import FluidModel


logger = logging.getLogger(__name__)


def energy_balance_segment(
    m_dot_hot: float, m_dot_cold: float, T_hot: float, T_cold: float, U: float, area: float
):
    """
    Compute change in stagnation enthalpy for hot and cold streams in a segment.

    Args:
        m_dot_hot: Hot stream mass flow rate (kg/s)
        m_dot_cold: Cold stream mass flow rate (kg/s)
        T_hot: Hot stream temperature (K)
        T_cold: Cold stream temperature (K)
        U: Overall heat transfer coefficient (W/m²·K)
        area: Heat transfer area (m²)

    Returns:
        tuple: (dh0_hot, dh0_cold, Q_segment)
        dh0 is change in stagnation enthalpy (J/kg)
        Q_segment is heat transferred (W)
    """
    # Heat transferred in this segment (positive if hot fluid loses heat)
    Q_segment = U * area * (T_hot - T_cold)

    # Change in stagnation enthalpy
    dh0_hot = -Q_segment / m_dot_hot if m_dot_hot != 0 else 0.0
    dh0_cold = Q_segment / m_dot_cold if m_dot_cold != 0 else 0.0

    return dh0_hot, dh0_cold, Q_segment


def momentum_balance_segment(
    G: float, rho_mean: float, D_h: float, length: float, f: float
) -> float:
    """
    Compute change in impulse function (F/A = p + G²/rho) over a segment.
    F/A_out - F/A_in = - tau_eff * P dx/A_cross_section = - tau_eff * 4dx/d_h
    Uses Fanning/Kays-London friction factor (includes mixing/wake pressure drop).

    Args:
        G: Mass velocity (kg/m²s)
        rho_mean: Mean density in segment (kg/m³)
        D_h: Hydraulic diameter (m)
        length: Segment length (m)
        f: Friction factor (-), Fanning/Kays-London definition

    Returns:
        float: Change in impulse function (Pa)
    """
    if rho_mean == 0 or D_h == 0:
        return 0.0
    return -f * (4 * length / D_h) * (G**2) / (2 * rho_mean)


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
    tol_T=1e-4,  # K tolerance for stagnation temperature change
    rel_tol_p=1e-3,  # % tolerance for pressure drop (strictly speaking of p + G^2/rho)
):
    r"""
    Solve simultaneously for T_not_a and p_not_b so that:
      1) Energy/stagnation enthalpy: (h_out + 0.5*(G^2/rho_out^2)) - (h_in + 0.5*(G^2/rho_in^2)) = dh0
      2) Momentum/impulse:           (p_out + G^2/rho_out) - (p_in + G^2/rho_in) = -tau_dA_over_A_c

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
    # Could improve guess by then using c_p(T_avg) to get T_guess
    if a_is_in:
        T_in = T_a
        cp_in = fluid.state(T_in, p_b).cp
        T_initial_guess = T_in + dh0 / cp_in if cp_in != 0 else T_in
        tol_dh0 = cp_in * tol_T
    else:
        T_out = T_a
        cp_out = fluid.state(T_out, p_b).cp
        T_initial_guess = T_out - dh0 / cp_out if cp_out != 0 else T_out
        tol_dh0 = cp_out * tol_T

    # For p_guess, a naive shift by dFA is typical (neglect density change)
    p_initial_guess = p_b - tau_dA_over_A_c if b_is_in else p_b + tau_dA_over_A_c

    tol_dFA = rel_tol_p / 100 * p_b

    # ------------------------------------------------------------
    # 2) Helper function: compute R1, R2 for a given guess of (T, p_unknown)
    # ------------------------------------------------------------
    def fluid_residuals(x: np.ndarray) -> np.ndarray:
        """
        Returns R1, R2 given the current guess of T, p
        """
        T_guess, p_guess = x
        # Guard against non-physical states that would break the fluid model.
        if T_guess <= 0 or p_guess <= 0:
            large_residual = 1e12
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

        # Residuals
        R1 = (h0_out - h0_in) - dh0
        R2 = (p_out_loc + G**2 / rho_out_loc) - (p_in_loc + G**2 / rho_in_loc) + tau_dA_over_A_c

        R2_scaled = R2 / tol_dFA * tol_dh0

        return np.array([R1, R2_scaled], dtype=float)

    # Note: Different methods have different convergence criteria:
    # - "hybr": checks if ||x_new - x_old|| < tol * (||x|| + tol) (NOT if ||F|| < tol directly!)
    # - "df-sane": minimizes max(|F_i|) (infinity norm) - better for mixed scales
    # - "lm": Levenberg-Marquardt - minimizes ||F||^2 with better convergence control
    #
    # Using hybr with increased iterations and tighter tolerance to handle scaling
    sol = root(
        fluid_residuals,
        np.array([T_initial_guess, p_initial_guess], dtype=float),
        method="hybr",
        tol=tol_dh0,
        options={"maxfev": max_iter},
    )

    x_sol = sol.x

    # Recompute residuals at solution to verify convergence
    # R1_final, R2_final_scaled = fluid_residuals(x_sol)
    R1_final, R2_final_scaled = sol.fun
    converged = sol.success and abs(R1_final) < tol_dh0 and abs(R2_final_scaled) < tol_dh0

    if not converged:
        logger.warning(
            (
                "Fluid step is not within desired tolerances: "
                "Individual residuals: |dh_t|=%.2e (want < %.2e), |d(p+G²/ρ)|=%.2e (want < %.2e) | "
                "State: (T_a=%.1f K, p_b=%.2e Pa) | "
            ),
            abs(R1_final),
            tol_dh0,
            abs(R2_final_scaled * tol_dFA / tol_dh0),
            tol_dFA,
            T_a,
            p_b,
        )

    T_solution = x_sol[0]
    p_solution = x_sol[1]

    return T_solution, p_solution
