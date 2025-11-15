from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.optimize import root


logger = logging.getLogger(__name__)


def get_Mach_from_mdot_A(
    mdot: float,
    A: float,
    T0: float,
    p0: float,
    c_p: float = 1005.0,
    gamma: float = 1.4,
    M_guess: float = 0.1,
) -> float:
    """
    Compute the subsonic Mach number from mass flow per unit area and stagnation
    (total) conditions for a perfect gas using a 1D isentropic relation.

    The equation is:
    mdot * np.sqrt(c_p * T0) / A / p0 = c * M * (1.0 + a * M**2) ** (-b)
    where c = γ / np.sqrt(γ - 1.0), a = (γ - 1.0) / 2.0, b = (γ + 1.0) / (2.0 * (γ - 1.0)).


    Parameters
    ----------
    mdot : float
        Mass flow rate [kg/s].
    A : float
        Frontal (flow) area [m^2].
    T0 : float
        Stagnation (total) temperature [K].
    p0 : float
        Stagnation (total) pressure [Pa].
    c_p : float, default 1005.0
        Specific heat at constant pressure [J/(kg·K)].
    gamma : float, default 1.4
        Ratio of specific heats (cp/cv).
    M_guess : float, default 0.1
        Initial guess for the Mach number (subsonic).

    Returns
    -------
    float
        Subsonic Mach number that satisfies the specified mass flux.
        Returns np.nan if the requested dimensionless mass flow exceeds
        the choked value.
    """
    if mdot <= 0.0 or A <= 0.0:
        warnings.warn("Non-positive mass flow or area; returning Mach = 0.0.", RuntimeWarning)
        return 0.0

    if T0 <= 0.0 or p0 <= 0.0:
        raise ValueError("Stagnation temperature and pressure must be positive.")

    # Mass flux and dimensionless mass-flow parameter y
    y = mdot * np.sqrt(c_p * T0) / A / p0

    # Maximum (choked) dimensionless mass-flow parameter at M = 1
    M_star = 1.0
    a = (gamma - 1.0) / 2.0
    b = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    c = gamma / np.sqrt(gamma - 1.0)
    y_star = c * M_star * (1.0 + a * M_star**2) ** (-b)

    if y > y_star * (1.0 + 1e-8):
        logger.warning(
            "get_Mach_from_mdot_A: y=%.6g exceeds choked value y*=%.6g (mdot=%.6g, A=%.6g, T0=%.6g, p0=%.6g)",
            y,
            y_star,
            mdot,
            A,
            T0,
            p0,
        )
        return float("nan")

    def residual(M_arr: np.ndarray) -> np.ndarray:
        """Residual for scipy.optimize.root: f(M) - y = 0."""
        M = M_arr[0]

        if M <= 0.0:
            # Strongly penalize non-physical negative/zero Mach.
            return np.array([y], dtype=float)

        f_M = c * M * (1.0 + a * M**2) ** (-b)
        return np.array([f_M - y], dtype=float)

    sol = root(
        residual,
        np.array([M_guess], dtype=float),
        method="hybr",
    )

    M_sol = float(sol.x[0])

    if not sol.success:
        logger.warning(
            "get_Mach_from_mdot_A did not fully converge: success=%s, message=%s, M_sol=%.6g",
            sol.success,
            sol.message,
            M_sol,
        )

    return M_sol
