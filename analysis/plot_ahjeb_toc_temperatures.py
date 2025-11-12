"""Plot Th and Tc arrays for the AHJE ToC case using compute_overall_performance.

This script:
  1) Loads the AHJE ToC case (CoolProp by default)
  2) Builds a 0D two-step initial guess
  3) Solves the shooting problem
  4) Runs compute_overall_performance
  5) Plots Th and Tc vs header index
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
from scipy.optimize import root

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting dependency
    raise SystemExit("matplotlib is required to run this plotting script. Please install it and try again.") from exc

from heat_exchanger.geometries.radial_spiral import (
    compute_overall_performance,
    rad_spiral_shoot,
)


def _import_analysis_helpers() -> object:
    """Import helper functions from analysis/comp_withZeli_involute_cases.py without requiring a package."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    analysis_dir = os.path.join(repo_root, "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    import comp_withZeli_involute_cases as helpers  # type: ignore[import-not-found]

    return helpers


def solve_shooting(case: str = "ahjeb_toc", fluid_model: str = "CoolProp"):
    """Return (geom, inputs, case_name, sol_x) where sol_x is the converged boundary vector."""
    helpers = _import_analysis_helpers()

    # Reduce noisy logs unless debugging
    logging.getLogger("heat_exchanger.conservation").setLevel(logging.WARNING)
    logging.getLogger("heat_exchanger.geometries.radial_spiral").setLevel(logging.INFO)

    geom, inputs, case_name = helpers.load_case(case, fluid_model)  # type: ignore[attr-defined]

    Th0, Ph0 = helpers._initial_guess_two_step_xflow(geom, inputs)  # type: ignore[attr-defined]

    tol_root = 1.0e-2
    Ph_in_known = inputs.Ph_in
    Ph_out_known = inputs.Ph_out
    tol_P_pct_of_Ph_in = 0.1
    tol_P = tol_P_pct_of_Ph_in / 100 * (Ph_in_known if Ph_in_known is not None else Ph_out_known)

    def _residuals(x: np.ndarray) -> np.ndarray:
        raw = rad_spiral_shoot(x, geom, inputs, property_solver_it_max=40, property_solver_tol_T=1e-2, rel_tol_p=1e-3)
        if raw.size == 2:
            return np.array([raw[0], raw[1] * (tol_root / tol_P)], dtype=float)  # type: ignore[arg-type]
        return np.array([raw[0]], dtype=float)

    x0 = np.array([Th0, Ph0], dtype=float) if Ph_in_known is not None else np.array([Th0], dtype=float)
    sol = root(_residuals, x0, method="hybr", tol=tol_root, options={"maxfev": 60})

    return geom, inputs, case_name, sol.x


def plot_temperatures(Th: np.ndarray, Tc: np.ndarray, title: str) -> None:
    """Plot hot and cold temperatures vs header index."""
    idx = np.arange(Th.size, dtype=int)
    plt.figure(figsize=(8, 5))
    plt.plot(idx, Th, label="Th (hot)", color="tab:red")
    plt.plot(idx, Tc, label="Tc (cold)", color="tab:blue")
    plt.xlabel("Header index (0 = cold inlet)")
    plt.ylabel("Temperature [K]")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    geom, inputs, case_name, sol_x = solve_shooting(case="ahjeb_toc", fluid_model="CoolProp")

    result = compute_overall_performance(
        sol_x, geom, inputs, property_solver_it_max=40, property_solver_tol_T=1e-2, rel_tol_p=1e-3
    )

    Th = result["Th"]  # type: ignore[index]
    Tc = result["Tc"]  # type: ignore[index]

    if not isinstance(Th, np.ndarray) or not isinstance(Tc, np.ndarray):
        raise RuntimeError("Unexpected result format: Th/Tc not found as numpy arrays.")

    plot_temperatures(Th, Tc, title=f"Temperature profiles for {case_name} (AHJE ToC)")


if __name__ == "__main__":
    main()
