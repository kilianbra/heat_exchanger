"""Plot Th and Tc arrays for the AHJE ToC case using spiral_hex_solver.

This script:
  1) Loads the AHJE ToC case (CoolProp by default)
  2) Solves using spiral_hex_solver (1D marching)
  3) Plots Th and Tc vs sector index
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting dependency
    raise SystemExit("matplotlib is required to run this plotting script. Please install it and try again.") from exc

from spiral_recup_cases import load_case  # type: ignore[import-not-found]

from heat_exchanger.geometries.radial_spiral import spiral_hex_solver
from heat_exchanger.logging_utils import configure_logging


def plot_temperatures(Th: np.ndarray, Tc: np.ndarray, title: str) -> None:
    """Plot hot and cold temperatures vs sector index (0 = cold inlet)."""
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
    # Configure logging
    configure_logging(logging.INFO)
    logging.getLogger("heat_exchanger.conservation").setLevel(logging.WARNING)
    logging.getLogger("heat_exchanger.geometries.radial_spiral").setLevel(logging.WARNING)

    # Load case and solve
    geom, inputs, case_name = load_case(case="ahjeb_toc", fluid_model="CoolProp")
    result = spiral_hex_solver(geom, inputs, method="1d")

    Th = result["Th"]  # type: ignore[index]
    Tc = result["Tc"]  # type: ignore[index]

    if not isinstance(Th, np.ndarray) or not isinstance(Tc, np.ndarray):
        raise RuntimeError("Unexpected result format: Th/Tc not found as numpy arrays.")

    plot_temperatures(Th, Tc, title=f"Temperature profiles for {case_name} (AHJE ToC)")


if __name__ == "__main__":
    main()
