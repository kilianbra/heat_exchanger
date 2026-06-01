"""Print thermophysical properties for a fluid at given T, P states.

Uses ``heat_exchanger.fluid_properties`` (CoolProp / REFPROP backends), similar to
the phase-diagram scripts but for quick tabulated lookups.

Examples:
    uv run analysis/standalone_tools/print_fluid_prop.py --fluid ParaHydrogen --T 20 77 --P-bar 1 3
    uv run analysis/standalone_tools/print_fluid_prop.py --fluid Air --T 288.15 --P-bar 1 --backend refprop
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
from tabulate import tabulate

from heat_exchanger.fluid_properties import (
    CoolPropProperties,
    RefPropProperties,
    configure_refprop,
)
from heat_exchanger.logging_utils import configure_logging

BAR_TO_PA = 1e5
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FluidState:
    t_k: float
    p_pa: float

    @property
    def p_bar(self) -> float:
        return self.p_pa / BAR_TO_PA


def _build_strategy(fluid: str, backend: str):
    backend = backend.lower()
    if backend == "coolprop":
        return CoolPropProperties(fluid)
    if backend == "refprop":
        return RefPropProperties(fluid)
    if backend == "auto":
        try:
            return RefPropProperties(fluid)
        except Exception:
            return CoolPropProperties(fluid)
    raise ValueError(f"Unknown backend '{backend}' (use auto, coolprop, or refprop).")


def _gamma(strategy, t_k: float, p_pa: float) -> float:
    cp = strategy.get_cp(t_k, p_pa)
    if hasattr(strategy, "R_specific"):
        cv = cp - strategy.R_specific
        return cp / cv if cv > 0 else float("nan")
    if hasattr(strategy, "CP") and hasattr(strategy, "fluid"):
        cv = strategy.CP.PropsSI("Cvmass", "T", t_k, "P", p_pa, strategy.fluid)
        return cp / cv if cv > 0 else float("nan")
    if hasattr(strategy, "mixture_state") and strategy.mixture_state is not None:
        strategy.mixture_state.update(strategy.mixture.CP.PT_INPUTS, p_pa, t_k)
        cv = strategy.mixture_state.cvmass()
        return cp / cv if cv > 0 else float("nan")
    return float("nan")


def query_row(strategy, state: FluidState) -> dict:
    t_k, p_pa = state.t_k, state.p_pa
    rho, cp, mu, k = strategy.get_transport_properties(t_k, p_pa)
    pr = cp * mu / k if k else float("nan")
    return {
        "T [K]": t_k,
        "P [bar]": state.p_bar,
        "rho [kg/m^3]": rho,
        "cp [J/kg-K]": cp,
        "mu [Pa·s]": mu,
        "k [W/m-K]": k,
        "Pr [-]": pr,
        "gamma [-]": _gamma(strategy, t_k, p_pa),
        "h [J/kg]": strategy.get_specific_enthalpy(t_k, p_pa),
        "s [J/kg-K]": strategy.get_specific_entropy(t_k, p_pa),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print fluid properties at given T, P.")
    p.add_argument("--fluid", default="ParaHydrogen", help="CoolProp / REFPROP fluid name.")
    p.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "coolprop", "refprop"),
        help="Property backend (default: try REFPROP, then CoolProp HEOS).",
    )
    p.add_argument(
        "--T",
        type=float,
        nargs="+",
        default=[20.0],
        help="Temperature(s) [K]. Pair with --P-bar if multiple pressures.",
    )
    p.add_argument(
        "--P-bar",
        type=float,
        nargs="+",
        default=[1.0],
        help="Pressure(s) [bar].",
    )
    return p.parse_args()


def main() -> None:
    configure_logging(logging.INFO)
    configure_refprop()
    args = parse_args()

    strategy = _build_strategy(args.fluid, args.backend)
    states = [FluidState(t_k=t, p_pa=p * BAR_TO_PA) for t in args.T for p in args.P_bar]

    rows = [query_row(strategy, st) for st in states]
    print(f"\n{args.fluid} ({type(strategy).__name__})")
    print(
        tabulate(
            rows,
            headers="keys",
            tablefmt="github",
            floatfmt=".6g",
        )
    )


if __name__ == "__main__":
    main()
