"""
Bridge between geometric HEx sweeps (fig6/fig7) and the simple cycle model.

Future workflow
---------------
1. Sweep NTU (and Ao/Ao_ref) via xflow → eps, dp_h, dp_c at reference geometry.
2. Pass RecuperatorInputs(eps, dp_h, dp_c) into solve_recuperated_cycle().
3. Build CycleWaterfallBreakdown via waterfall_from_solution() for bar charts.

The fig9 coupling (mdot solve at constant P_shaft with varying M_in) can be added
here when bar charts need fully coupled boundary conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cycle_assumptions import CycleAssumptions, RecuperatorInputs
from cycle_model import CycleSolution, solve_recuperated_cycle
from cycle_waterfall import CycleWaterfallBreakdown, waterfall_from_solution


@dataclass(frozen=True)
class HexSweepPoint:
    """One point from a length or diffusion sweep."""

    ntu: float
    eps: float
    dp_hot_frac: float
    dp_cold_frac: float
    ao_over_ao_ref: float | None = None
    a_over_a_ref: float | None = None


@dataclass(frozen=True)
class CoupledCyclePoint:
    """Recuperated cycle + waterfall at one HEx sweep point."""

    hex_point: HexSweepPoint
    cycle: CycleSolution
    waterfall: CycleWaterfallBreakdown


def evaluate_at_hex_point(
    point: HexSweepPoint,
    assumptions: CycleAssumptions | None = None,
) -> CoupledCyclePoint | None:
    """Solve recuperated cycle and waterfall for one (eps, dp) point."""
    recup = RecuperatorInputs(
        eps=point.eps,
        dp_hot_frac=point.dp_hot_frac,
        dp_cold_frac=point.dp_cold_frac,
    )
    sol = solve_recuperated_cycle(recup, assumptions)
    if sol is None:
        return None
    return CoupledCyclePoint(
        hex_point=point,
        cycle=sol,
        waterfall=waterfall_from_solution(sol),
    )


def evaluate_hex_sweep(
    points: list[HexSweepPoint],
    assumptions: CycleAssumptions | None = None,
) -> list[CoupledCyclePoint]:
    """Evaluate a list of sweep points; skip invalid cycle solutions."""
    out: list[CoupledCyclePoint] = []
    for pt in points:
        coupled = evaluate_at_hex_point(pt, assumptions)
        if coupled is not None:
            out.append(coupled)
    return out


def hex_point_from_arrays(
    ntu_v: np.ndarray,
    eps_v: np.ndarray,
    dp_h_v: np.ndarray,
    dp_c_v: np.ndarray,
    index: int,
    *,
    ao_over_ao_ref: float | None = None,
    a_over_a_ref: float | None = None,
) -> HexSweepPoint:
    """Build HexSweepPoint from fig6/fig7 sweep arrays at one index."""
    return HexSweepPoint(
        ntu=float(ntu_v[index]),
        eps=float(eps_v[index]),
        dp_hot_frac=float(dp_h_v[index]),
        dp_cold_frac=float(dp_c_v[index]),
        ao_over_ao_ref=ao_over_ao_ref,
        a_over_a_ref=a_over_a_ref,
    )
