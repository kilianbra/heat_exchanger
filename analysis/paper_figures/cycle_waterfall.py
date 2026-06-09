"""Practical-availability waterfall breakdown for cycle bar charts (normalized by Q_fuel)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cycle_model import CycleSolution, ideal_comp_work_J_per_kg, ideal_turb_work_J_per_kg
from xflow import practical_unavailable_creation_hex


@dataclass(frozen=True)
class CycleWaterfallBreakdown:
    """
    Dimensionless shaft-work / practical-availability waterfall vs Q_fuel.

    Signed visc_* terms are losses (negative). thermal_* are gains (positive).
    Closure: thermal_fuel + thermal_hex + visc_hex + visc_comp + visc_turb = total.
    """

    thermal_fuel: float
    thermal_hex: float
    visc_hex_signed: float
    visc_comp_signed: float
    visc_turb_signed: float
    total: float

    @property
    def visc_rest_signed(self) -> float:
        """Lumped comp + turb + any closure residual (negative = loss)."""
        return self.total - self.thermal_fuel - self.thermal_hex - self.visc_hex_signed

    @property
    def visc_rest(self) -> float:
        return abs(self.visc_rest_signed)

    @property
    def visc_hex(self) -> float:
        return abs(self.visc_hex_signed)


def _thermal_fuel_fraction(sol: CycleSolution) -> float:
    """mdot*cp*deltaT*(1-(p0/p_comb_in)^k) / Q_fuel with Q_fuel = mdot*cp*deltaT."""
    a = sol.assumptions
    p_comb_in = sol.combustor_inlet().p_Pa
    return 1.0 - (a.p0_Pa / p_comb_in) ** a.k


def _visc_comp_signed(sol: CycleSolution) -> float:
    """Compressor irreversibility vs Q_fuel (negative)."""
    w_ideal = ideal_comp_work_J_per_kg(sol.assumptions)
    delta_w = sol.w_comp_J_per_kg - w_ideal
    return -(delta_w / sol.q_fuel_J_per_kg)


def _visc_turb_signed(sol: CycleSolution) -> float:
    """Turbine irreversibility vs Q_fuel (negative)."""
    w_ideal = ideal_turb_work_J_per_kg(sol)
    delta_w = w_ideal - sol.w_turb_J_per_kg
    return -(delta_w / sol.q_fuel_J_per_kg)


def _hex_terms_from_xflow(sol: CycleSolution) -> tuple[float, float]:
    """
    Return (thermal_hex, visc_hex_signed) normalized by Q_fuel.

    Uses practical_unavailable_creation_hex at cycle-coupled hex boundary conditions.
    """
    if sol.recuperator is None:
        return 0.0, 0.0

    a = sol.assumptions
    recup = sol.recuperator
    cold_in = sol.hex_cold_inlet()
    cold_out = sol.hex_cold_outlet()
    hot_in = sol.hex_hot_inlet()
    hot_out = sol.hex_hot_outlet()

    q_hex_w = sol.mdot_kg_per_s * a.cp_J_per_kgK * (hot_in.T_K - hot_out.T_K)
    pressure_term = (a.p0_Pa / hot_out.p_Pa) ** a.k - (a.p0_Pa / cold_out.p_Pa) ** a.k

    t_act = hot_in.T_K / cold_in.T_K
    pu = practical_unavailable_creation_hex(
        np.array([recup.eps]),
        t_act,
        np.array([recup.dp_hot_frac]),
        np.array([recup.dp_cold_frac]),
        np.array([True]),
        p_cold_in_over_p_hot_in=cold_in.p_Pa / hot_in.p_Pa,
        p_dead_over_p_hot_in=a.p0_Pa / hot_in.p_Pa,
        gamma=a.gamma,
    )
    q_max_w = sol.mdot_kg_per_s * a.cp_J_per_kgK * (hot_in.T_K - cold_in.T_K)
    dq_o_m_w = -float(np.asarray(pu, dtype=float).flat[0]) * q_max_w

    q_fuel_w = sol.mdot_kg_per_s * sol.q_fuel_J_per_kg
    thermal_hex = pressure_term * q_hex_w / q_fuel_w
    visc_hex_signed = (dq_o_m_w - pressure_term * q_hex_w) / q_fuel_w
    return thermal_hex, visc_hex_signed


def open_cycle_waterfall(sol: CycleSolution) -> CycleWaterfallBreakdown:
    """No recuperator: fuel thermal gain + lumped viscous losses = eta."""
    thermal_fuel = _thermal_fuel_fraction(sol)
    total = sol.eta_cycle
    visc_comp = _visc_comp_signed(sol)
    visc_turb = _visc_turb_signed(sol)
    return CycleWaterfallBreakdown(
        thermal_fuel=thermal_fuel,
        thermal_hex=0.0,
        visc_hex_signed=0.0,
        visc_comp_signed=visc_comp,
        visc_turb_signed=visc_turb,
        total=total,
    )


def recuperated_cycle_waterfall(sol: CycleSolution) -> CycleWaterfallBreakdown:
    """Full recuperated breakdown including hex thermal/viscous terms."""
    thermal_fuel = _thermal_fuel_fraction(sol)
    thermal_hex, visc_hex_signed = _hex_terms_from_xflow(sol)
    visc_comp = _visc_comp_signed(sol)
    visc_turb = _visc_turb_signed(sol)
    total = sol.eta_cycle
    return CycleWaterfallBreakdown(
        thermal_fuel=thermal_fuel,
        thermal_hex=thermal_hex,
        visc_hex_signed=visc_hex_signed,
        visc_comp_signed=visc_comp,
        visc_turb_signed=visc_turb,
        total=total,
    )


def waterfall_from_solution(sol: CycleSolution) -> CycleWaterfallBreakdown:
    if sol.recuperator is None:
        return open_cycle_waterfall(sol)
    return recuperated_cycle_waterfall(sol)
