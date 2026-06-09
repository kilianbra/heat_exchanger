"""Shared Brayton-cycle assumptions for conference PPT bar charts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CycleAssumptions:
    """Ideal-gas open/recuperated cycle inputs (air, neglect fuel addition)."""

    P_net_shaft_W: float = 700e3
    pressure_ratio: float = 9.0
    T_turb_inlet_K: float = 1500.0
    p0_bar: float = 1.0
    T0_K: float = 288.0
    eta_poly_comp: float = 0.88
    eta_poly_turb: float = 0.84
    cp_J_per_kgK: float = 1070.0
    gamma: float = 1.4
    # Fuel LHV only needed for fuel mass-flow / weight estimates, not dimensionless eta.
    LHV_J_per_kg: float = 12.0 * 3.6e6

    @property
    def k(self) -> float:
        return (self.gamma - 1.0) / self.gamma

    @property
    def p0_Pa(self) -> float:
        return self.p0_bar * 1e5

    @property
    def R_J_per_kgK(self) -> float:
        return self.cp_J_per_kgK * (self.gamma - 1.0) / self.gamma


@dataclass(frozen=True)
class RecuperatorInputs:
    """Recuperator effectiveness and fractional pressure drops (dp / inlet p)."""

    eps: float
    dp_hot_frac: float
    dp_cold_frac: float


# Default case used across conf PPT plots (matches fig9_w_cycle_model).
DEFAULT_CYCLE = CycleAssumptions()
