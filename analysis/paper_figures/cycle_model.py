"""Simple Brayton cycle with optional recuperator (constant cp, gamma)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cycle_assumptions import CycleAssumptions, RecuperatorInputs


@dataclass(frozen=True)
class CycleStation:
    """Thermodynamic state at one station."""

    label: str
    T_K: float
    p_Pa: float


@dataclass(frozen=True)
class CycleSolution:
    """Solved cycle with stations and work/heat splits."""

    assumptions: CycleAssumptions
    stations: tuple[CycleStation, ...]
    recuperator: RecuperatorInputs | None
    w_comp_J_per_kg: float
    w_turb_J_per_kg: float
    w_net_J_per_kg: float
    q_fuel_J_per_kg: float
    eta_cycle: float  # w_net / q_fuel
    mdot_kg_per_s: float

    @property
    def P_shaft_W(self) -> float:
        return self.assumptions.P_net_shaft_W

    def station(self, label: str) -> CycleStation:
        for st in self.stations:
            if st.label == label:
                return st
        raise KeyError(f"unknown station {label!r}")

    def hex_cold_inlet(self) -> CycleStation:
        return self.station("comp_out")

    def hex_cold_outlet(self) -> CycleStation:
        return self.station("recup_cold_out")

    def hex_hot_inlet(self) -> CycleStation:
        return self.station("turb_out")

    def hex_hot_outlet(self) -> CycleStation:
        return self.station("recup_hot_out")

    def combustor_inlet(self) -> CycleStation:
        if self.recuperator is None:
            return self.station("comp_out")
        return self.station("recup_cold_out")

    def combustor_outlet(self) -> CycleStation:
        return self.station("comb_out")


def _isentropic_T_ratio(p_ratio: float, gamma: float) -> float:
    return p_ratio ** ((gamma - 1.0) / gamma)


def solve_open_cycle(assumptions: CycleAssumptions | None = None) -> CycleSolution:
    """Unrecuperated Brayton cycle: ambient -> comp -> combustor -> turb -> ambient."""
    a = assumptions or CycleAssumptions()
    cp = a.cp_J_per_kgK
    g = a.gamma
    p0 = a.p0_Pa
    T0 = a.T0_K
    pr = a.pressure_ratio
    tit = a.T_turb_inlet_K

    p_comp_out = p0 * pr
    T_comp_out = T0 * (pr ** ((g - 1.0) / (g * a.eta_poly_comp)))
    # Polytropic turbine: same exponent convention as compressor (k/eta), not fig9's k*eta.
    T_turb_out = tit * ((p0 / p_comp_out) ** ((g - 1.0) / (g * a.eta_poly_turb)))

    w_comp = cp * (T_comp_out - T0)
    w_turb = cp * (tit - T_turb_out)
    w_net = w_turb - w_comp
    q_fuel = cp * (tit - T_comp_out)
    eta = w_net / q_fuel
    mdot = a.P_net_shaft_W / w_net

    stations = (
        CycleStation("ambient", T0, p0),
        CycleStation("comp_out", T_comp_out, p_comp_out),
        CycleStation("comb_out", tit, p_comp_out),
        CycleStation("turb_out", T_turb_out, p0),
    )
    return CycleSolution(
        assumptions=a,
        stations=stations,
        recuperator=None,
        w_comp_J_per_kg=w_comp,
        w_turb_J_per_kg=w_turb,
        w_net_J_per_kg=w_net,
        q_fuel_J_per_kg=q_fuel,
        eta_cycle=eta,
        mdot_kg_per_s=mdot,
    )


def solve_recuperated_cycle(
    recup: RecuperatorInputs,
    assumptions: CycleAssumptions | None = None,
) -> CycleSolution | None:
    """
    Recuperated cycle with specified effectiveness and dp fractions.

    Station order: ambient, comp_out, recup_cold_out, comb_out, turb_out, recup_hot_out.
    Returns None if net work is non-positive.
    """
    a = assumptions or CycleAssumptions()
    cp = a.cp_J_per_kgK
    g = a.gamma
    p0 = a.p0_Pa
    T0 = a.T0_K
    pr = a.pressure_ratio
    tit = a.T_turb_inlet_K

    p_comp_out = p0 * pr
    T_comp_out = T0 * (pr ** ((g - 1.0) / (g * a.eta_poly_comp)))

    p_recup_cold_out = p_comp_out * (1.0 - recup.dp_cold_frac)
    p_turb_out = p0 / (1.0 - recup.dp_hot_frac)
    T_turb_out = tit * ((p_turb_out / p_recup_cold_out) ** ((g - 1.0) / (g * a.eta_poly_turb)))

    max_temp_rise = max(T_turb_out - T_comp_out, 0.0)
    T_recup_cold_out = T_comp_out + recup.eps * max_temp_rise
    T_recup_hot_out = T_turb_out - recup.eps * max_temp_rise
    p_recup_hot_out = p_turb_out * (1.0 - recup.dp_hot_frac)

    w_comp = cp * (T_comp_out - T0)
    w_turb = cp * (tit - T_turb_out)
    w_net = w_turb - w_comp
    q_fuel = cp * (tit - T_recup_cold_out)
    if w_net <= 0.0 or q_fuel <= 0.0 or not np.isfinite(w_net):
        return None

    eta = w_net / q_fuel
    mdot = a.P_net_shaft_W / w_net

    stations = (
        CycleStation("ambient", T0, p0),
        CycleStation("comp_out", T_comp_out, p_comp_out),
        CycleStation("recup_cold_out", T_recup_cold_out, p_recup_cold_out),
        CycleStation("comb_out", tit, p_recup_cold_out),
        CycleStation("turb_out", T_turb_out, p_turb_out),
        CycleStation("recup_hot_out", T_recup_hot_out, p_recup_hot_out),
    )
    return CycleSolution(
        assumptions=a,
        stations=stations,
        recuperator=recup,
        w_comp_J_per_kg=w_comp,
        w_turb_J_per_kg=w_turb,
        w_net_J_per_kg=w_net,
        q_fuel_J_per_kg=q_fuel,
        eta_cycle=eta,
        mdot_kg_per_s=mdot,
    )


def ideal_comp_work_J_per_kg(a: CycleAssumptions) -> float:
    T2s = a.T0_K * (a.pressure_ratio ** a.k)
    return a.cp_J_per_kgK * (T2s - a.T0_K)


def ideal_turb_work_J_per_kg(sol: CycleSolution) -> float:
    a = sol.assumptions
    p_in = sol.combustor_outlet().p_Pa
    tit = sol.combustor_outlet().T_K
    p_out = sol.station("turb_out").p_Pa
    T4s = tit * _isentropic_T_ratio(p_out / p_in, a.gamma)
    return a.cp_J_per_kgK * (tit - T4s)
