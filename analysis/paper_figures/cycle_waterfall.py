"""
Dimensional practical-availability waterfall for cycle bar charts [kW].

All terms are computed in waterfall_from_solution(); bar order is set in
cycle_bar_plot.py.

Nomenclature
------------
WM_A       = mdot * cp * T * (1 - (p0/p)^k),   k = (gamma-1)/gamma
             Mechanical practical available work potential (euergy, not exergy).
T_se(T, p) = T * (p0/p)^k
             Isentropic exit temperature — T reached after reversible expansion to p0.
             Conserved along an isentropic process.

Per component:  dWM_A = mdot * dh_t - dQ0M_v
  mdot * dh_t  : first-law enthalpy change (heat transfer Q to fluid - extracted shaft work W)
  dQ0M_v       : creation of practical unavailable energy

Bar chart convention
--------------------
Heat-transfer terms (up bars) — increase in available work potential:
  dWM_qin  = Q_in * (1 - (p0/p_comb_in)^k)               combustor
  - dQ0M_qrec = Q_rec * ((p0/p_ho)^k - (p0/p_co)^k) >0   recuperator thermal gain = dWM_qrec > 0

Viscous terms (down bars) — plotted as -dQ0M_v (negative, i.e. losses):
  -dQ0M_v_rec  recuperator:
               = -Q_max * [1/eta_C * ((p0/p_ho)^k - (p0/p_hi)^k)
                           + (1-eta_C)/eta_C * ((p0/p_co)^k - (p0/p_ci)^k)]
               where eta_C = 1 - T_ci/T_hi  (Carnot efficiency of the inlet temperatures)
               Because both streams have zero net enthalpy flux: dWM_v_rec = -dQ0M_v_rec exactly.
  -dQ0M_v_comp compressor: - mdot*cp*(T_se(T2,p2) - T_se(T1,p1))
               = -mdot*cp*(T_se(T2,p2) - T0) = mdot*cp*(T0 - T_se(T2,p2)) = dWM_comp - W_comp
  -dQ0M_v_turb turbine:    - mdot*cp*(T_se(T4,p_to) - T_se(TIT,p_ti))
               = mdot*cp*(T_se(TIT,p_ti) - T_se(T4,p_to)) = dWM_turb + W_turb

  Note: dQ0M_v_turbomachinery (cycle budget) differs from mdot*(w_comp - w_comp_isen) because
  viscous heating at high pressure retains some work potential.  The isentropic
  work deviations are printed separately for reference.

Closure: dWM_qin - dQ0M_qrec - dQ0M_v_rec - dQ0M_v_comp - dQ0M_v_turb = P_shaft  (exact)
"""

from __future__ import annotations

from dataclasses import dataclass

from cycle_model import CycleSolution, isentropic_comp_work_J_per_kg, isentropic_turb_work_J_per_kg, t_se


@dataclass(frozen=True)
class CycleWaterfallBreakdown:
    """Practical-availability waterfall [kW]. All bar values use dWM convention."""

    Q_in_kw: float  # combustor heat rate [kW]
    dwm_qin_kw: float  # bar up:   combustor available energy gain
    dwm_qrec_kw: float  # bar up:   recuperator thermal gain (0 if open cycle)
    dwm_v_rec_kw: float  # bar down: -dQ0M_v_rec = dWM_v_rec (0 if open cycle)
    dwm_comp_kw: float  # bar down: -dQ0M_v_comp (cycle budget)
    dwm_turb_kw: float  # bar down: -dQ0M_v_turb (cycle budget)
    # Isentropic work deviations for reference (not plotted):
    phi_comp_isen_kw: float  # mdot * (w_comp - w_comp_isen)  > 0
    phi_turb_isen_kw: float  # mdot * (w_turb_isen - w_turb)  > 0
    # Cycle operating point:
    w_net_J_per_kg: float  # w_turb - w_comp  [J/kg]
    mdot_kg_per_s: float
    p_shaft_kw: float
    eta_cycle: float

    @property
    def recuperated(self) -> bool:
        return self.dwm_qrec_kw != 0.0 or self.dwm_v_rec_kw != 0.0

    @property
    def closure_residual_kw(self) -> float:
        """Sum of bar dWM steps minus P_shaft — should be exactly 0."""
        total = self.dwm_qin_kw + self.dwm_comp_kw + self.dwm_turb_kw
        if self.recuperated:
            total += self.dwm_qrec_kw + self.dwm_v_rec_kw
        return total - self.p_shaft_kw


def _combustor_terms_kw(sol: CycleSolution) -> tuple[float, float]:
    """Return (Q_in [kW], dWM_qin [kW])."""
    a = sol.assumptions
    t_comb_in = sol.combustor_inlet().T_K
    p_comb_in = sol.combustor_inlet().p_Pa
    q_in_w = sol.mdot_kg_per_s * a.cp_J_per_kgK * (a.T_turb_inlet_K - t_comb_in)
    dwm_qin_w = q_in_w * (1.0 - (a.p0_Pa / p_comb_in) ** a.k)
    return q_in_w / 1e3, dwm_qin_w / 1e3


def _hex_terms_kw(sol: CycleSolution) -> tuple[float, float]:
    """Return (dWM_qrec, dWM_v_rec) [kW].  dWM_v_rec = -dQ0M_v_rec (no net enthalpy)."""
    if sol.recuperator is None:
        return 0.0, 0.0

    a = sol.assumptions
    mdot = sol.mdot_kg_per_s
    k = a.k
    p0 = a.p0_Pa
    cp = a.cp_J_per_kgK

    t_ci = sol.hex_cold_inlet().T_K
    t_hi = sol.hex_hot_inlet().T_K
    p_ci = sol.hex_cold_inlet().p_Pa
    p_co = sol.hex_cold_outlet().p_Pa
    p_hi = sol.hex_hot_inlet().p_Pa
    p_ho = sol.hex_hot_outlet().p_Pa

    q_max_w = mdot * cp * (t_hi - t_ci)
    q_rec_w = sol.recuperator.eps * q_max_w

    dwm_qrec_w = q_rec_w * ((p0 / p_ho) ** k - (p0 / p_co) ** k)

    eta_c = 1.0 - t_ci / t_hi
    if eta_c <= 0.0:
        raise ValueError(f"eta_C = {eta_c:.4f} invalid (T_ci={t_ci:.1f} K, T_hi={t_hi:.1f} K)")

    hot_term = (p0 / p_ho) ** k - (p0 / p_hi) ** k
    cold_term = (p0 / p_co) ** k - (p0 / p_ci) ** k
    dwm_v_rec_w = -q_max_w * ((1.0 / eta_c) * hot_term + ((1.0 - eta_c) / eta_c) * cold_term)

    return dwm_qrec_w / 1e3, dwm_v_rec_w / 1e3


def _compressor_terms_kw(sol: CycleSolution) -> tuple[float, float]:
    """Return (dWM_comp, phi_comp_isen) [kW].

    dWM_comp = mdot*cp*(T0 - T_se(T2,p2)) = -dQ0M_v_comp (cycle-budget bar value)
    phi_comp_isen = mdot*(w_comp - w_comp_isen) (isentropic deviation, printed not plotted)
    """
    a = sol.assumptions
    T2 = sol.station("comp_out").T_K
    p2 = sol.station("comp_out").p_Pa
    dwm_comp_w = sol.mdot_kg_per_s * a.cp_J_per_kgK * (a.T0_K - t_se(T2, p2, a.p0_Pa, a.k))
    phi_isen_w = sol.mdot_kg_per_s * (sol.w_comp_J_per_kg - isentropic_comp_work_J_per_kg(a))
    return dwm_comp_w / 1e3, phi_isen_w / 1e3


def _turbine_terms_kw(sol: CycleSolution) -> tuple[float, float]:
    """Return (dWM_turb, phi_turb_isen) [kW].

    dWM_turb = mdot*cp*(T_se(TIT,p_ti) - T_se(T4,p_to)) = -dQ0M_v_turb (bar value)
    phi_turb_isen = mdot*(w_turb_isen - w_turb) (isentropic deviation, printed not plotted)
    """
    a = sol.assumptions
    tit = sol.combustor_outlet().T_K
    p_t_in = sol.combustor_outlet().p_Pa
    T4 = sol.station("turb_out").T_K
    p_t_out = sol.station("turb_out").p_Pa
    dwm_turb_w = (
        sol.mdot_kg_per_s * a.cp_J_per_kgK * (t_se(tit, p_t_in, a.p0_Pa, a.k) - t_se(T4, p_t_out, a.p0_Pa, a.k))
    )
    phi_isen_w = sol.mdot_kg_per_s * (isentropic_turb_work_J_per_kg(sol) - sol.w_turb_J_per_kg)
    return dwm_turb_w / 1e3, phi_isen_w / 1e3


def waterfall_from_solution(sol: CycleSolution) -> CycleWaterfallBreakdown:
    """Build dimensional waterfall breakdown for one solved cycle."""
    q_in_kw, dwm_qin_kw = _combustor_terms_kw(sol)
    dwm_qrec_kw, dwm_v_rec_kw = _hex_terms_kw(sol)
    dwm_comp_kw, phi_comp_isen_kw = _compressor_terms_kw(sol)
    dwm_turb_kw, phi_turb_isen_kw = _turbine_terms_kw(sol)

    return CycleWaterfallBreakdown(
        Q_in_kw=q_in_kw,
        dwm_qin_kw=dwm_qin_kw,
        dwm_qrec_kw=dwm_qrec_kw,
        dwm_v_rec_kw=dwm_v_rec_kw,
        dwm_comp_kw=dwm_comp_kw,
        dwm_turb_kw=dwm_turb_kw,
        phi_comp_isen_kw=phi_comp_isen_kw,
        phi_turb_isen_kw=phi_turb_isen_kw,
        w_net_J_per_kg=sol.w_net_J_per_kg,
        mdot_kg_per_s=sol.mdot_kg_per_s,
        p_shaft_kw=sol.mdot_kg_per_s * sol.w_net_J_per_kg / 1e3,
        eta_cycle=(sol.mdot_kg_per_s * sol.w_net_J_per_kg / 1e3) / q_in_kw if q_in_kw > 0 else 0.0,
    )


def print_waterfall(bd: CycleWaterfallBreakdown, *, title: str = "Cycle waterfall") -> None:
    """Print waterfall breakdown.

    Heat-transfer bars (dWM_q*) represent the increase in available work potential.
    Viscous bars (-dQ0M_v_*) represent the decrease in available work potential from
    irreversibilities.  For the recuperator dWM_v_rec = -dQ0M_v_rec exactly (no net
    enthalpy flux through the combined hot+cold streams).
    For comp/turb the bar value (dWM, T_se formula) differs from the isentropic work
    deviation mdot*(w - w_isen) because dissipation at high pressure retains some
    work potential.
    """
    print(title)
    print(f"  Q_in:                   {bd.Q_in_kw:8.1f} kW  (combustor heat rate)")
    print(f"  dWM_qin:                {bd.dwm_qin_kw:8.1f} kW  (bar up)")
    if bd.recuperated:
        print(f"  -dQ0M_qrec:             {bd.dwm_qrec_kw:8.1f} kW  (bar up,   = dWM_qrec)")
        print(f"  -dQ0M_v_rec:            {bd.dwm_v_rec_kw:8.1f} kW  (bar down, = dWM_v_rec exact)")
    print(f"  -dQ0M_v_comp:           {bd.dwm_comp_kw:8.1f} kW  (bar down, = dWM_comp - W_comp)")
    print(f"    mdot*(w_comp-w_c,is): {bd.phi_comp_isen_kw:8.1f} kW  (isentropic deviation, not plotted)")
    print(f"  -dQ0M_v_turb:           {bd.dwm_turb_kw:8.1f} kW  (bar down, = dWM_turb + W_turb)")
    print(f"    mdot*(w_t,is-w_turb): {bd.phi_turb_isen_kw:8.1f} kW  (isentropic deviation, not plotted)")
    print(f"  P_shaft:                {bd.p_shaft_kw:8.1f} kW  (bar: green total)")
    print(f"  closure:                {bd.closure_residual_kw:8.3f} kW")
    print(
        f"  w_net = {bd.w_net_J_per_kg / 1e3:.2f} kJ/kg,  mdot = {bd.mdot_kg_per_s:.4f} kg/s, eta_cycle = {bd.eta_cycle:.2%}"
    )
