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


@dataclass(frozen=True)
class RecupHexGeometry:
    """HEx geometry ratios from fig9 coupled optimum (reference Ao/Ao_ref = 1, A/A_ref = 1)."""

    a_over_a_ref: float
    ao_over_ao_ref: float

    @property
    def ao_ref_over_ao(self) -> float:
        return 1.0 / self.ao_over_ao_ref


@dataclass(frozen=True)
class RecupPptCase:
    """One recuperated conf-PPT bar chart case."""

    stem: str
    label: str
    recup: RecuperatorInputs
    geom: RecupHexGeometry


def mach_ratio_first_order(mdot_kg_per_s: float, mdot_ref_kg_per_s: float, ao_ref_over_ao: float) -> float:
    """M/M_ref to first order, ignoring sqrt(T_hot_in/T_ref) from cycle coupling."""
    return (mdot_kg_per_s / mdot_ref_kg_per_s) * ao_ref_over_ao


# Default case used across conf PPT plots (matches fig9_w_cycle_model).
DEFAULT_CYCLE = CycleAssumptions()

# Recuperator cases for conf PPT bar charts (eps, dp_h/p_hi, dp_c/p_ci).
# REC_REF: fixed-BC reference from xflow at NTU_MATCH with f_c/f_h=1, M_h=0.1362
#   -> dp_h≈6.00%, dp_c≈4.11% (both pressure drops enter turbine exit = HEx hot inlet T).
REC_REF = RecuperatorInputs(eps=0.5966, dp_hot_frac=0.0600, dp_cold_frac=0.0411)
REC_FIX = RecuperatorInputs(eps=0.5376, dp_hot_frac=0.0191, dp_cold_frac=0.0131)
REC_GLOB = RecuperatorInputs(eps=0.6246, dp_hot_frac=0.0199, dp_cold_frac=0.0136)

# Geometry from fig9 red-line coupled optima (get_line_data).
GEOM_REF = RecupHexGeometry(a_over_a_ref=1.0, ao_over_ao_ref=1.0)
GEOM_FIX = RecupHexGeometry(a_over_a_ref=1.0, ao_over_ao_ref=1.5061)  # fixed mass, A/A_ref = 1
GEOM_GLOB = RecupHexGeometry(a_over_a_ref=1.5708, ao_over_ao_ref=1.7667)  # global aircraft-mass opt

RECUP_PPT_CASES: tuple[RecupPptCase, ...] = (
    RecupPptCase("rec_ref", "Reference", REC_REF, GEOM_REF),
    RecupPptCase("rec_fix", "Fixed mass opt", REC_FIX, GEOM_FIX),
    RecupPptCase("rec_glob", "Opt aircraft mass", REC_GLOB, GEOM_GLOB),
)

# Backward-compatible alias.
DEFAULT_RECUP = REC_REF
