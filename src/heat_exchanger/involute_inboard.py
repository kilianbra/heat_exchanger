"""Involute heat-exchanger shooting residuals for the inboard orientation.

This module reworks the legacy ``analysis/involute_hx_study.py`` solver so that the
boundary-value problem is exposed as a residual function suitable for SciPy's root
finders.  The goal is to factor geometry-specific logic (involute layout) away from
the generic marching / property evaluation code and to rely on the new ``FluidModel``
protocol for thermophysical properties.

The ``F_inboard`` function provided here computes the residuals (temperature,
pressure) at the known hot-side boundary when integrating outward from an inboard
guess.  This is the "worst case" shooting configuration where two thermodynamic
unknowns must be guessed on the marching side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from heat_exchanger.conservation import update_static_properties
from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    tube_bank_nusselt_number_and_friction_factor,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu
from heat_exchanger.fluids.protocols import FluidModel

RAD_PER_DEG = np.pi / 180.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RadialInvoluteGeometry:
    """Primary involute geometry inputs using the renamed conventions."""

    tube_outer_diam: float
    tube_thick: float
    tube_spacing_trv: float  # non-dimensional spacing ratio (Xt*)
    tube_spacing_long: float  # non-dimensional spacing ratio (Xl*)
    staggered: bool
    n_headers: int
    n_rows_per_header: int
    n_rows_axial: int
    radius_outer_whole_hex: float
    inv_angle_deg: float = 360.0

    def tube_inner_diam(self) -> float:
        return self.tube_outer_diam - 2.0 * self.tube_thick

    def radius_inner_whole_hex(self) -> float:
        n_rows_per_axial_section = self.n_rows_per_header * self.n_headers
        outer_radius_span = n_rows_per_axial_section * self.tube_spacing_long * self.tube_outer_diam
        return self.radius_outer_whole_hex - outer_radius_span

    def axial_length(self) -> float:
        return self.n_rows_axial * self.tube_spacing_trv * self.tube_outer_diam

    def frontal_area_outer(self) -> float:
        return (
            2.0
            * np.pi
            * self.radius_outer_whole_hex
            * self.n_rows_axial
            * self.tube_outer_diam
            * self.tube_spacing_trv
        )


@dataclass(frozen=True)
class MarchingOptions:
    """Tuning knobs and numerical tolerances for the marching solver."""

    heat_transfer_tuning: float = 1.0
    pressure_drop_tuning: float = 1.0
    property_solver_iterations: int = 20
    # ruff: noqa: N815 (allow mixed case variables)
    property_solver_tol_T: float = 1e-2
    property_solver_rel_tol_p: float = 1e-3


@dataclass(frozen=True)
class _GeometryCache:  # For now this is involute specific, later make it more general
    """Pre-computed geometry arrays for the involute layout."""

    radii: np.ndarray
    theta: np.ndarray
    tube_length: np.ndarray
    area_ht_hot: np.ndarray
    area_ht_cold: np.ndarray
    area_frontal_hot: np.ndarray
    area_free_hot: np.ndarray
    area_free_cold: np.ndarray
    d_h_hot: np.ndarray
    # ruff: noqa: N815 (allow mixed case variables)
    dR: float


def _compute_geometry_arrays(geom: RadialInvoluteGeometry) -> _GeometryCache:
    """Compute the geometry arrays for the involute layout.
    Each array calculates the heat transfer area (area_ht) or free flow area (area_free)
    for one sector of the hot and cold fluids."""
    tube_inner_diam = geom.tube_inner_diam()
    radius_inner = geom.radius_inner_whole_hex()
    radius_outer = geom.radius_outer_whole_hex
    if radius_inner <= 0 or radius_inner >= radius_outer:
        raise ValueError(
            "Invalid geometry: inner radius must be positive and less than outer radius"
        )

    spacing_trv = geom.tube_spacing_trv * geom.tube_outer_diam
    spacing_long = geom.tube_spacing_long * geom.tube_outer_diam
    axial_length = geom.axial_length()
    # n_rows_total = geom.n_rows_per_header * geom.n_headers
    n_tubes_total = geom.n_rows_axial * geom.n_headers * geom.n_rows_per_header
    n_layers = geom.n_headers
    dR = (radius_outer - radius_inner) / n_layers

    radii = radius_inner + np.arange(n_layers + 1, dtype=float) * dR
    inv_b = (radius_outer - radius_inner) / np.deg2rad(geom.inv_angle_deg)
    theta = (radii - radius_inner) / inv_b

    tubes_per_layer = geom.n_rows_per_header * geom.n_rows_axial
    area_ht_hot = np.zeros(n_layers)
    area_ht_cold = np.zeros(n_layers)
    area_frontal_hot = np.zeros(n_layers)
    area_free_hot = np.zeros(n_layers)
    area_free_cold = np.zeros(n_layers)
    tube_length = np.zeros(n_layers)
    d_h_hot = np.zeros(n_layers)

    # Free-area ratios (global) depend only on spacing
    sigma_hot = (spacing_trv - geom.tube_outer_diam) / spacing_trv
    if geom.staggered:
        diag_spacing = np.sqrt(spacing_long**2 + (0.5 * spacing_trv) ** 2)
        sigma_hot = min(sigma_hot, 2.0 * (diag_spacing - geom.tube_outer_diam) / spacing_trv)
    # sigma_cold = np.pi * tube_inner_diam**2 / (4.0 * spacing_trv * spacing_long)

    Lf = geom.n_rows_per_header * spacing_long

    for j in range(n_layers):
        tube_length[j] = np.trapezoid(
            np.sqrt(radii[j : j + 2] ** 2 + inv_b**2),
            theta[j : j + 2],
        )
        area_frontal_hot[j] = axial_length * 2.0 * np.pi * radii[j] / geom.n_headers

        area_ht_hot[j] = np.pi * geom.tube_outer_diam * tube_length[j] * tubes_per_layer
        area_ht_cold[j] = np.pi * tube_inner_diam * tube_length[j] * tubes_per_layer

        area_free_hot[j] = area_frontal_hot[j] * sigma_hot
        # The equation below is only valid for this segmentation of the HEx
        area_free_cold[j] = np.pi * tube_inner_diam**2 / 4.0 * n_tubes_total / geom.n_headers
        d_h_hot[j] = (4.0 * area_free_hot[j] * Lf) / area_ht_hot[j]

    return _GeometryCache(
        radii=radii,
        theta=theta,
        tube_length=tube_length,
        area_ht_hot=area_ht_hot,
        area_ht_cold=area_ht_cold,
        area_frontal_hot=area_frontal_hot,
        area_free_hot=area_free_hot,
        area_free_cold=area_free_cold,
        d_h_hot=d_h_hot,
        dR=dR,
    )


def _compute_overall_performance(
    diagnostics: dict[str, float],
    Th: np.ndarray,
    Ph: np.ndarray,
    Tc: np.ndarray,
    Pc: np.ndarray,
    Th_in: float,
    Ph_known: float,
    Tc_in: float,
    Pc_in: float,
    fluid_hot: FluidModel,
    fluid_cold: FluidModel,
    mdot_h_total: float,
    mdot_c_total: float,
    geom_cache: _GeometryCache,
    n_headers: int,
    geometry: RadialInvoluteGeometry,
    opts: MarchingOptions,
    outlet_pressure_known: bool = False,
) -> None:
    """Calculate overall heat exchanger performance metrics.

    Adds the following keys to diagnostics:
    - Q_total: Total heat transfer rate (W)
    - epsilon: Overall effectiveness (-)
    - NTU: Overall number of transfer units (-)
    - dP_hot: Hot side pressure drop (Pa)
    - dP_hot_pct: Hot side pressure drop as % of inlet
    - dP_cold: Cold side pressure drop (Pa)
    - dP_cold_pct: Cold side pressure drop as % of inlet
    - Th_out: Hot side outlet temperature (K)
    - Tc_out: Cold side outlet temperature (K)
    - Re_h_in/out: Hot side Reynolds numbers based on OD at inlet/outlet (-)
    - St_h_in/out: Hot side Stanton numbers at inlet/outlet (-)
    - f_h_in/out: Hot side friction factors (tube bank) at inlet/outlet (-)
    - Ec_h_in/out: Hot side Eckert numbers at inlet/outlet (-)
    """
    # Hot side: flows from outer (index -1) to inner (index 0)
    # Cold side: flows from inner (index 0) to outer (index -1)
    Th_out = Th[0]
    Tc_out = Tc[-1]
    Pc_out = Pc[-1]

    if outlet_pressure_known:
        Ph_out = Ph_known
        Ph_in = Ph[-1]
    else:
        Ph_in = Ph_known
        Ph_out = Ph[0]

    # Get states at inlet and outlet for both fluids
    state_h_in = fluid_hot.state(Th_in, Ph_in)
    state_h_out = fluid_hot.state(Th_out, Ph_out)
    state_c_in = fluid_cold.state(Tc_in, Pc_in)
    state_c_out = fluid_cold.state(Tc_out, Pc_out)

    area_free_hot_in = geom_cache.area_free_hot[-1] * n_headers
    area_free_hot_out = geom_cache.area_free_hot[0] * n_headers
    area_free_cold = geom_cache.area_free_cold[0] * n_headers

    G_h_in = mdot_h_total / area_free_hot_in
    G_h_out = mdot_h_total / area_free_hot_out
    G_c = mdot_c_total / area_free_cold

    h_stag_in_hot = state_h_in.h + 0.5 * (G_h_in / state_h_in.rho) ** 2
    h_stag_out_hot = state_h_out.h + 0.5 * (G_h_out / state_h_out.rho) ** 2
    h_stag_in_cold = state_c_in.h + 0.5 * (G_c / state_c_in.rho) ** 2
    h_stag_out_cold = state_c_out.h + 0.5 * (G_c / state_c_out.rho) ** 2

    # Calculate heat transfer rates from enthalpy changes
    Q_hot = mdot_h_total * (h_stag_in_hot - h_stag_out_hot)
    Q_cold = mdot_c_total * (h_stag_out_cold - h_stag_in_cold)

    # Check if Q_hot and Q_cold are similar
    if abs(Q_hot - Q_cold) / max(abs(Q_hot), abs(Q_cold)) > 0.01 * 1e6:
        logger.warning(
            "Q_hot and Q_cold are not similar: Q_hot=%.2f MW, Q_cold=%.2f MW",
            Q_hot / 1e6,
            Q_cold / 1e6,
        )

    # Calculate overall effectiveness using mean cp
    cp_h_avg = (h_stag_in_hot - h_stag_out_hot) / (Th_in - Th_out)
    cp_c_avg = (h_stag_out_cold - h_stag_in_cold) / (Tc_out - Tc_in)
    C_h = mdot_h_total * cp_h_avg
    C_c = mdot_c_total * cp_c_avg
    C_min = min(C_h, C_c)
    Q_max = C_min * (Th_in - Tc_in)
    epsilon = Q_hot / Q_max if Q_max > 0 else 0.0

    # Calculate NTU from epsilon using inverse relationship
    # For counterflow: NTU = ln((1-epsilon*Cr)/(1-epsilon)) / (1-Cr)
    Cr = C_min / max(C_h, C_c)
    NTU = diagnostics["total_UA"] / C_min

    # Pressure drops
    dP_hot = Ph_in - Ph_out
    dP_cold = Pc_out - Pc_in
    dP_hot_pct = 100.0 * dP_hot / Ph_in if Ph_in > 0 else 0.0
    dP_cold_pct = 100.0 * dP_cold / Pc_in if Pc_in > 0 else 0.0

    # Store in diagnostics
    diagnostics["Q_total"] = float(Q_hot)
    diagnostics["Q_hot"] = float(Q_hot)
    diagnostics["Q_cold"] = float(Q_cold)
    diagnostics["epsilon"] = float(epsilon)
    diagnostics["NTU"] = float(NTU)
    diagnostics["Cr"] = float(Cr)
    diagnostics["dP_hot"] = float(dP_hot)
    diagnostics["dP_hot_pct"] = float(dP_hot_pct)
    diagnostics["dP_cold"] = float(dP_cold)
    diagnostics["dP_cold_pct"] = float(dP_cold_pct)
    diagnostics["Th_out"] = float(Th_out)
    diagnostics["Tc_out"] = float(Tc_out)
    diagnostics["Ph_out"] = float(Ph_out)
    diagnostics["Pc_out"] = float(Pc_out)

    # Hot-side inlet/exit Reynolds, Stanton, friction, and Eckert numbers
    # Recompute local transport properties at the two ends
    mu_h_in = state_h_in.mu
    mu_h_out = state_h_out.mu
    k_h_in = state_h_in.k
    k_h_out = state_h_out.k
    cp_h_in = state_h_in.cp
    cp_h_out = state_h_out.cp
    rho_h_in = state_h_in.rho
    rho_h_out = state_h_out.rho

    # For cold side at corresponding radii: use inner for hot-out, outer for hot-in
    state_c_at_hot_out = state_c_in
    state_c_at_hot_in = state_c_out
    mu_c_in = state_c_at_hot_in.mu
    mu_c_out = state_c_at_hot_out.mu
    k_c_in = state_c_at_hot_in.k
    k_c_out = state_c_at_hot_out.k
    cp_c_in = state_c_at_hot_in.cp
    cp_c_out = state_c_at_hot_out.cp

    # Outer/inner diameters
    # Note: for Stanton based on tube-bank external flow, use OD-based Nu and Re
    # Access OD/ID via geometry values available from radii spacing through total HT areas; simpler to pull from ratios used elsewhere
    # We don't carry geometry here, so infer from area ratios is cumbersome; instead, stash OD/ID in diagnostics earlier

    # Use geometry-provided diameters directly
    Do_eff = geometry.tube_outer_diam
    Di_eff = geometry.tube_inner_diam()

    # Mass fluxes and velocities
    V_h_in = G_h_in / rho_h_in if rho_h_in > 0 else float("nan")
    V_h_out = G_h_out / rho_h_out if rho_h_out > 0 else float("nan")

    # Reynolds based on OD
    Re_h_in = G_h_in * Do_eff / mu_h_in if np.isfinite(Do_eff) and mu_h_in > 0 else float("nan")
    Re_h_out = G_h_out * Do_eff / mu_h_out if np.isfinite(Do_eff) and mu_h_out > 0 else float("nan")

    # Prandtl numbers
    Pr_h_in = mu_h_in * cp_h_in / k_h_in if k_h_in > 0 else float("nan")
    Pr_h_out = mu_h_out * cp_h_out / k_h_out if k_h_out > 0 else float("nan")

    # Tube bank Nu and f at inlet and outlet
    Nu_h_in, f_h_in = (
        tube_bank_nusselt_number_and_friction_factor(
            Re_h_in,
            geometry.tube_spacing_long,
            geometry.tube_spacing_trv,
            Pr_h_in,
            inline=(not geometry.staggered),
            n_rows=geometry.n_rows_per_header * n_headers,
        )
        if np.isfinite(Re_h_in) and np.isfinite(Pr_h_in)
        else (float("nan"), float("nan"))
    )
    Nu_h_out, f_h_out = (
        tube_bank_nusselt_number_and_friction_factor(
            Re_h_out,
            geometry.tube_spacing_long,
            geometry.tube_spacing_trv,
            Pr_h_out,
            inline=(not geometry.staggered),
            n_rows=geometry.n_rows_per_header * n_headers,
        )
        if np.isfinite(Re_h_out) and np.isfinite(Pr_h_out)
        else (float("nan"), float("nan"))
    )

    # Stanton numbers
    St_h_in = (
        Nu_h_in / (Re_h_in * Pr_h_in)
        if np.isfinite(Nu_h_in) and Re_h_in > 0 and Pr_h_in > 0
        else float("nan")
    )
    St_h_out = (
        Nu_h_out / (Re_h_out * Pr_h_out)
        if np.isfinite(Nu_h_out) and Re_h_out > 0 and Pr_h_out > 0
        else float("nan")
    )

    # Local h_h and h_c at inlet and outlet (use OD based for hot, ID for cold)
    Di_eff = Di_eff
    h_h_in = (
        Nu_h_in * k_h_in / Do_eff
        if np.isfinite(Nu_h_in) and np.isfinite(Do_eff) and Do_eff > 0
        else float("nan")
    )
    h_h_out = (
        Nu_h_out * k_h_out / Do_eff
        if np.isfinite(Nu_h_out) and np.isfinite(Do_eff) and Do_eff > 0
        else float("nan")
    )
    # Apply heat transfer tuning consistent with marching
    h_h_in *= opts.heat_transfer_tuning if np.isfinite(h_h_in) else 1.0
    h_h_out *= opts.heat_transfer_tuning if np.isfinite(h_h_out) else 1.0

    # Cold side
    # Re_c and Nu_c at the two ends (Re_c is same if G_c and mu don't vary with radius much; mu varies with T)
    tube_inner_diam = geometry.tube_inner_diam()
    Re_c_in = (
        G_c * tube_inner_diam / mu_c_in
        if np.isfinite(tube_inner_diam) and mu_c_in > 0
        else float("nan")
    )
    Re_c_out = (
        G_c * tube_inner_diam / mu_c_out
        if np.isfinite(tube_inner_diam) and mu_c_out > 0
        else float("nan")
    )
    Pr_c_in = mu_c_in * cp_c_in / k_c_in if k_c_in > 0 else float("nan")
    Pr_c_out = mu_c_out * cp_c_out / k_c_out if k_c_out > 0 else float("nan")
    Nu_c_in = (
        circular_pipe_nusselt(Re_c_in, 0, prandtl=Pr_c_in)
        if np.isfinite(Re_c_in) and np.isfinite(Pr_c_in)
        else float("nan")
    )
    Nu_c_out = (
        circular_pipe_nusselt(Re_c_out, 0, prandtl=Pr_c_out)
        if np.isfinite(Re_c_out) and np.isfinite(Pr_c_out)
        else float("nan")
    )
    h_c_in = (
        Nu_c_in * k_c_in / tube_inner_diam
        if np.isfinite(Nu_c_in) and np.isfinite(tube_inner_diam) and tube_inner_diam > 0
        else float("nan")
    )
    h_c_out = (
        Nu_c_out * k_c_out / tube_inner_diam
        if np.isfinite(Nu_c_out) and np.isfinite(tube_inner_diam) and tube_inner_diam > 0
        else float("nan")
    )

    # Wall conduction term
    wall_term = (
        Do_eff
        / (2.0 * (diagnostics.get("_wall_k", float("nan"))))
        * np.log(Do_eff / tube_inner_diam)
        if np.isfinite(Do_eff)
        and np.isfinite(tube_inner_diam)
        and Do_eff > 0
        and tube_inner_diam > 0
        else float("nan")
    )

    def _U_local(hh: float, hc: float) -> float:
        if not (np.isfinite(hh) and np.isfinite(hc) and np.isfinite(wall_term)):
            return float("nan")
        return 1.0 / (1.0 / hh + (1.0 / hc) * (Do_eff / tube_inner_diam) + wall_term)

    U_in = _U_local(h_h_in, h_c_in)
    U_out = _U_local(h_h_out, h_c_out)

    # Wall temperatures on hot side: Tw_hot = Th - q''/h_h, q'' = U*(Th-Tc)
    deltaT_in = Th_in - Tc_out
    deltaT_out = Th_out - Tc_in
    qpp_in = U_in * deltaT_in if np.isfinite(U_in) else float("nan")
    qpp_out = U_out * deltaT_out if np.isfinite(U_out) else float("nan")
    Tw_h_in = (
        Th_in - (qpp_in / h_h_in)
        if np.isfinite(qpp_in) and np.isfinite(h_h_in) and h_h_in > 0
        else float("nan")
    )
    Tw_h_out = (
        Th_out - (qpp_out / h_h_out)
        if np.isfinite(qpp_out) and np.isfinite(h_h_out) and h_h_out > 0
        else float("nan")
    )

    # Eckert numbers
    Ec_h_in = (
        (V_h_in**2) / (cp_h_in * max(Th_in - Tw_h_in, 1e-16))
        if np.isfinite(V_h_in) and np.isfinite(Tw_h_in)
        else float("nan")
    )
    Ec_h_out = (
        (V_h_out**2) / (cp_h_out * max(Th_out - Tw_h_out, 1e-16))
        if np.isfinite(V_h_out) and np.isfinite(Tw_h_out)
        else float("nan")
    )

    diagnostics["Re_h_in"] = float(Re_h_in)
    diagnostics["Re_h_out"] = float(Re_h_out)
    diagnostics["St_h_in"] = float(St_h_in)
    diagnostics["St_h_out"] = float(St_h_out)
    diagnostics["f_h_in"] = float(f_h_in)
    diagnostics["f_h_out"] = float(f_h_out)
    diagnostics["Ec_h_in"] = float(Ec_h_in)
    diagnostics["Ec_h_out"] = float(Ec_h_out)
    # Store supporting quantities for explanation/logging
    diagnostics["V_h_in"] = float(V_h_in)
    diagnostics["V_h_out"] = float(V_h_out)
    diagnostics["V2_h_in"] = float(V_h_in**2) if np.isfinite(V_h_in) else float("nan")
    diagnostics["V2_h_out"] = float(V_h_out**2) if np.isfinite(V_h_out) else float("nan")
    diagnostics["cp_h_in"] = float(cp_h_in)
    diagnostics["cp_h_out"] = float(cp_h_out)
    diagnostics["dT_hw_in"] = float(Th_in - Tw_h_in) if np.isfinite(Tw_h_in) else float("nan")
    diagnostics["dT_hw_out"] = float(Th_out - Tw_h_out) if np.isfinite(Tw_h_out) else float("nan")


def F_inboard(
    Th_out_guess: float,
    Ph_out_guess: float,
    *,
    geometry: RadialInvoluteGeometry,
    fluid_hot: FluidModel,
    fluid_cold: FluidModel,
    Th_in: float,
    Ph_known: float,
    Tc_in: float,
    Pc_in: float,
    mdot_h_total: float,
    mdot_c_total: float,
    wall_conductivity: float,
    options: MarchingOptions | None = None,
    diagnostics: dict[str, float] | None = None,
    outlet_pressure_known: bool = False,
) -> np.ndarray:
    """Return residuals at the hot outer boundary for an inboard shoot.

    For an inboard radial involute HEx, it is assumed that the hot fluid flows from
    the outer radius to the inner radius, and the cold fluid flows (inside the tubes)
    from the inner radius to the outer radius.
    Assuming the flow inlet properties are known, this means that the known boundary conditions are:
      - Hot inlet (at outer radius): ``Th_in, Ph_in, mdot_h_total``
      - Cold inlet (at inner radius): ``Tc_in, Pc_in, mdot_c_total``
    As the marching solver marches outward from inner radius to outer radius,
    the hot exit (at inner radius) is guessed as Th_out_guess, Ph_out_guess

    Returns a vector of how accurate this guess of Th_out_guess, Ph_out_guess was in terms
    of whether the hot inlet temperatures calculated from the marching solver match to the known
    hot inlet temperature and pressure.

    Returns:
    - [0] = temperature residual Th_in_calc - Th_in
    - [1] = pressure residual Ph_in_calc - Ph_in (or 0 if outlet pressure is known)
    """

    opts = options or MarchingOptions()
    geom_cache = _compute_geometry_arrays(geometry)

    tube_inner_diam = geometry.tube_inner_diam()
    # axial_length = geometry.axial_length()
    # radius_inner = geometry.radius_inner_whole_hex()
    # hx_area = np.pi * (geometry.radius_outer_whole_hex**2 - radius_inner**2)

    m_dot_hot = mdot_h_total / geometry.n_headers
    m_dot_cold = mdot_c_total / geometry.n_headers

    n_nodes = geom_cache.radii.size
    Th = np.zeros(n_nodes)
    Ph = np.zeros(n_nodes)
    Tc = np.zeros(n_nodes)
    Pc = np.zeros(n_nodes)
    UA_sum = 0.0
    # Boundary conditions at the inner radius (start of march)
    Th[0] = Th_out_guess
    if not outlet_pressure_known:
        Ph[0] = Ph_out_guess
    else:
        Ph[0] = Ph_known

    Tc[0] = Tc_in
    Pc[0] = Pc_in

    state_hot = fluid_hot.state(Th[0], Ph[0])
    rho_h = state_hot.rho

    state_cold = fluid_cold.state(Tc[0], Pc[0])
    rho_c = state_cold.rho

    for j in range(n_nodes - 1):
        if not np.all(np.isfinite([Th[j], Ph[j], Tc[j], Pc[j]])):
            logger.warning(
                "Non-finite state before layer %d: Th=%s K, Ph=%s Pa, Tc=%s K, Pc=%s Pa",
                j,
                Th[j],
                Ph[j],
                Tc[j],
                Pc[j],
            )
            return np.array([np.nan, np.nan], dtype=float)

        state_hot = fluid_hot.state(Th[j], Ph[j])
        state_cold = fluid_cold.state(Tc[j], Pc[j])

        cp_h = state_hot.cp
        mu_h = state_hot.mu
        k_h = state_hot.k
        rho_h = state_hot.rho
        Pr_h = mu_h * cp_h / k_h

        cp_c = state_cold.cp
        mu_c = state_cold.mu
        k_c = state_cold.k
        rho_c = state_cold.rho
        Pr_c = mu_c * cp_c / k_c

        area_free_hot = geom_cache.area_free_hot[j]
        area_free_cold = geom_cache.area_free_cold[j]

        G_h = m_dot_hot / area_free_hot
        G_c = m_dot_cold / area_free_cold

        # Re_h = G_h * geom_cache.d_h_hot[j] / max(mu_h, 1e-16)
        Re_h_od = G_h * geometry.tube_outer_diam / mu_h
        Re_c = G_c * tube_inner_diam / mu_c

        Nu_c = circular_pipe_nusselt(Re_c, 0, prandtl=Pr_c)
        f_c = circular_pipe_friction_factor(Re_c, 0)
        h_c = Nu_c * k_c / tube_inner_diam

        Nu_h, f_h = tube_bank_nusselt_number_and_friction_factor(
            Re_h_od,
            geometry.tube_spacing_long,
            geometry.tube_spacing_trv,
            Pr_h,
            inline=(not geometry.staggered),
            n_rows=geometry.n_rows_per_header * geometry.n_headers,  # use total rows
        )
        h_h = Nu_h * k_h / geometry.tube_outer_diam
        h_h *= opts.heat_transfer_tuning

        wall_term = (
            geometry.tube_outer_diam
            / (2.0 * wall_conductivity)
            * np.log(geometry.tube_outer_diam / tube_inner_diam)
        )

        h_h_inv = 1.0 / h_h
        h_c_inv = 1.0 / h_c

        U_hot = 1.0 / (h_h_inv + h_c_inv * (geometry.tube_outer_diam / tube_inner_diam) + wall_term)
        # U_cold = 1.0 / (
        #    h_c_inv + h_h_inv * (tube_inner_diam / max(geometry.tube_outer_diam, 1e-16)) + wall_term
        # )
        UA_sum += U_hot * geom_cache.area_ht_hot[j] * geometry.n_headers

        C_h = m_dot_hot * cp_h
        C_c = m_dot_cold * cp_c
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        Cr = C_min / C_max

        NTU_local = U_hot * geom_cache.area_ht_hot[j] / C_min
        flow_type = "Cmax_mixed" if C_h > C_c else "Cmin_mixed"
        eps_local = epsilon_ntu(
            NTU_local,
            Cr,
            exchanger_type="cross_flow",
            flow_type=flow_type,
            n_passes=1,
        )

        delta_T = Th[j] - Tc[j]
        q_max = C_min * delta_T
        q = eps_local * q_max

        if j == n_nodes - 1 and n_nodes > 2 and logger.isEnabledFor(logging.DEBUG):
            mid_index = n_nodes // 2
            logger.debug(
                "1D geometry ratios total: A_hot/Aff_hot=%5.3e, A_cold/Aff_cold=%5.3e",
                np.sum(geom_cache.area_ht_hot) / area_free_hot[mid_index],
                np.sum(geom_cache.area_ht_cold) / area_free_cold[mid_index],
            )

        dh0_hot = -q / m_dot_hot
        dh0_cold = q / m_dot_cold

        tau_hot = (
            opts.pressure_drop_tuning
            * f_h
            * (geom_cache.area_ht_hot[j] / area_free_hot)
            * (G_h**2)
            / (2.0 * rho_h)
        )
        tau_cold = (
            opts.pressure_drop_tuning
            * f_c
            * (geom_cache.area_ht_cold[j] / area_free_cold)
            * (G_c**2)
            / (2.0 * rho_c)
        )

        Th[j + 1], Ph[j + 1] = update_static_properties(
            fluid_hot,
            G_h,
            dh0_hot,
            tau_hot,
            T_a=Th[j],
            p_b=Ph[j],
            a_is_in=False,
            b_is_in=False,
            max_iter=opts.property_solver_iterations,
            tol_T=opts.property_solver_tol_T,
            rel_tol_p=opts.property_solver_rel_tol_p,
        )

        Tc[j + 1], Pc[j + 1] = update_static_properties(
            fluid_cold,
            G_c,
            dh0_cold,
            tau_cold,
            T_a=Tc[j],
            p_b=Pc[j],
            a_is_in=True,
            b_is_in=True,
            max_iter=opts.property_solver_iterations,
            tol_T=opts.property_solver_tol_T,
            rel_tol_p=opts.property_solver_rel_tol_p,
        )

        if not np.all(np.isfinite([Th[j + 1], Ph[j + 1], Tc[j + 1], Pc[j + 1]])):
            logger.warning(
                "Non-finite state after layer %d of %d: Th=%s K, Ph=%s Pa, Tc=%s K, Pc=%s Pa",
                j,
                n_nodes - 1,
                Th[j + 1],
                Ph[j + 1],
                Tc[j + 1],
                Pc[j + 1],
            )
            return np.array([np.nan, np.nan], dtype=float)

    Th_in_calc = Th[-1]
    Ph_in_calc = Ph[-1]

    residual_T = Th_in_calc - Th_in
    residual_P = Ph_in_calc - Ph_known if not outlet_pressure_known else 0.0

    if diagnostics is not None:
        total_area_ht_hot = np.sum(geom_cache.area_ht_hot)
        total_UA = UA_sum
        diagnostics["total_UA"] = float(total_UA)
        diagnostics["total_area_ht_hot"] = float(total_area_ht_hot)
        # Stash geometry and wall info for downstream diagnostics
        diagnostics["_Do"] = float(geometry.tube_outer_diam)
        diagnostics["_Di"] = float(tube_inner_diam)
        diagnostics["_wall_k"] = float(wall_conductivity)
        mid_index = n_nodes // 2
        total_aff_hot = geom_cache.area_free_hot[mid_index]
        total_aff_cold = geom_cache.area_free_cold[mid_index]
        diagnostics["area_ratio_hot_total"] = (
            float(np.sum(geom_cache.area_ht_hot)) / total_aff_hot
            if total_aff_hot > 0.0
            else float("nan")
        )
        diagnostics["area_ratio_cold_total"] = (
            float(np.sum(geom_cache.area_ht_cold)) / total_aff_cold
            if total_aff_cold > 0.0
            else float("nan")
        )
        diagnostics["n_layers"] = float(n_nodes - 1)

        # Calculate overall performance metrics
        _compute_overall_performance(
            diagnostics,
            Th,
            Ph,
            Tc,
            Pc,
            Th_in,
            Ph_known,
            Tc_in,
            Pc_in,
            fluid_hot,
            fluid_cold,
            mdot_h_total,
            mdot_c_total,
            geom_cache,
            geometry.n_headers,
            geometry,
            opts,
            outlet_pressure_known,
        )

    return np.array([residual_T, residual_P])


__all__ = [
    "RadialInvoluteGeometry",
    "MarchingOptions",
    "F_inboard",
]
