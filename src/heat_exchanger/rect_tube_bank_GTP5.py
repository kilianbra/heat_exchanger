"""Rectangular tube bank heat-exchanger solver with backward marching (GTP5 variant).

This module mirrors the structure of `involute_inboard.py` but for a straight/
rectangular tube bank geometry. We march from the last row (outlet) to the first
(inlet), using one step per pass. Within each pass we assume a single, average
heat-transfer coefficient based on the Reynolds numbers computed from:
- Hot/outside:  G_h = mdot_h_total / area_free_flow_outer
- Cold/inside:  G_c = mdot_c_total / area_free_flow_inner

Within each pass the crossflow epsilon-NTU relationship is used. The flow outside
the tubes (hot) is mixed, while the tube-side (cold) is unmixed. The appropriate
crossflow relation is chosen based on whether the hot or cold stream is Cmax.

cp evaluation: do one iteration per pass.
- First, assume properties at the outlet boundary (hot-out/ cold-in).
- Then, recompute using cp = Δh/ΔT over the pass.

Optionally, the hot-side outlet pressure can be treated as fixed (no need to guess).
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
from heat_exchanger.tube_bank_straight import TubeBankStraightGeometry


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RectTubeBankGeometry(TubeBankStraightGeometry):
    """Rectangular/straight tube bank geometry (alias of `TubeBankStraightGeometry`)."""


@dataclass(frozen=True)
class MarchingOptions:
    """Tuning knobs and numerical tolerances for the marching solver."""

    heat_transfer_tuning: float = 1.0
    pressure_drop_tuning: float = 1.0
    property_solver_iterations: int = 20
    # ruff: noqa: N815 (allow mixed case variables)
    property_solver_tol_T: float = 1e-2
    property_solver_rel_tol_p: float = 1e-3
    # Perform one cp iteration per pass (first with outlet properties, then Δh/ΔT)
    cp_iterations: int = 2


def _compute_overall_performance(
    diagnostics: dict[str, float],
    Th: np.ndarray,
    Ph: np.ndarray,
    Tc: np.ndarray,
    Pc: np.ndarray,
    Th_in: float,
    Ph_in: float,
    Tc_in: float,
    Pc_in: float,
    fluid_hot: FluidModel,
    fluid_cold: FluidModel,
    mdot_h_total: float,
    mdot_c_total: float,
    geometry: RectTubeBankGeometry,
) -> None:
    """Calculate overall heat exchanger performance metrics and store in diagnostics."""
    Th_out = Th[-1]
    Ph_out = Ph[-1]
    Tc_out = Tc[-1]
    Pc_out = Pc[-1]

    state_h_in = fluid_hot.state(Th_in, Ph_in)
    state_h_out = fluid_hot.state(Th_out, Ph_out)
    state_c_in = fluid_cold.state(Tc_in, Pc_in)
    state_c_out = fluid_cold.state(Tc_out, Pc_out)

    area_free_hot = geometry.area_free_flow_outer()
    area_free_cold = geometry.area_free_flow_inner()

    G_h = mdot_h_total / area_free_hot
    G_c = mdot_c_total / area_free_cold

    h_stag_in_hot = state_h_in.h + 0.5 * (G_h / state_h_in.rho) ** 2
    h_stag_out_hot = state_h_out.h + 0.5 * (G_h / state_h_out.rho) ** 2
    h_stag_in_cold = state_c_in.h + 0.5 * (G_c / state_c_in.rho) ** 2
    h_stag_out_cold = state_c_out.h + 0.5 * (G_c / state_c_out.rho) ** 2

    Q_hot = mdot_h_total * (h_stag_in_hot - h_stag_out_hot)
    Q_cold = mdot_c_total * (h_stag_out_cold - h_stag_in_cold)

    if abs(Q_hot - Q_cold) / max(abs(Q_hot), abs(Q_cold), 1.0) > 0.01:
        logger.warning(
            "Q_hot and Q_cold differ: Q_hot=%.2f kW, Q_cold=%.2f kW",
            Q_hot / 1e3,
            Q_cold / 1e3,
        )

    # Effectiveness
    cp_h_avg = (h_stag_in_hot - h_stag_out_hot) / (Th_in - Th_out) if Th_in != Th_out else state_h_in.cp
    cp_c_avg = (h_stag_out_cold - h_stag_in_cold) / (Tc_out - Tc_in) if Tc_out != Tc_in else state_c_in.cp
    C_h = mdot_h_total * cp_h_avg
    C_c = mdot_c_total * cp_c_avg
    C_min = min(C_h, C_c)
    Q_max = C_min * (Th_in - Tc_in)
    epsilon = Q_hot / Q_max if Q_max > 0 else 0.0

    NTU = diagnostics.get("total_UA", 0.0) / C_min if C_min > 0 else 0.0

    dP_hot = Ph_in - Ph_out
    dP_cold = Pc_out - Pc_in
    dP_hot_pct = 100.0 * dP_hot / Ph_in if Ph_in > 0 else 0.0
    dP_cold_pct = 100.0 * dP_cold / Pc_in if Pc_in > 0 else 0.0

    diagnostics["Q_total"] = float(Q_hot)
    diagnostics["Q_hot"] = float(Q_hot)
    diagnostics["Q_cold"] = float(Q_cold)
    diagnostics["epsilon"] = float(epsilon)
    diagnostics["NTU"] = float(NTU)
    diagnostics["Cr"] = float(C_min / max(C_h, C_c))
    diagnostics["dP_hot"] = float(dP_hot)
    diagnostics["dP_hot_pct"] = float(dP_hot_pct)
    diagnostics["dP_cold"] = float(dP_cold)
    diagnostics["dP_cold_pct"] = float(dP_cold_pct)
    diagnostics["Th_out"] = float(Th_out)
    diagnostics["Tc_out"] = float(Tc_out)
    diagnostics["Ph_out"] = float(Ph_out)
    diagnostics["Pc_out"] = float(Pc_out)


def F_backward(
    Th_out_guess: float,
    Ph_out_guess: float,
    *,
    geometry: RectTubeBankGeometry,
    fluid_hot: FluidModel,
    fluid_cold: FluidModel,
    Th_in: float,
    Ph_in: float,
    Tc_in: float,
    Pc_in: float,
    mdot_h_total: float,
    mdot_c_total: float,
    wall_conductivity: float,
    options: MarchingOptions | None = None,
    diagnostics: dict[str, float] | None = None,
    # If provided, treat hot outlet pressure as fixed (not a guessed unknown)
    Ph_out_specified: float | None = None,
) -> np.ndarray:
    """Return residuals at the hot inlet boundary for a backward march.

    Default mode: Th_out and Ph_out are guessed unknowns; Tc_in and Pc_in are known.
    Optional: pass `Ph_out_specified` to fix the hot-side outlet pressure (only T residual).

    Marching:
      - From last row to first, one step per pass.
      - At each pass, use a single, average heat-transfer coefficient for the given Re.
      - Within the pass, use crossflow epsilon-NTU with hot mixed and cold unmixed;
        choose Cmax_mixed or Cmin_mixed based on which stream is Cmax.
      - cp evaluation uses one iteration per pass (initial props at outlet, then Δh/ΔT).
    """
    opts = options or MarchingOptions()

    # Determine whether pressure is an unknown or fixed at outlet
    pressure_fixed = Ph_out_specified is not None
    Ph_out_boundary = Ph_out_specified if pressure_fixed else Ph_out_guess

    n_passes = geometry.n_passes
    tube_inner_diam = geometry.tube_inner_diam()
    area_ht_hot_per_row = geometry.area_heat_transfer_outer_per_row()
    area_ht_cold_per_row = geometry.area_heat_transfer_inner_per_row()

    # Free-flow areas (constant across passes)
    area_free_hot = geometry.area_free_flow_outer()
    area_free_cold = geometry.area_free_flow_inner()

    # Mass velocities
    G_h = mdot_h_total / area_free_hot
    G_c = mdot_c_total / area_free_cold

    # Arrays: 0 = inlet, -1 = outlet
    n_nodes = n_passes + 1
    Th = np.zeros(n_nodes)
    Ph = np.zeros(n_nodes)
    Tc = np.zeros(n_nodes)
    Pc = np.zeros(n_nodes)
    UA_sum = 0.0

    # Boundary conditions at outlet (start of backward march)
    Th[-1] = Th_out_guess
    Ph[-1] = Ph_out_boundary
    # Cold side inlet at HX inlet
    Tc[0] = Tc_in
    Pc[0] = Pc_in

    # March backward through passes
    for j in range(n_passes - 1, -1, -1):
        # Known before pass j:
        #   hot outlet (downstream): Th[j+1], Ph[j+1]
        #   cold inlet (upstream):   Tc[j],   Pc[j]
        if not np.all(np.isfinite([Th[j + 1], Ph[j + 1], Tc[j], Pc[j]])):
            logger.warning(
                "Non-finite state before pass %d: Th=%s K, Ph=%s Pa, Tc=%s K, Pc=%s Pa",
                j,
                Th[j + 1],
                Ph[j + 1],
                Tc[j],
                Pc[j],
            )
            return np.array([np.nan, np.nan], dtype=float) if not pressure_fixed else np.array([np.nan])

        # Initial values for cp iteration
        Th_pass_out = np.nan
        Tc_pass_out = np.nan
        q = np.nan

        for iter_idx in range(opts.cp_iterations):
            # Use outlet (hot) and inlet (cold) properties first; then average based on Δh/ΔT
            state_h_out = fluid_hot.state(Th[j + 1], Ph[j + 1])
            state_c_in = fluid_cold.state(Tc[j], Pc[j])

            if iter_idx == 0:
                cp_h = state_h_out.cp
                cp_c = state_c_in.cp
                mu_h = state_h_out.mu
                k_h = state_h_out.k
                rho_h = state_h_out.rho
                mu_c = state_c_in.mu
                k_c = state_c_in.k
                rho_c = state_c_in.rho
                # crude initial temperature updates for cp iteration
                Th_pass_in = Th[j + 1]
            else:
                # Use Δh/ΔT to recompute cp over the pass (requires pass-out predictions)
                state_h_in = fluid_hot.state(Th_pass_out, Ph[j + 1])  # pressure approx: outlet P
                state_c_out = fluid_cold.state(Tc_pass_out, Pc[j])    # pressure approx: inlet P
                dh_h = state_h_in.h - state_h_out.h
                dT_h = Th_pass_out - Th[j + 1]
                cp_h = dh_h / dT_h if abs(dT_h) > 0.1 else state_h_out.cp

                dh_c = state_c_out.h - state_c_in.h
                dT_c = Tc_pass_out - Tc[j]
                cp_c = dh_c / dT_c if abs(dT_c) > 0.1 else state_c_in.cp

                mu_h = 0.5 * (state_h_in.mu + state_h_out.mu)
                k_h = 0.5 * (state_h_in.k + state_h_out.k)
                rho_h = 0.5 * (state_h_in.rho + state_h_out.rho)
                mu_c = 0.5 * (state_c_in.mu + state_c_out.mu)
                k_c = 0.5 * (state_c_in.k + state_c_out.k)
                rho_c = 0.5 * (state_c_in.rho + state_c_out.rho)
                Th_pass_in = Th[j + 1]

            Pr_h = mu_h * cp_h / max(k_h, 1e-16)
            Pr_c = mu_c * cp_c / max(k_c, 1e-16)

            Re_h_od = G_h * geometry.tube_outer_diam / max(mu_h, 1e-16)
            Re_c = G_c * tube_inner_diam / max(mu_c, 1e-16)

            # Heat transfer coefficients and friction factors
            Nu_c = circular_pipe_nusselt(Re_c, 0, prandtl=Pr_c)
            f_c = circular_pipe_friction_factor(Re_c, 0)
            h_c = Nu_c * k_c / max(tube_inner_diam, 1e-16)

            Nu_h, f_h = tube_bank_nusselt_number_and_friction_factor(
                Re_h_od,
                geometry.tube_spacing_long,
                geometry.tube_spacing_trv,
                Pr_h,
                inline=(not geometry.staggered),
                n_rows=geometry.n_rows_per_pass,
            )
            h_h = Nu_h * k_h / max(geometry.tube_outer_diam, 1e-16)
            h_h *= opts.heat_transfer_tuning

            # Overall U (based on outer area)
            wall_term = (
                geometry.tube_outer_diam / (2.0 * wall_conductivity) * np.log(
                    max(geometry.tube_outer_diam / max(tube_inner_diam, 1e-16), 1e-16)
                )
            )
            U_hot = 1.0 / (1.0 / h_h + (1.0 / h_c) * (geometry.tube_outer_diam / max(tube_inner_diam, 1e-16)) + wall_term)

            # Pass areas and UA
            area_ht_hot_pass = area_ht_hot_per_row * geometry.n_rows_per_pass
            area_ht_cold_pass = area_ht_cold_per_row * geometry.n_rows_per_pass
            UA_pass = U_hot * area_ht_hot_pass

            # Capacity rates
            C_h = mdot_h_total * cp_h
            C_c = mdot_c_total * cp_c
            C_min = min(C_h, C_c)
            C_max = max(C_h, C_c)
            Cr = C_min / C_max if C_max > 0 else 0.0

            # Crossflow: hot mixed, cold unmixed
            flow_type = "Cmax_mixed" if C_h >= C_c else "Cmin_mixed"
            NTU_pass = UA_pass / max(C_min, 1e-16)
            eps_pass = epsilon_ntu(NTU_pass, Cr, exchanger_type="cross_flow", flow_type=flow_type, n_passes=1)

            # Heat for this pass; use current boundary temps
            delta_T = Th_pass_in - Tc[j]
            q_max = C_min * delta_T
            q = eps_pass * q_max

            # Temperature updates across pass (backward for hot, forward for cold)
            dT_h = -q / max(C_h, 1e-16)
            dT_c = q / max(C_c, 1e-16)
            Th_pass_out = Th_pass_in + dT_h
            Tc_pass_out = Tc[j] + dT_c

        # Pressure/thermo updates over pass using update_static_properties
        dh0_hot = -q / mdot_h_total
        tau_hot = (
            opts.pressure_drop_tuning * f_h * (area_ht_hot_pass / area_free_hot) * (G_h**2) / (2.0 * rho_h)
        )
        Th[j], Ph[j] = update_static_properties(
            fluid_hot,
            G_h,
            dh0_hot,
            tau_hot,
            T_a=Th[j + 1],
            p_b=Ph[j + 1],
            a_is_in=False,
            b_is_in=False,
            max_iter=opts.property_solver_iterations,
            tol_T=opts.property_solver_tol_T,
            rel_tol_p=opts.property_solver_rel_tol_p,
        )

        dh0_cold = q / mdot_c_total
        tau_cold = (
            opts.pressure_drop_tuning * f_c * (area_ht_cold_pass / area_free_cold) * (G_c**2) / (2.0 * rho_c)
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

        if not np.all(np.isfinite([Th[j], Ph[j], Tc[j + 1], Pc[j + 1]])):
            logger.warning(
                "Non-finite state after pass %d of %d: Th=%s K, Ph=%s Pa, Tc=%s K, Pc=%s Pa",
                j,
                n_passes,
                Th[j],
                Ph[j],
                Tc[j + 1],
                Pc[j + 1],
            )
            return np.array([np.nan, np.nan], dtype=float) if not pressure_fixed else np.array([np.nan])

        UA_sum += UA_pass

    # Residuals at inlet
    Th_in_calc = Th[0]
    Ph_in_calc = Ph[0]
    residual_T = Th_in_calc - Th_in
    residual_P = Ph_in_calc - Ph_in

    # Diagnostics
    if diagnostics is not None:
        diagnostics["total_UA"] = float(UA_sum)
        diagnostics["total_area_ht_hot"] = float(area_ht_hot_per_row * geometry.n_rows_per_pass * n_passes)
        diagnostics["total_area_ht_cold"] = float(area_ht_cold_per_row * geometry.n_rows_per_pass * n_passes)
        diagnostics["area_ratio_hot_total"] = (
            float(area_ht_hot_per_row * geometry.n_rows_per_pass * n_passes / area_free_hot)
            if area_free_hot > 0
            else float("nan")
        )
        diagnostics["area_ratio_cold_total"] = (
            float(area_ht_cold_per_row * geometry.n_rows_per_pass * n_passes / area_free_cold)
            if area_free_cold > 0
            else float("nan")
        )
        diagnostics["n_passes"] = float(n_passes)
        _compute_overall_performance(
            diagnostics,
            Th,
            Ph,
            Tc,
            Pc,
            Th_in,
            Ph_in,
            Tc_in,
            Pc_in,
            fluid_hot,
            fluid_cold,
            mdot_h_total,
            mdot_c_total,
            geometry,
        )

    # Return residual vector
    if pressure_fixed:
        return np.array([residual_T], dtype=float)
    return np.array([residual_T, residual_P], dtype=float)


__all__ = [
    "RectTubeBankGeometry",
    "MarchingOptions",
    "F_backward",
]

