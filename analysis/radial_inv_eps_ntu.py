"""Investigate epsilon-NTU relationship for Radial Involute HEx with constant U.

This script tests whether a classical epsilon-NTU relationship can be established
for the radial involute geometry when using a constant overall heat transfer
coefficient (U) and perfect gas properties (constant cp).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from heat_exchanger.conservation import update_static_properties
from heat_exchanger.correlations import (
    circular_pipe_friction_factor,
    circular_pipe_nusselt,
    tube_bank_nusselt_number_and_friction_factor,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu
from heat_exchanger.fluids.protocols import FluidModel, PerfectGasFluid
from heat_exchanger.involute_inboard import (
    F_inboard,
    MarchingOptions,
    RadialInvoluteGeometry,
    _GeometryCache,
    _compute_geometry_arrays,
    _compute_overall_performance,
)
from heat_exchanger.logging_utils import configure_logging

logger = logging.getLogger(__name__)


WALL_CONDUCTIVITY_304_SS = 14.0


def F_inboard_constant_U(
    Th_out_guess: float,
    Ph_out_guess: float,
    *,
    geometry: RadialInvoluteGeometry,
    fluid_hot: FluidModel,
    fluid_cold: FluidModel,
    Th_in: float,
    Ph_in: float,
    Tc_in: float,
    Pc_in: float,
    mdot_h_total: float,
    mdot_c_total: float,
    wall_conductivity: float,
    U_hot_override: float,
    options: MarchingOptions | None = None,
    diagnostics: dict[str, float] | None = None,
) -> np.ndarray:
    """Modified F_inboard that uses a constant U_hot instead of calculating it locally.

    This allows testing the epsilon-NTU relationship under the assumption of
    constant overall heat transfer coefficient, similar to classical HEx theory.

    Parameters
    ----------
    U_hot_override : float
        The constant overall heat transfer coefficient (W/m²·K) to use throughout
        the heat exchanger instead of calculating it locally from correlations.
    
    All other parameters match F_inboard.
    
    Returns
    -------
    np.ndarray
        [temperature_residual, pressure_residual] at the hot inlet boundary.
    """
    opts = options or MarchingOptions()
    geom_cache = _compute_geometry_arrays(geometry)

    tube_inner_diam = geometry.tube_inner_diam()

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
    Ph[0] = Ph_out_guess
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

        Re_h_od = G_h * geometry.tube_outer_diam / mu_h
        Re_c = G_c * tube_inner_diam / mu_c

        # Still need friction factors for pressure drop
        Nu_c = circular_pipe_nusselt(Re_c, 0, prandtl=Pr_c)
        f_c = circular_pipe_friction_factor(Re_c, 0)

        Nu_h, f_h = tube_bank_nusselt_number_and_friction_factor(
            Re_h_od,
            geometry.tube_spacing_long,
            geometry.tube_spacing_trv,
            Pr_h,
            inline=(not geometry.staggered),
            n_rows=geometry.n_rows_per_header * geometry.n_headers,
        )

        # KEY DIFFERENCE: Use constant U_hot instead of calculating from correlations
        U_hot = U_hot_override
        UA_sum += U_hot * geom_cache.area_ht_hot[j]

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
    residual_P = Ph_in_calc - Ph_in

    if diagnostics is not None:
        total_area_ht_hot = np.sum(geom_cache.area_ht_hot)
        total_UA = UA_sum
        diagnostics["total_UA"] = float(total_UA)
        diagnostics["total_area_ht_hot"] = float(total_area_ht_hot)
        diagnostics["U_hot_used"] = float(U_hot_override)
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
            Ph_in,
            Tc_in,
            Pc_in,
            fluid_hot,
            fluid_cold,
            mdot_h_total,
            mdot_c_total,
            geom_cache,
            geometry.n_headers,
        )

    return np.array([residual_T, residual_P])


def solve_hex_with_constant_U(
    geometry: RadialInvoluteGeometry,
    fluid_hot: FluidModel,
    fluid_cold: FluidModel,
    Th_in: float,
    Ph_in: float,
    Tc_in: float,
    Pc_in: float,
    mdot_h_total: float,
    mdot_c_total: float,
    wall_conductivity: float,
    U_hot_override: float,
    initial_guess: tuple[float, float],
    tol_T: float = 1e-2,
    tol_P: float = 100.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve the HEx with a constant U value.

    Parameters
    ----------
    U_hot_override : float
        Constant overall heat transfer coefficient (W/m²·K).
    initial_guess : tuple[float, float]
        Initial guess for (Th_out, Ph_out).
    tol_T : float
        Absolute temperature tolerance (K).
    tol_P : float
        Absolute pressure tolerance (Pa).

    Returns
    -------
    solution : np.ndarray
        Converged [Th_out, Ph_out].
    diagnostics : dict[str, float]
        Performance metrics.
    """
    eval_count = {"count": 0}
    tol_root = 1e-2

    def residuals(x: np.ndarray) -> np.ndarray:
        eval_count["count"] += 1
        raw = F_inboard_constant_U(
            Th_out_guess=x[0],
            Ph_out_guess=x[1],
            geometry=geometry,
            fluid_hot=fluid_hot,
            fluid_cold=fluid_cold,
            Th_in=Th_in,
            Ph_in=Ph_in,
            Tc_in=Tc_in,
            Pc_in=Pc_in,
            mdot_h_total=mdot_h_total,
            mdot_c_total=mdot_c_total,
            wall_conductivity=wall_conductivity,
            U_hot_override=U_hot_override,
            options=MarchingOptions(),
        )
        # Scale pressure residual to match temperature tolerance scale
        scaled = np.array([raw[0], raw[1] * (tol_root / tol_P)], dtype=float)
        return scaled

    x0 = np.array(initial_guess, dtype=float)
    sol = root(residuals, x0, method="hybr", tol=tol_root, options={"maxfev": 40})

    # Get final diagnostics
    final_diag: dict[str, float] = {}
    F_inboard_constant_U(
        Th_out_guess=sol.x[0],
        Ph_out_guess=sol.x[1],
        geometry=geometry,
        fluid_hot=fluid_hot,
        fluid_cold=fluid_cold,
        Th_in=Th_in,
        Ph_in=Ph_in,
        Tc_in=Tc_in,
        Pc_in=Pc_in,
        mdot_h_total=mdot_h_total,
        mdot_c_total=mdot_c_total,
        wall_conductivity=wall_conductivity,
        U_hot_override=U_hot_override,
        options=MarchingOptions(),
        diagnostics=final_diag,
    )

    logger.info(
        "  Constant-U solver: %d iterations, Th_out=%.2f K, Ph_out=%.2e Pa",
        eval_count["count"],
        sol.x[0],
        sol.x[1],
    )

    return sol.x, final_diag


def main() -> None:
    """Main analysis routine."""
    configure_logging(logging.INFO)
    logging.getLogger("heat_exchanger.conservation").setLevel(logging.WARNING)
    logging.getLogger("heat_exchanger.involute_inboard").setLevel(logging.WARNING)
    logging.getLogger("__main__").setLevel(logging.INFO)

    # ========================================================================
    # STEP 1: Baseline calculation with standard F_inboard (viper case)
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: Baseline calculation with viper case (PerfectGas)")
    logger.info("=" * 80)

    # Load viper case parameters
    air = PerfectGasFluid.from_name("Air")
    helium = PerfectGasFluid.from_name("Helium")

    geom_baseline = RadialInvoluteGeometry(
        tube_outer_diam=0.98e-3,
        tube_thick=0.04e-3,
        tube_spacing_trv=2.5,
        tube_spacing_long=1.1,
        staggered=True,
        n_headers=31,
        n_rows_per_header=4,
        n_rows_axial=200,
        radius_outer_whole_hex=478e-3,
        inv_angle_deg=360.0,
    )

    Th_in = 298.0
    Ph_in = 1.02e5
    Tc_in = 96.0
    Pc_in = 150e5
    mflow_h_total = 12.26
    mflow_c_total = 1.945

    # Get baseline solution with standard solver
    from comp_withZeli_involute_cases import _zero_d_two_step_guess

    Th0, Ph0 = _zero_d_two_step_guess(
        geom_baseline,
        air,
        helium,
        mflow_h_total,
        mflow_c_total,
        Th_in,
        Ph_in,
        Tc_in,
        Pc_in,
        wall_k=WALL_CONDUCTIVITY_304_SS,
    )

    eval_state = {"count": 0}
    tol_root = 1.0e-2
    tol_P = 0.001 * Ph_in  # 0.1% of inlet pressure

    def residuals_baseline(x: np.ndarray) -> np.ndarray:
        eval_state["count"] += 1
        raw = F_inboard(
            Th_out_guess=x[0],
            Ph_out_guess=x[1],
            geometry=geom_baseline,
            fluid_hot=air,
            fluid_cold=helium,
            Th_in=Th_in,
            Ph_in=Ph_in,
            Tc_in=Tc_in,
            Pc_in=Pc_in,
            mdot_h_total=mflow_h_total,
            mdot_c_total=mflow_c_total,
            wall_conductivity=WALL_CONDUCTIVITY_304_SS,
            options=MarchingOptions(),
        )
        scaled = np.array([raw[0], raw[1] * (tol_root / tol_P)], dtype=float)
        return scaled

    x0 = np.array([Th0, Ph0], dtype=float)
    sol_baseline = root(residuals_baseline, x0, method="hybr", tol=tol_root, options={"maxfev": 40})

    # Get baseline diagnostics
    diag_baseline: dict[str, float] = {}
    F_inboard(
        Th_out_guess=sol_baseline.x[0],
        Ph_out_guess=sol_baseline.x[1],
        geometry=geom_baseline,
        fluid_hot=air,
        fluid_cold=helium,
        Th_in=Th_in,
        Ph_in=Ph_in,
        Tc_in=Tc_in,
        Pc_in=Pc_in,
        mdot_h_total=mflow_h_total,
        mdot_c_total=mflow_c_total,
        wall_conductivity=WALL_CONDUCTIVITY_304_SS,
        options=MarchingOptions(),
        diagnostics=diag_baseline,
    )

    logger.info("Baseline solution converged in %d iterations", eval_state["count"])
    logger.info("  Th_out = %.2f K, Ph_out = %.2e Pa", sol_baseline.x[0], sol_baseline.x[1])
    logger.info("  epsilon = %.4f, NTU = %.3f, Cr = %.4f", 
                diag_baseline["epsilon"], diag_baseline["NTU"], diag_baseline["Cr"])
    logger.info("  Q_total = %.3f MW", diag_baseline["Q_total"] / 1e6)

    # Extract U_avg
    U_avg_baseline = diag_baseline["total_UA"] / diag_baseline["total_area_ht_hot"]
    logger.info("  UA_total = %.3e W/K", diag_baseline["total_UA"])
    logger.info("  A_total = %.3f m²", diag_baseline["total_area_ht_hot"])
    logger.info("  U_avg = %.2f W/(m²·K)", U_avg_baseline)

    # ========================================================================
    # STEP 2: Re-solve with constant U
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Re-solve with constant U = %.2f W/(m²·K)", U_avg_baseline)
    logger.info("=" * 80)

    sol_const_U, diag_const_U = solve_hex_with_constant_U(
        geometry=geom_baseline,
        fluid_hot=air,
        fluid_cold=helium,
        Th_in=Th_in,
        Ph_in=Ph_in,
        Tc_in=Tc_in,
        Pc_in=Pc_in,
        mdot_h_total=mflow_h_total,
        mdot_c_total=mflow_c_total,
        wall_conductivity=WALL_CONDUCTIVITY_304_SS,
        U_hot_override=U_avg_baseline,
        initial_guess=(sol_baseline.x[0], sol_baseline.x[1]),
        tol_P=tol_P,
    )

    logger.info("Constant-U solution:")
    logger.info("  Th_out = %.2f K, Ph_out = %.2e Pa", sol_const_U[0], sol_const_U[1])
    logger.info("  epsilon = %.4f, NTU = %.3f, Cr = %.4f",
                diag_const_U["epsilon"], diag_const_U["NTU"], diag_const_U["Cr"])
    logger.info("  Q_total = %.3f MW", diag_const_U["Q_total"] / 1e6)
    
    logger.info("\nComparison (Baseline vs Constant-U):")
    logger.info("  Δepsilon = %.4f (%.2f%%)",
                diag_const_U["epsilon"] - diag_baseline["epsilon"],
                100 * (diag_const_U["epsilon"] - diag_baseline["epsilon"]) / diag_baseline["epsilon"])
    logger.info("  ΔQ = %.3f kW (%.2f%%)",
                (diag_const_U["Q_total"] - diag_baseline["Q_total"]) / 1e3,
                100 * (diag_const_U["Q_total"] - diag_baseline["Q_total"]) / diag_baseline["Q_total"])

    # ========================================================================
    # STEP 3: Parametric study
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Parametric study with constant-U model")
    logger.info("=" * 80)

    # Define parameter variations
    mdot_h_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    mdot_c_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    radius_factors = [0.9, 0.95, 1.0, 1.05, 1.1]

    # Study 1: Vary hot mass flow rate
    logger.info("\n--- Study 1: Varying hot-side mass flow rate ---")
    logger.info("mdot_h_factor, mdot_h [kg/s], epsilon, NTU, Cr, Q [MW], U_avg [W/m²K]")
    
    for factor in mdot_h_factors:
        mdot_h = mflow_h_total * factor
        
        # Need new initial guess - use baseline as starting point
        sol, diag = solve_hex_with_constant_U(
            geometry=geom_baseline,
            fluid_hot=air,
            fluid_cold=helium,
            Th_in=Th_in,
            Ph_in=Ph_in,
            Tc_in=Tc_in,
            Pc_in=Pc_in,
            mdot_h_total=mdot_h,
            mdot_c_total=mflow_c_total,
            wall_conductivity=WALL_CONDUCTIVITY_304_SS,
            U_hot_override=U_avg_baseline,
            initial_guess=(sol_baseline.x[0], sol_baseline.x[1]),
            tol_P=tol_P,
        )
        
        logger.info("  %.2f, %.3f, %.4f, %.3f, %.4f, %.3f, %.2f",
                    factor, mdot_h, diag["epsilon"], diag["NTU"], diag["Cr"],
                    diag["Q_total"] / 1e6, diag["U_hot_used"])

    # Study 2: Vary cold mass flow rate
    logger.info("\n--- Study 2: Varying cold-side mass flow rate ---")
    logger.info("mdot_c_factor, mdot_c [kg/s], epsilon, NTU, Cr, Q [MW], U_avg [W/m²K]")
    
    for factor in mdot_c_factors:
        mdot_c = mflow_c_total * factor
        
        sol, diag = solve_hex_with_constant_U(
            geometry=geom_baseline,
            fluid_hot=air,
            fluid_cold=helium,
            Th_in=Th_in,
            Ph_in=Ph_in,
            Tc_in=Tc_in,
            Pc_in=Pc_in,
            mdot_h_total=mflow_h_total,
            mdot_c_total=mdot_c,
            wall_conductivity=WALL_CONDUCTIVITY_304_SS,
            U_hot_override=U_avg_baseline,
            initial_guess=(sol_baseline.x[0], sol_baseline.x[1]),
            tol_P=tol_P,
        )
        
        logger.info("  %.2f, %.3f, %.4f, %.3f, %.4f, %.3f, %.2f",
                    factor, mdot_c, diag["epsilon"], diag["NTU"], diag["Cr"],
                    diag["Q_total"] / 1e6, diag["U_hot_used"])

    # Study 3: Vary outer radius (affects geometry)
    logger.info("\n--- Study 3: Varying outer radius ---")
    logger.info("radius_factor, R_outer [mm], epsilon, NTU, Cr, Q [MW], U_avg [W/m²K], A_total [m²]")
    
    for factor in radius_factors:
        geom_varied = RadialInvoluteGeometry(
            tube_outer_diam=geom_baseline.tube_outer_diam,
            tube_thick=geom_baseline.tube_thick,
            tube_spacing_trv=geom_baseline.tube_spacing_trv,
            tube_spacing_long=geom_baseline.tube_spacing_long,
            staggered=geom_baseline.staggered,
            n_headers=geom_baseline.n_headers,
            n_rows_per_header=geom_baseline.n_rows_per_header,
            n_rows_axial=geom_baseline.n_rows_axial,
            radius_outer_whole_hex=geom_baseline.radius_outer_whole_hex * factor,
            inv_angle_deg=geom_baseline.inv_angle_deg,
        )
        
        sol, diag = solve_hex_with_constant_U(
            geometry=geom_varied,
            fluid_hot=air,
            fluid_cold=helium,
            Th_in=Th_in,
            Ph_in=Ph_in,
            Tc_in=Tc_in,
            Pc_in=Pc_in,
            mdot_h_total=mflow_h_total,
            mdot_c_total=mflow_c_total,
            wall_conductivity=WALL_CONDUCTIVITY_304_SS,
            U_hot_override=U_avg_baseline,
            initial_guess=(sol_baseline.x[0], sol_baseline.x[1]),
            tol_P=tol_P,
        )
        
        logger.info("  %.2f, %.2f, %.4f, %.3f, %.4f, %.3f, %.2f, %.3f",
                    factor, geom_varied.radius_outer_whole_hex * 1e3,
                    diag["epsilon"], diag["NTU"], diag["Cr"],
                    diag["Q_total"] / 1e6, diag["U_hot_used"],
                    diag["total_area_ht_hot"])

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

