"""AHJE rectangular recuperator model using the new tube_bank_normal geometry and solver."""

import logging

import numpy as np

from fluid_files.H2_recirc import calculate_recirc_fraction_coolprop
from heat_exchanger.fluids.protocols import CoolPropFluid, FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.tube_bank_normal import TubeBankNormalSpec, tube_bank_normal_0d_solver
from heat_exchanger.logging_utils import configure_logging

# Configure logging
configure_logging(logging.INFO)
logger = logging.getLogger(__name__)


def solve_ahje_recuperator(
    n_rows: int,
    n_passes_cold: int = 1,
    mdot_hot: float = 64.316,
    mdot_cold: float = 0.427 * (1 + 0.787),
    model: str = "PG",
    temp_hot_in: float = 574.0,
    temp_cold_in: float = 275.0,
    p_hot_in: float = 0.368e5,
    p_cold_in: float = 25e5,
    total_diameter_outer: float = 2 * 0.871,
    total_diameter_inner: float = 2 * 0.541,
    spacing_trans: float = 2.5,
    spacing_long: float = 1.5,
    tube_diameter_outer: float = 1.0e-3 + 2 * 0.040e-3,
    t_tubes: float = 0.040e-3,
    tube_bank_correction_factor_hot: float = 1.0,
    p_cold_MTO: float | None = 70e5,
    p_hot_MTO: float | None = 1.26e5,
    T_hot_MTO: float | None = 718.0,
    T_base: float | None = 290.0,
) -> dict[str, object]:
    """Solve the AHJE rectangular recuperator for given geometry parameters.

    Parameters
    ----------
    n_rows : int
        Number of rows in the tube bank.
    n_passes_cold : int, optional
        Number of passes for cold fluid (default: 1).
    mdot_hot : float, optional
        Hot fluid mass flow rate in kg/s (default: 64.316).
    mdot_cold : float, optional
        Cold fluid mass flow rate in kg/s (default: 0.427 * (1 + 0.787)).
    model : str, optional
        Fluid model type: "CP", "PG", or "RP" (default: "PG").
    temp_hot_in : float, optional
        Hot inlet temperature in K (default: 574.0).
    temp_cold_in : float, optional
        Cold inlet temperature in K (default: 275.0).
    p_hot_in : float, optional
        Hot inlet pressure in Pa (default: 0.368e5).
    p_cold_in : float, optional
        Cold inlet pressure in Pa (default: 25e5).
    total_diameter_outer : float, optional
        Outer diameter of the annulus in m (default: 2 * 0.871).
    total_diameter_inner : float, optional
        Inner diameter of the annulus in m (default: 2 * 0.541).
    spacing_trans : float, optional
        Transverse spacing ratio (default: 2.5).
    spacing_long : float, optional
        Longitudinal spacing ratio (default: 1.5).
    tube_diameter_outer : float, optional
        Outer diameter of tubes in m (default: 1.0e-3 + 2 * 0.040e-3).
    t_tubes : float, optional
        Tube wall thickness in m (default: 0.040e-3).
    tube_bank_correction_factor_hot : float, optional
        Correction factor for hot-side tube bank correlations (default: 1.0).
    p_cold_MTO : float | None, optional
        Cold-side pressure at MTO conditions in Pa (default: 70e5).
    p_hot_MTO : float | None, optional
        Hot-side pressure at MTO conditions in Pa (default: 1.26e5).
    T_hot_MTO : float | None, optional
        Hot-side temperature at MTO conditions in K (default: 718.0).
    T_base : float | None, optional
        Base temperature for thermal expansion in K (default: 290.0).

    Returns
    -------
    dict
        Dictionary containing results from tube_bank_normal_0d_solver plus additional
        diagnostics like recirc_fraction.
    """
    # Fluid model selection
    match model:
        case "RP":
            # Need to implement RefProp versions
            raise NotImplementedError("RP model not yet implemented")
        case "CP":
            hot_air = CoolPropFluid("air")
            cold_hydrogen = CoolPropFluid("para_h2")
        case "PG":
            hot_air = PerfectGasFluid.from_name("h2comb_ahje")
            cold_hydrogen = PerfectGasFluid.from_name("para_h2")

    # Calculate number of tubes per row
    n_tubes_per_row = round(
        np.pi * total_diameter_inner**2 / (spacing_trans * tube_diameter_outer)
    )

    n_rows_per_pass = int(n_rows / n_passes_cold)

    # Calculate frontal area
    area_frontal = np.pi * (total_diameter_outer**2 - total_diameter_inner**2) / 4

    # Create geometry object
    geom = TubeBankNormalSpec(
        tube_outer_diam=tube_diameter_outer,
        tube_thick=t_tubes,
        tube_spacing_trv=spacing_trans,
        tube_spacing_long=spacing_long,
        staggered=False,  # inline
        n_rows_per_pass=n_rows_per_pass,
        n_passes=n_passes_cold,
        n_tubes_per_row=n_tubes_per_row,
        frontal_area_outer=area_frontal,
        annular_not_box=True,
        total_diameter_inner=total_diameter_inner,
        total_diameter_outer=total_diameter_outer,
    )

    logger.info(
        "Geometry: n_rows=%d, n_passes=%d, n_tubes_per_row=%d, axial_length=%.2f m, n_tubes_total=%d",
        n_rows,
        n_passes_cold,
        n_tubes_per_row,
        geom.axial_length,
        geom.n_tubes_total,
    )

    # Create fluid inputs
    f_in = FluidInputs(
        hot=hot_air,
        cold=cold_hydrogen,
        m_dot_hot=mdot_hot,
        m_dot_cold=mdot_cold,
        Th_in=temp_hot_in,
        Ph_in=p_hot_in,
        Tc_in=temp_cold_in,
        Pc_in=p_cold_in,
    )

    # Material properties for wall calculations
    sigma_yield_wall = 205e6  # Pa - 304 Stainless Steel
    thermal_expansion_coefficient_wall = 16e-6  # K^-1
    rho_wall = 7930  # kg/m^3 - 304 Stainless Steel

    # Solve using 0D solver
    result = tube_bank_normal_0d_solver(
        geom=geom,
        f_in=f_in,
        tube_bank_correction_factor_hot=tube_bank_correction_factor_hot,
        p_cold_MTO=p_cold_MTO,
        p_hot_MTO=p_hot_MTO,
        T_hot_MTO=T_hot_MTO,
        T_base=T_base,
        sigma_yield_wall=sigma_yield_wall,
        thermal_expansion_coefficient_wall=thermal_expansion_coefficient_wall,
        rho_wall=rho_wall,
    )

    # Calculate recirculation fraction
    recirc_fraction = calculate_recirc_fraction_coolprop(
        40, temp_cold_in, result["Tc_out"], p_cold_in / 1e5
    )
    result["recirc_fraction"] = recirc_fraction[0]

    logger.info("Recirculation fraction: %.2f", recirc_fraction[0])

    # Return result with geometry for convenience
    return {"result": result, "geometry": geom}


if __name__ == "__main__":
    # Default parameters
    n_rows_default = 6
    n_passes_cold_default = 1

    # Run with default parameters
    output = solve_ahje_recuperator(
        n_rows=n_rows_default,
        n_passes_cold=n_passes_cold_default,
    )
    result = output["result"]
    geom = output["geometry"]

    logger.info("=" * 60)
    logger.info("Results summary:")
    logger.info("Effectiveness: %.2f%%", result["epsilon"] * 100)
    logger.info("Hot pressure drop: %.2f%%", result["dP_hot_pct"])
    logger.info("Axial length: %.2f m", geom.axial_length)
    logger.info("Outer heat transfer area: %.2f m²", geom.area_heat_transfer_outer_total)

    # # Future code: Loop through different n_rows values
    # logger.info("=" * 60)
    # logger.info("Sweeping n_rows from 1 to 10:")
    # logger.info("n_rows | Effectiveness [%%] | dP_hot [%%] | Axial length [m] | A_ht_hot [m²]")
    # logger.info("-" * 70)
    #
    # for n_rows in range(1, 11):
    #     output = solve_ahje_recuperator(n_rows=n_rows, n_passes_cold=1)
    #     result = output["result"]
    #     geom = output["geometry"]
    #     logger.info(
    #         "%6d | %15.2f | %11.2f | %15.2f | %12.2f",
    #         n_rows,
    #         result["epsilon"] * 100,
    #         result["dP_hot_pct"],
    #         geom.axial_length,
    #         geom.area_heat_transfer_outer_total,
    #     )
