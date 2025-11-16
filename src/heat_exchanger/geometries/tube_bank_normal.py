"""Normal (straight) tube bank heat-exchanger geometry protocol.
Provides 0D cached properties and a 0D solver for crossflow heat exchanger analysis.
Can handle both annular and box geometries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np

from heat_exchanger.correlations import (
    circular_pipe_friction_factor as _circ_fric,
    circular_pipe_nusselt as _circ_nu,
    tube_bank_nusselt_number_and_friction_factor as _bank_corr,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu as _eps_ntu
from heat_exchanger.hex_basic import dp_tube_bank, ntu as _ntu_func
from heat_exchanger.fluids.protocols import FluidInputsProtocol as _FluidInputs

logger = logging.getLogger(__name__)

WALL_CONDUCTIVITY_304_SS = 14.0
WALL_DENSITY_304_SS = 7930.0


class TubeBankCorrelationGeometry(Protocol):
    """Minimal protocol for tube-bank correlations and 0D metrics."""

    tube_outer_diam: float
    tube_thick: float
    tube_spacing_trv: float
    tube_spacing_long: float
    staggered: bool
    n_tubes_per_row: int
    n_tubes_total: int
    n_rows_total: int

    @property
    def sigma_outer(self) -> float: ...

    @property
    def tube_inner_diam(self) -> float: ...


class TubeBankNormalProtocol(TubeBankCorrelationGeometry, Protocol):
    """Protocol describing a normal (straight) tube bank heat exchanger geometry.
    Can be either annular (involute) or box-shaped.

    Required inputs:
      - tube_outer_diam, tube_thick
      - tube_spacing_trv, tube_spacing_long (non-dimensional spacing ratios)
      - staggered (True for staggered, False for inline)
      - n_rows_per_pass, n_passes, n_tubes_per_row
      - frontal_area_outer
      - annular_not_box (True for annular/involute, False for box)

    0D analysis uses the cached-style properties below and the tube_bank_normal_0d_solver function.
    """

    # Core, non-cached inputs (implementers provide these as attributes)
    tube_outer_diam: float
    tube_thick: float
    tube_spacing_trv: float  # non-dimensional spacing ratio (Xt*)
    tube_spacing_long: float  # non-dimensional spacing ratio (Xl*)
    staggered: bool
    n_rows_per_pass: int
    n_passes: int
    n_tubes_per_row: int
    frontal_area_outer: float
    annular_not_box: bool = True
    wall_conductivity: float = WALL_CONDUCTIVITY_304_SS
    total_diameter_inner: float | None = None  # If provided, used directly; otherwise derived from row_width
    total_diameter_outer: float | None = None  # If provided, used directly; otherwise derived from frontal_area_outer

    # ---------- Cached-style 0D properties (default implementations) ----------
    @cached_property
    def tube_inner_diam(self) -> float:
        return self.tube_outer_diam - 2.0 * self.tube_thick

    @cached_property
    def row_width(self) -> float:
        """Width of one row of tubes. If annular, this is pi D_i. If box this is box width."""
        return self.tube_spacing_trv * self.tube_outer_diam * self.n_tubes_per_row

    @cached_property
    def tube_length(self) -> float:
        """Length of tubes. For annular, uses involute formula. For box, uses frontal area."""
        if self.annular_not_box:
            # Use total_diameter_inner and total_diameter_outer if provided, otherwise derive
            if self.total_diameter_inner is not None and self.total_diameter_outer is not None:
                D_i = self.total_diameter_inner
                D_o = self.total_diameter_outer
            elif self.total_diameter_inner is not None:
                D_i = self.total_diameter_inner
                D_o = np.sqrt(D_i**2 + self.frontal_area_outer / np.pi)
            else:
                D_i = self.row_width / np.pi
                D_o = np.sqrt(D_i**2 + self.frontal_area_outer / np.pi)
            return D_i / 4 * ((D_o / D_i) ** 2 - 1)
        else:
            W = self.row_width
            return self.frontal_area_outer / W

    @cached_property
    def passage_height(self) -> float:
        """Height of the passage. For annular, this is (D_o - D_i)/2. For box, same as tube_length."""
        if self.annular_not_box:
            # Use total_diameter_inner and total_diameter_outer if provided, otherwise derive
            if self.total_diameter_inner is not None and self.total_diameter_outer is not None:
                D_i = self.total_diameter_inner
                D_o = self.total_diameter_outer
            elif self.total_diameter_inner is not None:
                D_i = self.total_diameter_inner
                D_o = np.sqrt(D_i**2 + self.frontal_area_outer / np.pi)
            else:
                D_i = self.row_width / np.pi
                D_o = np.sqrt(D_i**2 + self.frontal_area_outer / np.pi)
            return (D_o - D_i) / 2
        else:  # It is the same as tube length for box
            return self.tube_length

    @cached_property
    def axial_length(self) -> float:
        """Axial length of the heat exchanger."""
        return self.tube_spacing_long * self.tube_outer_diam * self.n_rows_per_pass * self.n_passes

    @cached_property
    def sigma_outer(self) -> float:
        """Free-area ratio for the hot-side external crossflow."""
        interim = (self.tube_spacing_trv - 1) / self.tube_spacing_trv
        if self.staggered:
            diag_spacing = np.sqrt(self.tube_spacing_long**2 + (0.5 * self.tube_spacing_trv) ** 2)
            interim = min(interim, 2.0 * (diag_spacing - 1) / self.tube_spacing_trv)
        return interim

    @cached_property
    def area_free_flow_outer(self) -> float:
        """Free flow area for hot fluid (external to tubes)."""
        return self.frontal_area_outer * self.sigma_outer

    @cached_property
    def n_rows_total(self) -> int:
        """Total number of rows."""
        return self.n_rows_per_pass * self.n_passes

    @cached_property
    def n_tubes_total(self) -> int:
        """Total number of tubes."""
        return self.n_tubes_per_row * self.n_rows_total

    @cached_property
    def n_tubes_per_pass(self) -> int:
        """Number of tubes per pass."""
        return self.n_tubes_per_row * self.n_rows_per_pass

    @cached_property
    def area_free_flow_inner(self) -> float:
        """Free flow area for cold fluid (inside tubes)."""
        return np.pi / 4 * self.tube_inner_diam**2 * self.n_tubes_per_pass

    @cached_property
    def area_heat_transfer_outer_total(self) -> float:
        """Total heat transfer area on outer (hot) side."""
        return np.pi * self.tube_outer_diam * self.tube_length * self.n_rows_total * self.n_tubes_per_row

    @cached_property
    def area_heat_transfer_inner_total(self) -> float:
        """Total heat transfer area on inner (cold) side."""
        return np.pi * self.tube_inner_diam * self.tube_length * self.n_tubes_total

    @cached_property
    def area_heat_transfer_outer_per_row(self) -> float:
        """Heat transfer area on outer side per row."""
        return np.pi * self.tube_outer_diam * self.tube_length * self.n_tubes_per_row

    @cached_property
    def area_heat_transfer_inner_per_row(self) -> float:
        """Heat transfer area on inner side per row."""
        return np.pi * self.tube_inner_diam * self.tube_length * self.n_tubes_per_row


@dataclass(frozen=True)
class TubeBankNormalSpec(TubeBankNormalProtocol):
    """Concrete container implementing the TubeBankNormalProtocol."""

    tube_outer_diam: float
    tube_thick: float
    tube_spacing_trv: float
    tube_spacing_long: float
    staggered: bool
    n_rows_per_pass: int
    n_passes: int
    n_tubes_per_row: int
    frontal_area_outer: float
    annular_not_box: bool = True
    wall_conductivity: float = WALL_CONDUCTIVITY_304_SS
    total_diameter_inner: float | None = None
    total_diameter_outer: float | None = None


def tube_bank_normal_0d_solver(
    geom: TubeBankNormalProtocol,
    f_in: _FluidInputs,
    tube_bank_correction_factor_hot: float = 1.0,
    *,
    p_cold_MTO: float | None = None,
    p_hot_MTO: float | None = None,
    T_hot_MTO: float | None = None,
    T_base: float | None = None,
    sigma_yield_wall: float = 205e6,
    thermal_expansion_coefficient_wall: float = 16e-6,
    rho_wall: float = WALL_DENSITY_304_SS,
) -> dict[str, object]:
    """Solve the normal tube bank heat exchanger using 0D analysis.

    This function performs a 0D (lumped) analysis of a crossflow tube bank heat exchanger,
    similar to the calculations in recup_model_ahje_rect.py.

    Parameters
    ----------
    geom : TubeBankNormalProtocol
        Geometry specification for the tube bank.
    f_in : FluidInputsProtocol
        Fluid inputs including hot and cold fluid states, mass flow rates, and inlet conditions.
    tube_bank_correction_factor_hot : float, optional
        Correction factor for hot-side tube bank correlations (default: 1.0).
    p_cold_MTO : float, optional
        Cold-side pressure at maximum take-off (MTO) conditions (Pa). If provided,
        thermal expansion and thickness calculations will be performed.
    p_hot_MTO : float, optional
        Hot-side pressure at MTO conditions (Pa). Defaults to cruise Ph_in if not provided.
    T_hot_MTO : float, optional
        Hot-side temperature at MTO conditions (K). Defaults to cruise Th_in if not provided.
    T_base : float, optional
        Base/reference temperature for thermal expansion calculations (K). Defaults to
        cruise Tc_in if not provided.
    sigma_yield_wall : float, optional
        Yield strength of wall material (Pa). Default: 205e6 Pa for 304 SS.
    thermal_expansion_coefficient_wall : float, optional
        Thermal expansion coefficient of wall material (K^-1). Default: 16e-6 K^-1 for 304 SS.
    rho_wall : float, optional
        Density of wall material (kg/m³). Default: 7930 kg/m³ for 304 SS.

    Returns
    -------
    dict
        Dictionary containing:
        - Th_out: Hot outlet temperature (K)
        - Tc_out: Cold outlet temperature (K)
        - Ph_out: Hot outlet pressure (Pa)
        - Pc_out: Cold outlet pressure (Pa)
        - Q_total: Total heat transfer rate (W)
        - epsilon: Effectiveness
        - NTU: Number of transfer units
        - Cr: Capacity ratio
        - dP_hot: Hot-side pressure drop (Pa)
        - dP_hot_pct: Hot-side pressure drop as percentage of inlet pressure
        - dP_cold: Cold-side pressure drop (Pa)
        - dP_cold_pct: Cold-side pressure drop as percentage of inlet pressure
        - diagnostics: Additional diagnostic information
    """
    logger = logging.getLogger(__name__ + ".tube_bank_normal_0d_solver")

    # Geometry calculations
    area_frontal = geom.frontal_area_outer
    sigma = geom.sigma_outer
    area_free_flow_hot = geom.area_free_flow_outer
    tube_length = geom.tube_length
    area_heat_transfer_hot = geom.area_heat_transfer_outer_total
    area_heat_transfer_cold = geom.area_heat_transfer_inner_total
    area_free_flow_cold = geom.area_free_flow_inner

    logger.info(
        "Geometry: axial_length=%.2f m, n_tubes_total=%d, sigma=%.3f",
        geom.axial_length,
        geom.n_tubes_total,
        sigma,
    )
    logger.info(
        "Areas: A_ht_hot=%.2f m², A_ht_cold=%.2f m², A_ff_hot=%.4f m², A_ff_cold=%.4f m²",
        area_heat_transfer_hot,
        area_heat_transfer_cold,
        area_free_flow_hot,
        area_free_flow_cold,
    )

    # Inlet states
    hot_in = f_in.hot.state(f_in.Th_in, f_in.Ph_in if f_in.Ph_in is not None else f_in.Ph_out)
    cold_in = f_in.cold.state(f_in.Tc_in, f_in.Pc_in)

    logger.info(
        "Hot inlet: Th_in=%.2f K, Ph=%.2e Pa, mu=%.2e Pa·s",
        f_in.Th_in,
        f_in.Ph_in if f_in.Ph_in is not None else f_in.Ph_out,
        hot_in.mu,
    )

    # Reynolds numbers
    reynolds_hot_in = f_in.m_dot_hot / area_free_flow_hot / hot_in.mu * geom.tube_outer_diam
    reynolds_cold_in = f_in.m_dot_cold / area_free_flow_cold / cold_in.mu * geom.tube_inner_diam

    logger.info("Reynolds numbers: Re_hot=%.2e, Re_cold=%.2e", reynolds_hot_in, reynolds_cold_in)

    # Nusselt numbers and friction factors
    Pr_hot = hot_in.mu * hot_in.cp / hot_in.k
    Pr_cold = cold_in.mu * cold_in.cp / cold_in.k

    nusselt_hot, f_hot = _bank_corr(
        reynolds_hot_in,
        geom.tube_spacing_long,
        geom.tube_spacing_trv,
        Pr_hot,
        inline=(not geom.staggered),
        n_rows=geom.n_rows_total,
    )
    f_hot = f_hot * tube_bank_correction_factor_hot
    nusselt_hot = nusselt_hot * tube_bank_correction_factor_hot  # keep j/f propto Nu/f constant

    nusselt_cold = _circ_nu(reynolds_cold_in, 0, prandtl=Pr_cold)
    f_cold = _circ_fric(reynolds_cold_in, 0)

    logger.info(
        "Nusselt numbers: Nu_hot=%.2f, Nu_cold=%.2f",
        nusselt_hot,
        nusselt_cold,
    )
    logger.info(
        "Friction factors: f_hot=%.4f, f_cold=%.4f",
        f_hot,
        f_cold,
    )

    # Stanton numbers
    stanton_hot = nusselt_hot / reynolds_hot_in / Pr_hot
    stanton_cold = nusselt_cold / reynolds_cold_in / Pr_cold

    # Heat capacity fluxes
    heat_capacity_flux_hot = f_in.m_dot_hot * hot_in.cp
    heat_capacity_flux_cold = f_in.m_dot_cold * cold_in.cp

    # Area ratios
    area_ratio_q_over_o_hot = area_heat_transfer_hot / area_free_flow_hot
    area_ratio_q_over_o_cold = area_heat_transfer_cold / area_free_flow_cold

    logger.info(
        "Area ratios (A_ht/A_ff): hot=%.2f, cold=%.2f",
        area_ratio_q_over_o_hot,
        area_ratio_q_over_o_cold,
    )

    # NTU calculation
    c_min = np.minimum(heat_capacity_flux_hot, heat_capacity_flux_cold)
    inv_h = 1 / stanton_hot / area_ratio_q_over_o_hot * c_min / heat_capacity_flux_hot
    inv_c = 1 / stanton_cold / area_ratio_q_over_o_cold * c_min / heat_capacity_flux_cold
    total_resistance = inv_h + inv_c
    ntu_value = 1 / total_resistance

    # Percentage contribution to total thermal resistance
    perc_hot = inv_h / total_resistance * 100
    perc_cold = inv_c / total_resistance * 100

    logger.info(
        "Thermal resistance contribution: hot=%.0f%%, cold=%.0f%%",
        perc_hot,
        perc_cold,
    )

    # Capacity ratio and flow type
    if heat_capacity_flux_hot > heat_capacity_flux_cold:
        c_min = heat_capacity_flux_cold
        c_ratio = heat_capacity_flux_cold / heat_capacity_flux_hot
        flow_type_description = "Cmax_mixed"
    else:
        c_min = heat_capacity_flux_hot
        c_ratio = heat_capacity_flux_hot / heat_capacity_flux_cold
        flow_type_description = "Cmin_mixed"

    # Effectiveness
    epsilon = _eps_ntu(
        ntu_value,
        c_ratio,
        exchanger_type="cross_flow",
        flow_type=flow_type_description,
        n_passes=geom.n_passes,
    )

    # Heat transfer
    heat_transfer = epsilon * c_min * (f_in.Th_in - f_in.Tc_in)
    logger.info("Heat transfer: Q=%.2f MW", heat_transfer / 1e6)

    # Outlet temperatures
    temp_hot_out = f_in.Th_in - heat_transfer / heat_capacity_flux_hot
    temp_cold_out = f_in.Tc_in + heat_transfer / heat_capacity_flux_cold

    logger.info(
        "Outlet temperatures: Th_out=%.2f K, Tc_out=%.2f K",
        temp_hot_out,
        temp_cold_out,
    )

    # Pressure drop calculations
    # Hot side
    rho_hot_in = hot_in.rho
    # Use Ph_in if available, otherwise Ph_out for property evaluation
    Ph_for_props = f_in.Ph_in if f_in.Ph_in is not None else f_in.Ph_out
    if Ph_for_props is None:
        raise ValueError("Either Ph_in or Ph_out must be provided.")
    hot_out_approx = f_in.hot.state(temp_hot_out, Ph_for_props)
    rho_hot_out_approx = hot_out_approx.rho

    mass_velocity_hot = f_in.m_dot_hot / area_free_flow_hot
    dp_hot = dp_tube_bank(
        area_ratio_q_over_o_hot,
        mass_velocity_hot,
        rho_hot_in,
        rho_hot_out_approx,
        sigma,
        f_hot,
    )

    if f_in.Ph_in is not None:
        Ph_in = f_in.Ph_in
        Ph_out = Ph_in - dp_hot
        dP_hot_pct = dp_hot / Ph_in * 100.0
    else:
        # Ph_out is given, calculate Ph_in
        Ph_out = f_in.Ph_out
        if Ph_out is None:
            raise ValueError("Either Ph_in or Ph_out must be provided.")
        Ph_in = Ph_out + dp_hot
        dP_hot_pct = dp_hot / Ph_in * 100.0

    # Cold side (simplified - using mean density)
    rho_cold_in = cold_in.rho
    cold_out_approx = f_in.cold.state(temp_cold_out, f_in.Pc_in)
    rho_cold_out_approx = cold_out_approx.rho
    one_over_rho_mean_cold = (1 / rho_cold_in + 1 / rho_cold_out_approx) / 2

    mass_velocity_cold = f_in.m_dot_cold / area_free_flow_cold
    dp_cold = (
        f_cold
        * (area_heat_transfer_cold / area_free_flow_cold)
        * (mass_velocity_cold**2)
        / 2
        * one_over_rho_mean_cold
    )

    Pc_out = f_in.Pc_in - dp_cold  # Cold side pressure decreases
    dP_cold_pct = dp_cold / f_in.Pc_in * 100.0

    logger.info(
        "Pressure drops: dP_hot=%.2f%% (%.1f Pa), dP_cold=%.2f%% (%.1f Pa)",
        dP_hot_pct,
        dp_hot,
        dP_cold_pct,
        dp_cold,
    )

    # Wall mass calculation
    wall_volume = (
        geom.n_rows_total
        * geom.n_tubes_per_row
        * np.pi
        * (geom.tube_outer_diam**2 - geom.tube_inner_diam**2)
        * tube_length
        / 4
    )
    wall_mass = wall_volume * rho_wall
    logger.info("Wall mass: %.2f kg", wall_mass)

    # Initialize diagnostics
    diagnostics = {
        "NTU": float(ntu_value),
        "epsilon": float(epsilon),
        "Cr": float(c_ratio),
        "Q_total": float(heat_transfer),
        "dP_hot": float(dp_hot),
        "dP_hot_pct": float(dP_hot_pct),
        "dP_cold": float(dp_cold),
        "dP_cold_pct": float(dP_cold_pct),
        "Re_hot": float(reynolds_hot_in),
        "Re_cold": float(reynolds_cold_in),
        "Nu_hot": float(nusselt_hot),
        "Nu_cold": float(nusselt_cold),
        "f_hot": float(f_hot),
        "f_cold": float(f_cold),
        "perc_hot_resistance": float(perc_hot),
        "perc_cold_resistance": float(perc_cold),
        "wall_mass": float(wall_mass),
    }

    # MTO conditions and thermal expansion/thickness calculations
    if p_cold_MTO is not None:
        # Use MTO conditions if provided, otherwise default to cruise conditions
        p_hot_mto = p_hot_MTO if p_hot_MTO is not None else Ph_in
        T_hot_mto = T_hot_MTO if T_hot_MTO is not None else f_in.Th_in
        T_base_mto = T_base if T_base is not None else f_in.Tc_in

        logger.info(
            "MTO conditions: p_cold_MTO=%.2e Pa, p_hot_MTO=%.2e Pa, T_hot_MTO=%.2f K, T_base=%.2f K",
            p_cold_MTO,
            p_hot_mto,
            T_hot_mto,
            T_base_mto,
        )

        # Thermal expansion calculation
        delta_T_takeoff = T_hot_mto - T_base_mto
        thermal_strain_takeoff = thermal_expansion_coefficient_wall * delta_T_takeoff
        delta_length_tube_takeoff = thermal_strain_takeoff * tube_length

        logger.info(
            "Tube axial thermal expansion from %.0f K to %.0f K: %.2f%% (ΔL ≈ %.4f m for tube length %.2f m)",
            T_base_mto,
            T_hot_mto,
            thermal_strain_takeoff * 100,
            delta_length_tube_takeoff,
            tube_length,
        )

        # Interpret axial expansion as an equivalent rotation at the inner radius
        if geom.annular_not_box:
            inner_radius_annulus = geom.row_width / (2 * np.pi)
            rotation_rad = delta_length_tube_takeoff / inner_radius_annulus
            rotation_deg = rotation_rad * 180.0 / np.pi

            logger.info(
                "If the added tube length were taken up purely by circumferential inclination at "
                "the inner radius (R_i = %.3f m), the inner radius would rotate by ≈ %.2f° (%.3f rad).",
                inner_radius_annulus,
                rotation_deg,
                rotation_rad,
            )

        # Thin-walled tube hoop stress check at take-off
        # σ_hoop ≈ p * r_i / t  =>  t_required = p * r_i / σ_yield
        delta_p_takeoff = p_cold_MTO  # Pa, external pressure neglected
        inner_radius = geom.tube_inner_diam / 2
        t_required_hoop = delta_p_takeoff * inner_radius / sigma_yield_wall

        logger.info(
            "Required wall thickness for hoop stress at %.1f bar: %.3f mm",
            delta_p_takeoff / 1e5,
            t_required_hoop * 1e3,
        )

        if geom.tube_thick >= t_required_hoop:
            logger.info(
                "Actual wall thickness %.3f mm is sufficient; margin over required thickness: %.1f%%.",
                geom.tube_thick * 1e3,
                (geom.tube_thick / t_required_hoop - 1) * 100,
            )
        else:
            logger.warning(
                "Actual wall thickness %.3f mm is INSUFFICIENT; requires at least %.3f mm.",
                geom.tube_thick * 1e3,
                t_required_hoop * 1e3,
            )

        # Add MTO diagnostics
        diagnostics["thermal_strain_takeoff"] = float(thermal_strain_takeoff)
        diagnostics["delta_length_tube_takeoff"] = float(delta_length_tube_takeoff)
        diagnostics["t_required_hoop"] = float(t_required_hoop)
        diagnostics["t_actual"] = float(geom.tube_thick)
        diagnostics["hoop_stress_margin_pct"] = float((geom.tube_thick / t_required_hoop - 1) * 100) if t_required_hoop > 0 else float("nan")

    return {
        "Th_out": float(temp_hot_out),
        "Tc_out": float(temp_cold_out),
        "Ph_out": float(Ph_out),
        "Pc_out": float(Pc_out),
        "Q_total": float(heat_transfer),
        "epsilon": float(epsilon),
        "NTU": float(ntu_value),
        "Cr": float(c_ratio),
        "dP_hot": float(dp_hot),
        "dP_hot_pct": float(dP_hot_pct),
        "dP_cold": float(dp_cold),
        "dP_cold_pct": float(dP_cold_pct),
        "diagnostics": diagnostics,
    }


__all__ = [
    "TubeBankCorrelationGeometry",
    "TubeBankNormalProtocol",
    "TubeBankNormalSpec",
    "tube_bank_normal_0d_solver",
]

