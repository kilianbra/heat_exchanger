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
    tube_spacing_trv: float  # non-dimensional spacing ratio (pitch / diameter)
    tube_spacing_long: float  # non-dimensional spacing ratio (pitch / diameter)
    staggered: bool
    n_headers: int
    n_rows_per_header: int
    n_rows_axial: int
    radius_outer_whole_hex: float
    inv_angle_deg: float = 360.0
    rectangular: bool = False

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
class _GeometryCache:
    """Pre-computed geometry arrays for the involute layout."""

    radii: np.ndarray
    theta: np.ndarray
    tube_length: np.ndarray
    area_ht_hot: np.ndarray
    area_ht_cold: np.ndarray
    area_frontal_hot: np.ndarray
    area_free_hot: np.ndarray
    area_free_cold: np.ndarray
    dh_hot: np.ndarray
    # ruff: noqa: N815 (allow mixed case variables)
    dR: float


def _compute_geometry_arrays(geom: RadialInvoluteGeometry) -> _GeometryCache:
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
    dh_hot = np.zeros(n_layers)

    if geom.rectangular:
        frontal_scale = geom.frontal_area_outer() / geom.n_headers

    # Free-area ratios (global) depend only on spacing
    sigma_hot = (spacing_trv - geom.tube_outer_diam) / spacing_trv
    if geom.staggered:
        diag_spacing = np.sqrt(spacing_long**2 + (0.5 * spacing_trv) ** 2)
        sigma_hot = min(sigma_hot, 2.0 * (diag_spacing - geom.tube_outer_diam) / spacing_trv)
    # sigma_cold = np.pi * tube_inner_diam**2 / (4.0 * spacing_trv * spacing_long)

    Lf = geom.n_rows_per_header * spacing_long

    for j in range(n_layers):
        if geom.rectangular:
            tube_length[j] = axial_length
            area_frontal_hot[j] = frontal_scale
        else:
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
        dh_hot[j] = (4.0 * area_free_hot[j] * Lf) / area_ht_hot[j]

    return _GeometryCache(
        radii=radii,
        theta=theta,
        tube_length=tube_length,
        area_ht_hot=area_ht_hot,
        area_ht_cold=area_ht_cold,
        area_frontal_hot=area_frontal_hot,
        area_free_hot=area_free_hot,
        area_free_cold=area_free_cold,
        dh_hot=dh_hot,
        dR=dR,
    )


def F_inboard(
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
    options: MarchingOptions | None = None,
    diagnostics: dict[str, float] | None = None,
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
    - [1] = pressure residual Ph_in_calc - Ph_in
    """

    opts = options or MarchingOptions()
    geom_cache = _compute_geometry_arrays(geometry)

    tube_inner_diam = geometry.tube_inner_diam()
    # axial_length = geometry.axial_length()
    # radius_inner = geometry.radius_inner_whole_hex()
    # hx_area = np.pi * (geometry.radius_outer_whole_hex**2 - radius_inner**2)
    # if geometry.rectangular:
    #    hx_area = geom_cache.dR * geometry.radius_outer_whole_hex

    m_dot_hot = mdot_h_total / geometry.n_headers
    m_dot_cold = mdot_c_total / geometry.n_headers

    n_nodes = geom_cache.radii.size
    Th = np.zeros(n_nodes)
    Ph = np.zeros(n_nodes)
    Tc = np.zeros(n_nodes)
    Pc = np.zeros(n_nodes)
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

        # Re_h = G_h * geom_cache.dh_hot[j] / max(mu_h, 1e-16)
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
    residual_P = Ph_in_calc - Ph_in

    if diagnostics is not None:
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

    return np.array([residual_T, residual_P])


__all__ = [
    "RadialInvoluteGeometry",
    "MarchingOptions",
    "F_inboard",
]
