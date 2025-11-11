"""Involute heat-exchanger geometry protocol.
Provides 0D cached properties and an on-demand method to compute 1D arrays
for a single sector used in radial marching (inner radius j=0 to outer j=n_headers).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np

from heat_exchanger.conservation import update_static_properties as _upd_stat_prop
from heat_exchanger.correlations import (
    circular_pipe_friction_factor as _circ_fric,
)
from heat_exchanger.correlations import (
    circular_pipe_nusselt as _circ_nu,
)
from heat_exchanger.correlations import (
    tube_bank_nusselt_number_and_friction_factor as _bank_corr,
)
from heat_exchanger.epsilon_ntu import epsilon_ntu as _eps_ntu
from heat_exchanger.fluids.protocols import FluidInputsProtocol as _FluidInputs

logger = logging.getLogger(__name__)

WALL_CONDUCTIVITY_304_SS = 14.0
WALL_DENSITY_304_SS = 7930.0


class RadialSpiralProtocol(Protocol):
    """Protocol describing a heat-exchanger geometry where a tube bank is wrapped in a spiral
    and the flow outside the tubes is flowing radially. This produces a local crossflow
    but overall counterflow configuration.

    Required inputs:
      - tube_outer_diam, tube_thick
      - tube_spacing_trv, tube_spacing_long (non-dimensional spacing ratios)
      - staggered (True for staggered, False for inline)
      - n_headers, n_rows_per_header, n_tubes_per_row
      - radius_outer_whole_hex
      - inv_angle_deg (involute sweep angle in degrees)

    0D analysis uses the cached-style properties below. 1D analysis calls
    _1d_arrays_for_one_sector() to compute all arrays needed for stepping.
    """

    # Core, non-cached inputs (implementers provide these as attributes)
    tube_outer_diam: float
    tube_thick: float
    tube_spacing_trv: float  # non-dimensional spacing ratio (Xt*)
    tube_spacing_long: float  # non-dimensional spacing ratio (Xl*)
    staggered: bool
    n_headers: int
    n_rows_per_header: int
    n_tubes_per_row: int
    radius_outer_whole_hex: float
    inv_angle_deg: float = 360.0
    wall_conductivity: float = WALL_CONDUCTIVITY_304_SS

    # ---------- Cached-style 0D properties (default implementations) ----------
    @cached_property
    def tube_inner_diam(self) -> float:
        return self.tube_outer_diam - 2.0 * self.tube_thick

    @cached_property
    def radius_inner_whole_hex(self) -> float:
        n_rows_per_axial_section = self.n_rows_per_header * self.n_headers
        outer_radius_span = n_rows_per_axial_section * self.tube_spacing_long * self.tube_outer_diam
        if outer_radius_span >= self.radius_outer_whole_hex:
            raise ValueError(
                f"Invalid geometry: too many rows in axial section for given outer radius: "
                f"{outer_radius_span:.2f} m > {self.radius_outer_whole_hex:.2f} m for "
                f"{n_rows_per_axial_section} rows of tubes spaced by "
                f"{self.tube_spacing_long * self.tube_outer_diam:.2f} m"
            )
        elif outer_radius_span <= 0:
            raise ValueError(
                f"Invalid geometry: negative width of annulus: {outer_radius_span:.2f} m < 0 m for "
                f"{n_rows_per_axial_section} rows of tubes spaced by "
                f"{self.tube_spacing_long * self.tube_outer_diam:.2f} m"
            )
        return self.radius_outer_whole_hex - outer_radius_span

    @cached_property
    def axial_length(self) -> float:
        return self.n_tubes_per_row * self.tube_spacing_trv * self.tube_outer_diam

    @cached_property
    def frontal_area_outer(self) -> float:
        return (
            2.0
            * np.pi
            * self.radius_outer_whole_hex
            * self.n_tubes_per_row
            * self.tube_outer_diam
            * self.tube_spacing_trv
        )

    @cached_property
    def n_rows(self) -> int:
        return self.n_rows_per_header * self.n_headers

    @cached_property
    def n_tubes_total(self) -> int:
        return self.n_tubes_per_row * self.n_rows

    @cached_property
    def n_tubes_per_header(self) -> int:
        return self.n_rows_per_header * self.n_tubes_per_row

    @cached_property
    def tube_spacing_diag(self) -> float:
        return float(np.sqrt(self.tube_spacing_trv**2 + (0.5 * self.tube_spacing_long) ** 2))

    @cached_property
    def sigma_outer(self) -> float:
        """Free-area ratio for the hot-side external crossflow at the outer section.
        For staggered layouts, the controlling throat may be the diagonal.
        """
        sigma_main = (self.tube_spacing_trv - 1) / self.tube_spacing_trv
        if self.staggered:
            sigma_diag = 2.0 * (self.tube_spacing_diag - 1) / self.tube_spacing_trv
            return min(sigma_main, sigma_diag)
        return sigma_main

    # ---------- 1D arrays (computed on demand for a single hot-sector) ----------
    def _1d_arrays_for_one_sector(self) -> dict[str, np.ndarray | float]:
        """Compute and return geometry arrays of length n_headers for 1D marching in one sector.

        Returns a dict with keys:
          - radii, theta, tube_length, area_ht_hot, area_ht_cold,
            area_frontal_hot, area_free_hot, area_free_cold, d_h_hot
          - dR (float)
        """
        dR = (self.radius_outer_whole_hex - self.radius_inner_whole_hex) / self.n_headers

        radii = self.radius_inner_whole_hex + np.arange(self.n_headers + 1, dtype=float) * dR
        inv_b = (self.radius_outer_whole_hex - self.radius_inner_whole_hex) / np.deg2rad(
            self.inv_angle_deg
        )
        theta = (radii - self.radius_inner_whole_hex) / inv_b

        area_ht_hot = np.zeros(self.n_headers)
        area_ht_cold = np.zeros(self.n_headers)
        area_frontal_hot = np.zeros(self.n_headers)
        area_free_hot = np.zeros(self.n_headers)
        area_free_cold = np.zeros(self.n_headers)
        tube_length = np.zeros(self.n_headers)
        d_h_hot = np.zeros(self.n_headers)

        length_flow_outer_per_header = (
            self.n_rows_per_header * self.tube_spacing_long * self.tube_outer_diam
        )

        for j in range(self.n_headers):
            tube_length[j] = np.trapezoid(
                np.sqrt(radii[j : j + 2] ** 2 + inv_b**2),
                theta[j : j + 2],
            )
            area_frontal_hot[j] = self.axial_length * 2.0 * np.pi * radii[j] / self.n_headers

            area_ht_hot[j] = np.pi * self.tube_outer_diam * tube_length[j] * self.n_tubes_per_header
            area_ht_cold[j] = (
                np.pi * self.tube_inner_diam * tube_length[j] * self.n_tubes_per_header
            )

            area_free_hot[j] = area_frontal_hot[j] * self.sigma_outer
            # Valid for this segmentation of the HEx
            area_free_cold[j] = (
                np.pi * self.tube_inner_diam**2 / 4.0 * self.n_tubes_total / self.n_headers
            )
            d_h_hot[j] = (4.0 * area_free_hot[j] * length_flow_outer_per_header) / area_ht_hot[j]

        return {
            "radii": radii,
            "theta": theta,
            "tube_length": tube_length,
            "area_ht_hot": area_ht_hot,
            "area_ht_cold": area_ht_cold,
            "area_frontal_hot": area_frontal_hot,
            "area_free_hot": area_free_hot,
            "area_free_cold": area_free_cold,
            "d_h_hot": d_h_hot,
            "dR": float(dR),
        }


@dataclass(frozen=True)
class RadialSpiralSpec(RadialSpiralProtocol):
    """Concrete container implementing the RadialSpiralGeometry protocol."""

    tube_outer_diam: float
    tube_thick: float
    tube_spacing_trv: float
    tube_spacing_long: float
    staggered: bool
    n_headers: int
    n_rows_per_header: int
    n_tubes_per_row: int
    radius_outer_whole_hex: float
    inv_angle_deg: float = 360.0
    wall_conductivity: float = WALL_CONDUCTIVITY_304_SS


# ---------- Marching solver (inboard) ----------
def calc_eps_local_and_tau(
    *,
    geometry: RadialSpiralProtocol,
    sh,
    sc,
    G_h: float,
    G_c: float,
    area_ht_hot_j: float,
    area_ht_cold_j: float,
    area_free_hot_j: float,
    area_free_cold_j: float,
    mdot_hot_per_header: float,
    mdot_cold_per_header: float,
) -> tuple[float, float, float]:
    """Return eps_local * C_min_local and (tau_hot, tau_cold) for layer j using local states and areas."""
    cp_h = sh.cp
    mu_h = sh.mu
    k_h = sh.k
    rho_h = sh.rho
    Pr_h = mu_h * cp_h / k_h

    cp_c = sc.cp
    mu_c = sc.mu
    k_c = sc.k
    rho_c = sc.rho
    Pr_c = mu_c * cp_c / k_c

    Re_h_od = G_h * geometry.tube_outer_diam / mu_h
    Re_c = G_c * geometry.tube_inner_diam / mu_c

    Nu_c = _circ_nu(Re_c, 0, prandtl=Pr_c)
    f_c = _circ_fric(Re_c, 0)
    h_c = Nu_c * k_c / geometry.tube_inner_diam

    Nu_h, f_h = _bank_corr(
        Re_h_od,
        geometry.tube_spacing_long,
        geometry.tube_spacing_trv,
        Pr_h,
        inline=(not geometry.staggered),
        n_rows=geometry.n_rows_per_header * geometry.n_headers,
    )
    h_h = Nu_h * k_h / geometry.tube_outer_diam

    wall_term = (
        geometry.tube_outer_diam
        / (2.0 * geometry.wall_conductivity)
        * np.log(geometry.tube_outer_diam / geometry.tube_inner_diam)
    )

    U_hot = 1.0 / (
        (1.0 / h_h)
        + (1.0 / h_c) * (geometry.tube_outer_diam / geometry.tube_inner_diam)
        + wall_term
    )

    C_h_j = mdot_hot_per_header * cp_h
    C_c_j = mdot_cold_per_header * cp_c
    C_min_j = min(C_h_j, C_c_j)
    C_max_j = max(C_h_j, C_c_j)
    Cr_j = C_min_j / C_max_j if C_max_j > 0 else 0.0

    NTU_local = U_hot * area_ht_hot_j / C_min_j if C_min_j > 0 else 0.0
    flow_type = "Cmax_mixed" if C_h_j > C_c_j else "Cmin_mixed"
    eps_local = _eps_ntu(
        NTU_local,
        Cr_j,
        exchanger_type="cross_flow",
        flow_type=flow_type,
        n_passes=1,
    )

    eps_C_min_local = eps_local * C_min_j

    tau_hot = f_h * (area_ht_hot_j / area_free_hot_j) * (G_h**2) / (2.0 * rho_h)
    tau_cold = f_c * (area_ht_cold_j / area_free_cold_j) * (G_c**2) / (2.0 * rho_c)

    return float(eps_C_min_local), float(tau_hot), float(tau_cold)


def rad_spiral_shoot(
    boundary_guess: np.ndarray | list[float],
    geometry: RadialSpiralProtocol,
    fluids: _FluidInputs,
    *,
    property_solver_it_max: int = 20,
    # ruff: noqa: N815 (allow mixed case variables)
    property_solver_tol_T: float = 1e-2,
    rel_tol_p: float = 1e-3,
) -> np.ndarray:
    """Inboard shooting residuals for a radial involute HX.
    For now hard coded that hot fluid flows outside tubes and cold inside.

    if fluids.Ph_in is provided:
        boundary_guess: [Th_out_guess, Ph_out_guess]
        Returns: [residual_T, residual_P]
    if fluids.Ph_out is provided:
        boundary_guess: [Th_out_guess]
        Returns: [residual_T]
    """
    has_Ph_in = fluids.Ph_in is not None
    has_Ph_out = fluids.Ph_out is not None
    if has_Ph_in == has_Ph_out:
        raise ValueError("Specify exactly one of Ph_in or Ph_out in FluidInputs.")

    x = np.asarray(boundary_guess, dtype=float)
    if has_Ph_in:
        if x.size != 2:
            raise ValueError(
                "boundary_guess must be [Th_out_guess, Ph_out_guess] when Ph_in is given."
            )
        Th_out_guess = float(x[0])
        Ph_out_guess = float(x[1])
    else:
        if x.size != 1:
            raise ValueError("boundary_guess must be [Th_out_guess] when Ph_out is given.")
        Th_out_guess = float(x[0])
        Ph_out_guess = float("nan")  # unused

    # Geometry arrays for one sector
    geo = geometry._1d_arrays_for_one_sector()
    area_ht_hot_of_sector = geo["area_ht_hot"]
    area_ht_cold_of_sector = geo["area_ht_cold"]
    area_free_hot_of_sector = geo["area_free_hot"]
    area_free_cold_of_sector = geo["area_free_cold"]

    # Per-header mass flows
    mdot_hot_per_header = fluids.m_dot_hot / geometry.n_headers
    mdot_cold_per_header = fluids.m_dot_cold / geometry.n_headers

    Th = np.zeros(geometry.n_headers)
    Ph = np.zeros(geometry.n_headers)
    Tc = np.zeros(geometry.n_headers)
    Pc = np.zeros(geometry.n_headers)

    # Inner boundary (start of outward march)
    Th[0] = Th_out_guess
    if has_Ph_in:
        Ph[0] = Ph_out_guess
    else:
        Ph[0] = fluids.Ph_out
    Tc[0] = fluids.Tc_in
    Pc[0] = fluids.Pc_in

    # Marching
    for j in range(geometry.n_headers - 1):
        G_h = mdot_hot_per_header / area_free_hot_of_sector[j]
        G_c = mdot_cold_per_header / area_free_cold_of_sector[j]

        sh = fluids.hot.state(Th[j], Ph[j])
        sc = fluids.cold.state(Tc[j], Pc[j])

        eps_C_min_local, tau_hot, tau_cold = calc_eps_local_and_tau(
            geometry=geometry,
            sh=sh,
            sc=sc,
            G_h=G_h,
            G_c=G_c,
            area_ht_hot_j=area_ht_hot_of_sector[j],
            area_ht_cold_j=area_ht_cold_of_sector[j],
            area_free_hot_j=area_free_hot_of_sector[j],
            area_free_cold_j=area_free_cold_of_sector[j],
            mdot_hot_per_header=mdot_hot_per_header,
            mdot_cold_per_header=mdot_cold_per_header,
        )

        q = eps_C_min_local * (Th[j] - Tc[j])

        dh0_hot = -q / mdot_hot_per_header
        dh0_cold = q / mdot_cold_per_header

        Th[j + 1], Ph[j + 1] = _upd_stat_prop(
            fluids.hot,
            G_h,
            dh0_hot,
            tau_hot,
            T_a=Th[j],
            p_b=Ph[j],
            a_is_in=False,
            b_is_in=False,
            max_iter=property_solver_it_max,
            tol_T=property_solver_tol_T,
            rel_tol_p=rel_tol_p,
        )

        Tc[j + 1], Pc[j + 1] = _upd_stat_prop(
            fluids.cold,
            G_c,
            dh0_cold,
            tau_cold,
            T_a=Tc[j],
            p_b=Pc[j],
            a_is_in=True,
            b_is_in=True,
            max_iter=property_solver_it_max,
            tol_T=property_solver_tol_T,
            rel_tol_p=rel_tol_p,
        )

        if not np.all(np.isfinite([Th[j + 1], Ph[j + 1], Tc[j + 1], Pc[j + 1]])):
            return np.array([np.nan] if not has_Ph_in else [np.nan, np.nan], dtype=float)

    Th_in_calc = Th[-1]
    Ph_in_calc = Ph[-1]

    residual_T = Th_in_calc - fluids.Th_in
    if has_Ph_in:
        residual_P = Ph_in_calc - fluids.Ph_in
        return np.array([residual_T, residual_P], dtype=float)
    else:
        return np.array([residual_T], dtype=float)


# ---------- Overall performance (diagnostics) ----------
def compute_overall_performance(
    boundary_converged: np.ndarray | list[float],
    geometry: RadialSpiralProtocol,
    fluids: _FluidInputs,
    *,
    property_solver_it_max: int = 20,
    property_solver_tol_T: float = 1e-2,
    rel_tol_p: float = 1e-3,
) -> dict[str, object]:
    """Run march and return residuals, state arrays, and diagnostics for inboard case."""
    geo = geometry._1d_arrays_for_one_sector()
    area_ht_hot = geo["area_ht_hot"]
    area_ht_cold = geo["area_ht_cold"]
    area_free_hot = geo["area_free_hot"]
    area_free_cold = geo["area_free_cold"]

    has_Ph_in = fluids.Ph_in is not None
    x = np.asarray(boundary_converged, dtype=float)
    if has_Ph_in and x.size != 2:
        raise ValueError(
            "boundary_converged must be [Th_out_converged, Ph_out_converged] when Ph_in is given."
        )
    if (not has_Ph_in) and x.size != 1:
        raise ValueError("boundary_converged must be [Th_out_converged] when Ph_out is given.")

    Th = np.zeros(geometry.n_headers)
    Ph = np.zeros(geometry.n_headers)
    Tc = np.zeros(geometry.n_headers)
    Pc = np.zeros(geometry.n_headers)

    Th[0] = float(x[0])
    Ph[0] = float(x[1]) if has_Ph_in else float(fluids.Ph_out)  # type: ignore[arg-type]
    Tc[0] = float(fluids.Tc_in)
    Pc[0] = float(fluids.Pc_in)

    mdot_h = fluids.m_dot_hot / geometry.n_headers
    mdot_c = fluids.m_dot_cold / geometry.n_headers

    UA_sum = 0.0
    for j in range(geometry.n_headers - 1):
        G_h = mdot_h / area_free_hot[j]
        G_c = mdot_c / area_free_cold[j]
        sh = fluids.hot.state(Th[j], Ph[j])
        sc = fluids.cold.state(Tc[j], Pc[j])

        eps_C_min_local, tau_h, tau_c = calc_eps_local_and_tau(
            geometry=geometry,
            sh=sh,
            sc=sc,
            G_h=G_h,
            G_c=G_c,
            area_ht_hot_j=area_ht_hot[j],
            area_ht_cold_j=area_ht_cold[j],
            area_free_hot_j=area_free_hot[j],
            area_free_cold_j=area_free_cold[j],
            mdot_hot_per_header=mdot_h,
            mdot_cold_per_header=mdot_c,
        )

        q = eps_C_min_local * (Th[j] - Tc[j])

        # Recompute U for UA sum (kept local to avoid altering helper return signature)
        mu_h = sh.mu
        k_h = sh.k
        mu_c = sc.mu
        k_c = sc.k
        Pr_h = mu_h * sh.cp / k_h
        Pr_c = mu_c * sc.cp / k_c
        Re_h_od = G_h * geometry.tube_outer_diam / mu_h
        Re_c = G_c * geometry.tube_inner_diam / mu_c
        Nu_h, _ = _bank_corr(
            Re_h_od,
            geometry.tube_spacing_long,
            geometry.tube_spacing_trv,
            Pr_h,
            inline=(not geometry.staggered),
            n_rows=geometry.n_rows_per_header * geometry.n_headers,
        )
        Nu_c = _circ_nu(Re_c, 0, prandtl=Pr_c)
        h_h = Nu_h * k_h / geometry.tube_outer_diam
        h_c = Nu_c * k_c / geometry.tube_inner_diam
        wall_term = (
            geometry.tube_outer_diam
            / (2.0 * geometry.wall_conductivity)
            * np.log(geometry.tube_outer_diam / geometry.tube_inner_diam)
        )
        U_hot = 1.0 / (
            (1.0 / h_h)
            + (1.0 / h_c) * (geometry.tube_outer_diam / geometry.tube_inner_diam)
            + wall_term
        )
        UA_sum += U_hot * area_ht_hot[j] * geometry.n_headers

        Th[j + 1], Ph[j + 1] = _upd_stat_prop(
            fluids.hot,
            G_h,
            -q / mdot_h,
            tau_h,
            T_a=Th[j],
            p_b=Ph[j],
            a_is_in=False,
            b_is_in=False,
            max_iter=property_solver_it_max,
            tol_T=property_solver_tol_T,
            rel_tol_p=rel_tol_p,
        )
        Tc[j + 1], Pc[j + 1] = _upd_stat_prop(
            fluids.cold,
            G_c,
            q / mdot_c,
            tau_c,
            T_a=Tc[j],
            p_b=Pc[j],
            a_is_in=True,
            b_is_in=True,
            max_iter=property_solver_it_max,
            tol_T=property_solver_tol_T,
            rel_tol_p=rel_tol_p,
        )

    # Residuals
    Th_in_calc = Th[-1]
    Ph_in_calc = Ph[-1]
    residual_T = Th_in_calc - float(fluids.Th_in)
    if has_Ph_in:
        residuals = np.array([residual_T, Ph_in_calc - float(fluids.Ph_in)], dtype=float)  # type: ignore[arg-type]
    else:
        residuals = np.array([residual_T], dtype=float)

    # Performance diagnostics
    Th_out = Th[0]
    Tc_out = Tc[-1]
    Pc_out = Pc[-1]
    if has_Ph_in:
        Ph_in = float(fluids.Ph_in)  # type: ignore[arg-type]
        Ph_out = Ph[0]
    else:
        Ph_in = Ph[-1]
        Ph_out = float(fluids.Ph_out)  # type: ignore[arg-type]

    state_h_in = fluids.hot.state(float(fluids.Th_in), Ph_in)
    state_h_out = fluids.hot.state(Th_out, Ph_out)
    state_c_in = fluids.cold.state(float(fluids.Tc_in), float(fluids.Pc_in))
    state_c_out = fluids.cold.state(Tc_out, Pc_out)

    area_free_hot_in = area_free_hot[-1] * geometry.n_headers
    area_free_hot_out = area_free_hot[0] * geometry.n_headers
    area_free_cold_total = area_free_cold[0] * geometry.n_headers
    G_h_in = fluids.m_dot_hot / area_free_hot_in
    G_h_out = fluids.m_dot_hot / area_free_hot_out
    G_c_total = fluids.m_dot_cold / area_free_cold_total

    h_stag_in_hot = state_h_in.h + 0.5 * (G_h_in / state_h_in.rho) ** 2
    h_stag_out_hot = state_h_out.h + 0.5 * (G_h_out / state_h_out.rho) ** 2
    h_stag_in_cold = state_c_in.h + 0.5 * (G_c_total / state_c_in.rho) ** 2
    h_stag_out_cold = state_c_out.h + 0.5 * (G_c_total / state_c_out.rho) ** 2

    Q_hot = fluids.m_dot_hot * (h_stag_in_hot - h_stag_out_hot)
    Q_cold = fluids.m_dot_cold * (h_stag_out_cold - h_stag_in_cold)

    cp_h_avg = (h_stag_in_hot - h_stag_out_hot) / (float(fluids.Th_in) - Th_out)
    cp_c_avg = (h_stag_out_cold - h_stag_in_cold) / (Tc_out - float(fluids.Tc_in))
    C_h = fluids.m_dot_hot * cp_h_avg
    C_c = fluids.m_dot_cold * cp_c_avg
    C_min = min(C_h, C_c)
    Cr = C_min / max(C_h, C_c) if max(C_h, C_c) > 0 else float("nan")
    NTU = UA_sum / C_min if C_min > 0 else float("nan")
    Q_max = C_min * (float(fluids.Th_in) - float(fluids.Tc_in))
    epsilon = Q_hot / Q_max if Q_max > 0 else 0.0

    dP_hot = Ph_in - Ph_out
    dP_cold = Pc_out - float(fluids.Pc_in)
    dP_hot_pct = 100.0 * dP_hot / Ph_in if Ph_in > 0 else 0.0
    dP_cold_pct = 100.0 * dP_cold / float(fluids.Pc_in) if float(fluids.Pc_in) > 0 else 0.0

    # Inlet/exit transport properties for non-dimensional groups (hot side)
    mu_h_in = state_h_in.mu
    mu_h_out = state_h_out.mu
    k_h_in = state_h_in.k
    k_h_out = state_h_out.k
    cp_h_in = state_h_in.cp
    cp_h_out = state_h_out.cp
    rho_h_in = state_h_in.rho
    rho_h_out = state_h_out.rho

    # Cold states "paired" with hot ends: use inner (c_in) at hot_out; outer (c_out) at hot_in
    state_c_at_hot_out = state_c_in
    state_c_at_hot_in = state_c_out
    mu_c_in = state_c_at_hot_in.mu
    mu_c_out = state_c_at_hot_out.mu
    k_c_in = state_c_at_hot_in.k
    k_c_out = state_c_at_hot_out.k
    cp_c_in = state_c_at_hot_in.cp
    cp_c_out = state_c_at_hot_out.cp

    Do_eff = geometry.tube_outer_diam
    Di_eff = geometry.tube_inner_diam

    # Velocities and mass flux relations
    V_h_in = G_h_in / rho_h_in if rho_h_in > 0 else float("nan")
    V_h_out = G_h_out / rho_h_out if rho_h_out > 0 else float("nan")

    # Reynolds (OD based) and Prandtl
    Re_h_in = G_h_in * Do_eff / mu_h_in if mu_h_in > 0 else float("nan")
    Re_h_out = G_h_out * Do_eff / mu_h_out if mu_h_out > 0 else float("nan")
    Pr_h_in = mu_h_in * cp_h_in / k_h_in if k_h_in > 0 else float("nan")
    Pr_h_out = mu_h_out * cp_h_out / k_h_out if k_h_out > 0 else float("nan")

    # Tube bank Nu and f at inlet and outlet
    n_rows_total = geometry.n_rows_per_header * geometry.n_headers
    Nu_h_in, f_h_in = (
        _bank_corr(
            Re_h_in,
            geometry.tube_spacing_long,
            geometry.tube_spacing_trv,
            Pr_h_in,
            inline=(not geometry.staggered),
            n_rows=n_rows_total,
        )
        if np.isfinite(Re_h_in) and np.isfinite(Pr_h_in)
        else (float("nan"), float("nan"))
    )
    Nu_h_out, f_h_out = (
        _bank_corr(
            Re_h_out,
            geometry.tube_spacing_long,
            geometry.tube_spacing_trv,
            Pr_h_out,
            inline=(not geometry.staggered),
            n_rows=n_rows_total,
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

    # Local h_h and h_c at inlet and outlet (use OD for hot, ID for cold)
    h_h_in = Nu_h_in * k_h_in / Do_eff if np.isfinite(Nu_h_in) and Do_eff > 0 else float("nan")
    h_h_out = Nu_h_out * k_h_out / Do_eff if np.isfinite(Nu_h_out) and Do_eff > 0 else float("nan")

    # Cold-side Re, Pr, Nu at corresponding ends
    Re_c_in = G_c_total * Di_eff / mu_c_in if mu_c_in > 0 and Di_eff > 0 else float("nan")
    Re_c_out = G_c_total * Di_eff / mu_c_out if mu_c_out > 0 and Di_eff > 0 else float("nan")
    Pr_c_in = mu_c_in * cp_c_in / k_c_in if k_c_in > 0 else float("nan")
    Pr_c_out = mu_c_out * cp_c_out / k_c_out if k_c_out > 0 else float("nan")
    Nu_c_in = (
        _circ_nu(Re_c_in, 0, prandtl=Pr_c_in)
        if np.isfinite(Re_c_in) and np.isfinite(Pr_c_in)
        else float("nan")
    )
    Nu_c_out = (
        _circ_nu(Re_c_out, 0, prandtl=Pr_c_out)
        if np.isfinite(Re_c_out) and np.isfinite(Pr_c_out)
        else float("nan")
    )
    h_c_in = Nu_c_in * k_c_in / Di_eff if np.isfinite(Nu_c_in) and Di_eff > 0 else float("nan")
    h_c_out = Nu_c_out * k_c_out / Di_eff if np.isfinite(Nu_c_out) and Di_eff > 0 else float("nan")

    # Wall conduction term and overall U at ends
    wall_term = (
        Do_eff / (2.0 * geometry.wall_conductivity) * np.log(Do_eff / Di_eff)
        if Do_eff > 0 and Di_eff > 0 and geometry.wall_conductivity > 0
        else float("nan")
    )

    def _U_local(hh: float, hc: float) -> float:
        if not (np.isfinite(hh) and np.isfinite(hc) and np.isfinite(wall_term)):
            return float("nan")
        return 1.0 / (1.0 / hh + (1.0 / hc) * (Do_eff / Di_eff) + wall_term)

    U_in = _U_local(h_h_in, h_c_in)
    U_out = _U_local(h_h_out, h_c_out)

    # Wall temperatures on hot side: Tw_hot = Th - q''/h_h, q'' = U*(Th-Tc)
    deltaT_in = float(fluids.Th_in) - Tc_out
    deltaT_out = Th_out - float(fluids.Tc_in)
    qpp_in = U_in * deltaT_in if np.isfinite(U_in) else float("nan")
    qpp_out = U_out * deltaT_out if np.isfinite(U_out) else float("nan")
    Tw_h_in = (
        float(fluids.Th_in) - (qpp_in / h_h_in)
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
        (V_h_in**2) / (cp_h_in * max(float(fluids.Th_in) - Tw_h_in, 1e-16))
        if np.isfinite(V_h_in) and np.isfinite(Tw_h_in)
        else float("nan")
    )
    Ec_h_out = (
        (V_h_out**2) / (cp_h_out * max(Th_out - Tw_h_out, 1e-16))
        if np.isfinite(V_h_out) and np.isfinite(Tw_h_out)
        else float("nan")
    )

    diagnostics: dict[str, float] = {
        "total_UA": float(UA_sum),
        "Q_total": float(Q_hot),
        "Q_hot": float(Q_hot),
        "Q_cold": float(Q_cold),
        "epsilon": float(epsilon),
        "NTU": float(NTU),
        "Cr": float(Cr),
        "dP_hot": float(dP_hot),
        "dP_hot_pct": float(dP_hot_pct),
        "dP_cold": float(dP_cold),
        "dP_cold_pct": float(dP_cold_pct),
        "Th_out": float(Th_out),
        "Tc_out": float(Tc_out),
        "Ph_out": float(Ph_out),
        "Pc_out": float(Pc_out),
    }

    # Store non-dimensional groups and supporting quantities
    diagnostics["Re_h_in"] = float(Re_h_in)
    diagnostics["Re_h_out"] = float(Re_h_out)
    diagnostics["St_h_in"] = float(St_h_in)
    diagnostics["St_h_out"] = float(St_h_out)
    diagnostics["f_h_in"] = float(f_h_in)
    diagnostics["f_h_out"] = float(f_h_out)
    diagnostics["Ec_h_in"] = float(Ec_h_in)
    diagnostics["Ec_h_out"] = float(Ec_h_out)
    diagnostics["V_h_in"] = float(V_h_in)
    diagnostics["V_h_out"] = float(V_h_out)
    diagnostics["V2_h_in"] = float(V_h_in**2) if np.isfinite(V_h_in) else float("nan")
    diagnostics["V2_h_out"] = float(V_h_out**2) if np.isfinite(V_h_out) else float("nan")
    diagnostics["cp_h_in"] = float(cp_h_in)
    diagnostics["cp_h_out"] = float(cp_h_out)
    diagnostics["dT_hw_in"] = (
        float(float(fluids.Th_in) - Tw_h_in) if np.isfinite(Tw_h_in) else float("nan")
    )
    diagnostics["dT_hw_out"] = float(Th_out - Tw_h_out) if np.isfinite(Tw_h_out) else float("nan")

    return {
        "residuals": residuals,
        "Th": Th,
        "Ph": Ph,
        "Tc": Tc,
        "Pc": Pc,
        "diagnostics": diagnostics,
    }


__all__ = [
    "RadialSpiralProtocol",
    "RadialSpiralSpec",
    "rad_spiral_shoot",
    "compute_overall_performance",
]
