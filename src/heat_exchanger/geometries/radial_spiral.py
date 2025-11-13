"""Involute heat-exchanger geometry protocol.
Provides 0D cached properties and an on-demand method to compute 1D arrays
for a single sector used in radial marching. Arrays are oriented so that
index 0 corresponds to the flow in tubes / cold inlet (which may be at the inner or outer
radius depending on configuration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np
from scipy.optimize import root

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
    def frontal_area_outer(self) -> float: ...

    @property
    def sigma_outer(self) -> float: ...

    @property
    def tube_inner_diam(self) -> float: ...


class RadialSpiralProtocol(TubeBankCorrelationGeometry, Protocol):
    """Protocol describing a heat-exchanger geometry where a tube bank is wrapped in a spiral
    and the flow outside the tubes is flowing radially. This produces a local crossflow
    but overall counterflow configuration.

    Required inputs:
      - tube_outer_diam, tube_thick
      - tube_spacing_trv, tube_spacing_long (non-dimensional spacing ratios)
      - staggered (True for staggered, False for inline)
      - n_headers, n_rows_per_header, n_tubes_per_row
      - radius_outer_hex
      - inv_angle_deg (involute sweep angle in degrees)

    0D analysis uses the cached-style properties below. 1D analysis calls
    _1d_arrays_for_one_sector() to compute all arrays needed for stepping.

    Orientation:
      - ext_fluid_flows_radially_inwards: True for inboard (external hot fluid
        flows from outer to inner radius), False for outboard (external hot
        fluid flows from inner to outer radius).
      - All returned 1D arrays are oriented such that index 0 is at the cold
        inlet, so marching proceeds from cold inlet to cold outlet.
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
    radius_outer_hex: float
    inv_angle_deg: float = 360.0
    wall_conductivity: float = WALL_CONDUCTIVITY_304_SS
    # Orientation flag: True=inboard (external flow inward), False=outboard (external flow outward)
    ext_fluid_flows_radially_inwards: bool = True

    # ---------- Cached-style 0D properties (default implementations) ----------
    @cached_property
    def tube_inner_diam(self) -> float:
        return self.tube_outer_diam - 2.0 * self.tube_thick

    @cached_property
    def radius_inner_hex(self) -> float:
        n_rows_per_axial_section = self.n_rows_per_header * self.n_headers
        outer_radius_span = n_rows_per_axial_section * self.tube_spacing_long * self.tube_outer_diam
        if outer_radius_span >= self.radius_outer_hex:
            raise ValueError(
                f"Invalid geometry: too many rows in axial section for given outer radius: "
                f"{outer_radius_span:.2f} m > {self.radius_outer_hex:.2f} m for "
                f"{n_rows_per_axial_section} rows of tubes spaced by "
                f"{self.tube_spacing_long * self.tube_outer_diam:.2f} m"
            )
        elif outer_radius_span <= 0:
            raise ValueError(
                f"Invalid geometry: negative width of annulus: {outer_radius_span:.2f} m < 0 m for "
                f"{n_rows_per_axial_section} rows of tubes spaced by "
                f"{self.tube_spacing_long * self.tube_outer_diam:.2f} m"
            )
        return self.radius_outer_hex - outer_radius_span

    @cached_property
    def axial_length(self) -> float:
        """Axial Length of total Heat Exchanger"""
        return self.n_tubes_per_row * self.tube_spacing_trv * self.tube_outer_diam

    @cached_property
    def spiral_b(self) -> float:
        """Archimedian spiral parameter b.
        2 pi b is the radial distance between two points of spiral on same ray to origin.
        """
        return (self.radius_outer_hex - self.radius_inner_hex) / np.deg2rad(self.inv_angle_deg)

    @cached_property
    def spiral_length(self) -> float:
        """Exact analytical formula for archimedian spiral arc length.

        Formula from Wikipedia:
            b/2 [θ√(1 + θ²) + ln(θ + √(1 + θ²))] evaluated from θ₁ to θ₂

        Where θ₁ = r_min/b and θ₂ = r_max/b.
        """
        # Compute theta values
        theta_1 = self.radius_inner_hex / self.spiral_b
        theta_2 = self.radius_outer_hex / self.spiral_b

        # Helper function for the integrand antiderivative
        def antiderivative(theta: float) -> float:
            """Antiderivative: θ√(1 + θ²) + ln(θ + √(1 + θ²))"""
            sqrt_term = np.sqrt(1 + theta**2)
            return theta * sqrt_term + np.log(theta + sqrt_term)

        # Evaluate at bounds
        F_theta_2 = antiderivative(theta_2)
        F_theta_1 = antiderivative(theta_1)

        # Apply the formula: b/2 * [F(θ₂) - F(θ₁)]
        return (self.spiral_b / 2.0) * (F_theta_2 - F_theta_1)

    @cached_property
    def frontal_area_outer(self) -> float:
        return 2.0 * np.pi * self.radius_outer_hex * self.n_tubes_per_row * self.tube_outer_diam * self.tube_spacing_trv

    @cached_property
    def area_heat_transfer_outer_total(self) -> float:
        return np.pi * self.tube_outer_diam * self.spiral_length * self.n_tubes_total

    @cached_property
    def area_heat_transfer_inner_total(self) -> float:
        return np.pi * self.tube_inner_diam * self.spiral_length * self.n_tubes_total

    @cached_property
    def frontal_area_outer_total(self) -> float:
        return 2.0 * np.pi * self.radius_outer_hex * self.n_tubes_per_row * self.tube_outer_diam * self.tube_spacing_trv

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

        Arrays are oriented such that index 0 corresponds to the flow in tubes / cold inlet.

        Returns a dict with keys:
          - radii, theta, tube_length, area_ht_hot, area_ht_cold,
            area_frontal_hot, area_free_hot, area_free_cold, d_h_hot
          - dR (float)
        """
        dR = (self.radius_outer_hex - self.radius_inner_hex) / self.n_headers

        radii = self.radius_inner_hex + np.arange(self.n_headers + 1, dtype=float) * dR
        inv_b = (self.radius_outer_hex - self.radius_inner_hex) / np.deg2rad(self.inv_angle_deg)
        theta = (radii - self.radius_inner_hex) / inv_b

        area_ht_hot = np.zeros(self.n_headers)
        area_ht_cold = np.zeros(self.n_headers)
        area_frontal_hot = np.zeros(self.n_headers)
        area_free_hot = np.zeros(self.n_headers)
        area_free_cold = np.zeros(self.n_headers)
        tube_length = np.zeros(self.n_headers)
        d_h_hot = np.zeros(self.n_headers)

        length_flow_outer_per_header = self.n_rows_per_header * self.tube_spacing_long * self.tube_outer_diam

        for j in range(self.n_headers):
            tube_length[j] = np.trapezoid(
                np.sqrt(radii[j : j + 2] ** 2 + inv_b**2),
                theta[j : j + 2],
            )
            area_frontal_hot[j] = self.axial_length * 2.0 * np.pi * radii[j] / self.n_headers

            area_ht_hot[j] = np.pi * self.tube_outer_diam * tube_length[j] * self.n_tubes_per_header
            area_ht_cold[j] = np.pi * self.tube_inner_diam * tube_length[j] * self.n_tubes_per_header

            area_free_hot[j] = area_frontal_hot[j] * self.sigma_outer
            # Valid for this segmentation of the HEx
            area_free_cold[j] = np.pi * self.tube_inner_diam**2 / 4.0 * self.n_tubes_total / self.n_headers
            d_h_hot[j] = (4.0 * area_free_hot[j] * length_flow_outer_per_header) / area_ht_hot[j]

        # Orient arrays so index 0 corresponds to cold inlet.
        # For inboard (external flow inward), cold inlet is at inner radius (current order).
        # For outboard (external flow outward), cold inlet is at outer radius -> reverse arrays.
        if not self.ext_fluid_flows_radially_inwards:
            # Reverse per-header arrays
            area_ht_hot = area_ht_hot[::-1]
            area_ht_cold = area_ht_cold[::-1]
            area_frontal_hot = area_frontal_hot[::-1]
            area_free_hot = area_free_hot[::-1]
            area_free_cold = area_free_cold[::-1]
            tube_length = tube_length[::-1]
            d_h_hot = d_h_hot[::-1]
            # Reverse node arrays (n_headers + 1)
            radii = radii[::-1]
            theta = theta[::-1]

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
    radius_outer_hex: float
    inv_angle_deg: float = 360.0
    wall_conductivity: float = WALL_CONDUCTIVITY_304_SS
    ext_fluid_flows_radially_inwards: bool = True


def spiral_hex_solver(
    geom: RadialSpiralProtocol,
    f_in: _FluidInputs,
    method: str = "0d",
) -> dict[str, object]:
    """Solve the radial spiral heat exchanger.
    Method can be "0d" for 0D guess assuming counterflow HEx, or "1d" for 1D marching.
    """

    logger = logging.getLogger(__name__ + ".spiral_hex_solver")

    logger.info(
        "Hot inlet parameters: \t \t \t Th_in =%.2f K, Ph_known =%.2e Pa (at %s)",
        f_in.Th_in,
        f_in.Ph_in if f_in.Ph_in is not None else f_in.Ph_out,
        "inlet" if f_in.Ph_in is not None else "outlet",
    )

    # 0D two-step initial guess assuming counterflow epsilon-NTU relationship
    # (first using inlet properties as average, then using average of first guess
    # outlet and known inlet to get a better average properties guess)
    Th0, Ph0 = xflow_guess_0d(geom, f_in)

    dP_P_in = (1 - Ph0 / f_in.Ph_in) * 100.0 if f_in.Ph_in is not None else (1 - f_in.Ph_out / Ph0) * 100.0

    logger.info(
        "0D guess (inner b.) \t Th_out=%.2f K, ΔPh/Ph_in=%.1f %%",
        Th0,
        dP_P_in,
    )

    if method != "0d":
        eval_state = {"count": 0}
        tol_root = 1.0e-2
        tol_P_pct_of_Ph_in = 0.1
        tol_P = (
            tol_P_pct_of_Ph_in
            / 100
            * (f_in.Ph_in if f_in.Ph_in is not None else f_in.Ph_out if f_in.Ph_out is not None else float("nan"))
        )  # absolute Pa tolerance

        def residuals(x: np.ndarray) -> np.ndarray:
            """Residuals for the hot-side shooting problem.

            Inputs
            -------
            x : ndarray
                boundary guess vector; [Th_out, Ph_out] if Ph_in known; [Th_out] if Ph_out known.

            Returns
            --------
            scaled : np.ndarray shape (1 or 2,)
                Temperature residual ``Th_in_calc - Th_in`` unscaled;
                pressure residual scaled so that |ΔP| == tol_P maps to tol_root (when present).
            """
            eval_state["count"] += 1
            raw = rad_spiral_shoot(x, geom, f_in, property_solver_it_max=40, property_solver_tol_T=1e-2, rel_tol_p=1e-3)
            # Root solver uses a single scalar tolerance (tol_root). Temperature residual uses it directly;
            # pressure residual is normalised so that when |ΔP| == tol_P the scaled residual equals tol_root.
            if raw.size == 2:
                scaled = np.array([raw[0], raw[1] * (tol_root / tol_P)], dtype=float)
            else:
                scaled = np.array([raw[0]], dtype=float)
            logger.debug(
                ("Residual eval %d: x=%s -> Th_in_calc-Th_in=%+.1e K (target %.1e)%s"),
                eval_state["count"],
                np.array2string(x, precision=3),
                raw[0],
                tol_root,
                "" if raw.size == 1 else f", Ph_in_calc-Ph_in={raw[1]:+.1e} Pa (target {tol_P:.1e} Pa)",
            )
            return scaled

        x0 = np.array([Th0, Ph0], dtype=float) if f_in.Ph_in is not None else np.array([Th0], dtype=float)
        sol = root(residuals, x0, method="hybr", tol=tol_root, options={"maxfev": 60})

        logger.debug(
            "SciPy root success=%s, nfev=%s, message=%s",
            sol.success,
            getattr(sol, "nfev", None),
            sol.message,
        )

        bound_converged = sol.x

    else:  # 0D method
        if f_in.Ph_in is not None:
            bound_converged = [Th0, Ph0]
            dP_P_in = (1 - Ph0 / f_in.Ph_in) * 100.0
        else:
            bound_converged = [Th0]
            dP_P_in = (1 - f_in.Ph_out / Ph0) * 100.0

    result = compute_overall_performance(
        bound_converged, geom, f_in, property_solver_it_max=40, property_solver_tol_T=1e-2, rel_tol_p=1e-3
    )
    final_diag: dict[str, float] = result["diagnostics"]  # type: ignore[assignment]
    final_raw = result["residuals"]  # type: ignore[assignment]

    if method != "0d":
        dP_P_in = (
            (1 - bound_converged[1] / f_in.Ph_in) * 100.0
            if f_in.Ph_in is not None
            else final_diag.get("dP_hot_pct", float("nan"))
        )
        logger.info(
            "Solution after %d iterations: \t \t Th_out=%.2f K, ΔPh/Ph_in=%.1f %%",
            eval_state["count"],
            bound_converged[0],
            dP_P_in,
        )

        if final_raw.size == 2:
            logger.info(
                "Residuals: Th_in_calc-Th_in=%.1e K (tol %.1e K), Ph_in_calc-Ph_in=%.1e Pa (tol %.1e Pa)",
                final_raw[0],
                tol_root,
                final_raw[1],
                tol_P,
            )
        else:
            logger.info("Residual: Th_in_calc-Th_in=%.1e K (tol %.1e K)", final_raw[0], tol_root)
    if final_diag:
        logger.info(
            "Performance: ε=%.3f, NTU=%.2f, Cr=%.3f, Q_hot=%.2f MW, Q_cold=%.2f MW",
            final_diag.get("epsilon", float("nan")),
            final_diag.get("NTU", float("nan")),
            final_diag.get("Cr", float("nan")),
            final_diag.get("Q_total", float("nan")) / 1e6,
            final_diag.get("Q_cold", float("nan")) / 1e6,
        )
        logger.info(
            "Pressure drops: ΔPh=%.1f%% (%.1f Pa), ΔPc=%.1f%% (%.1f Pa)",
            final_diag.get("dP_hot_pct", float("nan")),
            final_diag.get("dP_hot", float("nan")),
            final_diag.get("dP_cold_pct", float("nan")),
            final_diag.get("dP_cold", float("nan")),
        )
        logger.info(
            "Outlet conditions: Th_out=%.2f K, Tc_out=%.2f K",
            final_diag.get("Th_out", float("nan")),
            final_diag.get("Tc_out", float("nan")),
        )
        # Hot-side non-dimensionals and Eckert numbers at inlet and exit
        logger.debug(
            "Hot inlet (outer r): Re=%.3e, St=%.3e, f=%.3e, Ec=%.3e, V=%.3f m/s (V^2=%.1e)",
            final_diag.get("Re_h_in", float("nan")),
            final_diag.get("St_h_in", float("nan")),
            final_diag.get("f_h_in", float("nan")),
            final_diag.get("Ec_h_in", float("nan")),
            final_diag.get("V_h_in", float("nan")),
            final_diag.get("V2_h_in", float("nan")),
        )
        logger.debug(
            "  Ec rationale (inlet): Ec = V^2 / (cp*(Th-Tw)) = %.1e / (%.3e*%.3f) = %.3e",
            final_diag.get("V2_h_in", float("nan")),
            final_diag.get("cp_h_in", float("nan")),
            final_diag.get("dT_hw_in", float("nan")),
            final_diag.get("Ec_h_in", float("nan")),
        )
        logger.debug(
            "Hot exit (inner r): Re=%.3e, St=%.3e, f=%.3e, Ec=%.3e, V=%.3f m/s (V^2=%.1e)",
            final_diag.get("Re_h_out", float("nan")),
            final_diag.get("St_h_out", float("nan")),
            final_diag.get("f_h_out", float("nan")),
            final_diag.get("Ec_h_out", float("nan")),
            final_diag.get("V_h_out", float("nan")),
            final_diag.get("V2_h_out", float("nan")),
        )
        logger.debug(
            "  Ec rationale (exit):  Ec = V^2 / (cp*(Th-Tw)) = %.1e / (%.3e*%.3f) = %.3e",
            final_diag.get("V2_h_out", float("nan")),
            final_diag.get("cp_h_out", float("nan")),
            final_diag.get("dT_hw_out", float("nan")),
            final_diag.get("Ec_h_out", float("nan")),
        )
        logger.debug(
            "Total Q_hot=%.2f MW, Q_cold=%.2f MW",
            final_diag.get("Q_hot", float("nan")) / 1e6,
            final_diag.get("Q_cold", float("nan")) / 1e6,
        )

    return result


# ---------- Initial guess (0D estimate) ----------
def xflow_guess_0d(
    geom: RadialSpiralProtocol,
    f_in: _FluidInputs,
) -> tuple[float, float]:
    """Return (Th_inner_guess, Ph_other_guess) using a two-step 0D estimate.
    Assumes that the Radial Spiral Geometry is close enough to a counterflow configuration.
    This tends to be off by only a few percentage points

    The first step evaluates properties at the inlet conditions; the second step
    re-evaluates at the mean of the inlet and the first-step outlet to refine the
    guess.
    """

    logger = logging.getLogger(__name__ + ".xflow_guess_0d")

    # Validate boundary pressure specification
    if (f_in.Ph_in is None and f_in.Ph_out is None) or (f_in.Ph_in is not None and f_in.Ph_out is not None):
        raise ValueError("Specify exactly one of Ph_in or Ph_out in FluidInputs.")

    A_total_hot0 = geom.area_heat_transfer_outer_total
    A_total_cold0 = geom.area_heat_transfer_inner_total

    # Frontal/free areas at mid-radius and total cold frontal area (per sector/header)
    r_mid = 0.5 * (geom.radius_inner_hex + geom.radius_outer_hex)
    Afr_hot_mid = geom.axial_length * 2.0 * np.pi * r_mid
    Aff_hot_mid = Afr_hot_mid * geom.sigma_outer

    Aff_cold_total = geom.n_tubes_total * np.pi * geom.tube_inner_diam**2 / 4.0

    G_h0 = f_in.m_dot_hot / Aff_hot_mid
    G_c0 = f_in.m_dot_cold / Aff_cold_total

    def _0d_xflow_guess(
        _Th_in: float,
        _Ph_in: float,
        _Tc_in: float,
        _Pc_in: float,
        Th_eval: float,
        Ph_eval: float,
        Tc_eval: float,
        Pc_eval: float,
        _Ph_out: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Single 0D estimate using property evaluation at (eval) and inlets (b).
        Approximates the heat exchanger as counterflow
        If _Ph_out is specified, then any input at _Ph_in is ignored. The inlet pressure
        is then calculated and returned as Ph_not_b.
        If no _Ph_out is specified, then like with the other three, _Ph_in is used
        and the exit pressure is returned as Ph_not_b."""
        sh = f_in.hot.state(Th_eval, Ph_eval)
        sc = f_in.cold.state(Tc_eval, Pc_eval)
        Pr_h = sh.mu * sh.cp / sh.k
        Pr_c = sc.mu * sc.cp / sc.k
        Re_h_OD = G_h0 * geom.tube_outer_diam / sh.mu
        Re_c = G_c0 * geom.tube_inner_diam / sc.mu

        Nu_h, f_h = _bank_corr(
            Re_h_OD,
            geom.tube_spacing_long,
            geom.tube_spacing_trv,
            Pr_h,
            inline=(not geom.staggered),
            n_rows=geom.n_rows_per_header * geom.n_headers,
        )
        logger.info(
            "0D guess tube bank for Re_od=%5.2e: St_h=%5.2f, f_h=%5.2e",
            Re_h_OD,
            Nu_h / Re_h_OD / Pr_h,
            f_h,
        )
        Nu_c = _circ_nu(Re_c, 0, prandtl=Pr_c)
        f_c = _circ_fric(Re_c, 0)

        h_h = Nu_h * sh.k / geom.tube_outer_diam
        h_c = Nu_c * sc.k / geom.tube_inner_diam

        wall_term = (
            geom.tube_outer_diam / (2.0 * geom.wall_conductivity) * np.log(geom.tube_outer_diam / geom.tube_inner_diam)
        )
        U_h = 1.0 / (1.0 / h_h + 1.0 / h_c * (geom.tube_outer_diam / geom.tube_inner_diam) + wall_term)

        C_h_tot = f_in.m_dot_hot * sh.cp
        C_c_tot = f_in.m_dot_cold * sc.cp
        C_min = min(C_h_tot, C_c_tot)
        C_max = max(C_h_tot, C_c_tot)
        Cr = C_min / C_max

        NTU = U_h * A_total_hot0 / C_min
        eps = _eps_ntu(NTU, Cr, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)
        logger.info("0D guess epsilon-NTU: NTU=%5.2f, eps=%5.2f", NTU, eps)
        Q = eps * C_min * (_Th_in - _Tc_in)

        tau_h = f_h * (A_total_hot0 / Aff_hot_mid) * (G_h0**2) / (2.0 * sh.rho)
        tau_c = f_c * (A_total_cold0 / Aff_cold_total) * (G_c0**2) / (2.0 * sc.rho)

        dh0_h = -Q / f_in.m_dot_hot
        dh0_c = Q / f_in.m_dot_cold
        Th_out, Ph_not_b = _upd_stat_prop(
            f_in.hot,
            G_h0,
            dh0_h,
            tau_h,
            T_a=_Th_in,
            p_b=_Ph_out if _Ph_out is not None else _Ph_in,
            a_is_in=True,
            b_is_in=(_Ph_out is None),
            max_iter=100,
            tol_T=1e-2,
            rel_tol_p=1e-2,
        )

        Tc_out, Pc_out = _upd_stat_prop(
            f_in.cold,
            G_c0,
            dh0_c,
            tau_c,
            T_a=_Tc_in,
            p_b=_Pc_in,
            a_is_in=True,
            b_is_in=True,
            max_iter=100,
            tol_T=1e-2,
            rel_tol_p=1e-2,
        )

        return Th_out, Tc_out, Ph_not_b, Pc_out

    logger.info(
        "0D guess inputs: Th_in =%5.2f K, Tc_in =%5.2f K, Ph_in =%s Pa, Pc_in =%5.2e Pa (Ph_out=%s)",
        f_in.Th_in,
        f_in.Tc_in,
        f"{f_in.Ph_in:.2e}" if f_in.Ph_in is not None else "N/A",
        f_in.Pc_in,
        f"{f_in.Ph_out:.2e}" if f_in.Ph_out is not None else "N/A",
    )
    Th_o1, Tc_o1, Ph_not_b1, Pc_o1 = _0d_xflow_guess(
        _Th_in=f_in.Th_in,
        _Ph_in=f_in.Ph_in if f_in.Ph_in is not None else float("nan"),
        _Tc_in=f_in.Tc_in,
        _Pc_in=f_in.Pc_in,
        Th_eval=f_in.Th_in,
        Ph_eval=f_in.Ph_out if f_in.Ph_out is not None else f_in.Ph_in,
        Tc_eval=f_in.Tc_in,
        Pc_eval=f_in.Pc_in,
        _Ph_out=f_in.Ph_out,
    )
    if f_in.Ph_out is not None:
        logger.info(
            "0D guess 1: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa (Ph_in_guess=%5.2e Pa)",
            Th_o1,
            Tc_o1,
            f_in.Ph_out,
            Pc_o1,
            Ph_not_b1,
        )
    else:
        logger.info(
            "0D guess 1: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa",
            Th_o1,
            Tc_o1,
            Ph_not_b1,
            Pc_o1,
        )
    Th_mean = 0.5 * (f_in.Th_in + Th_o1)
    Tc_mean = 0.5 * (f_in.Tc_in + Tc_o1)
    Ph_known = f_in.Ph_out if f_in.Ph_out is not None else f_in.Ph_in
    if Ph_known is None:
        raise ValueError("Either Ph_in or Ph_out must be provided for the 0D guess.")
    Ph_mean = 0.5 * (Ph_known + Ph_not_b1)
    Pc_mean = 0.5 * (f_in.Pc_in + Pc_o1)
    Th_o2, Tc_o2, Ph_not_b2, Pc_o2 = _0d_xflow_guess(
        _Th_in=f_in.Th_in,
        _Ph_in=f_in.Ph_in if f_in.Ph_in is not None else float("nan"),
        _Tc_in=f_in.Tc_in,
        _Pc_in=f_in.Pc_in,
        Th_eval=Th_mean,
        Ph_eval=Ph_mean,
        Tc_eval=Tc_mean,
        Pc_eval=Pc_mean,
        _Ph_out=f_in.Ph_out,
    )
    if f_in.Ph_out is not None:
        logger.info(
            "0D guess 2: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa (Ph_in_guess=%5.2e Pa)",
            Th_o2,
            Tc_o2,
            f_in.Ph_out,
            Pc_o2,
            Ph_not_b2,
        )
    else:
        logger.info(
            "0D guess 2: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa",
            Th_o2,
            Tc_o2,
            Ph_not_b2,
            Pc_o2,
        )
    # Hot inner boundary (inboard shoot) guess equals the 0D outlet
    return float(Th_o2), float(Ph_not_b2)


def rad_spiral_shoot(
    boundary_guess: np.ndarray | list[float],
    geometry: RadialSpiralProtocol,
    fluids: _FluidInputs,
    *,
    property_solver_it_max: int = 20,
    # noqa: N815 (allow mixed case variables)
    property_solver_tol_T: float = 1e-2,
    rel_tol_p: float = 1e-3,
) -> np.ndarray:
    """For a radial spiral HEx, where the cold fluid is assumed to flow inside the tubes
    and the hot fluid around them, this function takes a guess at unknown hot boundary conditions
    at the cold inlet and marches towards the hot inlet.
    It then returns the difference between the derived hot inlet condition from the guess
    and the known hot inlet condition.

    if fluids.Ph_in is provided:
        boundary_guess: [Th_out_guess, Ph_out_guess]
        Returns: [Th_in_calc - fluids.Th_in, Ph_in_calc - fluids.Ph_in]
    if fluids.Ph_out is provided:
        boundary_guess: [Th_out_guess]
        Returns: [Th_in_calc - fluids.Th_in]

    Whether it is an inboard or outboard configuration is determined by the
    geometry.ext_fluid_flows_radially_inwards flag. An inboard configuration means the
    hot fluid flows from outer to inner radius, and an outboard configuration means the
    hot fluid flows from inner to outer radius (cold fluid flows in the opposite direction).
    """
    has_Ph_in = fluids.Ph_in is not None
    has_Ph_out = fluids.Ph_out is not None
    if has_Ph_in == has_Ph_out:
        raise ValueError("Specify exactly one of Ph_in or Ph_out in FluidInputs.")

    x = np.asarray(boundary_guess, dtype=float)
    if has_Ph_in:
        if x.size != 2:
            raise ValueError("boundary_guess must be [Th_out_guess, Ph_out_guess] when Ph_in is given.")
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


# ---------- Marching solver ----------
def calc_eps_local_and_tau(
    *,
    geometry: RadialSpiralProtocol,
    sh,
    sc,
    area_ht_hot_j: float,
    area_ht_cold_j: float,
    area_free_hot_j: float,
    area_free_cold_j: float,
    mdot_hot_per_header: float,
    mdot_cold_per_header: float,
) -> tuple[float, float, float]:
    """Return eps_local * C_min_local and (tau_hot, tau_cold) for layer j using local states and areas."""
    G_h = mdot_hot_per_header / area_free_hot_j
    G_c = mdot_cold_per_header / area_free_cold_j
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

    U_hot = 1.0 / ((1.0 / h_h) + (1.0 / h_c) * (geometry.tube_outer_diam / geometry.tube_inner_diam) + wall_term)

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
    """Run march and return residuals, state arrays, and diagnostics.

    Marching is performed from the cold inlet (inner or outer radius depending
    on configuration). Diagnostics map hot-side inlet/outlet consistently based
    on array orientation (hot inlet at index -1, hot outlet at index 0).
    """
    geo = geometry._1d_arrays_for_one_sector()
    area_ht_hot = geo["area_ht_hot"]
    area_ht_cold = geo["area_ht_cold"]
    area_free_hot = geo["area_free_hot"]
    area_free_cold = geo["area_free_cold"]

    has_Ph_in = fluids.Ph_in is not None
    x = np.asarray(boundary_converged, dtype=float)
    if has_Ph_in and x.size != 2:
        raise ValueError("boundary_converged must be [Th_out_converged, Ph_out_converged] when Ph_in is given.")
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
        sh = fluids.hot.state(Th[j], Ph[j])
        sc = fluids.cold.state(Tc[j], Pc[j])

        eps_C_min_local, tau_h, tau_c = calc_eps_local_and_tau(
            geometry=geometry,
            sh=sh,
            sc=sc,
            area_ht_hot_j=area_ht_hot[j],
            area_ht_cold_j=area_ht_cold[j],
            area_free_hot_j=area_free_hot[j],
            area_free_cold_j=area_free_cold[j],
            mdot_hot_per_header=mdot_h,
            mdot_cold_per_header=mdot_c,
        )

        q = eps_C_min_local * (Th[j] - Tc[j])

        G_h = mdot_h / area_free_hot[j]
        G_c = mdot_c / area_free_cold[j]

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
        U_hot = 1.0 / ((1.0 / h_h) + (1.0 / h_c) * (geometry.tube_outer_diam / geometry.tube_inner_diam) + wall_term)
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
    St_h_in = Nu_h_in / (Re_h_in * Pr_h_in) if np.isfinite(Nu_h_in) and Re_h_in > 0 and Pr_h_in > 0 else float("nan")
    St_h_out = (
        Nu_h_out / (Re_h_out * Pr_h_out) if np.isfinite(Nu_h_out) and Re_h_out > 0 and Pr_h_out > 0 else float("nan")
    )

    # Local h_h and h_c at inlet and outlet (use OD for hot, ID for cold)
    h_h_in = Nu_h_in * k_h_in / Do_eff if np.isfinite(Nu_h_in) and Do_eff > 0 else float("nan")
    h_h_out = Nu_h_out * k_h_out / Do_eff if np.isfinite(Nu_h_out) and Do_eff > 0 else float("nan")

    # Cold-side Re, Pr, Nu at corresponding ends
    Re_c_in = G_c_total * Di_eff / mu_c_in if mu_c_in > 0 and Di_eff > 0 else float("nan")
    Re_c_out = G_c_total * Di_eff / mu_c_out if mu_c_out > 0 and Di_eff > 0 else float("nan")
    Pr_c_in = mu_c_in * cp_c_in / k_c_in if k_c_in > 0 else float("nan")
    Pr_c_out = mu_c_out * cp_c_out / k_c_out if k_c_out > 0 else float("nan")
    Nu_c_in = _circ_nu(Re_c_in, 0, prandtl=Pr_c_in) if np.isfinite(Re_c_in) and np.isfinite(Pr_c_in) else float("nan")
    Nu_c_out = (
        _circ_nu(Re_c_out, 0, prandtl=Pr_c_out) if np.isfinite(Re_c_out) and np.isfinite(Pr_c_out) else float("nan")
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
        Th_out - (qpp_out / h_h_out) if np.isfinite(qpp_out) and np.isfinite(h_h_out) and h_h_out > 0 else float("nan")
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
    diagnostics["dT_hw_in"] = float(float(fluids.Th_in) - Tw_h_in) if np.isfinite(Tw_h_in) else float("nan")
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
    "TubeBankCorrelationGeometry",
    "RadialSpiralProtocol",
    "RadialSpiralSpec",
    "rad_spiral_shoot",
    "xflow_guess_0d",
    "compute_overall_performance",
]
