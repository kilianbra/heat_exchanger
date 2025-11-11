"""Involute heat-exchanger geometry protocol.
Provides 0D cached properties and an on-demand method to compute 1D arrays
for a single sector used in radial marching (inner radius j=0 to outer j=n_headers).
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

WALL_CONDUCTIVITY_304_SS = 14.0
WALL_DENSITY_304_SS = 7930.0


class RadialSpiralGeometry(Protocol):
    """Protocol describing an involute heat-exchanger geometry.

    Required inputs:
      - tube_outer_diam, tube_thick
      - tube_spacing_trv, tube_spacing_long (non-dimensional spacing ratios)
      - staggered (True for staggered, False for inline)
      - n_headers, n_rows_per_header, n_rows_axial
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
    n_rows_axial: int
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
        return self.n_rows_axial * self.tube_spacing_trv * self.tube_outer_diam

    @cached_property
    def frontal_area_outer(self) -> float:
        return (
            2.0
            * np.pi
            * self.radius_outer_whole_hex
            * self.n_rows_axial
            * self.tube_outer_diam
            * self.tube_spacing_trv
        )

    @cached_property
    def n_tubes_total(self) -> int:
        return self.n_rows_axial * self.n_headers * self.n_rows_per_header

    @cached_property
    def tubes_per_layer(self) -> int:
        return self.n_rows_per_header * self.n_rows_axial

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

            area_ht_hot[j] = np.pi * self.tube_outer_diam * tube_length[j] * self.tubes_per_layer
            area_ht_cold[j] = np.pi * self.tube_inner_diam * tube_length[j] * self.tubes_per_layer

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


""" Marching options from old
    property_solver_iterations: int = 20
    # ruff: noqa: N815 (allow mixed case variables)
    property_solver_tol_T: float = 1e-2
    property_solver_rel_tol_p: float = 1e-3
"""


__all__ = ["RadialSpiralGeometry"]
