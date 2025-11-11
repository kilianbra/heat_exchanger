from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TubeBankStraightGeometry:
    """Geometry of a straight tube bank. Can either be a box tube bank or an axial involute tube bank."""

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

    @property
    def tube_inner_diam(self) -> float:
        return self.tube_outer_diam - 2 * self.tube_thick

    def row_width(self) -> float:
        """Width of one row of tubes. If annular, this is pi D_i. If box this is box width."""
        return self.tube_spacing_trv * self.tube_outer_diam * self.n_tubes_per_row

    def tube_length(self) -> float:
        if self.annular_not_box:
            D_i = self.row_width() / np.pi
            D_o = np.sqrt(D_i**2 + self.frontal_area_outer / np.pi)
            return D_i / 4 * ((D_o / D_i) ** 2 - 1)
        else:
            W = self.row_width()
            return self.frontal_area_outer / W

    def passage_height(self) -> float:
        if self.annular_not_box:
            D_i = self.row_width() / np.pi
            D_o = np.sqrt(D_i**2 + self.frontal_area_outer / np.pi)
            return (D_o - D_i) / 2
        else:
            W = self.row_width()
            return self.frontal_area_outer / W

    @property
    def axial_length(self) -> float:
        return self.tube_spacing_long * self.tube_outer_diam * self.n_rows_per_pass * self.n_passes

    @property
    def sigma_outer(self) -> float:
        interim = (self.tube_spacing_trv - 1) / self.tube_spacing_trv
        if self.staggered:
            diag_spacing = np.sqrt(self.tube_spacing_long**2 + (0.5 * self.tube_spacing_trv) ** 2)
            interim = min(interim, 2.0 * (diag_spacing - 1) / self.tube_spacing_trv)
        return interim

    def area_free_flow_outer(self) -> float:
        return self.frontal_area_outer * self.sigma_outer

    @property
    def n_rows_total(self) -> int:
        return self.n_rows_per_pass * self.n_passes

    @property
    def n_tubes_total(self) -> int:
        return self.n_tubes_per_row * self.n_rows_total

    def n_tubes_per_pass(self) -> int:
        return self.n_tubes_per_row * self.n_rows_per_pass

    def area_free_flow_inner(self) -> float:
        return np.pi / 4 * self.tube_inner_diam**2 * self.n_tubes_per_pass()

    def area_heat_transfer_outer_per_row(self) -> float:
        return np.pi * self.tube_outer_diam * self.tube_length() * self.n_tubes_per_row

    def area_heat_transfer_inner_per_row(self) -> float:
        return np.pi * self.tube_inner_diam * self.tube_length() * self.n_tubes_per_row
