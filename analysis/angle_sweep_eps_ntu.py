from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt
from spiral_recup_cases import load_case  # type: ignore

from heat_exchanger.epsilon_ntu import epsilon_ntu as _eps_ntu
from heat_exchanger.geometries.radial_spiral import (
    RadialSpiralProtocol,
    RadialSpiralSpec,
    spiral_hex_solver,
)
from heat_exchanger.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def _geometry_with_new_angle(base_geom: RadialSpiralProtocol, inv_angle_deg: float) -> RadialSpiralSpec:
    """Create a new RadialSpiralSpec with the same parameters as base_geom but with a new involute angle."""
    return RadialSpiralSpec(
        tube_outer_diam=base_geom.tube_outer_diam,
        tube_thick=base_geom.tube_thick,
        tube_spacing_trv=base_geom.tube_spacing_trv,
        tube_spacing_long=base_geom.tube_spacing_long,
        staggered=base_geom.staggered,
        n_headers=base_geom.n_headers,
        n_rows_per_header=base_geom.n_rows_per_header,
        n_tubes_per_row=base_geom.n_tubes_per_row,
        radius_outer_hex=base_geom.radius_outer_hex,
        inv_angle_deg=inv_angle_deg,
        wall_conductivity=base_geom.wall_conductivity,
        ext_fluid_flows_radially_inwards=base_geom.ext_fluid_flows_radially_inwards,
    )


def _representative_capacity_ratio_and_mixing(geom: RadialSpiralProtocol, inputs) -> tuple[float, str]:
    """Estimate a representative overall C_r and choose fair crossflow mixing type based on which stream is Cmax."""
    # Use inlet cp as a simple, robust proxy for overall capacity rates
    # For hot-side pressure, use whichever boundary is specified; cp is weakly pressure-sensitive here.
    Ph_for_cp = inputs.Ph_in if inputs.Ph_in is not None else inputs.Ph_out
    hot_in = inputs.hot.state(inputs.Th_in, Ph_for_cp if Ph_for_cp is not None else 1.0e5)
    cold_in = inputs.cold.state(inputs.Tc_in, inputs.Pc_in)
    C_h_in = inputs.m_dot_hot * hot_in.cp
    C_c_in = inputs.m_dot_cold * cold_in.cp
    C_min = min(C_h_in, C_c_in)
    C_max = max(C_h_in, C_c_in)
    Cr = C_min / C_max if C_max > 0 else 0.0
    flow_type = "Cmax_mixed" if C_h_in > C_c_in else "Cmin_mixed"
    return float(Cr), flow_type


def run_sweep(case: str = "ahjeb_toc_k1", fluid_model: str = "CoolProp") -> None:
    configure_logging(logging.INFO)
    logging.getLogger("heat_exchanger.geometries.radial_spiral").setLevel(logging.WARNING)

    base_geom, inputs, case_name = load_case(case, fluid_model)
    step_deg = 360.0 / base_geom.n_headers

    # Build list of angles from full 360 down in steps of 360/n_headers, staying > 0
    angles: list[float] = []
    angle = 360.0
    while angle > 0.0 + 1e-9:
        angles.append(angle)
        angle -= step_deg

    logger.info("Sweeping inv_angle_deg for %s: %d steps, step=%.2f deg", case_name, len(angles), step_deg)

    # Representative bounds parameters from inlets (stable across the sweep)
    Cr_rep, xflow_mix_type = _representative_capacity_ratio_and_mixing(base_geom, inputs)
    logger.info(f"Representative C_r={Cr_rep:.3f}, crossflow bound uses '{xflow_mix_type}'")

    # Collect results
    eps_list: list[float] = []
    NTU_list: list[float] = []
    Cr_list: list[float] = []
    ang_list: list[float] = []
    dP_list: list[float] = []

    for ang in angles:
        geom = _geometry_with_new_angle(base_geom, ang)
        try:
            res = spiral_hex_solver(geom, inputs, method="1d")
            diag = res["diagnostics"]  # type: ignore[assignment]
            eps_list.append(float(diag["epsilon"]))
            NTU_list.append(float(diag["NTU"]))
            Cr_list.append(float(diag["Cr"]))
            ang_list.append(float(ang))
            dP_list.append(float(diag["dP_hot_pct"]))
            logger.info(
                "Angle %7.2f deg -> epsilon=%.4f, NTU=%.3f, Cr=%.3f, ΔPh=%.1f %%, ΔPc=%.1f %%",
                ang,
                diag["epsilon"],
                diag["NTU"],
                diag["Cr"],
                diag["dP_hot_pct"],
                diag["dP_cold_pct"],
            )
        except Exception as exc:
            logger.warning("Angle %.2f deg failed: %s", ang, exc)

    if not NTU_list:
        logger.error("No successful solutions; aborting plot.")
        return

    NTU_arr = np.array(NTU_list, dtype=float)
    eps_arr = np.array(eps_list, dtype=float)
    dP_arr = np.array(dP_list, dtype=float)

    # Bounds curves (use a representative Cr and appropriate crossflow mixing)
    NTU_grid = np.linspace(0.0, max(1.05 * np.nanmax(NTU_arr), 1.0), 300)
    eps_counter = _eps_ntu(NTU_grid, Cr_rep, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)
    eps_xflow = _eps_ntu(NTU_grid, Cr_rep, exchanger_type="cross_flow", flow_type=xflow_mix_type, n_passes=1)

    # Plot
    plt.figure(figsize=(7.5, 5.0))
    sc = plt.scatter(NTU_arr, eps_arr, c=dP_arr, cmap="viridis", s=40, edgecolors="k", linewidths=0.5)
    cbar = plt.colorbar(sc)
    cbar.set_label("Hot-side pressure drop (%)")
    plt.plot(NTU_grid, eps_counter, "r--", label=f"Counterflow bound (C_r={Cr_rep:.3f})")
    plt.plot(NTU_grid, eps_xflow, "b-.", label=f"Crossflow bound {xflow_mix_type} (C_r={Cr_rep:.3f})")
    plt.xlabel("NTU = UA/C_min (-)")
    plt.ylabel("Effectiveness, ε (-)")
    plt.title(f"{case_name}: ε vs NTU as involute angle decreases by 360/n_headers")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_sweep(case="ahjeb_toc_k1", fluid_model="PerfectGas")
