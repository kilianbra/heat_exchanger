"""Solve the inboard involute HX shooting problem for Radial Involute HEx.
Compare the results with Zeli's analysis.
"""

from __future__ import annotations

import logging

from heat_exchanger.epsilon_ntu import epsilon_ntu as _eps_ntu
from heat_exchanger.fluids.protocols import (
    CoolPropFluid,
    FluidInputs,
    PerfectGasFluid,
    RefPropFluid,
)
from heat_exchanger.geometries.radial_spiral import (
    RadialSpiralProtocol,
    RadialSpiralSpec,
    spiral_hex_solver,
)
from heat_exchanger.logging_utils import configure_logging

logger = logging.getLogger(__name__)


WALL_CONDUCTIVITY_304_SS = 14.0
WALL_DENSITY_304_SS = 7930.0
WALL_MATERIAL_304_SS = "304 Stainless Steel"


def load_case(case: str, fluid_model: str = "PerfectGas") -> tuple[RadialSpiralProtocol, FluidInputs, str]:
    """Return (geometry, f_in, case_name) for a named case.

    Supported case identifiers (case-insensitive):
      - ``viper``
      - ``custom`` ()
      - ``ahjeb`` (Base Adv H2 Jet Engine Design)
      - ``ahjeb_toc`` (Top of Climb Variant)
      - ``ahjeb_toc_outb`` (Outboard ToC Variant)
      - ``ahjeb_mto_k1`` (Maximum take-off, k=1 variant)
      - ``chinese`` (preset_chinese == 1, K. He et al. 2024)
    """
    if fluid_model == "PerfectGas":
        air = PerfectGasFluid.from_name("Air")
        helium = PerfectGasFluid.from_name("Helium")
        # h2 = PerfectGasFluid.from_name("H2")  # should be parahydrogen
        h2comb_ahje = PerfectGasFluid.from_name("H2_Combustion_Products_AHJE")
        h2 = PerfectGasFluid.from_name("Para_Hydrogen")
    elif fluid_model == "CoolProp":
        air = CoolPropFluid("Air")
        helium = CoolPropFluid("Helium")
        h2 = CoolPropFluid("ParaHydrogen")
        h2comb_ahje = air
    elif fluid_model == "RefProp":
        air = RefPropFluid("Air")
        helium = RefPropFluid("Helium")
        h2 = RefPropFluid("ParaHydrogen")
        h2comb_ahje = air

    case_key = case.strip().lower()
    for sep in (" ", "-", "/"):
        case_key = case_key.replace(sep, "_")
    aliases = {
        "preset_custom": "custom",
        "custom_design": "custom",
        "preset_ahjeb": "ahjeb",
        "ahje_version_b": "ahjeb",
        "preset_ahjeb_toc": "ahjeb_toc",
        "preset_ahjeb_toc_outb": "ahjeb_toc_outb",
        "preset_chinese": "chinese",
        "k._he_et_al._2024": "chinese",
        "k._he_et_al._2024_case": "chinese",
    }
    canonical = aliases.get(case_key, case_key)

    cases: dict[str, dict[str, object]] = {
        "viper": {
            "case_name": "VIPER (REL)",
            "fluid_hot": air,
            "fluid_cold": helium,
            "Th_in": 298.0,
            "Ph_in": 1.02e5,
            "Tc_in": 96.0,
            "Pc_in": 150e5,
            "tube_outer_diam": 0.98e-3,
            "tube_thick": 0.04e-3,
            "tube_spacing_trv": 2.5,
            "tube_spacing_long": 1.1,
            "staggered": True,
            "n_headers": 31,
            "n_rows_per_header": 4,
            "n_tubes_per_row": 200,
            "radius_outer_hex": 478e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 12.26,
            "mflow_c_total": 1.945,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "custom": {
            "case_name": "AHJE w Viper tube",  # I think No work cycle (unclear if MTO or ToC?
            "fluid_hot": h2comb_ahje,
            "fluid_cold": h2,
            "Th_in": 500.0,
            "Ph_in": 1.02e5,
            "Tc_in": 40.0,
            "Pc_in": 50e5,
            "tube_outer_diam": 0.98e-3,
            "tube_thick": 0.04e-3,
            "tube_spacing_trv": 2.5,
            "tube_spacing_long": 1.5,
            "staggered": True,
            "n_headers": 21,
            "n_rows_per_header": 4,
            "n_tubes_per_row": int(round(540e-3 / (2.5 * 0.98e-3))),  # 220 from axial length
            "radius_outer_hex": 325e-3 + 21 * 4 * 1.5 * 0.98e-3,  # 448.5e-3
            "inv_angle_deg": 360.0,
            "mflow_h_total": 12.0,  # 60 * 0.2
            "mflow_c_total": 0.3,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "ahjeb": {  # Zelim at MTO conditions
            "case_name": "AHJE",  # version B (MTO but no work cycle?)
            "fluid_hot": h2comb_ahje,
            "fluid_cold": h2,
            "Th_in": 500.0,
            "Ph_in": 1.02e5,
            "Tc_in": 40.0,
            "Pc_in": 50e5,
            "tube_outer_diam": 1.067e-3,  # based on 19gT/W from needleworks
            "tube_thick": 0.129e-3,  # based on 19gT/W from needleworks
            "tube_spacing_trv": 2.5,
            "tube_spacing_long": 1.5,
            "staggered": True,
            "n_headers": 21,
            "n_rows_per_header": 4,
            "n_tubes_per_row": int(round(690e-3 / (2.5 * 1.067e-3))),  # 259 from axial length
            "radius_outer_hex": 460e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 12.0,
            "mflow_c_total": 0.3,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "ahjeb_toc": {
            "case_name": "AHJE ToC Work",  # version B - H2TOCv2 ExPHT
            "fluid_hot": h2comb_ahje,
            "fluid_cold": h2,
            "Th_in": 574.0,
            "Ph_in": 0.368e5,
            "Tc_in": 287.0,
            "Pc_in": 150e5,  # This is work cycle - Hydrogen turbine included after
            "tube_outer_diam": 1.067e-3,
            "tube_thick": 0.129e-3,
            "tube_spacing_trv": 3.0,  # Why higher than for ahje?
            "tube_spacing_long": 1.5,
            "staggered": True,
            "n_headers": 21,
            "n_rows_per_header": 4,
            "n_tubes_per_row": int(round(690e-3 / (3.0 * 1.067e-3))),  # 216 tubes per row (Axially)
            "radius_outer_hex": 460e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 12.0,
            "mflow_c_total": 0.76,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "ahjeb_toc_outb": {
            "case_name": "AHJE ToC Outb Work",  # version B - H2TOCv2 ExPHT (Outboard)
            "fluid_hot": h2comb_ahje,
            "fluid_cold": h2,
            "Th_in": 574.0,
            "Ph_in": 0.368e5,
            "Tc_in": 287.0,
            "Pc_in": 150e5,  # This is work cycle - Hydrogen turbine included after
            "tube_outer_diam": 1.067e-3,
            "tube_thick": 0.129e-3,
            "tube_spacing_trv": 3.0,  # Why higher than for ahje?
            "tube_spacing_long": 1.5,
            "staggered": True,
            "n_headers": 21,
            "n_rows_per_header": 4,
            "n_tubes_per_row": int(round(690e-3 / (3.0 * 1.067e-3))),
            "radius_outer_hex": 680e-3 + 21 * 4 * 1.5 * 1.067e-3,  # 0.814 m  From inner radius
            "inv_angle_deg": 360.0,
            "mflow_h_total": 12.0,
            "mflow_c_total": 0.76,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
            "ext_fluid_flows_radially_inwards": False,
        },
        "chinese": {
            "case_name": "K. He 2024",
            "fluid_hot": air,
            "fluid_cold": h2,
            "Th_in": 734.0,
            "Ph_in": 2.62e5,
            "Tc_in": 90.0,
            "Pc_in": 150e5,
            "tube_outer_diam": 1.0e-3,
            "tube_thick": 0.07e-3,
            "tube_spacing_trv": 2.0,
            "tube_spacing_long": 1.5,
            "staggered": True,
            "n_headers": 8,
            "n_rows_per_header": 4,
            "n_tubes_per_row": int(round(2.08 / (2.0 * 1.0e-3))),
            "radius_outer_hex": 0.112 + 8 * 4 * 1.5 * 1.0e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 24.0,
            "mflow_c_total": 2.0,
            "wall_conductivity": 11.4,  # Inconel 718
        },
        "ahjeb_mto_k1": {  # Uses conditions from 28/10 presentation
            "case_name": "AHJE MTO k=1",
            "fluid_hot": h2comb_ahje,
            "fluid_cold": h2,
            "Th_in": 718.1,
            "Ph_out": 1.01e5,
            "Tc_in": 287.0,
            "Pc_in": 96.05e5,
            "tube_outer_diam": 1.067e-3,
            "tube_thick": 0.129e-3,
            "tube_spacing_trv": 3.0,  # Why higher than for ahje?
            "tube_spacing_long": 1.5,
            "staggered": True,
            "n_headers": 11,
            "n_rows_per_header": 4,
            "n_tubes_per_row": int(round(690e-3 / (3.0 * 1.067e-3))),
            "radius_outer_hex": 460e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 162.04 * 0.2,
            "mflow_c_total": 1.316,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "ahjeb_toc_k1": {  # Uses conditions from 28/10 presentation
            "case_name": "AHJE ToC k=1",
            "fluid_hot": h2comb_ahje,
            "fluid_cold": h2,
            "Th_in": 571.65,
            "Ph_out": 0.22631e5,
            "Tc_in": 287.0,
            "Pc_in": 27.9e5,
            "tube_outer_diam": 1.067e-3,
            "tube_thick": 0.129e-3,
            "tube_spacing_trv": 3.0,  # Why higher than for ahje?
            "tube_spacing_long": 1.5,
            "staggered": True,
            "n_headers": 21,
            "n_rows_per_header": 4,
            "n_tubes_per_row": int(round(690e-3 / (3.0 * 1.067e-3))),
            "radius_outer_hex": 460e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 64.3 * 0.2,
            "mflow_c_total": 0.418,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
    }

    if canonical not in cases:
        available = ", ".join(sorted(cases))
        raise ValueError(f"Unknown case '{case}'. Currently supported: {available}.")

    params = dict(cases[canonical])

    # Build geometry
    geom = RadialSpiralSpec(
        tube_outer_diam=params["tube_outer_diam"],
        tube_thick=params["tube_thick"],
        tube_spacing_trv=params["tube_spacing_trv"],
        tube_spacing_long=params["tube_spacing_long"],
        staggered=params["staggered"],
        n_headers=params["n_headers"],
        n_rows_per_header=params["n_rows_per_header"],
        n_tubes_per_row=params["n_tubes_per_row"],
        radius_outer_hex=params["radius_outer_hex"],
        inv_angle_deg=params["inv_angle_deg"],
        wall_conductivity=params["wall_conductivity"],
        ext_fluid_flows_radially_inwards=params.get("ext_fluid_flows_radially_inwards", True),
    )

    # Build f_in
    f_in = FluidInputs(
        hot=params["fluid_hot"],
        cold=params["fluid_cold"],
        m_dot_hot=params["mflow_h_total"],
        m_dot_cold=params["mflow_c_total"],
        Tc_in=params["Tc_in"],
        Pc_in=params["Pc_in"],
        Th_in=params["Th_in"],
        Ph_in=params.get("Ph_in"),
        Ph_out=params.get("Ph_out"),
    )

    return geom, f_in, params["case_name"]


def main(case: str = "viper", fluid_model: str = "PerfectGas") -> None:
    # region logging config
    # logging levels  DEBUG < INFO < WARNING < ERROR < CRITICAL (Default is WARNING)
    configure_logging(logging.INFO)

    # Control logging levels for different modules
    # conservation.py returns a debug message if at any step the properties aren't both conv
    logging.getLogger("heat_exchanger.conservation").setLevel(logging.WARNING)
    # Radial spiral has most of the code, nothing should come out under warning
    logging.getLogger("heat_exchanger.geometries.radial_spiral").setLevel(logging.WARNING)
    # Function specific loggers
    logging.getLogger("heat_exchanger.geometries.radial_spiral.xflow_guess_0d").setLevel(logging.WARNING)
    logging.getLogger("heat_exchanger.geometries.radial_spiral.spiral_hex_solver").setLevel(logging.WARNING)

    # endregion

    geom, f_in, case_name = load_case(case, fluid_model)

    result = spiral_hex_solver(geom, f_in, method="1d")

    final_diag = result["diagnostics"]

    eps_counterflow = _eps_ntu(
        final_diag.get("NTU", float("nan")),
        final_diag.get("Cr", float("nan")),
        exchanger_type="aligned_flow",
        flow_type="counterflow",
        n_passes=1,
    )

    print(
        f"for case {case_name:>20}, NTU={final_diag.get('NTU', float('nan')):5.2f}, "
        f"eps={final_diag.get('epsilon', float('nan')) * 100.0:.2f} %, "
        f"dP_hot={final_diag.get('dP_hot_pct', float('nan')):5.1f} %, "
        f"C_r={final_diag.get('Cr', float('nan')):.3f}, "
        f"eps_xflow(NTU)={eps_counterflow * 100.0:.2f} %"
    )


if __name__ == "__main__":
    fluid_model = "PerfectGas"  # "CoolProp"

    main(case="custom", fluid_model=fluid_model)
    main(case="ahjeb", fluid_model=fluid_model)
    main(case="ahjeb_toc", fluid_model=fluid_model)
    main(case="ahjeb_toc_outb", fluid_model=fluid_model)
    # main(case="viper", fluid_model="RefProp")
    main(case="viper", fluid_model=fluid_model)
    main(case="chinese", fluid_model=fluid_model)

    main(case="ahjeb_MTO_k1", fluid_model=fluid_model)
    main(case="ahjeb_toc_k1", fluid_model=fluid_model)
