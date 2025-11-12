"""Solve the inboard involute HX shooting problem for Radial Involute HEx.
Compare the results with Zeli's analysis.
"""

from __future__ import annotations

import logging

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
from heat_exchanger.fluids.protocols import (
    CoolPropFluid,
    FluidInputs,
    PerfectGasFluid,
    RefPropFluid,
)
from heat_exchanger.geometries.radial_spiral import (
    RadialSpiralProtocol,
    RadialSpiralSpec,
    compute_overall_performance,
    rad_spiral_shoot,
)
from heat_exchanger.logging_utils import configure_logging

logger = logging.getLogger(__name__)


WALL_CONDUCTIVITY_304_SS = 14.0
WALL_DENSITY_304_SS = 7930.0
WALL_MATERIAL_304_SS = "304 Stainless Steel"


def load_case(case: str, fluid_model: str = "PerfectGas") -> tuple[RadialSpiralProtocol, FluidInputs, str]:
    """Return (geometry, inputs, case_name) for a named case.

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
        h2 = PerfectGasFluid.from_name("H2")  # should be parahydrogen
    elif fluid_model == "CoolProp":
        air = CoolPropFluid("Air")
        helium = CoolPropFluid("Helium")
        h2 = CoolPropFluid("ParaHydrogen")
    elif fluid_model == "RefProp":
        air = RefPropFluid("Air")
        helium = RefPropFluid("Helium")
        h2 = RefPropFluid("ParaHydrogen")

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
            "radius_outer_whole_hex": 478e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 12.26,
            "mflow_c_total": 1.945,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "custom": {
            "case_name": "Custom Design",
            "fluid_hot": air,
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
            "radius_outer_whole_hex": 325e-3 + 21 * 4 * 1.5 * 0.98e-3,  # 448.5e-3
            "inv_angle_deg": 360.0,
            "mflow_h_total": 12.0,  # 60 * 0.2
            "mflow_c_total": 0.3,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "ahjeb": {
            "case_name": "AHJE",  # version B
            "fluid_hot": air,
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
            "radius_outer_whole_hex": 460e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 12.0,
            "mflow_c_total": 0.3,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "ahjeb_toc": {
            "case_name": "AHJE ToC",  # version B - H2TOCv2 ExPHT
            "fluid_hot": air,
            "fluid_cold": h2,
            "Th_in": 574.0,
            "Ph_in": 0.368e5,
            "Tc_in": 287.0,
            "Pc_in": 150e5,
            "tube_outer_diam": 1.067e-3,
            "tube_thick": 0.129e-3,
            "tube_spacing_trv": 3.0,  # Why higher than for ahje?
            "tube_spacing_long": 1.5,
            "staggered": True,
            "n_headers": 21,
            "n_rows_per_header": 4,
            "n_tubes_per_row": int(round(690e-3 / (3.0 * 1.067e-3))),  # 216 tubes per row (Axially)
            "radius_outer_whole_hex": 460e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 12.0,
            "mflow_c_total": 0.76,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "ahjeb_toc_outb": {
            "case_name": "AHJE ToC Outb",  # version B - H2TOCv2 ExPHT (Outboard)
            "fluid_hot": air,
            "fluid_cold": h2,
            "Th_in": 574.0,
            "Ph_in": 0.368e5,
            "Tc_in": 287.0,
            "Pc_in": 150e5,
            "tube_outer_diam": 1.067e-3,
            "tube_thick": 0.129e-3,
            "tube_spacing_trv": 3.0,  # Why higher than for ahje?
            "tube_spacing_long": 1.5,
            "staggered": True,
            "n_headers": 21,
            "n_rows_per_header": 4,
            "n_tubes_per_row": int(round(690e-3 / (3.0 * 1.067e-3))),
            "radius_outer_whole_hex": 680e-3 + 21 * 4 * 1.5 * 1.067e-3,  # 0.814 m  From inner radius
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
            "radius_outer_whole_hex": 0.112 + 8 * 4 * 1.5 * 1.0e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 24.0,
            "mflow_c_total": 2.0,
            "wall_conductivity": 11.4,  # Inconel 718
        },
        "ahjeb_mto_k1": {  # Uses conditions from 28/10 presentation
            "case_name": "AHJE MTO k=1",
            "fluid_hot": air,
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
            "radius_outer_whole_hex": 460e-3,
            "inv_angle_deg": 360.0,
            "mflow_h_total": 162.04 * 0.2,
            "mflow_c_total": 1.316,
            "wall_conductivity": WALL_CONDUCTIVITY_304_SS,
        },
        "ahjeb_toc_k1": {  # Uses conditions from 28/10 presentation
            "case_name": "AHJE ToC k=1",
            "fluid_hot": air,
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
            "radius_outer_whole_hex": 460e-3,
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
        radius_outer_whole_hex=params["radius_outer_whole_hex"],
        inv_angle_deg=params["inv_angle_deg"],
        wall_conductivity=params["wall_conductivity"],
        ext_fluid_flows_radially_inwards=params.get("ext_fluid_flows_radially_inwards", True),
    )

    # Build inputs
    inputs = FluidInputs(
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

    return geom, inputs, params["case_name"]


def _compute_spiral_length(radius_inner: float, radius_outer: float, inv_angle_deg: float, n_points: int = 64) -> float:
    theta_vals = np.linspace(0.0, np.deg2rad(inv_angle_deg), n_points)
    b = (radius_outer - radius_inner) / np.deg2rad(inv_angle_deg)
    r_vals = radius_inner + b * theta_vals
    return float(np.trapezoid(np.sqrt(r_vals**2 + b**2), theta_vals))


def _initial_guess_two_step_xflow(
    geom: RadialSpiralProtocol,
    inputs: FluidInputs,
) -> tuple[float, float]:
    """Return (Th_inner_guess, Ph_other_guess) using a two-step 0D estimate.

    The first step evaluates properties at the inlet conditions; the second step
    re-evaluates at the mean of the inlet and the first-step outlet to refine the
    guess.
    """

    # Unpack inputs
    # Convenience aliases (optional)
    fluid_hot = inputs.hot  # noqa: F841
    fluid_cold = inputs.cold  # noqa: F841
    mflow_h_total = inputs.m_dot_hot
    mflow_c_total = inputs.m_dot_cold
    Th_in = inputs.Th_in
    Tc_in = inputs.Tc_in
    Pc_in = inputs.Pc_in
    Ph_in = inputs.Ph_in
    Ph_out = inputs.Ph_out

    # Validate boundary pressure specification
    if (Ph_in is None and Ph_out is None) or (Ph_in is not None and Ph_out is not None):
        raise ValueError("Specify exactly one of Ph_in or Ph_out in FluidInputs.")

    tube_ID = geom.tube_inner_diam
    r_in = geom.radius_inner_whole_hex
    r_out = geom.radius_outer_whole_hex
    Lx = geom.axial_length

    # Global single-tube areas using total involute length
    inv_len = _compute_spiral_length(r_in, r_out, geom.inv_angle_deg)
    A_ht_hot_one = np.pi * geom.tube_outer_diam * inv_len
    A_ht_cold_one = np.pi * tube_ID * inv_len

    n_rows_radial = geom.n_rows_per_header * geom.n_headers
    n_tubes_total = geom.n_tubes_total
    A_total_hot0 = A_ht_hot_one * n_tubes_total
    A_total_cold0 = A_ht_cold_one * n_tubes_total

    # Frontal/free areas at mid-radius and total cold frontal area (per sector/header)
    r_mid = 0.5 * (r_in + r_out)
    Afr_hot_mid = Lx * 2.0 * np.pi * r_mid
    Aff_hot_mid = Afr_hot_mid * geom.sigma_outer

    Aff_cold_total = n_tubes_total * np.pi * tube_ID**2 / 4.0

    G_h0 = mflow_h_total / Aff_hot_mid
    G_c0 = mflow_c_total / Aff_cold_total

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
        sh = fluid_hot.state(Th_eval, Ph_eval)
        sc = fluid_cold.state(Tc_eval, Pc_eval)
        Pr_h = sh.mu * sh.cp / sh.k
        Pr_c = sc.mu * sc.cp / sc.k
        rho_h = sh.rho
        rho_c = sc.rho
        Re_h_OD = G_h0 * geom.tube_outer_diam / sh.mu
        Re_c = G_c0 * tube_ID / sc.mu

        Nu_h, f_h = _bank_corr(
            Re_h_OD,
            geom.tube_spacing_long,
            geom.tube_spacing_trv,
            Pr_h,
            inline=(not geom.staggered),
            n_rows=n_rows_radial,
        )
        logger.debug(
            "0D guess tube bank for Re_od=%5.2e: St_h=%5.2f, f_h=%5.2e",
            Re_h_OD,
            Nu_h / Re_h_OD / Pr_h,
            f_h,
        )
        Nu_c = _circ_nu(Re_c, 0, prandtl=Pr_c)
        f_c = _circ_fric(Re_c, 0)

        h_h = Nu_h * sh.k / geom.tube_outer_diam
        h_c = Nu_c * sc.k / tube_ID

        wall_term = geom.tube_outer_diam / (2.0 * geom.wall_conductivity) * np.log(geom.tube_outer_diam / tube_ID)
        U = 1.0 / (1.0 / h_h + 1.0 / h_c * (geom.tube_outer_diam / tube_ID) + wall_term)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "0D guess geometry ratios: UA_master=%5.3e, A_hot/Aff_hot=%5.3e, A_cold/Aff_cold=%5.3e",
                U * A_total_hot0,
                A_total_hot0 / Aff_hot_mid,
                A_total_cold0 / Aff_cold_total,
            )

        C_h_tot = mflow_h_total * sh.cp
        C_c_tot = mflow_c_total * sc.cp
        C_min = min(C_h_tot, C_c_tot)
        C_max = max(C_h_tot, C_c_tot)
        Cr = C_min / C_max

        NTU = U * A_total_hot0 / C_min
        eps = _eps_ntu(NTU, Cr, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)
        logger.debug("0D guess epsilon-NTU: NTU=%5.2f, eps=%5.2f", NTU, eps)
        Q = eps * C_min * (_Th_in - _Tc_in)

        tau_h = f_h * (A_total_hot0 / Aff_hot_mid) * (G_h0**2) / (2.0 * rho_h)
        tau_c = f_c * (A_total_cold0 / Aff_cold_total) * (G_c0**2) / (2.0 * rho_c)

        dh0_h = -Q / mflow_h_total
        dh0_c = Q / mflow_c_total
        Th_out, Ph_not_b = _upd_stat_prop(
            fluid_hot,
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
            fluid_cold,
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

    logger.debug(
        "0D guess 0: Th_in =%5.2f K, Tc_in =%5.2f K, Ph_in =%s Pa, Pc_in =%5.2e Pa (Ph_out=%s)",
        Th_in,
        Tc_in,
        f"{Ph_in:.2e}" if Ph_in is not None else "N/A",
        Pc_in,
        f"{Ph_out:.2e}" if Ph_out is not None else "N/A",
    )
    Th_o1, Tc_o1, Ph_not_b1, Pc_o1 = _0d_xflow_guess(
        Th_in,
        Ph_in if Ph_in is not None else float("nan"),
        Tc_in,
        Pc_in,
        Th_in,
        Ph_in if Ph_out is None else Ph_out,
        Tc_in,
        Pc_in,
        _Ph_out=Ph_out,
    )
    if Ph_out is not None:
        logger.debug(
            "0D guess 1: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa (Ph_in_guess=%5.2e Pa)",
            Th_o1,
            Tc_o1,
            Ph_out,
            Pc_o1,
            Ph_not_b1,
        )
    else:
        logger.debug(
            "0D guess 1: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa",
            Th_o1,
            Tc_o1,
            Ph_not_b1,
            Pc_o1,
        )
    Th_mean = 0.5 * (Th_in + Th_o1)
    Tc_mean = 0.5 * (Tc_in + Tc_o1)
    Ph_known = Ph_out if Ph_out is not None else Ph_in
    if Ph_known is None:
        raise ValueError("Either Ph_in or Ph_out must be provided for the 0D guess.")
    Ph_mean = 0.5 * (Ph_known + Ph_not_b1)
    Pc_mean = 0.5 * (Pc_in + Pc_o1)
    Th_o2, Tc_o2, Ph_not_b2, Pc_o2 = _0d_xflow_guess(
        Th_in,
        Ph_in if Ph_in is not None else float("nan"),
        Tc_in,
        Pc_in,
        Th_mean,
        Ph_mean,
        Tc_mean,
        Pc_mean,
        _Ph_out=Ph_out,
    )
    if Ph_out is not None:
        logger.debug(
            "0D guess 2: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa (Ph_in_guess=%5.2e Pa)",
            Th_o2,
            Tc_o2,
            Ph_out,
            Pc_o2,
            Ph_not_b2,
        )
    else:
        logger.debug(
            "0D guess 2: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa",
            Th_o2,
            Tc_o2,
            Ph_not_b2,
            Pc_o2,
        )
    # Hot inner boundary (inboard shoot) guess equals the 0D outlet
    return float(Th_o2), float(Ph_not_b2)


def main(case: str = "viper", fluid_model: str = "PerfectGas") -> None:
    # logging levels  DEBUG < INFO < WARNING < ERROR < CRITICAL (Default is WARNING)
    configure_logging(logging.WARNING)

    # Control logging levels for different modules
    # Suppress debug from conservation.py
    logging.getLogger("heat_exchanger.conservation").setLevel(logging.WARNING)
    logging.getLogger("heat_exchanger.geometries.radial_spiral").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.DEBUG)  # Main loop stays at INFO

    geom, inputs, case_name = load_case(case, fluid_model)

    # 0D two-step initial guess assuming counterflow epsilon-NTU relationship
    # (first using inlet properties as average, then using average of first guess
    # outlet and known inlet to get a better average properties guess)
    Ph_in_known = inputs.Ph_in
    Ph_out_known = inputs.Ph_out
    Th0, Ph0 = _initial_guess_two_step_xflow(geom, inputs)
    logger.info(
        "Hot inlet parameters: \t \t \t Th_in =%.2f K, Ph_known =%.2e Pa (at %s)",
        inputs.Th_in,
        Ph_in_known if Ph_in_known is not None else Ph_out_known,
        "inlet" if Ph_in_known is not None else "outlet",
    )
    dP_P_in = (1 - Ph0 / Ph_in_known) * 100.0 if Ph_in_known is not None else (1 - Ph_out_known / Ph0) * 100.0
    logger.info(
        "Case %10s: 0D guess (inner b.) \t Th_out=%.2f K, ΔPh/Ph_in=%.1f %%",
        case_name,
        Th0,
        dP_P_in,
    )

    eval_state = {"count": 0}
    tol_root = 1.0e-2
    tol_P_pct_of_Ph_in = 0.1
    tol_P = (
        tol_P_pct_of_Ph_in / 100 * (Ph_in_known if Ph_in_known is not None else Ph_out_known)
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
        raw = rad_spiral_shoot(x, geom, inputs, property_solver_it_max=40, property_solver_tol_T=1e-2, rel_tol_p=1e-3)
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

    x0 = np.array([Th0, Ph0], dtype=float) if Ph_in_known is not None else np.array([Th0], dtype=float)
    sol = root(residuals, x0, method="hybr", tol=tol_root, options={"maxfev": 60})

    result = compute_overall_performance(
        sol.x, geom, inputs, property_solver_it_max=40, property_solver_tol_T=1e-2, rel_tol_p=1e-3
    )
    final_diag: dict[str, float] = result["diagnostics"]  # type: ignore[assignment]
    final_raw = result["residuals"]  # type: ignore[assignment]

    logger.debug(
        "SciPy root success=%s, nfev=%s, message=%s",
        sol.success,
        getattr(sol, "nfev", None),
        sol.message,
    )
    dP_P_in = (
        (1 - sol.x[1] / Ph_in_known) * 100.0 if Ph_in_known is not None else final_diag.get("dP_hot_pct", float("nan"))
    )
    logger.info(
        "Solution after %d iterations: \t \t Th_out=%.2f K, ΔPh/Ph_in=%.1f %%",
        eval_state["count"],
        sol.x[0],
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
        eps_counterflow = _eps_ntu(
            final_diag.get("NTU", float("nan")),
            final_diag.get("Cr", float("nan")),
            exchanger_type="aligned_flow",
            flow_type="counterflow",
            n_passes=1,
        )
        logger.info(
            "Performance: ε=%.3f (vs xflow ε=%.3f), NTU=%.2f, Cr=%.3f, Q_hot=%.2f MW, Q_cold=%.2f MW",
            final_diag.get("epsilon", float("nan")),
            eps_counterflow,
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
        logger.warning(
            "for case %20s, NTU=%5.2f, eps=%.2f %%, dP_hot=%5.1f %%, C_r=%.3f, eps_xflow=%.2f %%",
            case_name,
            final_diag.get("NTU", float("nan")),
            final_diag.get("epsilon", float("nan")) * 100.0,
            final_diag.get("dP_hot_pct", float("nan")),
            final_diag.get("Cr", float("nan")),
            eps_counterflow * 100.0,
        )


if __name__ == "__main__":
    # main(case="custom", fluid_model="CoolProp")
    # main(case="ahjeb", fluid_model="CoolProp")
    main(case="ahjeb_toc", fluid_model="CoolProp")
    # main(case="ahjeb_toc_outb", fluid_model="CoolProp")
    # # main(case="viper", fluid_model="RefProp")
    # main(case="viper", fluid_model="CoolProp")
    # main(case="chinese", fluid_model="CoolProp")

    # main(case="ahjeb_MTO_k1", fluid_model="CoolProp")
    # main(case="ahjeb_toc_k1", fluid_model="CoolProp")
