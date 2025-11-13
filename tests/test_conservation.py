from __future__ import annotations

import math

import pytest

from heat_exchanger.conservation import update_static_properties
from heat_exchanger.fluids.protocols import PerfectGasFluid


def _compute_residuals(
    fluid: PerfectGasFluid,
    *,
    G: float,
    dh0: float,
    tau_dA_over_A_c: float,
    T_a: float,
    p_b: float,
    T_candidate: float,
    p_candidate: float,
    a_is_in: bool,
    b_is_in: bool,
) -> tuple[float, float, float, float]:
    if b_is_in:
        p_in = p_b
        p_out = p_candidate
    else:
        p_in = p_candidate
        p_out = p_b

    if a_is_in:
        T_in = T_a
        T_out = T_candidate
    else:
        T_in = T_candidate
        T_out = T_a

    state_in = fluid.state(T_in, p_in)
    state_out = fluid.state(T_out, p_out)

    rho_in = state_in.rho
    rho_out = state_out.rho

    h0_in = state_in.h + 0.5 * (G / rho_in) ** 2
    h0_out = state_out.h + 0.5 * (G / rho_out) ** 2

    residual_energy = (h0_out - h0_in) - dh0
    residual_impulse = (p_out + G**2 / rho_out) - (p_in + G**2 / rho_in) + tau_dA_over_A_c

    return residual_energy, residual_impulse, T_in, p_in


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "no_change_inlet",
            "G": 50.0,
            "dh0": 0.0,
            "tau_dA_over_A_c": 0.0,
            "T_a": 500.0,
            "p_b": 150_000.0,
            "a_is_in": True,
            "b_is_in": True,
        },
        {
            "name": "heat_added_pressure_drop",
            "G": 120.0,
            "dh0": 20_000.0,
            "tau_dA_over_A_c": 800.0,
            "T_a": 700.0,
            "p_b": 250_000.0,
            "a_is_in": True,
            "b_is_in": True,
        },
        {
            "name": "cooling_with_known_exit_pressure",
            "G": 90.0,
            "dh0": -15_000.0,
            "tau_dA_over_A_c": 600.0,
            "T_a": 650.0,
            "p_b": 220_000.0,
            "a_is_in": True,
            "b_is_in": False,
        },
        {
            "name": "heating_with_known_exit_temperature",
            "G": 110.0,
            "dh0": 10_000.0,
            "tau_dA_over_A_c": 500.0,
            "T_a": 620.0,
            "p_b": 210_000.0,
            "a_is_in": False,
            "b_is_in": True,
        },
    ],
    ids=lambda case: case["name"],
)
def test_update_static_properties_balances(case, request):
    fluid = PerfectGasFluid.from_name("air")

    tol_T = 1e-3
    rel_tol_p = 1e-3  # percent

    T_solution, p_solution = update_static_properties(
        fluid,
        case["G"],
        case["dh0"],
        case["tau_dA_over_A_c"],
        case["T_a"],
        case["p_b"],
        a_is_in=case["a_is_in"],
        b_is_in=case["b_is_in"],
        tol_T=tol_T,
        rel_tol_p=rel_tol_p,
        max_iter=100,
    )

    residual_h0, residual_fa, T_in, p_in = _compute_residuals(
        fluid,
        G=case["G"],
        dh0=case["dh0"],
        tau_dA_over_A_c=case["tau_dA_over_A_c"],
        T_a=case["T_a"],
        p_b=case["p_b"],
        T_candidate=T_solution,
        p_candidate=p_solution,
        a_is_in=case["a_is_in"],
        b_is_in=case["b_is_in"],
    )

    cp_reference = fluid.state(case["T_a"], case["p_b"]).cp
    tol_dh0 = cp_reference * tol_T
    tol_dFA = (rel_tol_p / 100.0) * case["p_b"]

    ratio_energy = abs(residual_h0) / tol_dh0 if tol_dh0 > 0 else math.inf
    ratio_impulse = abs(residual_fa) / tol_dFA if tol_dFA > 0 else math.inf

    T_out = T_solution if case["a_is_in"] else case["T_a"]
    p_out = p_solution if case["b_is_in"] else case["p_b"]

    # to print this run: uv run pytest tests/test_conservation.py -vv -s
    if request.config.getoption("verbose") > 0:
        message = (
            f"{case['name']}: "
            f"T_in={T_in:.2f} K → T_out={T_out:.2f} K, "
            f"P_in={p_in / 1e5:.3f} bar → P_out={p_out / 1e5:.3f} bar | "
            f"|Δh0|/tol = {ratio_energy:.2e} < 1.0, "
            f"|Δ(p+G²/ρ)|/tol = {ratio_impulse:.2e} < 1.0"
        )
        print(message)  # noqa: T201

    assert ratio_energy < 1.0
    assert ratio_impulse < 1.0
