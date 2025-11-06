import numpy as np
import pytest

from heat_exchanger.conservation import update_static_properties
from heat_exchanger.fluid_properties import PerfectGasProperties


@pytest.mark.parametrize(
    "a_is_in,b_is_in",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_update_static_properties_air_ideal(a_is_in, b_is_in):
    # Ideal gas air properties (constant cp)
    air = PerfectGasProperties(molecular_weight=28.97, gamma=1.4)

    # Reference state
    T_ref = 300.0  # K
    p_ref = 101325.0  # Pa
    rho_ref = air.get_density(T_ref, p_ref)

    # Inputs
    G = 20.0  # kg/m^2/s (arbitrary positive mass velocity)
    dh0 = 1e3 * 100.0  # J/kg
    tau_dA_over_A_c = 100.0  # Pa

    # Provide known side values
    T_a = T_ref
    rho_a = rho_ref
    p_b = p_ref

    T_not_a, p_not_b, rho_not_a = update_static_properties(
        fluid_props=air,
        G=G,
        dh0=dh0,
        tau_dA_over_A_c=tau_dA_over_A_c,
        T_a=T_a,
        rho_a=rho_a,
        p_b=p_b,
        a_is_in=a_is_in,
        b_is_in=b_is_in,
        max_iter=25,  # give a bit more headroom
    )

    # Basic sanity checks: finite, positive where expected
    assert np.isfinite(T_not_a) and T_not_a > 0.0
    assert np.isfinite(p_not_b) and p_not_b > 0.0
    assert np.isfinite(rho_not_a) and rho_not_a > 0.0
    if b_is_in:
        assert p_not_b < p_b
    else:
        assert p_not_b > p_b
