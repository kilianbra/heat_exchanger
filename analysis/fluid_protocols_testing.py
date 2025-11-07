import logging

from heat_exchanger.fluids.protocols import PerfectGasFluid
from heat_exchanger.logging_utils import configure_logging

if __name__ == "__main__":
    # logging levels  DEBUG < INFO < WARNING < ERROR < CRITICAL (Default is WARNING)
    configure_logging(logging.INFO)

    # Can now either initiaie
    air = PerfectGasFluid(M=28.97, S=111.0, T_ref=273.15, mu_ref=1.716e-5)
    air_state = air.state(T=600.0, P=2e5)
    print("Air cp(T=600 K, P=2e5 Pa) = {air_state.cp:.1f} J/(kg·K)")
    print("Air mu(T=600 K, P=2e5 Pa) = {air_state.mu:.2e} Pa·s")

    # Or use presets
    air = PerfectGasFluid.from_name("Air")
    air_state = air.state(T=600.0, P=2e5)
    print("Air cp(T=600 K, P=2e5 Pa) = {air_state.cp:.1f} J/(kg·K)")
    print("Air mu(T=600 K, P=2e5 Pa) = {air_state.mu:.2e} Pa·s")

    print("Available perfect-gas presets: " + ", ".join(PerfectGasFluid.available_presets()))

    hydrogen = PerfectGasFluid.from_name("H2")
    hydrogen_state = hydrogen.state(T=800.0, P=5e5)
    print("Hydrogen rho(T=800 K, P=5e5 Pa) = {hydrogen_state.rho:.2f} kg/m^3")

    try:
        PerfectGasFluid.from_name("Argon")
    except ValueError:
        print("Argon preset not yet implemented.")
