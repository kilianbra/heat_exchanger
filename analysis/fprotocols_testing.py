import logging

from heat_exchanger.fluids.protocols import CoolPropFluid, PerfectGasFluid, RefPropFluid
from heat_exchanger.logging_utils import configure_logging

if __name__ == "__main__":
    # logging levels  DEBUG < INFO < WARNING < ERROR < CRITICAL (Default is WARNING)
    configure_logging(logging.INFO)

    # Can now either initiaie
    air = PerfectGasFluid(M=28.97, S=111.0, T_ref=273.15, mu_ref=1.716e-5)
    air_state = air.state(T=600.0, P=2e5)
    print(f"Air cp(T=600 K, P=2e5 Pa) = {air_state.cp:.1f} J/(kg·K)")
    print(f"Air mu(T=600 K, P=2e5 Pa) = {air_state.mu:.2e} Pa·s")

    # Or use presets
    air = PerfectGasFluid.from_name("Air")
    air_state = air.state(T=600.0, P=2e5)
    print(f"Air cp(T=600 K, P=2e5 Pa) = {air_state.cp:.1f} J/(kg·K)")
    print(f"Air mu(T=600 K, P=2e5 Pa) = {air_state.mu:.2e} Pa·s")

    print("Available perfect-gas presets: " + ", ".join(PerfectGasFluid.available_presets()))

    hydrogen = PerfectGasFluid.from_name("H2")
    hydrogen_state = hydrogen.state(T=200.0, P=5e5)
    print(f"Hydrogen cp(T=200 K, P=5e5 Pa) = {hydrogen_state.cp:.1f} J/(kg·K)")
    print(f"Hydrogen mu(T=200 K, P=5e5 Pa) = {hydrogen_state.mu:.2e} Pa·s")
    print(f"Hydrogen rho(T=200 K, P=5e5 Pa) = {hydrogen_state.rho:.2f} kg/m^3")

    try:
        PerfectGasFluid.from_name("Argon")
    except ValueError:
        print("Argon preset not yet implemented.")

    air_coolprop = CoolPropFluid("Air")
    air_coolprop_state = air_coolprop.state(T=600.0, P=2e5)
    print(f"Air cp(T=600 K, P=2e5 Pa) = {air_coolprop_state.cp:.1f} J/(kg·K)")
    print(f"Air mu(T=600 K, P=2e5 Pa) = {air_coolprop_state.mu:.2e} Pa·s")

    parahydrogen_coolprop = CoolPropFluid("ParaHydrogen")
    parahydrogen_coolprop_state = parahydrogen_coolprop.state(T=200.0, P=5e5)
    print(f"ParaHydrogen cp(T=200 K, P=5e5 Pa) = {parahydrogen_coolprop_state.cp:.1f} J/(kg·K)")
    print(f"ParaHydrogen mu(T=200 K, P=5e5 Pa) = {parahydrogen_coolprop_state.mu:.2e} Pa·s")
    print(f"ParaHydrogen rho(T=200 K, P=5e5 Pa) = {parahydrogen_coolprop_state.rho:.2f} kg/m^3")

    parahydrogen_refprop = RefPropFluid("ParaHydrogen")
    parahydrogen_refprop_state = parahydrogen_refprop.state(T=200.0, P=5e5)
    print(f"ParaHydrogen cp(T=200 K, P=5e5 Pa) = {parahydrogen_refprop_state.cp:.1f} J/(kg·K)")
    print(f"ParaHydrogen mu(T=200 K, P=5e5 Pa) = {parahydrogen_refprop_state.mu:.2e} Pa·s")
    print(f"ParaHydrogen rho(T=200 K, P=5e5 Pa) = {parahydrogen_refprop_state.rho:.2f} kg/m^3")
