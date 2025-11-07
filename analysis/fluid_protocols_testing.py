import logging

from heat_exchanger.fluids.protocols import FluidModel, PerfectGasFluid


def simulate(fluid: FluidModel):
    hot_in = fluid.state(T=600.0, P=2e5)
    cp_hot_in = hot_in.cp
    print(f"cp: {cp_hot_in:.1f} J/(kg·K)")


if __name__ == "__main__":
    # logging levels  DEBUG < INFO < WARNING < ERROR < CRITICAL (Default is WARNING)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    simulate(PerfectGasFluid(molecular_weight=28.97))
