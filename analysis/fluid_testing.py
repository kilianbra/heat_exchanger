import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from tabulate import tabulate

from heat_exchanger.fluid_properties import (
    CombustionProductsProperties,
    CoolPropProperties,
    PerfectGasProperties,
    RefPropProperties,
)

BAR_TO_PA = 1e5


def build_strategies():
    strategies = {}
    # Perfect gas: rough hydrogen constants
    strategies["PerfectGas-Hydrogen"] = PerfectGasProperties(
        molecular_weight=2.016, gamma=1.4, Pr=0.7, mu_ref=8.4e-6, T_ref=273.15, S=110.4
    )
    # CoolProp HEOS
    try:
        strategies["CoolProp-Hydrogen"] = CoolPropProperties("Hydrogen")
    except Exception as e:
        strategies["CoolProp-Hydrogen"] = f"Error: {e}"
    # ParaHydrogen in CoolProp
    try:
        strategies["CoolProp-ParaHydrogen"] = CoolPropProperties("ParaHydrogen")
    except Exception as e:
        strategies["CoolProp-ParaHydrogen"] = f"Error: {e}"
    # REFPROP hydrogen
    try:
        strategies["REFPROP-Hydrogen"] = RefPropProperties("HYDROGEN")
    except Exception as e:
        strategies["REFPROP-Hydrogen"] = f"Error: {e}"
    # REFPROP para-hydrogen: try common aliases
    refprop_para_aliases = [
        "PARAHYDROGEN",
        "PARA-HYDROGEN",
        "PARA_HYDROGEN",
        "PARAH2",
        "PARA-H2",
        "PARA_H2",
    ]
    refprop_para = None
    for alias in refprop_para_aliases:
        try:
            refprop_para = RefPropProperties(alias)
            break
        except Exception:
            continue
    if refprop_para is None:
        # last resort: sometimes REFPROP mixtures require key like H2;PARAH2 with z
        strategies["REFPROP-ParaHydrogen"] = "Error: No REFPROP para-hydrogen alias worked"
    else:
        strategies["REFPROP-ParaHydrogen"] = refprop_para
    return strategies


def build_hot_models():
    models = {}
    # Perfect gas air constants
    models["PerfectGas-Air"] = PerfectGasProperties(
        molecular_weight=28.97, gamma=1.4, Pr=0.7, mu_ref=1.8e-5, T_ref=300.0, S=110.4
    )
    # CoolProp Air
    try:
        models["CoolProp-Air"] = CoolPropProperties("Air")
    except Exception as e:
        models["CoolProp-Air"] = f"Error: {e}"

    # Combustion products H2 with given FAR
    FAR = 9.95 / (1144 - 9.95)
    try:
        models["CombustionProducts-H2-CoolPropFirst"] = CombustionProductsProperties(
            fuel_type="H2", FAR_mass=FAR, prefer_refprop=False
        )
    except Exception as e:
        models["CombustionProducts-H2-CoolPropFirst"] = f"Error: {e}"

    try:
        models["CombustionProducts-H2-RefpropFirst"] = CombustionProductsProperties(
            fuel_type="H2", FAR_mass=FAR, prefer_refprop=True
        )
    except Exception as e:
        models["CombustionProducts-H2-RefpropFirst"] = f"Error: {e}"

    return models


def query_properties(strategy, T, P):
    try:
        if hasattr(strategy, "get_transport_properties"):
            rho, cp, mu, k = strategy.get_transport_properties(T, P)
            return {
                "rho [kg/m^3]": rho,
                "cp [J/kg-K]": cp,
                "mu [Pa·s]": mu,
                "k [W/m-K]": k,
            }
        else:
            return (
                {"error": str(strategy)}
                if isinstance(strategy, str)
                else {"error": "Unsupported strategy"}
            )
    except Exception as e:
        return {"error": str(e)}


def print_table(title, rows):
    print(f"\n{title}")
    print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".6e"))


def plot_cp_kerosene_products_three_FAR_as_function_of_temperature():
    """
    This function plots the cp of kerosene products for different FARs at 1 bar.
    It was visually inspected on 27/10/2025 that Refprop returns errors for a few temperature values
    at FAR = 0.0135 but other than that agrees quite well with the lecture not c_p up until about 1200 K
    Then the c_p returned is lower than that in the reference plot. Hence quite good agreement.
    The coolprop approach only works until about 600 K and then all collapses to the same value as FAR=0.

    """
    FARs = [0.0, 0.0135, 0.027]
    # H2_FARs = [0.0, 0.0036, 0.0072]
    T_vals = np.linspace(300.0, 1800.0, 50)
    P = 1.0 * BAR_TO_PA

    # REFPROP-first plot
    plt.figure()
    for FAR in FARs:
        model = CombustionProductsProperties(fuel_type="C092H2", FAR_mass=FAR, prefer_refprop=True)
        cps = [model.get_cp(T, P) for T in T_vals]
        plt.scatter(T_vals, cps, marker="+", label=f"FAR={FAR}")
    plt.ylim([1e3, 1.4e3])
    plt.xlabel("T [K]")
    plt.ylabel("cp [J/kg-K]")
    plt.title("Kerosene (C092H2) products cp(T), REFPROP-first, p=1 bar")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # CoolProp-first plot
    plt.figure()
    for FAR in FARs:
        model = CombustionProductsProperties(fuel_type="C092H2", FAR_mass=FAR, prefer_refprop=False)
        cps = [model.get_cp(T, P) for T in T_vals]
        plt.scatter(T_vals, cps, marker="+", label=f"FAR={FAR}")
    # plt.ylim([1e3, 1.4e3])
    plt.xlabel("T [K]")
    plt.ylabel("cp [J/kg-K]")
    plt.title("Kerosene (C092H2) products cp(T), CoolProp-first, p=1 bar")
    plt.legend()
    plt.grid(True, alpha=0.3)


def plot_mu_combustion_products_one_temp_as_function_of_FAR(
    T: float,
    P: float = 1.0 * BAR_TO_PA,
    FAR_H2: np.ndarray | None = None,
    FAR_kero: np.ndarray | None = None,
    prefer_refprop: bool = True,
) -> None:
    """
    Plot relative change in viscosity vs air as a function of FAR at fixed temperature.

    Two curves are shown:
    - H2 combustion products (blue)
    - Kerosene (C0.92H2) combustion products (red)

    The y-axis shows (mu_mix - mu_air)/mu_air in percent. The axis label includes mu_air.
    """
    if FAR_H2 is None:
        FAR_H2 = np.linspace(0.002, 0.015, 20)
    if FAR_kero is None:
        FAR_kero = np.linspace(0.02, 0.05, 20)

    # Baseline air viscosity at given T, 1 bar by default
    try:
        air = CoolPropProperties("Air")
        mu_air = air.get_viscosity(T, P)
    except Exception:
        # Fallback to a simple perfect gas estimate if CoolProp fails
        air_pg = PerfectGasProperties(
            molecular_weight=28.97, gamma=1.4, Pr=0.7, mu_ref=1.8e-5, T_ref=300.0, S=110.4
        )
        mu_air = air_pg.get_viscosity(T, P)

    rel_mu_H2 = []
    for far in FAR_H2:
        try:
            mix = CombustionProductsProperties("H2", far, prefer_refprop=prefer_refprop)
            mu = mix.get_viscosity(T, P)
            rel_mu_H2.append(mu / mu_air - 1.0)
        except Exception:
            rel_mu_H2.append(np.nan)

    rel_mu_kero = []
    for far in FAR_kero:
        try:
            mix = CombustionProductsProperties("C092H2", far, prefer_refprop=prefer_refprop)
            mu = mix.get_viscosity(T, P)
            rel_mu_kero.append(mu / mu_air - 1.0)
        except Exception:
            rel_mu_kero.append(np.nan)

    fig, ax = plt.subplots()
    ax.plot(FAR_H2, rel_mu_H2, label="H2 products", color="blue")
    ax.plot(FAR_kero, rel_mu_kero, label="C0.92H2 products", color="red")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("FAR [-]")
    ax.set_ylabel(f"Relative viscosity vs air at {T:.0f} K (mu_air={mu_air:.3e} Pa·s)")
    ax.set_title(
        f"Viscosity change of combustion products vs air at T={T:.0f} K, p={P / BAR_TO_PA:.1f} bar"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()


def main_table():
    # Cold side (H2 / p-H2) at 28 bar
    P_cold = 28.0 * BAR_TO_PA
    cold_states = [(264.0, P_cold), (677.0, P_cold)]
    strategies = build_strategies()

    for T, Pstate in cold_states:
        rows = []
        for name, strat in strategies.items():
            props = query_properties(strat, T, Pstate)
            row = {"Model": name}
            if "error" in props:
                row.update(
                    {
                        "rho [kg/m^3]": None,
                        "cp [J/kg-K]": None,
                        "mu [Pa·s]": None,
                        "k [W/m-K]": None,
                        "error": props["error"],
                    }
                )
            else:
                row.update(props)
            rows.append(row)
        print_table(f"Transport properties at T={T:.1f} K, P=28 bar", rows)

    # Hot side at 0.4 bar
    P_hot = 0.4 * BAR_TO_PA
    hot_states = [(778.0, P_hot), (733.0, P_hot)]
    models = build_hot_models()

    for T, Pstate in hot_states:
        rows = []
        for name, strat in models.items():
            props = query_properties(strat, T, Pstate)
            row = {"Model": name}
            if "error" in props:
                row.update(
                    {
                        "rho [kg/m^3]": None,
                        "cp [J/kg-K]": None,
                        "mu [Pa·s]": None,
                        "k [W/m-K]": None,
                        "error": props["error"],
                    }
                )
            else:
                row.update(props)
            rows.append(row)
        print_table(f"Hot-side transport properties at T={T:.1f} K, P=0.4 bar", rows)


if __name__ == "__main__":
    FAR = 0.0135
    P = 1.0e5
    model = CombustionProductsProperties(fuel_type="C092H2", FAR_mass=FAR, prefer_refprop=True)
    for T in np.linspace(630.0, 1000.0, 5):
        # cps = model.get_cp(T, P)
        cps = model.mixture_state.cpmass()
        print(f"T={T:.1f} K, cp={cps:.3e} J/kg-K")
    # main_table()
    # Kerosene products cp(T) plots
    plot_cp_kerosene_products_three_FAR_as_function_of_temperature()
    plt.show()

    # plot_mu_combustion_products_one_temp_as_function_of_FAR(T=750.0, P=0.4 * BAR_TO_PA, FAR_H2=None, FAR_kero=None, prefer_refprop=True)
    # plt.show()
