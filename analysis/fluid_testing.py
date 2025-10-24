from tabulate import tabulate
from heat_exchanger.fluid_properties import (
    CoolPropProperties,
    PerfectGasProperties,
    RefPropProperties,
)
from heat_exchanger.fluid_properties import CombustionProductsProperties
import numpy as np
import matplotlib.pyplot as plt

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


def plot_cp_kerosene_products():
    FARs = [0.0, 0.0135, 0.027]
    T_vals = np.linspace(300.0, 1800.0, 200)
    P = 1.0 * BAR_TO_PA

    # REFPROP-first plot
    plt.figure()
    for FAR in FARs:
        model = CombustionProductsProperties(fuel_type="C092H2", FAR_mass=FAR, prefer_refprop=True)
        cps = [model.get_cp(T, P) for T in T_vals]
        plt.plot(T_vals, cps, label=f"FAR={FAR}")
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
        plt.plot(T_vals, cps, label=f"FAR={FAR}")
    plt.xlabel("T [K]")
    plt.ylabel("cp [J/kg-K]")
    plt.title("Kerosene (C092H2) products cp(T), CoolProp-first, p=1 bar")
    plt.legend()
    plt.grid(True, alpha=0.3)


def main():
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

    # Kerosene products cp(T) plots
    plot_cp_kerosene_products()
    plt.show()


if __name__ == "__main__":
    main()
