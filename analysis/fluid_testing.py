from contextlib import suppress
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from tabulate import tabulate

from heat_exchanger.fluid_properties import (
    CombustionProductsProperties,
    CoolPropProperties,
    PerfectGasProperties,
    RefPropProperties,
    configure_refprop,
)
from heat_exchanger.logging_utils import configure_logging

BAR_TO_PA = 1e5


logger = logging.getLogger(__name__)


def build_strategies():
    strategies = {}
    # Perfect gas: rough hydrogen constants
    strategies["PerfectGas-H2"] = PerfectGasProperties(
        molecular_weight=2.016, gamma=1.4, Pr=0.7, mu_ref=8.4e-6, T_ref=273.15, S=110.4
    )
    # CoolProp HEOS
    try:
        strategies["CoolProp-H2"] = CoolPropProperties("Hydrogen")
    except Exception as e:
        strategies["CoolProp-H2"] = f"Error: {e}"
    # ParaHydrogen in CoolProp
    try:
        strategies["CoolProp-ParaH2"] = CoolPropProperties("ParaHydrogen")
    except Exception as e:
        strategies["CoolProp-ParaH2"] = f"Error: {e}"
    # REFPROP hydrogen
    try:
        strategies["REFPROP-H2"] = RefPropProperties("HYDROGEN")
    except Exception as e:
        strategies["REFPROP-H2"] = f"Error: {e}"
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
        strategies["REFPROP-ParaH2"] = "Error: No REFPROP para-hydrogen alias worked"
    else:
        strategies["REFPROP-ParaH2"] = refprop_para
    return strategies


def build_h2_flue_models(FAR: float = 9.95 / (1144 - 9.95)):
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

    try:
        models["CombProdH2-CP"] = CombustionProductsProperties(fuel_type="H2", FAR_mass=FAR, prefer_refprop=False)
    except Exception as e:
        models["CombProdH2-CP"] = f"Error: {e}"

    try:
        models["CombProdH2-RP"] = CombustionProductsProperties(fuel_type="H2", FAR_mass=FAR, prefer_refprop=True)
    except Exception as e:
        models["CombProdH2-RP"] = f"Error: {e}"

    return models


def query_properties(strategy, T, P):
    try:
        if hasattr(strategy, "get_transport_properties"):
            rho, cp, mu, k = strategy.get_transport_properties(T, P)
            # Compute Pr and gamma
            pr_val = (cp * mu / k) if (k is not None and k != 0.0) else float("nan")
            gamma_val = float("nan")
            try:
                # Preferred: use specific gas constant when available
                if hasattr(strategy, "R_specific"):
                    cv = cp - strategy.R_specific
                    gamma_val = (cp / cv) if cv != 0.0 else float("nan")
                # CoolProp/RefProp pure fluids
                elif hasattr(strategy, "CP") and hasattr(strategy, "fluid"):
                    cv = strategy.CP.PropsSI("Cvmass", "T", T, "P", P, strategy.fluid)
                    gamma_val = (cp / cv) if cv != 0.0 else float("nan")
                # Mixtures/combustion products via mixture_state (REFPROP)
                elif hasattr(strategy, "mixture_state") and strategy.mixture_state is not None:
                    # Use low-level interface if available
                    strategy.mixture_state.update(strategy.mixture.CP.PT_INPUTS, P, T)
                    cv = strategy.mixture_state.cvmass()
                    gamma_val = (cp / cv) if cv != 0.0 else float("nan")
            except Exception:
                gamma_val = float("nan")

            return {
                "rho [kg/m^3]": rho,
                "cp [J/kg-K]": cp,
                "mu [Pa·s]": mu,
                # Format Pr and gamma to two decimals for legibility in tables
                "Pr [-]": f"{pr_val:.2f}" if np.isfinite(pr_val) else "nan",
                "gamma [-]": f"{gamma_val:.2f}" if np.isfinite(gamma_val) else "nan",
            }
        else:
            return {"error": str(strategy)} if isinstance(strategy, str) else {"error": "Unsupported strategy"}
    except Exception as e:
        return {"error": str(e)}


def print_table(title, rows):
    print(f"\n{title}")
    # Ensure Pr and gamma print with two decimals regardless of underlying types
    formatted_rows = []
    for row in rows:
        new_row = dict(row)
        for col in ("Pr [-]", "gamma [-]"):
            if col in new_row and new_row[col] is not None:
                with suppress(Exception):
                    new_row[col] = f"{float(new_row[col]):.2f}"
        formatted_rows.append(new_row)
    # Compute disable_numparse indices based on first-row key order
    col_names = list(formatted_rows[0].keys()) if formatted_rows else []
    disable_cols: list[int] = []
    for name in ("Pr [-]", "gamma [-]"):
        if name in col_names:
            disable_cols.append(col_names.index(name))
    print(
        tabulate(
            formatted_rows,
            headers="keys",
            tablefmt="github",
            floatfmt=".6e",
            disable_numparse=disable_cols,
        )
    )


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
        air_pg = PerfectGasProperties(molecular_weight=28.97, gamma=1.4, Pr=0.7, mu_ref=1.8e-5, T_ref=300.0, S=110.4)
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
    ax.set_title(f"Viscosity change of combustion products vs air at T={T:.0f} K, p={P / BAR_TO_PA:.1f} bar")
    ax.grid(True, alpha=0.3)
    ax.legend()


def main_table():
    print("Brewer Fluid Properties")
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
                        "Pr [-]": None,
                        "gamma [-]": None,
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
    FAR_Brewer = 9.95 / (1144 - 9.95)
    print(f"Brewer FAR = 9.95 / (1144 - 9.95) = {FAR_Brewer:.6f}")
    models = build_h2_flue_models(FAR_Brewer)

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
                        "Pr [-]": None,
                        "gamma [-]": None,
                        "error": props["error"],
                    }
                )
            else:
                row.update(props)
            rows.append(row)
        print_table(f"Hot-side transport properties at T={T:.1f} K, P=0.4 bar", rows)

    # Helper to choose a preferred valid model
    def pick_first_valid(models_dict, preferred_keys):
        for key in preferred_keys:
            if key in models_dict:
                val = models_dict[key]
                if not isinstance(val, str) and hasattr(val, "get_specific_enthalpy"):
                    return key, val
        # Fallback: first valid
        for key, val in models_dict.items():
            if not isinstance(val, str) and hasattr(val, "get_specific_enthalpy"):
                return key, val
        return None, None

    # Mean cp for cold stream (use preferred hydrogen/para-hydrogen model at 28 bar)
    (T_c_in, P_cold) = cold_states[0]
    (T_c_out, _) = cold_states[1]
    cold_key, cold_model = pick_first_valid(
        strategies,
        [
            "REFPROP-ParaH2",
            "REFPROP-H2",
            "CoolProp-ParaH2",
            "CoolProp-H2",
            "PerfectGas-H2",
        ],
    )
    if cold_model is not None:
        h_c_in = cold_model.get_specific_enthalpy(T_c_in, P_cold)
        h_c_out = cold_model.get_specific_enthalpy(T_c_out, P_cold)
        cp_mean_cold = (h_c_out - h_c_in) / (T_c_out - T_c_in)
        cp_c_in = cold_model.get_cp(T_c_in, P_cold)
        cp_c_out = cold_model.get_cp(T_c_out, P_cold)
        c_in_pct = (cp_c_in / cp_mean_cold - 1.0) * 100.0
        c_out_pct = (cp_c_out / cp_mean_cold - 1.0) * 100.0
    else:
        cp_mean_cold = float("nan")
        c_in_pct = float("nan")
        c_out_pct = float("nan")
        cold_key = "n/a"

    # Mean cp for hot stream (use preferred combustion products model at 0.4 bar)
    (T_h_in, P_hot) = hot_states[0]
    (T_h_out, _) = hot_states[1]
    hot_key, hot_model = pick_first_valid(
        models,
        [
            "CombProdH2-RP",
            "CombProdH2-CP",
        ],
    )
    if hot_model is not None:
        h_h_in = hot_model.get_specific_enthalpy(T_h_in, P_hot)
        h_h_out = hot_model.get_specific_enthalpy(T_h_out, P_hot)
        cp_mean_hot = (h_h_out - h_h_in) / (T_h_out - T_h_in)
        cp_h_in = hot_model.get_cp(T_h_in, P_hot)
        cp_h_out = hot_model.get_cp(T_h_out, P_hot)
        h_in_pct = (cp_h_in / cp_mean_hot - 1.0) * 100.0
        h_out_pct = (cp_h_out / cp_mean_hot - 1.0) * 100.0
    else:
        cp_mean_hot = float("nan")
        h_in_pct = float("nan")
        h_out_pct = float("nan")
        hot_key = "n/a"

    # Two-line Brewer mean cp summary (match AHJE format)
    print(f"Brewer mean cp (in/out % diff) (cold={cold_key}, hot={hot_key})")
    print(f"  cold c_p = {cp_mean_cold:.1f} J/kg-K ({c_in_pct:+.0f} %, {c_out_pct:+.0f} %)")
    print(f"  hot  c_p = {cp_mean_hot:.1f} J/kg-K ({h_in_pct:+.0f} %, {h_out_pct:+.0f} %)")

    # Helper to back-calculate perfect-gas parameters for a model using two states
    def backcalc_pg_params(
        label: str,
        model,
        T_in: float,
        P_in: float,
        T_out: float,
        P_out: float,
        cp_mean: float,
        T_ref: float = 273.15,
    ) -> None:
        R_UNIVERSAL_LOCAL = 8314.462618  # J/(kmol·K)
        # Compute R_specific via ideal relation and molecular masses
        rho_in = model.get_density(T_in, P_in)
        rho_out = model.get_density(T_out, P_out)
        R_in = P_in / (rho_in * T_in)
        R_out = P_out / (rho_out * T_out)
        M_in = R_UNIVERSAL_LOCAL / R_in
        M_out = R_UNIVERSAL_LOCAL / R_out
        # mu_ref at T_ref (at inlet pressure context)
        mu_ref = model.get_viscosity(T_ref, P_in)

        # Back-calc Sutherland constant from Sutherland's law at each state
        def _S_from(T: float, mu_T: float) -> float:
            A = (mu_T / mu_ref) / ((T / T_ref) ** 1.5)
            return (T_ref - A * T) / (A - 1.0)

        mu_in = model.get_viscosity(T_in, P_in)
        mu_out = model.get_viscosity(T_out, P_out)
        S_in = _S_from(T_in, mu_in)
        S_out = _S_from(T_out, mu_out)

        print(f"PG backcalc [{label}]")
        print(f"  M_in={M_in:.3f} kg/kmol, M_out={M_out:.3f} kg/kmol")
        print(f"  cp_mean={cp_mean:.1f} J/kg-K  |  mu_ref(T_ref={T_ref:.2f} K)={mu_ref:.3e} Pa·s")
        print(f"  S_in={S_in:.2f} K, S_out={S_out:.2f} K")

    # Brewer pair: para-H2 (cold) and H2 combustion products (hot)
    if cold_model is not None:
        backcalc_pg_params(
            f"Brewer cold ({cold_key}) para-H2/H2",
            cold_model,
            T_c_in,
            P_cold,
            T_c_out,
            P_cold,
            cp_mean_cold,
        )
    if hot_model is not None:
        backcalc_pg_params(
            f"Brewer hot ({hot_key}) H2 comb. products",
            hot_model,
            T_h_in,
            P_hot,
            T_h_out,
            P_hot,
            cp_mean_hot,
            T_ref=350.0,
        )

    # ======================
    # AHJE conditions: MTO
    # ======================
    FAR_MTO = 1.316 / (161.12 + 1.316)  # FAR = 1.316/(161.12+1.316)
    print(f"\nAHJE MTO FAR = 1.316/(161.12+1.316) = {FAR_MTO:.6f}")
    # Hot stream (combustion products of H2)
    T_h_in_MTO = 718.0
    P_h_in_MTO = 1.26 * BAR_TO_PA
    T_h_out_MTO = 718.0 - (718.0 - 60.0 - 290.0) / 2.0  # T = (718 - (718-60-290)/2)
    P_h_out_MTO = 1.15 * BAR_TO_PA
    # Cold stream (hydrogen side)
    T_c_in_MTO = 290.0
    P_c_in_MTO = 69.0 * BAR_TO_PA
    T_c_out_MTO = 718.0 - 60.0  # 658 K
    P_c_out_MTO = 68.0 * BAR_TO_PA

    models_MTO = build_h2_flue_models(FAR_MTO)

    # Cold inlet (MTO)
    rows = []
    for name, strat in strategies.items():
        props = query_properties(strat, T_c_in_MTO, P_c_in_MTO)
        row = {"Model": name}
        if "error" in props:
            row.update(
                {
                    "rho [kg/m^3]": None,
                    "cp [J/kg-K]": None,
                    "mu [Pa·s]": None,
                    "Pr [-]": None,
                    "gamma [-]": None,
                    "error": props["error"],
                }
            )
        else:
            row.update(props)
        rows.append(row)
    print_table(
        f"AHJE MTO cold inlet prop@ T={T_c_in_MTO:.1f} K, P={P_c_in_MTO / BAR_TO_PA:.2f} bar",
        rows,
    )

    # Cold outlet (MTO)
    rows = []
    for name, strat in strategies.items():
        props = query_properties(strat, T_c_out_MTO, P_c_out_MTO)
        row = {"Model": name}
        if "error" in props:
            row.update(
                {
                    "rho [kg/m^3]": None,
                    "cp [J/kg-K]": None,
                    "mu [Pa·s]": None,
                    "Pr [-]": None,
                    "gamma [-]": None,
                    "error": props["error"],
                }
            )
        else:
            row.update(props)
        rows.append(row)
    print_table(
        f"AHJE MTO cold outlet prop@ T={T_c_out_MTO:.1f} K, P={P_c_out_MTO / BAR_TO_PA:.2f} bar",
        rows,
    )

    # Hot inlet (MTO)
    rows = []
    for name, strat in models_MTO.items():
        props = query_properties(strat, T_h_in_MTO, P_h_in_MTO)
        row = {"Model": name}
        if "error" in props:
            row.update(
                {
                    "rho [kg/m^3]": None,
                    "cp [J/kg-K]": None,
                    "mu [Pa·s]": None,
                    "Pr [-]": None,
                    "gamma [-]": None,
                    "error": props["error"],
                }
            )
        else:
            row.update(props)
        rows.append(row)
    print_table(
        f"AHJE MTO hot inlet prop@ T={T_h_in_MTO:.1f} K, P={P_h_in_MTO / BAR_TO_PA:.2f} bar",
        rows,
    )

    # Hot outlet (MTO)
    rows = []
    for name, strat in models_MTO.items():
        props = query_properties(strat, T_h_out_MTO, P_h_out_MTO)
        row = {"Model": name}
        if "error" in props:
            row.update(
                {
                    "rho [kg/m^3]": None,
                    "cp [J/kg-K]": None,
                    "mu [Pa·s]": None,
                    "Pr [-]": None,
                    "gamma [-]": None,
                    "error": props["error"],
                }
            )
        else:
            row.update(props)
        rows.append(row)
    print_table(
        f"AHJE MTO hot outlet prop@ T={T_h_out_MTO:.1f} K, P={P_h_out_MTO / BAR_TO_PA:.2f} bar",
        rows,
    )

    # Mean cp summary for AHJE MTO
    cold_key_MTO, cold_model_MTO = pick_first_valid(
        strategies,
        [
            "REFPROP-ParaH2",
            "REFPROP-H2",
            "CoolProp-ParaH2",
            "CoolProp-H2",
            "PerfectGas-H2",
        ],
    )
    hot_key_MTO, hot_model_MTO = pick_first_valid(
        models_MTO,
        [
            "CombProdH2-RP",
            "CombProdH2-CP",
        ],
    )
    # Cold mean cp (MTO)
    if cold_model_MTO is not None:
        h_c_in = cold_model_MTO.get_specific_enthalpy(T_c_in_MTO, P_c_in_MTO)
        h_c_out = cold_model_MTO.get_specific_enthalpy(T_c_out_MTO, P_c_out_MTO)
        cp_mean_cold_MTO = (h_c_out - h_c_in) / (T_c_out_MTO - T_c_in_MTO)
        cp_c_in = cold_model_MTO.get_cp(T_c_in_MTO, P_c_in_MTO)
        cp_c_out = cold_model_MTO.get_cp(T_c_out_MTO, P_c_out_MTO)
        c_in_pct = (cp_c_in / cp_mean_cold_MTO - 1.0) * 100.0
        c_out_pct = (cp_c_out / cp_mean_cold_MTO - 1.0) * 100.0
    else:
        cp_mean_cold_MTO = float("nan")
        c_in_pct = float("nan")
        c_out_pct = float("nan")
        cold_key_MTO = "n/a"
    # Hot mean cp (MTO)
    if hot_model_MTO is not None:
        h_h_in = hot_model_MTO.get_specific_enthalpy(T_h_in_MTO, P_h_in_MTO)
        h_h_out = hot_model_MTO.get_specific_enthalpy(T_h_out_MTO, P_h_out_MTO)
        cp_mean_hot_MTO = (h_h_out - h_h_in) / (T_h_out_MTO - T_h_in_MTO)
        cp_h_in = hot_model_MTO.get_cp(T_h_in_MTO, P_h_in_MTO)
        cp_h_out = hot_model_MTO.get_cp(T_h_out_MTO, P_h_out_MTO)
        h_in_pct = (cp_h_in / cp_mean_hot_MTO - 1.0) * 100.0
        h_out_pct = (cp_h_out / cp_mean_hot_MTO - 1.0) * 100.0
    else:
        cp_mean_hot_MTO = float("nan")
        h_in_pct = float("nan")
        h_out_pct = float("nan")
        hot_key_MTO = "n/a"
    print(f"AHJE MTO mean cp (in/out % diff) (cold={cold_key_MTO}, hot={hot_key_MTO})")
    print(f"  cold c_p = {cp_mean_cold_MTO:.1f} J/kg-K ({c_in_pct:+.0f} %, {c_out_pct:+.0f} %)")
    print(f"  hot  c_p = {cp_mean_hot_MTO:.1f} J/kg-K ({h_in_pct:+.0f} %, {h_out_pct:+.0f} %)")

    # MTO pair: para-H2 (cold) and H2 combustion products (hot)
    if cold_model_MTO is not None:
        backcalc_pg_params(
            f"AHJE MTO cold ({cold_key_MTO}) para-H2/H2",
            cold_model_MTO,
            T_c_in_MTO,
            P_c_in_MTO,
            T_c_out_MTO,
            P_c_out_MTO,
            cp_mean_cold_MTO,
        )
    if hot_model_MTO is not None:
        backcalc_pg_params(
            f"AHJE MTO hot ({hot_key_MTO}) H2 comb. products",
            hot_model_MTO,
            T_h_in_MTO,
            P_h_in_MTO,
            T_h_out_MTO,
            P_h_out_MTO,
            cp_mean_hot_MTO,
            T_ref=350.0,
        )

    # ======================
    # AHJE conditions: ToC
    # ======================
    FAR_TOC = 0.418 / (63.898 + 0.418)  # FAR = 0.418/(63.898+0.418)
    print(f"\nAHJE ToC FAR = 0.418/(63.898+0.418) = {FAR_TOC:.6f}")
    # Hot stream
    T_h_in_TOC = 570.0
    P_h_in_TOC = 0.368 * BAR_TO_PA
    T_h_out_TOC = 570.0 - (570.0 - 3.0 - 290.0) / 2.0  # T = 570 - (570-3-290)/2
    P_h_out_TOC = 0.268 * BAR_TO_PA
    # Cold stream
    T_c_in_TOC = 290.0
    P_c_in_TOC = 25.0 * BAR_TO_PA
    T_c_out_TOC = 570.0 - 3.0  # 567 K
    P_c_out_TOC = 23.0 * BAR_TO_PA

    models_TOC = build_h2_flue_models(FAR_TOC)

    # Cold inlet (ToC)
    rows = []
    for name, strat in strategies.items():
        props = query_properties(strat, T_c_in_TOC, P_c_in_TOC)
        row = {"Model": name}
        if "error" in props:
            row.update(
                {
                    "rho [kg/m^3]": None,
                    "cp [J/kg-K]": None,
                    "mu [Pa·s]": None,
                    "Pr [-]": None,
                    "gamma [-]": None,
                    "error": props["error"],
                }
            )
        else:
            row.update(props)
        rows.append(row)
    print_table(f"AHJE ToC cold inlet prop@ T={T_c_in_TOC:.1f} K, P={P_c_in_TOC / BAR_TO_PA:.2f} bar", rows)

    # Cold outlet (ToC)
    rows = []
    for name, strat in strategies.items():
        props = query_properties(strat, T_c_out_TOC, P_c_out_TOC)
        row = {"Model": name}
        if "error" in props:
            row.update(
                {
                    "rho [kg/m^3]": None,
                    "cp [J/kg-K]": None,
                    "mu [Pa·s]": None,
                    "Pr [-]": None,
                    "gamma [-]": None,
                    "error": props["error"],
                }
            )
        else:
            row.update(props)
        rows.append(row)
    print_table(f"AHJE ToC cold outlet prop@ T={T_c_out_TOC:.1f} K, P={P_c_out_TOC / BAR_TO_PA:.2f} bar", rows)

    # Hot inlet (ToC)
    rows = []
    for name, strat in models_TOC.items():
        props = query_properties(strat, T_h_in_TOC, P_h_in_TOC)
        row = {"Model": name}
        if "error" in props:
            row.update(
                {
                    "rho [kg/m^3]": None,
                    "cp [J/kg-K]": None,
                    "mu [Pa·s]": None,
                    "Pr [-]": None,
                    "γ [-]": None,
                    "error": props["error"],
                }
            )
        else:
            row.update(props)
        rows.append(row)
    print_table(f"AHJE ToC hot inlet prop@ T={T_h_in_TOC:.1f} K, P={P_h_in_TOC / BAR_TO_PA:.3f} bar", rows)

    # Hot outlet (ToC)
    rows = []
    for name, strat in models_TOC.items():
        props = query_properties(strat, T_h_out_TOC, P_h_out_TOC)
        row = {"Model": name}
        if "error" in props:
            row.update(
                {
                    "rho [kg/m^3]": None,
                    "cp [J/kg-K]": None,
                    "mu [Pa·s]": None,
                    "Pr [-]": None,
                    "γ [-]": None,
                    "error": props["error"],
                }
            )
        else:
            row.update(props)
        rows.append(row)
    print_table(f"AHJE ToC hot outlet prop@ T={T_h_out_TOC:.1f} K, P={P_h_out_TOC / BAR_TO_PA:.3f} bar", rows)

    # Mean cp summary for AHJE ToC
    cold_key_TOC, cold_model_TOC = pick_first_valid(
        strategies,
        [
            "REFPROP-ParaH2",
            "REFPROP-H2",
            "CoolProp-ParaH2",
            "CoolProp-H2",
            "PerfectGas-H2",
        ],
    )
    hot_key_TOC, hot_model_TOC = pick_first_valid(
        models_TOC,
        [
            "CombProdH2-RP",
            "CombProdH2-CP",
        ],
    )
    # Cold mean cp (ToC)
    if cold_model_TOC is not None:
        h_c_in = cold_model_TOC.get_specific_enthalpy(T_c_in_TOC, P_c_in_TOC)
        h_c_out = cold_model_TOC.get_specific_enthalpy(T_c_out_TOC, P_c_out_TOC)
        cp_mean_cold_TOC = (h_c_out - h_c_in) / (T_c_out_TOC - T_c_in_TOC)
        cp_c_in = cold_model_TOC.get_cp(T_c_in_TOC, P_c_in_TOC)
        cp_c_out = cold_model_TOC.get_cp(T_c_out_TOC, P_c_out_TOC)
        c_in_pct = (cp_c_in / cp_mean_cold_TOC - 1.0) * 100.0
        c_out_pct = (cp_c_out / cp_mean_cold_TOC - 1.0) * 100.0
    else:
        cp_mean_cold_TOC = float("nan")
        c_in_pct = float("nan")
        c_out_pct = float("nan")
        cold_key_TOC = "n/a"
    # Hot mean cp (ToC)
    if hot_model_TOC is not None:
        h_h_in = hot_model_TOC.get_specific_enthalpy(T_h_in_TOC, P_h_in_TOC)
        h_h_out = hot_model_TOC.get_specific_enthalpy(T_h_out_TOC, P_h_out_TOC)
        cp_mean_hot_TOC = (h_h_out - h_h_in) / (T_h_out_TOC - T_h_in_TOC)
        cp_h_in = hot_model_TOC.get_cp(T_h_in_TOC, P_h_in_TOC)
        cp_h_out = hot_model_TOC.get_cp(T_h_out_TOC, P_h_out_TOC)
        h_in_pct = (cp_h_in / cp_mean_hot_TOC - 1.0) * 100.0
        h_out_pct = (cp_h_out / cp_mean_hot_TOC - 1.0) * 100.0
    else:
        cp_mean_hot_TOC = float("nan")
        h_in_pct = float("nan")
        h_out_pct = float("nan")
        hot_key_TOC = "n/a"
    print(f"AHJE ToC mean cp (in/out % diff) (cold={cold_key_TOC}, hot={hot_key_TOC})")
    print(f"  cold c_p = {cp_mean_cold_TOC:.1f} J/kg-K ({c_in_pct:+.0f} %, {c_out_pct:+.0f} %)")
    print(f"  hot  c_p = {cp_mean_hot_TOC:.1f} J/kg-K ({h_in_pct:+.0f} %, {h_out_pct:+.0f} %)")

    # ToC pair: para-H2 (cold) and H2 combustion products (hot)
    if cold_model_TOC is not None:
        backcalc_pg_params(
            f"AHJE ToC cold ({cold_key_TOC}) para-H2/H2",
            cold_model_TOC,
            T_c_in_TOC,
            P_c_in_TOC,
            T_c_out_TOC,
            P_c_out_TOC,
            cp_mean_cold_TOC,
        )
    if hot_model_TOC is not None:
        backcalc_pg_params(
            f"AHJE ToC hot ({hot_key_TOC}) H2 comb. products",
            hot_model_TOC,
            T_h_in_TOC,
            P_h_in_TOC,
            T_h_out_TOC,
            P_h_out_TOC,
            cp_mean_hot_TOC,
            T_ref=350.0,
        )


if __name__ == "__main__":
    # logging levels  DEBUG < INFO < WARNING < ERROR < CRITICAL (Default is WARNING)
    configure_logging(logging.INFO)
    configure_refprop()  # Annoying to have this here but if I want to have logging from
    # refprop configuration and path then I need to either configure logging before I import
    # the fluid_properties module or I need to configure refprop after configuring logging

    FAR = 0.0135
    P = 1.0e5
    model = CombustionProductsProperties(fuel_type="C092H2", FAR_mass=FAR, prefer_refprop=True)
    for T in np.linspace(630.0, 1000.0, 5):
        if model.mixture_state is not None:
            model.mixture_state.update(model.mixture.CP.PT_INPUTS, P, T)
            cps = model.mixture_state.cpmass()
        else:
            cps = model.get_cp(T, P)
        # For digit alignment even if negative, add a space for positives via {cps: +10.3e}
        logger.info(f"T={T:6.1f} K, cp={cps:+8.3e} J/kg-K")
        # TODO: dig deeper into why returning -inf cp

    main_table()
    # Kerosene products cp(T) plots
    # plot_cp_kerosene_products_three_FAR_as_function_of_temperature()
    # plt.show()

    # plot_mu_combustion_products_one_temp_as_function_of_FAR(T=750.0, P=0.4 * BAR_TO_PA, FAR_H2=None, FAR_kero=None, prefer_refprop=True)
    # plt.show()
