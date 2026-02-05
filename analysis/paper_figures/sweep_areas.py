"""
Sweep analysis for A_fr and A_q parameter space.

This script separates data generation from plotting:
- Data sweep over A_fr (frontal area) and A_q (heat transfer area)
- Saves results to parquet for reuse
- Plots Figures 6, 7, 8 for the paper

Figure 6: Contour plot of work potential vs A_q and g² (Heli or Brewer)
Figure 7: Δm_TO / ṁ vs m_hex/ṁ for Helicopter case
Figure 8: Δm_TO / ṁ vs m_hex/ṁ for Brewer case
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import griddata

from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.general_counterflow import rate_hex_simple

# =============================================================================
# Configuration
# =============================================================================
RERUN_SWEEP = False  # Set to False to load existing data from parquet
DATA_FILE = Path(__file__).parent / "sw_data.parquet"


# =============================================================================
# Case Configurations
# =============================================================================
def get_case_config(case: str) -> dict:
    """Return configuration dictionary for a given case."""
    if case == "Heli":
        fluid_hot = PerfectGasFluid.from_name("kerocomb_helicopter")
        fluid_cold = PerfectGasFluid.from_name("air")

        f_in = FluidInputs(
            hot=fluid_hot,
            cold=fluid_cold,
            m_dot_hot=1.6,
            m_dot_cold=1.6,
            Th_in=980,
            Ph_in=1.06e5,
            Tc_in=576,
            Pc_in=7.2e5,
        )

        return {
            "f_in": f_in,
            "ls_over_dh": 0.7,
            "t_over_dhc": 0.02,
            "sigma_r": 2.0,
            "sigma_w": 2.0,
            "d_h_c": 1e-3,
            "rho_wall_t": 8000 * 0.02 * 1e-3,
            "mass_engine": 250,
            "mission_hours": 2,
            "lhv_kwh_per_kg": 43.2 / 3.6,
            "eta_ov_over_eta_turb": 0.2 / 0.8,
            "eta_ov_recup_max_over_eta_turb": 0.4 / 0.8,
            "kg_hex_fixed": 6,
            "alpha_hex_kg": 0.7,
            "a_fr_start": 0.5,
            "dp_max": 0.2,
            "aq_baseline": 43.0,
            "aq_sweep_range": (1, 80, 50),
        }

    elif case == "Brewer":
        fluid_cold = PerfectGasFluid.from_name("para_h2")
        fluid_hot = PerfectGasFluid(M=27.5, S=150.0, T_ref=350.0, mu_ref=1.12e-5, gamma=1.37, Pr=0.74, cp=1170.0)

        f_in = FluidInputs(
            hot=fluid_hot,
            cold=fluid_cold,
            m_dot_hot=19.07,
            m_dot_cold=0.166,
            Th_in=778,
            Ph_in=0.388e5,
            Tc_in=264,
            Pc_in=16.8e5,
        )

        return {
            "f_in": f_in,
            "ls_over_dh": 60.0,
            "t_over_dhc": 0.06,
            "sigma_r": 250.0,
            "sigma_w": 1.14,
            "d_h_c": 5e-3,
            "rho_wall_t": 8000 * 0.06 * 5e-3,
            "mass_engine": 6000,
            "mission_hours": 10,
            "lhv_kwh_per_kg": 120 / 3.6,
            "eta_ov_over_eta_turb": 0.363 / 0.88,
            "eta_ov_recup_max_over_eta_turb": 0.4 / 0.88,
            "kg_hex_fixed": 0.2,
            "alpha_hex_kg": 0.9,
            "a_fr_start": 10.0,
            "dp_max": 0.1,
            "aq_baseline": 15.0,
            "aq_sweep_range": (0.5, 100, 200),
        }
    else:
        raise ValueError(f"Unknown case: {case}")


def compute_derived_params(cfg: dict) -> dict:
    """Compute derived parameters from configuration."""
    f_in = cfg["f_in"]

    # A_fr / A_o ratios
    a_fr_over_ao_c = (1 + cfg["sigma_r"]) + 2 * cfg["t_over_dhc"] * (1 + cfg["sigma_w"])
    a_fr_over_ao_h = cfg["sigma_r"] * a_fr_over_ao_c
    aoh_over_afr_ratio = cfg["sigma_r"] / a_fr_over_ao_c

    # Heat capacity rates
    c_hot = f_in.m_dot_hot * f_in.hot.state(T=f_in.Th_in, P=f_in.Ph_in).cp
    c_cold = f_in.m_dot_cold * f_in.cold.state(T=f_in.Tc_in, P=f_in.Pc_in).cp
    c_min = min(c_hot, c_cold)
    q_max = c_min * (f_in.Th_in - f_in.Tc_in)

    # Fuel conversion factor
    fuel_per_heat = cfg["mission_hours"] / cfg["lhv_kwh_per_kg"]

    # Initial g² calculation
    rho_in_hot = f_in.hot.state(T=f_in.Th_in, P=f_in.Ph_in).rho
    ao_h_start = cfg["a_fr_start"] * aoh_over_afr_ratio
    g_in2_start = (f_in.m_dot_hot / ao_h_start) ** 2 / f_in.Ph_in / rho_in_hot

    return {
        "a_fr_over_ao_c": a_fr_over_ao_c,
        "a_fr_over_ao_h": a_fr_over_ao_h,
        "aoh_over_afr_ratio": aoh_over_afr_ratio,
        "c_min": c_min,
        "q_max": q_max,
        "fuel_per_heat": fuel_per_heat,
        "rho_in_hot": rho_in_hot,
        "g_in2_start": g_in2_start,
    }


# =============================================================================
# Data Generation
# =============================================================================
def run_sweep(case: str) -> pd.DataFrame:
    """Run the A_fr/A_q sweep for a given case and return results as DataFrame."""
    cfg = get_case_config(case)
    f_in = cfg["f_in"]

    # Sweep parameters
    aq_start, aq_end, aq_n = cfg["aq_sweep_range"]
    aq_sweep = np.linspace(aq_start, aq_end, aq_n)
    afr_start = cfg["a_fr_start"]
    a_fr_decrease_ratio_start = 0.999

    results = []

    for aq in aq_sweep:
        a_fr = afr_start
        a_fr_decrease_ratio = a_fr_decrease_ratio_start
        it = 0

        while it < 10:
            r_s = rate_hex_simple(
                A_fr=a_fr,
                A_q=aq,
                f_in=f_in,
                d_h_c=cfg["d_h_c"],
                sigma_r=cfg["sigma_r"],
                sigma_w=cfg["sigma_w"],
                t_over_dhc=cfg["t_over_dhc"],
                ls_over_dh=cfg["ls_over_dh"],
            )

            # Reject designs exceeding pressure drop limits
            if r_s["dp_hot"] > cfg["dp_max"] or r_s["dp_cold"] > cfg["dp_max"] or r_s["dp_hot"] < 0:
                break

            results.append(
                {
                    "case": case,
                    "A_q": aq,
                    "A_fr": a_fr,
                    "euergy": -r_s["dW_pot_Eu_norm"],
                    "exergy": -r_s["dW_pot_Ex_norm"],
                    "dp_hot": r_s["dp_hot"],
                    "dp_cold": r_s["dp_cold"],
                    "effectiveness": r_s.get("effectiveness", np.nan),
                }
            )

            if it < 9:
                a_fr = a_fr * a_fr_decrease_ratio
                it += 1
            else:
                it = 0
                a_fr_decrease_ratio = a_fr_decrease_ratio * a_fr_decrease_ratio_start
                a_fr = a_fr * a_fr_decrease_ratio

    return pd.DataFrame(results)


def generate_all_data() -> pd.DataFrame:
    """Generate sweep data for all cases."""
    df_heli = run_sweep("Heli")
    df_brewer = run_sweep("Brewer")
    return pd.concat([df_heli, df_brewer], ignore_index=True)


def load_or_generate_data() -> pd.DataFrame:
    """Load data from parquet or generate if RERUN_SWEEP is True."""
    if RERUN_SWEEP or not DATA_FILE.exists():
        print("Running sweep for both cases...")
        df = generate_all_data()
        df.to_parquet(DATA_FILE)
        print(f"Data saved to {DATA_FILE}")
    else:
        print(f"Loading data from {DATA_FILE}")
        df = pd.read_parquet(DATA_FILE)
    return df


# =============================================================================
# Plotting Functions
# =============================================================================
def prepare_grid_data(df: pd.DataFrame, case: str):
    """Prepare interpolated grid data for a given case."""
    df_case = df[df["case"] == case].copy()

    aq_array = df_case["A_q"].values
    afr_array = df_case["A_fr"].values
    euergy_array = df_case["euergy"].values
    exergy_array = df_case["exergy"].values

    # Create interpolation grid
    xi = np.linspace(np.min(aq_array), np.max(aq_array), 100)
    yi = np.linspace(np.min(afr_array), np.max(afr_array), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi_euergy = griddata((aq_array, afr_array), euergy_array, (Xi, Yi), method="linear")
    Zi_exergy = griddata((aq_array, afr_array), exergy_array, (Xi, Yi), method="linear")

    return xi, yi, Xi, Yi, Zi_euergy, Zi_exergy


def plot_figure_6(df: pd.DataFrame, case: str = "Heli", use_euergy: bool = True):
    """
    Figure 6: Contour plot of work potential vs A_q and g².

    Args:
        df: DataFrame with sweep results
        case: "Heli" or "Brewer"
        use_euergy: If True, plot euergy; if False, plot exergy
    """
    cfg = get_case_config(case)
    derived = compute_derived_params(cfg)
    f_in = cfg["f_in"]

    xi, yi, Xi, Yi, Zi_euergy, Zi_exergy = prepare_grid_data(df, case)
    Zi = Zi_euergy if use_euergy else Zi_exergy

    # Convert y-axis to g² (dimensionless mass flux squared)
    yi_g2 = (f_in.m_dot_hot / yi) ** 2 / f_in.Ph_in / derived["rho_in_hot"]
    Yi_g2 = (f_in.m_dot_hot / Yi) ** 2 / f_in.Ph_in / derived["rho_in_hot"]

    # Find optimal g² for each A_q (euergy-based)
    idx_max = np.nanargmax(Zi_euergy, axis=0)
    y_at_max = yi[idx_max]
    y_at_max_g2 = (f_in.m_dot_hot / y_at_max) ** 2 / f_in.Ph_in / derived["rho_in_hot"]

    # Find optimal A_q for each g²
    idx_max_for_afr = np.nanargmax(Zi_euergy, axis=1)
    x_at_max_for_afr = xi[idx_max_for_afr]
    y_i_nd = (f_in.m_dot_hot / yi) ** 2 / f_in.Ph_in / derived["rho_in_hot"]

    # X-axis dimensionalisation
    dim_x = cfg["rho_wall_t"] / f_in.m_dot_hot
    x_label = "m_hex / mdot_air (kg/(kg/s))"

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(Xi * dim_x, Yi_g2, Zi, cmap="binary_r", levels=20)

    title = "Euergy" if use_euergy else "Exergy"
    plt.title(f"Contour plot of {title} creation / Q_max vs A_q and g² ({case})")

    plt.plot(xi * dim_x, y_at_max_g2, "r--", lw=2, label="Optimal g² for a given m_hex")
    plt.plot(x_at_max_for_afr * dim_x, y_i_nd, "b-", lw=2, label="Optimal m_hex for a given g²")

    cbar = plt.colorbar(cp, label="Work Potential creation / Q_max", format=PercentFormatter(xmax=1.0, decimals=0))
    plt.xlabel(x_label)
    plt.ylabel("g²_in (-)")

    if case == "Heli":
        plt.ylim(0, 0.1)
    elif case == "Brewer":
        plt.ylim(0, 0.25)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_delta_m_to(df: pd.DataFrame, case: str):
    """
    Plot Δm_TO / ṁ vs m_hex/ṁ for a given case.

    Figure 7: Helicopter case
    Figure 8: Brewer case

    Y-axis: Δm_TO / ṁ = m_hex/ṁ - m_fuel_saved/ṁ
    Negative values indicate net mass saved (beneficial).
    """
    cfg = get_case_config(case)
    derived = compute_derived_params(cfg)
    f_in = cfg["f_in"]

    # x,y are Aq and Afr - capitals are meshgrids, lowercase is 1D array

    area_q, area_fr, Xi, Yi, Zi_euergy, Zi_exergy = prepare_grid_data(df, case)

    # For each A_q, find the optimal A_fr (max euergy)
    max_euergy_for_aq = np.nanmax(Zi_euergy, axis=0)

    # X-axis: total HEx mass per unit mass flow (kg/(kg/s))

    m_hex = area_q * cfg["rho_wall_t"] * (1 + cfg["alpha_hex_kg"]) + cfg["kg_hex_fixed"]

    # Fuel saved per unit mass flow (kg/(kg/s))
    # m_fuel_saved/ṁ = ξ_Eu * Q_max/ṁ * (η_turb/η_ov) * t_mission/LHV
    # Note: ξ_Eu = euergy (already normalized by Q_max), so:
    # m_fuel_saved/ṁ = ξ_Eu * Q_max * (η_turb/η_ov) * FUEL_PER_HEAT / ṁ / 1000 (kW to W)
    m_fuel_saved_over_mdot = (
        max_euergy_for_aq
        * derived["q_max"]
        / f_in.m_dot_hot
        / cfg["eta_ov_over_eta_turb"]
        * derived["fuel_per_heat"]
        / 1000
    )

    # Δm_TO / ṁ = m_hex/ṁ - m_fuel_saved/ṁ
    delta_m_to_over_mdot = m_hex / f_in.m_dot_hot - m_fuel_saved_over_mdot

    # Also compute for higher efficiency cycle
    m_fuel_saved_recup_max = (
        max_euergy_for_aq
        * derived["q_max"]
        / cfg["eta_ov_recup_max_over_eta_turb"]
        * derived["fuel_per_heat"]
        / 1000
        / f_in.m_dot_hot
    )
    delta_m_to_recup_max = m_hex / f_in.m_dot_hot - m_fuel_saved_recup_max

    # Plotting
    plt.figure(figsize=(8, 6))

    if case == "Heli":
        label_base = "20% efficient cycle"
        label_recup = "40% efficient cycle"
    else:
        label_base = "36% efficient cycle"
        label_recup = "40% efficient cycle"

    plt.plot(m_hex / f_in.m_dot_hot, delta_m_to_over_mdot, "r--", lw=2, label=label_base)
    plt.plot(m_hex / f_in.m_dot_hot, delta_m_to_recup_max, "k-.", lw=2, label=label_recup)
    plt.axhline(y=0, color="g", linestyle="-", lw=1.5, label="Break even (Δm_TO = 0)")

    plt.xlabel("m_hex_overall / ṁ (kg/(kg/s))")
    plt.ylabel("Δm_TO / ṁ (kg/(kg/s))")
    plt.title(f"Take-off mass change per unit mass flow ({case})")

    # Find and annotate optimal points
    idx_opt_base = np.nanargmin(delta_m_to_over_mdot)
    idx_opt_recup = np.nanargmin(delta_m_to_recup_max)

    if not np.isnan(delta_m_to_over_mdot[idx_opt_base]):
        plt.scatter(
            m_hex[idx_opt_base] / f_in.m_dot_hot,
            delta_m_to_over_mdot[idx_opt_base],
            color="r",
            s=80,
            zorder=5,
            marker="o",
        )
    if not np.isnan(delta_m_to_recup_max[idx_opt_recup]):
        plt.scatter(
            m_hex[idx_opt_recup] / f_in.m_dot_hot,
            delta_m_to_recup_max[idx_opt_recup],
            color="k",
            s=80,
            zorder=5,
            marker="o",
        )

    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_figure_7(df: pd.DataFrame):
    """Figure 7: Δm_TO / ṁ for Helicopter case."""
    plot_delta_m_to(df, "Heli")


def plot_figure_8(df: pd.DataFrame):
    """Figure 8: Δm_TO / ṁ for Brewer case."""
    plot_delta_m_to(df, "Brewer")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Load or generate data
    df = load_or_generate_data()

    print("\nData summary:")
    print(f"  Heli points: {len(df[df['case'] == 'Heli'])}")
    print(f"  Brewer points: {len(df[df['case'] == 'Brewer'])}")

    # Plot all figures
    print("\nPlotting Figure 6 (Contour - Heli)...")
    plot_figure_6(df, case="Heli", use_euergy=True)

    print("\nPlotting Figure 7 (Δm_TO/ṁ - Heli)...")
    plot_figure_7(df)

    print("\nPlotting Figure 8 (Δm_TO/ṁ - Brewer)...")
    plot_figure_8(df)
