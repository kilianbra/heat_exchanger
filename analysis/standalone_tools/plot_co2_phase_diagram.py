"""Standalone CO2 phase diagram with density contours.

Features:
- x-axis: temperature in degC
- y-axis: pressure in atm
- vapor-liquid equilibrium curve (liquidus / vapor-pressure line)
- solid-liquid equilibrium (melting line, when backend supports it)
- triple and critical points
- supercritical region shading
- density contour lines and filled contours (kg/L)

Uses REFPROP backend when available, otherwise falls back to HEOS.

Run:
    uv run analysis/standalone_tools/plot_co2_phase_diagram.py --p-min-atm 1 --out-dir analysis/standalone_tools/figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np

PA_PER_ATM = 101325.0
KELVIN_OFFSET = 273.15


def _density_kg_per_l(fluid: str, t_k: float, p_pa: float) -> float:
    """Return density in kg/L, with phase-imposed fallbacks near boundaries."""
    # First try standard state resolution.
    try:
        return CP.PropsSI("Dmass", "T", t_k, "P", p_pa, fluid) / 1000.0
    except Exception:
        pass

    # Around saturation/melting boundaries, explicit phase hints often recover values.
    for phase in ("gas", "liquid", "supercritical", "supercritical_gas", "supercritical_liquid"):
        try:
            return CP.PropsSI("Dmass", f"T|{phase}", t_k, "P", p_pa, fluid) / 1000.0
        except Exception:
            continue

    return np.nan


def _get_state(fluid: str = "CO2") -> tuple[CP.AbstractState, str]:
    """Return an AbstractState, preferring REFPROP."""
    for backend in ("REFPROP", "HEOS"):
        try:
            state = CP.AbstractState(backend, fluid)
            return state, backend
        except Exception:
            continue
    raise RuntimeError("Could not initialize CO2 state with REFPROP or HEOS.")


def _build_density_grid(
    fluid: str,
    t_c_min: float,
    t_c_max: float,
    p_atm_min: float,
    p_atm_max: float,
    n_t: int,
    n_p: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build T/P mesh and density field in kg/L."""
    t_c = np.linspace(t_c_min, t_c_max, n_t)
    p_atm = np.logspace(np.log10(p_atm_min), np.log10(p_atm_max), n_p)
    tt_c, pp_atm = np.meshgrid(t_c, p_atm)

    rho_kg_l = np.full_like(tt_c, np.nan, dtype=float)
    for i in range(n_p):
        for j in range(n_t):
            t_k = tt_c[i, j] + KELVIN_OFFSET
            p_pa = pp_atm[i, j] * PA_PER_ATM
            rho_kg_l[i, j] = _density_kg_per_l(fluid=fluid, t_k=t_k, p_pa=p_pa)
    return tt_c, pp_atm, rho_kg_l


def _saturation_curve(fluid: str, t_triple_k: float, t_crit_k: float) -> tuple[np.ndarray, np.ndarray]:
    """Return vapor-pressure line (liquid-vapor equilibrium) in degC, atm."""
    t_sat_k = np.linspace(t_triple_k + 1e-5, t_crit_k - 1e-5, 300)
    p_sat_pa = np.array([CP.PropsSI("P", "T", t_k, "Q", 0, fluid) for t_k in t_sat_k])
    return t_sat_k - KELVIN_OFFSET, p_sat_pa / PA_PER_ATM


def _melting_curve(state: CP.AbstractState, t_triple_k: float, t_max_k: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Return solid-liquid equilibrium in degC, atm when available."""
    if not state.has_melting_line():
        return None

    t_melt_k = np.linspace(t_triple_k + 0.1, t_max_k, 220)
    p_melt_pa: list[float] = []
    t_valid_k: list[float] = []
    for t_k in t_melt_k:
        try:
            p_pa = state.melting_line(CP.iP, CP.iT, float(t_k))
            if np.isfinite(p_pa) and p_pa > 0:
                p_melt_pa.append(p_pa)
                t_valid_k.append(float(t_k))
        except Exception:
            continue

    if not t_valid_k:
        return None
    return np.array(t_valid_k) - KELVIN_OFFSET, np.array(p_melt_pa) / PA_PER_ATM


def make_plot(output_dir: Path, t_c_min: float, t_c_max: float, p_atm_min: float, p_atm_max: float) -> None:
    fluid = "CO2"
    state, backend = _get_state(fluid=fluid)

    t_triple_k = CP.PropsSI("Ttriple", fluid)
    p_triple_atm = CP.PropsSI("ptriple", fluid) / PA_PER_ATM
    t_crit_k = CP.PropsSI("Tcrit", fluid)
    p_crit_atm = CP.PropsSI("pcrit", fluid) / PA_PER_ATM

    tt_c, pp_atm, rho = _build_density_grid(
        fluid=fluid,
        t_c_min=t_c_min,
        t_c_max=t_c_max,
        p_atm_min=p_atm_min,
        p_atm_max=p_atm_max,
        n_t=420,
        n_p=420,
    )
    t_sat_c, p_sat_atm = _saturation_curve(fluid=fluid, t_triple_k=t_triple_k, t_crit_k=t_crit_k)
    melt = _melting_curve(state=state, t_triple_k=t_triple_k, t_max_k=t_c_max + KELVIN_OFFSET)

    output_dir.mkdir(parents=True, exist_ok=True)
    line_fig = output_dir / "co2_phase_diagram_density_lines.png"
    fill_fig = output_dir / "co2_phase_diagram_density_filled.png"

    # Figure 1: line contours
    plt.figure(figsize=(11, 8))
    # Dense low-density levels to resolve the bottom-left region on log-pressure plots.
    low_levels = np.array(
        [
            0.01,
            0.015,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
        ]
    )
    high_levels = np.linspace(0.1, np.nanpercentile(rho, 98), 16)
    line_levels = np.unique(np.concatenate((low_levels, high_levels)))
    cs = plt.contour(tt_c, pp_atm, rho, levels=line_levels, colors="0.35", linewidths=0.9)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    plt.plot(t_sat_c, p_sat_atm, color="tab:blue", linewidth=2.0, label="Liquid-Vapor line (liquidus)")
    if melt is not None:
        plt.plot(melt[0], melt[1], color="tab:red", linewidth=2.0, label="Solid-Liquid line (solidus)")

    plt.scatter(t_triple_k - KELVIN_OFFSET, p_triple_atm, color="black", s=40, zorder=5, label="Triple point")
    plt.scatter(t_crit_k - KELVIN_OFFSET, p_crit_atm, color="purple", s=40, zorder=5, label="Critical point")

    # Supercritical region (above both Tcrit and Pcrit) within selected window
    t_sc_min = max(t_crit_k - KELVIN_OFFSET, t_c_min)
    p_sc_min = max(p_crit_atm, p_atm_min)
    if t_sc_min < t_c_max and p_sc_min < p_atm_max:
        plt.fill_between(
            [t_sc_min, t_c_max],
            [p_sc_min, p_sc_min],
            [p_atm_max, p_atm_max],
            color="mediumpurple",
            alpha=0.12,
            label="Supercritical region",
        )

    plt.xlabel("Temperature [degC]")
    plt.ylabel("Pressure [atm]")
    plt.title(f"CO2 Phase Diagram with Density Contours [kg/L] (backend={backend})")
    plt.xlim(t_c_min, t_c_max)
    plt.ylim(p_atm_min, p_atm_max)
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(line_fig, dpi=180)
    plt.close()

    # Figure 2: filled contour map
    plt.figure(figsize=(11, 8))
    fill_levels = np.unique(np.concatenate((np.linspace(0.01, 0.09, 26), np.linspace(0.1, np.nanpercentile(rho, 98), 36))))
    cf = plt.contourf(tt_c, pp_atm, rho, levels=fill_levels, cmap="viridis")
    cbar = plt.colorbar(cf)
    cbar.set_label("Density [kg/L]")

    plt.contour(tt_c, pp_atm, rho, levels=line_levels, colors="white", linewidths=0.4, alpha=0.8)
    plt.plot(t_sat_c, p_sat_atm, color="tab:orange", linewidth=2.0, label="Liquid-Vapor line")
    if melt is not None:
        plt.plot(melt[0], melt[1], color="tab:red", linewidth=2.0, label="Solid-Liquid line")
    plt.scatter(t_triple_k - KELVIN_OFFSET, p_triple_atm, color="black", s=35, zorder=5, label="Triple point")
    plt.scatter(t_crit_k - KELVIN_OFFSET, p_crit_atm, color="magenta", s=35, zorder=5, label="Critical point")

    plt.xlabel("Temperature [degC]")
    plt.ylabel("Pressure [atm]")
    plt.title(f"CO2 Density Contour Map [kg/L] with Phase Boundaries (backend={backend})")
    plt.xlim(t_c_min, t_c_max)
    plt.ylim(p_atm_min, p_atm_max)
    plt.yscale("log")
    plt.grid(alpha=0.2)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fill_fig, dpi=180)
    plt.close()

    print(f"Backend used: {backend}")
    print(f"Triple point: T={t_triple_k - KELVIN_OFFSET:.3f} degC, P={p_triple_atm:.4f} atm")
    print(f"Critical point: T={t_crit_k - KELVIN_OFFSET:.3f} degC, P={p_crit_atm:.4f} atm")
    print(f"Saved: {line_fig}")
    print(f"Saved: {fill_fig}")
    if melt is None:
        print("Solid-liquid line unavailable for selected backend.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CO2 phase diagram with density contours.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/standalone_tools/figs"),
        help="Output directory for figures.",
    )
    parser.add_argument("--t-min-c", type=float, default=-70.0, help="Minimum temperature [degC].")
    parser.add_argument("--t-max-c", type=float, default=80.0, help="Maximum temperature [degC].")
    parser.add_argument("--p-min-atm", type=float, default=1.0, help="Minimum pressure [atm].")
    parser.add_argument("--p-max-atm", type=float, default=600.0, help="Maximum pressure [atm].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_plot(
        output_dir=args.out_dir,
        t_c_min=args.t_min_c,
        t_c_max=args.t_max_c,
        p_atm_min=args.p_min_atm,
        p_atm_max=args.p_max_atm,
    )


if __name__ == "__main__":
    main()
