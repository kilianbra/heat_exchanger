"""Standalone parahydrogen T-s diagram at low temperatures.

Plots:
- Saturation dome (sat. liquid and sat. vapor) for parahydrogen from 15–50 K
- Three points overlaid:
  - saturated liquid at P1 = 3.447 bar (pump inlet)
  - pump outlet at P2_TO  = 51.552 bar with eta = 69.0%
  - pump outlet at P2_ToC = 17.595 bar with eta = 67.2%

Also prints the pump outlet temperatures (TO and ToC), and compares the implied
specific pump work (Δh_actual) to the provided specific powers.

Run:
    uv run analysis/fluid_files/plot_parahydrogen_ts_diagram.py --out-dir analysis/fluid_files/figs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np

from heat_exchanger.fluid_properties import configure_refprop

PA_PER_BAR = 1e5


@dataclass(frozen=True)
class PumpCase:
    name: str
    p_out_bar: float
    eta: float
    power_kw: float
    mdot_kg_s: float

    @property
    def p_out_pa(self) -> float:
        return self.p_out_bar * PA_PER_BAR

    @property
    def w_spec_given_kj_kg(self) -> float:
        # kW / (kg/s) = kJ/kg
        return self.power_kw / self.mdot_kg_s


def _get_state_refprop_first(fluid: str) -> tuple[CP.AbstractState, str]:
    """Return an AbstractState, preferring REFPROP."""
    for backend in ("REFPROP", "HEOS"):
        try:
            state = CP.AbstractState(backend, fluid)
            return state, backend
        except Exception:
            continue
    raise RuntimeError(f"Could not initialize state for '{fluid}' with REFPROP or HEOS.")


def _resolve_hydrogen_fluid_strings() -> dict[str, str]:
    """Best-effort mapping to REFPROP-compatible fluid strings."""
    # REFPROP naming can vary between installations; try common variants.
    candidates_parah2 = [
        "PARAHYDROGEN",
        "PARAHYD",
        "PARA-HYDROGEN",
        "ParaHydrogen",
        "p-HYDROGEN",
        "p-H2",
    ]
    candidates_normalh2 = [
        "HYDROGEN",
        "H2",
        "NormalHydrogen",
        "NORMALHYDROGEN",
        "n-HYDROGEN",
        "n-H2",
    ]

    def first_working(cands: list[str]) -> str:
        for f in cands:
            try:
                _get_state_refprop_first(f)
                return f
            except Exception:
                continue
        # Return the first candidate so the error message later is still meaningful.
        return cands[0]

    return {
        "parah2": first_working(candidates_parah2),
        "normalh2": first_working(candidates_normalh2),
    }


def _sat_smass_over_t_range(fluid: str, t_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (s_sat_liq, s_sat_vap) arrays in J/(kg*K)."""
    s_l = np.full_like(t_k, np.nan, dtype=float)
    s_v = np.full_like(t_k, np.nan, dtype=float)
    for i, t in enumerate(t_k):
        try:
            s_l[i] = CP.PropsSI("Smass", "T", float(t), "Q", 0, fluid)
            s_v[i] = CP.PropsSI("Smass", "T", float(t), "Q", 1, fluid)
        except Exception:
            # Near triple point or if saturation is undefined, keep NaN.
            continue
    return s_l, s_v


def _smass_at_t_p_refined(fluid: str, t_k: float, p_pa: float) -> float:
    """Entropy at (T,P), with phase-hints to avoid boundary failures."""
    try:
        return float(CP.PropsSI("Smass", "T", float(t_k), "P", float(p_pa), fluid))
    except Exception:
        pass

    # Try a few phase strings that commonly recover values.
    for phase in ("liquid", "gas", "supercritical", "supercritical_liquid", "supercritical_gas"):
        try:
            return float(CP.PropsSI("Smass", f"T|{phase}", float(t_k), "P", float(p_pa), fluid))
        except Exception:
            continue

    return float("nan")


def _isobar_ts_curve(
    fluid: str,
    p_pa: float,
    t_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (s, T) arrays for an isobar in T-s space (s in J/kg/K)."""
    s = np.array([_smass_at_t_p_refined(fluid, float(t), float(p_pa)) for t in t_k], dtype=float)
    valid = np.isfinite(s)
    return s[valid], t_k[valid]


def _pump_outlet_from_sat_liq_inlet(
    fluid: str,
    p_in_pa: float,
    p_out_pa: float,
    eta: float,
) -> dict[str, float]:
    """Compute pump outlet from saturated liquid inlet, given isentropic efficiency."""
    # Inlet: saturated liquid at p_in
    t1 = CP.PropsSI("T", "P", p_in_pa, "Q", 0, fluid)
    h1 = CP.PropsSI("Hmass", "P", p_in_pa, "Q", 0, fluid)
    s1 = CP.PropsSI("Smass", "P", p_in_pa, "Q", 0, fluid)
    rho1 = CP.PropsSI("Dmass", "P", p_in_pa, "Q", 0, fluid)

    # Isentropic outlet at (P2, s1)
    h2s = CP.PropsSI("Hmass", "P", p_out_pa, "Smass", s1, fluid)

    # Pump efficiency definition: eta = (h2s - h1) / (h2 - h1)  -> h2 = h1 + (h2s-h1)/eta
    h2 = h1 + (h2s - h1) / eta
    t2 = CP.PropsSI("T", "P", p_out_pa, "Hmass", h2, fluid)
    s2 = CP.PropsSI("Smass", "P", p_out_pa, "Hmass", h2, fluid)
    rho2 = CP.PropsSI("Dmass", "P", p_out_pa, "Hmass", h2, fluid)

    return {
        "T1_K": float(t1),
        "P1_Pa": float(p_in_pa),
        "h1_J_kg": float(h1),
        "s1_J_kgK": float(s1),
        "rho1_kg_m3": float(rho1),
        "T2_K": float(t2),
        "P2_Pa": float(p_out_pa),
        "h2_J_kg": float(h2),
        "s2_J_kgK": float(s2),
        "rho2_kg_m3": float(rho2),
        "h2s_J_kg": float(h2s),
    }


def _isentropic_compression_from_sat_vapour(
    fluid: str,
    p_in_pa: float,
    p_out_pa: float,
) -> dict[str, float]:
    """Isentropic compression starting from saturated vapour at p_in."""
    t1 = CP.PropsSI("T", "P", p_in_pa, "Q", 1, fluid)
    h1 = CP.PropsSI("Hmass", "P", p_in_pa, "Q", 1, fluid)
    s1 = CP.PropsSI("Smass", "P", p_in_pa, "Q", 1, fluid)
    rho1 = CP.PropsSI("Dmass", "P", p_in_pa, "Q", 1, fluid)

    h2s = CP.PropsSI("Hmass", "P", p_out_pa, "Smass", s1, fluid)
    t2s = CP.PropsSI("T", "P", p_out_pa, "Hmass", h2s, fluid)
    rho2s = CP.PropsSI("Dmass", "P", p_out_pa, "Hmass", h2s, fluid)

    return {
        "T1_K": float(t1),
        "P1_Pa": float(p_in_pa),
        "h1_J_kg": float(h1),
        "s1_J_kgK": float(s1),
        "rho1_kg_m3": float(rho1),
        "T2s_K": float(t2s),
        "P2_Pa": float(p_out_pa),
        "h2s_J_kg": float(h2s),
        "rho2s_kg_m3": float(rho2s),
    }


def _heating_sat_liquid_to_300k_breakdown(fluid: str, p_pa: float) -> dict[str, float]:
    """Enthalpy rise to heat sat liquid at p to 300 K at same pressure.

    Returns enthalpies in J/kg and a latent fraction based on h_fg/(h_300 - h_f).
    """
    t_sat = CP.PropsSI("T", "P", p_pa, "Q", 0, fluid)
    h_f = CP.PropsSI("Hmass", "P", p_pa, "Q", 0, fluid)
    h_g = CP.PropsSI("Hmass", "P", p_pa, "Q", 1, fluid)
    h_300 = CP.PropsSI("Hmass", "T", 300.0, "P", p_pa, fluid)

    dh_total = h_300 - h_f
    dh_latent = h_g - h_f
    frac_latent = dh_latent / dh_total if dh_total > 0 else float("nan")

    return {
        "P_Pa": float(p_pa),
        "T_sat_K": float(t_sat),
        "h_f_J_kg": float(h_f),
        "h_g_J_kg": float(h_g),
        "h_300_J_kg": float(h_300),
        "dh_total_J_kg": float(dh_total),
        "dh_latent_J_kg": float(dh_latent),
        "frac_latent": float(frac_latent),
    }


def make_plot(
    out_dir: Path,
    t_min_k: float,
    t_max_k: float,
    n_t: int,
    also_try_normal_h2: bool,
) -> None:
    configure_refprop()

    fluids = _resolve_hydrogen_fluid_strings()
    fluid_parah2 = fluids["parah2"]
    fluid_normalh2 = fluids["normalh2"]

    # Prefer REFPROP, but we’ll still plot if it falls back (prints backend).
    _, backend = _get_state_refprop_first(fluid_parah2)

    t_k = np.linspace(t_min_k, t_max_k, n_t)
    s_l, s_v = _sat_smass_over_t_range(fluid_parah2, t_k=t_k)

    valid = np.isfinite(s_l) & np.isfinite(s_v)
    if not np.any(valid):
        raise RuntimeError(
            "Could not compute saturation properties over requested T range. "
            "This usually means REFPROP isn’t available/configured for parahydrogen."
        )

    # s range guidance: from sat liquid to sat vap + 30%*(s_v - s_l), across the T range.
    s_min = float(np.nanmin(s_l[valid]))
    s_max = float(np.nanmax(s_v[valid] + 0.3 * (s_v[valid] - s_l[valid])))

    # Pump cases (Brewer operating conditions per your message)
    p1_bar = 3.447
    p1_pa = p1_bar * PA_PER_BAR

    cases = [
        PumpCase(name="TO", p_out_bar=51.552, eta=0.690, power_kw=44.7, mdot_kg_s=0.411),
        PumpCase(name="ToC", p_out_bar=17.595, eta=0.672, power_kw=5.46, mdot_kg_s=0.166),
    ]

    inlet = _pump_outlet_from_sat_liq_inlet(fluid_parah2, p_in_pa=p1_pa, p_out_pa=p1_pa, eta=1.0)
    outlet_states = {c.name: _pump_outlet_from_sat_liq_inlet(fluid_parah2, p_in_pa=p1_pa, p_out_pa=c.p_out_pa, eta=c.eta) for c in cases}

    # Plot
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parahydrogen_ts_diagram.png"

    plt.figure(figsize=(10.5, 7.5))
    # Plot only the valid saturation range so the dome "closes" cleanly (up to critical).
    t_sat = t_k[valid]
    s_l_sat = s_l[valid]
    s_v_sat = s_v[valid]
    plt.plot(s_l_sat / 1000.0, t_sat, color="tab:blue", linewidth=2.0, label="_nolegend_")
    plt.plot(s_v_sat / 1000.0, t_sat, color="tab:orange", linewidth=2.0, label="_nolegend_")

    # Isobars through the three points (inlet + two outlets).
    p_crit_pa = float(CP.PropsSI("pcrit", fluid_parah2))
    isobar_specs = [
        ("3.447 bar", p1_pa, "0.45"),
        ("51.552 bar", cases[0].p_out_pa, "0.35"),
        ("17.595 bar", cases[1].p_out_pa, "0.35"),
        ("pcrit", p_crit_pa, "0.25"),
    ]
    for p_label, p_pa, color in isobar_specs:
        s_iso, t_iso = _isobar_ts_curve(fluid_parah2, p_pa=p_pa, t_k=t_k)
        if s_iso.size >= 2:
            plt.plot(s_iso / 1000.0, t_iso, color=color, linewidth=1.0, alpha=0.65, label="_nolegend_")

            # Inline label: p = xx.x bar
            p_bar = p_pa / PA_PER_BAR
            idx = int(0.75 * (t_iso.size - 1))
            plt.text(
                (s_iso[idx] / 1000.0) + 0.01,
                t_iso[idx],
                f"p = {p_bar:.1f} bar",
                fontsize=9,
                color=color,
                alpha=0.9,
            )

    # Mark inlet + outlets
    h_in = plt.scatter(
        inlet["s1_J_kgK"] / 1000.0,
        inlet["T1_K"],
        s=55,
        color="black",
        zorder=6,
        label=f"Inlet: sat liq @ {p1_bar:.3f} bar",
    )
    circle_handles = [h_in]
    for c in cases:
        st = outlet_states[c.name]
        h_out = plt.scatter(
            st["s2_J_kgK"] / 1000.0,
            st["T2_K"],
            s=65,
            zorder=7,
            label=f"{c.name} out: {c.p_out_bar:.3f} bar, η={c.eta*100:.1f}%",
        )
        circle_handles.append(h_out)
        # Light line from inlet to outlet
        plt.plot(
            [inlet["s1_J_kgK"] / 1000.0, st["s2_J_kgK"] / 1000.0],
            [inlet["T1_K"], st["T2_K"]],
            color="0.35",
            linewidth=1.2,
            alpha=0.7,
        )

    plt.xlabel("Specific entropy, s [kJ/(kg·K)]")
    plt.ylabel("Temperature, T [K]")
    plt.title(f"Parahydrogen T–s (15–50 K) with pump points (backend={backend})")
    plt.xlim(s_min / 1000.0, s_max / 1000.0)
    plt.ylim(t_min_k, t_max_k)
    plt.grid(alpha=0.25)
    plt.legend(handles=circle_handles, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Fluid (para): {fluid_parah2}  (backend={backend})")
    print(f"Saved: {out_path}")
    print()
    print(f"Inlet: saturated liquid at {p1_bar:.3f} bar -> T1 = {inlet['T1_K']:.4f} K")
    print(f"  rho_in = {inlet['rho1_kg_m3']:.3f} kg/m^3")
    print()

    for c in cases:
        st = outlet_states[c.name]
        dh_actual_kj_kg = (st["h2_J_kg"] - inlet["h1_J_kg"]) / 1000.0
        dh_isent_kj_kg = (st["h2s_J_kg"] - inlet["h1_J_kg"]) / 1000.0
        print(f"{c.name}: P2={c.p_out_bar:.3f} bar, eta={c.eta*100:.2f}%")
        print(f"  T_out = {st['T2_K']:.4f} K")
        print(f"  rho_out = {st['rho2_kg_m3']:.3f} kg/m^3")
        print(f"  dH_isent = {dh_isent_kj_kg:.3f} kJ/kg")
        print(f"  dH_actual= {dh_actual_kj_kg:.3f} kJ/kg")
        print(f"  Given specific power = {c.w_spec_given_kj_kg:.3f} kJ/kg")
        print(f"  (dH_actual - given) = {dh_actual_kj_kg - c.w_spec_given_kj_kg:+.3f} kJ/kg")
        print()

    # Vapour-compressor comparison: start at sat vapour @ p1 and compress isentropically.
    try:
        vap_in = _isentropic_compression_from_sat_vapour(fluid_parah2, p_in_pa=p1_pa, p_out_pa=p1_pa)
        print("Isentropic vapour compression comparison (para-H2):")
        print(f"  Vapour inlet: sat vap @ {p1_bar:.3f} bar -> T1 = {vap_in['T1_K']:.4f} K, rho1 = {vap_in['rho1_kg_m3']:.5f} kg/m^3")
        print("  Compressor assumed 100% efficient (w = h2s-h1). Pump uses given eta from sat liquid inlet.")
        print()

        for c in cases:
            vap = _isentropic_compression_from_sat_vapour(fluid_parah2, p_in_pa=p1_pa, p_out_pa=c.p_out_pa)
            w_comp_kj_kg = (vap["h2s_J_kg"] - vap["h1_J_kg"]) / 1000.0  # 100% eff compressor

            pump_state = outlet_states[c.name]
            w_pump_isent_kj_kg = (pump_state["h2s_J_kg"] - inlet["h1_J_kg"]) / 1000.0
            w_pump_actual_kj_kg = (pump_state["h2_J_kg"] - inlet["h1_J_kg"]) / 1000.0

            print(f"  {c.name}: to P2={c.p_out_bar:.3f} bar")
            print(f"    Vapour compressor (isentropic): w = {w_comp_kj_kg:.3f} kJ/kg (T2s={vap['T2s_K']:.3f} K)")
            print(f"    Pump from sat liquid: w_isent={w_pump_isent_kj_kg:.3f} kJ/kg, w_actual(eta={c.eta*100:.1f}%)={w_pump_actual_kj_kg:.3f} kJ/kg")
            print(f"    Difference (compressor - pump_actual) = {w_comp_kj_kg - w_pump_actual_kj_kg:+.3f} kJ/kg")
            if w_pump_actual_kj_kg > 0:
                print(f"    Ratio (compressor / pump_actual) = {w_comp_kj_kg / w_pump_actual_kj_kg:.2f}")
            print()
    except Exception as exc:
        print()
        print(f"Isentropic vapour compression comparison skipped due to property evaluation error: {exc}")

    # Heating breakdown: sat liquid at p -> 300 K at same p, and latent fraction.
    print()
    print("Heating saturated parahydrogen (sat liq) to 300 K at constant pressure:")
    for p_bar in (1.0, 3.0, 5.0, 10.0):
        p_pa = p_bar * PA_PER_BAR
        try:
            b = _heating_sat_liquid_to_300k_breakdown(fluid_parah2, p_pa=p_pa)
            dh_total_kj_kg = b["dh_total_J_kg"] / 1000.0
            dh_latent_kj_kg = b["dh_latent_J_kg"] / 1000.0
            frac_pct = 100.0 * b["frac_latent"]
            print(f"  p = {p_bar:.1f} bar (Tsat={b['T_sat_K']:.3f} K): dH_total = {dh_total_kj_kg:.1f} kJ/kg")
            print(f"    phase change (h_fg) = {dh_latent_kj_kg:.1f} kJ/kg  ->  {frac_pct:.1f}% of total")
        except Exception as exc:
            print(f"  p = {p_bar:.1f} bar: skipped (property evaluation error: {exc})")

    if also_try_normal_h2:
        try:
            _, backend_n = _get_state_refprop_first(fluid_normalh2)
            inlet_n = _pump_outlet_from_sat_liq_inlet(fluid_normalh2, p_in_pa=p1_pa, p_out_pa=p1_pa, eta=1.0)
            print(f"Fluid (normal): {fluid_normalh2}  (backend={backend_n})")
            print(f"Inlet (normal H2): sat liq @ {p1_bar:.3f} bar -> T1 = {inlet_n['T1_K']:.4f} K")
            for c in cases:
                st_n = _pump_outlet_from_sat_liq_inlet(fluid_normalh2, p_in_pa=p1_pa, p_out_pa=c.p_out_pa, eta=c.eta)
                print(f"{c.name} (normal H2): T_out = {st_n['T2_K']:.4f} K")
        except Exception as exc:
            print()
            print(f"Normal-hydrogen comparison skipped (could not evaluate '{fluid_normalh2}'): {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a low-temperature T-s diagram for parahydrogen using REFPROP when available.")
    p.add_argument("--out-dir", type=Path, default=Path("analysis/fluid_files/figs"), help="Output directory for figure.")
    p.add_argument("--t-min-k", type=float, default=15.0, help="Minimum temperature [K].")
    p.add_argument("--t-max-k", type=float, default=50.0, help="Maximum temperature [K].")
    p.add_argument("--n-t", type=int, default=350, help="Number of temperature samples for saturation curve.")
    p.add_argument("--also-try-normal-h2", action="store_true", help="Also print outlet temperatures using normal hydrogen (if available).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    make_plot(
        out_dir=args.out_dir,
        t_min_k=args.t_min_k,
        t_max_k=args.t_max_k,
        n_t=args.n_t,
        also_try_normal_h2=bool(args.also_try_normal_h2),
    )


if __name__ == "__main__":
    main()

