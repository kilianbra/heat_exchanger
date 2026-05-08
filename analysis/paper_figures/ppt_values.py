"""
Presentation helper: single-point availability (Fig 4-style) + NTU sweep (Fig 6c-style).

- Baseline point: fixed epsilon; hot dp/p_in = DP_HOT_OF_INLET; cold dp/p_in follows xflow
  (same rule as the sweep: pressure_drop_ratio * dp_hot).
- Sweep: balanced counterflow epsilon–NTU model from xflow; linear dp vs NTU (SHOW_CUBIC off).
- Area ratio: A/A_ref = NTU / NTU_ref where NTU_ref matches EPSILON_POINT on the sweep.
"""

from __future__ import annotations

import numpy as np
import xflow
from fig4_bar_chart import static_pressure_from_stagnation, static_temperature_from_stagnation
from xflow import (
    calculate_epsilon_ntu_curve,
    calculate_pressure_drop_ratio,
    classical_unavailable_creation_hex,
    practical_unavailable_creation_hex,
)

# --- Match fig6c / fig8 geometric defaults ---
C_COLD_OVER_C_HOT = 1.0
ST_OVER_F = 0.4
F_C_OVER_F_H = 0.25
D_R = 0.25
A_R = 0.92
MOLAR_MASS_RATIO = 1.0
PRESSURE_DROP_ASSUMPTION = "inlet_density"  # (dp_c/p_c,in)/(dp_h/p_h,in) = 0.56
# PRESSURE_DROP_ASSUMPTION = "dp_c=dp_h"  # (dp_c/p_c,in)/(dp_h/p_h,in) = 1.0

# --- New-case inlets (stagnation); Mach same both sides ---
GAMMA = 1.4
T_CIN_STAG = 600.0  # K
T_HIN_STAG = 900.0  # K
T0_STAG = 288.0  # K
P0 = 1.0  # bar (dead)
P_CIN = 10.0  # bar stagnation at cold inlet
P_HIN = 1.1  # bar stagnation at hot inlet

MACH_H = 0.11
MACH_C = 0.05

# --- Point calculation ---
EPSILON_POINT = 0.6
# Hot-side relative loss; cold side = pressure_drop_ratio * DP_HOT_OF_INLET (see xflow.calculate_epsilon_ntu_curve)
DP_HOT_OF_INLET = 0.06

# --- Sweep limits ---
# fig6c uses ntu_max=15, dp_max=0.2; for this presentation case the post-optimum branch
# often needs more NTU before av_prac falls back to the fixed-epsilon baseline (dp hits cap first).
NTU_MAX_SWEEP = 40.0
DP_MAX_SWEEP = 0.35


def baseline_point():
    """Fig 4-style totals at EPSILON_POINT; cold dp matches sweep assumption (ratio * dp_hot)."""
    p_hin = P_HIN
    t0_static = static_temperature_from_stagnation(T0_STAG, 0.0, GAMMA)
    t_cin_static = static_temperature_from_stagnation(T_CIN_STAG, MACH_C, GAMMA)
    p_hin_static = static_pressure_from_stagnation(p_hin, MACH_H, GAMMA)
    p_cin_static = static_pressure_from_stagnation(P_CIN, MACH_C, GAMMA)

    t_ratio_stag = T_HIN_STAG / T_CIN_STAG
    t_dead_over_t_cold_in = t0_static / t_cin_static
    p_cold_in_over_p_hot_in = p_cin_static / p_hin_static
    p_dead_over_p_hot_in = P0 / p_hin_static

    sigma_r = D_R * A_R
    pressure_drop_ratio = calculate_pressure_drop_ratio(
        PRESSURE_DROP_ASSUMPTION,
        C_COLD_OVER_C_HOT,
        t_ratio_stag,
        D_R,
        MOLAR_MASS_RATIO,
        sigma_r,
        p_cold_in_over_p_hot_in,
    )
    dp_hot = float(DP_HOT_OF_INLET)
    dp_cold = float(pressure_drop_ratio * dp_hot)
    if dp_cold < 0.0 or dp_cold >= 1.0:
        raise ValueError(
            f"Derived cold dp/p_in = {dp_cold:.4g} invalid; check DP_HOT_OF_INLET and PRESSURE_DROP_ASSUMPTION"
        )

    eps_a = np.atleast_1d(EPSILON_POINT)
    dp_h = np.atleast_1d(dp_hot)
    dp_c = np.atleast_1d(dp_cold)
    mask = np.ones_like(eps_a, dtype=bool)

    av_class_total = float(
        np.atleast_1d(
            -classical_unavailable_creation_hex(
                eps_a,
                t_ratio_stag,
                dp_h,
                dp_c,
                mask,
                t_dead_over_t_cold_in=t_dead_over_t_cold_in,
                gamma=GAMMA,
            )
        ).flat[0]
    )
    av_prac_total = float(
        np.atleast_1d(
            -practical_unavailable_creation_hex(
                eps_a,
                t_ratio_stag,
                dp_h,
                dp_c,
                mask,
                p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
                p_dead_over_p_hot_in=p_dead_over_p_hot_in,
                gamma=GAMMA,
            )
        ).flat[0]
    )

    return {
        "t_ratio_stag": t_ratio_stag,
        "t_dead_over_t_cold_in": t_dead_over_t_cold_in,
        "p_cold_in_over_p_hot_in": p_cold_in_over_p_hot_in,
        "p_dead_over_p_hot_in": p_dead_over_p_hot_in,
        "T0_static": t0_static,
        "T_cin_static": t_cin_static,
        "P_hin_static": p_hin_static,
        "P_cin_static": p_cin_static,
        "P_hin_stag": p_hin,
        "pressure_drop_ratio": pressure_drop_ratio,
        "dp_hot_point": dp_hot,
        "dp_cold_point": dp_cold,
        "av_class_total": av_class_total,
        "av_prac_total": av_prac_total,
    }


def _interp_x_at_y(x: np.ndarray, y: np.ndarray, y_target: float) -> float:
    """Linear interpolation: find x where y(x) = y_target (single root between brackets)."""
    if len(x) < 2:
        raise ValueError("Need at least two points for interpolation")
    dy = y - y_target
    for i in range(len(dy) - 1):
        if dy[i] == 0:
            return float(x[i])
        if dy[i] * dy[i + 1] < 0:
            t = -dy[i] / (dy[i + 1] - dy[i])
            return float(x[i] + t * (x[i + 1] - x[i]))
    raise ValueError(f"No bracket for y_target={y_target} in y range [{y.min()}, {y.max()}]")


def sweep_practical(
    t_ratio_stag: float,
    t_dead_over_t_cold_in: float,
    p_cold_in_over_p_hot_in: float,
    p_dead_over_p_hot_in: float,
    g2_h: float,
    pressure_drop_ratio: float,
):
    xflow.SHOW_CUBIC = False

    ntu, epsilon, dp_hot, dp_cold, validity_mask = calculate_epsilon_ntu_curve(
        C_COLD_OVER_C_HOT,
        ST_OVER_F,
        F_C_OVER_F_H,
        D_R,
        g2_h,
        ntu_max=NTU_MAX_SWEEP,
        dp_max=DP_MAX_SWEEP,
        pressure_drop_percent_ratio_cold_over_hot=pressure_drop_ratio,
    )

    ntu_v = ntu[validity_mask]
    eps_v = epsilon[validity_mask]
    dp_h_v = dp_hot[validity_mask]
    dp_c_v = dp_cold[validity_mask]

    pu = practical_unavailable_creation_hex(
        epsilon,
        t_ratio_stag,
        dp_hot,
        dp_cold,
        validity_mask,
        p_cold_in_over_p_hot_in=p_cold_in_over_p_hot_in,
        p_dead_over_p_hot_in=p_dead_over_p_hot_in,
        gamma=GAMMA,
    )
    av_prac_v = -np.asarray(pu, dtype=float)

    cu = classical_unavailable_creation_hex(
        epsilon,
        t_ratio_stag,
        dp_hot,
        dp_cold,
        validity_mask,
        t_dead_over_t_cold_in=t_dead_over_t_cold_in,
        gamma=GAMMA,
    )
    av_class_v = -np.asarray(cu, dtype=float)

    return ntu_v, eps_v, dp_h_v, dp_c_v, av_prac_v, av_class_v


def main():
    base = baseline_point()
    target_av = base["av_prac_total"]

    g2_h = 0.5 * GAMMA * MACH_H**2
    pressure_drop_ratio = base["pressure_drop_ratio"]

    ntu_v, eps_v, dp_h_v, dp_c_v, av_prac_v, _ = sweep_practical(
        base["t_ratio_stag"],
        base["t_dead_over_t_cold_in"],
        base["p_cold_in_over_p_hot_in"],
        base["p_dead_over_p_hot_in"],
        g2_h,
        pressure_drop_ratio,
    )

    if len(ntu_v) < 3:
        raise RuntimeError("Sweep produced too few valid NTU points; raise DP_MAX or check inputs")

    # NTU_ref: match EPSILON_POINT on the sweep (area reference)
    if eps_v.min() <= EPSILON_POINT <= eps_v.max():
        ntu_ref = _interp_x_at_y(ntu_v, eps_v, EPSILON_POINT)
    else:
        raise RuntimeError(f"EPSILON_POINT={EPSILON_POINT} outside sweep range [{eps_v.min():.4f}, {eps_v.max():.4f}]")

    a_over_a_ref = ntu_v / ntu_ref

    idx_opt = int(np.nanargmax(av_prac_v))
    if idx_opt == 0 or idx_opt == len(av_prac_v) - 1:
        print("Warning: practical optimum lies on sweep boundary; results may be unreliable")

    ntu_opt = float(ntu_v[idx_opt])
    a_opt = float(a_over_a_ref[idx_opt])
    print("=" * 72)
    print("ppt_values: presentation numbers")
    print("=" * 72)
    print(f"Inlets (stagnation): T_h,in = {T_HIN_STAG:.0f} K, T_c,in = {T_CIN_STAG:.0f} K")
    print(f"Mach: M_h = {MACH_H:.3f}, M_c = {MACH_C:.3f}; gamma = {GAMMA}")
    print(f"P_c,in (stagnation) = {P_CIN:.1f} bar; P_h,in (stagnation) = {P_HIN:.1f} bar")
    print(f"  baseline dp_h/p_h,in = {DP_HOT_OF_INLET:.3f}")
    print(f"Dead state: T0 = {T0_STAG:.0f} K, P0 = {P0:.1f} bar")
    print(f"Static cold inlet: T = {base['T_cin_static']:.2f} K, P = {base['P_cin_static']:.4f} bar")
    print(
        f"Static hot inlet:  T = {static_temperature_from_stagnation(T_HIN_STAG, MACH_H, GAMMA):.2f} K, P = {base['P_hin_static']:.4f} bar"
    )
    print(
        f"t = T_h,stag/T_c,stag = {base['t_ratio_stag']:.4f}; T0_static/T_c,static = {base['t_dead_over_t_cold_in']:.4f}"
    )
    print(
        f"p_c,in/p_h,in (static) = {base['p_cold_in_over_p_hot_in']:.4f}; p_dead/p_h,in = {base['p_dead_over_p_hot_in']:.4f}"
    )
    print()
    print(
        f"--- Baseline point (fixed epsilon; dp split = {PRESSURE_DROP_ASSUMPTION!r}, "
        f"(dp_c/p_c,in)/(dp_h/p_h,in) = {pressure_drop_ratio:.6f}) ---"
    )
    print(
        f"epsilon = {EPSILON_POINT}; dp_h/p_h,in = {base['dp_hot_point']:.4f}; "
        f"dp_c/p_c,in = {base['dp_cold_point']:.4f}"
    )
    print(f"av_class_total / Q_max = {base['av_class_total'] * 100:.4f} %")
    print(f"av_prac_total / Q_max  = {base['av_prac_total'] * 100:.4f} %")
    print()
    print(f"--- Sweep (xflow): pressure-drop assumption = {PRESSURE_DROP_ASSUMPTION!r} ---")
    print(f"(dp_c/p_c,in)/(dp_h/p_h,in) = {pressure_drop_ratio:.6f}; g2_h = {g2_h:.4e}; NTU in [0.1, {NTU_MAX_SWEEP}]")
    print(f"NTU_ref (epsilon = {EPSILON_POINT}) = {ntu_ref:.4f}  =>  A/A_ref = NTU/NTU_ref")
    print()
    print("--- Optimum (max practical availability on sweep) ---")
    print(f"NTU_opt = {ntu_opt:.4f}, A/A_ref_opt = {a_opt:.4f}, epsilon_opt = {eps_v[idx_opt]:.4f}")
    print(
        f"dp_h/p_h,in = {dp_h_v[idx_opt] * 100:.2f} %, dp_c/p_c,in = {dp_c_v[idx_opt] * 100:.2f} %, "
        f"sum = {(dp_h_v[idx_opt] + dp_c_v[idx_opt]) * 100:.2f} % (not additive frac of same basis)"
    )
    print(f"av_prac_total_opt / Q_max = {av_prac_v[idx_opt] * 100:.4f} %")
    print()

    # Match baseline av_prac on the high-NTU side of optimum
    rhs_ntu = ntu_v[ntu_v > ntu_opt]
    rhs_av = av_prac_v[ntu_v > ntu_opt]
    if len(rhs_ntu) < 2:
        raise RuntimeError("Not enough points past optimum for crossing search")

    diff = rhs_av - target_av
    crossed = False
    ntu_match = None
    for i in range(len(diff) - 1):
        if diff[i] == 0:
            crossed = True
            ntu_match = float(rhs_ntu[i])
            break
        if diff[i] * diff[i + 1] < 0:
            crossed = True
            t = -diff[i] / (diff[i + 1] - diff[i])
            ntu_match = float(rhs_ntu[i] + t * (rhs_ntu[i + 1] - rhs_ntu[i]))
            break

    if not crossed or ntu_match is None:
        print("--- Matched design (same av_prac as baseline, NTU > NTU_opt): NOT FOUND in sweep ---")
        print(f"  Target av_prac / Q_max = {target_av * 100:.4f} %")
        print("  Try increasing NTU_MAX_SWEEP or DP_MAX_SWEEP, or check monotonicity past optimum.")
    else:
        eps_m = float(np.interp(ntu_match, ntu_v, eps_v))
        dp_h_m = float(np.interp(ntu_match, ntu_v, dp_h_v))
        dp_c_m = float(np.interp(ntu_match, ntu_v, dp_c_v))
        av_m = float(np.interp(ntu_match, ntu_v, av_prac_v))
        a_m = ntu_match / ntu_ref
        print("--- Matched design (same av_prac as baseline, other side of optimum) ---")
        print(f"NTU = {ntu_match:.4f}, A/A_ref = {a_m:.4f}, epsilon = {eps_m:.4f}")
        print(
            f"dp_h/p_h,in = {dp_h_m * 100:.2f} %, dp_c/p_c,in = {dp_c_m * 100:.2f} %, "
            f"sum = {(dp_h_m + dp_c_m) * 100:.2f} %"
        )
        print(f"av_prac_total / Q_max = {av_m * 100:.4f} % (target {target_av * 100:.4f} %)")


if __name__ == "__main__":
    main()
