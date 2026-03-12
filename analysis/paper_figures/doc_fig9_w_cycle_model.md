# Fig8 Full Cycle Analysis — Summary

## Overview

`fig8_full_cycle.py` performs a **cycle-coupled heat exchanger (HEx) analysis** for a helicopter recuperated gas turbine. It uses parameters from `xflow.py`, solves for mass flow at constant shaft power, and produces three weight-delta curves vs. a baseline unrecuperated engine:

1. **Delta fuel** — fuel mass change from cycle efficiency
2. **Delta fuel + HEx** — fuel plus heat exchanger core mass
3. **Delta fuel + HEx + engine** — total mass change including scaled engine mass

The script sweeps HEx sizes, optimizes for minimum total mass, and compares results with the baseline and between a cycle-efficiency model and a dQ_o^M (practical unavailable creation) model.

---

## Key Equations

### 1. Power–Mass Flow Constraint

At constant shaft power P_shaft:

```
P_shaft = mdot × w_net    ⇒    mdot = P_ref / w_net
```

- **w_net**: net specific work from the recuperated cycle (J/kg)
- **P_ref** = 697 kW: reference shaft power

---

### 2. Mach Number and g²h (Dynamic Head Parameter)

```
M_in = M_ref × (mdot/mdot_ref) × √(T_hot_in/T_hot_in_ref) / (A_o/A_o_ref)
```

```
g²h = (1/2) × γ × M_in²
```

- **M_ref** = 0.1, **mdot_ref** = 2.3 kg/s
- **A_o**: hot-side free-flow area
- **g²h** couples flow rate, inlet temperature, and geometry via dynamic pressure

---

### 3. Heat Exchanger Effectiveness (ε–NTU)

For counterflow, aligned-flow exchanger with C_ratio = C_min/C_max:

```
ε = [1 − exp(−NTU×(1 − C_ratio))] / [1 − C_ratio × exp(−NTU×(1 − C_ratio))]
```

(Kays & London, counterflow)

For C_ratio ≈ 1: **ε = NTU / (NTU + 1)**

---

### 4. Pressure Drop (Fraction of Inlet Pressure)

```
Δp_hot/p_in_hot = g²h × NTU × [ 1/(St/f_hot) × C_min/C_hot
                                + 1/(f_c/f_hot) × 1/(St/f_c) × d_r × C_min/C_cold ]
```

```
Δp_cold/p_in_cold = pressure_drop_ratio × Δp_hot/p_in_hot
```

- **d_r** = σ_r/A_r (cold/hot)
- **St/f**: Stanton number to friction factor ratio
- Pressure drop scales linearly with NTU

---

### 5. Area / Mass Scaling

```
A/A_ref = (NTU / NTU_MATCH) × (A_o/A_o_ref)^0.587
```

```
m_HEx = (A/A_ref) × m_hex_ref    with  m_hex_ref = 13.3 kg
```

- NTU and A_o linked via an empirical exponent ≈ 0.587

---

### 6. Cycle Model (recuperated)

The recuperated cycle is computed in `calculate_recuperated_cycle_dp_eps()` with:

- **Compressor**: T₂ = T₁ × PR^((γ−1)/(γ×η_poly,c))
- **Combustor**: heats to TIT at constant pressure
- **Turbine**: T₄ = T₃ × (PR_turb^−1)^((γ−1)/γ × η_poly,t)
- **Recuperator**: temperature rise ΔT = ε × max(T₄ − T₂, 0)
- **Efficiency**: η = w_net / Q_in × 100%

---

### 7. Mass Deltas vs. Baseline Unrecuperated

```
Δm_fuel = (mdot_fuel − mdot_fuel_baseline) × t_mission
```
where **mdot_fuel** = P_shaft / (LHV × η/100)

```
Δm_engine = (mdot − mdot_baseline) × 23 kg/(kg/s)
```

```
Total Δm = Δm_fuel + m_HEx + Δm_engine
```

- **t_mission** = 2 h, **LHV** = 43.2 MJ/kg
- Engine scaling: 23 kg per kg/s of air flow

---

### 8. Practical Unavailable Creation dQ_o^M / Q_max

Implemented by `practical_unavailable_creation_hex()` in xflow.py. Used only for the **red comparison lines** in the plot.

- **Framework**: Work potential (exergy) change relative to dead state, **not** entropy generation or classical unavailable energy.
- **Inputs**: ε, T_hot_in/T_cold_in, Δp_hot, Δp_cold, p_cold_in/p_hot_in, p_dead/p_hot_in, γ.
- **Output**: Work potential change (dQ_o^M) normalized by Q_max. Positive = work potential created (recuperation); negative = destroyed.
- **Scaling to fuel mass**: delta_fuel_dqom = dQ_o^M × factor_fuel, where factor_fuel = mission_hours/(LHV_kWh/kg) × η_turb/η_ov × Q_max.

---

## Where Cycle Model vs. dQ_o^M Are Used

**Cycle model (black lines)** — used for optimization and primary curves:

- **Optimization**: `_optimal_ao_for_each_a_over_a_ref` minimizes `delta_fuel_cycle + m_hex + delta_engine` using cycle efficiency η from `calculate_recuperated_cycle_dp_eps`. Design points (ao, NTU, mdot, ε, Δp) come from this.
- **Delta fuel**: `delta_fuel = (mdot_fuel − mdot_fuel_baseline) × t_mission` with `mdot_fuel = P_shaft/(LHV × η/100)`.
- **Black dashed** = delta fuel only. **Black solid** = delta fuel + m_hex + delta_engine.

**dQ_o^M model (red lines)** — used only for plotting comparison:

- **Same design points** as black (same ε, Δp, mdot from the cycle-coupled solve), but a different way to estimate fuel impact.
- At each point, `solve_mdot_at_constant_power` returns `dq` from `practical_unavailable_creation_hex` (work potential change / Q_max).
- **Delta fuel**: `delta_fuel_dqom = dq_o_m_opt × factor_fuel`.
- **Red dashed** = delta_fuel_dqom. **Red solid** = delta_fuel_dqom + m_hex + delta_engine.
- The optimizer does **not** use dQ_o^M; red lines are for comparison only.

---

## Main Assumptions

| Assumption | Value / Description |
|------------|---------------------|
| **Constant shaft power** | P_shaft = 697 kW |
| **Gas properties** | γ = 1.4, ideal gas |
| **Cycle parameters** | PR = 9, TIT = 1500 K, η_poly,c = 0.88, η_poly,t = 0.84 |
| **Capacity ratio** | C_cold/C_hot = 1 |
| **Reference Mach** | M_in = 0.1 at reference design |
| **Pressure drop limit** | Δp/p ≤ 20% on both sides |
| **NTU range** | 0.02 ≤ NTU ≤ 5 |
| **Pressure drop model** | "inlet_density" for cold/hot Δp ratio |
| **Area scaling** | A/A_ref ∝ (A_o/A_o_ref)^0.587 × NTU |
| **Mission** | 2 h, LHV = 43.2 MJ/kg |
| **Engine mass** | 23 kg per kg/s air |
| **Temperature ratio** | T = T_hot_in/T_cold_in = 898/588 |
| **Pressure ratios** | p_cold_in/p_hot_in ≈ 8.48, p_dead/p_hot_in ≈ 0.96 |

---

## Algorithm Flow

1. **Baseline**: Compute unrecuperated cycle and mdot_baseline at P_shaft.
2. **Reference**: Get ε, Δp, w_net at g²h_ref, NTU_MATCH = 1.824.
3. **Sweep** over A/A_ref (≈ 80 points).
4. **Optimize**: For each A/A_ref, sweep A_o/A_o_ref (100 points), choose NTU from area relation, and solve for mdot at constant power.
5. **Solve** coupled system iteratively: mdot → g²h → ε, Δp → w_net → mdot_new.
6. **Objective**: Minimize Δm_fuel + m_HEx + Δm_engine.
7. **Plot**: Δm vs. m_HEx for both cycle-η and dQ_o^M models.

---

## Outputs

- **Figure**: Δm vs. HEx core mass. Black circle = design that minimizes cycle-model total; red square = design (same sweep) that minimizes dQ_o^M-based total.
- **Exports**: SVG, TIFF, PNG, PDF in `analysis/paper_figures/`.
- **Console**: Reference conditions, baseline, optima, and mass deltas.
