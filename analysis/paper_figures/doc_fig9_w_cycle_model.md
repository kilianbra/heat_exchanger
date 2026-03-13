# Fig9 Cycle-Coupled HEx Analysis — Summary

## Overview

`fig9_w_cycle_model.py` performs a **cycle-coupled heat exchanger (HEx) analysis** for a helicopter recuperated gas turbine. It uses parameters from `xflow.py`, solves for mass flow at constant shaft power, and produces weight-delta curves vs. a baseline unrecuperated engine. The script compares two fuel-delta models: (1) cycle efficiency η, and (2) dQ_o^M (practical unavailable creation).

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

### 7. Delta Mass Formulae — Cycle Model (η-based)

**Fuel delta** (from cycle efficiency):

```
Δm_fuel = (ṁ_fuel − ṁ_fuel,baseline) × t_mission
```
where **ṁ_fuel** = P_shaft / (LHV × η/100)

**HEx mass**:

```
m_HEx = (A/A_ref) × m_hex_ref    (m_hex_ref = 13.3 kg)
```

**Engine delta** (scaled with airflow):

```
Δm_engine = (ṁ − ṁ_baseline) × 23 kg/(kg/s)
```

**Cumulative (cycle model, red lines)**:
- cum_fuel = Δm_fuel
- cum_fuel+hex = Δm_fuel + m_HEx
- cum_total = Δm_fuel + m_HEx + Δm_engine

*t_mission* = 2 h, *LHV* = 43.2 MJ/kg, engine scaling = 23 kg/(kg/s)

---

### 8. Delta Mass Formulae — HEx Model (dQ_o^M–based)

**Fuel scaling factor**:

```
Q_max = ṁ_hot × c_p × (T_h,in − T_c,in)
factor_fuel = t_mission / (LHV_kWh/kg) × η_turb/η_ov × Q_max    [kg]
```

**Fuel delta** (from practical unavailable creation dQ_o^M):

```
Δm_fuel,dQoM = (dQ_o^M / Q_max) × factor_fuel
```

where **dQ_o^M/Q_max** is computed by `practical_unavailable_creation_hex()` in xflow.py (work potential change normalised by Q_max).

**Cumulative (HEx model, black lines, fuel + HEx only)**:
- cum_fuel (dQo^M) = Δm_fuel,dQoM
- cum_fuel+hex (dQo^M) = Δm_fuel,dQoM + m_HEx

(The full black solid plot line adds engine: cum_total = Δm_fuel,dQoM + m_HEx + Δm_engine, but the printed HEx model table omits engine mass.)

---

## Where Cycle Model vs. dQ_o^M Are Used

**Cycle model (red lines)** — used for optimization and primary curves:

- **Optimization**: `_optimal_ao_for_each_a_over_a_ref` minimizes `delta_fuel + m_hex + delta_engine` using cycle efficiency η from `calculate_recuperated_cycle_dp_eps`. Design points (ao, NTU, mdot, ε, Δp) come from this.
- **Red dashed** = delta_fuel. **Red solid** = delta_fuel + m_hex + delta_engine.

**dQ_o^M model (black lines)** — used for plotting comparison:

- **Same design points** as red (same ε, Δp, mdot from the cycle-coupled solve), but a different way to estimate fuel impact.
- At each point, `solve_mdot_at_constant_power` returns `dq` from `practical_unavailable_creation_hex` (dQ_o^M / Q_max).
- **Black dashed** = delta_fuel_dqom. **Black solid** = delta_fuel_dqom + m_hex + delta_engine.
- The optimizer does **not** use dQ_o^M; black lines are for comparison only.

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

- **Figure**: Δm vs. HEx core mass. Red square = design that minimizes cycle-model total (cum_total); red+black star = design that minimizes dQ_o^M-based total (cum_total from HEx model).
- **Exports**: SVG, TIFF, PNG, PDF in `Figs_current/`.
- **Console**: Reference conditions, baseline, optima, and mass deltas (including HEx model cumulatives: cum_fuel, cum_fuel+hex, no engine).
