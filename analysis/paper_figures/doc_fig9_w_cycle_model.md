# Fig9 Cycle-Coupled HEx Analysis — Summary

## Overview

`fig9_w_cycle_model.py` performs a **cycle-coupled heat exchanger (HEx) analysis** for a helicopter recuperated gas turbine. It uses parameters from `xflow.py`, solves for mass flow at constant shaft power, and produces weight-delta curves vs. a baseline unrecuperated engine. The script compares the **cycle-efficiency model** (red lines) with several **dQ_o^M–based variants** (black lines) that isolate two effects the cycle model captures: (1) **changing boundary conditions** (T_hot_in, pressure ratios), and (2) **changing mass flow** (→ Mach, ε, Δp). A printed variant comparison table shows optimal m_HEx and min Δm for each.

---

## Unit Conventions (SI)

| Quantity | Unit | Notes |
|----------|------|-------|
| **LHV** | 43.2 MJ/kg | Single definition (`LHV_MJ_per_kg`); derived `LHV_J_per_kg = 43.2e6` for all formulae |
| **w_net** | J/kg | Cycle uses c_p = 1070 J/(kg·K), T in K ⇒ work in J/kg |
| **P_shaft** | W | 697e3 W |
| **mdot**, **ṁ_fuel** | kg/s | |
| **Q_max** | W | ṁ × c_p × ΔT with c_p in J/(kg·K) |
| **t_mission** | s | mission_seconds = mission_hours × 3600 |

---

## Key Equations

### 1. Power–Mass Flow Constraint

At constant shaft power P_shaft:

```
P_shaft = mdot × w_net    ⇒    mdot = P_ref / w_net
```

- **w_net**: net specific work from the recuperated cycle (J/kg)
- **P_ref** = 697 kW: reference shaft power

**Coupled solve** (mdot, T_hot_in are coupled): `scipy.optimize.root` on residuals

```
res[0] = mdot − P_ref / w_net
res[1] = T_hot_in − T[4]
```

where T[4] is turbine exit (hot side inlet to recuperator) from the cycle; w_net, T[4] depend on ε, Δp, which depend on g²h, which depends on mdot and T_hot_in.

---

### 2. Mach Number and g²h (Dynamic Head Parameter)

```
M_in = M_ref × (mdot/mdot_ref) × √(T_hot_in/T_hot_in_ref) / (A_o/A_o_ref)
```

```
g²h = (1/2) × γ × M_in²
```

- **M_ref** = 0.11, **mdot_ref** ≈ 2.24 kg/s (from P_shaft / w_net at reference design)
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
where **ṁ_fuel** = P_shaft / (LHV × η/100)  [kg/s] with LHV in J/kg, η in %, P_shaft in W

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

**Fixed fuel scaling factor** (reference Q_max; same for all dQ^M variants):

```
Q_max_ref = ṁ_ref × c_p × (T_h,in − T_c,in)_ref     [W]
factor_fuel = t_mission_s / LHV_J_per_kg × η_turb/η_ov × Q_max_ref   [kg]
```

**Fuel delta** (from practical unavailable creation dQ_o^M):

```
Δm_fuel,dQoM = (dQ_o^M / Q_max) × factor_fuel
```

- **dQ_o^M/Q_max** from `practical_unavailable_creation_hex()` in xflow.py.
- **factor_fuel** is fixed at reference; dQ^M itself varies with boundary conditions and Mach.

**Cumulative (dQ^M variants)**:
- Effect 1: objective = dQ^M × factor_fuel + m_HEx (no delta_engine).
- Effect 2 / Effects 1+2: objective = dQ^M × factor_fuel + m_HEx [+ delta_engine if INCLUDE_DELTA_ENGINE].

---

## Toggle Flags (top of script)

| Flag | Default | Description |
|------|---------|-------------|
| **INCLUDE_DELTA_ENGINE** | `False` | If `True`, adds `(mdot − mdot_baseline) × 23` to all dQ^M-based objectives. Effect 1 (BC) uses uncoupled cycle and reference mdot, so delta_engine ≈ 0 regardless. |
| **HIDE_BLACK_LINES** | `False` | If `True`, black lines and black star are hidden in the plot (all printouts still shown). |

---

## Where Cycle Model vs. dQ_o^M Are Used

**Cycle model (red lines)** — primary optimization and curves:

- **Optimization**: `_optimal_ao_r_ref_for_each_a_r_ref` minimizes `delta_fuel + m_hex + delta_engine` using cycle efficiency η. Always includes delta_engine.
- **Red dashed** = delta_fuel. **Red solid** = delta_fuel + m_hex + delta_engine.
- **Red square + red star** = cycle-model optimum.

**dQ_o^M variants (black lines)** — isolate effects the cycle captures vs. a pure dQ^M approach:

| Variant | Effect | Description | delta_engine |
|---------|--------|-------------|--------------|
| **Effect 1 only (BC)** | Changing boundary conditions | Uncoupled cycle for T_hot_in; ref Mach for g²h. Actual t, p_ratios in dQ^M. | No (ref mdot) |
| **Effect 2 only (mdot/Mach)** | Changing mdot → Mach → ε, Δp | Coupled cycle; actual Mach; fixed DEFAULT BCs in dQ^M. | Toggle |
| **Effects 1+2 (BC+mdot)** | Both | Coupled cycle; actual BCs and actual Mach in dQ^M. | Toggle |

- **Black dashed** = Effect 1 only. **Black solid** = Effects 1+2.
- All dQ^M variants use a **fixed reference factor_fuel** (from Q_max at reference conditions) to convert dQ^M/Q_max → fuel mass.
- Effect 2 optimum is in the printed table only (not plotted).

---

## Key Functions

| Function | Purpose |
|----------|---------|
| `solve_mdot_at_constant_power` | Coupled solve for mdot, T_hot_in at given ao_r_ref, ntu. Returns mdot, w_net, T_hot_in, eps, dp_hot, dp_cold, dq (dQ^M/Q_max), eff, M_in. |
| `_optimal_ao_r_ref_for_each_a_r_ref` | Main sweep: for each A/A_ref, sweeps Ao/Ao_ref calling the coupled cycle once per point. Returns dict `{red, mdot_dqom, bc_mdot_dqom}` with optima for cycle model, Effect 2 (mdot), and Effects 1+2 (BC+mdot). |
| `_sweep_bc_only` | Effect 1 only: uncoupled cycle, reference Mach, actual BCs in dQ^M. Returns dict with `a`, `ao`, `obj` arrays. |

---

## Main Assumptions

| Assumption | Value / Description |
|------------|---------------------|
| **Constant shaft power** | P_shaft = 697 kW |
| **Gas properties** | γ = 1.4, ideal gas |
| **Cycle parameters** | PR = 9, TIT = 1500 K, η_poly,c = 0.88, η_poly,t = 0.84 |
| **Capacity ratio** | C_cold/C_hot = 1 |
| **Reference Mach** | M_in = 0.11 at reference design |
| **Pressure drop limit** | Δp/p ≤ 20% on both sides |
| **NTU range** | 0.02 ≤ NTU ≤ 8 |
| **Pressure drop model** | "inlet_density" for cold/hot Δp ratio |
| **Area scaling** | A/A_ref ∝ (A_o/A_o_ref)^0.587 × NTU |
| **Mission** | 2 h, LHV = 43.2 MJ/kg |
| **Engine mass** | 23 kg per kg/s air |
| **Temperature ratio** | T = T_hot_in/T_cold_in = 898/588 |
| **Pressure ratios** | p_cold_in/p_hot_in ≈ 8.48, p_dead/p_hot_in ≈ 0.96 |

---

## Algorithm Flow

1. **Baseline**: Compute unrecuperated cycle and mdot_baseline at P_shaft.
2. **Reference**: Get ε, Δp, w_net at g²h_ref, NTU_MATCH = 1.479.
3. **Coupled sweep**: For each A/A_ref, sweep A_o/A_o_ref, call `solve_mdot_at_constant_power` once per point. Simultaneously track optima for:
   - **red**: cycle η → delta_fuel + m_HEx + delta_engine
   - **mdot_dqom**: dQ^M (fixed DEFAULT BCs, actual Mach) × factor_fuel + m_HEx [+ delta_engine]
   - **bc_mdot_dqom**: dQ^M (actual BCs, actual Mach) × factor_fuel + m_HEx [+ delta_engine]
4. **BC-only sweep**: Separate `_sweep_bc_only` — uncoupled cycle, ref Mach, actual BCs in dQ^M. No delta_engine.
5. **Print** variant comparison table (opt m_HEx, min Δm per variant).
6. **Plot**: Red lines (cycle model), black lines (dQ^M variants) if not HIDE_BLACK_LINES.

---

## Outputs

- **Figure**: Δm vs. HEx core mass.
  - Red square + red star = cycle-model optimum (min cum_total).
  - Black star = BC+mdot dQ^M optimum (when HIDE_BLACK_LINES is False).
  - Red + black marker at baseline design (ao=1).
- **Exports**: SVG, TIFF, PNG, PDF in `Figs_current/`.
- **Console**:
  - INCLUDE_DELTA_ENGINE and factor_fuel at start.
  - Variant comparison table: opt m_HEx, min Δm for Effect 1, Effect 2, Effects 1+2, and red line.
  - Detailed table: 1st, ref, ref_opt, square (red opt), star (BC+mdot opt), last, unrecup.
