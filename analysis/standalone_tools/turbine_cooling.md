# Optimal Turbine Cooling Bleed Fraction: Lowest-Order Cycle Model Specification

## Purpose

Find the coolant mass flow fraction `ψ` that **maximises specific thrust** (or thermal efficiency) of a simple turbofan/turbojet core cycle, capturing the fundamental trade-off:

- **Too little cooling** → blade metal temperature limit constrains TET below its thermodynamic optimum → low efficiency.
- **Too much cooling** → compressed air bypasses combustion → lost turbine work exceeds the benefit of higher TET.

The model should recover **ψ ≈ 15–25 % of core flow** as optimal for modern aero-engine parameters (TET ≈ 1800–2000 K, OPR ≈ 40–50, T_blade_max ≈ 1100–1200 K with TBC).

---

## Literature Basis

| Reference                                                     | Contribution used                                                                                                      |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Holland & Thake (1980), _J. Aircraft_ 17, 412–418             | Semi-empirical cooling flow fraction formula (heat-exchanger analogy, constant blade temperature)                      |
| Young & Wilcock (2002), ASME GT-2002-30132 Part 1 & 2         | Modified Holland–Thake with TBC Biot number; row-by-row cooling loss accounting via entropy generation                 |
| El-Masri (1986), _J. Eng. Gas Turbines Power_ 108, 151–159    | Continuous cooled-expansion model (turbine as expander with heat extraction through walls)                             |
| Horlock, Watson & Jones (2000), ASME 2000-GT-0353             | "Limitations on gas turbine performance imposed by large turbine cooling flows" — frames the ψ-vs-TET trade explicitly |
| Wilcock, Young & Horlock (2005), _J. Turbomach._ 127, 109–116 | Comprehensive cooling loss breakdown: thermal mixing, aerodynamic mixing, kinetic energy dissipation                   |
| Masci & Sciubba (2019), _Energies_ 4(4), 36                   | Cooled-stage expansion model comparing El-Masri vs discrete mixer approaches                                           |
| Schneider (2024), _CEAS Aero. J._ (DLR)                       | Survey of 5 cooling-air estimation methods; confirms 20–25 % of HPC flow typical for modern aero HPT                   |

---

## Model Architecture (3 coupled sub-models)

```
┌───────────────────────────────────────────────────────┐
│              OPTIMISATION LOOP                        │
│   sweep ψ from 0 to 0.35  (or use scipy.optimize)    │
│                                                       │
│   For each ψ:                                         │
│     1. COOLING MODEL  → required T_ET given ψ         │
│        (or: given T_ET, check if ψ is sufficient)     │
│     2. CYCLE MODEL    → W_specific, η_th, SFC         │
│     3. Evaluate objective (e.g. max W_specific)        │
│                                                       │
│   Plot objective vs ψ → find optimum                  │
└───────────────────────────────────────────────────────┘
```

---

## Sub-Model 1: Cooling Flow Fraction (Holland–Thake / Young–Wilcock)

### Physical picture

The blade row is a heat exchanger. Hot gas at `T_g` transfers heat to the blade surface at `T_b`. Internal coolant enters at `T_c_in` and picks up heat. The coolant mass flow must be large enough that the maximum blade metal temperature `T_b` stays below the material limit `T_b_max`.

### Assumptions

1. **Constant blade metal temperature** along span (averaged representation of the peak — this is the Halls/Holland–Thake simplification).
2. **One lumped HPT stage** (NGV + rotor combined into a single equivalent cooled row). This is the key "lowest order" simplification. For a more refined version, do NGV and rotor separately.
3. **Film cooling included** via an overall film effectiveness parameter `ε_f`.
4. **Internal cooling** characterised by an internal cooling effectiveness `η_c`.
5. **Optional TBC** via a Biot number `Bi_TBC`.

### Definitions

```
Symbol      Description                                     Typical value
──────      ───────────                                     ─────────────
T_g         Gas temperature entering the cooled stage (≈TET) [K]   1800–2000
T_c_in      Coolant inlet temperature (= HPC delivery) [K]         800–950
T_b_max     Max allowable avg blade metal temperature [K]           1050–1200 (with TBC)
ε_f         Overall film cooling effectiveness [-]                  0.30–0.50
η_c         Internal cooling effectiveness [-]                      0.60–0.80
Bi_TBC      TBC Biot number (= h_g * t_TBC / k_TBC) [-]           0.10–0.20  (0 if no TBC)
St_g        External (gas-side) Stanton number [-]                  0.005–0.010
A_g/A_c     Ratio of gas-side to coolant-side wetted areas [-]     2.0–4.0
cp_g        Hot gas specific heat [J/(kg·K)]                        1150
cp_c        Coolant (air) specific heat [J/(kg·K)]                  1005
```

### Equation: Cooling flow fraction per blade row

From Young & Wilcock (2002), the required coolant-to-gas mass flow ratio for a single row is:

```
         cp_g          St_g * (A_g / A_passage)       (T_g_eff - T_b_max)
ψ_row = ───── × ──────────────────────────────── × ─────────────────────
         cp_c           η_c * (1 + Bi_TBC)            (T_b_max - T_c_in)
```

where the **effective gas temperature** seen by the blade (accounting for film cooling) is:

```
T_g_eff = T_g - ε_f * (T_g - T_c_in)
```

and `A_g / A_passage` is essentially `(blade perimeter × span) / (passage cross-section area)`, which collapses into the product `St_g × (A_g/A_c)` when non-dimensionalised. For a lumped single-stage model, define:

```
         cp_g       St_g * (A_g/A_c)     [T_g(1 - ε_f) + ε_f * T_c_in - T_b_max]
ψ_HPT = ───── × ──────────────────── × ─────────────────────────────────────────────
         cp_c     η_c * (1 + Bi_TBC)              (T_b_max - T_c_in)
```

**Simplified single-parameter form** (for quick prototyping):

Lump all the heat-transfer technology into one parameter `C_cool`:

```
         (T_g - T_b_max)
ψ = C_cool × ───────────────
              (T_b_max - T_c_in)
```

where `C_cool ≈ 0.045` for modern film-cooled blades with TBC (calibrate to give ψ ≈ 0.20 at T_g = 1900 K, T_c_in = 850 K, T_b_max = 1150 K).

### How ψ couples to T_ET

**Mode A — "Given T_ET, compute required ψ":** plug T_g = T_ET into equation above.

**Mode B — "Given ψ, compute max allowable T_ET":** invert:

```
T_ET_max(ψ) = T_b_max + (T_b_max - T_c_in) × ψ / C_cool
```

(with film cooling included, the full inversion from the detailed equation is straightforward algebra).

**For the optimisation, use Mode A:** sweep T_ET from 1400 K to 2200 K, compute required ψ(T_ET), then feed both into the cycle model.

---

## Sub-Model 2: Brayton Cycle with Cooling Bleed

### Assumptions

1. **Ideal gas with constant cp** (separate values for air and combustion products).
2. **Single-spool turbojet core** (compressor → combustor → HPT → nozzle or power turbine). Extension to two-spool is straightforward but unnecessary for capturing the trade-off.
3. **Coolant is bled from HPC exit** (worst case for work penalty; most representative).
4. **Coolant re-enters at HPT exit** after mixing (simple constant-pressure mixer). This is the "non-chargeable" cooling assumption used in most 0-D codes. Alternatively, use the "chargeable" model where coolant does partial work — see option below.
5. **Polytropic efficiencies** for compressor and turbine.
6. **No afterburner, no bypass** (pure core cycle for clarity; thrust = core specific thrust).

### Station numbering

```
Station   Location
───────   ────────
0         Ambient (freestream, static)
2         Compressor inlet (= 0 for ground-static)
3         Compressor exit / combustor inlet
4         Combustor exit = Turbine Entry (TET = T_04)
4.1       HPT exit (after cooled expansion + coolant mixing)
5         Nozzle exit (or free power turbine exit)
```

### Input parameters

```
Symbol         Description                              Baseline value
──────         ───────────                              ──────────────
T_0            Ambient temperature [K]                  288.15  (ISA SLS)
p_0            Ambient pressure [Pa]                    101325
OPR            Overall pressure ratio [-]               40
η_pc           Compressor polytropic efficiency [-]     0.90
η_pt           Turbine polytropic efficiency [-]        0.90
T_04           Turbine entry temperature [K]            VARIABLE (1400–2200)
cp_a           Specific heat of air [J/(kg·K)]          1005
cp_g           Specific heat of combustion gas [J/(kg·K)] 1148
γ_a            Ratio of specific heats, air [-]         1.40
γ_g            Ratio of specific heats, gas [-]         1.33
η_b            Combustor efficiency [-]                 0.99
η_mech         Mechanical efficiency [-]                0.99
Δp_b/p_03      Combustor fractional pressure loss [-]   0.05
Q_R            Fuel heating value [J/kg]                43.0e6
ψ              Coolant mass fraction (= m_cool/m_core) VARIABLE (0–0.35)
```

### Equations (evaluate sequentially)

#### Step 1: Compressor

```python
# Temperature ratio
T_03 = T_0 * OPR ** ((γ_a - 1) / (γ_a * η_pc))

# Specific compressor work
w_c = cp_a * (T_03 - T_0)     # [J/kg of core air]

# Coolant temperature (bled from HPC exit)
T_c_in = T_03
```

#### Step 2: Combustor (reduced mass flow through combustor)

Only `(1 - ψ)` of the core air goes through the combustor.

```python
# Fuel-air ratio (per unit of AIR entering combustor, not total core)
f = (cp_g * T_04 - cp_a * T_03) / (η_b * Q_R - cp_g * T_04)

# Fuel flow per unit TOTAL core mass flow
f_total = f * (1 - ψ)
```

#### Step 3: HPT cooled expansion

**Option A — Simple mixer model (recommended for lowest order):**

The hot gas `(1 - ψ)(1 + f)` expands through the HPT, then mixes at constant pressure with coolant `ψ` at temperature `T_c_in`.

First, compute uncooled HPT exit temperature. The HPT must produce enough work to drive the compressor:

```python
# Power balance: compressor work = turbine work (per unit total core flow)
# w_c * 1.0 = η_mech * (1 - ψ) * (1 + f) * cp_g * (T_04 - T_045_hot)
# where T_045_hot is the hot-gas temperature at HPT exit (before mixing)

T_045_hot = T_04 - w_c / (η_mech * (1 - ψ) * (1 + f) * cp_g)
```

Check that HPT expansion ratio is feasible:

```python
# HPT pressure ratio (from polytropic relation)
π_HPT = (T_04 / T_045_hot) ** (γ_g * η_pt / (γ_g - 1))

# Combustor exit pressure
p_04 = p_0 * OPR * (1 - Δp_b/p_03)

# HPT exit pressure
p_045 = p_04 / π_HPT
```

Now mix with coolant at constant pressure:

```python
# Mixing (enthalpy balance)
# [(1-ψ)(1+f) * cp_g * T_045_hot + ψ * cp_a * T_c_in]
# = [(1-ψ)(1+f) + ψ] * cp_mix * T_045_mixed

m_hot = (1 - ψ) * (1 + f)
m_total_after_mix = m_hot + ψ

# For simplicity, use mass-weighted cp:
cp_mix = (m_hot * cp_g + ψ * cp_a) / m_total_after_mix

T_045_mixed = (m_hot * cp_g * T_045_hot + ψ * cp_a * T_c_in) / (m_total_after_mix * cp_mix)
```

**Option B — Continuous cooled expansion (El-Masri style):**

Model the expansion as a polytropic process with continuous heat loss to the coolant. This is more physical but adds an ODE. For lowest-order, Option A is sufficient and is what most 0-D codes (GasTurb, Turbomatch) effectively do.

#### Step 4: Nozzle / Exhaust

Expand from `p_045` to `p_0`:

```python
# Nozzle pressure ratio
NPR = p_045 / p_0

# Nozzle exit temperature (isentropic with γ_mix or γ_g)
γ_mix = cp_mix / (cp_mix - R_mix)    # or just use γ_g as approximation
T_5 = T_045_mixed * (1 / NPR) ** ((γ_mix - 1) / γ_mix)

# Exit velocity
V_j = sqrt(2 * cp_mix * (T_045_mixed - T_5))
```

#### Step 5: Performance metrics

```python
# Specific thrust [N·s/kg] (per unit total core mass flow)
F_s = m_total_after_mix * V_j      # for ground-static (V_0 = 0)

# SFC [kg/(N·s)]
SFC = f_total / F_s

# Thermal efficiency
η_th = F_s * V_j / (2 * f_total * Q_R)    # simplified; exact uses KE balance

# Alternative: specific work (for shaft-power version)
# w_net = w_turbine_total - w_c  (all per unit core mass flow)
```

---

## Sub-Model 3: Coupling and Optimisation

### Procedure

```python
import numpy as np
from scipy.optimize import minimize_scalar

def evaluate_cycle(T_ET, params):
    """Given T_ET, compute required ψ, then compute cycle performance."""

    # 1. Cooling model: required ψ
    T_c_in = params.T_0 * params.OPR ** ((params.γ_a - 1) / (params.γ_a * params.η_pc))
    ψ = params.C_cool * (T_ET - params.T_b_max) / (params.T_b_max - T_c_in)
    ψ = max(ψ, 0.0)  # no cooling needed if T_ET < T_b_max

    # 2. Cycle model (equations from Sub-Model 2)
    # ... compute F_s, SFC, η_th ...

    return F_s, SFC, η_th, ψ

# Sweep T_ET and find optimum
T_ET_range = np.linspace(1400, 2200, 200)
results = [evaluate_cycle(T, params) for T in T_ET_range]

# The optimum T_ET (and corresponding ψ) maximises F_s or minimises SFC
```

### What to plot

1. **ψ vs T_ET** — the cooling demand curve (monotonically increasing).
2. **Specific thrust vs T_ET** — rises, then rolls over as cooling penalty dominates.
3. **SFC vs T_ET** — dips to a minimum, then rises.
4. **Specific thrust vs ψ** — the key trade-off plot. Should show a clear peak at ψ ≈ 0.15–0.25.
5. **η_th vs ψ** — shows the efficiency cliff as cooling fraction grows.

---

## Baseline Parameter Set (calibrated to "modern high-BPR turbofan core")

```
Parameter       Value       Notes
─────────       ─────       ─────
T_0             288.15 K    ISA sea level
p_0             101325 Pa   ISA sea level
OPR             40          Typical of Trent 1000 / GEnx class
η_pc            0.90        Polytropic
η_pt            0.90        Polytropic (uncooled value)
T_b_max         1150 K      Ni superalloy + TBC
C_cool          0.045       Calibrated to ψ≈0.20 at T_ET=1900K
Q_R             43.0e6 J/kg Jet-A
η_b             0.99
η_mech          0.99
Δp_b/p_03       0.05
cp_a            1005 J/(kg·K)
cp_g            1148 J/(kg·K)
γ_a             1.40
γ_g             1.33
```

With these values, the model should predict:

- Optimal T_ET ≈ 1850–1950 K
- Optimal ψ ≈ 0.18–0.22
- These are consistent with published data for modern civil aero HPTs

---

## Sensitivity Studies to Run

Once the baseline works, vary one parameter at a time to build intuition:

1. **T_b_max**: 1000 K → 1300 K (effect of better materials / TBC).
2. **C_cool**: 0.03 → 0.07 (effect of better cooling technology — lower C_cool means less flow needed per ΔT, shifts optimum to higher T_ET).
3. **OPR**: 20 → 60 (higher OPR → hotter coolant → more cooling needed → earlier diminishing returns on T_ET).
4. **η_pt**: 0.85 → 0.93 (as found by Wilcock et al., this is a dominant parameter).

---

## Key Physical Insight the Model Should Reveal

At low ψ (low T_ET), the cycle is "leaving performance on the table" — Carnot efficiency improves with T_ET and there is negligible cooling penalty. At high ψ, the marginal benefit of extra T_ET is small (diminishing returns on Carnot) while the marginal cost of extra coolant is large (every extra 1% coolant costs ~0.5–1% in SFC). The optimum sits where `∂(SFC)/∂(T_ET) = 0`, which implicitly defines the optimal ψ through the cooling model coupling.

The OPR-vs-T_ET interaction is critical: higher OPR raises T_c_in (compressor delivery temperature), which degrades the cooling ΔT denominator `(T_b_max - T_c_in)`, dramatically increasing the required ψ for a given T_ET. This is why simply cranking up OPR without raising T_ET (or improving cooling tech) hits a wall — and why cooled cooling air (CCA) heat exchangers are being pursued for next-gen engines.

---

## Extensions (Optional, for V2)

1. **Two-spool model**: separate HPT and LPT, with HPT cooling only. More realistic work split.
2. **Row-by-row cooling**: separate NGV (uses ~60% of cooling air) and rotor (uses ~40%). Different T_g seen by each row due to expansion.
3. **Cooled turbine efficiency penalty**: η_pt_cooled = η_pt_uncooled - K_loss × ψ, where K_loss ≈ 0.1–0.3 (accounts for aerodynamic mixing losses from film holes). Young & Wilcock (2002) give entropy-based formulation.
4. **Variable cp** via polynomial fits (more accurate at high T but adds complexity).
5. **Bypass ratio coupling**: for a turbofan, higher ψ reduces core power available to drive the fan → less bypass thrust.
