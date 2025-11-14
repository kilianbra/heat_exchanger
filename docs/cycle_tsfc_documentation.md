# Documentation for `zeli_og.py`

## Overview

This script performs a thermodynamic cycle analysis for a jet engine with a heat exchanger (HX) system. It calculates Thrust Specific Fuel Consumption (TSFC) for both baseline and preheated (with HX) configurations, and performs sensitivity studies on heat exchanger effectiveness and pressure drop.

## Key Components

### 1. Input Parameters

#### Mass Flow Rates

- `m_core = 64` [kg/s] - Core air mass flow rate
- `BPR = 10.89` - Bypass ratio
- `m_bypass = BPR * m_core` [kg/s] - Bypass air mass flow rate
- `m_combustor = 48.2` [kg/s] - Air mass flow into combustor (after cooling bleed)
- `m_split_core = 0.8 * m_core` [kg/s] - Core flow that bypasses HX (80%)
- `m_split_hx = 0.2 * m_core` [kg/s] - Core flow through HX (20%)
- `m_split_hx_coolant = 0.063 * m_split_hx` [kg/s] - Hydrogen coolant mass flow rate

#### Flight Conditions

- `flight_altitude_m = 11000` [m] - Flight altitude (overrides initial 39000 ft)
- $M_{\text{flight}} = 0.85$ - Flight Mach number
- $T_0$ [K] - Ambient temperature (calculated from altitude using ISA model)

  - For $h \leq 11000$ m:

    $$
    \begin{equation}
    T_0 = 288.15 - 0.0065 \cdot h
    \end{equation}
    $$

  - For $11000 < h \leq 20000$ m: $T_0 = 216.65$ (isothermal stratosphere)

- $p_0$ [Pa] - Ambient pressure (calculated from altitude)

  $$
  \begin{equation}
  p_0 = 101325 \cdot \left(1 - \frac{0.0065 \cdot h}{288.15}\right)^{5.2561}
  \end{equation}
  $$

#### Cycle States (Brayton Cycle)

The script defines thermodynamic states for the engine cycle:

| State | Description               | Temperature [K]    | Pressure [Pa]                     |
| ----- | ------------------------- | ------------------ | --------------------------------- |
| 0     | Ambient/freestream        | $T_0$ (calculated) | $p_0$ (calculated)                |
| 1     | After inlet/ram           | $T_1 = 248$        | $p_1 = 0.362 \times 10^5$         |
| 2     | After compressor          | $T_2 = 907$        | $p_2 = 25.37 \times 10^5$         |
| 3     | After combustor           | $T_3 = 1610$       | $p_3 = p_2$ (no combustor losses) |
| 4     | After turbine (baseline)  | $T_4 = 575$        | $p_4 = 0.368 \times 10^5$         |
| 4h2   | After HX (preheated case) | Calculated         | $p_{4\text{h}2} = p_4 \cdot dP$   |

#### Coolant Conditions

- `fluid_c = "parahydrogen"` - Coolant fluid
- `Tc_inlet = 40` [K] - Actual hydrogen inlet temperature
- `Tc_inlet_real = 300` [K] - Preheated inlet temperature (to avoid frosting)
- `Pc_inlet = 150e5` [Pa] - Coolant inlet pressure
- `Pc_outlet = 150e5 * 0.9` [Pa] - Coolant outlet pressure (10% drop)

#### Heat Exchanger Performance

- `eps = 0.90` - Heat exchanger effectiveness (90%)
- `dP = 0.85` - Pressure retention factor (85% retained = 15% drop)

#### Nozzle Areas

- `Abypass = 6.78` [m²] - Bypass nozzle area
- `Acore = 1.024` [m²] - Core nozzle area (baseline)
- `Asplit_main = 0.816` [m²] - Main split nozzle area
- `Asplit_hx = 0.184` [m²] - HX split nozzle area

#### Fuel Properties

- `fuel_LHV = 120e6` [J/kg] - Lower heating value of hydrogen

### 2. Core Calculations

#### 2.1 Heat Exchanger Analysis

**Heat capacity rates:**

$$
\begin{equation}
C_{\text{hot}} = \dot{m}_{\text{HX}} \cdot c_{p,\text{hot}}(T_4, p_4)
\end{equation}
$$

$$
\begin{equation}
C_{\text{cold}} = \dot{m}_{\text{coolant}} \cdot c_{p,\text{cold}}(T_{c,\text{inlet,real}}, p_{c,\text{inlet}})
\end{equation}
$$

**Maximum possible heat transfer:**

$$
\begin{equation}
q_{\max} = \min(C_{\text{hot}}, C_{\text{cold}}) \cdot (T_4 - T_{c,\text{inlet,real}})
\end{equation}
$$

**Actual heat transfer:**

$$
\begin{equation}
q = \varepsilon \cdot q_{\max}
\end{equation}
$$

**Outlet temperatures:**

$$
\begin{equation}
T_{c,\text{outlet}} = T_{c,\text{inlet,real}} + \frac{q}{C_{\text{cold}}}
\end{equation}
$$

$$
\begin{equation}
T_{4\text{h}2} = T_4 - \frac{q}{C_{\text{hot}}}
\end{equation}
$$

**Hot side outlet pressure:**

$$
\begin{equation}
p_{4\text{h}2} = p_4 \cdot dP
\end{equation}
$$

where $dP = 0.85$ (means 15% pressure drop, 85% retained).

#### 2.2 Jet Velocity Calculation (`calc_vjet`)

Calculates exhaust jet velocity assuming isentropic compressible flow in a 1D nozzle.

**Function inputs:**

- $p_{0,\text{in}}$: Stagnation pressure at nozzle inlet [Pa]
- $T_{0,\text{in}}$: Stagnation temperature at nozzle inlet [K]

**Function outputs:**

- $V_e$: Exit velocity [m/s]
- $p_e$: Exit pressure [Pa]
- $T_e$: Exit temperature [K]

**Logic:**

- Checks if flow is choked:

  $$
  \begin{equation}
  \frac{p_0}{p_{0,\text{in}}} < p_{r,\text{crit}}
  \end{equation}
  $$

  - If choked: Exit Mach number $M_e = 1$, exit pressure:

    $$
    \begin{equation}
    p_e = p_{0,\text{in}} \cdot p_{r,\text{crit}}
    \end{equation}
    $$

  - If not choked: Exit pressure $p_e = p_0$ (ambient), calculate Mach number from pressure ratio

- Uses ambient pressure $p_0$ for back pressure

**Critical pressure ratio:**

$$
\begin{equation}
p_{r,\text{crit}} = \left(\frac{2}{\gamma + 1}\right)^{\frac{\gamma}{\gamma - 1}}
\end{equation}
$$

For $\gamma = 1.4$, $p_{r,\text{crit}} \approx 0.528$.

#### 2.3 Thrust Calculations

**Baseline (no HX):**

$$
\begin{equation}
F_{\text{net,baseline}} = \dot{m}_{\text{core}} \cdot (V_4 - V_0) + A_{\text{core}} \cdot (p_{4e} - p_0)
\end{equation}
$$

**Preheated (with HX):**

$$
\begin{equation}
F_{\text{net,preheated}} = \dot{m}_{\text{split,core}} \cdot (V_4 - V_0) + \dot{m}_{\text{split,HX}} \cdot (V_{4\text{h}2} - V_0) + A_{\text{split,main}} \cdot (p_{4e} - p_0) + A_{\text{split,HX}} \cdot (p_{4\text{h}2e} - p_0)
\end{equation}
$$

**Total thrust (including bypass):**

$$
\begin{equation}
F_{\text{net,total,baseline}} = F_{\text{net,baseline}} + F_{\text{net,bypass}}
\end{equation}
$$

$$
\begin{equation}
F_{\text{net,total,preheated}} = F_{\text{net,preheated}} + F_{\text{net,bypass}}
\end{equation}
$$

#### 2.4 Fuel Consumption Calculations

**Baseline:**

$$
\begin{equation}
\dot{Q}_{\text{addition}} = \dot{m}_{\text{combustor}} \cdot \left[T_3 \cdot c_p(T_3, p_3) - T_2 \cdot c_p(T_2, p_2)\right]
\end{equation}
$$

$$
\begin{equation}
\dot{m}_{f,\text{baseline}} = \frac{\dot{Q}_{\text{addition}}}{\text{LHV}}
\end{equation}
$$

$$
\begin{equation}
\text{TSFC}_{\text{baseline}} = \frac{\dot{m}_{f,\text{baseline}}}{F_{\text{net,baseline}}}
\end{equation}
$$

**Preheated (with HX heat recovery):**

$$
\begin{equation}
Q_{\text{H}_2} = T_{c,\text{outlet}} \cdot c_p(T_{c,\text{outlet}}, p_{c,\text{outlet}}) - T_{c,\text{inlet}} \cdot c_p(T_{c,\text{inlet}}, p_{c,\text{inlet}})
\end{equation}
$$

$$
\begin{equation}
\dot{m}_{f,\text{preheated}} = \frac{\dot{Q}_{\text{addition}}}{\text{LHV} + Q_{\text{H}_2}}
\end{equation}
$$

$$
\begin{equation}
\text{TSFC}_{\text{preheated}} = \frac{\dot{m}_{f,\text{preheated}}}{F_{\text{net,preheated}}}
\end{equation}
$$

**Note:** The heat recovery term $Q_{\text{H}_2}$ represents the sensible heat picked up by hydrogen, which reduces the required fuel input.

### 3. Sensitivity Studies

#### 3.1 Pressure Drop Sensitivity

- **Fixed:** Effectiveness $\varepsilon = 0.90$
- **Varied:** Pressure drop from 0% to 47.5% (in steps of 2.5%)
- **Outputs:** Jet velocity excess $(V_{\text{EXIT,HX}} - V_{\text{flight}})$, net thrust, TSFC change

#### 3.2 Effectiveness Sensitivity

- **Fixed:** Pressure drop $(1 - dP) \times 100 = 15\%$

- **Varied:** Effectiveness from 70% to 100%
- **Outputs:** Jet velocity excess, net thrust, TSFC change, fuel mass flow

#### 3.3 Combined Sensitivity Study

- **Varied:** Both effectiveness (70-100%) and pressure drop (0-35%)
- **Outputs:** 2D contour plots showing:
  - Net thrust $(F_{\text{NET,HX}})$ vs effectiveness & pressure drop
  - TSFC change vs effectiveness & pressure drop

### 4. Outputs and Plots

#### 4.1 Printed Results

- Cycle state temperatures and pressures
- Jet velocities $(V_0, V_4, V_{4\text{h}2}, V_{\text{bp}})$
- Net thrust values (bypass, baseline, preheated)
- Fuel mass flow rates and TSFC values
- Percentage changes in thrust and TSFC

#### 4.2 Generated Plots

1. **T-s Diagram:** Temperature-entropy diagram showing cycle states and isobars
2. **Pressure Drop Sensitivity:** 3-axis plot showing velocity excess, thrust, and TSFC change
3. **Effectiveness Sensitivity:** 3-axis plot showing velocity excess, thrust, and TSFC change
4. **Combined Sensitivity Contours:** 2-panel figure with:
   - Net thrust contours (left)
   - TSFC change contours (right)
   - Design points marked (Design A: 83.6% eps, 18.2% dP; Design B: 86.7% eps, 4.2% dP)

### 5. Key Assumptions

1. **Perfect Gas Model:** Uses CoolProp's perfect gas model for air and parahydrogen
2. **No Combustor Losses:** $p_3 = p_2$ (no pressure drop in combustor)
3. **Isentropic Nozzle Flow:** Jet velocity calculated assuming isentropic expansion
4. **Constant Properties:** Heat capacity rates calculated at inlet conditions
5. **Heat Exchanger:** NTU-effectiveness method with $\varepsilon = 0.90$
6. **Bypass Stream:** Fixed conditions $p_{\text{bp}} = 0.45 \times 10^5$ Pa, $T_{\text{bp}} = 276$ K

### 6. Important Notes

- The script uses **CoolProp** for thermodynamic property calculations
- Pressure drop is defined as a **retention factor** $dP = 0.85$ (means 15% drop)

- The heat recovery calculation uses $c_p \cdot T$ differences (which equals enthalpy for perfect gases)
- The `calc_vjet` function uses the ambient pressure $p_0$ for back pressure
- Design points A and B are marked on contour plots for reference

### 7. File Dependencies

- Requires CoolProp library for property calculations
- Uses matplotlib for plotting
- Uses numpy for numerical operations
- Path manipulation to import heat exchanger modules (though not actively used in this script)
