# Sweep Analysis: Dimensionalisations, Variables, and Take-off Mass Trade-off

This document explains the `sweep_Afr_and_Aq.py` analysis, including the meaning of variables, dimensionalisations, assumptions, and the mathematical framework for converting exergy/euergy creation into take-off mass changes.

---

## 1. Overview of the Sweep

The script sweeps over two geometric parameters:
- **A_q**: Total heat transfer area (m²), defined as the average of hot and cold side areas: `A_q = (A_h + A_c) / 2`
- **A_fr**: Frontal area of the heat exchanger (m²)

For each combination, it computes the heat exchanger performance using the `rate_hex_simple()` function, which returns:
- Effectiveness (ε)
- Pressure drops (Δp/p for hot and cold sides)
- **Normalised exergy change** (`dW_pot_Ex_norm`)
- **Normalised euergy change** (`dW_pot_Eu_norm`)

---

## 2. Key Variables and Their Meaning

### 2.1 The `xi` Variable

In the plotting code, `xi` is simply:

$$\texttt{xi} = A_q \quad \text{(the x-axis grid for interpolation)}$$

It spans from the minimum to maximum values of `A_q` in the sweep.

### 2.2 Dimensionalisation of the X-axis

The x-axis is converted from area to **mass per unit mass flow rate**:

$$\texttt{dimensionalisation\_x} = \frac{\rho_{\text{wall}} \cdot t}{\.{m}_{\text{hot}}}$$

Where:
- $\rho_{\text{wall}} \cdot t$ = `RHO_WALL_T` = wall density × wall thickness (kg/m² of heat transfer area)
- $\dot{m}_{\text{hot}}$ = hot-side mass flow rate (kg/s)

The total HEx mass (including packaging) per unit mass flow is:

$$\frac{m_{\text{HEx}}}{\dot{m}} = A_q \cdot \frac{\rho_{\text{wall}} \cdot t}{\dot{m}} \cdot (1 + \alpha_{\text{HEx}}) + m_{\text{fixed}}$$

Where:
- $\alpha_{\text{HEx}}$ = `ALPHA_HEX_KG` ≈ 0.7–0.9 (packaging mass fraction relative to matrix mass)
- $m_{\text{fixed}}$ = `KG_HEX_FIXED` (fixed mass per kg/s of air, e.g., headers, manifolds)

### 2.3 The Y-axis: $g^2$ (dimensionless mass flux squared)

When `PLOT_BOTH_PER_AQ = False`, the y-axis shows:

$$g^2 = \frac{G^2}{p_{\text{in}} \cdot \rho_{\text{in}}} = \frac{(\dot{m}/A_o)^2}{p_{\text{in}} \cdot \rho_{\text{in}}}$$

This is a key dimensionless parameter controlling:
- Pressure drop (scales with $g^2$)
- Heat transfer enhancement through flow velocity
- Trade-off between friction losses and heat transfer

---

## 3. What Is Plotted

### 3.1 Contour Plot Mode (`PLOT_BOTH_PER_AQ = False`)

**X-axis**: $A_q \cdot \frac{\rho_{\text{wall}} t}{\dot{m}}$ = HEx matrix mass per unit mass flow (kg/(kg/s))

**Y-axis**: $g^2$ = dimensionless inlet mass flux squared

**Contours**: Work potential creation normalised by $Q_{\max}$:
- **Euergy mode** (`PLOT_EUERGY_NOT_EXERGY = True`): $-\Delta \dot{W}_{\text{pot,Eu}} / Q_{\max}$
- **Exergy mode**: $-\Delta \dot{W}_{\text{pot,Ex}} / Q_{\max}$

**Overlaid lines**:
- Red dashed: Optimal $g^2$ for each HEx mass (maximises euergy)
- Blue solid: Optimal HEx mass for each $g^2$

### 3.2 1D Slice Mode (`PLOT_BOTH_PER_AQ = True`)

**X-axis**: Total HEx mass per unit mass flow = $m_{\text{HEx}} / \dot{m}$ (kg/(kg/s))

**Y-axis**: **Fuel mass avoided per unit HEx mass** (dimensionless ratio)

The y-value plotted is:

$$\frac{m_{\text{fuel,saved}}}{m_{\text{HEx}}} = \frac{\xi_{\text{Eu}} \cdot Q_{\max} \cdot \eta_{\text{turb}} / \eta_{\text{ov}} \cdot t_{\text{mission}} / \text{LHV}}{m_{\text{HEx}}}$$

Where $\xi_{\text{Eu}}$ is the normalised euergy creation.

---

## 4. Mathematical Definitions

### 4.1 Exergy Change (Entropy-based)

The normalised exergy change is the entropy generation scaled by the dead-state temperature:

$$\frac{\Delta \dot{W}_{\text{pot,Ex}}}{Q_{\max}} = \frac{T_d \Delta \dot{S}}{C_{\min}(T_{h,in} - T_{c,in})}$$

Explicitly:

$$\frac{\Delta \dot{W}_{\text{pot,Ex}}}{Q_{\max}} = \frac{C_h \ln(T_{h,o}/T_{h,i}) + C_c \ln(T_{c,o}/T_{c,i}) - C_h \frac{\gamma_h-1}{\gamma_h}\ln(p_{h,o}/p_{h,i}) - C_c \frac{\gamma_c-1}{\gamma_c}\ln(p_{c,o}/p_{c,i})}{C_{\min}(T_{h,in} - T_{c,in})}$$

### 4.2 Euergy Change (Isentropic Work Potential)

Euergy accounts for the actual work potential relative to a reference state $(T_d, p_d)$:

$$\Delta w_{\text{pot,Eu,hot}} = \left(\frac{p_d}{p_{h,in}}\right)^{(\gamma-1)/\gamma} T_{h,in} \left[\frac{T_{h,o}}{T_{h,in}} \left(\frac{p_{h,in}}{p_{h,o}}\right)^{(\gamma-1)/\gamma} - 1\right]$$

$$\Delta w_{\text{pot,Eu,cold}} = \left(\frac{p_d}{p_{c,in}}\right)^{(\gamma-1)/\gamma} T_{c,in} \left[\frac{T_{c,o}}{T_{c,in}} \left(\frac{p_{c,in}}{p_{c,o}}\right)^{(\gamma-1)/\gamma} - 1\right]$$

The total normalised euergy change:

$$\frac{\Delta \dot{W}_{\text{pot,Eu}}}{Q_{\max}} = \frac{C_h \Delta w_{\text{pot,Eu,hot}} + C_c \Delta w_{\text{pot,Eu,cold}}}{C_{\min}(T_{h,in} - T_{c,in})}$$

**Sign convention**: Negative values mean **work potential is destroyed** (bad). Positive values mean **work potential is created** (good, as in recuperation).

---

## 5. Assumptions for Fuel Savings

### 5.1 Core Assumption

The work potential created in the heat exchanger (euergy) represents additional power available at the turbine. This power:

1. **Replaces fuel that would otherwise be burned** to produce the same shaft power
2. **Is converted to useful work at the turbine efficiency** $\eta_{\text{turb}}$
3. **Scales inversely with the overall cycle efficiency** $\eta_{\text{ov}}$ because a more efficient baseline cycle requires more fuel per unit of shaft power deficit

### 5.2 Mathematical Framework

**Work potential rate created** (W):

$$\dot{W}_{\text{pot}} = -\xi_{\text{Eu}} \cdot Q_{\max}$$

where $\xi_{\text{Eu}} = -\texttt{dW\_pot\_Eu\_norm}$ is the positive euergy creation.

**Equivalent shaft power** (assuming turbine converts this work potential):

$$\dot{W}_{\text{shaft}} = \dot{W}_{\text{pot}} \cdot \eta_{\text{turb}}$$

**Fuel power avoided** (fuel that would have been needed to produce this shaft power in the baseline cycle):

$$\dot{Q}_{\text{fuel,avoided}} = \frac{\dot{W}_{\text{shaft}}}{\eta_{\text{ov}}} = \dot{W}_{\text{pot}} \cdot \frac{\eta_{\text{turb}}}{\eta_{\text{ov}}}$$

**Fuel mass saved over mission**:

$$m_{\text{fuel,saved}} = \frac{\dot{Q}_{\text{fuel,avoided}} \cdot t_{\text{mission}}}{\text{LHV}}$$

Substituting:

$$\boxed{m_{\text{fuel,saved}} = \xi_{\text{Eu}} \cdot Q_{\max} \cdot \frac{\eta_{\text{turb}}}{\eta_{\text{ov}}} \cdot \frac{t_{\text{mission}}}{\text{LHV}}}$$

### 5.3 Key Parameters in the Code

| Parameter | Heli Case | Brewer Case | Meaning |
|-----------|-----------|-------------|---------|
| `ETA_OV_OVER_ETA_TURB` | 0.2/0.8 = 0.25 | 0.363/0.88 ≈ 0.41 | $\eta_{\text{ov}}/\eta_{\text{turb}}$ |
| `FUEL_PER_HEAT` | 2 hrs / 12 kWh/kg | 10 hrs / 33.3 kWh/kg | $t_{\text{mission}}/\text{LHV}$ (kg/kW) |
| `LHV_KWH_PER_KG_FUEL` | 43.2/3.6 ≈ 12 | 120/3.6 ≈ 33.3 | Lower heating value (kWh/kg) |
| `MISSION_HOURS` | 2 | 10 | Flight duration |

---

## 6. Take-off Mass Change Analysis

### 6.1 The Trade-off

Adding a recuperator changes take-off mass through two competing effects:

1. **Mass increase**: The heat exchanger adds structural mass
2. **Mass decrease**: Less fuel is required for the mission

### 6.2 Equations for Take-off Mass Change

**HEx mass**:

$$m_{\text{HEx}} = A_q \cdot \rho_{\text{wall}} \cdot t \cdot (1 + \alpha_{\text{HEx}}) + m_{\text{fixed}} \cdot \dot{m}$$

**Fuel mass saved** (from Section 5):

$$m_{\text{fuel,saved}} = \xi_{\text{Eu}} \cdot Q_{\max} \cdot \frac{\eta_{\text{turb}}}{\eta_{\text{ov}}} \cdot \frac{t_{\text{mission}}}{\text{LHV}}$$

**Net change in take-off mass**:

$$\boxed{\Delta m_{\text{TO}} = m_{\text{HEx}} - m_{\text{fuel,saved}}}$$

If $\Delta m_{\text{TO}} < 0$, the recuperator provides a **net benefit**.

### 6.3 Break-even Condition

The break-even occurs when fuel saved equals HEx mass:

$$\frac{m_{\text{fuel,saved}}}{m_{\text{HEx}}} = 1$$

The y-axis in `PLOT_BOTH_PER_AQ = True` mode shows exactly this ratio. The **green horizontal line at y = 1** represents break-even.

### 6.4 Dimensionless Form

Normalising by $\dot{m}$ (mass flow rate):

$$\frac{\Delta m_{\text{TO}}}{\dot{m}} = \underbrace{\frac{m_{\text{HEx}}}{\dot{m}}}_{\text{x-axis}} - \underbrace{\frac{\xi_{\text{Eu}} \cdot Q_{\max}}{\dot{m}} \cdot \frac{\eta_{\text{turb}}}{\eta_{\text{ov}}} \cdot \frac{t_{\text{mission}}}{\text{LHV}}}_{\text{fuel saved per unit flow}}$$

Or, rearranging to show fuel saved **per kg of HEx** (the y-axis):

$$\frac{m_{\text{fuel,saved}}}{m_{\text{HEx}}} = \frac{\xi_{\text{Eu}} \cdot Q_{\max}}{m_{\text{HEx}}} \cdot \frac{\eta_{\text{turb}}}{\eta_{\text{ov}}} \cdot \frac{t_{\text{mission}}}{\text{LHV}}$$

### 6.5 Complete Take-off Mass Model

For a full take-off mass analysis, you would compute:

$$\Delta m_{\text{TO}} = m_{\text{HEx}}(A_q, A_{fr}) - m_{\text{fuel,saved}}(\xi_{\text{Eu}}(A_q, A_{fr}))$$

The optimal design **minimises** $\Delta m_{\text{TO}}$ (or maximises the negative value, i.e., the mass saved).

---

## 7. Summary of Plot Interpretation

### For `PLOT_BOTH_PER_AQ = True` (Figure 7 style):

| Feature | Interpretation |
|---------|---------------|
| **Y > 1** | Fuel saved exceeds HEx mass → **net TO mass reduction** |
| **Y < 1** | HEx mass exceeds fuel saved → **net TO mass increase** |
| **Y = 1 (green line)** | Break-even point |
| **Optimal point** | Peak of the curve: best fuel-to-HEx mass ratio |
| **X-axis** | Specific HEx mass: how much HEx per kg/s of air |

### For `PLOT_BOTH_PER_AQ = False` (Figure 6 style):

| Feature | Interpretation |
|---------|---------------|
| **Contour value** | Work potential creation as fraction of $Q_{\max}$ |
| **Higher contours** | More beneficial recuperation |
| **Red line** | Optimal mass flux for given HEx size |
| **Feasible region** | Below pressure drop limits |

---

## 8. Proposed Take-off Mass Plot

To directly visualise take-off mass change, one could plot:

**X-axis**: $m_{\text{HEx}} / \dot{m}$ (kg/(kg/s))

**Y-axis**: $\Delta m_{\text{TO}} / \dot{m}$ = $m_{\text{HEx}}/\dot{m} - m_{\text{fuel,saved}}/\dot{m}$ (kg/(kg/s))

Using the equations above:

$$\frac{\Delta m_{\text{TO}}}{\dot{m}} = \frac{m_{\text{HEx}}}{\dot{m}} \left(1 - \frac{m_{\text{fuel,saved}}}{m_{\text{HEx}}}\right)$$

Since the current plot shows $m_{\text{fuel,saved}}/m_{\text{HEx}}$ on the y-axis:

$$\frac{\Delta m_{\text{TO}}}{\dot{m}} = x \cdot (1 - y_{\text{current}})$$

Where:
- $x$ = x-axis value (specific HEx mass)
- $y_{\text{current}}$ = current y-axis value (fuel saved per HEx mass)

**Regions**:
- $\Delta m_{\text{TO}} / \dot{m} < 0$ → Net mass saved (beneficial)
- $\Delta m_{\text{TO}} / \dot{m} > 0$ → Net mass added (detrimental)

The **optimal design** would be the point that **minimises** $\Delta m_{\text{TO}} / \dot{m}$.

---

## 9. Caveats and Limitations

1. **Fixed cycle efficiency**: The analysis assumes $\eta_{\text{ov}}$ is constant, but adding a recuperator changes the cycle efficiency
2. **Linear mission model**: Fuel burn is constant over the mission (no Breguet range effects)
3. **Static conditions**: Does not account for varying flight conditions
4. **Perfect conversion**: Assumes all euergy creation converts to shaft power at $\eta_{\text{turb}}$
5. **No installation effects**: Pressure drops in ducts, additional cooling, etc. not included
6. **Packaging estimates**: The $\alpha_{\text{HEx}}$ and fixed mass terms are approximate
