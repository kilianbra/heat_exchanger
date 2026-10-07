# Radial Spiral Heat Exchanger Geometry

## Overview

The radial spiral heat exchanger (`radial_spiral.py`) implements an involute spiral geometry where a tube bank is wrapped in a spiral pattern. The external (hot) fluid flows radially while the internal (cold) fluid flows through the tubes. This configuration produces **local crossflow** but an **overall counterflow** arrangement, which is advantageous for heat transfer effectiveness.

## Key Geometric Parameters

### Tube Geometry

- `tube_outer_diam`: Outer diameter of tubes $d_{\text{o}}$ [m]
- `tube_thick`: Tube wall thickness $t$ [m]
- `tube_inner_diam`: Inner diameter $d_{\text{i}}$ [m], where:

$$
\begin{equation}
d_{\text{i}} = d_{\text{o}} - 2t
\end{equation}
$$

### Tube Bank Layout

- `tube_spacing_trv`: Non-dimensional transverse spacing ratio (Xt\*) - spacing perpendicular to flow direction
- `tube_spacing_long`: Non-dimensional longitudinal spacing ratio (Xl\*) - spacing along flow direction
- `staggered`: Boolean flag - `True` for staggered layout, `False` for inline layout
- `n_tubes_per_row`: Number of tubes in each row

### Spiral Configuration

- `n_headers`: Number of radial sectors/headers (discretization for 1D marching)
- `n_rows_per_header`: Number of tube rows per header/sector
- `radius_outer_hex`: Outer radius of heat exchanger [m]
- `radius_inner_hex`: Calculated inner radius [m] - depends on number of rows and spacing
- `inv_angle_deg`: Involute sweep angle in degrees (default: 360° for full spiral)

### Flow Orientation

- `ext_fluid_flows_radially_inwards`: Boolean flag
  - `True` = **Inboard configuration**: External hot fluid flows from outer to inner radius
  - `False` = **Outboard configuration**: External hot fluid flows from inner to outer radius
- Cold fluid always flows in the opposite direction to maintain counterflow

### Material Properties

- `wall_conductivity`: Thermal conductivity of tube wall [W/(m·K)] (default: 14.0 W/(m·K) for 304 stainless steel)
- `wall_density`: Density of tube wall material [kg/m³] (default: 7930 kg/m³ for 304 stainless steel)

## Key Assumptions

### 1. **Involute Spiral Geometry**

- Tubes follow an **Archimedean spiral** path: $r = r_0 + b\theta$

- The spiral parameter $b$ relates the radial change to the angular sweep:

$$
\begin{equation}
b = \frac{r_{\text{outer}} - r_{\text{inner}}}{\theta_{\text{total}}}
\end{equation}
$$

where $\theta_{\text{total}}$ is the total involute sweep angle in radians.

- Spiral arc length uses exact analytical formula:

$$
\begin{equation}
L = \frac{b}{2}\left[\theta\sqrt{1+\theta^2} + \ln(\theta + \sqrt{1+\theta^{2}})\right]_{\theta_1}^{\theta_2}
\end{equation}
$$

### 2. **Flow Configuration**

- **Hot side (external)**: Crossflow over tube bank, flowing radially
- **Cold side (internal)**: Flow through tubes, following spiral path
- **Overall arrangement**: Counterflow (hot and cold flow in opposite directions)
- **Local arrangement**: Crossflow at each radial position

### 3. **Tube Bank Correlations**

- External flow uses **tube bank correlations** for Nusselt number and friction factor
- Correlations depend on:
  - Reynolds number based on outer diameter: $Re_{\text{OD}} = G_{\text{h}} \cdot d_{\text{o}} / \mu_{\text{h}}$
  - Spacing ratios (`tube_spacing_long`, `tube_spacing_trv`)
  - Staggered vs inline layout
  - Total number of rows
- Internal flow uses **circular pipe correlations** for Nusselt number and friction factor

### 4. **Free Area Ratio (σ)**

The free-area ratio for external flow accounts for tube blockage:

- **Inline layout**: $\sigma = \frac{X_t^* - 1}{X_t^*}$
- **Staggered layout**: $\sigma = \min\left(\frac{X_t^* - 1}{X_t^*}, \frac{2(X_d^* - 1)}{X_t^*}\right)$
  where $X_d^* = \sqrt{(X_t^*/2)^2 + X_l^{*2}}$ is the diagonal spacing

### 5. **1D Discretization**

- Heat exchanger is divided into `n_headers` radial sectors
- Each sector contains `n_rows_per_header` rows of tubes
- Arrays are oriented so index 0 corresponds to cold inlet
- For inboard: cold inlet at inner radius (index 0)
- For outboard: cold inlet at outer radius (arrays reversed)

### 6. **Heat Transfer Assumptions**

- **Overall heat transfer coefficient** accounts for:
  - External convection: $h_{\text{h}}$ (from tube bank correlation)
  - Internal convection: $h_{\text{c}}$ (from circular pipe correlation)
  - Wall conduction: $\frac{d_{\text{o}}}{2k_w}\ln(d_{\text{o}}/d_{\text{i}})$

$$
\begin{equation}
\frac{1}{U} = \frac{1}{h_{\text{h}}} + \frac{1}{h_{\text{c}}}\frac{d_{\text{o}}}{d_{\text{i}}} + \frac{d_{\text{o}}}{2k_w}\ln\frac{d_{\text{o}}}{d_{\text{i}}}
\end{equation}
$$

### 7. **0D Initial Guess**

- Assumes **counterflow** configuration for initial estimate
- Uses epsilon-NTU method with:
  - Properties evaluated at inlet conditions (first pass)
  - Properties re-evaluated at mean conditions (second pass for refinement)
- Typically accurate to within a few percentage points

### 8. **1D Marching Solver**

- Uses **shooting method** to solve boundary value problem
- Marches from cold inlet (inner or outer radius) toward hot inlet
- At each header:
  - Calculates local effectiveness using crossflow epsilon-NTU
  - Updates fluid states accounting for:
    - Heat transfer: $\Delta h = \pm q / \dot{m}$
    - Pressure drop:

$$
\begin{equation}
\Delta P = f \cdot \frac{A_{\text{HT}}}{A_{\text{free}}} \cdot \frac{G^2}{2\rho}
\end{equation}
$$

- Solves for unknown boundary conditions (typically $T_{\text{h},\text{out}}$ and optionally $P_{\text{h},\text{out}}$)

### 9. **Mass Flux Calculation**

$$
\begin{equation}
G_{\text{h}} = \frac{\dot{m}_{\text{h}} / n_{\text{headers}}}{A_{\text{free},\text{hot}}}
\end{equation}
$$

$$
\begin{equation}
G_{\text{c}} = \frac{\dot{m}_{\text{c}} / n_{\text{headers}}}{A_{\text{free},\text{cold}}}
\end{equation}
$$

- Free areas vary with radius (hot side) but constant for cold side (tube cross-section)

### 10. **Geometric Constraints**

$$
\begin{equation}
r_{\text{inner}} = r_{\text{outer}} - n_{\text{rows}} \cdot X_l^* \cdot d_{\text{o}}
\end{equation}
$$

- Validation checks ensure:
  - $r_{\text{inner}} > 0$ (positive annulus width)
  - $r_{\text{inner}} < r_{\text{outer}}$ (valid geometry)

## Calculated Properties

### Geometric Properties

- `axial_length`: Total axial length = $n_{\text{tubes}/\text{row}} \cdot X_t^* \cdot d_{\text{o}}$
- `spiral_length`: Total spiral arc length (analytical formula)
- `frontal_area_outer`: Frontal area at outer radius = $2\pi r_{\text{outer}} \cdot L_{\text{axial}} \cdot X_t^*$
- `area_heat_transfer_outer_total`: Total external heat transfer area
- `area_heat_transfer_inner_total`: Total internal heat transfer area
- `n_tubes_total`: Total number of tubes = $n_{\text{tubes}/\text{row}} \cdot n_{\text{rows}}$

### Flow Properties (per sector)

- `area_frontal_hot`: Frontal area for hot flow at each radial position
- `area_free_hot`: Free flow area for hot side (accounts for tube blockage)
- `area_free_cold`: Free flow area for cold side (tube cross-sections)
- `d_h_hot`: Hydraulic diameter for hot side flow ($d_{\text{h},\text{hot}}$)
- `tube_length`: Spiral arc length per header

## Solver Methods

### 0D Method (`method="0d"`)

- Fast initial estimate
- Assumes counterflow configuration
- Two-step refinement using mean properties
- Returns outlet temperatures and pressures

### 1D Method (`method="1d"`)

- Full marching solution
- Accounts for radial variation in geometry and flow properties
- Uses shooting method with root finding
- More accurate but computationally expensive
- Tolerances:
  - Temperature: 1e-2 K
  - Pressure: 0.1% of inlet pressure

## Key Limitations

1. **Perfect Gas Assumption**: Uses perfect gas model for fluid properties (constant cp, ideal gas law)
2. **Constant Properties**: Properties evaluated locally but assumed constant within each header
3. **Crossflow Approximation**: Local heat transfer uses crossflow epsilon-NTU, though overall is counterflow
4. **No Axial Variation**: Assumes uniform conditions in axial direction (perpendicular to radial flow)
5. **Tube Bank Correlation Validity**: Correlations valid for specific Reynolds number ranges and spacing ratios
6. **No Bypass Flow**: Assumes all flow passes through heat exchanger (no leakage)

## Usage Notes

- The geometry protocol allows for flexible implementation
- `RadialSpiralSpec` provides concrete dataclass implementation
- Arrays from `_1d_arrays_for_one_sector()` are oriented for marching from cold inlet
- For outboard configuration, arrays are automatically reversed
- Diagnostics include non-dimensional groups (Re, St, Ec) at inlet and outlet

## References

- Archimedean spiral arc length formula from mathematical literature
- Tube bank correlations from standard heat transfer textbooks
- Circular pipe correlations (Dittus-Boelter for Nu, standard friction factor)
