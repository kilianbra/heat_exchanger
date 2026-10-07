# Straight Tube Bank Heat Exchanger Geometry

## Overview

The straight tube bank geometry (`tube_bank_straight.py`) defines a simple rectangular or annular tube bank configuration where tubes are arranged in a straight pattern. This geometry provides the foundation for 0D heat exchanger analysis and can represent either a **box** (rectangular) or **annular** (cylindrical) configuration.

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

### Bank Configuration

- `n_rows_per_pass`: Number of tube rows per pass
- `n_passes`: Number of passes (for multi-pass arrangements)
- `n_rows_total`: Total number of rows $n_{\text{rows},\text{total}}$, where:

$$
\begin{equation}
n_{\text{rows},\text{total}} = n_{\text{rows}/\text{pass}} \cdot n_{\text{passes}}
\end{equation}
$$

- `n_tubes_total`: Total number of tubes $n_{\text{tubes},\text{total}}$, where:

$$
\begin{equation}
n_{\text{tubes},\text{total}} = n_{\text{tubes}/\text{row}} \cdot n_{\text{rows},\text{total}}
\end{equation}
$$

### Geometry Type

- `annular_not_box`: Boolean flag
  - `True` = **Annular configuration**: Cylindrical geometry with inner and outer diameters
  - `False` = **Box configuration**: Rectangular cross-section

### Frontal Area

- `frontal_area_outer`: Frontal area for external flow [m²]
  - For annular: Cross-sectional area of annulus
  - For box: Width × Height

## Key Assumptions

### 1. **Straight Tube Arrangement**

- All tubes are **straight** (no curvature or spiral)
- Tubes run parallel to each other
- Uniform spacing in both transverse and longitudinal directions

### 2. **Annular Configuration** (`annular_not_box = True`)

- Geometry is cylindrical with:
  - **Inner diameter** $D_{\text{i}}$: Calculated from row width
  - **Outer diameter** $D_{\text{o}}$: Calculated from frontal area
  - **Row width**: $W = \pi D_{\text{i}} = X_t^* \cdot D_{\text{o}} \cdot n_{\text{tubes}/\text{row}}$
  - **Tube length**: Calculated from annulus geometry:

$$
\begin{equation}
L = \frac{D_{\text{i}}}{4}\left[\left(\frac{D_{\text{o}}}{D_{\text{i}}}\right)^2 - 1\right]
\end{equation}
$$

- **Passage height**: Annular gap height:

$$
\begin{equation}
H = \frac{D_{\text{o}} - D_{\text{i}}}{2}
\end{equation}
$$

### 3. **Box Configuration** (`annular_not_box = False`)

- Geometry is rectangular with:
  - **Row width**: $W = X_t^* \cdot D_{\text{o}} \cdot n_{\text{tubes}/\text{row}}$
  - **Tube length**: $L = \frac{A_{\text{frontal}}}{W}$
  - **Passage height**: Same as tube length (flow direction perpendicular to tubes)

### 4. **Free Area Ratio (σ)**

The free-area ratio accounts for tube blockage in external flow:

- **Inline layout**:

$$
\begin{equation}
\sigma = \frac{X_t^* - 1}{X_t^*}
\end{equation}
$$

- **Staggered layout**:

$$
\begin{equation}
\sigma = \min\left(\frac{X_t^* - 1}{X_t^*}, \frac{2(X_d^* - 1)}{X_t^*}\right)
\end{equation}
$$

where diagonal spacing: $X_d^* = \sqrt{X_l^{*2} + (X_t^*/2)^2}$

### 5. **Flow Areas**

- **External (hot) free flow area**:

$$
\begin{equation}
A_{\text{free},\text{outer}} = A_{\text{frontal}} \cdot \sigma
\end{equation}
$$

- **Internal (cold) free flow area**:

$$
\begin{equation}
A_{\text{free},\text{inner}} = \frac{\pi D_{\text{i}}^2}{4} \cdot n_{\text{tubes}/\text{pass}}
\end{equation}
$$

where $n_{\text{tubes}/\text{pass}} = n_{\text{tubes}/\text{row}} \cdot n_{\text{rows}/\text{pass}}$

### 6. **Heat Transfer Areas**

- **External heat transfer area per row**:

$$
\begin{equation}
A_{\text{HT},\text{outer}} = \pi D_{\text{o}} \cdot L \cdot n_{\text{tubes}/\text{row}}
\end{equation}
$$

- **Internal heat transfer area per row**:

$$
\begin{equation}
A_{\text{HT},\text{inner}} = \pi D_{\text{i}} \cdot L \cdot n_{\text{tubes}/\text{row}}
\end{equation}
$$

### 7. **Axial Length**

- Total axial length (flow direction):

$$
\begin{equation}
L_{\text{axial}} = X_l^* \cdot D_{\text{o}} \cdot n_{\text{rows}/\text{pass}} \cdot n_{\text{passes}}
\end{equation}
$$

### 8. **Multi-Pass Configuration**

- Geometry supports multiple passes
- Each pass has `n_rows_per_pass` rows
- Total rows = `n_rows_per_pass × n_passes`
- Useful for:
  - Increasing heat transfer area
  - Achieving counterflow arrangement
  - Managing pressure drop

## Calculated Properties

### Geometric Dimensions

- `row_width`: Width of one row of tubes
  - Annular: $\pi D_{\text{i}}$
  - Box: $X_t^* \cdot D_{\text{o}} \cdot n_{\text{tubes}/\text{row}}$
- `tube_length`: Length of tubes (perpendicular to flow)
- `passage_height`: Height of flow passage
  - Annular: $(D_{\text{o}} - D_{\text{i}})/2$
  - Box: Same as tube length
- `axial_length`: Total length in flow direction

### Flow Areas

- `area_free_flow_outer`: Free flow area for external fluid
- `area_free_flow_inner`: Free flow area for internal fluid (per pass)

### Heat Transfer Areas

- `area_heat_transfer_outer_per_row`: External heat transfer area per row
- `area_heat_transfer_inner_per_row`: Internal heat transfer area per row

### Tube Counts

- `n_tubes_per_pass`: Number of tubes per pass
- `n_tubes_total`: Total number of tubes in entire bank

## Key Limitations

1. **Geometry Definition Only**: This class provides **geometric properties only** - it does not include a solver
2. **Constant Properties**: Assumes uniform geometry throughout (no tapering or variation)
3. **Perfect Alignment**: Assumes tubes are perfectly aligned in rows
4. **Uniform Spacing**: Spacing ratios are constant throughout the bank
5. **No End Effects**: Does not account for entrance/exit effects
6. **Simplified Annulus**: Annular geometry uses simplified calculation (assumes thin annulus approximation in some formulas)

## Usage Notes

- This geometry class is typically used with:
  - 0D heat exchanger solvers
  - Tube bank correlation functions
  - Performance analysis tools
- The `annular_not_box` flag determines calculation method for tube length and passage height
- Spacing ratios (`tube_spacing_trv`, `tube_spacing_long`) are non-dimensional (normalized by tube diameter)
- All properties are cached for efficiency (computed once, reused)

## Comparison with Radial Spiral

| Feature            | Straight Tube Bank | Radial Spiral                  |
| ------------------ | ------------------ | ------------------------------ |
| Tube path          | Straight           | Involute spiral                |
| Flow pattern       | Crossflow          | Radial crossflow → counterflow |
| Geometry variation | Constant           | Varies with radius             |
| Solver included    | No                 | Yes (0D and 1D)                |
| Configuration      | Box or Annular     | Always annular                 |
| Complexity         | Simple             | Complex                        |

## Typical Applications

- **Box configuration**:
  - Rectangular heat exchangers
  - Simple crossflow arrangements
  - Easy to manufacture
- **Annular configuration**:
  - Cylindrical heat exchangers
  - Compact designs
  - Fits in annular spaces

## Design Considerations

1. **Spacing Ratios**:

   - Typical range: 1.25 - 2.5 for $X_t^*$ and $X_l^*$
   - Closer spacing → higher heat transfer but higher pressure drop
   - Staggered layout generally better than inline

2. **Number of Passes**:

   - Single pass: Simple, pure crossflow
   - Multiple passes: Can achieve counterflow, higher effectiveness

3. **Frontal Area**:

   - Determines overall size
   - For annular: Sets outer diameter
   - For box: Sets one dimension (width or height)

4. **Tube Count**:
   - More tubes → more heat transfer area
   - But also more complex manufacturing
   - Balance between performance and cost
