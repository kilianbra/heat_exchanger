"""
Script to generate publication-quality figure from hex_Afr_isolation.py
Case B: Air and combustion products heat exchanger
Black and white, no sliders, saves as SVG

Self-contained version - all functions included for portability.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, FuncFormatter


# ============================================================================
# CORRELATION FUNCTIONS (from correlations.py)
# ============================================================================
def general_hex_j_factor(Re: float, l_s_over_d_h: float, show_warnings: bool = False) -> float:
    """
    Calculate j-factor for general heat exchangers.
    From Milten (2024) eqn (15), based on HEx from Kays and London (1984) like LaHaye (1974).
    """
    import warnings

    if show_warnings and (Re < 2e3 or Re > 2e4):
        warnings.warn(f"Reynolds number {Re:.1e} outside correlation range of 2k-20k")

    if show_warnings and (l_s_over_d_h < 0.645 or l_s_over_d_h > 73.8):
        warnings.warn(f"l_s_over_d_h ratio {l_s_over_d_h:.2f} outside correlation range of 0.645-73.8")

    return 0.360 * l_s_over_d_h**-0.401 * Re**-0.413 + 2.13e-5 * l_s_over_d_h


def general_hex_friction_factor(Re: float, l_s_over_d_h: float, show_warnings: bool = False) -> float:
    """
    Calculate friction factor for general heat exchangers.
    From Milten (2024) eqn (16), based on HEx from Kays and London (1984) like LaHaye (1974).
    """
    import warnings

    if show_warnings and (Re < 2e3 or Re > 2e4):
        warnings.warn(f"Reynolds number {Re:.1e} outside correlation range of 2k-20k")

    if show_warnings and (l_s_over_d_h < 0.645 or l_s_over_d_h > 73.8):
        warnings.warn(f"l_s_over_d_h ratio {l_s_over_d_h:.2f} outside correlation range of 0.645-73.8")

    return 0.492 * l_s_over_d_h**-0.501 * Re**-0.232


# ============================================================================
# HEAT EXCHANGER FUNCTIONS (from use_cases_shared/hex_in_isolation.py)
# ============================================================================
def calculate_temperature_ratio(eps, t, hot_fluid=True, C_h_c=1.0):
    """
    Calculate the outlet/inlet temperature ratio for a given effectiveness and temperature ratio.

    Parameters:
    -----------
    eps : float or array
        Heat exchanger effectiveness
    t : float
        Temperature ratio T_h_in/T_c_in
    hot_fluid : bool
        Whether this is for the hot fluid (True) or cold fluid (False)
    C_h_c : float
        Capacity ratio C_hot/C_cold (default 1.0 for balanced exchanger)

    Returns:
    --------
    float or array
        Temperature ratio T_out/T_in

    Notes:
    ------
    For a counterflow heat exchanger:
    if C_hot > C_cold (Cr = 1/C_h_c < 1):
    - Cold side is C_min: Q = eps * C_cold * (T_h_in - T_c_in)
    - Hot side temp change: dT_hot = Q / C_hot = eps * (T_h_in - T_c_in) / C_h_c
    - Cold side temp change: dT_cold = Q / C_cold = eps * (T_h_in - T_c_in)
    if C_hot < C_cold (Cr = C_h_c < 1):
    - Hot side is C_min: Q = eps * C_hot * (T_h_in - T_c_in)
    - Cold side temp change: dT_cold = Q / C_cold = eps * (T_h_in - T_c_in) * C_h_c
    - Hot side temp change: dT_hot = Q / C_hot = eps * (T_h_in - T_c_in)

    Then convert dT to T_out/T_in
    for hot_fluid
    - T_out/T_in = 1 - dT_hot/T_h_in
    for cold_fluid
    - T_out/T_in = 1 + dT_cold/T_c_in
    """
    if hot_fluid:
        if C_h_c > 1:  # cold side is C_min
            return 1 - eps / C_h_c * (1 - 1 / t)
        else:
            return 1 - eps * (1 - 1 / t)
    else:  # cold fluid
        if C_h_c > 1:  # hot side is C_min
            return 1 + eps * (t - 1)
        else:
            return 1 + eps * C_h_c * (t - 1)


def calculate_pressure_ratio(L_dh, f, gd2, T_i_Td, p_i_pd, eps, t, hot_fluid=True, C_h_c=1.0, max_iter=100, tol=0.0001):
    """
    Calculate the pressure ratio p_out/p_in using an iterative approach that accounts for
    both friction losses and density changes due to heating.

    This uses g_d^2 = G^2/rho_d/p_d

    Parameters:
    -----------
    L_dh : float or array
        Length to hydraulic diameter ratio
    f : float or array
        Friction factor
    gd2 : float or array
        Square of the mass flux parameter (g_d^2 = G^2/rho_d/p_d)
    T_i_Td : float
        Inlet temperature ratio (T_in/T_d)
    p_i_pd : float
        Inlet pressure ratio (p_in/p_d)
    eps : float or array
        Heat exchanger effectiveness
    t : float
        Temperature ratio T_h_in/T_c_in
    hot_fluid : bool
        Whether this is for the hot fluid (True) or cold fluid (False)
    Cr : float
        Capacity ratio C_min/C_max (default 1.0)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Convergence tolerance (default: 0.0001)

    Returns:
    --------
    float or array
        Pressure ratio p_out/p_in
    """
    # Initialize pressure ratio (start with no pressure drop)
    p_o_pi = np.ones_like(L_dh)

    # Calculate temperature ratio for a given effectiveness
    T_o_Ti = calculate_temperature_ratio(eps, t, hot_fluid, C_h_c)

    # Iterate to find converged pressure ratio
    for _ in range(max_iter):
        # Store old value for convergence check
        p_o_pi_old = p_o_pi.copy()

        # Calculate new pressure ratio using the iteration formula
        p_o_pi_new = 1 - gd2 * (1 / p_i_pd) ** 2 * (T_i_Td) * (
            0.5 * f * 4 * L_dh * (1 + T_o_Ti * 1 / p_o_pi) / 2 + (T_o_Ti * 1 / p_o_pi - 1)
        )

        # Apply relaxation for stability
        p_o_pi = 0.5 * p_o_pi_old + 0.5 * p_o_pi_new

        # Check convergence
        if np.all(np.abs(p_o_pi - p_o_pi_old) < tol):
            break

    return p_o_pi


# ============================================================================
# CASE B PARAMETERS
# ============================================================================
# Fluid properties
mdot = 1.6  # kg/s for both fluids
cp_hot = 1150  # J/(kg·K) for combustion products
cp_cold = 1040  # J/(kg·K) for air

# Inlet conditions
T_hot_in = 980  # K
P_hot_in = 1.06e5  # Pa (1.06 bar)
T_cold_in = 576  # K
P_cold_in = 7.2e5  # Pa (7.2 bar)

# Reference conditions (for non-dimensionalization)
Td = 300  # K
pd = 1e5  # Pa

# Geometry
t_over_dh = 0.02  # t/d_h = 0.02 (85 micron t over 4 mm walls)
sigma_r = 2.0  # Ratio of free flow areas (Ao_h/Ao_c)
A_fr_over_Ao_c = (1 + sigma_r) * (1 + 2 * t_over_dh)  # Ratio of frontal area to cold side free flow area (A_fr/Ao_c)
d_h = 4e-3  # m, hydraulic diameter for both sides

# Start from 0.05 m² and go down

Aq_baseline = 43.0  # m², baseline heat transfer area

# Calculate temperature ratios
Th_in_Td = T_hot_in / Td
Tc_in_Td = T_cold_in / Td
Th_in_Tc_in = T_hot_in / T_cold_in

# Pressure ratios
ph_in_pd = P_hot_in / pd
pc_in_pd = P_cold_in / pd

# Geometry parameters
Ao_c_over_Ao_h = 1.0 / sigma_r  # Ao_c/Ao_h = 1 / (Ao_h/Ao_c)
ls_over_dh_hot = 5.0  # Will be calculated from geometry
ls_over_dh_cold = 5.0  # Will be calculated from geometry

# Gamma values for work potential calculations
gamma_hot = 1.33  # For combustion products
gamma_cold = 1.4  # For air
gm1og_hot = (gamma_hot - 1) / gamma_hot
gm1og_cold = (gamma_cold - 1) / gamma_cold

# Capacity ratio Cr = C_min / C_max
C_hot = mdot * cp_hot
C_cold = mdot * cp_cold
C_min = min(C_hot, C_cold)
C_max = max(C_hot, C_cold)
Cr = C_min / C_max  # Cr < 1 since cp_cold < cp_hot (cold side is C_min)
C_h_c = C_hot / C_cold

# Plotting parameters
TICK_LABEL_SIZE = 12
AXIS_LABEL_SIZE = 14
LEGEND_SIZE = 12
FLIP_Y_AXIS = True
max_dp = 0.06  # Maximum pressure drop (6%)

# ============================================================================
# FLUID PROPERTIES CALCULATION
# ============================================================================
# Use perfect gas model for simplicity (can be changed to CoolProp if needed)
# For air: molecular weight ~28.97 kg/kmol, gamma ~1.4
# For combustion products: approximate as air with higher cp
R_air = 287  # J/(kg·K)
gamma_air = 1.4

# Calculate density and viscosity at inlet conditions
# Using ideal gas law and Sutherland's law approximation
rho_hot_in = P_hot_in / (R_air * T_hot_in)  # Approximate as air
rho_cold_in = P_cold_in / (R_air * T_cold_in)

# Viscosity using Sutherland's law (approximate)
mu_ref = 1.8e-5  # Pa·s at 300 K
T_ref = 300  # K
S = 110.4  # K (Sutherland's constant)
mu_hot_in = mu_ref * ((T_ref + S) / (T_hot_in + S)) * ((T_hot_in / T_ref) ** 1.5)
mu_cold_in = mu_ref * ((T_ref + S) / (T_cold_in + S)) * ((T_cold_in / T_ref) ** 1.5)

print("=" * 80)
print("CASE B: Air and Combustion Products Heat Exchanger")
print("=" * 80)
print("Hot fluid (combustion products):")
print(f"  T_in = {T_hot_in} K, P_in = {P_hot_in / 1e5:.2f} bar")
print(f"  rho = {rho_hot_in:.3f} kg/m³, mu = {mu_hot_in:.2e} Pa·s, cp = {cp_hot} J/(kg·K)")
print("Cold fluid (air):")
print(f"  T_in = {T_cold_in} K, P_in = {P_cold_in / 1e5:.2f} bar")
print(f"  rho = {rho_cold_in:.3f} kg/m³, mu = {mu_cold_in:.2e} Pa·s, cp = {cp_cold} J/(kg·K)")
print(f"Mass flow rate: {mdot} kg/s (both sides)")
print(
    f"Geometry: d_h = {d_h * 1000:.1f} mm, sigma_r = {sigma_r} (Ao_h/Ao_c), "
    f"A_fr/Ao_c = {A_fr_over_Ao_c}, Aq_baseline = {Aq_baseline} m²"
)
print(f"Capacity ratio: Cr = C_min/C_max = {Cr:.4f} (cold side is C_min)")
print("=" * 80)

# ============================================================================
# FRONTAL AREA SWEEP TO FIND BASELINE
# ============================================================================
# A_fr_over_Ao_c is the ratio of frontal area to cold side free flow area
# sigma_r = Ao_h/Ao_c is the ratio of hot to cold free flow areas

# Start from 0.05 m² and go down
A_fr_sweep = np.linspace(0.03, 0.9, 1000)  # 1000 points from 0.05 to 0.001 m²

# Calculate properties for each frontal area
print("\nFrontal area sweep (filtering out dp_hot < 0% or dp_hot > 20%):")
header = (
    f"{'A_fr (m²)':<12} {'eps':<8} {'dp_hot (%)':<12} {'dp_cold (%)':<12} "
    f"{'Re_hot':<12} {'Re_cold':<12} {'g²_hot':<12} {'g²_cold':<12}"
)
print(header)
print("-" * 100)

results = []
dp_hot_max = 20.0  # Maximum allowed pressure drop (20%)

for A_fr in A_fr_sweep:
    # Calculate free flow areas
    # A_fr_over_Ao_c = A_fr/Ao_c, so Ao_c = A_fr / A_fr_over_Ao_c
    Ao_c = A_fr / A_fr_over_Ao_c
    # sigma_r = Ao_h/Ao_c, so Ao_h = Ao_c * sigma_r
    Ao_h = Ao_c * sigma_r

    # Calculate mass flux G = mdot / Ao
    G_hot = mdot / Ao_h
    G_cold = mdot / Ao_c

    # Calculate Reynolds numbers: Re = G * d_h / mu
    Re_hot = G_hot * d_h / mu_hot_in
    Re_cold = G_cold * d_h / mu_cold_in

    # Calculate g^2 = (mdot/Ao)^2 / 4 / p_in / rho_in
    g2_hot = (mdot / Ao_h) ** 2 / 4 / P_hot_in / rho_hot_in
    g2_cold = (mdot / Ao_c) ** 2 / 4 / P_cold_in / rho_cold_in

    # Calculate friction factors
    f_hot = general_hex_friction_factor(Re_hot, ls_over_dh_hot)
    f_cold = general_hex_friction_factor(Re_cold, ls_over_dh_cold)

    # Calculate Stanton numbers
    Pr = 0.7  # Prandtl number
    j_hot = general_hex_j_factor(Re_hot, ls_over_dh_hot)
    j_cold = general_hex_j_factor(Re_cold, ls_over_dh_cold)
    St_hot = j_hot * Pr ** (-2 / 3)
    St_cold = j_cold * Pr ** (-2 / 3)

    # Calculate L/dh from Aq and Ao
    # From d_h = 4 * Ao * L / A, we get L/d_h = A / (4 * Ao)
    # For the hot side: L/d_h = A_h / (4 * Ao_h)
    # Since d_h is the same for both sides, A_h/Ao_h = A_c/Ao_c, meaning A_h/A_c = sigma_r
    # So A_h = Aq_baseline represents the hot-side heat transfer area
    L_dh_hot = Aq_baseline / (4 * Ao_h)
    # For cold side: same L and d_h, so L/d_h is identical
    # (A_c = Aq_baseline / sigma_r, Ao_c = Ao_h / sigma_r, so A_c/(4*Ao_c) = L/d_h)
    L_dh_cold = L_dh_hot  # Same physical flow length and hydraulic diameter

    # Calculate NTU and effectiveness
    # NTU = UA / C_min, where UA = 1 / (1/(h_hot*A_hot) + 1/(h_cold*A_cold))
    # h = St * G * cp, and A_hot = Aq/2, A_cold = Aq/2 (for balanced areas)
    # Simplified: NTU = 1 / ((1/St_hot + Ao_c/Ao_h/St_cold) / (Aq/Ao_h))
    Aq_over_Ao_h = Aq_baseline / Ao_h
    NTU = 1 / ((1 / St_hot + Ao_c_over_Ao_h / St_cold) / Aq_over_Ao_h)

    if Cr < 0.99:  # If significantly unbalanced
        eps = (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr)))
    else:
        eps = NTU / (1 + NTU)

    # Calculate pressure ratios
    t = Th_in_Tc_in
    p_hot_out_p_hot_in = calculate_pressure_ratio(
        L_dh_hot, f_hot, g2_hot, T_i_Td=Th_in_Td, p_i_pd=ph_in_pd, eps=eps, t=t, hot_fluid=True, C_h_c=C_h_c
    )
    p_cold_out_p_cold_in = calculate_pressure_ratio(
        L_dh_cold, f_cold, g2_cold, T_i_Td=Tc_in_Td, p_i_pd=pc_in_pd, eps=eps, t=t, hot_fluid=False, C_h_c=C_h_c
    )

    dp_hot = (1 - p_hot_out_p_hot_in) * 100
    dp_cold = (1 - p_cold_out_p_cold_in) * 100

    # Skip this A_fr if dp_hot is negative or exceeds 20%
    if dp_hot < 0 or dp_hot > dp_hot_max:
        continue  # Move to next A_fr value

    results.append(
        {
            "A_fr": A_fr,
            "Ao_h": Ao_h,
            "eps": eps,
            "dp_hot": dp_hot,
            "dp_cold": dp_cold,
            "Re_hot": Re_hot,
            "Re_cold": Re_cold,
            "g2_hot": g2_hot,
            "g2_cold": g2_cold,
        }
    )

# Check if any valid results were found
if len(results) == 0:
    print("\n" + "=" * 80)
    print("WARNING: No valid A_fr values found!")
    print(f"All frontal areas in the sweep had dp_hot < 0% or dp_hot > {dp_hot_max}%")
    print("Consider adjusting the sweep range or geometry parameters.")
    print("=" * 80)
    raise ValueError("No valid results found - all A_fr values had invalid pressure drops")

# Select 10 evenly spaced values from the valid results
n_values = min(10, len(results))
indices = np.linspace(0, len(results) - 1, n_values, dtype=int)
selected_results = [results[i] for i in indices]

for r in selected_results:
    print(
        f"{r['A_fr']:<12.6f} {r['eps']:<8.4f} {r['dp_hot']:<12.4f} {r['dp_cold']:<12.4f} "
        f"{r['Re_hot']:<12.1f} {r['Re_cold']:<12.1f} {r['g2_hot']:<12.2e} {r['g2_cold']:<12.2e}"
    )

# Find the one closest to 60% effectiveness
target_eps = 0.60
closest_idx = np.argmin([abs(r["eps"] - target_eps) for r in results])
baseline_result = results[closest_idx]

print("\n" + "=" * 80)
print(f"Baseline selected (closest to {target_eps * 100:.0f}% effectiveness):")
print(f"  A_fr_baseline = {baseline_result['A_fr']:.6f} m²")
print(f"  Ao_h_baseline = {baseline_result['Ao_h']:.6f} m²")
print(f"  Effectiveness = {baseline_result['eps']:.4f} ({baseline_result['eps'] * 100:.2f}%)")
print(f"  Pressure drop (hot) = {baseline_result['dp_hot']:.4f}%")
print(f"  Pressure drop (cold) = {baseline_result['dp_cold']:.4f}%")
print(f"  Re_hot = {baseline_result['Re_hot']:.1f}")
print(f"  Re_cold = {baseline_result['Re_cold']:.1f}")
print(f"  g²_hot = {baseline_result['g2_hot']:.2e}")
print(f"  g²_cold = {baseline_result['g2_cold']:.2e}")
print("=" * 80)

# Set baseline values
A_fr_baseline = baseline_result["A_fr"]
Ao_h_baseline = baseline_result["Ao_h"]
Re_h_baseline = baseline_result["Re_hot"]
Re_c_baseline = baseline_result["Re_cold"]
g2_hot_baseline = baseline_result["g2_hot"]
g2_cold_baseline = baseline_result["g2_cold"]

# Calculate Aq_over_Aoh_baseline for the baseline
Aq_over_Aoh_baseline = Aq_baseline / Ao_h_baseline

A_fr_min = 0.2  # Minimum A_fr/A_fr_baseline

# ============================================================================
# CALCULATIONS FOR PLOT
# ============================================================================

# Create array of A_fr_over_A_fr_baseline values
A_fr_over_A_fr_baseline = np.linspace(A_fr_min, 2.0, 1000)

# Calculate work potentials with modified function that uses different gamma values
# We'll calculate most things the same way, but use different gamma for hot and cold
Re_h = Re_h_baseline / A_fr_over_A_fr_baseline
Re_c = Re_c_baseline / A_fr_over_A_fr_baseline

g2_hot = g2_hot_baseline / A_fr_over_A_fr_baseline**2
g2_cold = g2_cold_baseline / A_fr_over_A_fr_baseline**2

f_hot = general_hex_friction_factor(Re_h, ls_over_dh_hot)
f_cold = general_hex_friction_factor(Re_c, ls_over_dh_cold)

Pr = 0.7
St_hot = general_hex_j_factor(Re_h, ls_over_dh_hot) * Pr ** (-2 / 3)
St_cold = general_hex_j_factor(Re_c, ls_over_dh_cold) * Pr ** (-2 / 3)

L_dh_hot_array = Aq_over_Aoh_baseline / A_fr_over_A_fr_baseline / 4

# Calculate NTU and effectiveness
NTU = 1 / ((1 / St_hot + Ao_c_over_Ao_h / St_cold) / (Aq_over_Aoh_baseline / A_fr_over_A_fr_baseline))
eps = NTU / (1 + NTU)

# Calculate temperature ratios
t = Th_in_Td / Tc_in_Td
T_hot_out_T_hot_in = calculate_temperature_ratio(eps, t, hot_fluid=True, C_h_c=C_h_c)
T_cold_out_T_cold_in = calculate_temperature_ratio(eps, t, hot_fluid=False, C_h_c=C_h_c)

# Calculate pressure ratios
p_hot_out_p_hot_in = calculate_pressure_ratio(
    L_dh_hot_array, f_hot, g2_hot, T_i_Td=Th_in_Td, p_i_pd=ph_in_pd, eps=eps, t=t, hot_fluid=True, C_h_c=C_h_c
)

# L and d_h are the same for both sides, so L/d_h is identical
L_dh_cold_array = L_dh_hot_array  # Same physical flow length and hydraulic diameter
p_cold_out_p_cold_in = calculate_pressure_ratio(
    L_dh_cold_array, f_cold, g2_cold, T_i_Td=Tc_in_Td, p_i_pd=pc_in_pd, eps=eps, t=t, hot_fluid=False, C_h_c=C_h_c
)

# Find cutoff index for pressure drops
first_invalid_hot = np.argmax(p_hot_out_p_hot_in < (1 - max_dp))
first_invalid_cold = np.argmax(p_cold_out_p_cold_in < (1 - max_dp))

if first_invalid_hot == 0 and first_invalid_cold == 0:
    cutoff_idx = len(L_dh_hot_array)
else:
    cutoff_idx = min(
        first_invalid_hot if first_invalid_hot != 0 else len(L_dh_hot_array),
        first_invalid_cold if first_invalid_cold != 0 else len(L_dh_hot_array),
    )

# Initialize arrays with NaNs
dW_pot_Ex = np.full_like(L_dh_hot_array, np.nan)
dW_pot_Eu = np.full_like(L_dh_hot_array, np.nan)

# Create mask that is True up to cutoff_idx
valid_pressure_mask = np.arange(len(L_dh_hot_array)) < cutoff_idx

# Calculate work potentials with different gamma values for hot and cold
# For Exergy, use average gm1og (or could use separate terms)
gm1og_avg = 0.5 * (gm1og_hot + gm1og_cold)
log_p_hot = np.log(p_hot_out_p_hot_in[valid_pressure_mask])
log_p_cold = np.log(p_cold_out_p_cold_in[valid_pressure_mask])
dW_pot_Ex[valid_pressure_mask] = (
    np.log(T_hot_out_T_hot_in[valid_pressure_mask])
    + np.log(T_cold_out_T_cold_in[valid_pressure_mask])
    - gm1og_avg * (log_p_hot + log_p_cold)
)

# Calculate Euergy change with different gamma for hot and cold
dW_pot_Eu_hot = (
    (1 / ph_in_pd) ** gm1og_hot * Th_in_Td * (T_hot_out_T_hot_in * (1 / p_hot_out_p_hot_in) ** gm1og_hot - 1)
)
dW_pot_Eu_cold = (
    (1 / pc_in_pd) ** gm1og_cold * Tc_in_Td * (T_cold_out_T_cold_in * (1 / p_cold_out_p_cold_in) ** gm1og_cold - 1)
)
dW_pot_Eu[valid_pressure_mask] = dW_pot_Eu_hot[valid_pressure_mask] + dW_pot_Eu_cold[valid_pressure_mask]

# Normalize by Q_max
norm = 1 / (Th_in_Td - Tc_in_Td)
dW_pot_Ex_norm = dW_pot_Ex * norm
dW_pot_Eu_norm = dW_pot_Eu * norm

# ============================================================================
# PLOTTING
# ============================================================================

# Create figure
fig, ax = plt.subplots(figsize=(6, 5))

# Plot Exergy (dashed, black)
ax.plot(A_fr_over_A_fr_baseline, dW_pot_Ex_norm, "k--", label="Exergy", linewidth=1.5)

# Plot Euergy (solid, black)
ax.plot(A_fr_over_A_fr_baseline, dW_pot_Eu_norm, "k-", label="Euergy", linewidth=1.5)

# Find and mark optimal points
# Euergy: minimum of dW_pot_Eu
valid_mask_eu = ~np.isnan(dW_pot_Eu_norm)
if np.any(valid_mask_eu):
    eu_min_idx = np.nanargmin(dW_pot_Eu_norm[valid_mask_eu])
    eu_min_x = A_fr_over_A_fr_baseline[valid_mask_eu][eu_min_idx]
    eu_min_y = dW_pot_Eu_norm[valid_mask_eu][eu_min_idx]
    ax.scatter(eu_min_x, eu_min_y, color="black", s=100, zorder=5, marker="o")
    ax.plot([eu_min_x, eu_min_x], [0, eu_min_y], "k--", alpha=0.5, linewidth=1)

# Exergy: minimum of dW_pot_Ex
valid_mask_ex = ~np.isnan(dW_pot_Ex_norm)
if np.any(valid_mask_ex):
    ex_min_idx = np.nanargmin(dW_pot_Ex_norm[valid_mask_ex])
    ex_min_x = A_fr_over_A_fr_baseline[valid_mask_ex][ex_min_idx]
    ex_min_y = dW_pot_Ex_norm[valid_mask_ex][ex_min_idx]
    ax.scatter(ex_min_x, ex_min_y, color="black", s=100, zorder=5, marker="s")
    ax.plot([ex_min_x, ex_min_x], [0, ex_min_y], "k:", alpha=0.5, linewidth=1)

# Set labels
ax.set_xlabel("$A_{fr}/A_{fr,baseline}$", fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel(r"$\Delta Q_0/Q_{\mathrm{max}}$", fontsize=AXIS_LABEL_SIZE)

# Set tick label sizes
ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

# Use PercentFormatter for y-axis
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))

# Set x-axis to logarithmic scale
ax.set_xscale("log")

# Flip y-axis if requested
if FLIP_Y_AXIS:
    ax.invert_yaxis()

# Set x-axis limits to only show valid range (inverted: high to low left to right)
valid_mask = ~np.isnan(dW_pot_Ex_norm)
if np.any(valid_mask):
    x_min = A_fr_over_A_fr_baseline[valid_mask][0]
    x_max = A_fr_over_A_fr_baseline[valid_mask][-1]
    # Invert: set limits so high values are on left, low on right
    ax.set_xlim(x_max, x_min)

    # Set ticks at 0.1 spacing, but only display labels for specified values
    # Values to display labels for
    label_values = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]

    # Generate all ticks at 0.1 spacing within the axis limits
    x_range_min = min(x_min, x_max)
    x_range_max = max(x_min, x_max)

    # Start from the first 0.1 increment above/below the range
    start_val = np.ceil(x_range_min * 10) / 10
    end_val = np.floor(x_range_max * 10) / 10

    # Generate all ticks at 0.1 spacing
    all_ticks = np.arange(start_val, end_val + 0.05, 0.1)  # +0.05 to include end_val

    # Set all ticks
    ax.set_xticks(all_ticks)

    # Custom formatter: only show labels for specified values
    def format_tick(x, pos):
        # Round to 1 decimal place to handle floating point precision
        x_rounded = round(x, 1)
        if x_rounded in label_values:
            return f"{x_rounded:.1f}"
        else:
            return ""

    ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
else:
    # Fallback if no valid data
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

# Add grid
ax.grid(True, alpha=0.3)

# Add legend
ax.legend(loc="best", fontsize=LEGEND_SIZE, frameon=True)

# Tight layout
plt.tight_layout()

# Save as SVG
output_filename = "hex_Afr_isolation_figure.svg"
# plt.savefig(output_filename, format="svg", bbox_inches="tight")
# print(f"Figure saved as {output_filename}")

# Optionally show the plot
plt.show()
