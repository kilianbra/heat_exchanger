"""Importing public libraries"""

import os

import numpy as np
from CoolProp.CoolProp import PropsSI
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

os.system("cls")

# mflow splits
m_core = 60.3
BPR = 14.15
m_bypass = BPR * m_core

# mflow splits NEW
m_core = 64
BPR = 10.89
m_bypass = BPR * m_core

m_combustor = 48.2  # air mass flow into combustor after accounting for cooling bleed air
m_split_core = 0.8 * m_core
m_split_hx = 0.2 * m_core
m_split_hx_coolant = 0.063 * m_split_hx
fluid_h = "air"

# Example: Set flight altitude in feet and convert to meters
flight_altitude_ft = 39000  # flight altitude in feet
flight_altitude_m = flight_altitude_ft * 0.3048  # convert feet to meters
flight_altitude_m = 11000
# Calculate reference environment temperature (T0) based on altitude
# Use International Standard Atmosphere (ISA) lapse rate for troposphere (up to 11,000 m)
# T0 = 288.15 - 0.0065 * h (h in meters), but for altitudes above 11,000 m, T0 is constant at 216.65 K
if flight_altitude_m <= 11000:
    T0 = 288.15 - 0.0065 * flight_altitude_m
elif flight_altitude_m > 11000 and flight_altitude_m <= 20000:
    T0 = 216.65  # isothermal stratosphere
elif flight_altitude_m > 20000:
    print("Altitude above 20,000 m not supported")
    exit()
P0 = 101325 * (1 - 0.0065 * flight_altitude_m / 288.15) ** 5.2561


# states for cycle
p0 = P0
s0 = PropsSI("S", "T", T0, "P", p0, fluid_h)  # T0 is static, others total I think.. need to confirm
T1 = 248
p1 = 0.362e5
s1 = PropsSI("S", "T", T1, "P", p1, fluid_h)
T2 = 907
p2 = 25.37e5
s2 = PropsSI("S", "T", T2, "P", p2, fluid_h)
T3 = 1610
# p3 = 21.2e5 # actual
p3 = p2  # assume no combustor losses
s3 = PropsSI("S", "T", T3, "P", p3, fluid_h)
T4 = 575
p4 = 0.368e5
s4 = PropsSI("S", "T", T4, "P", p4, fluid_h)

# Hydrogen coolant conditions
fluid_c = "parahydrogen"
Tc_inlet = 40
Tc_inlet_real = 300  # coolant preheated before entering HX to avoid frosting
Pc_inlet = 150e5
Pc_outlet = 150e5 * 0.9

# HX performance
eps = 0.90
dP = 0.85
# Coolant side
C_hot = m_split_hx * PropsSI("C", "T", T4, "P", p4, fluid_h)
C_cold = m_split_hx_coolant * PropsSI("C", "T", Tc_inlet_real, "P", Pc_inlet, fluid_c)
qmax = min(C_hot, C_cold) * (T4 - Tc_inlet_real)
q = eps * qmax
Tc_outlet = Tc_inlet_real + q / C_cold
# Hot side
T4h2 = T4 - q / C_hot
p4h2 = p4 * dP
s4h2 = PropsSI("S", "T", T4h2, "P", p4h2, fluid_h)
print(f"T4h2: {T4h2:.0f} K, p4h2: {p4h2:.0f} Pa, s4h2: {s4h2:.2f} J/kgK")

# isobar curves
qT = np.linspace(100, 2000, 100)  # 0-2000 K queries
qS0 = PropsSI("S", "T", qT, "P", p0, fluid_h)
qS1 = PropsSI("S", "T", qT, "P", p1, fluid_h)
qS2 = PropsSI("S", "T", qT, "P", p2, fluid_h)
# cycle curve
s_cycle = np.array(
    [s0] + [s1] + [PropsSI("S", "T", t, "P", p2, fluid_h) for t in np.linspace(T2, T3, 10)] + [s4] + [s4h2]
)
T_cycle = np.array([T0] + [T1] + list(np.linspace(T2, T3, 10)) + [T4] + [T4h2])

Mflight = 0.85  # flight Mach number
V0 = Mflight * np.sqrt(1.4 * 287 * T0)


def calc_vjet(p0in, T0in):
    # p0 = 0.226 bar, atm static pressure
    gamma = 1.4
    pr_crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    if p0 / p0in < pr_crit:  # flow choked
        pe = p0in * pr_crit
        Me = 1
        Te = T0in / (1 + (gamma - 1) / 2)
        Ve = np.sqrt(gamma * 287 * Te)
    else:
        pe = p0
        # Check if the expression inside sqrt is positive to avoid RuntimeWarning
        sqrt_arg = 2 / (gamma - 1) * ((p0 / p0in) ** ((gamma - 1) / (-gamma)) - 1)
        if sqrt_arg < 0:
            # If negative, set Me to 0 (no flow) or handle as choked flow
            Me = np.nan
            Te = T0in
            Ve = np.nan
        else:
            Me = np.sqrt(sqrt_arg)
            Te = T0in / (1 + (gamma - 1) / 2 * Me**2)
            ae = np.sqrt(gamma * 287 * Te)
            Ve = Me * ae
    return Ve, pe, Te


V4, p4e, T4e = calc_vjet(p4, T4)
V4h2, p4h2e, T4h2e = calc_vjet(p4h2, T4h2)
pbp = 0.45e5
Tbp = 276
Vbp, pbpe, Tbpe = calc_vjet(pbp, Tbp)

# area splits from JY
Abypass = 6.78  # bypass area [m2]
Acore = 1.024  # core area [m2]
Asplit_main = 0.816  # main split area [m2]
Asplit_hx = 0.184  # hx split area [m2]

# thrust calculations
Fnet_bypass = m_bypass * (Vbp - V0) + Abypass * (pbpe - p0)
Fnet_baseline = m_core * (V4 - V0) + Acore * (p4e - p0)
Fnet_preheated = (
    m_split_core * (V4 - V0) + m_split_hx * (V4h2 - V0) + Asplit_main * (p4e - p0) + Asplit_hx * (p4h2e - p0)
)
Fnet_total_baseline = Fnet_baseline + Fnet_bypass
Fnet_total_preheated = Fnet_preheated + Fnet_bypass
# thrust changes
dFnet_core = 1 - Fnet_preheated / Fnet_baseline
dFnet_total = 1 - Fnet_total_preheated / Fnet_total_baseline

fuel_LHV = 120e6  # lower heating value of hydrogen, J/kg
# baseline
heat_addition = m_combustor * (
    T3 * PropsSI("C", "T", T3, "P", p3, fluid_h) - T2 * PropsSI("C", "T", T2, "P", p2, fluid_h)
)
fuel_massflow = heat_addition / fuel_LHV  # mf*LCV = m_combustor*heat_addition, tf mf in kg/s
fuel_massflow_frac = fuel_massflow / m_combustor
tsfc_baseline_core = fuel_massflow / Fnet_baseline
tsfc_baseline_total = fuel_massflow / Fnet_total_baseline

# preheated
H2_q = Tc_outlet * PropsSI("C", "T", Tc_outlet, "P", Pc_outlet, fluid_c) - Tc_inlet * PropsSI(
    "C", "T", Tc_inlet, "P", Pc_inlet, fluid_c
)
fuel_massflow_preheated = heat_addition / (fuel_LHV + H2_q)  # mf*LCV = m_combustor*heat_addition, tf mf in kg/s
heat_frac = H2_q / fuel_LHV
fuel_massflow_preheated_frac = fuel_massflow_preheated / m_combustor
tsfc_preheated_core = fuel_massflow_preheated / Fnet_preheated
tsfc_preheated_total = fuel_massflow_preheated / Fnet_total_preheated

# change in tsfc and fuel flow
d_fuel_massflow = (fuel_massflow_preheated / fuel_massflow - 1) * 100
d_tsfc_core = (tsfc_preheated_core / tsfc_baseline_core - 1) * 100
d_tsfc_total = (tsfc_preheated_total / tsfc_baseline_total - 1) * 100

# PRINT RESULTS
print(f"V0: {V0:.0f} m/s, V4: {V4:.0f} m/s, V4h2: {V4h2:.0f} m/s, Vbp: {Vbp:.0f} m/s")
print(f"T0: {T0:.0f} K, T4e: {T4e:.0f} K, T4h2e: {T4h2e:.0f} K, Tbpe: {Tbpe:.0f} K")
print(
    f"Fnet_bypass: {Fnet_bypass / 1e3:.0f} kN, Fnet_baseline: {Fnet_baseline / 1e3:.0f} kN, Fnet_preheated (split): {Fnet_preheated / 1e3:.0f} kN"
)
print(
    f"Change in thrust, core only (exc. bypass): {dFnet_core * 100:.0f}%, total (inc. bypass): {dFnet_total * 100:.0f}%"
)
print(
    f"fuel/air: {fuel_massflow_frac * 100:.2f}%, fuel mass flow: {fuel_massflow:.2f} kg/s, tsfc_baseline_core: {tsfc_baseline_core:.2f} kg/s/N, tsfc_baseline_total: {tsfc_baseline_total:.2f} kg/s/N"
)
print(
    f"fuel/air: {fuel_massflow_preheated_frac * 100:.2f}%, fuel mass flow: {fuel_massflow_preheated:.2f} kg/s, tsfc_baseline_core: {tsfc_preheated_core:.2f} kg/s/N, tsfc_baseline_total: {tsfc_preheated_total:.2f} kg/s/N"
)
print(f"fraction of sensible heat pick up to fuel LHV: {heat_frac * 100:.2f}%")
print(
    f"Change in fuel mass flow: {d_fuel_massflow:.2f}%, change in core tsfc: {d_tsfc_core:.2f}%, change in total tsfc (inc. bypass): {d_tsfc_total:.2f}%"
)

# pressure drop sensitivity study, constant T4h2, vary p4h2 and plot dV (V4-V0)
n_samples = 20
qp4h2 = np.zeros(n_samples)
qV4h2 = np.zeros(n_samples)
for jj in range(n_samples):
    qp4h2[jj] = p4 * (1 - jj * 0.025)
    qV4h2[jj] = calc_vjet(qp4h2[jj], T4h2)[0]
dVel = qV4h2 - V0

# effectiveness sensitivity study, constant p4h2, vary T4h2 based on eps
qeps = np.zeros(n_samples)
qT4h2_eps = np.zeros(n_samples)
qV4h2_eps = np.zeros(n_samples)
qd_fuel_massflow_eps = np.zeros(n_samples)
# Define the range for effectiveness (e.g., from 0.7 to 1)
eps_min = 0.3
eps_max = 1
for jj in range(n_samples):
    current_eps = eps_min + jj * (eps_max - eps_min) / (n_samples - 1)
    qeps[jj] = current_eps
    # Recalculate heat capacity rates for current effectiveness
    C_hot_current = m_split_hx * PropsSI("C", "T", T4, "P", p4, fluid_h)
    C_cold_current = m_split_hx_coolant * PropsSI("C", "T", Tc_inlet_real, "P", Pc_inlet, fluid_c)
    qmax_current = min(C_hot_current, C_cold_current) * (T4 - Tc_inlet_real)
    # Calculate heat transfer and temperatures
    q_current = current_eps * qmax_current
    Tc_outlet_current = Tc_inlet_real + q_current / C_cold_current
    qT4h2_eps[jj] = T4 - q_current / C_hot_current
    # Calculate heat recovery from hydrogen (use Tc_inlet for consistency)
    H2_q_current = Tc_outlet_current * PropsSI(
        "C", "T", Tc_outlet_current, "P", Pc_outlet, fluid_c
    ) - Tc_inlet * PropsSI("C", "T", Tc_inlet, "P", Pc_inlet, fluid_c)
    # Calculate fuel mass flow for this case
    qd_fuel_massflow_eps[jj] = heat_addition / (fuel_LHV + H2_q_current)
    qV4h2_eps[jj] = calc_vjet(p4 * dP, qT4h2_eps[jj])[0]
dVel_eps = qV4h2_eps - V0

# Combined sensitivity study for effectiveness and pressure drop
# Initialize 2D arrays for results, using n_samples for both dimensions
qp4h2_both = np.zeros((n_samples, n_samples))
qT4h2_both = np.zeros((n_samples, n_samples))
qV4h2_both = np.zeros((n_samples, n_samples))
dVel_both = np.zeros((n_samples, n_samples))
eps_cold_both = np.zeros((n_samples, n_samples))  # Store H2 effectiveness
# Define ranges for pressure drop and effectiveness
dp_percent_values = np.linspace(0, 35, n_samples)
# Effectiveness values: from eps_min to eps_max
eps_values = np.linspace(eps_min, eps_max, n_samples)
for i_dp in range(n_samples):  # Iterate over pressure drop
    # Calculate current pressure at HX exit based on pressure drop percentage
    current_p4h2 = p4 * (1 - dp_percent_values[i_dp] / 100)
    for i_eps in range(n_samples):  # Iterate over effectiveness
        current_eps = eps_values[i_eps]
        # Recalculate heat capacity rates for current conditions
        C_hot_current = m_split_hx * PropsSI("C", "T", T4, "P", p4, fluid_h)
        C_cold_current = m_split_hx_coolant * PropsSI("C", "T", Tc_inlet_real, "P", Pc_inlet, fluid_c)
        C_min_current = min(C_hot_current, C_cold_current)
        # Convert to H2 effectiveness: eps = C_cold / C_min * eps_cold, so eps_cold = eps * C_min / C_cold
        eps_cold_both[i_dp, i_eps] = current_eps * C_min_current / C_cold_current
        qmax_current = C_min_current * (T4 - Tc_inlet_real)
        # Calculate heat transfer and temperatures
        q_current = current_eps * qmax_current
        Tc_outlet_current = Tc_inlet_real + q_current / C_cold_current
        qT4h2_both[i_dp, i_eps] = T4 - q_current / C_hot_current
        # Store the pressure for this iteration
        qp4h2_both[i_dp, i_eps] = current_p4h2
        # Calculate V4h2 using the calculated p4h2 and T4h2
        qV4h2_both[i_dp, i_eps] = calc_vjet(qp4h2_both[i_dp, i_eps], qT4h2_both[i_dp, i_eps])[0]
        # Calculate dVel
        dVel_both[i_dp, i_eps] = qV4h2_both[i_dp, i_eps] - V0
qFnet_HX_both = m_split_hx * dVel_both
qFnet_preheated_both = m_split_core * (V4 - V0) + qFnet_HX_both
# Calculate lost thrust: baseline HX thrust - actual HX thrust
# Baseline HX thrust would be m_split_hx * (V4 - V0) if there was no pressure drop/temp reduction
baseline_HX_thrust = m_split_hx * (V4 - V0)
lost_thrust_HX_both = baseline_HX_thrust - qFnet_HX_both  # Lost thrust due to HX

# Recalculate fuel mass flow for each effectiveness case
qd_fuel_massflow_both = np.zeros((n_samples, n_samples))
qd_tsfc_both = np.zeros((n_samples, n_samples))
for i_dp in range(n_samples):
    for i_eps in range(n_samples):
        # Recalculate heat recovery for this effectiveness
        current_eps = eps_values[i_eps]
        C_hot_current = m_split_hx * PropsSI("C", "T", T4, "P", p4, fluid_h)
        C_cold_current = m_split_hx_coolant * PropsSI("C", "T", Tc_inlet_real, "P", Pc_inlet, fluid_c)
        qmax_current = min(C_hot_current, C_cold_current) * (T4 - Tc_inlet_real)
        q_current = current_eps * qmax_current
        Tc_outlet_current = Tc_inlet_real + q_current / C_cold_current
        # Calculate heat recovery from hydrogen (use Tc_inlet for consistency with original calculation)
        H2_q_current = Tc_outlet_current * PropsSI(
            "C", "T", Tc_outlet_current, "P", Pc_outlet, fluid_c
        ) - Tc_inlet * PropsSI("C", "T", Tc_inlet, "P", Pc_inlet, fluid_c)
        # Calculate fuel mass flow for this case
        qd_fuel_massflow_both[i_dp, i_eps] = heat_addition / (fuel_LHV + H2_q_current)
        # Calculate TSFC change
        qd_tsfc_both[i_dp, i_eps] = (
            (qd_fuel_massflow_both[i_dp, i_eps] / (qFnet_preheated_both[i_dp, i_eps] + Fnet_bypass))
            / tsfc_baseline_total
            - 1
        ) * 100


# T-S DIAGRAM FUNCTION (commented out - call plot_ts_diagram() to generate)
def plot_ts_diagram():
    """Plot T-s diagram for the cycle."""
    plt.figure()
    plt.title("T-s diagram")
    plt.scatter([s0, s1, s2, s3, s4, s4h2], [T0, T1, T2, T3, T4, T4h2], c="black", marker="o")
    plt.plot(s_cycle, T_cycle, c="black", label="Core + Split streams")
    plt.plot(qS0, qT, c="black", linestyle="--", linewidth=0.5, label="Atm")
    plt.plot(qS1, qT, c="black", linestyle="--", linewidth=0.5, label="ramPressure")
    plt.plot(qS2, qT, c="black", linestyle="--", linewidth=0.5, label="CPR")
    plt.xlabel("Entropy")
    plt.ylabel("Temperature")
    plt.xlim(round(s1 - 500, -2), round(s4 + 500, -2))
    plt.ylim(0, 2000)
    plt.legend()
    plt.show()


# Uncomment the line below to generate the T-s diagram
# plot_ts_diagram()

# COMBINED SENSITIVITY - CONTOUR PLOTS
# Create 2D contour plots with H2 Effectiveness (eps_cold) vs Lost Thrust of HX
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Create regular grid for interpolation: H2 effectiveness (x) vs lost thrust (y)
eps_cold_min = eps_cold_both.min()
eps_cold_max = eps_cold_both.max()
eps_cold_grid = np.linspace(eps_cold_min * 100, eps_cold_max * 100, n_samples)
lost_thrust_grid = np.linspace(lost_thrust_HX_both.min(), lost_thrust_HX_both.max(), n_samples)
EpsCold_grid, LostThrust_grid = np.meshgrid(eps_cold_grid, lost_thrust_grid)

# Prepare data points for interpolation (flatten the arrays)
# The data is organized as [i_dp, i_eps], so we need to create points correctly
eps_cold_points = []
lost_thrust_points = []
dp_points = []
tsfc_points = []
for i_dp in range(n_samples):
    for i_eps in range(n_samples):
        eps_cold_points.append(eps_cold_both[i_dp, i_eps] * 100)
        lost_thrust_points.append(lost_thrust_HX_both[i_dp, i_eps])
        dp_points.append(dp_percent_values[i_dp])
        tsfc_points.append(qd_tsfc_both[i_dp, i_eps])

eps_cold_points = np.array(eps_cold_points)
lost_thrust_points = np.array(lost_thrust_points)
dp_points = np.array(dp_points)
tsfc_points = np.array(tsfc_points)

# Interpolate pressure drop and TSFC onto the new grid
dp_interp = griddata(
    (eps_cold_points, lost_thrust_points),
    dp_points,
    (EpsCold_grid, LostThrust_grid),
    method="linear",
    fill_value=np.nan,
)
tsfc_interp = griddata(
    (eps_cold_points, lost_thrust_points),
    tsfc_points,
    (EpsCold_grid, LostThrust_grid),
    method="linear",
    fill_value=np.nan,
)

# Left plot: Pressure drop contours as black lines
contour1 = ax1.contour(
    EpsCold_grid,
    LostThrust_grid,
    dp_interp,
    levels=10,
    colors="black",
    linewidths=1.5,
    alpha=0.8,
)
ax1.clabel(contour1, inline=True, fontsize=9, fmt="%g%%")
ax1.set_title("Pressure Drop [%] vs H2 Effectiveness & Lost Thrust")
ax1.set_xlabel("H2 Effectiveness [%]")
ax1.set_ylabel("Lost Thrust of HX [N]")
ax1.grid(True, alpha=0.3)

# Right plot: TSFC Change contours as colormap
norm2 = TwoSlopeNorm(vmin=-4.5, vcenter=0, vmax=0.5)
contour2 = ax2.contourf(EpsCold_grid, LostThrust_grid, tsfc_interp, levels=50, cmap="coolwarm", norm=norm2)
ax2.contour(EpsCold_grid, LostThrust_grid, tsfc_interp, levels=10, colors="black", alpha=0.3, linewidths=0.5)
ax2.set_title("TSFC Change [%] vs H2 Effectiveness & Lost Thrust")
ax2.set_xlabel("H2 Effectiveness [%]")
ax2.set_ylabel("Lost Thrust of HX [N]")
ax2.grid(True, alpha=0.3)
cbar2 = fig.colorbar(contour2, ax=ax2, shrink=0.8)
cbar2.set_label("TSFC Change [%]")
# Invert the TSFC colorbar so negative values (better efficiency) are green
cbar2.ax.invert_yaxis()

# Find design points in new coordinate system (H2 effectiveness, lost thrust)
# Design A: 83.6% eps, 18.2% dP
# Design B: 86.7% eps, 4.2% dP
# Need to convert eps to eps_cold for design points
# First, get the heat capacity rates (they should be constant)
C_hot_design = m_split_hx * PropsSI("C", "T", T4, "P", p4, fluid_h)
C_cold_design = m_split_hx_coolant * PropsSI("C", "T", Tc_inlet_real, "P", Pc_inlet, fluid_c)
C_min_design = min(C_hot_design, C_cold_design)

design_A_eps = 83.6 / 100  # Convert to fraction
design_A_eps_cold = design_A_eps * C_min_design / C_cold_design
design_A_dp = 18.2
design_B_eps = 86.7 / 100  # Convert to fraction
design_B_eps_cold = design_B_eps * C_min_design / C_cold_design
design_B_dp = 4.2

# Find lost thrust for design points by interpolating
design_A_lost_thrust = griddata(
    (eps_cold_points, dp_points),
    lost_thrust_points,
    (design_A_eps_cold * 100, design_A_dp),
    method="linear",
)
design_B_lost_thrust = griddata(
    (eps_cold_points, dp_points),
    lost_thrust_points,
    (design_B_eps_cold * 100, design_B_dp),
    method="linear",
)

# Add design points as star markers
ax1.scatter(
    design_A_eps_cold * 100,
    design_A_lost_thrust,
    marker="*",
    s=200,
    color="silver",
    edgecolor="black",
    linewidth=1,
    label="Design A (Inboard)",
    zorder=5,
)
ax2.scatter(
    design_A_eps_cold * 100,
    design_A_lost_thrust,
    marker="*",
    s=200,
    color="silver",
    edgecolor="black",
    linewidth=1,
    label="Design A (Inboard, 140 kg)",
    zorder=5,
)

ax1.scatter(
    design_B_eps_cold * 100,
    design_B_lost_thrust,
    marker="*",
    s=200,
    color="gold",
    edgecolor="black",
    linewidth=1,
    label="Design B (Outboard)",
    zorder=5,
)
ax2.scatter(
    design_B_eps_cold * 100,
    design_B_lost_thrust,
    marker="*",
    s=200,
    color="gold",
    edgecolor="black",
    linewidth=1,
    label="Design B (Outboard, 260 kg)",
    zorder=5,
)

# Add legends to both plots
ax1.legend(loc="upper right", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()

plt.show()
print()
