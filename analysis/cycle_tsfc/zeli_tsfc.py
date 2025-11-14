"""Importing public libraries"""

import numpy as np
from CoolProp.CoolProp import PropsSI
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid


def calc_vjet(p0in, T0in, p_back, gamma=1.4, R=287):
    """Calculate jet velocity for given stagnation pressure and temperature.
    Assumes Isentropic Compressible flow in a 1D steady nozzle
    Assumes the exit plane is the throat or that the exit is perfectly expanded to p_back"""
    pr_crit = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    if p_back / p0in < pr_crit:  # flow choked - cannot achieve exit pressure so will have
        pe = p0in * pr_crit
        Me = 1
        Te = T0in / (1 + (gamma - 1) / 2)
        Ve = np.sqrt(gamma * 287 * Te)
    else:
        pe = p_back
        # Check if the expression inside sqrt is positive to avoid RuntimeWarning
        sqrt_arg = 2 / (gamma - 1) * ((p_back / p0in) ** ((gamma - 1) / (-gamma)) - 1)
        if sqrt_arg < 0:
            raise ValueError(f"Exit pressure {p_back:.1e} Pa too low to achieve back pressure {p0in:.1e} Pa")
        else:
            Me = np.sqrt(sqrt_arg)
            Te = T0in / (1 + (gamma - 1) / 2 * Me**2)
            ae = np.sqrt(gamma * 287 * Te)
            Ve = Me * ae
    return Ve, pe, Te


def calculate_cycle_tsfc(
    # Cycle inputs
    T0,
    p0,
    T1,
    p1,
    T2,
    p2,
    T3,
    p3,
    T4,
    p4,
    # Mass flows
    m_core,
    m_bypass,
    m_combustor,
    m_split_core,
    m_split_hx,
    m_split_hx_coolant,
    # Coolant conditions
    Tc_pump_exit,
    Tc_hex_inlet,
    Pc_inlet,
    # HX performance
    eps,
    dp_h_pct_of_in,
    dp_c_pct_of_in,
    # Flight conditions
    Mflight,
    # Areas
    Abypass,
    Acore,
    Asplit_main,
    Asplit_hx,
    # Fuel properties
    fuel_LHV,
    # Fluid models
    f_h,
    f_c,
):
    """
    Calculate TSFC and other cycle outputs for given inputs.

    Creates a FluidInputs object for the heat exchanger and performs all calculations.

    Returns:
        dict: Dictionary containing all calculated outputs including TSFC, thrust, etc.
    """
    # Create cycle state objects
    state_0 = f_h.state(T0, p0)
    state_1 = f_h.state(T1, p1)
    state_2 = f_h.state(T2, p2)
    state_3 = f_h.state(T3, p3)
    state_4 = f_h.state(T4, p4)

    # Extract properties from states
    s0 = state_0.s
    s1 = state_1.s
    s2 = state_2.s
    s3 = state_3.s
    s4 = state_4.s

    # Create coolant state objects
    cold_in = f_c.state(Tc_hex_inlet, Pc_inlet)  # Use real inlet temp for calculations

    # HX performance calculations
    C_hot = m_split_hx * state_4.cp
    C_cold = m_split_hx_coolant * cold_in.cp
    qmax = min(C_hot, C_cold) * (T4 - Tc_hex_inlet)
    q = eps * qmax
    Tc_outlet = Tc_hex_inlet + q / C_cold
    Pc_outlet = Pc_inlet * (1 - dp_c_pct_of_in / 100)

    # Hot side HEx exit
    T4h2 = T4 - q / C_hot
    p4h2 = p4 * (1 - dp_h_pct_of_in / 100)
    state_4h2 = f_h.state(T4h2, p4h2)
    s4h2 = state_4h2.s

    # Create cold outlet state after HX
    cold_out = f_c.state(Tc_outlet, Pc_outlet)

    # Create FluidInputs object for heat exchanger
    fluid_inputs = FluidInputs(
        hot=f_h,
        cold=f_c,
        m_dot_hot=m_split_hx,
        m_dot_cold=m_split_hx_coolant,
        Tc_in=Tc_hex_inlet,
        Pc_in=Pc_inlet,
        Th_in=T4,
        Ph_in=p4,
    )

    # Flight velocity
    V0 = Mflight * np.sqrt(1.4 * 287 * T0)

    # Jet velocities
    V4, p4e, T4e = calc_vjet(p4, T4, p0)
    V4h2, p4h2e, T4h2e = calc_vjet(p4h2, T4h2, p0)
    pbp = 0.45e5  # KB: Fixed for given altitude ?
    Tbp = 276  # KB: Fixed for given altitude ?
    Vbp, pbpe, Tbpe = calc_vjet(pbp, Tbp, p0)

    # Thrust calculations
    Fnet_bypass = m_bypass * (Vbp - V0) + Abypass * (pbpe - p0)
    Fnet_baseline = m_core * (V4 - V0) + Acore * (p4e - p0)
    Fnet_preheated = (
        m_split_core * (V4 - V0) + m_split_hx * (V4h2 - V0) + Asplit_main * (p4e - p0) + Asplit_hx * (p4h2e - p0)
    )
    Fnet_total_baseline = Fnet_baseline + Fnet_bypass
    Fnet_total_preheated = Fnet_preheated + Fnet_bypass

    # Thrust changes
    dFnet_core = 1 - Fnet_preheated / Fnet_baseline
    dFnet_total = 1 - Fnet_total_preheated / Fnet_total_baseline

    # Baseline fuel calculations
    heat_addition = m_combustor * (state_3.h - state_2.h)
    fuel_massflow = heat_addition / fuel_LHV
    fuel_massflow_frac = fuel_massflow / m_combustor
    tsfc_baseline_core = fuel_massflow / Fnet_baseline
    tsfc_baseline_total = fuel_massflow / Fnet_total_baseline

    # Preheated fuel calculations
    cold_pump_exit = f_c.state(Tc_pump_exit, Pc_inlet)  # Reference state at actual inlet temp
    H2_q = cold_out.h - cold_pump_exit.h
    fuel_massflow_preheated = heat_addition / (fuel_LHV + H2_q)
    heat_frac = H2_q / fuel_LHV
    fuel_massflow_preheated_frac = fuel_massflow_preheated / m_combustor
    tsfc_preheated_core = fuel_massflow_preheated / Fnet_preheated
    tsfc_preheated_total = fuel_massflow_preheated / Fnet_total_preheated

    # Change in TSFC and fuel flow
    d_fuel_massflow = (fuel_massflow_preheated / fuel_massflow - 1) * 100
    d_tsfc_core = (tsfc_preheated_core / tsfc_baseline_core - 1) * 100
    d_tsfc_total = (tsfc_preheated_total / tsfc_baseline_total - 1) * 100

    # Calculate lost thrust
    baseline_HX_thrust = m_split_hx * (V4 - V0)
    lost_thrust_HX = baseline_HX_thrust - m_split_hx * (V4h2 - V0)

    # Calculate H2 effectiveness (eps_cold)
    C_min = min(C_hot, C_cold)
    eps_cold = eps * C_min / C_cold

    return {
        "fluid_inputs": fluid_inputs,
        "V0": V0,
        "V4": V4,
        "V4h2": V4h2,
        "Vbp": Vbp,
        "T4e": T4e,
        "T4h2e": T4h2e,
        "Tbpe": Tbpe,
        "Fnet_bypass": Fnet_bypass,
        "Fnet_baseline": Fnet_baseline,
        "Fnet_preheated": Fnet_preheated,
        "Fnet_total_baseline": Fnet_total_baseline,
        "Fnet_total_preheated": Fnet_total_preheated,
        "dFnet_core": dFnet_core,
        "dFnet_total": dFnet_total,
        "fuel_massflow": fuel_massflow,
        "fuel_massflow_frac": fuel_massflow_frac,
        "tsfc_baseline_core": tsfc_baseline_core,
        "tsfc_baseline_total": tsfc_baseline_total,
        "fuel_massflow_preheated": fuel_massflow_preheated,
        "fuel_massflow_preheated_frac": fuel_massflow_preheated_frac,
        "tsfc_preheated_core": tsfc_preheated_core,
        "tsfc_preheated_total": tsfc_preheated_total,
        "heat_frac": heat_frac,
        "d_fuel_massflow": d_fuel_massflow,
        "d_tsfc_core": d_tsfc_core,
        "d_tsfc_total": d_tsfc_total,
        "lost_thrust_HX": lost_thrust_HX,
        "eps_cold": eps_cold,
        "T4h2": T4h2,
        "p4h2": p4h2,
        "s4h2": s4h2,
        "state_0": state_0,
        "state_1": state_1,
        "state_2": state_2,
        "state_3": state_3,
        "state_4": state_4,
        "state_4h2": state_4h2,
        "cold_in": cold_in,
        "cold_out": cold_out,
        "s0": s0,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "C_hot": C_hot,
        "C_cold": C_cold,
        "C_min": C_min,
        "T4": T4,
        "p4": p4,
        "Tc_pump_exit": Tc_pump_exit,
        "Tc_hex_inlet": Tc_hex_inlet,
        "Pc_outlet": Pc_outlet,
    }


def main():
    """Main function that sets up inputs and performs plotting."""

    # ========== INPUTS ==========
    # mflow splits
    m_core = 64
    BPR = 10.89
    m_bypass = BPR * m_core

    m_combustor = 48.2  # air mass flow into combustor after accounting for cooling bleed air
    fraction_core_to_hex = 0.2
    m_split_core = (1 - fraction_core_to_hex) * m_core
    m_split_hx = fraction_core_to_hex * m_core
    m_split_hx_coolant = 0.063 * m_split_hx  # chosen to get C_ratio = 1, no relationship to fuel reqd

    # Fluid models
    f_h = PerfectGasFluid.from_name("air")
    f_c = PerfectGasFluid.from_name("parahydrogen")

    # Flight conditions
    flight_altitude_ft = 39000  # flight altitude in feet
    flight_altitude_m = flight_altitude_ft * 0.3048  # convert feet to meters
    flight_altitude_m = 11000
    # Calculate reference environment temperature (T0) based on altitude
    if flight_altitude_m <= 11000:
        T0 = 288.15 - 0.0065 * flight_altitude_m
    elif flight_altitude_m > 11000 and flight_altitude_m <= 20000:
        T0 = 216.65  # isothermal stratosphere
    elif flight_altitude_m > 20000:
        print("Altitude above 20,000 m not supported")
        exit()
    P0 = 101325 * (1 - 0.0065 * flight_altitude_m / 288.15) ** 5.2561

    # Cycle state temperatures and pressures
    T1 = 248
    p1 = 0.362e5
    T2 = 907
    p2 = 25.37e5
    T3 = 1610
    p3 = p2  # assume no combustor losses
    T4 = 575
    p4 = 0.368e5

    # Hydrogen coolant conditions
    Tc_pump_exit = 40
    Tc_hex_inlet = 275  # coolant preheated before entering HX to avoid frosting
    Pc_inlet = 150e5

    # HX performance
    eps = 0.90
    dp_h_pct_of_in = 15  # % drop
    dp_c_pct_of_in = 10  # % drop

    # Flight conditions
    Mflight = 0.85  # flight Mach number

    # Areas
    Abypass = 6.78  # bypass area [m2]
    Acore = 1.024  # core area [m2]
    Asplit_main = 0.816  # main split area [m2]
    Asplit_hx = 0.184  # hx split area [m2]

    # Fuel properties
    fuel_LHV = 120e6  # lower heating value of hydrogen, J/kg

    # ========== BASELINE CALCULATION ==========
    results = calculate_cycle_tsfc(
        T0=T0,
        p0=P0,
        T1=T1,
        p1=p1,
        T2=T2,
        p2=p2,
        T3=T3,
        p3=p3,
        T4=T4,
        p4=p4,
        m_core=m_core,
        m_bypass=m_bypass,
        m_combustor=m_combustor,
        m_split_core=m_split_core,
        m_split_hx=m_split_hx,
        m_split_hx_coolant=m_split_hx_coolant,
        Tc_pump_exit=Tc_pump_exit,
        Tc_hex_inlet=Tc_hex_inlet,
        Pc_inlet=Pc_inlet,
        eps=eps,
        dp_h_pct_of_in=dp_h_pct_of_in,
        dp_c_pct_of_in=dp_c_pct_of_in,
        Mflight=Mflight,
        Abypass=Abypass,
        Acore=Acore,
        Asplit_main=Asplit_main,
        Asplit_hx=Asplit_hx,
        fuel_LHV=fuel_LHV,
        f_h=f_h,
        f_c=f_c,
    )

    # Print baseline results
    print(f"T4h2: {results['T4h2']:.0f} K, p4h2: {results['p4h2']:.0f} Pa, s4h2: {results['s4h2']:.2f} J/kgK")
    print(
        f"V0: {results['V0']:.0f} m/s, V4: {results['V4']:.0f} m/s, V4h2: {results['V4h2']:.0f} m/s, Vbp: {results['Vbp']:.0f} m/s"
    )
    print(
        f"T0: {T0:.0f} K, T4e: {results['T4e']:.0f} K, T4h2e: {results['T4h2e']:.0f} K, Tbpe: {results['Tbpe']:.0f} K"
    )
    print(
        f"Fnet_bypass: {results['Fnet_bypass'] / 1e3:.0f} kN, Fnet_baseline: {results['Fnet_baseline'] / 1e3:.0f} kN, Fnet_preheated (split): {results['Fnet_preheated'] / 1e3:.0f} kN"
    )
    print(
        f"Change in thrust, core only (exc. bypass): {results['dFnet_core'] * 100:.0f}%, total (inc. bypass): {results['dFnet_total'] * 100:.0f}%"
    )
    print(
        f"fuel/air: {results['fuel_massflow_frac'] * 100:.2f}%, fuel mass flow: {results['fuel_massflow']:.2f} kg/s, tsfc_baseline_core: {results['tsfc_baseline_core']:.1e} kg/s/N, tsfc_baseline_total: {results['tsfc_baseline_total']:.1e} kg/s/N"
    )
    print(
        f"fuel/air: {results['fuel_massflow_preheated_frac'] * 100:.2f}%, fuel mass flow: {results['fuel_massflow_preheated']:.2f} kg/s, tsfc_baseline_core: {results['tsfc_preheated_core']:.1e} kg/s/N, tsfc_baseline_total: {results['tsfc_preheated_total']:.1e} kg/s/N"
    )
    print(f"fraction of sensible heat pick up to fuel LHV: {results['heat_frac'] * 100:.2f}%")
    print(
        f"Change in fuel mass flow: {results['d_fuel_massflow']:.2f}%, change in core tsfc: {results['d_tsfc_core']:.2f}%, change in total tsfc (inc. bypass): {results['d_tsfc_total']:.2f}%"
    )

    # Store baseline values for sensitivity studies
    tsfc_baseline_total = results["tsfc_baseline_total"]
    Fnet_baseline = results["Fnet_baseline"]
    Fnet_bypass = results["Fnet_bypass"]
    V0 = results["V0"]

    # ========== SENSITIVITY STUDIES ==========
    n_samples = 20
    eps_min = 0.3
    eps_max = 1

    # Combined sensitivity study for effectiveness and pressure drop
    qp4h2_both = np.zeros((n_samples, n_samples))
    qT4h2_both = np.zeros((n_samples, n_samples))
    qV4h2_both = np.zeros((n_samples, n_samples))
    dVel_both = np.zeros((n_samples, n_samples))
    eps_cold_both = np.zeros((n_samples, n_samples))  # Store H2 effectiveness
    lost_thrust_HX_both = np.zeros((n_samples, n_samples))
    qd_tsfc_both = np.zeros((n_samples, n_samples))

    # Define ranges for pressure drop and effectiveness
    dp_percent_values = np.linspace(0, 35, n_samples)
    eps_values = np.linspace(eps_min, eps_max, n_samples)

    for i_dp in range(n_samples):  # Iterate over pressure drop
        current_dp_h_pct = dp_percent_values[i_dp]
        for i_eps in range(n_samples):  # Iterate over effectiveness
            current_eps = eps_values[i_eps]

            # Call calculation function for this combination
            loop_results = calculate_cycle_tsfc(
                T0=T0,
                p0=P0,
                T1=T1,
                p1=p1,
                T2=T2,
                p2=p2,
                T3=T3,
                p3=p3,
                T4=T4,
                p4=p4,
                m_core=m_core,
                m_bypass=m_bypass,
                m_combustor=m_combustor,
                m_split_core=m_split_core,
                m_split_hx=m_split_hx,
                m_split_hx_coolant=m_split_hx_coolant,
                Tc_pump_exit=Tc_pump_exit,
                Tc_hex_inlet=Tc_hex_inlet,
                Pc_inlet=Pc_inlet,
                eps=current_eps,
                dp_h_pct_of_in=current_dp_h_pct,
                dp_c_pct_of_in=dp_c_pct_of_in,
                Mflight=Mflight,
                Abypass=Abypass,
                Acore=Acore,
                Asplit_main=Asplit_main,
                Asplit_hx=Asplit_hx,
                fuel_LHV=fuel_LHV,
                f_h=f_h,
                f_c=f_c,
            )

            # Store results
            eps_cold_both[i_dp, i_eps] = loop_results["eps_cold"]
            qT4h2_both[i_dp, i_eps] = loop_results["T4h2"]
            qp4h2_both[i_dp, i_eps] = loop_results["p4h2"]
            qV4h2_both[i_dp, i_eps] = loop_results["V4h2"]
            dVel_both[i_dp, i_eps] = loop_results["V4h2"] - V0
            lost_thrust_HX_both[i_dp, i_eps] = loop_results["lost_thrust_HX"]

            # Calculate TSFC change
            qFnet_preheated_current = loop_results["Fnet_preheated"]
            fuel_massflow_current = loop_results["fuel_massflow_preheated"]
            qd_tsfc_both[i_dp, i_eps] = (
                (fuel_massflow_current / (qFnet_preheated_current + Fnet_bypass)) / tsfc_baseline_total - 1
            ) * 100

    # Calculate ideal 0% pressure drop line
    eps_ideal = np.linspace(eps_min, eps_max, n_samples)
    eps_cold_ideal = np.zeros(n_samples)
    lost_thrust_ideal = np.zeros(n_samples)

    for jj in range(n_samples):
        current_eps = eps_ideal[jj]

        # Call calculation function for ideal case (0% pressure drop)
        ideal_results = calculate_cycle_tsfc(
            T0=T0,
            p0=P0,
            T1=T1,
            p1=p1,
            T2=T2,
            p2=p2,
            T3=T3,
            p3=p3,
            T4=T4,
            p4=p4,
            m_core=m_core,
            m_bypass=m_bypass,
            m_combustor=m_combustor,
            m_split_core=m_split_core,
            m_split_hx=m_split_hx,
            m_split_hx_coolant=m_split_hx_coolant,
            Tc_pump_exit=Tc_pump_exit,
            Tc_hex_inlet=Tc_hex_inlet,
            Pc_inlet=Pc_inlet,
            eps=current_eps,
            dp_h_pct_of_in=0.0,  # 0% pressure drop
            dp_c_pct_of_in=dp_c_pct_of_in,
            Mflight=Mflight,
            Abypass=Abypass,
            Acore=Acore,
            Asplit_main=Asplit_main,
            Asplit_hx=Asplit_hx,
            fuel_LHV=fuel_LHV,
            f_h=f_h,
            f_c=f_c,
        )

        eps_cold_ideal[jj] = ideal_results["eps_cold"]
        lost_thrust_ideal[jj] = ideal_results["lost_thrust_HX"]

    # ========== PLOTTING ==========
    # T-S DIAGRAM FUNCTION (commented out - call plot_ts_diagram() to generate)
    def plot_ts_diagram():
        """Plot T-s diagram for the cycle."""
        # cycle curve
        s_cycle = np.array(
            [results["s0"]]
            + [results["s1"]]
            + [f_h.state(t, p2).s for t in np.linspace(T2, T3, 10)]
            + [results["s4"]]
            + [results["state_4h2"].s]
        )
        T_cycle = np.array([T0] + [T1] + list(np.linspace(T2, T3, 10)) + [T4] + [results["T4h2"]])

        # isobar curves
        qT = np.linspace(100, 2000, 100)  # 100-2000 K queries
        fluid_h = "air"
        qS0 = PropsSI("S", "T", qT, "P", P0, fluid_h)  # KB: PropsSI is vectorised!
        qS1 = PropsSI("S", "T", qT, "P", p1, fluid_h)
        qS2 = PropsSI("S", "T", qT, "P", p2, fluid_h)
        plt.figure()
        plt.title("T-s diagram")
        plt.scatter(
            [results["s0"], results["s1"], results["s2"], results["s3"], results["s4"], results["s4h2"]],
            [T0, T1, T2, T3, T4, results["T4h2"]],
            c="black",
            marker="o",
        )
        plt.plot(s_cycle, T_cycle, c="black", label="Core + Split streams")
        plt.plot(qS0, qT, c="black", linestyle="--", linewidth=0.5, label="Atm")
        plt.plot(qS1, qT, c="black", linestyle="--", linewidth=0.5, label="ramPressure")
        plt.plot(qS2, qT, c="black", linestyle="--", linewidth=0.5, label="CPR")
        plt.xlabel("Entropy")
        plt.ylabel("Temperature")
        plt.xlim(round(results["s1"] - 500, -2), round(results["s4"] + 500, -2))
        plt.ylim(0, 2000)
        plt.legend()
        plt.show()

    # Uncomment the line below to generate the T-s diagram
    # plot_ts_diagram()

    # COMBINED SENSITIVITY - CONTOUR PLOTS
    # Create 2D contour plots with H2 Effectiveness (eps_cold) vs Lost Thrust of HX
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 2]})

    # Create regular grid for interpolation: H2 effectiveness (x) vs lost thrust (y)
    eps_cold_min = eps_cold_both.min()
    eps_cold_max = eps_cold_both.max()
    eps_cold_grid = np.linspace(eps_cold_min * 100, eps_cold_max * 100, n_samples)
    lost_thrust_grid = np.linspace(lost_thrust_HX_both.min(), lost_thrust_HX_both.max(), n_samples)
    EpsCold_grid, LostThrust_grid = np.meshgrid(eps_cold_grid, lost_thrust_grid)

    # Prepare data points for interpolation (flatten the arrays)
    thrust_normalisation = Fnet_baseline / 100

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
        LostThrust_grid / thrust_normalisation,
        dp_interp,
        levels=10,
        colors="black",
        linewidths=1.5,
        alpha=0.8,
    )
    ax1.clabel(contour1, inline=True, fontsize=9, fmt="%g%%")
    # Add ideal 0% pressure drop line
    ax1.plot(
        eps_cold_ideal * 100,
        lost_thrust_ideal / thrust_normalisation,
        color="red",
        linestyle="--",
        linewidth=2,
        label="0% Pressure Drop (Ideal)",
        zorder=10,
    )
    ax1.set_title("Pressure Drop [%] vs H2 Effectiveness & Lost Thrust")
    ax1.set_xlabel("H2 Effectiveness [%]")
    ax1.set_ylabel("Lost Thrust due to HEx [% of core thrust]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # Right plot: TSFC Change contours as colormap with overlay lines
    norm2 = TwoSlopeNorm(vmin=-4.5, vcenter=0, vmax=2.0)
    contour2_filled = ax2.contourf(
        EpsCold_grid, LostThrust_grid / thrust_normalisation, tsfc_interp, levels=50, cmap="coolwarm", norm=norm2
    )
    # Overlay specific contour lines at 1%, 0%, -1%, -2%, -3%
    contour2_lines = ax2.contour(
        EpsCold_grid,
        LostThrust_grid / thrust_normalisation,
        tsfc_interp,
        levels=[-3, -2, -1, 0, 1],
        colors="black",
        linewidths=1.5,
        alpha=0.8,
    )
    ax2.clabel(contour2_lines, inline=True, fontsize=9, fmt="%g%%")
    ax2.set_title("TSFC Change [%] vs H2 Effectiveness & Lost Thrust")
    ax2.set_xlabel("H2 Effectiveness [%]")
    ax2.grid(True, alpha=0.3)
    cbar2 = fig.colorbar(contour2_filled, ax=ax2, shrink=0.8)
    cbar2.set_label("TSFC Change [%]")
    # Invert the TSFC colorbar so negative values (better efficiency) are green
    cbar2.ax.invert_yaxis()

    # Find design points in new coordinate system (H2 effectiveness, lost thrust)
    # Design A: 83.6% eps, 18.2% dp_h_pct_of_in
    # Design B: 86.7% eps, 4.2% dp_h_pct_of_in
    # Need to convert eps to eps_cold for design points
    # First, get the heat capacity rates (they should be constant)
    C_cold_design = results["C_cold"]
    C_min_design = results["C_min"]

    design_A_eps = 83.6 / 100  # Convert to fraction
    design_A_eps_cold = design_A_eps * C_min_design / C_cold_design
    design_A_dp = 18.2
    design_B_eps = 86.7 / 100  # Convert to fraction
    design_B_eps_cold = design_B_eps * C_min_design / C_cold_design
    design_B_dp = 4.2

    # Find lost thrust for design points by interpolating
    design_A_lost_thrust = griddata(
        (eps_cold_points, dp_points),
        lost_thrust_points / thrust_normalisation,
        (design_A_eps_cold * 100, design_A_dp),
        method="linear",
    )
    design_B_lost_thrust = griddata(
        (eps_cold_points, dp_points),
        lost_thrust_points / thrust_normalisation,
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


if __name__ == "__main__":
    main()
