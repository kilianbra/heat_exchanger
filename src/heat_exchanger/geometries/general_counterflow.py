"""
General counterflow heat exchanger models.

This module provides two approaches for modeling counterflow heat exchangers:
1. Simple model: Uses NTU-effectiveness method with iterative pressure drop
2. Compressible flow model: Uses Sturas 1971 equations for friction and heat transfer

References:
- Shah and Sekulic (2003) "Fundamentals of Heat Exchanger Design"
- Sturas (1971) NASA Technical Report
- Kays and London (1984) "Compact Heat Exchangers"
"""

from __future__ import annotations

import logging

import numpy as np

from heat_exchanger.conservation import update_static_properties as _upd_stat_prop
from heat_exchanger.correlations import general_hex_friction_factor, general_hex_j_factor
from heat_exchanger.epsilon_ntu import epsilon_ntu as _eps_ntu
from heat_exchanger.fluids.compressible_flow_friction_heat import (
    find_ksi_lim_adaptive as _find_ksi_lim,
)
from heat_exchanger.fluids.compressible_flow_friction_heat import (
    p_static_over_p_static_in,
    solve_M_from_ksi,
)
from heat_exchanger.fluids.protocols import FluidInputs

logger = logging.getLogger(__name__)

# ============================================================================
# SIMPLE MODEL FUNCTIONS (Incompressible / Low Mach)
# ============================================================================


def calculate_pressure_ratio(
    aq_over_ao,
    f,
    gd2,
    t_i_td,
    p_i_pd,
    eps,
    t,
    hot_fluid=True,
    c_h_c=1.0,
    max_iter=100,
    tol=0.001,
):
    """
    Calculate the pressure ratio p_out/p_in using iterative approach.

    Accounts for both friction losses and density changes due to heating.
    Uses the non-dimensional parameter g_d^2 = G^2 / (rho_d * p_d).

    Parameters
    ----------
    aq_over_ao : float or array
        Heat transfer area to free flow area ratio (A_q/A_o = 4*L/d_h)
    f : float or array
        Fanning friction factor
    gd2 : float or array
        Square of dimensionless mass flux: g_d^2 = G^2 / (rho_d * p_d)
    t_i_td : float
        Inlet temperature ratio: T_in / T_d (reference temperature)
    p_i_pd : float
        Inlet pressure ratio: p_in / p_d (reference pressure)
    eps : float or array
        Heat exchanger effectiveness
    t : float
        Temperature ratio T_h_in / T_c_in
    hot_fluid : bool
        Whether this is for the hot fluid (True) or cold fluid (False)
    c_h_c : float
        Capacity ratio C_hot/C_cold (default 1.0)
    max_iter : int
        Maximum number of iterations (default: 100)
    tol : float
        Convergence tolerance (default: 0.0001)

    Returns
    -------
    float or array
        Pressure ratio p_out / p_in
    """
    # Convert A_q/A_o to L/d_h: A_q/A_o = 4*L/d_h, so L/d_h = A_q/A_o / 4
    l_dh = aq_over_ao / 4.0

    # Initialize pressure ratio (start with no pressure drop)
    p_o_pi = np.ones_like(l_dh)

    # Calculate temperature ratio for a given effectiveness
    if hot_fluid:
        if c_h_c > 1:  # cold side is C_min
            t_o_ti = 1 - eps / c_h_c * (1 - 1 / t)
        else:
            t_o_ti = 1 - eps * (1 - 1 / t)
    else:  # cold fluid
        if c_h_c > 1:  # cold side is C_min
            t_o_ti = 1 + eps * (t - 1)
        else:
            t_o_ti = 1 + eps * c_h_c * (t - 1)

    # Iterate to find converged pressure ratio
    for _ in range(max_iter):
        # Store old value for convergence check
        p_o_pi_old = p_o_pi.copy()

        # Calculate new pressure ratio using the iteration formula
        # Based on momentum equation with friction and density change
        p_o_pi_new = 1 - gd2 * (1 / p_i_pd) ** 2 * t_i_td * (
            0.5 * f * 4 * l_dh * (1 + t_o_ti * 1 / p_o_pi) / 2 + (t_o_ti * 1 / p_o_pi - 1)
        )

        # Apply relaxation for stability
        p_o_pi = 0.5 * p_o_pi_old + 0.5 * p_o_pi_new

        # Check convergence
        if np.all(np.abs(p_o_pi - p_o_pi_old) < tol):
            break

    return p_o_pi


def rate_hex_simple(
    A_fr,
    A_q,
    f_in: FluidInputs,
    d_h_c=4e-3,
    sigma_r=1.0,
    sigma_w=1.0,
    t_over_dhc=0.02,
    ls_over_dh=5.0,
):
    """
    Rate a counterflow heat exchanger using the simple model.

    Parameters
    ----------
    A_fr : float
        Frontal area [m²]
    A_q : float
        Average heat transfer area (A_h + A_c) / 2 [m²]
    f_in : FluidInputs
        Fluid inputs containing fluid models, mass flow rates, and inlet conditions
    d_h_c : float
        Cold side hydraulic diameter [m] (default 4 mm) -> dh_h = d_h_c * sigma_r / sigma_w
    sigma_r : float
        Free flow area ratio A_o_h/A_o_c (default 1.0)
    sigma_w : float
        Heat transfer area ratio A_h/A_c (default 1.0)
    t_over_dhc : float
        Wall thickness to cold hydraulic diameter ratio (default 0.02)
    ls_over_dh : float
        Strip length to hydraulic diameter ratio for correlations

    Notes
    -----
    The physical flow length L is assumed to be the same for both hot and cold sides.
    This could be relaxed in future implementations to allow different flow lengths.

    Returns
    -------
    dict
        Dictionary containing results
    """
    # Extract values from FluidInputs
    fluid_hot = f_in.hot
    fluid_cold = f_in.cold
    mdot = f_in.m_dot_hot  # Assume same for both sides
    T_hot_in = f_in.Th_in
    P_hot_in = f_in.Ph_in if f_in.Ph_in is not None else f_in.Ph_out
    T_cold_in = f_in.Tc_in
    P_cold_in = f_in.Pc_in

    # Create fluid states at inlet conditions
    state_hot_in = fluid_hot.state(T=T_hot_in, P=P_hot_in)
    state_cold_in = fluid_cold.state(T=T_cold_in, P=P_cold_in)

    # Get fluid properties
    cp_hot = state_hot_in.cp
    cp_cold = state_cold_in.cp
    rho_hot_in = state_hot_in.rho
    rho_cold_in = state_cold_in.rho
    mu_hot_in = state_hot_in.mu
    mu_cold_in = state_cold_in.mu
    gamma_hot = state_hot_in.gamma
    gamma_cold = state_cold_in.gamma
    gm1og_hot = (gamma_hot - 1) / gamma_hot
    gm1og_cold = (gamma_cold - 1) / gamma_cold

    # Get model-level properties
    Pr_hot = fluid_hot.Pr
    Pr_cold = fluid_cold.Pr

    # Reference conditions
    T_d = 300.0  # K
    p_d = 1e5  # Pa

    # Calculate heat transfer areas from total A_q and sigma_w
    # A_q = A_h + A_c, and sigma_w = A_h / A_c
    # Solving: A_c = 2*A_q / (1+sigma_w), A_h = A_c * sigma_w
    A_c = 2 * A_q / (1 + sigma_w)
    A_h = A_c * sigma_w

    # Calculate hydraulic diameters: dh_h/dh_c = sigma_r / sigma_w
    d_h_h = d_h_c * sigma_r / sigma_w

    # Calculate free flow areas: A_fr = A_o_c * ( (1 + sigma_r) + 2*t/d_h * (1+sigma_w) )
    A_fr_over_Ao_c = (1 + sigma_r) + 2 * t_over_dhc * (1 + sigma_w)
    Ao_c = A_fr / A_fr_over_Ao_c
    Ao_h = Ao_c * sigma_r

    # Calculate mass flux G = mdot / A_o
    G_hot = mdot / Ao_h
    G_cold = mdot / Ao_c

    # Calculate Reynolds numbers: Re = G * d_h / mu
    Re_hot = G_hot * d_h_h / mu_hot_in
    Re_cold = G_cold * d_h_c / mu_cold_in

    # Calculate g² = (mdot/A_o)² / 4 / p_in / rho_in
    g2_hot = (mdot / Ao_h) ** 2 / 4 / P_hot_in / rho_hot_in
    g2_cold = (mdot / Ao_c) ** 2 / 4 / P_cold_in / rho_cold_in

    # Calculate friction factors and j-factors
    # Note: ls_over_dh uses the respective hydraulic diameter for each side
    ls_over_dh_hot = ls_over_dh
    ls_over_dh_cold = ls_over_dh
    f_hot = general_hex_friction_factor(Re_hot, ls_over_dh_hot)
    f_cold = general_hex_friction_factor(Re_cold, ls_over_dh_cold)
    j_hot = general_hex_j_factor(Re_hot, ls_over_dh_hot)
    j_cold = general_hex_j_factor(Re_cold, ls_over_dh_cold)

    # Calculate Stanton numbers: St = j * Pr^(-2/3)
    St_hot = j_hot * Pr_hot ** (-2 / 3)
    St_cold = j_cold * Pr_cold ** (-2 / 3)

    # Calculate A_q/A_o for each side: A_q/A_o = 4*L/d_h
    # Note: L is the same for both sides (physical flow length)
    # This could be relaxed in future implementations to allow different flow lengths
    A_q_over_Ao_h = A_h / Ao_h
    A_q_over_Ao_c = A_c / Ao_c

    # Capacity ratio
    C_hot = mdot * cp_hot
    C_cold = mdot * cp_cold
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    C_r = C_min / C_max
    C_h_c = C_hot / C_cold

    # NTU_i = h_i A_i / (mdot cp)_i with i = h,c
    # Hence NTU_i = St_i * A_i/A_oi

    NTU_h = St_hot * A_q_over_Ao_h
    NTU_c = St_cold * A_q_over_Ao_c

    NTU = 1 / (C_min / C_hot / NTU_h + C_min / C_cold / NTU_c)

    # Calculate effectiveness: eps = NTU/(1+NTU) for balanced, or counterflow formula
    if C_r < 0.99:
        eps = (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))
    else:
        eps = NTU / (1 + NTU)

    # Temperature and pressure ratios
    T_h_in_Td = T_hot_in / T_d
    T_c_in_Td = T_cold_in / T_d
    t = T_hot_in / T_cold_in
    p_h_in_pd = P_hot_in / p_d
    p_c_in_pd = P_cold_in / p_d

    # Calculate pressure ratios using A_q/A_o
    # Note: Physical flow length L is the same for both sides
    # This could be relaxed in future implementations
    P_hot_out_P_hot_in = calculate_pressure_ratio(
        A_q_over_Ao_h, f_hot, g2_hot, t_i_td=T_h_in_Td, p_i_pd=p_h_in_pd, eps=eps, t=t, hot_fluid=True, c_h_c=C_h_c
    )
    P_cold_out_P_cold_in = calculate_pressure_ratio(
        A_q_over_Ao_c, f_cold, g2_cold, t_i_td=T_c_in_Td, p_i_pd=p_c_in_pd, eps=eps, t=t, hot_fluid=False, c_h_c=C_h_c
    )

    # Calculate outlet temperatures
    if C_h_c > 1:  # cold side is C_min
        T_o_Ti_hot = 1 - eps / C_h_c * (1 - 1 / t)
        T_o_Ti_cold = 1 + eps * (t - 1)
    else:  # hot side is C_min
        T_o_Ti_hot = 1 - eps * (1 - 1 / t)
        T_o_Ti_cold = 1 + eps * C_h_c * (t - 1)
    T_hot_out = T_hot_in * T_o_Ti_hot
    T_cold_out = T_cold_in * T_o_Ti_cold

    # Calculate work potentials with different gamma values for hot and cold
    # This is Td delta s / cp/Td = delta s / cp
    log_p_hot = np.log(P_hot_out_P_hot_in)
    log_p_cold = np.log(P_cold_out_P_cold_in)
    dW_pot_Ex = (
        np.log(T_hot_out / T_hot_in) + np.log(T_cold_out / T_cold_in) - gm1og_hot * log_p_hot - gm1og_cold * log_p_cold
    )

    # Calculate Euergy change with different gamma for hot and cold

    dW_pot_Eu_hot = (
        (1 / p_h_in_pd) ** gm1og_hot * T_hot_in * (T_hot_out / T_hot_in * (1 / P_hot_out_P_hot_in) ** gm1og_hot - 1)
    )
    dW_pot_Eu_cold = (
        (1 / p_c_in_pd) ** gm1og_cold
        * T_cold_in
        * (T_cold_out / T_cold_in * (1 / P_cold_out_P_cold_in) ** gm1og_cold - 1)
    )
    dW_pot_Eu = dW_pot_Eu_hot + dW_pot_Eu_cold

    # Normalize by Q_max
    norm = 1 / (T_hot_in - T_cold_in)
    dW_pot_Ex_norm = dW_pot_Ex * norm
    dW_pot_Eu_norm = dW_pot_Eu * norm

    return {
        "eps": eps,
        "dp_hot": 1 - P_hot_out_P_hot_in,
        "dp_cold": 1 - P_cold_out_P_cold_in,
        "t_hot_out": T_hot_out,
        "t_cold_out": T_cold_out,
        "re_hot": Re_hot,
        "re_cold": Re_cold,
        "ntu": NTU,
        "g2_hot": g2_hot,
        "g2_cold": g2_cold,
        "Aq_over_Ao_c": A_q_over_Ao_c,
        "Aq_over_Ao_h": A_q_over_Ao_h,
        "dW_pot_Ex_norm": dW_pot_Ex_norm,
        "dW_pot_Eu_norm": dW_pot_Eu_norm,
    }


# ============================================================================
# COMPRESSIBLE FLOW MODEL (Sturas 1971)
# ============================================================================


def rate_hex_compressible_two_stream(
    A_fr,
    A_q,
    f_in: FluidInputs,
    d_h_c=4e-3,
    sigma_r=1.0,
    sigma_w=1.0,
    t_over_dhc=0.02,
    ls_over_dh=5.0,
    inlet_is_stagnation=True,
):
    """
    Rate a counterflow heat exchanger using compressible flow model for both streams.

    Two-step approach:
    1. First iteration uses inlet Reynolds number
    2. Second iteration uses average Reynolds number (viscosity varies with temperature)

    Process for each iteration:
    - Calculate heat transfer coefficients from j-factor correlations
    - Compute overall heat transfer coefficient and total Q
    - Calculate ksi = 4*f*L/d_h for each stream
    - Calculate k = Q / (ksi * mdot * cp * T_stag_in) for each stream
    - Solve for outlet Mach number using Sturas equations

    ASSUMPTION: Heat transfer is equally distributed along the length.
    This is not very accurate for C_r << 1, but provides a reasonable approximation.

    Parameters
    ----------
    A_fr : float
        Frontal area [m²]
    A_q : float
        Average heat transfer area (A_h + A_c) / 2 [m²]
    f_in : FluidInputs
        Fluid inputs containing fluid models, mass flow rates, and inlet conditions
    d_h_c : float
        Cold side hydraulic diameter [m] (default 4 mm) -> dh_h = d_h_c * sigma_r / sigma_w
    sigma_r : float
        Free flow area ratio A_o_h/A_o_c (default 1.0)
    sigma_w : float
        Heat transfer area ratio A_h/A_c (default 1.0)
    t_over_dhc : float
        Wall thickness to hydraulic diameter ratio (default 0.02)
    ls_over_dh : float
        Strip length to hydraulic diameter ratio
    inlet_is_stagnation : bool
        If True, inlet conditions are stagnation values (default True)

    Returns
    -------
    dict
        Dictionary containing results for both streams
    """
    # Extract values from FluidInputs
    fluid_hot = f_in.hot
    fluid_cold = f_in.cold
    mdot_hot = f_in.m_dot_hot
    mdot_cold = f_in.m_dot_cold
    T_hot_in = f_in.Th_in
    P_hot_in = f_in.Ph_in if f_in.Ph_in is not None else f_in.Ph_out
    T_cold_in = f_in.Tc_in
    P_cold_in = f_in.Pc_in

    # Calculate heat transfer areas from total A_q and sigma_w
    # A_q = A_h + A_c, and sigma_w = A_h / A_c
    # Solving: A_c = 2*A_q / (1+sigma_w), A_h = A_c * sigma_w
    A_q_c = 2 * A_q / (1 + sigma_w)
    A_q_h = A_q_c * sigma_w

    # Calculate hydraulic diameters: dh_h/dh_c = sigma_r / sigma_w
    d_h_h = d_h_c * sigma_r / sigma_w

    # Calculate free flow areas: A_fr = A_o_c * ( (1 + sigma_r) + 2*t/d_h * (1+sigma_w) )
    A_fr_over_Ao_c = (1 + sigma_r) + 2 * t_over_dhc * (1 + sigma_w)
    Ao_c = A_fr / A_fr_over_Ao_c
    Ao_h = Ao_c * sigma_r

    # Mass fluxes
    G_hot = mdot_hot / Ao_h
    G_cold = mdot_cold / Ao_c

    # =========================================================================
    # Calculate inlet conditions (simplified: assume rho ≈ rho_stag for low Mach)
    # =========================================================================
    if inlet_is_stagnation:
        # Hot side: stagnation inputs → derive static
        T_stag_hot_in = T_hot_in
        P_stag_hot_in = P_hot_in
        state_stag_hot = fluid_hot.state(T=T_stag_hot_in, P=P_stag_hot_in)
        cp_hot = state_stag_hot.cp
        gamma_hot = state_stag_hot.gamma
        V_hot = G_hot / state_stag_hot.rho  # Approximate: V ≈ G/rho_stag
        T_static_hot_in = T_stag_hot_in - V_hot**2 / (2 * cp_hot)
        P_static_hot_in = P_stag_hot_in * (T_static_hot_in / T_stag_hot_in) ** (gamma_hot / (gamma_hot - 1))
        state_hot_in = fluid_hot.state(T=T_static_hot_in, P=P_static_hot_in)
        M_hot_in = V_hot / state_hot_in.a

        # Cold side: stagnation inputs → derive static
        T_stag_cold_in = T_cold_in
        P_stag_cold_in = P_cold_in
        state_stag_cold = fluid_cold.state(T=T_stag_cold_in, P=P_stag_cold_in)
        cp_cold = state_stag_cold.cp
        gamma_cold = state_stag_cold.gamma
        V_cold = G_cold / state_stag_cold.rho
        T_static_cold_in = T_stag_cold_in - V_cold**2 / (2 * cp_cold)
        P_static_cold_in = P_stag_cold_in * (T_static_cold_in / T_stag_cold_in) ** (gamma_cold / (gamma_cold - 1))
        state_cold_in = fluid_cold.state(T=T_static_cold_in, P=P_static_cold_in)
        M_cold_in = V_cold / state_cold_in.a
    else:
        # Hot side: static inputs → derive stagnation
        T_static_hot_in = T_hot_in
        P_static_hot_in = P_hot_in
        state_hot_in = fluid_hot.state(T=T_static_hot_in, P=P_static_hot_in)
        cp_hot = state_hot_in.cp
        gamma_hot = state_hot_in.gamma
        V_hot = G_hot / state_hot_in.rho
        M_hot_in = V_hot / state_hot_in.a
        T_stag_hot_in = T_static_hot_in + V_hot**2 / (2 * cp_hot)
        P_stag_hot_in = P_static_hot_in * (T_stag_hot_in / T_static_hot_in) ** (gamma_hot / (gamma_hot - 1))

        # Cold side: static inputs → derive stagnation
        T_static_cold_in = T_cold_in
        P_static_cold_in = P_cold_in
        state_cold_in = fluid_cold.state(T=T_static_cold_in, P=P_static_cold_in)
        cp_cold = state_cold_in.cp
        gamma_cold = state_cold_in.gamma
        V_cold = G_cold / state_cold_in.rho
        M_cold_in = V_cold / state_cold_in.a
        T_stag_cold_in = T_static_cold_in + V_cold**2 / (2 * cp_cold)
        P_stag_cold_in = P_static_cold_in * (T_stag_cold_in / T_static_cold_in) ** (gamma_cold / (gamma_cold - 1))

    Pr_hot = fluid_hot.Pr
    Pr_cold = fluid_cold.Pr

    # Capacity rates
    C_hot = mdot_hot * cp_hot
    C_cold = mdot_cold * cp_cold
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    C_r = C_min / C_max

    # Maximum possible heat transfer
    Q_max = C_min * (T_stag_hot_in - T_stag_cold_in)

    # =========================================================================
    # TWO-STEP APPROACH: First with inlet Re, then with average Re
    # =========================================================================

    # Initialize with inlet temperature estimates for output
    T_stag_hot_out = T_stag_hot_in
    T_stag_cold_out = T_stag_cold_in

    for iteration in range(2):
        # Step 1 (iteration=0): Use inlet Reynolds number
        # Step 2 (iteration=1): Use average Reynolds number based on temperature

        if iteration == 0:
            # Use inlet viscosities
            mu_hot = state_hot_in.mu
            mu_cold = state_cold_in.mu
        else:
            # Use average temperature for viscosity (only viscosity changes due to temperature)
            T_avg_hot = 0.5 * (T_stag_hot_in + T_stag_hot_out)
            T_avg_cold = 0.5 * (T_stag_cold_in + T_stag_cold_out)
            # Get viscosity at average temperature (use inlet pressure as approximation)
            state_hot_avg = fluid_hot.state(T=T_avg_hot, P=P_static_hot_in)
            state_cold_avg = fluid_cold.state(T=T_avg_cold, P=P_static_cold_in)
            mu_hot = state_hot_avg.mu
            mu_cold = state_cold_avg.mu

        # Calculate Reynolds numbers: Re = G * d_h / mu
        Re_hot = G_hot * d_h_h / mu_hot
        Re_cold = G_cold * d_h_c / mu_cold

        # Calculate j-factors and friction factors (same Re for both j and f)
        j_hot = general_hex_j_factor(Re_hot, ls_over_dh)
        j_cold = general_hex_j_factor(Re_cold, ls_over_dh)
        f_hot = general_hex_friction_factor(Re_hot, ls_over_dh)
        f_cold = general_hex_friction_factor(Re_cold, ls_over_dh)

        # Calculate Stanton numbers: St = j * Pr^(-2/3)
        St_hot = j_hot * Pr_hot ** (-2 / 3)
        St_cold = j_cold * Pr_cold ** (-2 / 3)

        # Calculate NTU_i = St_i * A_q_i/A_oi
        NTU_h = St_hot * A_q_h / Ao_h
        NTU_c = St_cold * A_q_c / Ao_c
        NTU = 1 / (C_min / C_hot / NTU_h + C_min / C_cold / NTU_c)

        # Calculate effectiveness (counterflow)
        if C_r < 0.99:
            eps = (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))
        else:
            eps = NTU / (1 + NTU)

        # Calculate total heat transfer
        Q = eps * Q_max

        # Calculate ksi = f A_q_i / A_oi
        ksi_hot = f_hot * A_q_h / Ao_h
        ksi_cold = f_cold * A_q_c / Ao_c

        # =====================================================================
        # ASSUMPTION: Heat transfer is equally distributed along the length.
        # This is not very accurate for C_r << 1 but will do for now.
        # =====================================================================

        # Calculate k = delta_T_stag / (T_stag_in * ksi) for each stream
        # delta_T_stag = Q / (mdot * cp)
        # For hot side: Q is removed (negative delta_T_stag)
        # For cold side: Q is added (positive delta_T_stag)

        delta_T_stag_hot = -Q / (mdot_hot * cp_hot)  # Hot side loses heat
        delta_T_stag_cold = Q / (mdot_cold * cp_cold)  # Cold side gains heat

        if ksi_hot > 1e-6:
            k_hot = delta_T_stag_hot / (T_stag_hot_in * ksi_hot)
        else:
            k_hot = 0.0

        if ksi_cold > 1e-6:
            k_cold = delta_T_stag_cold / (T_stag_cold_in * ksi_cold)
        else:
            k_cold = 0.0

        # Solve for outlet Mach numbers using Sturas equations
        choked_hot = False
        choked_cold = False

        try:
            _, M_hot_out = solve_M_from_ksi(ksi_hot, M_hot_in, k_hot, gamma=gamma_hot)
        except (ValueError, RuntimeError):
            M_hot_out = np.nan
            choked_hot = True

        try:
            _, M_cold_out = solve_M_from_ksi(ksi_cold, M_cold_in, k_cold, gamma=gamma_cold)
        except (ValueError, RuntimeError):
            M_cold_out = np.nan
            choked_cold = True

        T_stag_hot_out = T_stag_hot_in + delta_T_stag_hot
        T_stag_cold_out = T_stag_cold_in + delta_T_stag_cold

    # Calculate outlet static temperatures
    if not choked_hot and not np.isnan(M_hot_out):
        ratio_out_hot = 1 + (gamma_hot - 1) / 2 * M_hot_out**2
        T_static_hot_out = T_stag_hot_out / ratio_out_hot
        # Calculate static pressure ratio using Sturas equation
        p_ratio_hot = p_static_over_p_static_in(M_hot_in, M_hot_out, k_hot, ksi_hot, gamma_hot)
        P_static_hot_out = P_static_hot_in * p_ratio_hot
        P_stag_hot_out = P_static_hot_out * ratio_out_hot ** (gamma_hot / (gamma_hot - 1))
    else:
        T_static_hot_out = np.nan
        P_static_hot_out = np.nan
        p_ratio_hot = np.nan

    if not choked_cold and not np.isnan(M_cold_out):
        ratio_out_cold = 1 + (gamma_cold - 1) / 2 * M_cold_out**2
        T_static_cold_out = T_stag_cold_out / ratio_out_cold
        # Calculate static pressure ratio using Sturas equation
        p_ratio_cold = p_static_over_p_static_in(M_cold_in, M_cold_out, k_cold, ksi_cold, gamma_cold)
        P_static_cold_out = P_static_cold_in * p_ratio_cold
        P_stag_cold_out = P_static_cold_out * ratio_out_cold ** (gamma_cold / (gamma_cold - 1))
    else:
        T_static_cold_out = np.nan
        P_static_cold_out = np.nan
        p_ratio_cold = np.nan

    dp_hot_friction = gamma_hot * (M_hot_in**2 * A_q_h / Ao_h / 2 * f_hot)
    dp_hot_heat = (
        gamma_hot / (gamma_hot - 1) * (delta_T_stag_hot / T_static_hot_in - np.log(T_static_hot_out / T_static_hot_in))
    )

    # Build result dictionaries
    result_hot = {
        "mach_in": M_hot_in,
        "mach_out": M_hot_out,
        "t_stag_in": T_stag_hot_in,
        "t_stag_out": T_stag_hot_out,
        "t_static_in": T_static_hot_in,
        "t_static_out": T_static_hot_out,
        "p_stag_in": P_stag_hot_in,
        "p_static_in": P_static_hot_in,
        "p_static_out": P_static_hot_out,
        "p_static_out_p_static_in": p_ratio_hot,
        "ksi": ksi_hot,
        "k": k_hot,
        "re": Re_hot,
        "choked": choked_hot,
    }

    result_cold = {
        "mach_in": M_cold_in,
        "mach_out": M_cold_out,
        "t_stag_in": T_stag_cold_in,
        "t_stag_out": T_stag_cold_out,
        "t_static_in": T_static_cold_in,
        "t_static_out": T_static_cold_out,
        "p_stag_in": P_stag_cold_in,
        "p_static_in": P_static_cold_in,
        "p_static_out": P_static_cold_out,
        "p_static_out_p_static_in": p_ratio_cold,
        "ksi": ksi_cold,
        "k": k_cold,
        "re": Re_cold,
        "choked": choked_cold,
    }

    return {
        "hot": result_hot,
        "cold": result_cold,
        "eps": eps,
        "dp_hot": 1 - p_ratio_hot,
        "dp_hot_friction": dp_hot_friction,
        "dp_hot_heat": dp_hot_heat,
        "dp_hot_stag": 1 - P_stag_hot_out / P_stag_hot_in,
        "dp_cold": 1 - p_ratio_cold,
        "dp_cold_stag": 1 - P_stag_cold_out / P_stag_cold_in,
        "t_hot_out": T_static_hot_out,
        "t_cold_out": T_static_cold_out,
        "t_stag_hot_out": T_stag_hot_out,
        "t_stag_cold_out": T_stag_cold_out,
        "re_hot": Re_hot,
        "re_cold": Re_cold,
        "ntu": NTU,
        "g2_hot": G_hot**2 / P_static_hot_in / state_hot_in.rho / 4,
        "g2_cold": G_cold**2 / P_static_cold_in / state_cold_in.rho / 4,
        "q": Q,
        "c_r": C_r,
        "Aq_over_Ao_c": A_q_c / Ao_c,
        "Aq_over_Ao_h": A_q_h / Ao_h,
    }


def xflow_guess_0d(
    geom,
    f_in: FluidInputs,
) -> tuple[float, float]:
    """Return (Th_inner_guess, Ph_other_guess) using a two-step 0D estimate.
    Assumes that the Radial Spiral Geometry is close enough to a counterflow configuration.
    This tends to be off by only a few percentage points

    The first step evaluates properties at the inlet conditions; the second step
    re-evaluates at the mean of the inlet and the first-step outlet to refine the
    guess.
    """

    logger = logging.getLogger(__name__ + ".xflow_guess_0d")

    # Validate boundary pressure specification
    if (f_in.Ph_in is None and f_in.Ph_out is None) or (f_in.Ph_in is not None and f_in.Ph_out is not None):
        raise ValueError("Specify exactly one of Ph_in or Ph_out in FluidInputs.")

    A_q_c = 2 * geom["A_q"] / (1 + geom["sigma_w"])
    A_q_h = A_q_c * geom["sigma_w"]
    d_h_c = geom["d_h_c"]
    d_h_h = geom["d_h_c"] * geom["sigma_r"] / geom["sigma_w"]
    Ao_c = geom["A_fr"] / (1 + geom["sigma_r"]) + 2 * geom["t_over_dhc"] * (1 + geom["sigma_w"])
    Ao_h = Ao_c * geom["sigma_r"]
    ls_over_dh = geom["ls_over_dh"]

    A_total_hot0 = A_q_h
    A_total_cold0 = A_q_c

    # Frontal/free areas at mid-radius and total cold frontal area (per sector/header)
    Afr_hot_mid = Ao_h
    Aff_hot_mid = Afr_hot_mid * 1

    Aff_cold_total = Ao_c

    G_h0 = f_in.m_dot_hot / Aff_hot_mid
    G_c0 = f_in.m_dot_cold / Aff_cold_total

    def _0d_xflow_guess(
        _Th_in: float,
        _Ph_in: float,
        _Tc_in: float,
        _Pc_in: float,
        Th_eval: float,
        Ph_eval: float,
        Tc_eval: float,
        Pc_eval: float,
        _Ph_out: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Single 0D estimate using property evaluation at (eval) and inlets (b).
        Approximates the heat exchanger as counterflow
        If _Ph_out is specified, then any input at _Ph_in is ignored. The inlet pressure
        is then calculated and returned as Ph_not_b.
        If no _Ph_out is specified, then like with the other three, _Ph_in is used
        and the exit pressure is returned as Ph_not_b."""
        sh = f_in.hot.state(Th_eval, Ph_eval)
        sc = f_in.cold.state(Tc_eval, Pc_eval)
        Pr_h = sh.mu * sh.cp / sh.k
        Pr_c = sc.mu * sc.cp / sc.k
        Re_h = G_h0 * d_h_h / sh.mu
        Re_c = G_c0 * d_h_c / sc.mu

        j_hot = general_hex_j_factor(Re_h, ls_over_dh)
        j_cold = general_hex_j_factor(Re_c, ls_over_dh)
        f_hot = general_hex_friction_factor(Re_h, ls_over_dh)
        f_cold = general_hex_friction_factor(Re_c, ls_over_dh)

        St_h = j_hot * Pr_h ** (-2 / 3)
        St_c = j_cold * Pr_c ** (-2 / 3)
        logger.info(
            "0D guess tube bank for Re_h=%5.2e: St_h=%5.2f, f_h=%5.2e",
            Re_h,
            St_h,
            f_hot,
        )
        St_c = j_cold * Pr_c ** (-2 / 3)
        logger.info(
            "0D guess tube flow for Re_c=%5.2e: St_c=%5.2f, f_c=%5.2e",
            Re_c,
            St_c,
            f_cold,
        )

        h_h = St_h * G_h0 * sh.cp
        h_c = St_c * G_c0 * sc.cp

        wall_term = 0
        U_h = 1.0 / (1.0 / h_h + 1.0 / h_c * (A_q_h / A_q_c) + wall_term)

        C_h_tot = f_in.m_dot_hot * sh.cp
        C_c_tot = f_in.m_dot_cold * sc.cp
        C_min = min(C_h_tot, C_c_tot)
        C_max = max(C_h_tot, C_c_tot)
        Cr = C_min / C_max

        NTU = U_h * A_total_hot0 / C_min
        eps = _eps_ntu(NTU, Cr, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)
        logger.info("0D guess epsilon-NTU: NTU=%5.2f, eps=%5.2f", NTU, eps)
        Q = eps * C_min * (_Th_in - _Tc_in)

        tau_h = f_hot * (A_total_hot0 / Aff_hot_mid) * (G_h0**2) / (2.0 * sh.rho)
        tau_c = f_cold * (A_total_cold0 / Aff_cold_total) * (G_c0**2) / (2.0 * sc.rho)

        dh0_h = -Q / f_in.m_dot_hot
        dh0_c = Q / f_in.m_dot_cold

        ksi_h = f_hot * (A_total_hot0 / Aff_hot_mid)
        ksi_c = f_cold * (A_total_cold0 / Aff_cold_total)
        k_h = dh0_h / (sh.h * ksi_h)
        k_c = dh0_c / (sc.h * ksi_c)
        M_in_h = G_h0 / sh.rho / sh.a
        M_in_c = G_c0 / sc.rho / sc.a

        # Check for choking limit
        if abs(k_h) > 1e-10:  # Avoid division by zero
            ksi_lim_h, _ = _find_ksi_lim(M_in_h, k_h, gamma=sh.gamma)
            if not np.isnan(ksi_lim_h) and ksi_h > ksi_lim_h:
                logger.info(
                    "Hot fluid choking risk: ksi_h=%.1e > ksi_lim_h=%.1e (M_in=%.2f, k=%.3f)",
                    ksi_h,
                    ksi_lim_h,
                    M_in_h,
                    k_h,
                )
        if abs(k_c) > 1e-10:  # Avoid division by zero
            ksi_lim_c, _ = _find_ksi_lim(M_in_c, k_c, gamma=sc.gamma)
            if not np.isnan(ksi_lim_c) and ksi_c > ksi_lim_c:
                logger.info(
                    "Cold fluid choking risk: ksi_c=%.1e > ksi_lim_c=%.1e (M_in=%.2f, k=%.3f)",
                    ksi_c,
                    ksi_lim_c,
                    M_in_c,
                    k_c,
                )

        Th_out, Ph_not_b = _upd_stat_prop(
            f_in.hot,
            G_h0,
            dh0_h,
            tau_h,
            T_a=_Th_in,
            p_b=_Ph_out if _Ph_out is not None else _Ph_in,
            a_is_in=True,
            b_is_in=(_Ph_out is None),
            max_iter=100,
            tol_T=1e-2,
            rel_tol_p=1e-2,
        )

        Tc_out, Pc_out = _upd_stat_prop(
            f_in.cold,
            G_c0,
            dh0_c,
            tau_c,
            T_a=_Tc_in,
            p_b=_Pc_in,
            a_is_in=True,
            b_is_in=True,
            max_iter=100,
            tol_T=1e-2,
            rel_tol_p=1e-2,
        )
        if Pc_out < 0:
            logger.warning(f"Pc_out {Pc_out:.1e} <0, for {tau_c:.1e} setting to 0.1e5 Pa")
            Pc_out = 0.1e5

        return Th_out, Tc_out, Ph_not_b, Pc_out

    logger.info(
        "0D guess inputs: Th_in =%5.2f K, Tc_in =%5.2f K, Ph_in =%s Pa, Pc_in =%5.2e Pa (Ph_out=%s)",
        f_in.Th_in,
        f_in.Tc_in,
        f"{f_in.Ph_in:.2e}" if f_in.Ph_in is not None else "N/A",
        f_in.Pc_in,
        f"{f_in.Ph_out:.2e}" if f_in.Ph_out is not None else "N/A",
    )
    Th_o1, Tc_o1, Ph_not_b1, Pc_o1 = _0d_xflow_guess(
        _Th_in=f_in.Th_in,
        _Ph_in=f_in.Ph_in if f_in.Ph_in is not None else float("nan"),
        _Tc_in=f_in.Tc_in,
        _Pc_in=f_in.Pc_in,
        Th_eval=f_in.Th_in,
        Ph_eval=f_in.Ph_out if f_in.Ph_out is not None else f_in.Ph_in,
        Tc_eval=f_in.Tc_in,
        Pc_eval=f_in.Pc_in,
        _Ph_out=f_in.Ph_out,
    )
    if f_in.Ph_out is not None:
        logger.info(
            "0D guess 1: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa (Ph_in_guess=%5.2e Pa)",
            Th_o1,
            Tc_o1,
            f_in.Ph_out,
            Pc_o1,
            Ph_not_b1,
        )
    else:
        logger.info(
            "0D guess 1: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa",
            Th_o1,
            Tc_o1,
            Ph_not_b1,
            Pc_o1,
        )
    Th_mean = 0.5 * (f_in.Th_in + Th_o1)
    Tc_mean = 0.5 * (f_in.Tc_in + Tc_o1)
    Ph_known = f_in.Ph_out if f_in.Ph_out is not None else f_in.Ph_in
    if Ph_known is None:
        raise ValueError("Either Ph_in or Ph_out must be provided for the 0D guess.")
    Ph_mean = 0.5 * (Ph_known + Ph_not_b1)
    Pc_mean = 0.5 * (f_in.Pc_in + Pc_o1)
    Th_o2, Tc_o2, Ph_not_b2, Pc_o2 = _0d_xflow_guess(
        _Th_in=f_in.Th_in,
        _Ph_in=f_in.Ph_in if f_in.Ph_in is not None else float("nan"),
        _Tc_in=f_in.Tc_in,
        _Pc_in=f_in.Pc_in,
        Th_eval=Th_mean,
        Ph_eval=Ph_mean,
        Tc_eval=Tc_mean,
        Pc_eval=Pc_mean,
        _Ph_out=f_in.Ph_out,
    )
    if f_in.Ph_out is not None:
        logger.info(
            "0D guess 2: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa (Ph_in_guess=%5.2e Pa)",
            Th_o2,
            Tc_o2,
            f_in.Ph_out,
            Pc_o2,
            Ph_not_b2,
        )
    else:
        logger.info(
            "0D guess 2: Th_out=%5.2f K, Tc_out=%5.2f K, Ph_out=%5.2e Pa, Pc_out=%5.2e Pa",
            Th_o2,
            Tc_o2,
            Ph_not_b2,
            Pc_o2,
        )
    # Hot inner boundary (inboard shoot) guess equals the 0D outlet
    return float(Th_o2), float(Tc_o2), float(Ph_not_b2), float(Pc_o2)
