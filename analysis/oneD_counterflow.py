"""
counterflow_simulation_strategy.py

This module implements a counterflow simulation strategy for a heat exchanger.
In a counterflow configuration, the hot fluid enters at x = 0 and the cold fluid
enters at x = L. The simulation first generates an initial guess using an epsilon–NTU
approach (to estimate exit temperatures), then assumes linear temperature profiles.
An iterative procedure then updates the temperature and pressure in each cell until
convergence is achieved.

The algorithm proceeds as follows:
  1. Calculate local transport properties at the inlet conditions.
  2. Compute the overall heat transfer coefficient (U) at the inlet.
  3. Estimate the NTU and effectiveness using the standard counterflow ε–NTU formula.
  4. Determine exit temperatures for hot (at x = L) and cold (at x = 0) streams.
  5. Initialize temperature profiles:
       - Hot: linear from hot inlet (x=0) to hot outlet (x=L)
       - Cold: linear from cold outlet (x=0) to cold inlet (x=L)
  6. Assume constant pressures initially.
  7. Iterate:
       - For the hot fluid, march forward from x=0 to x=L (updating cell i+1 from cell i).
       - For the cold fluid, march backward from x=L to x=0 (updating cell j-1 from cell j).
       - Use the energy_balance_segment() and momentum_balance_segment() functions.
       - Use relaxation and check convergence.

Returns:
  A pandas DataFrame with columns:
    - 'x': position along the exchanger [m]
    - 'T_hot': hot fluid temperature [K]
    - 'T_cold': cold fluid temperature [K]
    - 'P_hot': hot fluid pressure [Pa]
    - 'P_cold': cold fluid pressure [Pa]
    - 'Re_hot': hot fluid Reynolds number
    - 'Re_cold': cold fluid Reynolds number
    - 'M_hot': hot fluid Mach number
    - 'M_cold': cold fluid Mach number

"""
import numpy as np
import pandas as pd
from math import pi, exp
from hex_model.correlations import (circular_pipe_nusselt,
                                   circular_pipe_friction_factor)
from hex_model.conservation import (update_static_properties,
                                   energy_balance_segment,
                                   momentum_balance_segment)
from hex_model.simulation_strategy import SimulationStrategy
from typing import Tuple, Dict
from geometries.base_geometry import BaseGeometry

import pdb #python debugger -> break with pdb.set_trace()


class CounterFlowSimulationStrategy(SimulationStrategy):
    """
    Counterflow heat exchanger simulation.
    Allows an optional single iteration of ε–NTU
    to refine the overall heat transfer coefficient using mean temperatures.

    Key flags/inputs:
      - boundary_conditions: dict specifying boundary T and p for hot & cold.
      - use_eNTU_iteration: bool to decide if we do one iteration or not.
    """
    def __init__(self, use_eNTU_iteration=False):
        self.use_eNTU_iteration = use_eNTU_iteration

    def simulate(self, exchanger) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulate counterflow heat exchanger with improved tracking of dissipation and performance.

        Returns:
            Tuple containing:
            1. Dictionary with DataFrames:
            - 'nodes': DataFrame with node values (N+1 points)
            - 'segments': DataFrame with segment values (N points)
            2. Dictionary with global performance metrics including:
            - Heat transfer metrics (Q_total, Q_max, effectiveness)
            - NTU metrics (NTU_1D, NTU_h_total, NTU_c_total)
            - Pressure drops (dP_hot_pct, dP_cold_pct)
            - Viscous dissipation totals
            - Entropy generation metrics
        """
        geom = exchanger.geometry
        hot = exchanger.hot_fluid
        cold = exchanger.cold_fluid

        N = geom.num_segments
        dx = geom.dx

        # Basic geometry data
        d_h_hot = geom.d_h_hot
        d_h_cold = geom.d_h_cold

        # Mass velocities
        G_h = hot.m_dot / geom.A_q_h
        G_c = cold.m_dot / geom.A_q_c

        # --- 0D eps-NTU calculation (two passes) ---
        # For clarity, define local "current" hot/cold inlet/outlet temperatures
        # that get updated each pass.
        # Assert that boundary temperatures are specified at inlets
        if not hot.T_boundary_at_inlet or not cold.T_boundary_at_inlet:
            raise NotImplementedError("Only inlet temperature boundaries are currently supported")
        T_hot_in = hot.boundary_T
        T_cold_in = cold.boundary_T

        # In second pass, not necessarily equal to inlet temperatures
        Th_property_eval = T_hot_in
        Tc_property_eval = T_cold_in
        Ph_property_eval = hot.boundary_P
        Pc_property_eval = cold.boundary_P


        for eNTU_iter in range(2):
            # 1) Get transport properties at the "inlet" temperatures (and the boundary pressures - which could be at exit).
            rho_h, cp_h, mu_h, k_h = hot.properties.get_transport_properties(
                Th_property_eval, Ph_property_eval)
            rho_c, cp_c, mu_c, k_c = cold.properties.get_transport_properties(
                Tc_property_eval, Pc_property_eval)

            # 2) Compute dimensionless groups (Re, Pr)
            Re_h = geom.calculate_reynolds_number(hot.m_dot, d_h_hot, geom.A_q_h, mu_h)
            Re_c = geom.calculate_reynolds_number(cold.m_dot, d_h_cold, geom.A_q_c, mu_c)

            Pr_h = cp_h * mu_h / k_h if k_h > 0 else 0.0
            Pr_c = cp_c * mu_c / k_c if k_c > 0 else 0.0

            # 3) Nusselt and convective coefficients
            Nu_h = geom.calculate_nusselt_number(Re_h, Pr_h, True)  # True for hot fluid
            Nu_c = geom.calculate_nusselt_number(Re_c, Pr_c, False)  # False for cold fluid

            h_h = Nu_h * k_h / d_h_hot
            h_c = Nu_c * k_c / d_h_cold

            # Calculate fin efficiency and overall surface efficiency
            # Get wall thermal conductivity (default to a reasonable value if not provided)
            lambda_wall = getattr(geom, 'lambda_wall', 237.0)  # Default to aluminum if not specified

            # Calculate fin efficiency for both sides
            eta_f_h = geom.fin_efficiency_hot(h_h, lambda_wall)
            eta_f_c = geom.fin_efficiency_cold(h_c, lambda_wall)

            # Calculate fin area ratio for both sides
            # For unfinned geometries, these will be 0
            A_fin_h = getattr(geom, 'A_fin_hot', 0.0)
            A_fin_c = getattr(geom, 'A_fin_cold', 0.0)
            A_q_h = geom.A_q_h
            A_q_c = geom.A_q_c

            A_fin_A_q_h = A_fin_h / A_q_h if A_q_h > 0 else 0.0
            A_fin_A_q_c = A_fin_c / A_q_c if A_q_c > 0 else 0.0

            # Calculate overall surface efficiency
            eta_o_h = 1.0 - A_fin_A_q_h * (1.0 - eta_f_h)
            eta_o_c = 1.0 - A_fin_A_q_c * (1.0 - eta_f_c)

            # 4) Overall U (neglecting wall resistance, etc.)
            U_base = 1.0 / ((1.0/(h_h * eta_o_h)) + (1.0/(h_c * eta_o_c))) if (h_h > 0 and h_c > 0) else 0.0
            A_total = geom.A_q_h

            # 5) Heat capacity rates
            C_hot = hot.m_dot  * cp_h
            C_cold = cold.m_dot * cp_c
            C_min = min(C_hot, C_cold)
            C_max = max(C_hot, C_cold)
            Cr = C_min / C_max if C_max > 0 else 0.0

            # 6) NTU and effectiveness (counterflow)
            NTU = (U_base * A_total / C_min) if C_min > 0 else 0.0

            if abs(1.0 - Cr) < 1e-6:
                # Cr ~ 1
                effectiveness = NTU / (1.0 + NTU)
            else:
                effectiveness = (1.0 - np.exp(-NTU * (1.0 - Cr))) / \
                                (1.0 - Cr * np.exp(-NTU * (1.0 - Cr)))

            # 7) Compute Q_max and actual Q_0D
            Q_max = C_min * (T_hot_in - T_cold_in)
            Q_0D = effectiveness * Q_max

            # 8) Calculate friction factors and impulse function changes
            f_h = geom.calculate_friction_factor(Re_h, True)  # True for hot fluid
            f_c = geom.calculate_friction_factor(Re_c, False)  # False for cold fluid

            dFA_h = momentum_balance_segment(G=G_h, rho_mean=rho_h, D_h=d_h_hot, length=geom.length, f=f_h)
            dFA_c = momentum_balance_segment(G=G_c, rho_mean=rho_c, D_h=d_h_cold, length=geom.length, f=f_c)

            # Calculate outlet conditions using energy and momentum balances
            T_hot_out_0D, P_hot_not_boundary_0D, _ = update_static_properties(
                hot.properties, G_h, -Q_0D/hot.m_dot, dFA_h,
                T_hot_in, hot.boundary_P, rho_h,
                reverse_pressure_marching=not hot.P_boundary_at_inlet
            )

            T_cold_out_0D, P_cold_not_boundary_0D, _ = update_static_properties(
                cold.properties, G_c, Q_0D/cold.m_dot, dFA_c,
                T_cold_in, cold.boundary_P, rho_c,
                reverse_pressure_marching=not cold.P_boundary_at_inlet
            )

            # For the second pass, refine using mean temperatures and pressures
            if eNTU_iter == 0:
                Th_property_eval = 0.5*(T_hot_in + T_hot_out_0D)
                Tc_property_eval = 0.5*(T_cold_in + T_cold_out_0D)
                Ph_property_eval = 0.5*(hot.boundary_P + P_hot_not_boundary_0D)
                Pc_property_eval = 0.5*(cold.boundary_P + P_cold_not_boundary_0D)


        # After the loop, Th_property_eval and Tc_property_eval hold the final 0D results of the second pass
        # This is not necessarily the average of the inlet and the second run outlet temperatures (could do third pass)

        # Create arrays for x and for storing results
        x = np.linspace(0, geom.length, N+1)
        # Fill initial T profiles linearly
        T_hot = np.linspace(T_hot_in, T_hot_out_0D, N+1)
        T_cold = np.linspace(T_cold_out_0D, T_cold_in, N+1)

        has_port_drops = hasattr(geom, 'port_mdot_multiplier') and hasattr(geom, 'port_diameter_cold_in') and hasattr(geom, 'port_diameter_cold_out')


        # Fill initial P profiles linearly
        if hot.P_boundary_at_inlet:
            P_hot = np.linspace(hot.boundary_P, P_hot_not_boundary_0D, N+1)
        else:
            P_hot = np.linspace(P_hot_not_boundary_0D, hot.boundary_P, N+1)
        if cold.P_boundary_at_inlet:
            P_cold = np.linspace(P_cold_not_boundary_0D, cold.boundary_P, N+1)
            if has_port_drops:
                rho_cold_in = cold.properties.get_density(T_cold_in, cold.boundary_P)
                rho_cold_out = cold.properties.get_density(T_cold_out_0D, P_cold_not_boundary_0D)

        else:
            P_cold = np.linspace(cold.boundary_P, P_cold_not_boundary_0D, N+1)


        # If N=1, we just skip any e–NTU iteration or further 1D stepping:
        if N == 1:

            # Re_h and M_h were calculated at segment level above - assume constant for nodes
            Re_hot_list = [Re_h, Re_h]
            Re_cold_list = [Re_c, Re_c]
            M_hot_list = [np.nan, np.nan]
            M_cold_list = [np.nan, np.nan]

        
            return results

        #region Initialisation of arrays
        # Initialize convergence tracking
        max_iterations = 200
        iteration = 0
        converged = False

        # Add storage for intermediate temperature profiles and convergence metrics

        # Initialize relaxation parameters
        relax = 0.2  # Initial relaxation factor
        relax_min = 0.05  # Minimum relaxation factor
        relax_max = 0.5   # Maximum relaxation factor
        oscillation_threshold = 1.0  # Threshold for detecting oscillations
        prev_diffs = None  # Store previous differences for oscillation detection




        while not converged and iteration < max_iterations:

            # Store previous iteration values
            T_hot_prev = T_hot.copy()
            T_cold_prev = T_cold.copy()
            P_hot_prev = P_hot.copy()
            P_cold_prev = P_cold.copy()

            # Store current temperature profiles
            T_hot_history.append(T_hot.copy())
            T_cold_history.append(T_cold.copy())
            iteration_numbers.append(iteration)

            # Initialize boundary conditions
            T_hot[0] = hot.boundary_T
            if hot.P_boundary_at_inlet:
                P_hot[0] = hot.boundary_P

            # Forward march for hot fluid and property calculations
            # Marching for each of the N segments, averaging properties of the N+1 nodes
            for i in range(0, N):
                # Calculate segment average properties
                T_hot_seg[i] = (T_hot_prev[i] + T_hot_prev[i+1])/2
                T_cold_seg[i] = (T_cold_prev[i] + T_cold_prev[i+1])/2
                P_hot_seg[i] = (P_hot_prev[i] + P_hot_prev[i+1])/2
                P_cold_seg[i] = (P_cold_prev[i] + P_cold_prev[i+1])/2

                # Get fluid properties in segment i
                rho_h_seg[i], cp_h_i, mu_h_i, k_h_i = hot.properties.get_transport_properties(T_hot_seg[i], P_hot_seg[i])
                rho_c_seg[i], cp_c_i, mu_c_i, k_c_i = cold.properties.get_transport_properties(T_cold_seg[i], P_cold_seg[i])

                M_hot_list[i] = G_h / rho_h_seg[i] / hot.properties.get_speed_of_sound(T_hot_seg[i], P_hot_seg[i])
                M_cold_list[i] = G_c / rho_c_seg[i] / cold.properties.get_speed_of_sound(T_cold_seg[i], P_cold_seg[i])

                # Calculate Reynolds, Nusselt, and friction factors
                Re_hot_list[i] = geom.calculate_reynolds_number(hot.m_dot, d_h_hot, geom.A_q_h, mu_h_i)
                Re_cold_list[i] = geom.calculate_reynolds_number(cold.m_dot, d_h_cold, geom.A_q_c, mu_c_i)

                Pr_h_i = mu_h_i * cp_h_i / k_h_i if k_h_i > 0.0 else 0.0
                Pr_c_i = mu_c_i * cp_c_i / k_c_i if k_c_i > 0.0 else 0.0

                Nu_h_i = geom.calculate_nusselt_number(Re_hot_list[i], Pr_h_i, True)  # True for hot fluid
                Nu_c_i = geom.calculate_nusselt_number(Re_cold_list[i], Pr_c_i, False)  # False for cold fluid

                f_h_i = geom.calculate_friction_factor(Re_hot_list[i], True)  # True for hot fluid
                f_c_i = geom.calculate_friction_factor(Re_cold_list[i], False)  # False for cold fluid

                h_h_i = Nu_h_i * k_h_i / d_h_hot
                h_c_i = Nu_c_i * k_c_i / d_h_cold

                # Calculate fin efficiency and overall surface efficiency
                # Get wall thermal conductivity (default to a reasonable value if not provided)
                lambda_wall = getattr(geom, 'lambda_wall', 237.0)  # Default to aluminum if not specified

                # Calculate fin efficiency for both sides
                eta_f_h = geom.fin_efficiency_hot(h_h_i, lambda_wall)
                eta_f_c = geom.fin_efficiency_cold(h_c_i, lambda_wall)

                # Calculate fin area ratio for both sides
                # For unfinned geometries, these will be 0
                A_fin_h = getattr(geom, 'A_fin_hot', 0.0)
                A_fin_c = getattr(geom, 'A_fin_cold', 0.0)
                A_q_h = geom.A_q_h
                A_q_c = geom.A_q_c

                A_fin_A_q_h = A_fin_h / A_q_h if A_q_h > 0 else 0.0
                A_fin_A_q_c = A_fin_c / A_q_c if A_q_c > 0 else 0.0

                # Calculate overall surface efficiency
                eta_o_h = 1.0 - A_fin_A_q_h * (1.0 - eta_f_h)
                eta_o_c = 1.0 - A_fin_A_q_c * (1.0 - eta_f_c)

                A_seg_hot = geom.get_heat_transfer_segment_area_hot()

                U_i = 1.0 / ((1.0/(h_h_i * eta_o_h)) + (A_q_h / (h_c_i * eta_o_c*A_q_c))) if (h_h_i > 0.0 and h_c_i > 0.0) else 0.0

                # Calculate and store energy/momentum changes
                dh0_hot, dh0_cold, Q_seg_list[i] = energy_balance_segment(m_dot_hot=hot.m_dot, m_dot_cold=cold.m_dot,
                    T_hot=T_hot_seg[i], T_cold=T_cold_seg[i], U=U_i, area=A_seg_hot)


                dFA_h = momentum_balance_segment(G=hot.m_dot/geom.A_q_h, rho_mean=rho_h_seg[i],
                    D_h=d_h_hot, length=dx, f=f_h_i)
                dFA_c = momentum_balance_segment(G=cold.m_dot/geom.A_q_c, rho_mean=rho_c_seg[i],
                    D_h=d_h_cold, length=dx, f=f_c_i)

                dh0_hot_list[i] = dh0_hot
                dh0_cold_list[i] = dh0_cold
                dFA_hot_list[i] = dFA_h
                dFA_cold_list[i] = dFA_c

                # Update hot fluid properties
                # if P specified at inlet, then can use P_hot of this iteration, it not use that of previous iteration
                T_hot[i+1], P_new, _ = update_static_properties(hot.properties,
                    hot.m_dot/geom.A_q_h, dh0_hot, dFA_h, T_hot[i],
                    P_hot[i] if hot.P_boundary_at_inlet else P_hot_prev[i],
                    hot.properties.get_density(T_hot[i], P_hot[i] if hot.P_boundary_at_inlet else P_hot_prev[i]), False)

                if hot.P_boundary_at_inlet:
                    P_hot[i+1] = P_new



            # Initialize cold boundary and march backwards
            T_cold[N] = cold.boundary_T
            if cold.P_boundary_at_inlet:
                P_cold[N] = cold.boundary_P
            if not hot.P_boundary_at_inlet:
                P_hot[N] = hot.boundary_P

            # Backward march for cold fluid and hot pressure (if needed)
            for i in range(N, 0, -1): # March from N, N-1 down to 2 and 1 (no 0)
                # Update cold fluid properties
                T_cold[i-1], P_new, _ = update_static_properties(cold.properties,
                    cold.m_dot/geom.A_q_c, dh0_cold_list[i-1], dFA_cold_list[i-1],
                    T_cold[i], P_cold[i] if cold.P_boundary_at_inlet else P_cold_prev[i],
                    cold.properties.get_density(T_cold[i], P_cold[i] if cold.P_boundary_at_inlet else P_cold_prev[i]), False)

                if cold.P_boundary_at_inlet:
                    P_cold[i-1] = P_new

                # Update hot pressure if boundary at outlet
                if not hot.P_boundary_at_inlet:
                    T_hot[i], P_hot[i-1], _ = update_static_properties(hot.properties,
                        hot.m_dot/geom.A_q_h, dh0_hot_list[i-1], dFA_hot_list[i-1],
                        T_hot[i-1], P_hot[i], hot.properties.get_density(T_hot[i-1], P_hot_prev[i-1]), True)

            # Final forward march for cold pressure if boundary at outlet
            if not cold.P_boundary_at_inlet:
                P_cold[0] = cold.boundary_P
                for i in range(0, N):
                    T_cold[i+1], P_cold[i+1], _ = update_static_properties(cold.properties,
                        cold.m_dot/geom.A_q_c, dh0_cold_list[i], dFA_cold_list[i],
                        T_cold[i], P_cold[i], cold.properties.get_density(T_cold[i], P_cold_prev[i]), True)

            # Check convergence
            T_hot_diff = np.max(np.abs(T_hot - T_hot_prev))
            T_cold_diff = np.max(np.abs(T_cold - T_cold_prev))
            P_hot_diff = np.max(np.abs(P_hot - P_hot_prev))
            P_cold_diff = np.max(np.abs(P_cold - P_cold_prev))

            # Store convergence metrics
            convergence_metrics.append({
                'T_hot_diff': T_hot_diff,
                'T_cold_diff': T_cold_diff,
                'P_hot_diff': P_hot_diff,
                'P_cold_diff': P_cold_diff
            })

            # Calculate current effectiveness
            Q_current = np.sum(Q_seg_list)
            effectiveness = Q_current / Q_max if Q_max > 0 else 0.0
            effectiveness_history.append(effectiveness)

            # Check temperature and pressure convergence separately
            T_converged = T_hot_diff < 0.1 and T_cold_diff < 0.1
            P_converged = P_hot_diff < 0.01e5 and P_cold_diff < 0.01e5

            if T_converged and P_converged:
                converged = True
            else:
                # Detect oscillations
                if prev_diffs is not None:
                    # Check if differences are oscillating
                    T_hot_osc = (T_hot_diff > prev_diffs['T_hot_diff'] and
                               T_hot_diff > oscillation_threshold * prev_diffs['T_hot_diff'])
                    T_cold_osc = (T_cold_diff > prev_diffs['T_cold_diff'] and
                                T_cold_diff > oscillation_threshold * prev_diffs['T_cold_diff'])

                    if T_hot_osc or T_cold_osc:
                        # Reduce relaxation factor if oscillating
                        relax = max(relax_min, relax * 0.8)
                    else:
                        # Gradually increase relaxation if not oscillating
                        relax = min(relax_max, relax * 1.1)

                # Store current differences for next iteration
                prev_diffs = {
                    'T_hot_diff': T_hot_diff,
                    'T_cold_diff': T_cold_diff,
                    'P_hot_diff': P_hot_diff,
                    'P_cold_diff': P_cold_diff
                }

                # Apply relaxation
                np.add(T_hot_prev, relax * (T_hot - T_hot_prev), out=T_hot)
                np.add(T_cold_prev, relax * (T_cold - T_cold_prev), out=T_cold)
                np.add(P_hot_prev, relax * (P_hot - P_hot_prev), out=P_hot)
                np.add(P_cold_prev, relax * (P_cold - P_cold_prev), out=P_cold)

                # Enforce temperature bounds
                # Hot fluid can't exceed its inlet temperature and can't go below cold's inlet temperature
                T_hot = np.minimum(T_hot, hot.boundary_T)
                T_hot = np.maximum(T_hot, cold.boundary_T)

                # Cold fluid can't go below its inlet temperature and can't exceed hot's inlet temperature
                T_cold = np.maximum(T_cold, cold.boundary_T)
                T_cold = np.minimum(T_cold, hot.boundary_T)

                # Enforce minimum temperature difference between hot and cold
                min_temp_diff = 0.01 / (iteration + 1)  # Decreases with iterations

                # If temperatures are too close, move them apart from the middle
                too_close_mask = (T_hot - T_cold) < min_temp_diff
                if np.any(too_close_mask):
                    mid_temp = (T_hot + T_cold) / 2
                    T_hot[too_close_mask] = mid_temp[too_close_mask] + min_temp_diff/2
                    T_cold[too_close_mask] = mid_temp[too_close_mask] - min_temp_diff/2

                iteration += 1
        # Check if convergence was due to max iterations
        if iteration >= max_iterations and not converged:
            print(f"Warning: Solution did not converge after {max_iterations} iterations")
        else:
            pass
            #print(f"Solution converged after {iteration} iterations")


        x_array = np.linspace(0, geom.length, N+1)

        for i in range(N): #With converged T,P calculated Mach and Reynolds (weren't used in correlations? bad idea)

             #Calculate derived properties

        return results
