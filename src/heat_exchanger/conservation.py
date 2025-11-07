import numpy as np


def energy_balance_segment(
    m_dot_hot: float, m_dot_cold: float, T_hot: float, T_cold: float, U: float, area: float
):
    """
    Compute change in stagnation enthalpy for hot and cold streams in a segment.

    Args:
        m_dot_hot: Hot stream mass flow rate (kg/s)
        m_dot_cold: Cold stream mass flow rate (kg/s)
        T_hot: Hot stream temperature (K)
        T_cold: Cold stream temperature (K)
        U: Overall heat transfer coefficient (W/m²·K)
        area: Heat transfer area (m²)

    Returns:
        tuple: (dh0_hot, dh0_cold, Q_segment)
        dh0 is change in stagnation enthalpy (J/kg)
        Q_segment is heat transferred (W)
    """
    # Heat transferred in this segment (positive if hot fluid loses heat)
    Q_segment = U * area * (T_hot - T_cold)

    # Change in stagnation enthalpy
    dh0_hot = -Q_segment / m_dot_hot if m_dot_hot != 0 else 0.0
    dh0_cold = Q_segment / m_dot_cold if m_dot_cold != 0 else 0.0

    return dh0_hot, dh0_cold, Q_segment


def momentum_balance_segment(
    G: float, rho_mean: float, D_h: float, length: float, f: float
) -> float:
    """
    Compute change in impulse function (F/A = p + G²/rho) over a segment.
    F/A_out - F/A_in = - tau_eff * P dx/A_cross_section = - tau_eff * 4dx/d_h
    Uses Fanning/Kays-London friction factor (includes mixing/wake pressure drop).

    Args:
        G: Mass velocity (kg/m²s)
        rho_mean: Mean density in segment (kg/m³)
        D_h: Hydraulic diameter (m)
        length: Segment length (m)
        f: Friction factor (-), Fanning/Kays-London definition

    Returns:
        float: Change in impulse function (Pa)
    """
    if rho_mean == 0 or D_h == 0:
        return 0.0
    return -f * (4 * length / D_h) * (G**2) / (2 * rho_mean)


def update_static_properties(
    fluid_props,
    G,
    dh0,
    tau_dA_over_A_c,
    T_a,
    rho_a,
    p_b,
    a_is_in=True,
    b_is_in=True,
    max_iter=10,
    tol_T=1e-2,
    rel_tol_p=1e-3,  # Pa tolerance
    fd_eps_T=1e-3,
    fd_eps_p=50.0,
):
    r"""
    Solve simultaneously for T_not_a and p_not_b so that:
      1) Energy/stagnation enthalpy: (h_out + 0.5*(G^2/rho_out^2)) - (h_in + 0.5*(G^2/rho_in^2)) = dh0
      2) Momentum/impulse:           (p_out + G^2/rho_out) - (p_in + G^2/rho_in) = -tau_dA_over_A_c

    a can either be in (if a_is_in is True) or out (if a_is_in is False) of the heat exchanger.
    b can either be in (if b_is_in is True) or out (if b_is_in is False) of the heat exchanger.

    Tolerances and finite-difference steps:
      - tol_T: Absolute convergence tolerance on the energy residual R1 (units of J/kg).
               When |R1| < tol_T, the energy equation is considered converged.
      - rel_tol_p: Relative convergence tolerance on the momentum residual R2, scaled by p_b.
                   Converged when |R2| < rel_tol_p * p_b (units of Pa).
      - fd_eps_T: Finite-difference perturbation on temperature used to estimate dR/dT (units of K).
                  This should be small enough to linearize but large enough to avoid numerical noise.
      - fd_eps_p: Finite-difference perturbation on pressure used to estimate dR/dp (units of Pa).
                  Since pressure convergence uses a relative tolerance, an absolute FD step of ~O(\(10^2\) Pa)
                  is typically adequate near 1 bar. Internally, an effective step is used:
                      fd_eps_p_eff = max(fd_eps_p, 0.5 * rel_tol_p * p_b)
                  so that the derivative remains well-scaled when p_b changes.

    Note:  tau_eff dA_friction / A_cross_section > 0.

    Returns:
        (T_not_a, p_not_b, rho_not_a)
    """

    # ------------------------------------------------------------
    # 1) Initial Guesses for T_non_a assumes no pressure drop for c_p
    # ------------------------------------------------------------
    # Could improve guess by then using c_p(T_avg) to get T_guess
    if a_is_in:
        T_in = T_a
        rho_in = rho_a
        cp_in = fluid_props.get_cp(T_in, p_b)
        T_guess = T_in + dh0 / cp_in
    else:
        T_out = T_a
        rho_out = rho_a
        cp_out = fluid_props.get_cp(T_out, p_b)
        T_guess = T_out - dh0 / cp_out
    # For p_guess, a naive shift by dFA is typical (neglect density change)
    if b_is_in:
        p_in = p_b
        p_guess = p_b - tau_dA_over_A_c
    else:
        p_out = p_b
        p_guess = p_b + tau_dA_over_A_c

    # ------------------------------------------------------------
    # 2) Helper function: compute R1, R2 for a given guess of (T, p_unknown)
    # ------------------------------------------------------------
    def compute_residuals(T_guess, p_guess):
        """
        Returns R1, R2 given the current guess of T, p
        """
        # Build local variables for both sides to avoid scoping issues
        # Pressures
        if b_is_in:
            p_in_loc = p_in
            p_out_loc = p_guess
        else:
            p_in_loc = p_guess
            p_out_loc = p_out

        # Temperatures
        if a_is_in:
            T_in_loc = T_a
            T_out_loc = T_guess
            rho_in_loc = rho_in
        else:
            T_in_loc = T_guess
            T_out_loc = T_a
            rho_out_loc = rho_out

        # Densities
        rho_in_loc = fluid_props.get_density(T_in_loc, p_in_loc)
        rho_out_loc = fluid_props.get_density(T_out_loc, p_out_loc)

        # Enthalpies
        h_in = fluid_props.get_specific_enthalpy(T_in_loc, p_in_loc)
        h_out = fluid_props.get_specific_enthalpy(T_out_loc, p_out_loc)

        # Stagnation enthalpies (per unit mass)
        h0_in = h_in + 0.5 * (G / rho_in_loc) ** 2
        h0_out = h_out + 0.5 * (G / rho_out_loc) ** 2

        # Residuals
        R1 = (h0_out - h0_in) - dh0
        R2 = (p_out_loc + G**2 / rho_out_loc) - (p_in_loc + G**2 / rho_in_loc) + tau_dA_over_A_c

        return R1, R2

    # ------------------------------------------------------------
    # 3) Newton Iteration (using finite-diff partial derivatives)
    # ------------------------------------------------------------
    converged = False
    for _iteration in range(max_iter):
        # Compute R1, R2 at current guesses
        R1, R2 = compute_residuals(T_guess, p_guess)

        # Check if close enough
        if abs(R1) < tol_T and abs(R2) < rel_tol_p * p_b:
            converged = True
            break

        # ~~~~~~~~ Finite-difference to get partial derivatives dR/dT, dR/dp ~~~~~~~~
        # We'll do a small shift in T:
        R1p, R2p = compute_residuals(T_guess + fd_eps_T, p_guess)
        dR1_dT = (R1p - R1) / fd_eps_T
        dR2_dT = (R2p - R2) / fd_eps_T

        # We'll do a small shift in p (scale step to relative tolerance):
        fd_eps_p_eff = max(fd_eps_p, 0.5 * rel_tol_p * p_b)
        R1p, R2p = compute_residuals(T_guess, p_guess + fd_eps_p_eff)
        dR1_dp = (R1p - R1) / fd_eps_p_eff
        dR2_dp = (R2p - R2) / fd_eps_p_eff

        # Build the Jacobian and the residual vector
        #    [ R1 ]   and   J = [ dR1/dT   dR1/dp ]
        #    [ R2 ]             [ dR2/dT   dR2/dp ]
        #
        # Newton step:  [ dT ] = - J^-1 [ R1 ]
        #               [ dp ]         [ R2 ]

        J = np.array([[dR1_dT, dR1_dp], [dR2_dT, dR2_dp]], dtype=float)
        R_vec = np.array([R1, R2], dtype=float)

        try:
            # Solve the linear system
            dX = np.linalg.solve(J, -R_vec)
        except np.linalg.LinAlgError:
            # If singular, just do an under-relaxed step
            dX = -0.5 * R_vec  # fallback

        # Update T_guess, p_guess
        T_guess_new = T_guess + dX[0]
        p_guess_new = p_guess + dX[1]

        # Optional: add some mild damping if needed
        alpha = 0.8
        T_guess = T_guess + alpha * (T_guess_new - T_guess)
        p_guess = p_guess + alpha * (p_guess_new - p_guess)

    # End iteration
    if not converged:
        raise ValueError(
            f"Failed to converge at T_a={T_a:.2f} K, p_b={p_b:.2e} Pa in {max_iter} iterations\n"
            f"residuals {abs(R1):.2e} > {tol_T:.2e} K or {abs(R2):.2e} > {rel_tol_p * p_b:.2e} Pa\n"
        )

    # Compute final residuals or do final get_density
    if a_is_in == b_is_in:  # then a = b (not a is not b)
        rho_not_a = fluid_props.get_density(T_guess, p_guess)
    else:  # they are different so not a is b
        rho_not_a = fluid_props.get_density(T_guess, p_b)

    return T_guess, p_guess, rho_not_a
