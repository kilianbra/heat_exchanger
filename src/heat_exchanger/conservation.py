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
    dFA,
    T_in,
    p_sp,
    rho_in,
    reverse_pressure_marching=False,
    max_iter=10,
    tol_T=1e-3,
    tol_p=10.0,  # Pa tolerance
    fd_eps_T=1e-3,
    fd_eps_p=50.0,
):
    """
    Solve simultaneously for (T_out, p_out) (or p_in) so that:
      1) (h_out + 0.5*(G^2/rho_out^2)) - (h_in + 0.5*(G^2/rho_in^2)) = dh0
      2) (p_out + G^2/rho_out) - (p_in + G^2/rho_in) = dFA

    If reverse_pressure_marching=True:
       - p_sp is p_out (the known outlet pressure),
         and we solve for p_in.
       - R2 is rearranged accordingly.

    Returns:
      (T_out, p_out, rho_out)  or  (T_out, p_in, rho_out)
      [depending on forward or reverse meaning of "p_out"]
    """

    # ------------------------------------------------------------
    # 1) Helper function: compute R1, R2 for a given guess of (T, p_unknown)
    # ------------------------------------------------------------
    def compute_residuals(T_guess, p_guess):
        """
        Returns R1, R2 given the current guess of T, p
        """

        # Evaluate fluid properties at 'specified' side
        #  (whichever side p_sp is meant for)

        if not reverse_pressure_marching:
            # Evaluate fluid properties at exit, unknown pressure
            rho_out = fluid_props.get_density(T_guess, p_guess)
            h_out = fluid_props.get_specific_enthalpy(T_guess, p_guess)
            h0_out = h_out + 0.5 * (G / rho_out) ** 2

            # Forward:  p_in = p_sp is known
            rho_in = fluid_props.get_density(T_in, p_sp)
            h_in = fluid_props.get_specific_enthalpy(T_in, p_sp)
            h0_in = h_in + 0.5 * (G / rho_in) ** 2  # "inlet" total enthalpy if forward
            # R1:  (h0_guess - h0_in) - dh0 = 0
            R1 = (h0_out - h0_in) - dh0

            # R2:  [p_guess + G^2/rho_guess] - [p_in + G^2/rho_in] - dFA = 0
            # We interpret p_guess as p_out
            R2 = (p_guess + G**2 / rho_out) - (p_sp + G**2 / rho_in) - dFA

        else:  # Reverse: p_out = p_sp is known
            # Evaluate fluid properties at exit, known pressure
            rho_out = fluid_props.get_density(T_guess, p_sp)
            h_out = fluid_props.get_specific_enthalpy(T_guess, p_sp)
            h0_out = h_out + 0.5 * (G / rho_out) ** 2

            rho_in = fluid_props.get_density(T_in, p_guess)  # used for KE reference
            h_in = fluid_props.get_specific_enthalpy(T_in, p_guess)
            h0_in = h_in + 0.5 * (G / rho_in) ** 2
            #        p_sp + G^2 / rho_sp
            # R1:  (h0_in - h0_out) - dh0 = 0  => sign might differ
            #      but let's keep the same sign as forward:
            # we want h0_guess (the 'unknown side') to differ from h0_sp by dh0
            R1 = (h0_out - h0_in) - dh0  # or negative of above, just keep consistent

            # R2: [p_sp + G^2/rho_sp] - [p_guess + G^2/rho_in] - dFA = 0
            #    p_guess is 'inlet' now
            R2 = (p_sp + G**2 / rho_out) - (p_guess + G**2 / rho_in) - dFA

        return R1, R2

    # ------------------------------------------------------------
    # 2) Initial Guesses for T_out and p_non_sp
    # ------------------------------------------------------------
    cp_in = fluid_props.get_cp(T_in, p_sp)

    # For T_guess, a naive shift by dh0 / cp is typical
    T_guess = T_in + dh0 / cp_in

    # Could improve guess by then using c_p(T_avg) to get T_guess

    # For p_guess, a naive shift by dFA is typical (small dp << p_sp assumption)
    if not reverse_pressure_marching:
        p_guess = p_sp + dFA  # forward => p_out ~ p_in + dFA (rough guess)
    else:
        p_guess = p_sp - dFA  # reverse => p_in ~ p_out - dFA (rough guess)

    # ------------------------------------------------------------
    # 3) Newton Iteration (using finite-diff partial derivatives)
    # ------------------------------------------------------------
    for iteration in range(max_iter):
        # Compute R1, R2 at current guesses
        R1, R2 = compute_residuals(T_guess, p_guess)

        # Check if close enough
        if abs(R1) < tol_T and abs(R2) < tol_p:
            break

        # ~~~~~~~~ Finite-difference to get partial derivatives dR/dT, dR/dp ~~~~~~~~
        # We'll do a small shift in T:
        R1p, R2p = compute_residuals(T_guess + fd_eps_T, p_guess)
        dR1_dT = (R1p - R1) / fd_eps_T
        dR2_dT = (R2p - R2) / fd_eps_T

        # We'll do a small shift in p:
        R1p, R2p = compute_residuals(T_guess, p_guess + fd_eps_p)
        dR1_dp = (R1p - R1) / fd_eps_p
        dR2_dp = (R2p - R2) / fd_eps_p

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
    if iteration == max_iter - 1:
        raise ValueError(
            f"Failed to converge at T_in={T_in:.2f} K, p_sp={p_sp:.2f} Pa in {max_iter} iterations"
        )
    # Compute final residuals or do final get_density
    if not reverse_pressure_marching:
        rho_out = fluid_props.get_density(T_guess, p_guess)
    else:
        rho_out = fluid_props.get_density(T_guess, p_sp)

    T_out = T_guess
    p_not_sp = p_guess

    return T_out, p_not_sp, rho_out
