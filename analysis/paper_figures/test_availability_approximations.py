"""
Test accuracy of practical availability approximations against the original formula.
Uses 1000 random samples of pressure drops and effectiveness; verifies exact forms are exact.
"""

import numpy as np

# Fixed params (typical recuperator)
GAMMA = 1.4
T_hi = 907.0   # K
T_ci = 588.0   # K
p_hi = 1.0     # bar (hot inlet, turbine exit)
p_ci = 9.0     # bar (cold inlet, compressor exit)
p_0 = 1.0      # bar (ambient)

k = 1.0 - 1.0 / GAMMA
eta_C = 1.0 - T_ci / T_hi
T_ci_over_1_minus_eta = T_ci / (1.0 - eta_C)  # = T_hi

def original_exact(eps, delta_h, delta_c):
    """Full original formula (Eq. 691-696 in Reference_draft)"""
    bracket_h = (1.0 - eps * eta_C) * (1.0 - delta_h)**(-k) - 1.0
    bracket_c = (1.0 + eps * eta_C / (1.0 - eta_C)) * (1.0 - delta_c)**(-k) - 1.0
    r_h = (p_0 / p_hi)**k
    r_c = (p_0 / p_ci)**k
    return -1.0 / (T_hi - T_ci) * (
        T_hi * r_h * bracket_h + T_ci * r_c * bracket_c
    )

def three_term_linearised(eps, delta_h, delta_c):
    """Thermal + viscous + coupling (first-order in delta)"""
    r_h = (p_0 / p_hi)**k
    r_c = (p_0 / p_ci)**k
    thermal = eps * (r_h - r_c)
    viscous = -k / eta_C * (r_h * delta_h + (1.0 - eta_C) * r_c * delta_c)
    coupling = eps * k * (r_h * delta_h - r_c * delta_c)
    return thermal + viscous + coupling

def exit_pressure_approx(eps, delta_h, delta_c):
    """Thermal+coupling at exit pressures + linearised viscous"""
    p_ho = p_hi * (1.0 - delta_h)
    p_co = p_ci * (1.0 - delta_c)
    thermal_plus_coupling = eps * ((p_0 / p_ho)**k - (p_0 / p_co)**k)
    r_h = (p_0 / p_hi)**k
    r_c = (p_0 / p_ci)**k
    viscous = -k / eta_C * (r_h * delta_h + (1.0 - eta_C) * r_c * delta_c)
    return thermal_plus_coupling + viscous

def exact_two_term(eps, delta_h, delta_c):
    """Exact two-term form: (T+C)_exact + V_exact (no approximation)"""
    p_ho = p_hi * (1.0 - delta_h)
    p_co = p_ci * (1.0 - delta_c)
    eta_J = 1.0 - (p_hi / p_ci)**k
    r_h = (p_0 / p_hi)**k
    thermal_plus_coupling = eps * ((p_0 / p_ho)**k - (p_0 / p_co)**k)
    viscous = -r_h / eta_C * (
        (p_hi / p_ho)**k - 1.0
        + (1.0 - eta_C) * (1.0 - eta_J) * ((p_ci / p_co)**k - 1.0)
    )
    return thermal_plus_coupling + viscous

def main():
    np.random.seed(42)
    n = 1000

    # Random: eps in (0.1, 0.95), delta_h and delta_c in (0.01, 0.15)
    eps = np.random.uniform(0.1, 0.95, n)
    delta_h = np.random.uniform(0.01, 0.15, n)
    delta_c = np.random.uniform(0.01, 0.15, n)

    # Clip to avoid singularities (1-delta must be > 0)
    delta_h = np.clip(delta_h, 0.005, 0.99)
    delta_c = np.clip(delta_c, 0.005, 0.99)

    orig = np.array([original_exact(e, dh, dc) for e, dh, dc in zip(eps, delta_h, delta_c, strict=False)])
    three = np.array([three_term_linearised(e, dh, dc) for e, dh, dc in zip(eps, delta_h, delta_c, strict=False)])
    exit_ = np.array([exit_pressure_approx(e, dh, dc) for e, dh, dc in zip(eps, delta_h, delta_c, strict=False)])
    exact2 = np.array([exact_two_term(e, dh, dc) for e, dh, dc in zip(eps, delta_h, delta_c, strict=False)])

    # Relative errors (avoid div by zero)
    tol = 1e-12
    scale = np.where(np.abs(orig) > tol, np.abs(orig), 1.0)
    err_three = np.abs(orig - three)
    err_exit = np.abs(orig - exit_)
    err_exact2 = np.abs(orig - exact2)

    print("=" * 60)
    print("Practical availability approximation tests (n=1000 samples)")
    print("=" * 60)
    print(f"Params: gamma={GAMMA}, T_hi={T_hi}, T_ci={T_ci}, p_hi={p_hi}, p_ci={p_ci}")
    print()

    # 1. Exact two-term form: must match original to machine precision
    max_err_exact = np.max(err_exact2)
    mean_err_exact = np.mean(err_exact2)
    print("1. EXACT two-term form vs original formula:")
    print(f"   Max |diff|:  {max_err_exact:.2e}")
    print(f"   Mean |diff|: {mean_err_exact:.2e}")
    if max_err_exact < 1e-10:
        print("   PASS: Exact form matches original to machine precision.")
    else:
        print("   FAIL: Exact form differs from original!")
    print()

    # 2. Three-term linearised approximation accuracy
    rel_err_three = err_three / scale
    print("2. Three-term linearised (thermal+viscous+coupling):")
    print(f"   Max |diff|:     {np.max(err_three):.2e}")
    print(f"   Max rel error:  {np.max(rel_err_three)*100:.2f}%")
    print(f"   Mean rel error: {np.mean(rel_err_three)*100:.2f}%")
    print(f"   Median rel err: {np.median(rel_err_three)*100:.2f}%")
    print()

    # 3. Exit-pressure approximation accuracy
    rel_err_exit = err_exit / scale
    print("3. Exit-pressure thermal+coupling + linear viscous:")
    print(f"   Max |diff|:     {np.max(err_exit):.2e}")
    print(f"   Max rel error:  {np.max(rel_err_exit)*100:.2f}%")
    print(f"   Mean rel error: {np.mean(rel_err_exit)*100:.2f}%")
    print()

    # 4. Subset with small pressure drops (linearisation should be good)
    small = (delta_h < 0.08) & (delta_c < 0.08)
    n_small = np.sum(small)
    rel_small_three = err_three[small] / scale[small]
    rel_small_exit = err_exit[small] / scale[small]
    print(f"4. Small pressure drops only (delta < 8%, n={n_small}):")
    print(f"   Three-term max rel error:  {np.max(rel_small_three)*100:.2f}%")
    print(f"   Exit-pressure max rel err:  {np.max(rel_small_exit)*100:.2f}%")
    print()

    # 5. Worst-case sample for each approximation
    idx_three = np.argmax(err_three)
    idx_exit = np.argmax(err_exit)
    print("5. Worst-case samples:")
    print(f"   Three-term:  eps={eps[idx_three]:.3f}, dh={delta_h[idx_three]:.3f}, dc={delta_c[idx_three]:.3f} -> err={err_three[idx_three]:.2e}"
          )
    print(f"   Exit-approx: eps={eps[idx_exit]:.3f}, dh={delta_h[idx_exit]:.3f}, dc={delta_c[idx_exit]:.3f} -> err={err_exit[idx_exit]:.2e}"
          )

if __name__ == "__main__":
    main()
