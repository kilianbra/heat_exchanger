"""
correlations.py

This module contains correlations for different heat exchanger geometries.
In some cases an explicit formula is used, in other cases a graph has been digitised and interpolated.
"""

import warnings

import numpy as np
from scipy.interpolate import interp1d

# Data from Kays & London 1984 CHX
# Fig 6-4 (p122) - Friction factor for laminar concentric annular flow
FRICTION_CONCENTRIC_LAMINAR_DATA = [
    (0, 16),
    (0.009708737864077666, 19.008130081300813),
    (0.016181229773462785, 19.6260162601626),
    (0.030744336569579298, 20.422764227642276),
    (0.05339805825242719, 21.170731707317074),
    (0.06957928802588997, 21.59349593495935),
    (0.08737864077669905, 21.9349593495935),
    (0.10032362459546929, 22.16260162601626),
    (0.12783171521035602, 22.504065040650406),
    (0.16828478964401297, 22.829268292682926),
    (0.21682847896440133, 23.138211382113823),
    (0.25889967637540456, 23.333333333333332),
    (0.3009708737864078, 23.479674796747968),
    (0.3462783171521036, 23.609756097560975),
    (0.40453074433656966, 23.723577235772357),
    (0.46601941747572817, 23.788617886178862),
    (0.5210355987055016, 23.83739837398374),
    (0.5760517799352751, 23.869918699186993),
    (0.6456310679611651, 23.934959349593495),
    (0.7233009708737865, 23.96747967479675),
    (1.0, 24),
]

# Fig 6-5 (p123) - Nusselt number for laminar concentric annular flow
# Wall boundary condition is constant heat flux
NUSSELT_CONCENTRIC_LAMINAR_DATA = [
    (0.05175600739371536, 16.018549747048905),
    (0.059149722735674676, 15.016863406408095),
    (0.06654343807763402, 14.03372681281619),
    (0.07948243992606285, 13.013490725126475),
    (0.09426987060998152, 12.030354131534569),
    (0.10720887245841038, 11.2141652613828),
    (0.12014787430683915, 10.602023608768972),
    (0.13493530499075784, 10.008431703204046),
    (0.1608133086876155, 9.247892074198989),
    (0.1922365988909427, 8.561551433389544),
    (0.22735674676524953, 8.005059021922428),
    (0.25508317929759705, 7.652613827993255),
    (0.2846580406654344, 7.355817875210793),
    (0.32532347504621073, 7.040472175379426),
    (0.365988909426987, 6.799325463743676),
    (0.40850277264325313, 6.576728499156831),
    (0.45286506469500915, 6.372681281618888),
    (0.5286506469500925, 6.1315345699831365),
    (0.600739371534196, 5.964586846543002),
    (0.6580406654343807, 5.834738617200674),
    (0.7597042513863217, 5.686340640809444),
    (0.822550831792976, 5.593591905564924),
    (0.9131238447319778, 5.500843170320405),
    (1.0, 5.445193929173694),
]

# Fig 6-7 - Turbulent Nusselt number for circular tube at prandtl=0.7
NUSSELT_CIRCULAR_TURBULENT_DATA = [
    (1e4, 29.006013007275083),
    (15497.970912116021, 39.278367251002734),
    (33119.560620331176, 69.26201500663633),
    (77259.03157239426, 137.34222066824626),
    (1e5, 171.98528456192847),
    (183768.3064825769, 280.4495968594034),
    (602735.6089958638, 753.0592582127344),
    (1e6, 1135.5715881362369),
]

# Data from Figure 6-7 p.129 (turbulent Nusselt for circular tube at different reynolds)
# Only digitised data of prandtl=0.7
NUSSELT_CONCENTRIC_RRATIO_0_2_PRANDTL_0_7 = [
    (1e4, 38.21022723912875),
    (14824.650245818953, 49.105375551726866),
    (21458.838814068622, 64.63304070095656),
    (34997.85687684415, 91.39207629014803),
    (46048.1763248177, 111.97103446004289),
    (69084.12134630402, 152.75500335676196),
    (1e5, 201.05783202558467),
    (139661.82738554175, 267.81482506285056),
    (204588.31666274174, 361.0241882087252),
    (299698.06423446094, 492.52247554590485),
    (439022.7710507395, 679.9921117380443),
    (598690.0943083635, 873.8830002276075),
    (1e6, 1327.5034291529391),
]

# Data fom Figure 6-14 p.132 (turbulent Nusselt for annular flow at different r_ratio)
# Only digitised nusselt_ii, htc to inner if outer insulated. data of prandtl=0.7 (also had prandtl = 0.01, not relevant)
NUSSELT_CONCENTRIC_REYNOLDS_1E5_PRANDTL_0_7 = [
    (0.10416666666666666, 232.20338983050846),
    (0.11631944444444445, 225.4237288135593),
    (0.12673611111111108, 219.83050847457628),
    (0.14409722222222224, 213.0508474576271),
    (0.15972222222222224, 207.28813559322032),
    (0.18055555555555555, 201.01694915254237),
    (0.21006944444444445, 193.72881355932202),
    (0.24999999999999997, 186.27118644067795),
    (0.2881944444444444, 180.50847457627117),
    (0.32118055555555547, 176.77966101694915),
    (0.359375, 173.22033898305085),
    (0.4184027777777778, 169.15254237288133),
    (0.48263888888888884, 166.27118644067798),
    (0.5416666666666666, 163.89830508474574),
    (0.626736111111111, 161.1864406779661),
    (0.7100694444444444, 158.9830508474576),
    (0.8177083333333334, 156.77966101694915),
    (0.9027777777777779, 155.4237288135593),
    (1, 154.40677966101694),
]

# Data from Kays & London 1984 Fig 6-2 p 121, x is 1/alpha = a/b where b is the longer side of the rectangle i.e. x<=1
# Exact values from Table 6-1 p.120 where it is specified for L/d_h > 100
FRICTION_RECTANGULAR_DUCT_LAMINAR_DATA = [
    (0, 24),
    (0.012643678160919936, 23.723316062176163),
    (0.02413793103448314, 23.358549222797926),
    (0.04022988505747166, 22.87979274611399),
    (0.05632183908046015, 22.37823834196891),
    (0.08160919540229927, 21.717098445595852),
    (0.11149425287356363, 20.964766839378235),
    (0.125, 20.6),
    (0.13678160919540272, 20.417616580310877),
    (0.16436781609195444, 19.824870466321244),
    (0.1804597701149429, 19.460103626943003),
    (0.2172413793103452, 18.798963730569948),
    (0.25, 18.3),
    (0.28620689655172454, 17.72746113989637),
    (0.3252873563218395, 17.203108808290153),
    (0.3758620689655176, 16.633160621761657),
    (0.4172413793103453, 16.222797927461137),
    (0.4747126436781613, 15.7440414507772),
    (0.5229885057471269, 15.402072538860104),
    (0.5712643678160925, 15.128497409326423),
    (0.6494252873563222, 14.763730569948185),
    (0.7459770114942532, 14.490155440414508),
    (0.8011494252873568, 14.398963730569948),
    (0.8540229885057473, 14.330569948186527),
    (0.9183908045977014, 14.262176165803108),
    (1.0, 14.2),
]
# Kays and London fig 6-3 p 121
NUSSELT_Q_CST_RECTANGULAR_DUCT_LAMINAR_DATA = [
    (0, 24),
    (0.012643678160919936, 23.723316062176163),
    (0.02413793103448314, 23.358549222797926),
    (0.04022988505747166, 22.87979274611399),
    (0.05632183908046015, 22.37823834196891),
    (0.08160919540229927, 21.717098445595852),
    (0.11149425287356363, 20.964766839378235),
    (0.125, 20.6),
    (0.13678160919540272, 20.417616580310877),
    (0.16436781609195444, 19.824870466321244),
    (0.1804597701149429, 19.460103626943003),
    (0.2172413793103452, 18.798963730569948),
    (0.25, 18.3),
    (0.28620689655172454, 17.72746113989637),
    (0.3252873563218395, 17.203108808290153),
    (0.3758620689655176, 16.633160621761657),
    (0.4172413793103453, 16.222797927461137),
    (0.4747126436781613, 15.7440414507772),
    (0.5229885057471269, 15.402072538860104),
    (0.5712643678160925, 15.128497409326423),
    (0.6494252873563222, 14.763730569948185),
    (0.7459770114942532, 14.490155440414508),
    (0.8011494252873568, 14.398963730569948),
    (0.8540229885057473, 14.330569948186527),
    (0.9183908045977014, 14.262176165803108),
    (1.0, 14.2),
]

# Create interpolation functions for laminar flow
_interp_friction = interp1d(*zip(*FRICTION_CONCENTRIC_LAMINAR_DATA, strict=True), kind="linear")
_interp_nusselt_laminar = interp1d(*zip(*NUSSELT_CONCENTRIC_LAMINAR_DATA, strict=True), kind="linear")

# Create interpolation functions for turbulent flow
_interp_nusselt_circular_turbulent = interp1d(*zip(*NUSSELT_CIRCULAR_TURBULENT_DATA, strict=True), kind="linear")
# in annular flow have two fixed graphs for prandtl=0.7: one with r_ratio = 0.2 and reynolds varying
# the other with reynolds = 1e5 and r_ratio varying
_interp_nusselt_reynolds = interp1d(
    *zip(*NUSSELT_CONCENTRIC_RRATIO_0_2_PRANDTL_0_7, strict=True), kind="linear", bounds_error=True
)
_interp_nusselt_rratio = interp1d(
    *zip(*NUSSELT_CONCENTRIC_REYNOLDS_1E5_PRANDTL_0_7, strict=True),
    kind="linear",
    bounds_error=True,
)

# Create interpolation functions for rectangular ducts
_interp_friction_rectangular = interp1d(*zip(*FRICTION_RECTANGULAR_DUCT_LAMINAR_DATA, strict=True), kind="linear")
_interp_nusselt_q_cst_rectangular = interp1d(
    *zip(*NUSSELT_Q_CST_RECTANGULAR_DUCT_LAMINAR_DATA, strict=True), kind="linear"
)


def von_karman_nikuradse_smooth(reynolds):
    """
    Calculate Fanning friction factor for turbulent flow in smooth tubes using von Karman-Nikuradse correlation.
    Uses iterative solution of implicit equation.
    TODO: add source!

    Args:
        reynolds: Reynolds number

    Returns:
        Fanning friction factor
    """
    f = 0.01  # Initial guess
    for _ in range(1000):
        f_new = 1 / (1.737 * np.log(reynolds * np.sqrt(f)) - 0.396) ** 2
        if abs(f_new - f) < 1e-6:
            return f_new
        f = f_new
    return None  # If no convergence after 1000 iterations


def nusselt_gnielinski(f, reynolds, prandtl):
    """
    Calculate Nusselt number using Gnielinski correlation.
    From Shah (2003) Equation 7.76.

    Args:
        f: Fanning friction factor
        reynolds: Reynolds number
        prandtl: Prandtl number

    Returns:
        Nusselt number

    Accuracy: ±10%
    Valid for:
    - 2300 ≤ reynolds ≤ 5×10⁶
    - 0.5 ≤ prandtl ≤ 2000
    Note: Not a good correlation in the transition regime
    """
    return (f / 2) * (reynolds - 1000) * prandtl / (1 + 12.7 * np.sqrt(f / 2) * (prandtl ** (2 / 3) - 1))


def circular_pipe_friction_factor(reynolds, r_ratio=0):
    """
    Calculate Fanning friction factor for circular (r_ratio=0) or annular tubes.
    Uses correlations from Kays & London 1984.
    Transitional interpolation is own work by KB

    Args:
        reynolds: Reynolds number
        r_ratio: Radius ratio (inner/outer) for annular flow, 0 for circular tube
    """
    if reynolds <= 0:
        return 0.0

    if reynolds <= 2300:  # Laminar
        if not (0 <= r_ratio <= 1):
            raise ValueError(f"r_ratio {r_ratio:.2f} must be between 0 and 1")
        return _interp_friction(r_ratio) / reynolds  # Uses Kays & London 1984 Fig 6-4 (p122)

    elif reynolds > 1e4:  # Turbulent
        return von_karman_nikuradse_smooth(
            reynolds
        )  # Kays & London 1984 Fig 6-6 uses 0.46 reynolds^-0.2 instead for this, but I chose to use another formula

    else:  # Transitional
        # Interpolate between laminar and turbulent
        reynolds_fit, n_fit = (
            4500,
            4,
        )  # values found from fidling around on Desmos and using the smooth pipe flow Fig from K&L (Fig 10-1?)
        f_lam = _interp_friction(r_ratio) / reynolds
        f_turb = 0.079 * reynolds ** (-0.25)
        return (f_lam + (reynolds / reynolds_fit) ** n_fit * f_turb) / (1 + (reynolds / reynolds_fit) ** n_fit)


def circular_pipe_nusselt(reynolds, r_ratio=0, prandtl=0.7, show_warnings=False):
    """
    Calculate Nusselt number for circular (r_ratio=0) or annular tubes.
    Uses correlations from Kays & London 1984.
    In laminar flow assumes constant wall heat flux
    Transitional interpolation is own work by KB

    Args:
        reynolds: Reynolds number
        r_ratio: Radius ratio (inner/outer) for annular flow, 0 for circular tube
        prandtl: Prandtl number (default 0.7)
        show_warnings: Whether to show warnings for transitional flow
    """
    if reynolds < 2300:  # Laminar
        if r_ratio == 0:
            return 4.36  # Circular tube, constant heat flux
        if not (0.052 <= r_ratio <= 1):
            raise ValueError(f"r_ratio {r_ratio:.2f} must be between 0.052 and 1")
        return _interp_nusselt_laminar(r_ratio)

    elif reynolds >= 1e4:  # Turbulent
        if r_ratio == 0:
            if not (1e4 <= reynolds <= 1e6):
                if show_warnings:
                    warnings.warn(f"reynolds={reynolds:.1e} outside correlation range", stacklevel=2)
                if reynolds > 1e6:
                    return 10 ** (-1.771548501143517) * reynolds ** (0.8027382977987497)
            return _interp_nusselt_circular_turbulent(reynolds)
        else:
            if not (0.105 <= r_ratio <= 1):
                raise ValueError(f"r_ratio of {r_ratio:.2f} must be between 0.105 and 1.")

            # Get nusselt at r_ratio=0.2 for given reynolds
            if reynolds <= 1e6:
                nusselt_at_fixed_rratio = _interp_nusselt_reynolds(reynolds)
            else:
                if show_warnings:
                    warnings.warn(f"Reynolds of {reynolds:.1e} above 1e6, extrapolating", stacklevel=2)
                nusselt_at_fixed_rratio = 10 ** (-1.555820526825431) * reynolds ** (0.7762764854498554)

            # Scale by r_ratio effect at reynolds=1e5
            nusselt_at_fixed_reynolds = _interp_nusselt_rratio(r_ratio)
            scaled_nusselt = nusselt_at_fixed_rratio * (nusselt_at_fixed_reynolds / _interp_nusselt_rratio(0.2))
            return scaled_nusselt

    else:  # Transitional
        if show_warnings:
            warnings.warn(f"Flow is transitional (reynolds={reynolds:.1e})", stacklevel=2)
        # Interpolate j/f between laminar and turbulent
        j_f_lam = (4.36 if r_ratio == 0 else _interp_nusselt_laminar(r_ratio)) / (
            prandtl ** (1 / 3) * _interp_friction(r_ratio)
        )
        j_f_turb = circular_pipe_nusselt(1e4, r_ratio) / (
            prandtl ** (1 / 3) * 1e4 * circular_pipe_friction_factor(1e4, r_ratio)
        )

        # Linear interpolation in log space
        a = (j_f_turb - j_f_lam) / (np.log10(1e4) - np.log10(2300))
        b = j_f_lam - a * np.log10(2300)
        j_f = a * np.log10(reynolds) + b

        return j_f * circular_pipe_friction_factor(reynolds, r_ratio) * reynolds * prandtl ** (1 / 3)


def rectangular_duct_friction_factor(reynolds, a_over_b=1, show_warnings=False):
    """
    Calculate friction factor for rectangular ducts.
    Uses correlations from Kays & London 1984 for laminar flow.
    Uses von Karman-Nikuradse for turbulent flow.
    Transitional interpolation follows same approach as circular pipe.

    Args:
        reynolds: Reynolds number
        a_over_b: Aspect ratio of rectangular duct (shorter/longer side), ≤ 1
        show_warnings: Whether to show warnings for transitional flow
    """
    if a_over_b < 0:
        raise ValueError(f"a_over_b ratio {a_over_b:.2f} must be non-negative")

    # Ensure a_over_b <= 1 by taking reciprocal if needed
    if a_over_b > 1:
        a_over_b = 1 / a_over_b

    if reynolds <= 0:
        return 0.0

    if reynolds <= 2300:  # Laminar
        return _interp_friction_rectangular(a_over_b) / reynolds

    elif reynolds > 1e4:  # Turbulent
        return von_karman_nikuradse_smooth(reynolds)

    else:  # Transitional
        if show_warnings:
            warnings.warn(f"Flow is transitional (reynolds={reynolds:.1e})", stacklevel=2)
        # Interpolate between laminar and turbulent
        reynolds_fit, n_fit = 4500, 4
        f_lam = _interp_friction_rectangular(a_over_b) / reynolds
        f_turb = von_karman_nikuradse_smooth(1e4)
        return (f_lam + (reynolds / reynolds_fit) ** n_fit * f_turb) / (1 + (reynolds / reynolds_fit) ** n_fit)


def rectangular_duct_nusselt(reynolds, a_over_b=1, prandtl=0.7, show_warnings=False):
    """
    Calculate Nusselt number for rectangular ducts.
    Uses interpolated data from Kays & London for laminar flow.
    Uses Gnielinski correlation for turbulent flow.

    Args:
        reynolds: Reynolds number
        a_over_b: Aspect ratio of rectangular duct (shorter/longer side), ≤ 1
        prandtl: Prandtl number
        show_warnings: Whether to show warnings

    Returns:
        Nusselt number
    """
    if a_over_b < 0:
        raise ValueError(f"a_over_b ratio {a_over_b:.2f} must be non-negative")

    # Ensure a_over_b <= 1 by taking reciprocal if needed
    if a_over_b > 1:
        a_over_b = 1 / a_over_b

    if reynolds <= 2300:  # Laminar
        return _interp_nusselt_q_cst_rectangular(a_over_b)

    elif reynolds > 1e4:  # Turbulent
        f = rectangular_duct_friction_factor(reynolds, a_over_b)
        return nusselt_gnielinski(f, reynolds, prandtl)

    else:  # Transitional
        if show_warnings:
            warnings.warn(f"Flow is transitional (reynolds={reynolds:.1e})", stacklevel=2)
        # Interpolate between laminar and turbulent using j/f method
        j_f_lam = _interp_nusselt_q_cst_rectangular(a_over_b) / (
            prandtl ** (1 / 3) * _interp_friction_rectangular(a_over_b)
        )

        # Get turbulent j/f at reynolds=1e4
        f_turb = rectangular_duct_friction_factor(1e4, a_over_b)
        nusselt_turb = nusselt_gnielinski(f_turb, 1e4, prandtl)
        j_f_turb = nusselt_turb / (prandtl ** (1 / 3) * 1e4 * f_turb)

        # Linear interpolation in log space
        a = (j_f_turb - j_f_lam) / (np.log10(1e4) - np.log10(2300))
        b = j_f_lam - a * np.log10(2300)
        j_f = a * np.log10(reynolds) + b

        return j_f * rectangular_duct_friction_factor(reynolds, a_over_b) * reynolds * prandtl ** (1 / 3)


def offset_strip_fin_friction_factor(
    reynolds: float,
    s_over_h_prime: float,
    delta_over_l_s: float,
    delta_over_s: float,
    show_warnings: bool = False,
) -> float:
    """
    Calculate friction factor for offset strip fins using Manglik and Bergles correlation.
    From Shah (2003) p516 eqn (7.124), based on Manglik and Bergles (1995).

    Args:
        reynolds: Reynolds number
        s_over_h_prime: Ratio of fin spacing to fin height
        delta_over_l_s: Ratio of fin thickness to fin length
        delta_over_s: Ratio of fin thickness to fin spacing
        show_warnings: Whether to show warnings when outside correlation range

    Returns:
        Fanning friction factor

    Valid for 120 < reynolds < 1e4
    """
    if show_warnings and (reynolds < 120 or reynolds > 1e4):
        warnings.warn(f"Reynolds number {reynolds:.1e} outside correlation range of 120-1e4", stacklevel=2)

    return (
        9.6243
        * reynolds**-0.7422
        * s_over_h_prime**-0.1856
        * delta_over_l_s**0.3053
        * delta_over_s**-0.2659
        * (1 + 7.669e-8 * reynolds**4.429 * s_over_h_prime**0.920 * delta_over_l_s**3.767 * delta_over_s**0.236) ** 0.1
    )


def offset_strip_fin_j_factor(
    reynolds: float,
    s_over_h_prime: float,
    delta_over_l_s: float,
    delta_over_s: float,
    show_warnings: bool = False,
) -> float:
    """
    Calculate j-factor for offset strip fins using Manglik and Bergles correlation.
    From Shah (2003) p516 eqn (7.124), based on Manglik and Bergles (1995).
    j = St * prandtl^(2/3), where St is the Stanton number.

    Args:
        reynolds: Reynolds number
        s_over_h_prime: Ratio of fin spacing to fin height
        delta_over_l_s: Ratio of fin thickness to fin length
        delta_over_s: Ratio of fin thickness to fin spacing
        show_warnings: Whether to show warnings when outside correlation range

    Returns:
        Colburn j-factor

    Valid for:
    - 1.2e2 < reynolds < 1e4
    - 0.5 < prandtl < 15
    """
    if show_warnings and (reynolds < 120 or reynolds > 1e4):
        warnings.warn(f"Reynolds number {reynolds:.1e} outside correlation range of 120-1e4", stacklevel=2)

    return (
        0.6522
        * reynolds**-0.5403
        * s_over_h_prime**-0.1541
        * delta_over_l_s**0.1499
        * delta_over_s**-0.0678
        * (1 + 5.269e-5 * reynolds**1.340 * s_over_h_prime**0.504 * delta_over_l_s**0.456 * delta_over_s**-1.055) ** 0.1
    )


def calculate_Hglam(Red, Xl, Xt, inline=False):
    # Shah 2003 Fundamentals of HX Equation 7.110
    if inline or (Xl >= 0.5 * (2 * Xt + 1) ** 0.5):
        Hglam = 140 * Red * ((Xl**0.5 - 0.6) ** 2 + 0.75) / (Xt**1.6 * (4 * Xt * Xl / np.pi - 1))
    else:
        Xd = (Xt**2 + Xl**2) ** 0.5
        Hglam = 140 * Red * ((Xl**0.5 - 0.6) ** 2 + 0.75) / (Xd**1.6 * (4 * Xt * Xl / np.pi - 1))
    test = Hglam
    assert isinstance(test, (int, float)) and not isinstance(test, complex), (
        f"The variable Hglam must be a real number but is {test}"
    )
    return Hglam


def calculate_Hgturb_i(Red, Xt, Xl, Nr=11):
    # Shah 2003 Fundamentals of HX Equation 7.111
    # Need to correct for number of tube rows
    if Nr > 10:
        phi_t_n = 0
    else:  # Shah Equation (7.114)
        if Nr < 5:
            # warn_with_custom_format(f"({Nr:.0f}<5 tube rows is too little for correlation")
            pass
        if Xl >= 0.5 * np.sqrt(2 * Xt + 1):
            phi_t_n = (1 / Nr - 1 / 10) / (2 * Xt**2)
        else:
            Xd = (Xt**2 + Xl**2) ** 0.5
            phi_t_n = 2 * (1 / Nr - 1 / 10) * ((Xd - 1) / (Xt * (Xt - 1))) ** 2

    #           First term is frictional pressure drop in the bundle                                                                                                  Second term inlet/outlet
    Hgturb_i = (
        (0.11 + 0.6 * (1 - 0.94 / Xl) ** 0.6 / (Xt - 0.85) ** 1.3) * 10 ** (0.47 * (Xl / Xt - 1.5))
        + 0.015 * (Xt - 1) * (Xl - 1)
    ) * Red ** (2 - 0.1 * Xl / Xt) + phi_t_n * Red**2
    assert isinstance(Hgturb_i, (int, float)) and not isinstance(Hgturb_i, complex), (
        f"The variable Hgturb_i must be a real number but is {Hgturb_i}"
    )
    return Hgturb_i


def calculate_Hgturb_s(Red, Xt, Xl, Nr):
    # Shah 2003 Fundamentals of HX Equation 7.112
    # Need to correct for number of tube rows
    if Nr > 10:
        phi_t_n = 0
    else:  # Shah Equation (7.114)
        if Nr < 5:
            # warn_with_custom_format(f"({Nr:.0f}<5 tube rows is too little for correlation")
            pass
        if Xl >= 0.5 * np.sqrt(2 * Xt + 1):
            phi_t_n = (1 / Nr - 1 / 10) / (2 * Xt**2)
        else:
            Xd = (Xt**2 + Xl**2) ** 0.5
            phi_t_n = 2 * (1 / Nr - 1 / 10) * ((Xd - 1) / (Xt * (Xt - 1))) ** 2
    #           First term is frictional pressure drop in the bundle                                                       Second term inlet/outlet
    Hgturb_s = (
        (1.25 + 0.6 / (Xt - 0.85) ** 1.08) + 0.2 * (Xl / Xt - 1) ** 3 - 0.005 * (Xt / Xl - 1) ** 3
    ) * Red**1.75 + phi_t_n * Red**2

    if Red > 250000:
        Hgturb_s = Hgturb_s * (1 + (Red - 250000) / 325000)  # Shah 2003 Fundamentals of HX Equation 7.113
    test = Hgturb_s
    assert isinstance(test, (int, float)) and not isinstance(test, complex), (
        f"The variable Hgturb_s must be a real number but is {test}"
    )
    return Hgturb_s


def calculate_Hg_dont_use(Red, Xl, Xt, inline=False, Nr=11):
    # Shah 2003 Fundamentals of HX Equation 7.109
    """Shah 2003 Fundamentals of HX Equation 7.109
    CONTAINS ERROR OF THE 1 THAT SHOULDNT BE THERE
    USE calculate_Hg NOW!
    Flow normal to a tube bundle by Zukauskas 1987 (shell side of shell and tube)
    1< Re_d < 3e5     Nr>=5
    validity (inline)        1.25 < Xt < 3        1.2 < Xl < 3.0
    validity (staggered)     1.25 < Xt < 3        0.6 < Xl < 3.0 and Xd > 1.25
    Correlations based on 7.9 < d_o < 73 mm experimental diameters

    Red : Reynolds number based on tube (outer) diameter
    Xt: tangential (normal to flow) spacing of tubes, normalised by tube outter diameter (Xt* in Shah)
    Xl: longitudinal (parrallel to flow) spacing of tubes, normalised by tube outter diameter (Xl* in Shah)
    """
    if inline:
        if not (1.25 <= Xt <= 3):
            # raise ValueError(f'Xt value {Xt:.2} is not within the range (1.25, 3)')
            pass
        if not (1.2 <= Xl <= 3.0):
            raise ValueError(f"Xl value {Xl:.2} is not within the range (1.2, 3.0)")
    else:
        Xd = (Xt**2 + Xl**2) ** 0.5
        if not (1.25 <= Xt <= 3):
            # raise ValueError(f'Xt value {Xt:.2} is not within the range (1.25, 3)')
            pass
        if not (0.6 <= Xl <= 3):
            raise ValueError(f"Xl value {Xl:.2} is not within the range (0.6, 3)")
        if not (Nr >= 5):
            raise ValueError(f"Nr value {Nr:} is not greater than 5")
        if Xd is None or Xd < 1.25:
            raise ValueError(f"X_d value {Xd:.2} is not greater than 1.25")

    Hglam = calculate_Hglam(Red, Xl, Xt, inline)
    if inline:
        Hgturb = calculate_Hgturb_i(Red, Xt, Xl, Nr)
        Hg = Hglam + Hgturb * (1 - np.exp(1 - (Red + 1000) / 2000))  # 1 is ERROR HERE!
    else:
        Hgturb = calculate_Hgturb_s(Red, Xt, Xl, Nr)
        Hg = Hglam + Hgturb * (1 - np.exp(1 - (Red + 200) / 1000))  # 1 is ERROR HERE!
    if Hg < 0:
        test = np.sqrt(-1)

    test = Hg
    assert isinstance(test, (int, float)) and not isinstance(test, complex), (
        f"The variable Hg must be a real number but is {test}"
    )

    return Hg


def calculate_Hg(Red, Xl, Xt, inline=False, Nr=11):
    """Use adapted correlation from Martin 2002 just without the added 1"""

    if inline:
        if not (1.25 <= Xt <= 3):
            # raise ValueError(f'Xt value {Xt:.2} is not within the range (1.25, 3)')
            pass
        if not (1.2 <= Xl <= 3.0):
            raise ValueError(f"Xl value {Xl:.2} is not within the range (1.2, 3.0)")
    else:
        Xd = (Xt**2 + Xl**2) ** 0.5
        if not (1.25 <= Xt <= 3):
            # raise ValueError(f'Xt value {Xt:.2} is not within the range (1.25, 3)')
            pass
        if not (0.6 <= Xl <= 3):
            raise ValueError(f"Xl value {Xl:.2} is not within the range (0.6, 3)")
        if not (Nr >= 5):
            raise ValueError(f"Nr value {Nr:} is not greater than 5")
        if Xd is None or Xd < 1.25:
            raise ValueError(f"X_d value {Xd:.2} is not greater than 1.25")

    Hglam = calculate_Hglam(Red, Xl, Xt, inline)
    if inline:
        Hgturb = calculate_Hgturb_i(Red, Xt, Xl, Nr)
        Hg = Hglam + Hgturb * (1 - np.exp(-(Red + 1000) / 2000))
    else:
        Hgturb = calculate_Hgturb_s(Red, Xt, Xl, Nr)
        Hg = Hglam + Hgturb * (1 - np.exp(-(Red + 200) / 1000))
    if Hg < 0:
        test = np.sqrt(-1)

    test = Hg
    assert isinstance(test, (int, float)) and not isinstance(test, complex), (
        f"The variable Hg must be a real number but is {test}"
    )

    return Hg


def calculate_Nu_n_Hg(Red, Pr, Xl, Xt, inline=False, Nr=11):
    """Shah 2003 Fundamentals of HX Equation 7.117 & 7.118
    Flow normal to a tube bundle by Martin 2002 (shell side of shell and tube)
    1< Re_d < 2e6     0.7 < Pr < 700  (also probably valid for larger than 700, defo not for lower than 0.6)
    validity (inline)        1.02 < Xt < 3        0.6 < Xl < 3    2< Nr <15 WEIRD, Xl<1 for inline is not physically possible
    validity (staggered)     1.02 < Xt < 3        0.6 < Xl < 3    4< Nr <80
    Correlations based on 7.9 < d < 73 mm
    Prediction of Nusselt number within +/- 20% for inline and 14% for staggered. Can improve with better friction data (experimental)

    Red : Reynolds number based on tube diameter and velocity at narrowest cross section
    Pr: Prandlt number of the fluid
    Xt: tangential (normal to flow) spacing of tubes, normalised by tube outter diameter Xt* in Shah
    Xl: longitudinal (parrallel to flow) spacing of tubes, normalised by tube outter diameter Xl* in Shah
    """

    Hg = calculate_Hg(Red, Xl, Xt, inline, Nr=Nr)
    if inline:
        Lq = 1.18 * Hg * Pr * (4 * Xt / np.pi - 1) / Xl
    elif Xl >= 1:
        Xd = (Xt**2 + Xl**2) ** 0.5
        Lq = 0.92 * Hg * Pr * (4 * Xt / np.pi - 1) / Xd
    else:
        Xd = (Xt**2 + Xl**2) ** 0.5
        Lq = 0.92 * Hg * Pr * (4 * Xt * Xl / np.pi - 1) / Xd / Xl
    test = Lq
    assert isinstance(test, (int, float)) and not isinstance(test, complex), (
        f"The variable Lq must be a real number but is {test}"
    )

    # Shah 2003 Fundamentals of HX Equation 7.117
    # by Martin 2002
    if inline:
        Nu = 0.404 * Lq ** (1.0 / 3) * ((Red + 1) / (Red + 1000)) ** 0.1
    else:
        Nu = 0.404 * Lq ** (1.0 / 3)
    test = Nu
    assert isinstance(test, (int, float)) and not isinstance(test, complex), (
        f"The variable Nu must be a real number but is {test}"
    )
    return Nu, Hg


def tube_bank_friction_factor(reynolds, spacing_long, spacing_trans, inline=True, n_rows=11):
    """Calculates the friction factor for a tube bank in cross flow.
    Implementation based on Shah 2003, reframed from Martin 2002, original from Gaddis and Gnielinski 1985.

    Args:
        reynolds: Reynolds number based on minimum free flow area and tube diameter.
        spacing_long: Longitudinal spacing between tubes, divided by tube outer diameter.
        spacing_trans: Transverse spacing between tubes, divided by tube outer diameter.
        inline: Whether the tubes are in line (True) or staggered (False).
        n_rows: Number of rows of tubes (if above 10 makes no difference).
    Returns:
        Kays and London equivalent friction factor.

    Valid for:
    - 1 <= reynolds <= 3e5
    - 5 <= n_rows

    The spacings used to create this data is unreliably presented in Shah and Martin. In the
    original Gaddis and Gnielinski paper, it is clearly stated that the following spacings had
    experimental data available, and are the basis of this correlation:

    spacing_trans x spacing_long:

    Re < 1e3:
        inline: 1.25 x 1.25, 1.5 x 1.5, 2.0 x 2.0
        staggered: 1.25 x 1.0825, 1.5 x 1.299, 1.768 x 0.884
        (these have spacing_diag = 1.25, 1.5 and 1.25 respectively)
    Re >= 1e3:
        inline: 1.25 <= spacing_trans <= 3.0, 1.2 <= spacing_long <= 3.0
        staggered: 1.25 <= spacing_trans <= 3.0, 0.6 <= spacing_long <= 3.0 but with
        spacing_diag >= 1.25 (diag = sqrt((spacing_trans/2)**2 + spacing_long**2))

    """

    assert 1 <= reynolds <= 3e5, f"Reynolds number {reynolds:.1e} outside correlation range of 1-3e5"
    assert n_rows >= 5, f"Number of rows {n_rows:.0f} outside correlation range of 5"
    assert 1.25 <= spacing_trans <= 3.0, f"Spacing trans {spacing_trans:.2f} outside correlation range of 1.25-3.0"
    if inline:
        # assert 1.2 <= spacing_long <= 3.0, (
        #    f"Inline Spacing long {spacing_long:.2f} outside correlation range of 1.2-3.0"
        # )
        pass
    else:
        assert 0.6 <= spacing_long <= 3.0, (
            f"Staggered Spacing long {spacing_long:.2f} outside correlation range of 0.6-3.0"
        )
        spacing_diag = ((spacing_trans / 2) ** 2 + spacing_long**2) ** 0.5
        assert spacing_diag >= 1.25, f"Staggered Spacing diag {spacing_diag:.2f} must be greater than 1.25"

    if reynolds < 1e3:
        # There are very few experimental data points for reynolds < 1e3, so we should warn user
        # Especially if they use non square spacing for inline
        # Or if they don't have diag spacing of 1.25 or 1.5 for staggered
        pass

    hagen_number = calculate_Hg(reynolds, spacing_long, spacing_trans, inline, n_rows)
    friction_factor_k_and_l = 2 * hagen_number / reynolds**2 * (spacing_trans - 1) / np.pi

    return friction_factor_k_and_l


def tube_bank_nusselt_from_hagen(hagen, reynolds, spacing_long, spacing_trans, prandtl=0.7, inline=True):
    """Calculates the Nusselt number and friction factor for a tube bank in cross flow.
    Implementation based on Shah 2003, reframed from Martin 2002, original from Gnielinski 1979.

    Args:
        reynolds: Reynolds number based on minimum free flow area and tube diameter.
        spacing_long: Longitudinal spacing between tubes, divided by tube outer diameter.
        spacing_trans: Transverse spacing between tubes, divided by tube outer diameter.
        prandtl: Prandtl number of the fluid.
        inline: Whether the tubes are in line (True) or staggered (False).
        n_rows: Number of rows of tubes (if above 10 makes no difference).
    Returns:
        Nusselt number and friction factor.
    """
    if inline:
        leveque = 1.18 * hagen * prandtl * (4 * spacing_trans / np.pi - 1) / spacing_long
    elif spacing_long >= 1:
        spacing_diag = ((spacing_trans / 2) ** 2 + spacing_long**2) ** 0.5
        leveque = 0.92 * hagen * prandtl * (4 * spacing_trans / np.pi - 1) / spacing_diag
    else:
        spacing_diag = ((spacing_trans / 2) ** 2 + spacing_long**2) ** 0.5
        leveque = 0.92 * hagen * prandtl * (4 * spacing_trans * spacing_long / np.pi - 1) / spacing_diag / spacing_long
    assert isinstance(leveque, (int | float)) and not isinstance(leveque, complex), (
        f"The Leveque number must be a real number but is {leveque:.2e}"
    )

    # Shah 2003 Fundamentals of HX Equation 7.117
    # by Martin 2002
    if inline:
        nusselt = 0.404 * leveque ** (1.0 / 3) * ((reynolds + 1) / (reynolds + 1000)) ** 0.1
    else:
        nusselt = 0.404 * leveque ** (1.0 / 3)
    assert isinstance(nusselt, (int | float)) and not isinstance(nusselt, complex), (
        f"The Nusselt number must be a real number but is {nusselt:.2e}"
    )

    return nusselt


def tube_bank_nusselt_number_and_friction_factor(
    reynolds, spacing_long, spacing_trans, prandtl=0.7, inline=True, n_rows=11
):
    """Calculates the Nusselt number and friction factor for a tube bank in cross flow.
    Implementation based on Shah 2003, reframed from Martin 2002, original from Gnielinski 1979.

    Args:
        reynolds: Reynolds number based on minimum free flow area and tube diameter.
        spacing_long: Longitudinal spacing between tubes, divided by tube outer diameter.
        spacing_trans: Transverse spacing between tubes, divided by tube outer diameter.
        prandtl: Prandtl number of the fluid.
        inline: Whether the tubes are in line (True) or staggered (False).
        n_rows: Number of rows of tubes (if above 10 makes no difference).
    Returns:
        Nusselt number and friction factor.
    """

    hagen = calculate_Hg(reynolds, spacing_long, spacing_trans, inline=inline, Nr=n_rows)
    nusselt = tube_bank_nusselt_from_hagen(hagen, reynolds, spacing_long, spacing_trans, prandtl, inline)

    friction_factor_k_and_l = 2 * hagen / reynolds**2 * (spacing_trans - 1) / np.pi

    return nusselt, friction_factor_k_and_l


def tube_bank_nusselt_gnielinski_vdi(
    reynolds,
    spacing_long,
    spacing_trans,
    prandtl=0.7,
    inline=True,
    n_rows=11,
    use_outside_bounds=True,
):
    """
    Correlation is based on Reynolds of the velocity outside the bundle! not in minimum cross section
    Correlation is based on longitudinal spacing > 1.2 diameters or with b/a >1
    """
    a = spacing_trans
    b = spacing_long

    if not inline and b < 0.5 * np.sqrt(2 * a + 1):  # throat is in diagonal
        d = np.sqrt((a / 2) ** 2 + (b) ** 2)
        sigma = 2 * (d - 1) / a
    else:
        sigma = (a - 1) / a

    reynolds_od = np.asarray(reynolds) if isinstance(reynolds, (list, np.ndarray)) else reynolds

    void_frac = 1 - np.pi / 4 / a if b >= 1 else 1 - np.pi / 4 / a / b
    # in VDI heat atlas Reynolds is based on l=pi/2 * d_o, not d_o
    # furthermore all other correlations are based on minimum free flow area, here is velocity outside bundle
    reynolds_psi = reynolds_od * np.pi / 2 / void_frac * sigma

    if not use_outside_bounds:
        mask = (reynolds_psi < 10) | (reynolds_psi > 1e6)
        reynolds_psi = np.where(mask, np.nan, reynolds_psi)

    nusselt_laminar = 0.664 * reynolds_psi**0.5 * prandtl ** (1 / 3)
    nusselt_turb = 0.037 * reynolds_psi**0.8 * prandtl / (1 + 2.443 * reynolds_psi ** (-0.1) * (prandtl ** (2 / 3) - 1))

    nusselt_uncorrected = 0.3 + np.sqrt(nusselt_laminar**2 + nusselt_turb**2)

    if inline:
        f_A = 1 + 0.7 * (b / a - 0.3) / void_frac**1.5 / (b / a + 0.7) ** 2
    else:
        f_A = 1 + 2 / 3 / b

    if n_rows >= 10:
        nusselt = f_A * nusselt_uncorrected / (np.pi / 2)  # division by pi to make result Nusselt in d_o lengthscale
    else:
        nusselt = (1 + (n_rows - 1) * f_A) / n_rows * nusselt_uncorrected

    return nusselt


def tube_bank_corrected_xi_gunter_and_shaw(
    reynolds, spacing_long, spacing_trans, bulk_to_wall_viscosity_ratio=1, use_outside_bounds=True
):
    """Calculates the corrected half friction factor for a tube bank in cross flow.
    Based on Gunter and Shaw 1945. Applicable to inline and staggered tube banks.
    Not xi/2 but xi!
    Correlation initially based on correcting Delta p/L rho d_v/G^2 with two factors + viscosity
    ratio.
    returns xi = Delta p / N_rows * 2 rho / G^2 (in paper a different 'f/2' is returned)
    Use volumetric hydraulic diameter in reynolds, so need to convert from outer diameter to d_v
    Args:
        reynolds: Reynolds number based on minimum free flow area and tube diameter.
        spacing_long: Longitudinal spacing between tubes, divided by tube outer diameter.
        spacing_trans: Transverse spacing between tubes, divided by tube outer diameter.
        bulk_to_wall_viscosity_ratio: Bulk to wall viscosity ratio.
        use_outside_bounds: Whether to use the outside bounds of the correlation.
    Bounds on Reynolds number dv 0.01 to 3e5
    """

    # For now ignore transition region from laminar to turbulent (slight curvature)

    ratio_dv_over_od = 4 * spacing_long * spacing_trans / np.pi - 1
    # tube outer diameter OD is used in many correlations and is used in inputed reynolds
    reynolds_od = np.asarray(reynolds) if isinstance(reynolds, (list, np.ndarray)) else reynolds
    reynolds_dv = reynolds_od * ratio_dv_over_od
    if not use_outside_bounds:
        # Return NaN where outside bounds (200 <= Re_dv <= 3e5)
        mask = (reynolds_dv < 1e-2) | (reynolds_dv > 3e5)
        reynolds_dv = np.where(mask, np.nan, reynolds_dv)
    phi = np.where(reynolds_dv < 200, 90 / reynolds_dv, 0.96 * reynolds_dv ** (-0.145))
    xi = (
        phi
        * bulk_to_wall_viscosity_ratio ** (-0.14)
        * (ratio_dv_over_od / spacing_trans) ** 0.4
        * (spacing_long / spacing_trans) ** 0.6
        * (spacing_long / ratio_dv_over_od)  # added term from D_v/L = D_v/X_l/N_r in paper
        * 2  # convert from half friction factor to full friction factor
    )

    return xi


def tube_bank_stanton_number_from_murray(
    reynolds_od, spacing_long, spacing_trans, prandtl=0.7, use_outside_bounds=True
):
    """
    This correlation is based on Murray 1998's thesis at the University of Bristol with Reaction
    Engines Ltd.
    In the thesis private communications with Bond A. at Reaction Engines is cited as source.
    The correlation uses the hydraulic diameter to calculate the Reynolds number.
    Re = G * d_h / mu
    This function uses the outer diameter and converts to hydraulic diameter.
    Returns Stanton number and friction factor.
    Beta = 0.184 is valid for Re_dh between 3e3 and 15e3
    Correction is made for lower Reynolds but was only used for Reynolds under 2e3 in thesis
    """
    reynolds_od = np.asarray(reynolds_od) if isinstance(reynolds_od, (list, np.ndarray)) else reynolds_od

    spacing_diag = ((spacing_trans / 2) ** 2 + spacing_long**2) ** 0.5
    xls = spacing_long
    xts = spacing_trans
    xlxt = xls * xts  # product of pitches
    kd = spacing_diag - 1
    kmin = min(2 * kd, xts - 1)
    kmax = max(2 * kd, xts - 1)

    reynolds_dh = reynolds_od * 4 * spacing_long * kmin / np.pi
    if not use_outside_bounds:
        mask = (reynolds_dh < 2e3) | (reynolds_dh > 5e3)
        reynolds_dh = np.where(mask, np.nan, reynolds_dh)
    # St4000 correlation
    j4000 = (0.002499 + 0.008261 * (xlxt - 1) - 0.000145 * (xlxt - 1) ** 2) / (kd * xls) ** 0.35
    St4000 = j4000 / prandtl ** (2 / 3)
    # St correlation
    St = St4000 * 25.6238 * reynolds_dh ** (-0.3913)

    f4000 = 0.0122 * (xlxt - 1) * (3 * xlxt - 2) / (kd * xls)

    beta = np.where(reynolds_dh >= 3000, 0.184, 0.184 + 0.2820 * (1 - 2 * kmax))

    f = np.where(
        reynolds_dh >= 3000,
        f4000 * (reynolds_dh / 4000) ** (-beta),
        f4000 * (reynolds_dh / 3000) ** (-beta),
    )

    return St, f


def htc_murray(G, Cp, Re, Pr, xls, xts, OD):
    """COPY PASTED FROM ZELI
    Inputs:
        G: mass velocity [kg/m^2/s]
        Cp: specific heat [J/kg/K]
        Re: Reynolds number based on hydraulic diameter
        Pr: Prandtl number
        xls: longitudinal pitch (normalised)
        xts: transverse pitch (normalised)
        OD: outer diameter [m]
    """
    xlxt = xls * xts  # product of pitches
    # distances between tube centers
    bt = 0.5 * OD * (xts - 1)  # transverse pitch
    bd = np.sqrt((xls * OD) ** 2 + (0.5 * xts * OD) ** 2)  # diagonal pitch
    # spacing ratios
    kt = bt / OD  # transverse spacing ratio
    kb = bd / OD  # diagonal spacing ratio
    kmin = min(kt, kb)  # minimum spacing ratio
    kmax = max(kt, kb)  # maximum spacing ratio

    # St4000 correlation
    St4000 = (0.002499 + 0.008261 * (xlxt - 1) - 0.000145 * (xlxt - 1) ** 2) / (kmin * xls) ** 0.35 / Pr ** (2 / 3)
    # St correlation
    St = St4000 * 25.6238 * Re ** (-0.3913)
    # htc
    h = St * Cp * G

    # 4000f correlation
    f4000 = 0.0122 * (xlxt - 1) * (3 * xlxt - 2) / (kmin * xls)
    # f correlation
    if Re > 3000:
        beta = 0.184
        f = f4000 * (Re / 4000) ** (-1 * beta)
    else:
        beta = 0.184 + 0.2820 * (1 - 2 * kmax)
        f = f4000 * (Re / 3000) ** (-1 * beta)

    return h, f


def general_hex_j_factor(reynolds: float, l_s_over_d_h: float, show_warnings: bool = False) -> float:
    """
    Calculate j-factor for general heat exchangers.
    From Milten (2024) eqn (15), based on HEx from Kays and London (1984) like LaHaye (1974).
    """
    if show_warnings and (reynolds < 2e3 or reynolds > 2e4):
        warnings.warn(f"Reynolds number {reynolds:.1e} outside correlation range of 2k-20k", stacklevel=2)

    if show_warnings and (l_s_over_d_h < 0.645 or l_s_over_d_h > 73.8):
        warnings.warn(
            f"l_s_over_d_h ratio {l_s_over_d_h:.2f} outside correlation range of 0.645-73.8",
            stacklevel=2,
        )

    return 0.360 * l_s_over_d_h**-0.401 * reynolds**-0.413 + 2.13e-5 * l_s_over_d_h


def general_hex_friction_factor(reynolds: float, l_s_over_d_h: float, show_warnings: bool = False) -> float:
    """
    Calculate friction factor for general heat exchangers.
    From Milten (2024) eqn (16), based on HEx from Kays and London (1984) like LaHaye (1974).
    """
    if show_warnings and (reynolds < 2e3 or reynolds > 2e4):
        warnings.warn(f"Reynolds number {reynolds:.1e} outside correlation range of 2k-20k", stacklevel=2)

    if show_warnings and (l_s_over_d_h < 0.645 or l_s_over_d_h > 73.8):
        warnings.warn(
            f"l_s_over_d_h ratio {l_s_over_d_h:.2f} outside correlation range of 0.645-73.8",
            stacklevel=2,
        )

    return 0.492 * l_s_over_d_h**-0.501 * reynolds**-0.232
