import numpy as np

SHOW_ALL_BAR_ELEMENTS = False

# Global variable containing all possible component names in order
ALL_COMPONENTS = [
    "Compressor",
    "Intercooler_hot",
    "Compressor_2",
    "Recup_cold",
    "Combustion",
    "Turbine",
    "Recup_hot",
    "Exhaust",
]

# Global variable for component colors (matches ALL_COMPONENTS order)
COMPONENT_COLORS = ["blue", "cyan", "darkblue", "lightblue", "red", "green", "lightcoral", "orange"]

#


def isobar(
    T_start,
    T_end,
    P_start,
    P_end=None,
    n_points=50,
    p_d=1e5,
    T_d=288,
    s_start=None,
    R=8314 / 28.97,
    c_p=8314 / 28.97 * 7 / 2,
):
    """
    Calculate entropy values for an isobar or pressure drop process

    Parameters:
    T_start, T_end: Start and end temperatures
    P_start: Start pressure
    P_end: End pressure (if None, assume constant pressure)
    n_points: Number of points
    p_d, T_d: Reference conditions for entropy calculation
    s_start: Starting entropy value (if None, calculate from reference)
    R, c_p: Gas properties

    Returns:
    T_range, s_range: Temperature and entropy arrays
    """
    T_range = np.linspace(T_start, T_end, n_points)

    # Calculate starting entropy if not provided
    if s_start is None:
        s_start = c_p * np.log(T_start / T_d) - R * np.log(P_start / p_d)

    if P_end is None or abs(P_end - P_start) < 100:
        # Constant pressure case
        s_range = s_start + c_p * np.log(T_range / T_start)
    else:
        # Pressure drop case - interpolate between inlet and outlet pressures
        s_no_dp = s_start + c_p * np.log(T_range / T_start)
        # Calculate what entropy would be at the end with pressure drop
        # s_end = s_start + c_p * np.log(T_end / T_start) - R * np.log(P_end / P_start)
        s_dp = s_start + c_p * np.log(T_range / T_start) - R * np.log(P_end / P_start)

        # Interpolate based on temperature position
        s_range = np.zeros_like(T_range)
        for i in range(n_points):
            if abs(T_end - T_start) > 1e-5:
                weight = (T_range[i] - T_start) / (T_end - T_start)
            else:
                weight = 1
            s_range[i] = (1 - weight) * s_no_dp[i] + weight * s_dp[i]

    return T_range, s_range


def main():
    pass
