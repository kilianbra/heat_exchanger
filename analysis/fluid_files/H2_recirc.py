"""Plot H2 recirculation fraction as a function of H2 HEx exit temperature.

This script plots the Recirculation Fraction (Pure recirculated H2/ combusted H2)
as a function of the H2 HEx exit temperature, comparing constant cp model
with CoolProp modeling.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button, Slider

from heat_exchanger.fluids.protocols import CoolPropFluid

BAR_TO_PA = 1e5


def calculate_recirc_fraction_cp_const(T_pe: float, T_cin: float, T_cout: np.ndarray) -> np.ndarray:
    """Calculate recirculation fraction using constant cp assumption.

    Formula: (T_cin - T_pe) / (T_cout - T_cin)

    Parameters
    ----------
    T_pe : float
        Pump exit temperature [K]
    T_cin : float
        HEx inlet H2 desired temperature (after mixing) [K]
    T_cout : np.ndarray
        HEx exit temperature (recirculated + combusted) [K]

    Returns
    -------
    np.ndarray
        Recirculation fraction
    """
    return (T_cin - T_pe) / (T_cout - T_cin)


def calculate_recirc_fraction_coolprop(T_pe: float, T_cin: float, T_cout: np.ndarray, P_bar: float) -> np.ndarray:
    """Calculate recirculation fraction using CoolProp enthalpy.

    Formula: (h_cin - h_pe) / (h_cout - h_cin)

    Parameters
    ----------
    T_pe : float
        Pump exit temperature [K]
    T_cin : float
        HEx inlet H2 desired temperature (after mixing) [K]
    T_cout : np.ndarray
        HEx exit temperature (recirculated + combusted) [K]
    P_bar : float
        H2 pressure [bar]

    Returns
    -------
    np.ndarray
        Recirculation fraction
    """
    # if Tcout is just a single value, need to convert to array
    if isinstance(T_cout, float):
        T_cout = np.array([T_cout])

    P_pa = P_bar * BAR_TO_PA
    h2_fluid = CoolPropFluid("ParaHydrogen")

    # Get enthalpies at the four temperatures
    h_pe = h2_fluid.state(T=T_pe, P=P_pa).h
    h_cin = h2_fluid.state(T=T_cin, P=P_pa).h

    # Calculate h_cout for each T_cout value
    h_cout = np.array([h2_fluid.state(T=T, P=P_pa).h for T in T_cout])

    # Calculate recirculation fraction
    return (h_cin - h_pe) / (h_cout - h_cin)


def main():
    # Initial values
    init_T_pe = 40.0  # K
    init_T_cin = 275.0  # K Brewer 200 K, AHJE 275-290
    init_P_bar = 30.0  # bar

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.7, bottom=0.1, top=0.9)

    # Initial calculation
    T_cout = np.linspace(init_T_cin + 20, 600.0, 500)

    recirc_cp_const = calculate_recirc_fraction_cp_const(init_T_pe, init_T_cin, T_cout)
    recirc_coolprop = calculate_recirc_fraction_coolprop(init_T_pe, init_T_cin, T_cout, init_P_bar)

    # Plot initial lines
    (line_cp_const,) = ax.plot(T_cout, recirc_cp_const, "b-", label="cp = const", linewidth=2)
    (line_coolprop,) = ax.plot(T_cout, recirc_coolprop, "r-", label="CoolProp", linewidth=2)

    # Set axis labels and limits
    ax.set_xlabel("H2 HEx exit temperature (recirculated + combusted) i.e. T_cout [K]", fontsize=11)
    ax.set_ylabel("Recirculation Fraction (Pure recirculated H2 / combusted H2)", fontsize=11)
    ax.set_xlim(0, 600)
    ax.set_yscale("log")

    # Format y-axis to use .0f format instead of scientific notation
    def log_formatter(x, pos):
        return f"{x:.2f}"

    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Recalculate button
    ax_btn = plt.axes([0.77, 0.85, 0.20, 0.04])
    btn_recalc = Button(ax_btn, "Recalculate")
    btn_recalc.ax.set_facecolor("#d0f0d0")

    # Slider axes (stacked on right side)
    axcolor = "0.95"
    slider_width = 0.20
    slider_height = 0.03
    slider_spacing = 0.05
    right_margin = 0.77
    top_start = 0.75

    # Text labels for sliders
    ax_text_T_pe = plt.axes([right_margin, top_start - slider_spacing * 0, slider_width, 0.02])
    ax_text_T_pe.axis("off")
    ax_text_T_pe.text(0.5, 0.5, "pump exit temperature", ha="center", va="center", fontsize=9)

    ax_text_T_cin = plt.axes([right_margin, top_start - slider_spacing * 2, slider_width, 0.02])
    ax_text_T_cin.axis("off")
    ax_text_T_cin.text(0.5, 0.5, "HEx inlet H2 desired Temp (after mixing)", ha="center", va="center", fontsize=9)

    # Slider for T_pe
    s_T_pe = plt.axes([right_margin, top_start - slider_spacing * 1, slider_width, slider_height], facecolor=axcolor)
    sl_T_pe = Slider(s_T_pe, r"$T_{p,e}$ [K]", 20.0, 50.0, valinit=init_T_pe, valstep=1.0)

    # Slider for T_cin
    s_T_cin = plt.axes([right_margin, top_start - slider_spacing * 3, slider_width, slider_height], facecolor=axcolor)
    sl_T_cin = Slider(s_T_cin, r"$T_{c,in}$ [K]", 50.0, 300.0, valinit=init_T_cin, valstep=1.0)

    # Slider for H2 pressure
    s_P = plt.axes([right_margin, top_start - slider_spacing * 4, slider_width, slider_height], facecolor=axcolor)
    sl_P = Slider(s_P, "P [bar]", 15.0, 150.0, valinit=init_P_bar, valstep=1.0)

    def recalculate(_):
        """Recalculate and update the plot."""
        T_pe = sl_T_pe.val
        T_cin = sl_T_cin.val
        P_bar = sl_P.val

        # Ensure T_cout starts from T_cin
        T_cout = np.linspace(T_cin + 20, 600.0, 500)

        # Calculate recirculation fractions
        recirc_cp_const = calculate_recirc_fraction_cp_const(T_pe, T_cin, T_cout)
        recirc_coolprop = calculate_recirc_fraction_coolprop(T_pe, T_cin, T_cout, P_bar)

        # Update plot lines
        line_cp_const.set_data(T_cout, recirc_cp_const)
        line_coolprop.set_data(T_cout, recirc_coolprop)

        # Update y-axis limits to fit the data
        # all_values = np.concatenate([recirc_cp_const, recirc_coolprop])
        # valid_values = all_values[np.isfinite(all_values)]

        ax.set_ylim(0.1, 10.0)

        fig.canvas.draw_idle()

    # Connect button to recalculate function
    btn_recalc.on_clicked(recalculate)

    # Initial plot
    recalculate(None)

    plt.show()


if __name__ == "__main__":
    main()
