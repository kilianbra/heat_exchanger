import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.widgets import Button, Slider

from heat_exchanger.fluids.protocols import FluidInputs, PerfectGasFluid
from heat_exchanger.geometries.general_counterflow import rate_hex_simple

case = "B"  # "Brewer"
if case == "Heli":
    # Fluid models (global)
    FLUID_HOT = PerfectGasFluid.from_name("kerocomb_helicopter")
    FLUID_COLD = PerfectGasFluid.from_name("air")

    # Initial values for sliders
    INIT_T_OVER_DHC = 0.02  # t/d_h_c = 0.02 (85 micron t over 4 mm walls)
    INIT_SIGMA_R = 2.0  # Ratio of free flow areas (Ao_h/Ao_c)
    INIT_SIGMA_W = 2.0  # Ratio of heat transfer areas (Ah/Ac)
    INIT_D_H_C = 4e-3  # m, cold side hydraulic diameter
    INIT_LS_OVER_DH = 5.0  # Strip length to hydraulic diameter ratio
    INIT_AQ_BASELINE = 43.0
    INIT_A_FR_BASELINE = 5.0

    # Fluid inputs initial values
    INIT_M_DOT_HOT = 1.6  # kg/s
    INIT_M_DOT_COLD = 1.6  # kg/s
    INIT_TH_IN = 980  # K
    INIT_PH_IN = 1.06e5  # Pa (1.06 bar)
    INIT_TC_IN = 576  # K
    INIT_PC_IN = 7.2e5  # Pa (7.2 bar)
    INIT_DP_MAX = 0.2

else:
    # Fluid models (global)
    FLUID_COLD = PerfectGasFluid.from_name("para_h2")
    FLUID_HOT = PerfectGasFluid(
        M=27.5, S=150.0, T_ref=350.0, mu_ref=1.12e-5, gamma=1.37, Pr=0.74, cp=1170.0
    )  # 1170 from dT

    INIT_M_DOT_HOT = 19.07  # kg/s
    INIT_M_DOT_COLD = 0.166  # kg/s
    INIT_TH_IN = 778  # K
    INIT_PH_IN = 0.388e5  # Pa (1.06 bar)
    INIT_TC_IN = 264  # K
    INIT_PC_IN = 16.8e5  # Pa (7.2 bar)

    # Geometry
    INIT_T_OVER_DHC = 0.06  # t/d_h_c = 0.02 (85 micron t over 4 mm walls)
    INIT_SIGMA_R = 250.0  # Ratio of free flow areas (Ao_h/Ao_c)
    INIT_SIGMA_W = 1.14  # Ratio of heat transfer areas (Ah/Ac)
    INIT_A_FR_OVER_AO_C = (1 + INIT_SIGMA_R) + 2 * INIT_T_OVER_DHC * (
        1 + INIT_SIGMA_W
    )  # Ratio of frontal area to cold side free flow area
    INIT_D_H_C = 1e-3  # m, cold side hydraulic diameter
    INIT_LS_OVER_DH = 60.0  # Strip length to hydraulic diameter ratio
    INIT_AQ_BASELINE = 16.0
    INIT_A_FR_BASELINE = 1.0
    INIT_DP_MAX = 0.15


F_IN = FluidInputs(
    hot=FLUID_HOT,
    cold=FLUID_COLD,
    m_dot_hot=INIT_M_DOT_HOT,
    m_dot_cold=INIT_M_DOT_COLD,
    Th_in=INIT_TH_IN,
    Ph_in=INIT_PH_IN,
    Tc_in=INIT_TC_IN,
    Pc_in=INIT_PC_IN,
)


# Initial boolean values
INIT_PLOT_AQ_SWEEP_NOT_AFR = True
INIT_PDOT_NOT_DP = False


def calculate_plot(
    plot_aq_sweep_not_afr,
    plot_pdot_not_dp,
    t_over_dhc,
    sigma_r,
    sigma_w,
    d_h_c,
    ls_over_dh,
    aq_baseline,
    a_fr_baseline,
    f_in,
    dp_max,
):
    """Calculate and return plot data"""
    # Create fluid inputs

    # Generate x values
    if plot_aq_sweep_not_afr:
        if case == "Heli":
            x = np.geomspace(2, 2000 / a_fr_baseline, 100)
        else:
            if a_fr_baseline < 1:
                x = np.geomspace(0.5, 500 / a_fr_baseline, 200)
            else:
                x = np.geomspace(0.5, 500 * a_fr_baseline, 200)
        Aq = x
        A_fr = a_fr_baseline
        title = "Heat transfer area variation Aq (m²) (HEx mass, length changes)"
    else:
        x = np.geomspace(a_fr_baseline, 0.001, 100)
        Aq = aq_baseline
        A_fr = x
        title = "Frontal area variation A_fr (m²) (Velocity, g^2)"
    # Calculate results
    r_s = rate_hex_simple(
        A_fr=A_fr,
        A_q=Aq,
        f_in=f_in,
        d_h_c=d_h_c,
        sigma_r=sigma_r,
        sigma_w=sigma_w,
        t_over_dhc=t_over_dhc,
        ls_over_dh=ls_over_dh,
    )

    # Find validity mask
    dp_hot = r_s["dp_hot"]
    dp_cold = r_s["dp_cold"]
    eps = r_s["eps"]
    if plot_pdot_not_dp:
        hot_in = f_in.hot.state(T=f_in.Th_in, P=f_in.Ph_in)
        cold_in = f_in.cold.state(T=f_in.Tc_in, P=f_in.Pc_in)
        Cmin = min(f_in.m_dot_hot * hot_in.cp, f_in.m_dot_cold * cold_in.cp)
        Q_max = Cmin * (f_in.Th_in - f_in.Tc_in)
        y_hot = f_in.m_dot_hot * dp_hot * f_in.Ph_in / hot_in.rho / Q_max
        y_cold = f_in.m_dot_cold * dp_cold * f_in.Pc_in / cold_in.rho / Q_max
    else:
        y_hot = dp_hot
        y_cold = dp_cold
    over_limit = (dp_hot >= dp_max) | (dp_cold >= dp_max) | (dp_hot < 0)
    if np.any(over_limit):
        idx_exceed = np.where(over_limit)[0]
        print("Exceeded dp_max or negative dp at indices:", idx_exceed)
        for i in idx_exceed[:5]:
            reasons = []
            if dp_hot[i] >= dp_max:
                reasons.append(f"dp_hot[{i}]={dp_hot[i]:.3g} >= dp_max={dp_max:.3g}")
            if dp_cold[i] >= dp_max:
                reasons.append(f"dp_cold[{i}]={dp_cold[i]:.3g} >= dp_max={dp_max:.3g}")
            if dp_hot[i] < 0:
                reasons.append(f"dp_hot[{i}]={dp_hot[i]:.3g} < 0")
            print(f"  At index {i}: " + ", ".join(reasons))
    if np.any(over_limit):
        first_exceed = np.argmax(over_limit)
        validity_mask = np.zeros_like(dp_hot, dtype=bool)
        validity_mask[:first_exceed] = True
    else:
        validity_mask = np.ones_like(dp_hot, dtype=bool)

    x_plot = r_s["ntu"]
    x_title = "NTU"

    return x_plot, eps, y_hot, y_cold, validity_mask, x_title, title, r_s


if __name__ == "__main__":
    # Create figure with space for sliders on the right
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    plt.subplots_adjust(right=0.75)  # Make room for sliders on the right

    # Initial plot
    x_plot, eps, y_hot, y_cold, validity_mask, x_title, title, r_s = calculate_plot(
        INIT_PLOT_AQ_SWEEP_NOT_AFR,
        INIT_PDOT_NOT_DP,
        INIT_T_OVER_DHC,
        INIT_SIGMA_R,
        INIT_SIGMA_W,
        INIT_D_H_C,
        INIT_LS_OVER_DH,
        INIT_AQ_BASELINE,
        INIT_A_FR_BASELINE,
        F_IN,
        INIT_DP_MAX,
    )

    if INIT_PLOT_AQ_SWEEP_NOT_AFR:
        print(
            f"NTU = {x_plot[0]:.2f}, g2_h: {r_s['g2_hot']:.2e}, g2_c: {r_s['g2_cold']:.2e},Rc {r_s['R_cold_over_R_tot'][0] * 100:.0f}%, Re_h: {r_s['re_hot']:.2e}, Re_c: {r_s['re_cold']:.2e}"
        )
    else:
        print(
            f"NTU = {x_plot[0]:.2f}, g2_h: {r_s['g2_hot'][0]:.2e}, g2_c: {r_s['g2_cold'][0]:.2e},Rc {r_s['R_cold_over_R_tot'][0] * 100:.0f}%, Re_h: {r_s['re_hot'][0]:.2e}, Re_c: {r_s['re_cold'][0]:.2e}"
        )
        print(
            f"NTU = {x_plot[validity_mask][-1]:.2f}, g2_h: {r_s['g2_hot'][validity_mask][-1]:.2e}, g2_c: {r_s['g2_cold'][validity_mask][-1]:.2e},Rc {r_s['R_cold_over_R_tot'][validity_mask][-1] * 100:.0f}%, Re_h: {r_s['re_hot'][validity_mask][-1]:.2e}, Re_c: {r_s['re_cold'][validity_mask][-1]:.2e}"
        )

    (line,) = ax.plot(x_plot[validity_mask], eps[validity_mask], "-", label="eps")
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    y_lab = "Pdot / Q_max" if INIT_PDOT_NOT_DP else "dp"
    (line2,) = ax2.plot(x_plot[validity_mask], y_hot[validity_mask], "r--", label=f"{y_lab}_hot")
    (line3,) = ax2.plot(x_plot[validity_mask], y_cold[validity_mask], "g--", label=f"{y_lab}_cold")
    ax2.set_ylim(0, INIT_DP_MAX)
    ax.legend()
    ax.set_xlabel(x_title)
    ax.set_ylabel("eps")
    ax2.set_ylabel(y_lab)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

    ax.set_title(title)

    # Create sliders on the right side
    slider_height = 0.03
    slider_spacing = 0.03
    start_y = 0.95

    # Slider positions (right side)
    slider_left = 0.86
    slider_width = 0.1

    # region Create sliders
    y_pos = start_y
    slider_t_over_dhc = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "T_OVER_DHC",
        0.001,
        0.1,
        valinit=INIT_T_OVER_DHC,
        valfmt="%.4f",
    )
    y_pos -= slider_spacing

    slider_sigma_r = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "SIGMA_R",
        0.1,
        10.0 if case == "Heli" else 1000.0,
        valinit=INIT_SIGMA_R,
        valfmt="%.2f",
    )
    y_pos -= slider_spacing

    slider_sigma_w = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "SIGMA_W",
        0.1,
        10.0,
        valinit=INIT_SIGMA_W,
        valfmt="%.2f",
    )
    y_pos -= slider_spacing

    slider_d_h_c = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "D_H_C (m)",
        1e-4,
        0.01 if case == "Heli" else 5e-2,
        valinit=INIT_D_H_C,
        valfmt="%.4f",
    )
    y_pos -= slider_spacing

    slider_ls_over_dh = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "LS_OVER_DH",
        0.7,
        70,
        valinit=INIT_LS_OVER_DH,
        valfmt="%.2f",
    )
    y_pos -= slider_spacing

    slider_aq_baseline = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "AQ_BASELINE",
        1.0,
        200.0,
        valinit=INIT_AQ_BASELINE,
        valfmt="%.1f",
    )
    y_pos -= slider_spacing

    slider_a_fr_baseline = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "A_FR_BASELINE",
        0.01,
        10.0,
        valinit=INIT_A_FR_BASELINE,
        valfmt="%.2f",
    )
    y_pos -= slider_spacing

    # Fluid input sliders
    slider_m_dot_hot = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "m_dot_hot",
        0.1,
        10.0 if case == "Heli" else 30.0,
        valinit=INIT_M_DOT_HOT,
        valfmt="%.2f",
    )
    y_pos -= slider_spacing

    slider_m_dot_cold = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "m_dot_cold",
        0.1,
        10.0 if case == "Heli" else 30.0,
        valinit=INIT_M_DOT_COLD,
        valfmt="%.2f",
    )
    y_pos -= slider_spacing

    slider_th_in = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "Th_in (K)",
        300,
        2000,
        valinit=INIT_TH_IN,
        valfmt="%.0f",
    )
    y_pos -= slider_spacing

    slider_ph_in = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "Ph_in (Pa)",
        1e4,
        1e6,
        valinit=INIT_PH_IN,
        valfmt="%.0e",
    )
    y_pos -= slider_spacing

    slider_tc_in = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "Tc_in (K)",
        200,
        1000,
        valinit=INIT_TC_IN,
        valfmt="%.0f",
    )
    y_pos -= slider_spacing

    slider_pc_in = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "Pc_in (Pa)",
        1e4,
        1e6 if case == "Heli" else 3e6,
        valinit=INIT_PC_IN,
        valfmt="%.0e",
    )
    y_pos -= slider_spacing

    slider_dp_max = Slider(
        plt.axes([slider_left, y_pos, slider_width, slider_height]),
        "DP_MAX",
        0.2 if case == "Heli" else 0.1,
        0.8 if case == "Heli" else 0.3,
        valinit=INIT_DP_MAX,
        valfmt="%.2f",
    )
    y_pos -= slider_spacing * 2

    # Create buttons for booleans
    button_height = 0.04
    button_spacing = 0.05

    button_aq_sweep = Button(
        plt.axes([slider_left, y_pos, slider_width, button_height]), f"AQ Sweep: {INIT_PLOT_AQ_SWEEP_NOT_AFR}"
    )
    y_pos -= button_spacing

    button_pdot_not_dp = Button(
        plt.axes([slider_left, y_pos, slider_width, button_height]), f"Pdot / Q_max: {INIT_PDOT_NOT_DP}"
    )

    # endregion Create sliders

    # Store boolean states
    plot_aq_sweep_not_afr = INIT_PLOT_AQ_SWEEP_NOT_AFR
    plot_pdot_not_dp = INIT_PDOT_NOT_DP

    def update_plot(val=None):
        """Update the plot when any slider changes"""
        global plot_aq_sweep_not_afr, plot_pdot_not_dp

        f_in = FluidInputs(
            hot=FLUID_HOT,
            cold=FLUID_COLD,
            m_dot_hot=slider_m_dot_hot.val,
            m_dot_cold=slider_m_dot_cold.val,
            Th_in=slider_th_in.val,
            Ph_in=slider_ph_in.val,
            Tc_in=slider_tc_in.val,
            Pc_in=slider_pc_in.val,
        )

        x_plot, eps, y_hot, y_cold, validity_mask, x_title, title, r_s = calculate_plot(
            plot_aq_sweep_not_afr,
            plot_pdot_not_dp,
            slider_t_over_dhc.val,
            slider_sigma_r.val,
            slider_sigma_w.val,
            slider_d_h_c.val,
            slider_ls_over_dh.val,
            slider_aq_baseline.val,
            slider_a_fr_baseline.val,
            f_in,
            slider_dp_max.val,
        )

        if plot_aq_sweep_not_afr:
            print(
                f"NTU = {x_plot[0]:.2f}, g2_h: {r_s['g2_hot']:.2e}, g2_c: {r_s['g2_cold']:.2e},Rc {r_s['R_cold_over_R_tot'][0] * 100:.0f}%, Re_h: {r_s['re_hot']:.2e}, Re_c: {r_s['re_cold']:.2e}"
            )
        else:
            print(
                f"NTU = {x_plot[0]:.2f}, g2_h: {r_s['g2_hot'][0]:.2e}, g2_c: {r_s['g2_cold'][0]:.2e},Rc {r_s['R_cold_over_R_tot'][0] * 100:.0f}%, Re_h: {r_s['re_hot'][0]:.2e}, Re_c: {r_s['re_cold'][0]:.2e}"
            )
            print(
                f"NTU = {x_plot[validity_mask][-1]:.2f}, g2_h: {r_s['g2_hot'][validity_mask][-1]:.2e}, g2_c: {r_s['g2_cold'][validity_mask][-1]:.2e},Rc {r_s['R_cold_over_R_tot'][validity_mask][-1] * 100:.0f}%, Re_h: {r_s['re_hot'][validity_mask][-1]:.2e}, Re_c: {r_s['re_cold'][validity_mask][-1]:.2e}"
            )

        line.set_data(x_plot[validity_mask], eps[validity_mask])
        line2.set_data(x_plot[validity_mask], y_hot[validity_mask])
        line3.set_data(x_plot[validity_mask], y_cold[validity_mask])
        ax.set_xlabel(x_title)
        y_lab = "Pdot / Q_max" if plot_pdot_not_dp else "dp/p_in"
        ax2.set_ylabel(y_lab)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_title(title)
        ax2.relim()
        ax.relim()
        ax2.set_ylim(0, slider_dp_max.val)
        ax.set_ylim(0, 1)
        if not plot_aq_sweep_not_afr:
            # Find first invalid point to show up to that point
            if np.any(~validity_mask):
                first_invalid_idx = np.where(~validity_mask)[0][0]
                x_max = x_plot[first_invalid_idx]
            else:
                x_max = x_plot[-1]
            ax.set_xlim(0, x_max)

        ax.autoscale_view()
        # Combine legend handles and labels from ax and ax2
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        ax.legend(handles, labels, loc="upper left")
        fig.canvas.draw_idle()

    def toggle_aq_sweep(event):
        """Toggle PLOT_AQ_SWEEP_NOT_AFR boolean"""
        global plot_aq_sweep_not_afr
        plot_aq_sweep_not_afr = not plot_aq_sweep_not_afr
        button_aq_sweep.label.set_text(f"AQ Sweep: {plot_aq_sweep_not_afr}")
        update_plot()

    def toggle_pdot_not_dp(event):
        """Toggle PLOT_PDOT_NOT_DP boolean"""
        global plot_pdot_not_dp
        plot_pdot_not_dp = not plot_pdot_not_dp
        button_pdot_not_dp.label.set_text(f"Pdot / Q_max: {plot_pdot_not_dp}")
        update_plot()

    # Connect sliders to update function
    slider_t_over_dhc.on_changed(update_plot)
    slider_sigma_r.on_changed(update_plot)
    slider_sigma_w.on_changed(update_plot)
    slider_d_h_c.on_changed(update_plot)
    slider_ls_over_dh.on_changed(update_plot)
    slider_aq_baseline.on_changed(update_plot)
    slider_a_fr_baseline.on_changed(update_plot)
    slider_m_dot_hot.on_changed(update_plot)
    slider_m_dot_cold.on_changed(update_plot)
    slider_th_in.on_changed(update_plot)
    slider_ph_in.on_changed(update_plot)
    slider_tc_in.on_changed(update_plot)
    slider_pc_in.on_changed(update_plot)
    slider_dp_max.on_changed(update_plot)

    # Connect buttons to toggle functions
    button_aq_sweep.on_clicked(toggle_aq_sweep)
    button_pdot_not_dp.on_clicked(toggle_pdot_not_dp)

    plt.show()
