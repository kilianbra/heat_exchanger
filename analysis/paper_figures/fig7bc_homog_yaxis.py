"""
Re-run fig7b/c aspect-ratio (Ao/Aoref) plots with shared y-axis: -20% to 30%.
Saves as fig7b_homog_yaxis and fig7c_homog_yaxis in Figs_current/.
"""

from fig7b_aspect_ratio import (
    DEFAULT_A_R,
    DEFAULT_C_COLD_OVER_C_HOT,
    DEFAULT_D_R,
    DEFAULT_DP_MAX,
    DEFAULT_F_C_OVER_F_H,
    DEFAULT_G2_H,
    DEFAULT_GAMMA,
    DEFAULT_MOLAR_MASS_RATIO,
    DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    DEFAULT_P_DEAD_OVER_P_HOT_IN,
    DEFAULT_PRESSURE_DROP_ASSUMPTION,
    DEFAULT_ST_OVER_F,
    DEFAULT_T,
    DEFAULT_T_DEAD_OVER_T_COLD_IN,
    save_figures as save_fig7b,
)
from fig7c_aspect_ratio import save_figures as save_fig7c

HOMOG_YLIM = (-0.2, 0.3)
# Label positions for homog y-axis (data coords: Ao_ref/Ao, availability fraction)
HOMOG_XYTEXT_BASELINE = (0.5, 0.10)  # 10%, shifted right (classical only)
HOMOG_XYTEXT_OPTIMAL = (0.25, 0.27)  # 27% (practical only)

_COMMON = dict(
    c_cold_over_c_hot=DEFAULT_C_COLD_OVER_C_HOT,
    st_over_f=DEFAULT_ST_OVER_F,
    f_c_over_f_h=DEFAULT_F_C_OVER_F_H,
    d_r=DEFAULT_D_R,
    g2_h=DEFAULT_G2_H,
    dp_max=DEFAULT_DP_MAX,
    t=DEFAULT_T,
    t_dead_over_t_cold_in=DEFAULT_T_DEAD_OVER_T_COLD_IN,
    p_cold_in_over_p_hot_in=DEFAULT_P_COLD_IN_OVER_P_HOT_IN,
    p_dead_over_p_hot_in=DEFAULT_P_DEAD_OVER_P_HOT_IN,
    gamma=DEFAULT_GAMMA,
    pressure_drop_assumption=DEFAULT_PRESSURE_DROP_ASSUMPTION,
    molar_mass_ratio=DEFAULT_MOLAR_MASS_RATIO,
    a_r=DEFAULT_A_R,
    plot_area_ratio_ref=True,
    ylim=HOMOG_YLIM,
)

if __name__ == "__main__":
    save_fig7b(
        **_COMMON,
        base_name="fig7b_homog_yaxis",
        xytext_baseline=HOMOG_XYTEXT_BASELINE,
        top_axis_label="reducing diffusion",
    )
    save_fig7c(
        **_COMMON,
        base_name="fig7c_homog_yaxis",
        xytext_optimal=HOMOG_XYTEXT_OPTIMAL,
        show_top_axis_label=False,
    )
