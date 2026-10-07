"""
Streamlit explorer for T* in adiabatic turbomachines vs polytropic efficiency
and pressure ratio.

T* = Δh_se / Δs is the mean isentropic-equivalent temperature from inlet to exit.
"""

import numpy as np


def k_from_gamma(gamma: float) -> float:
    """R/c_p for a perfect gas."""
    return (gamma - 1) / gamma


def tau_log_ratio(tau: np.ndarray) -> np.ndarray:
    """Evaluate (tau - 1) / ln(tau); limit is 1 at tau = 1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (tau - 1.0) / np.log(tau)
    is_one = np.abs(tau - 1.0) < 1e-10
    ratio = np.where(is_one, 1.0, ratio)
    invalid = (tau <= 0.0) | ~np.isfinite(ratio)
    return np.where(invalid, np.nan, ratio)


def compressor_tau(pressure_ratio: np.ndarray, eta_p: np.ndarray, *, gamma: float) -> np.ndarray:
    """T2s/T2 for a compressor: (p2/p1)^(k*(1 - 1/eta_p)) with p2/p1 = pressure_ratio."""
    k = k_from_gamma(gamma)
    return pressure_ratio ** (k * (1.0 - 1.0 / eta_p))


def turbine_tau(pressure_ratio: np.ndarray, eta_p: np.ndarray, *, gamma: float) -> np.ndarray:
    """T2s/T2 for a turbine: (p2/p1)^(k*(1 - eta_p)) with p2/p1 = 1/pressure_ratio."""
    k = k_from_gamma(gamma)
    return (1.0 / pressure_ratio) ** (k * (1.0 - eta_p))


def compressor_t_star_over_t1(
    pressure_ratio: np.ndarray,
    eta_p: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    """
    Compressor map with inlet at the dead/reference state (T1 = T0, p1 = p0).

    T*/T1 = (p2/p1)^(k(1/eta_p - 1)) * (T2s/T2 - 1) / ln(T2s/T2)
    """
    k = k_from_gamma(gamma)
    tau = compressor_tau(pressure_ratio, eta_p, gamma=gamma)
    prefactor = pressure_ratio ** (k * (1.0 / eta_p - 1.0))
    return prefactor * tau_log_ratio(tau)


def turbine_t_star_over_t2(
    pressure_ratio: np.ndarray,
    eta_p: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    """
    Turbine map: T*/T2 = (T2s/T2 - 1) / ln(T2s/T2) with p2/p1 = 1/PR.
    """
    tau = turbine_tau(pressure_ratio, eta_p, gamma=gamma)
    return tau_log_ratio(tau)


# Contour levels (values of T*/T1 or T*/T2)
# Compressor T*/T1 spans ~1.00–1.15 on the default grid (inlet = reference state).
COMPRESSOR_CONTOUR_LEVELS = tuple(np.round(np.arange(1.00, 1.151, 0.02), 2))
TURBINE_CONTOUR_LEVELS = tuple(np.round(np.arange(0.90, 1.001, 0.01), 2))


def line_contour_figure(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_grid: np.ndarray,
    levels: tuple[float, ...],
    *,
    labelformat: str = ".2f",
):
    """Single panel: black labeled contour lines, plain-text axes only."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for level in levels:
        fig.add_trace(
            go.Contour(
                x=x_axis,
                y=y_axis,
                z=z_grid,
                showscale=False,
                hoverinfo="skip",
                contours_coloring="lines",
                line=dict(color="black", width=1),
                contours=dict(
                    start=level,
                    end=level,
                    coloring="lines",
                    showlabels=True,
                    labelfont=dict(size=10, color="black"),
                    labelformat=labelformat,
                ),
            )
        )
    fig.update_layout(
        xaxis_title="Polytropic efficiency (%)",
        yaxis_title="Pressure ratio",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=550,
        margin=dict(l=60, r=20, t=20, b=60),
    )
    return fig


if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(page_title="T* in adiabatic turbomachines", layout="wide")
    st.title("Mean isentropic-equivalent temperature in adiabatic turbomachines")

    st.markdown(
        r"""
$T^*$ is the mean isentropic-equivalent temperature from inlet to exit,

$$
T^* = \frac{\Delta h_{se}}{\Delta s}
$$

It is the temperature that should weight irreversible entropy production in an adiabatic
turbomachine so that the loss in mechanical work potential is book-kept correctly.
Classical availability analysis uses the dead-state temperature $T_0$ instead.
"""
    )

    st.latex(
        r"k = \frac{\gamma - 1}{\gamma}, \qquad "
        r"\frac{T_{2s}}{T_2} = \left(\frac{p_2}{p_1}\right)^{k(1 - 1/\eta_p)} \text{ (compressor)}, \qquad "
        r"\frac{T_{2s}}{T_2} = \left(\frac{p_2}{p_1}\right)^{k(1 - \eta_p)} \text{ (turbine)}"
    )

    st.sidebar.header("Parameters")
    gamma = st.sidebar.slider(
        "Gamma",
        min_value=1.33,
        max_value=1.40,
        value=1.40,
        step=0.01,
    )
    n_points = st.sidebar.slider("Grid resolution", min_value=100, max_value=500, value=250, step=25)

    k = k_from_gamma(gamma)
    st.sidebar.metric("k = (gamma - 1) / gamma", f"{k:.4f}")

    eta_p_axis = np.linspace(0.80, 1.00, n_points)
    pr_axis = np.linspace(1.0, 50.0, n_points)
    eta_grid, pr_grid = np.meshgrid(eta_p_axis, pr_axis)
    eta_pct_axis = eta_p_axis * 100.0

    z_compressor = compressor_t_star_over_t1(pr_grid, eta_grid, gamma=gamma)
    z_turbine = turbine_t_star_over_t2(pr_grid, eta_grid, gamma=gamma)

    col_compressor, col_turbine = st.columns(2)

    with col_compressor:
        st.subheader("Compressor — relative to inlet / reference temperature")
        st.caption("Inlet taken as the dead/reference state (T1 = T0, p1 = p0).")
        st.latex(
            r"\frac{T^*}{T_1} = \left(\frac{p_2}{p_1}\right)^{k(1/\eta_p - 1)} "
            r"\frac{T_{2s}/T_2 - 1}{\ln(T_{2s}/T_2)}, \qquad "
            r"\frac{p_2}{p_1} = \mathrm{PR}"
        )
        if np.any(np.isfinite(z_compressor)):
            finite_c = z_compressor[np.isfinite(z_compressor)]
            st.caption(
                f"Plotted range of T*/T1 on this grid: "
                f"{finite_c.min():.3f} to {finite_c.max():.3f} "
                f"(contours every 0.02 from 1.00)."
            )
            st.plotly_chart(
                line_contour_figure(
                    eta_pct_axis,
                    pr_axis,
                    z_compressor,
                    COMPRESSOR_CONTOUR_LEVELS,
                ),
                use_container_width=True,
            )
        else:
            st.warning("No finite compressor values in the selected range.")

    with col_turbine:
        st.subheader("Turbine — relative to exit temperature")
        st.caption(
            "T*/T2 describes loss weighting at the machine exit. "
            "A dead-state comparison also depends on T2 relative to T0, "
            "so this map is shown only on an exit-temperature basis."
        )
        st.latex(
            r"\frac{T^*}{T_2} = \frac{T_{2s}/T_2 - 1}{\ln(T_{2s}/T_2)}, \qquad "
            r"\frac{p_2}{p_1} = 1/\mathrm{PR}"
        )
        if np.any(np.isfinite(z_turbine)):
            st.plotly_chart(
                line_contour_figure(
                    eta_pct_axis,
                    pr_axis,
                    z_turbine,
                    TURBINE_CONTOUR_LEVELS,
                ),
                use_container_width=True,
            )
        else:
            st.warning("No finite turbine values in the selected range.")

    st.caption(
        r"For PR $> 1$, $T_{2s}/T_2 < 1$ on both maps. "
        r"At $\eta_p = 100\%$, $T_{2s}/T_2 \to 1$; "
        r"then $T^*/T_1 \to 1$ (compressor) and $T^*/T_2 \to 1$ (turbine)."
    )
