"""
Turbine cooling bleed fraction and Brayton cycle model.

Implements the Holland–Thake / Young–Wilcock cooling model coupled with a
simple Brayton cycle to find optimal coolant mass flow fraction ψ.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass
class CycleParams:
    """Baseline cycle parameters."""

    T_0: float = 288.15
    p_0: float = 101325.0
    OPR: float = 40.0
    eta_pc: float = 0.90
    eta_pt: float = 0.90
    T_b_max: float = 1150.0
    C_cool: float = 0.045
    Q_R: float = 43.0e6
    eta_b: float = 0.99
    eta_mech: float = 0.99
    dp_b_p03: float = 0.05
    cp_a: float = 1005.0
    cp_g: float = 1148.0
    gamma_a: float = 1.40
    gamma_g: float = 1.33


def cooling_fraction(T_ET: float, params: CycleParams) -> float:
    """
    Compute required coolant mass fraction ψ for given turbine entry temperature.

    Mode A: Given T_ET, compute required ψ.
    ψ = C_cool * (T_ET - T_b_max) / (T_b_max - T_c_in)
    """
    T_c_in = params.T_0 * params.OPR ** ((params.gamma_a - 1) / (params.gamma_a * params.eta_pc))
    psi = params.C_cool * (T_ET - params.T_b_max) / (params.T_b_max - T_c_in)
    return max(psi, 0.0)


def evaluate_cycle(T_ET: float, psi: float | None, params: CycleParams) -> dict:
    """
    Evaluate Brayton cycle performance for given T_ET and ψ.

    If psi is None, compute it from the cooling model.
    Returns dict with F_s, SFC, eta_th, psi, T_045_mixed, V_j, etc.
    """
    if psi is None:
        psi = cooling_fraction(T_ET, params)

    # Step 1: Compressor
    T_03 = params.T_0 * params.OPR ** ((params.gamma_a - 1) / (params.gamma_a * params.eta_pc))
    w_c = params.cp_a * (T_03 - params.T_0)
    T_c_in = T_03

    # Step 2: Combustor (only (1-ψ) of core goes through combustor)
    f = (params.cp_g * T_ET - params.cp_a * T_03) / (params.eta_b * params.Q_R - params.cp_g * T_ET)
    f_total = f * (1 - psi)

    # Step 3: HPT cooled expansion + mixing
    m_hot = (1 - psi) * (1 + f)
    T_045_hot = T_ET - w_c / (params.eta_mech * m_hot * params.cp_g)

    # HPT pressure ratio
    pi_HPT = (T_ET / T_045_hot) ** (params.gamma_g * params.eta_pt / (params.gamma_g - 1))
    p_04 = params.p_0 * params.OPR * (1 - params.dp_b_p03)
    p_045 = p_04 / pi_HPT

    # Mixing with coolant
    m_total_after_mix = m_hot + psi
    cp_mix = (m_hot * params.cp_g + psi * params.cp_a) / m_total_after_mix
    T_045_mixed = (m_hot * params.cp_g * T_045_hot + psi * params.cp_a * T_c_in) / (m_total_after_mix * cp_mix)

    # R for mixed gas (approximate)
    R_mix = params.cp_g * (params.gamma_g - 1) / params.gamma_g
    gamma_mix = cp_mix / (cp_mix - R_mix)

    # Step 4: Nozzle
    NPR = p_045 / params.p_0
    T_5 = T_045_mixed * (1 / NPR) ** ((gamma_mix - 1) / gamma_mix)
    V_j = sqrt(2 * cp_mix * (T_045_mixed - T_5))

    # Step 5: Performance metrics
    F_s = V_j  # specific thrust [m/s] for ground-static
    SFC = f_total / F_s if F_s > 0 else np.inf  # kg/(N·s)
    eta_th = (F_s * V_j) / (2 * f_total * params.Q_R) if f_total > 0 else 0.0

    return {
        "F_s": F_s,
        "SFC": SFC,
        "eta_th": eta_th,
        "psi": psi,
        "T_045_mixed": T_045_mixed,
        "V_j": V_j,
        "f_total": f_total,
        "T_03": T_03,
        "T_c_in": T_c_in,
    }


def sweep_T_ET(T_ET_range: np.ndarray, params: CycleParams, use_cooling_model: bool = True) -> dict:
    """
    Sweep T_ET and compute cycle performance.

    If use_cooling_model=True, ψ is computed from cooling model for each T_ET.
    If False, sweep ψ directly (requires different interpretation - we use T_ET as driver).
    """
    F_s = []
    SFC = []
    eta_th = []
    psi_list = []
    T_ET_used = []

    for T in T_ET_range:
        psi = cooling_fraction(T, params) if use_cooling_model else None
        if use_cooling_model and psi > 0.55:
            continue  # Skip if cooling demand exceeds sweep range
        res = evaluate_cycle(T, psi, params)
        T_ET_used.append(T)
        F_s.append(res["F_s"])
        SFC.append(res["SFC"])
        eta_th.append(res["eta_th"])
        psi_list.append(res["psi"])

    return {
        "T_ET": np.array(T_ET_used),
        "F_s": np.array(F_s),
        "SFC": np.array(SFC),
        "eta_th": np.array(eta_th),
        "psi": np.array(psi_list),
    }


def sweep_psi(psi_range: np.ndarray, params: CycleParams) -> dict:
    """
    Sweep ψ and compute max allowable T_ET from cooling model, then cycle performance.

    T_ET_max(ψ) = T_b_max + (T_b_max - T_c_in) × ψ / C_cool
    """
    T_c_in = params.T_0 * params.OPR ** ((params.gamma_a - 1) / (params.gamma_a * params.eta_pc))
    T_ET_max = params.T_b_max + (params.T_b_max - T_c_in) * psi_range / params.C_cool

    F_s = []
    SFC = []
    eta_th = []

    for psi, T_ET in zip(psi_range, T_ET_max, strict=True):
        res = evaluate_cycle(T_ET, psi, params)
        F_s.append(res["F_s"])
        SFC.append(res["SFC"])
        eta_th.append(res["eta_th"])

    return {
        "psi": psi_range,
        "T_ET": T_ET_max,
        "F_s": np.array(F_s),
        "SFC": np.array(SFC),
        "eta_th": np.array(eta_th),
    }


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(page_title="Turbine Cooling Trade-off", layout="wide")
    st.title("Turbine Cooling Bleed Fraction vs Cycle Performance")

    # Sidebar: input parameters
    st.sidebar.header("Cycle parameters")

    T_04 = st.sidebar.slider(
        "Turbine Inlet Temperature (TIT) [K]",
        min_value=1400,
        max_value=2200,
        value=1900,
        step=25,
        help="TET = T_04",
    )
    OPR = st.sidebar.slider(
        "Overall Pressure Ratio (OPR)",
        min_value=20.0,
        max_value=60.0,
        value=40.0,
        step=2.0,
    )
    T_b_max = st.sidebar.slider(
        "Max blade metal temperature [K]",
        min_value=1000,
        max_value=1300,
        value=1150,
        step=25,
        help="With TBC: ~1150 K",
    )
    C_cool = st.sidebar.slider(
        "Cooling technology factor C_cool",
        min_value=0.02,
        max_value=0.08,
        value=0.045,
        step=0.005,
        help="Lower = better cooling tech",
    )
    eta_pc = st.sidebar.slider(
        "Compressor polytropic efficiency",
        min_value=0.85,
        max_value=0.95,
        value=0.90,
        step=0.01,
    )
    eta_pt = st.sidebar.slider(
        "Turbine polytropic efficiency",
        min_value=0.85,
        max_value=0.95,
        value=0.90,
        step=0.01,
    )
    eta_b = st.sidebar.slider(
        "Combustor efficiency",
        min_value=0.96,
        max_value=1.0,
        value=0.99,
        step=0.01,
    )
    dp_b = st.sidebar.slider(
        "Combustor pressure loss (Δp/p)",
        min_value=0.02,
        max_value=0.08,
        value=0.05,
        step=0.01,
    )

    params = CycleParams(
        OPR=OPR,
        eta_pc=eta_pc,
        eta_pt=eta_pt,
        T_b_max=T_b_max,
        C_cool=C_cool,
        eta_b=eta_b,
        dp_b_p03=dp_b,
    )

    # Sweep mode: T_ET or ψ
    sweep_mode = st.sidebar.radio(
        "Sweep variable",
        ["T_ET (Turbine Entry Temp)", "ψ (Cooling mass fraction)"],
        help="T_ET: cooling model gives ψ for each T_ET. ψ: invert to get max T_ET.",
    )

    # Build sweep and compute
    if "ψ" in sweep_mode:
        psi_range = np.linspace(0.02, 0.55, 180)
        results = sweep_psi(psi_range, params)
        x_data = results["psi"]
        x_label = "Cooling mass fraction ψ"
    else:
        T_ET_range = np.linspace(1400, 2800, 200)
        results = sweep_T_ET(T_ET_range, params)
        x_data = results["T_ET"]
        x_label = "Turbine entry temperature T_ET [K]"

    psi_arr = results["psi"]
    eta_th_arr = results["eta_th"]
    SFC_arr = results["SFC"]
    F_s_arr = results["F_s"]

    # Current point (single TIT from slider)
    psi_single = cooling_fraction(T_04, params)
    res_single = evaluate_cycle(T_04, psi_single, params)

    # Main plot: dual y-axis (Plotly for hover, zoom, pan)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if "ψ" in sweep_mode:
        idx = np.argmin(np.abs(psi_arr - psi_single))
    else:
        idx = np.argmin(np.abs(results["T_ET"] - T_04))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=eta_th_arr * 100,
            name="η_th [%]",
            line=dict(color="rgb(31, 119, 180)", width=2),
            hovertemplate="%{x:.3f}<br>η_th: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=SFC_arr * 1e6,
            name="SFC [mg/(N·s)]",
            line=dict(color="rgb(255, 127, 14)", width=2),
            hovertemplate="%{x:.3f}<br>SFC: %{y:.2f} mg/(N·s)<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=[x_data[idx]],
            y=[eta_th_arr[idx] * 100],
            mode="markers",
            name="Current point",
            marker=dict(size=12, color="rgb(31, 119, 180)", symbol="circle"),
            hovertemplate=f"Current<br>{x_label}: %{{x:.3f}}<br>η_th: %{{y:.2f}}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[x_data[idx]],
            y=[SFC_arr[idx] * 1e6],
            mode="markers",
            marker=dict(size=12, color="rgb(255, 127, 14)", symbol="circle"),
            showlegend=False,
            hovertemplate=f"Current<br>{x_label}: %{{x:.3f}}<br>SFC: %{{y:.2f}} mg/(N·s)<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        xaxis_title=x_label,
        hovermode="x unified",
        height=450,
        margin=dict(l=60, r=60),
    )
    fig.update_yaxes(title_text="Thermal efficiency η_th [%]", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="SFC [mg/(N·s)]", secondary_y=True, rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    # Second plot: Specific thrust vs cooling fraction
    idx2 = np.argmin(np.abs(psi_arr - psi_single))
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(
        go.Scatter(
            x=psi_arr,
            y=F_s_arr,
            name="F_s [m/s]",
            line=dict(color="rgb(44, 160, 44)", width=2),
            hovertemplate="ψ: %{x:.3f}<br>F_s: %{y:.0f} m/s<extra></extra>",
        ),
        secondary_y=False,
    )
    fig2.add_trace(
        go.Scatter(
            x=psi_arr,
            y=eta_th_arr * 100,
            name="η_th [%]",
            line=dict(color="rgb(31, 119, 180)", width=2),
            hovertemplate="ψ: %{x:.3f}<br>η_th: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig2.add_trace(
        go.Scatter(
            x=[psi_arr[idx2]],
            y=[F_s_arr[idx2]],
            mode="markers",
            name="Current",
            marker=dict(size=12, color="rgb(44, 160, 44)", symbol="circle"),
            hovertemplate="ψ: %{x:.3f}<br>F_s: %{y:.0f} m/s<extra></extra>",
        ),
        secondary_y=False,
    )
    fig2.add_trace(
        go.Scatter(
            x=[psi_arr[idx2]],
            y=[eta_th_arr[idx2] * 100],
            mode="markers",
            marker=dict(size=12, color="rgb(31, 119, 180)", symbol="circle"),
            showlegend=False,
            hovertemplate="ψ: %{x:.3f}<br>η_th: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig2.update_layout(
        xaxis_title="Cooling mass fraction ψ",
        hovermode="x unified",
        height=400,
        margin=dict(l=60, r=60),
    )
    fig2.update_yaxes(title_text="Specific thrust F_s [m/s]", secondary_y=False, rangemode="tozero")
    fig2.update_yaxes(title_text="Thermal efficiency η_th [%]", secondary_y=True, rangemode="tozero")
    st.plotly_chart(fig2, use_container_width=True)

    # Metrics at current TIT
    st.subheader("Performance at current TIT")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("η_th [%]", f"{res_single['eta_th']*100:.2f}")
    with col2:
        st.metric("ψ", f"{res_single['psi']:.3f}")
    with col3:
        st.metric("SFC [mg/(N·s)]", f"{res_single['SFC']*1e6:.2f}")
    with col4:
        st.metric("F_s [m/s]", f"{res_single['F_s']:.0f}")
    with col5:
        st.metric("T_c_in [K]", f"{res_single['T_c_in']:.0f}")
