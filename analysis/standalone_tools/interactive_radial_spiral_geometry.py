import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from heat_exchanger.geometries.radial_spiral import RadialSpiralSpec


def _build_geometry(
    R_o: float,
    n_r_h: int,
    n_r_a: int,
    n_h: int,
    Xt_star: float,
    Xl_star: float,
    inv_angle_deg: float,
    tube_OD: float,
    tube_thick: float = 0.129e-3,
    staggered: bool = True,
):
    n_r_h = int(max(1, round(n_r_h)))
    n_r_a = int(max(1, round(n_r_a)))
    n_h = int(max(1, round(n_h)))
    geom = RadialSpiralSpec(
        tube_outer_diam=float(tube_OD),
        tube_thick=float(tube_thick),
        tube_spacing_trv=float(Xt_star),
        tube_spacing_long=float(Xl_star),
        staggered=bool(staggered),
        n_headers=int(n_h),
        n_rows_per_header=int(n_r_h),
        n_tubes_per_row=int(n_r_a),
        radius_outer_whole_hex=float(R_o),
        inv_angle_deg=float(inv_angle_deg),
    )
    return geom


def _spiral_xy(radius_inner: float, radius_outer: float, inv_angle_deg: float, n: int = 800):
    theta_max = np.deg2rad(inv_angle_deg)
    # linear r(theta) model used in _compute_geometry_arrays
    b = (radius_outer - radius_inner) / theta_max if theta_max > 0 else 0.0
    theta = np.linspace(0.0, max(theta_max, 1e-12), n)
    r = radius_inner + b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _flow_angle_deg_at_point(position: np.ndarray, tangent: np.ndarray, eps: float = 1e-12) -> float:
    pos_norm = np.linalg.norm(position)
    tan_norm = np.linalg.norm(tangent)
    if pos_norm < eps or tan_norm < eps:
        return float("nan")
    # Angle between inward radial direction (-r_hat) and tangent vector (acute)
    r_inward_hat = -position / pos_norm
    cos_theta = np.dot(tangent, r_inward_hat) / tan_norm
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(abs(cos_theta))
    return float(np.rad2deg(theta))


def main():
    # Initial defaults from comp_withZeli_involute_cases.py -> case "ahjeb_mto_k1"
    init = {
        "R_o": 0.460,  # radius_outer_whole_hex [m]
        "n_r_h": 4,  # n_rows_per_header
        "n_r_a": 216,  # n_rows_axial (rounded from 690e-3 / (3.0 * 1.067e-3))
        "n_h": 21,  # n_headers
        "Xt_star": 3.0,  # tube_spacing_trv
        "Xl_star": 1.5,  # tube_spacing_long
        "inv_angle_deg": 360.0,
        "tube_OD": 1.067e-3,  # tube_outer_diam [m]
    }

    geom = _build_geometry(**init)
    # Prepare figure
    fig, ax = plt.subplots(figsize=(9, 8))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.33)

    # Plot spiral(s) and boundary circles
    x_sp_base, y_sp_base = _spiral_xy(geom.radius_inner_whole_hex, geom.radius_outer_whole_hex, geom.inv_angle_deg)
    lines_spirals: list = []
    for k in range(geom.n_headers):
        phi = 2.0 * np.pi * k / geom.n_headers
        cph, sph = np.cos(phi), np.sin(phi)
        xk = cph * x_sp_base - sph * y_sp_base
        yk = sph * x_sp_base + cph * y_sp_base
        label = "Base spiral" if k == 0 else None
        color = "red" if k == 0 else "black"
        (lk,) = ax.plot(xk, yk, "-", color=color, lw=1.2, alpha=0.9, label=label)
        lines_spirals.append(lk)

    theta_circ = np.linspace(0, 2 * np.pi, 512)
    x_in = geom.radius_inner_whole_hex * np.cos(theta_circ)
    y_in = geom.radius_inner_whole_hex * np.sin(theta_circ)
    x_out = geom.radius_outer_whole_hex * np.cos(theta_circ)
    y_out = geom.radius_outer_whole_hex * np.sin(theta_circ)
    (line_inner,) = ax.plot(x_in, y_in, "k--", alpha=0.6, label="Inner radius")
    (line_outer,) = ax.plot(x_out, y_out, "k--", alpha=0.6, label="Outer radius")
    # Header points on inner/outer circles
    thetas_hdr = np.linspace(0.0, 2.0 * np.pi, geom.n_headers, endpoint=False)
    x_hdr_in = geom.radius_inner_whole_hex * np.cos(thetas_hdr)
    y_hdr_in = geom.radius_inner_whole_hex * np.sin(thetas_hdr)
    x_hdr_out = geom.radius_outer_whole_hex * np.cos(thetas_hdr)
    y_hdr_out = geom.radius_outer_whole_hex * np.sin(thetas_hdr)
    (pts_inner,) = ax.plot(
        x_hdr_in,
        y_hdr_in,
        "o",
        color="tab:green",
        ms=3.0,
        linestyle="None",
        label="Header nodes (inner)",
    )
    (pts_outer,) = ax.plot(
        x_hdr_out,
        y_hdr_out,
        "o",
        color="tab:green",
        ms=3.0,
        linestyle="None",
        label="Header nodes (outer)",
    )

    ax.set_aspect("equal", adjustable="box")
    rmax = geom.radius_outer_whole_hex * 1.05
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.grid(True, alpha=0.25)
    # ax.legend(loc="upper left")

    # Areas
    def _areas_values(g: RadialSpiralSpec) -> tuple[float, float, float]:
        c = g._1d_arrays_for_one_sector()
        A_q = float(np.sum(c["area_ht_hot"])) * g.n_headers
        A_ff_i = float(c["area_free_hot"][0]) * g.n_headers
        A_ff_o = float(c["area_free_hot"][-1]) * g.n_headers
        return A_q, A_ff_i, A_ff_o

    def _areas_title(g: RadialSpiralSpec):
        A_q, A_ff_i, A_ff_o = _areas_values(g)
        return f"A_q={A_q:.3e} m^2  |  A_ff_i={A_ff_i:.3e} m^2  |  A_ff_o={A_ff_o:.3e} m^2"

    # Baseline (defaults) for normalization
    baseline_A_q, baseline_A_ff_i, baseline_A_ff_o = _areas_values(geom)
    normalize = False

    def _title_normalized(g: RadialSpiralSpec):
        A_q, A_ff_i, A_ff_o = _areas_values(g)
        pct_q = 100.0 * A_q / baseline_A_q if baseline_A_q > 0 else float("nan")
        pct_i = 100.0 * A_ff_i / baseline_A_ff_i if baseline_A_ff_i > 0 else float("nan")
        pct_o = 100.0 * A_ff_o / baseline_A_ff_o if baseline_A_ff_o > 0 else float("nan")
        return f"A_q={pct_q:.0f}%  |  A_ff_i={pct_i:.0f}%  |  A_ff_o={pct_o:.0f}%"

    def set_title_for_geometry(g: RadialSpiralSpec):
        ax.set_title(_title_normalized(g) if normalize else _areas_title(g))

    set_title_for_geometry(geom)

    # Button to toggle normalization
    ax_btn = plt.axes([0.82, 0.34, 0.12, 0.04])
    btn_norm = Button(ax_btn, "Normalized: OFF")
    btn_norm.ax.set_facecolor("#f0f0f0")

    def on_toggle_normalize(_evt):
        nonlocal normalize
        normalize = not normalize
        btn_norm.label.set_text("Normalized: ON" if normalize else "Normalized: OFF")
        btn_norm.ax.set_facecolor("#d0f0d0" if normalize else "#f0f0f0")
        set_title_for_geometry(
            _build_geometry(
                R_o=sl_Ro.val,
                n_r_h=int(sl_n_r_h.val),
                n_r_a=int(sl_n_r_a.val),
                n_h=int(sl_n_h.val),
                Xt_star=sl_Xt.val,
                Xl_star=sl_Xl.val,
                inv_angle_deg=sl_angle.val,
                tube_OD=sl_OD.val,
            )
        )
        fig.canvas.draw_idle()

    btn_norm.on_clicked(on_toggle_normalize)

    # Center text box: flow angle wrt tubes at inner/outer radii
    def _compute_flow_angles_text(x_sp: np.ndarray, y_sp: np.ndarray) -> str:
        if len(x_sp) >= 2:
            p_in = np.array([x_sp[0], y_sp[0]], dtype=float)
            t_in = np.array([x_sp[1] - x_sp[0], y_sp[1] - y_sp[0]], dtype=float)
            ang_in = _flow_angle_deg_at_point(p_in, t_in)
        else:
            ang_in = float("nan")
        if len(x_sp) >= 2:
            p_out = np.array([x_sp[-1], y_sp[-1]], dtype=float)
            t_out = np.array([x_sp[-1] - x_sp[-2], y_sp[-1] - y_sp[-2]], dtype=float)
            ang_out = _flow_angle_deg_at_point(p_out, t_out)
        else:
            ang_out = float("nan")
        return f"Flow angle wrt tubes: \n inner={ang_in:.1f}°, outer={ang_out:.1f}°"

    angle_text = ax.text(
        0.0,
        0.0,
        _compute_flow_angles_text(x_sp_base, y_sp_base),
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    # Slider axes (stacked bottom)
    axcolor = "0.95"
    s_Ro = plt.axes([0.10, 0.26, 0.80, 0.03], facecolor=axcolor)
    s_n_r_h = plt.axes([0.10, 0.22, 0.80, 0.03], facecolor=axcolor)
    s_n_r_a = plt.axes([0.10, 0.18, 0.80, 0.03], facecolor=axcolor)
    s_n_h = plt.axes([0.10, 0.14, 0.80, 0.03], facecolor=axcolor)
    s_Xt = plt.axes([0.10, 0.10, 0.80, 0.03], facecolor=axcolor)
    s_Xl = plt.axes([0.10, 0.06, 0.80, 0.03], facecolor=axcolor)
    s_angle = plt.axes([0.10, 0.02, 0.80, 0.03], facecolor=axcolor)

    sl_Ro = Slider(s_Ro, "R_o [m]", 0.05, 1.50, valinit=init["R_o"], valstep=0.001)
    sl_n_r_h = Slider(s_n_r_h, "n_r_h", 1, 40, valinit=init["n_r_h"], valstep=1)
    sl_n_r_a = Slider(s_n_r_a, "n_r_a", 1, 800, valinit=init["n_r_a"], valstep=1)
    sl_n_h = Slider(s_n_h, "n_h", 1, 64, valinit=init["n_h"], valstep=1)
    sl_Xt = Slider(s_Xt, "Xt*", 1.05, 5.0, valinit=init["Xt_star"], valstep=0.01)
    sl_Xl = Slider(s_Xl, "Xl*", 1.00, 4.0, valinit=init["Xl_star"], valstep=0.01)
    sl_angle = Slider(s_angle, "inv_angle [deg]", 30.0, 720.0, valinit=init["inv_angle_deg"], valstep=None)

    # Put tube OD slider on the right side to save vertical space
    s_OD = plt.axes([0.82, 0.40, 0.12, 0.45], facecolor=axcolor)
    sl_OD = Slider(
        s_OD,
        "tube_OD [m]",
        0.2e-3,
        5.0e-3,
        valinit=init["tube_OD"],
        valstep=0.001e-3,
        orientation="vertical",
    )

    def _snap_inv_angle_to_headers() -> None:
        """Snap inv_angle to nearest multiple of 360/n_h."""
        n_h_now = max(1, int(round(sl_n_h.val)))
        step = 360.0 / float(n_h_now)
        raw = float(sl_angle.val)
        # Round to nearest multiple and clamp to slider range
        snapped = step * round(raw / step)
        snapped = min(max(snapped, sl_angle.valmin), sl_angle.valmax)
        if abs(snapped - raw) > 1e-9:
            sl_angle.set_val(snapped)

    def update(_):
        # Ensure inv_angle respects discretization before rebuilding geometry
        _snap_inv_angle_to_headers()
        g = _build_geometry(
            R_o=sl_Ro.val,
            n_r_h=int(sl_n_r_h.val),
            n_r_a=int(sl_n_r_a.val),
            n_h=int(sl_n_h.val),
            Xt_star=sl_Xt.val,
            Xl_star=sl_Xl.val,
            inv_angle_deg=sl_angle.val,
            tube_OD=sl_OD.val,
        )
        try:
            # Recompute spiral base and circles
            x_sp, y_sp = _spiral_xy(g.radius_inner_whole_hex, g.radius_outer_whole_hex, g.inv_angle_deg)
            # Ensure correct number of spiral lines equals n_headers
            nonlocal lines_spirals
            # Remove extra
            while len(lines_spirals) > g.n_headers:
                ln = lines_spirals.pop()
                ln.remove()
            # Add missing
            while len(lines_spirals) < g.n_headers:
                (ln,) = ax.plot([], [], "-", color="black", lw=1.2, alpha=0.9)
                lines_spirals.append(ln)
            # Update all spiral lines with rotations
            for k, ln in enumerate(lines_spirals):
                phi = 2.0 * np.pi * k / g.n_headers if g.n_headers > 0 else 0.0
                cph, sph = np.cos(phi), np.sin(phi)
                xk = cph * x_sp - sph * y_sp
                yk = sph * x_sp + cph * y_sp
                ln.set_data(xk, yk)
                ln.set_color("red" if k == 0 else "black")

            x_in = g.radius_inner_whole_hex * np.cos(theta_circ)
            y_in = g.radius_inner_whole_hex * np.sin(theta_circ)
            x_out = g.radius_outer_whole_hex * np.cos(theta_circ)
            y_out = g.radius_outer_whole_hex * np.sin(theta_circ)
            line_inner.set_data(x_in, y_in)
            line_outer.set_data(x_out, y_out)
            # Update header points
            thetas_hdr = np.linspace(0.0, 2.0 * np.pi, g.n_headers, endpoint=False)
            pts_inner.set_data(
                g.radius_inner_whole_hex * np.cos(thetas_hdr),
                g.radius_inner_whole_hex * np.sin(thetas_hdr),
            )
            pts_outer.set_data(
                g.radius_outer_whole_hex * np.cos(thetas_hdr),
                g.radius_outer_whole_hex * np.sin(thetas_hdr),
            )

            rmax_local = g.radius_outer_whole_hex * 1.05
            ax.set_xlim(-rmax_local, rmax_local)
            ax.set_ylim(-rmax_local, rmax_local)

            set_title_for_geometry(g)
            # Update flow-angle center text
            angle_text.set_text(_compute_flow_angles_text(x_sp, y_sp))
        except Exception as exc:
            ax.set_title(f"Invalid geometry: {exc}")

        fig.canvas.draw_idle()

    for s in (sl_Ro, sl_n_r_h, sl_n_r_a, sl_n_h, sl_Xt, sl_Xl, sl_angle, sl_OD):
        s.on_changed(update)

    # Additionally enforce snapping when n_h or angle changes
    def _on_n_h_changed(_):
        _snap_inv_angle_to_headers()
        update(_)

    def _on_angle_changed(_):
        _snap_inv_angle_to_headers()
        update(_)

    sl_n_h.on_changed(_on_n_h_changed)
    sl_angle.on_changed(_on_angle_changed)

    plt.show()


if __name__ == "__main__":
    main()
