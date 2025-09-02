import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider


def calculate_spirals(flow_angle_deg, r_min):
    """Calculate both spiral types based on input parameters"""
    # 1) derive scale factor S for Archimedean spiral
    flow_angle_rad = np.deg2rad(flow_angle_deg)
    S = np.tan(flow_angle_rad)

    # 2) Archimedean spiral
    # param t in [0, S*(1-r_min)]
    t = np.linspace(0, S * (1 - r_min), 500)
    r_s = 1 - t / S
    theta_s = np.pi / 2 - t  # clockwise from top
    x_s = r_s * np.cos(theta_s)
    y_s = r_s * np.sin(theta_s)

    # 3) Involute of circle
    a = 1 / np.sqrt(1 + S**2)  # circle radius
    phi1 = S
    sqrt_arg = (r_min / a) ** 2 - 1
    phi0 = max(0, np.sqrt(sqrt_arg)) if sqrt_arg > 0 else 0
    phi = np.linspace(phi0, phi1, 500)

    x = a * (np.cos(phi) + phi * np.sin(phi))
    y = a * (np.sin(phi) - phi * np.cos(phi))

    # rotate so that point at phi1 lands at top (0,1)
    x1 = a * (np.cos(phi1) + phi1 * np.sin(phi1))
    y1 = a * (np.sin(phi1) - phi1 * np.cos(phi1))
    alpha = np.pi / 2 - np.arctan2(y1, x1)

    X = x * np.cos(alpha) - y * np.sin(alpha)
    Y = x * np.sin(alpha) + y * np.cos(alpha)

    return x_s, y_s, X, Y


def _rotate_points(x: np.ndarray, y: np.ndarray, angle_rad: float):
    ca = np.cos(angle_rad)
    sa = np.sin(angle_rad)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    return xr, yr


def _intersect_ray_with_segment(
    p: np.ndarray, d: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-9
):
    """Return lambda (distance along ray) and u (segment param) for intersection, tolerant to endpoints/FP error."""
    s = b - a
    M = np.array([[d[0], -s[0]], [d[1], -s[1]]], dtype=float)
    rhs = a - p
    det = np.linalg.det(M)
    if np.isclose(det, 0.0, atol=eps):
        return None, None
    sol = np.linalg.solve(M, rhs)
    lmbda, u = sol[0], sol[1]
    if lmbda >= -eps and (-eps) <= u <= (1.0 + eps):
        return max(lmbda, 0.0), min(max(u, 0.0), 1.0)
    return None, None


def _intersect_ray_with_polyline(
    p: np.ndarray, d: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray, eps: float = 1e-9
):
    """Return closest intersection point with polyline in direction d from p, or None (tolerant)."""
    best_dist = None
    best_point = None
    for i in range(len(poly_x) - 1):
        a = np.array([poly_x[i], poly_y[i]], dtype=float)
        b = np.array([poly_x[i + 1], poly_y[i + 1]], dtype=float)
        lmbda, u = _intersect_ray_with_segment(p, d, a, b, eps=eps)
        if lmbda is not None and lmbda >= -eps and (best_dist is None or lmbda < best_dist):
            best_dist = lmbda
            best_point = p + lmbda * d
    if best_point is None:
        # Fallback: snap to nearest endpoint that lies on the ray line (within tolerance)
        for i in (0, len(poly_x) - 1):
            q = np.array([poly_x[i], poly_y[i]], dtype=float)
            v = q - p
            proj = np.dot(v, d)
            if proj >= -eps:
                d_line = np.linalg.norm(v - proj * d / (np.linalg.norm(d) + eps))
                if d_line <= 1e-6:
                    return q
    return best_point


def update(val):
    """Update function for sliders"""
    flow_angle = slider1.val
    r_min_val = slider2.val
    offset_deg = offset_slider.val if offset_ax.get_visible() else 0.0

    # Recalculate spirals
    x_s, y_s, X, Y = calculate_spirals(flow_angle, r_min_val)

    # Update plot data
    line1.set_data(x_s, y_s)
    line2.set_data(X, Y)

    # Update circle mask
    circle_mask.set_radius(r_min_val)

    # Update title with current parameters
    ax.set_title(rf"Init. Flow Angle: {flow_angle:.1f}°, $r_{{{'min'}}}$: {r_min_val:.2f}")

    # Handle spacing/offset optional curves
    if offset_ax.get_visible():
        off_rad = np.deg2rad(offset_deg)
        # Spiral: clockwise shift => negative rotation
        x_s_off, y_s_off = _rotate_points(x_s, y_s, -off_rad)
        # Involute: anti-clockwise shift => positive rotation
        X_off, Y_off = _rotate_points(X, Y, off_rad)
        line1_off.set_data(x_s_off, y_s_off)
        line2_off.set_data(X_off, Y_off)
        line1_off.set_visible(True)
        line2_off.set_visible(True)

        # Compute orthogonal lines (solid black)
        # Recompute needed scalars for tangents
        flow_angle_rad = np.deg2rad(flow_angle)
        S = np.tan(flow_angle_rad)
        a_radius = 1.0 / np.sqrt(1.0 + S**2)

        # Spiral orthogonals
        # 1) main at r=r_min -> offset
        p_sp_in_main = np.array([x_s[-1], y_s[-1]])
        t_sp_in_main = np.array([x_s[-1] - x_s[-2], y_s[-1] - y_s[-2]])
        # candidate normals
        n1 = np.array([-t_sp_in_main[1], t_sp_in_main[0]])
        n2 = -n1
        hit1 = _intersect_ray_with_polyline(p_sp_in_main, n1, x_s_off, y_s_off)
        hit2 = _intersect_ray_with_polyline(p_sp_in_main, n2, x_s_off, y_s_off)
        hit = (
            hit1
            if (
                hit1 is not None
                and (
                    hit2 is None
                    or np.linalg.norm(hit1 - p_sp_in_main) <= np.linalg.norm(hit2 - p_sp_in_main)
                )
            )
            else hit2
        )
        orth_spiral_inner_main_to_off.set_data(
            [p_sp_in_main[0], hit[0]] if hit is not None else [],
            [p_sp_in_main[1], hit[1]] if hit is not None else [],
        )

        # 2) offset at r=1 -> main
        p_sp_out_off = np.array([x_s_off[0], y_s_off[0]])
        t_sp_out_off = np.array([x_s_off[1] - x_s_off[0], y_s_off[1] - y_s_off[0]])
        n1 = np.array([-t_sp_out_off[1], t_sp_out_off[0]])
        n2 = -n1
        hit1 = _intersect_ray_with_polyline(p_sp_out_off, n1, x_s, y_s)
        hit2 = _intersect_ray_with_polyline(p_sp_out_off, n2, x_s, y_s)
        hit = (
            hit1
            if (
                hit1 is not None
                and (
                    hit2 is None
                    or np.linalg.norm(hit1 - p_sp_out_off) <= np.linalg.norm(hit2 - p_sp_out_off)
                )
            )
            else hit2
        )
        orth_spiral_outer_off_to_main.set_data(
            [p_sp_out_off[0], hit[0]] if hit is not None else [],
            [p_sp_out_off[1], hit[1]] if hit is not None else [],
        )

        # Involute orthogonals
        # 3) offset at inner r=max(a,r_min) -> main (force clockwise direction)
        # When r_min < a, start from the second sample to ensure a usable tangent vector
        inner_idx = 1 if r_min_val < a_radius else 0
        inner_idx = min(max(inner_idx, 0), len(X_off) - 2)
        p_iv_in_off = np.array([X_off[inner_idx], Y_off[inner_idx]])
        # Tangent by forward difference
        t_iv_in_off = np.array(
            [
                X_off[inner_idx + 1] - X_off[inner_idx],
                Y_off[inner_idx + 1] - Y_off[inner_idx],
            ]
        )
        # Clockwise normal: rotate tangent by -90 deg => [t_y, -t_x]
        n_cw = np.array([t_iv_in_off[1], -t_iv_in_off[0]])
        hit = _intersect_ray_with_polyline(p_iv_in_off, n_cw, X, Y)
        if hit is None and inner_idx + 2 < len(X_off):
            # Try one step outward if no hit
            j = inner_idx + 1
            p2 = np.array([X_off[j], Y_off[j]])
            t2 = np.array([X_off[j + 1] - X_off[j], Y_off[j + 1] - Y_off[j]])
            n2_cw = np.array([t2[1], -t2[0]])
            hit = _intersect_ray_with_polyline(p2, n2_cw, X, Y)
            if hit is not None:
                p_iv_in_off = p2
        if hit is None and inner_idx - 1 >= 0:
            # Try one step inward as another fallback (guarded by bounds)
            j = inner_idx - 1
            p2 = np.array([X_off[j], Y_off[j]])
            t2 = np.array([X_off[j + 1] - X_off[j], Y_off[j + 1] - Y_off[j]])
            n2_cw = np.array([t2[1], -t2[0]])
            hit = _intersect_ray_with_polyline(p2, n2_cw, X, Y)
            if hit is not None:
                p_iv_in_off = p2
        if hit is None:
            # Final fallback: try counter-clockwise only to avoid drawing nothing
            n_ccw = np.array([-t_iv_in_off[1], t_iv_in_off[0]])
            hit = _intersect_ray_with_polyline(p_iv_in_off, n_ccw, X, Y)
        orth_involute_inner_off_to_main.set_data(
            [p_iv_in_off[0], hit[0]] if hit is not None else [],
            [p_iv_in_off[1], hit[1]] if hit is not None else [],
        )

        # 4) main at outer r=1 -> offset
        p_iv_out_main = np.array([X[-1], Y[-1]])
        t_iv_out_main = np.array([X[-1] - X[-2], Y[-1] - Y[-2]])
        n1 = np.array([-t_iv_out_main[1], t_iv_out_main[0]])
        n2 = -n1
        hit1 = _intersect_ray_with_polyline(p_iv_out_main, n1, X_off, Y_off)
        hit2 = _intersect_ray_with_polyline(p_iv_out_main, n2, X_off, Y_off)
        hit = (
            hit1
            if (
                hit1 is not None
                and (
                    hit2 is None
                    or np.linalg.norm(hit1 - p_iv_out_main) <= np.linalg.norm(hit2 - p_iv_out_main)
                )
            )
            else hit2
        )
        orth_involute_outer_main_to_off.set_data(
            [p_iv_out_main[0], hit[0]] if hit is not None else [],
            [p_iv_out_main[1], hit[1]] if hit is not None else [],
        )

        # Length ratios for title
        # Spiral lengths
        len_sp_r1 = np.nan
        len_sp_rmin = np.nan
        len_sp_rad_in = np.nan
        len_sp_rad_out = np.nan
        if orth_spiral_outer_off_to_main.get_xdata():
            p0 = p_sp_out_off
            p1 = np.array(
                [
                    orth_spiral_outer_off_to_main.get_xdata()[1],
                    orth_spiral_outer_off_to_main.get_ydata()[1],
                ]
            )
            len_sp_r1 = float(np.linalg.norm(p1 - p0))
        if orth_spiral_inner_main_to_off.get_xdata():
            p0 = p_sp_in_main
            p1 = np.array(
                [
                    orth_spiral_inner_main_to_off.get_xdata()[1],
                    orth_spiral_inner_main_to_off.get_ydata()[1],
                ]
            )
            len_sp_rmin = float(np.linalg.norm(p1 - p0))
        sp_ratio = (
            (len_sp_rmin / len_sp_r1)
            if (np.isfinite(len_sp_r1) and np.isfinite(len_sp_rmin) and len_sp_rmin > 0)
            else np.nan
        )

        # Spiral radial dotted lines from r=1
        # From offset spiral at r=1 radially inward to main
        p_off_outer = np.array([x_s_off[0], y_s_off[0]])
        d_inward = -p_off_outer
        hit_rad_in = _intersect_ray_with_polyline(p_off_outer, d_inward, x_s, y_s)
        if hit_rad_in is not None:
            spiral_rad_in_from_off.set_data(
                [p_off_outer[0], hit_rad_in[0]], [p_off_outer[1], hit_rad_in[1]]
            )
            spiral_rad_in_from_off.set_visible(True)
            len_sp_rad_in = float(np.linalg.norm(hit_rad_in - p_off_outer))
        else:
            spiral_rad_in_from_off.set_visible(False)

        # From main spiral at r=r_min radially outward to offset
        p_main_inner = np.array([x_s[-1], y_s[-1]])
        # Outward radial direction = normalized position vector
        norm = np.linalg.norm(p_main_inner)
        d_outward = p_main_inner / (norm if norm > 0 else 1.0)
        hit_rad_out = _intersect_ray_with_polyline(p_main_inner, d_outward, x_s_off, y_s_off)
        if hit_rad_out is not None:
            spiral_rad_out_from_main.set_data(
                [p_main_inner[0], hit_rad_out[0]], [p_main_inner[1], hit_rad_out[1]]
            )
            spiral_rad_out_from_main.set_visible(True)
            len_sp_rad_out = float(np.linalg.norm(hit_rad_out - p_main_inner))
        else:
            spiral_rad_out_from_main.set_visible(False)

        sp_rad_ratio = (
            (len_sp_rad_in / len_sp_rad_out)
            if (np.isfinite(len_sp_rad_in) and np.isfinite(len_sp_rad_out) and len_sp_rad_out > 0)
            else np.nan
        )

        # Involute lengths
        len_iv_r1 = np.nan
        len_iv_inner = np.nan
        if orth_involute_outer_main_to_off.get_xdata():
            p0 = p_iv_out_main
            p1 = np.array(
                [
                    orth_involute_outer_main_to_off.get_xdata()[1],
                    orth_involute_outer_main_to_off.get_ydata()[1],
                ]
            )
            len_iv_r1 = float(np.linalg.norm(p1 - p0))
        if orth_involute_inner_off_to_main.get_xdata():
            p0 = p_iv_in_off
            p1 = np.array(
                [
                    orth_involute_inner_off_to_main.get_xdata()[1],
                    orth_involute_inner_off_to_main.get_ydata()[1],
                ]
            )
            len_iv_inner = float(np.linalg.norm(p1 - p0))
        iv_ratio = (
            (len_iv_inner / len_iv_r1)
            if (np.isfinite(len_iv_r1) and np.isfinite(len_iv_inner) and len_iv_inner > 0)
            else np.nan
        )

        def _fmt_ratio(x):
            return f"{x:.0%}" if np.isfinite(x) else "n/a"

        ax.set_title(
            rf"Init. Flow Angle: {flow_angle:.1f}°, $r_{{{'min'}}}$: {r_min_val:.2f} | "
            f"spiral_r = {_fmt_ratio(sp_ratio)} ({_fmt_ratio(sp_rad_ratio)}), "
            f"inv_r = {_fmt_ratio(iv_ratio)}"
        )

        # Show only the orthogonal lines; hide previous tangent lines if any
        for ln in tangent_lines:
            ln.set_visible(False)
        for ln in orth_lines:
            ln.set_visible(True)
    else:
        line1_off.set_visible(False)
        line2_off.set_visible(False)
        for ln in tangent_lines:
            ln.set_visible(False)
        for ln in orth_lines:
            ln.set_visible(False)
        spiral_rad_in_from_off.set_visible(False)
        spiral_rad_out_from_main.set_visible(False)

    fig.canvas.draw_idle()


def on_radio_spacing(label: str):
    is_on = label.lower().endswith("on")
    # Show/Hide offset slider
    offset_ax.set_visible(is_on)
    fig.canvas.draw_idle()
    # Trigger update to show/hide offset curves
    update(None)


# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
# Reserve space on the right for controls
plt.subplots_adjust(right=0.75)

# Initial parameters
flow_angle_deg = 75  # default
r_min = 0.5  # default

# Calculate initial spirals
x_s, y_s, X, Y = calculate_spirals(flow_angle_deg, r_min)

# Plot the spirals
(line1,) = ax.plot(x_s, y_s, "b-", linewidth=1, label="Archimedean Spiral")
(line2,) = ax.plot(X, Y, "r-", linewidth=1, label="Involute of Circle")

# Plot reference unit circle
theta_circle = np.linspace(0, 2 * np.pi, 100)
x_circle = np.cos(theta_circle)
y_circle = np.sin(theta_circle)
ax.plot(x_circle, y_circle, "k--", linewidth=1, alpha=0.5, label="Unit Circle")

# Add inner circle mask to show r_min boundary
circle_mask = patches.Circle(
    (0, 0), r_min, fill=False, edgecolor="gray", linestyle="--", alpha=0.7, label=f"r_min = {r_min}"
)

# Add center point
ax.plot(0, 0, "ko", markersize=3)

# Set up the plot
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
ax.set_title(rf"Init. Flow Angle: {flow_angle_deg:.1f}°, $r_{{{'min'}}}$: {r_min:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Add the circle mask to the plot
ax.add_patch(circle_mask)

"""Controls on the right"""
# Sliders on the right side
ax_slider1 = plt.axes([0.81, 0.65, 0.15, 0.03])  # numbers mean left, bottom, width, height
ax_slider2 = plt.axes([0.81, 0.60, 0.15, 0.03])
slider1 = Slider(
    ax_slider1, "Flow Angle", 45, 89.9, valinit=flow_angle_deg, valstep=0.1, valfmt="%.1f°"
)
slider2 = Slider(ax_slider2, r"$r_{min}$", 0.01, 0.99, valinit=r_min, valstep=0.01)

# Radio buttons for spacing toggle
radio_ax = plt.axes([0.78, 0.48, 0.21, 0.10])
radio = RadioButtons(radio_ax, ("spacing off", "spacing on"), 0)
radio.on_clicked(on_radio_spacing)

# Offset slider (initially hidden)
offset_ax = plt.axes([0.81, 0.42, 0.15, 0.03])
offset_slider = Slider(offset_ax, "offset", 0.0, 30.0, valinit=15.0, valstep=0.1, valfmt="%.1f°")
offset_ax.set_visible(False)
offset_slider.on_changed(update)

# Connect sliders to update function
slider1.on_changed(update)
slider2.on_changed(update)

# Lines for offset curves (created but hidden initially)
(line1_off,) = ax.plot([], [], "b--", linewidth=1, alpha=0.8, label="Archimedean Spiral (offset)")
(line2_off,) = ax.plot([], [], "r--", linewidth=1, alpha=0.8, label="Involute of Circle (offset)")
line1_off.set_visible(False)
line2_off.set_visible(False)

# Tangent lines (solid black), hidden initially
(tan_spiral_outer_main_to_off,) = ax.plot([], [], "k-", linewidth=1.2)
(tan_spiral_outer_off_to_main,) = ax.plot([], [], "k-", linewidth=1.2)
(tan_spiral_inner_main_to_off,) = ax.plot([], [], "k-", linewidth=1.2)
(tan_spiral_inner_off_to_main,) = ax.plot([], [], "k-", linewidth=1.2)
(tan_involute_outer_main_to_off,) = ax.plot([], [], "k-", linewidth=1.2)
(tan_involute_outer_off_to_main,) = ax.plot([], [], "k-", linewidth=1.2)
(tan_involute_inner_main_to_off,) = ax.plot([], [], "k-", linewidth=1.2)
(tan_involute_inner_off_to_main,) = ax.plot([], [], "k-", linewidth=1.2)
tangent_lines = [
    tan_spiral_outer_main_to_off,
    tan_spiral_outer_off_to_main,
    tan_spiral_inner_main_to_off,
    tan_spiral_inner_off_to_main,
    tan_involute_outer_main_to_off,
    tan_involute_outer_off_to_main,
    tan_involute_inner_main_to_off,
    tan_involute_inner_off_to_main,
]
for ln in tangent_lines:
    ln.set_visible(False)

# Orthogonal lines (solid black), hidden initially (exactly four)
(orth_spiral_inner_main_to_off,) = ax.plot([], [], "k-", linewidth=1.6)
(orth_spiral_outer_off_to_main,) = ax.plot([], [], "k-", linewidth=1.6)
(orth_involute_inner_off_to_main,) = ax.plot([], [], "k-", linewidth=1.6)
(orth_involute_outer_main_to_off,) = ax.plot([], [], "k-", linewidth=1.6)
orth_lines = [
    orth_spiral_inner_main_to_off,
    orth_spiral_outer_off_to_main,
    orth_involute_inner_off_to_main,
    orth_involute_outer_main_to_off,
]
for ln in orth_lines:
    ln.set_visible(False)

"""Spiral radial dotted helper lines (hidden initially)"""
(spiral_rad_in_from_off,) = ax.plot([], [], "k:", linewidth=1.2)
(spiral_rad_out_from_main,) = ax.plot([], [], "k:", linewidth=1.2)
spiral_rad_in_from_off.set_visible(False)
spiral_rad_out_from_main.set_visible(False)

plt.show()
