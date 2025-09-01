import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


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
    phi0 = max(0, np.sqrt((r_min / a) ** 2 - 1))
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


def update(val):
    """Update function for sliders"""
    flow_angle = slider1.val
    r_min_val = slider2.val

    # Recalculate spirals
    x_s, y_s, X, Y = calculate_spirals(flow_angle, r_min_val)

    # Update plot data
    line1.set_data(x_s, y_s)
    line2.set_data(X, Y)

    # Update circle mask
    circle_mask.set_radius(r_min_val)

    # Update title with current parameters
    ax.set_title(f"Flow Angle: {flow_angle:.1f}°, r_min: {r_min_val:.2f}")

    fig.canvas.draw_idle()


# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)  # Make room for sliders

# Initial parameters
flow_angle_deg = 85  # default
r_min = 0.8  # default

# Calculate initial spirals
x_s, y_s, X, Y = calculate_spirals(flow_angle_deg, r_min)

# Plot the spirals
(line1,) = ax.plot(x_s, y_s, "b-", linewidth=2, label="Archimedean Spiral")
(line2,) = ax.plot(X, Y, "r-", linewidth=2, label="Involute of Circle")

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
ax.plot(0, 0, "ko", markersize=3, label="Center")

# Set up the plot
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title(f"Flow Angle: {flow_angle_deg}°, r_min: {r_min:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Add the circle mask to the plot
ax.add_patch(circle_mask)

# Create sliders
ax_slider1 = plt.axes([0.2, 0.1, 0.6, 0.03])
ax_slider2 = plt.axes([0.2, 0.05, 0.6, 0.03])

slider1 = Slider(ax_slider1, "Flow Angle (deg)", 45, 89.9, valinit=flow_angle_deg, valstep=0.1)
slider2 = Slider(ax_slider2, "r_min", 0.1, 0.99, valinit=r_min, valstep=0.01)

# Connect sliders to update function
slider1.on_changed(update)
slider2.on_changed(update)

plt.show()
