"""
Shared color scheme for fig4 (bar charts), fig7b, fig7c.
"""

# Subplot / panel titles (Classical availability, Conventional metrics, …)
TITLE_FONTSIZE = 12

# Marker size to match LaTeX document symbols $+$ $\bullet$ $\star$ (~6pt)
MARKER_SIZE_LATEX = 36  # scatter s=36 gives ~6pt; tune if figures scale differently in document

# Bar chart / filled region colors
COLOR_TOTAL = (1.0, 1.0, 1.0)  # white, no hatch
COLOR_VISC_HOT = (0.7, 0.82, 0.95, 0.65)  # light blue, transparent
COLOR_VISC_COLD = (0.45, 0.6, 0.88, 0.65)  # darker blue, transparent
COLOR_THERMAL = (0.95, 0.6, 0.6, 0.65)  # off red, transparent


def color_with_alpha(color, alpha: float) -> tuple[float, float, float, float]:
    """Same hue as *color* with a new alpha (for pale cycle fuel/rest bars)."""
    import matplotlib.colors as mcolors

    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)
