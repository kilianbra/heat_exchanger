"""Backward-compatible entry point: fig8 is now red-only (see fig8_red_only.py). """

from fig8_red_only import run_plot

if __name__ == "__main__":
    run_plot(base_name="fig8_red_only")
