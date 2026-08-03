"""
Live three-panel layout preview.

Model: the printed figure WIDTH is fixed (target journal column width). Each
panel image keeps its own native aspect ratio — read directly from the PNG's
pixel dimensions, e.g. fig6a's source is ~7.1x7cm, fig6b/c are ~6x7cm — no
per-panel width slider is needed, because a wider source image is simply
wider once all three panels share the same HEIGHT.

That leaves exactly 3 free variables (all may go negative: overlap / bleed
past the figure edge):
  gap    - spacing between adjacent panels (cm)
  left   - outer margin on the left edge (cm)
  right  - outer margin on the right edge (cm)

Panel HEIGHT is solved for, not chosen:
    FIG_WIDTH = left + H*ar_a + gap + H*ar_b + gap + H*ar_c + right
So shrinking the gap frees up width budget -> H goes up -> all three panels
grow taller together, undistorted. This is the behaviour you want when
"gaining space by reducing the gap."

Two windows: a small fixed-size control panel (3 sliders) and a preview
figure that resizes live as you drag them.

Usage
-----
  uv run python compose_three_panel.py            # fig6 preset (live)
  uv run python compose_three_panel.py fig7
  uv run python compose_three_panel.py fig9
  uv run python compose_three_panel.py fig6 --save   # write fig6_combined.*

Values (gap/left/right in cm, plus the resulting solved height and overall
figure size) print to the terminal on every change. Tuned layouts are stored
in PRESETS[*]["layout_cm"] and used by --save.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.widgets import Slider

from fig_paths import JOURNAL_PLOTS, ensure_fig_dirs

ensure_fig_dirs()

CM_PER_IN = 2.54
# Match body text (same as fig4: font.size == axes.titlesize == 8).
PANEL_TITLE_FONTSIZE = 8

# Fixed (non-slider) vertical allowances, in cm.
TITLE_ROW_CM = 0.9  # room above the panel row for ax.set_title text
BOTTOM_PAD_CM = 0.15  # small pad below the panel row
TITLE_PAD_PT = 4  # matplotlib title pad (points) between title and panel image

PRESETS: dict[str, dict] = {
    "fig6": {
        "stems": ("fig6a_NTU_w_plus", "fig6b_NTU_w_plus", "fig6c_NTU_w_plus"),
        "titles": (
            "Conventional metrics",
            "Classical availability (exergy)",
            "Practical availability (euergy)",
        ),
        "width_cm": 18.0,
        "out_stem": "fig6_combined",
        # Tuned interactively (gap / left / right in cm).
        "layout_cm": {"gap": -0.08, "left": 0.00, "right": 0.04},
    },
    "fig7": {
        "stems": ("fig7a_Ao_Aoref", "fig7b_Ao_Aoref", "fig7c_Ao_Aoref"),
        "titles": (
            "Conventional metrics",
            "Classical availability (exergy)",
            "Practical availability (euergy)",
        ),
        "width_cm": 18.0,
        "out_stem": "fig7_combined",
        # Tuned interactively (gap / left / right in cm).
        "layout_cm": {"gap": -0.10, "left": -0.12, "right": -0.04},
    },
    "fig9": {
        "stems": ("fig9a_baseline", "fig9b_fix_mass_opt", "fig9c_min_ac_mass"),
        "titles": (
            "Baseline\n" r"$\eta$ = 40.6%",
            "Fixed mass optimum\n" r"$\eta$ = 41.2%",
            "Aircraft level optimum\n" r"$\eta$ = 42.7%",
        ),
        "width_cm": 18.0,
        "out_stem": "fig9_combined",
        "title_row_cm": 0.85,  # two-line titles; tighter than single-line presets
        "title_pad_pt": 1,  # reduce whitespace between two-line title and image
        # Tuned interactively (gap / left / right in cm).
        "layout_cm": {"gap": -0.02, "left": 0.00, "right": 0.10},
    },
}

SAVE_EXTS = (".svg", ".png", ".pdf", ".tiff", ".eps")


def _load_png(stem: str, fig_dir: Path):
    path = fig_dir / f"{stem}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing panel image: {path}")
    img = mpimg.imread(path)
    h_px, w_px = img.shape[0], img.shape[1]
    aspect = w_px / h_px  # width / height, dimensionless
    return img, aspect, path


def _axes_box_center_x(img, *, dark_thresh: float = 0.25) -> float:
    """Horizontal centre of the plot frame within *img*, as a fraction of width.

    Panel titles default to centering on the full PNG (ylabel margins included),
    which pushes long titles left of the actual axes box on panels with a tall
    left label and no right-axis label.  Detect the top spine of the black
    axes rectangle and return its midpoint so ``ax.set_title(..., x=...)``
    sits over the box.
    """
    rgb = img[..., :3] if getattr(img, "ndim", 0) == 3 else img
    gray = np.asarray(rgb.mean(axis=-1) if getattr(rgb, "ndim", 0) == 3 else rgb, dtype=float)
    # mpimg may return 0–1 float or 0–255 int
    if gray.max() > 1.5:
        gray = gray / 255.0
    dark = gray < dark_thresh
    h, w = dark.shape
    best: tuple[int, int, int] | None = None  # (length, left, right)
    for y in range(int(0.02 * h), int(0.45 * h)):
        row = dark[y]
        padded = np.concatenate(([False], row, [False]))
        edges = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        if starts.size == 0:
            continue
        lengths = ends - starts
        i = int(lengths.argmax())
        if lengths[i] < 0.30 * w:
            continue
        left, right = int(starts[i]), int(ends[i])
        if best is None or lengths[i] > best[0]:
            best = (int(lengths[i]), left, right)
    if best is None:
        return 0.5
    _, left, right = best
    return 0.5 * (left + right) / w


def _solve_height_cm(aspects: list[float], fig_width_cm: float, gap_cm: float, left_cm: float, right_cm: float):
    """Panel height (cm) so panels + gaps + margins exactly fill fig_width_cm."""
    n_gaps = len(aspects) - 1
    budget = fig_width_cm - left_cm - right_cm - n_gaps * gap_cm
    denom = sum(aspects)
    h_cm = budget / denom if denom > 0 else 0.0
    return max(h_cm, 0.05)  # floor to avoid zero/negative degenerate layouts


def _panel_x_widths_cm(aspects: list[float], h_cm: float, gap_cm: float, left_cm: float):
    x = left_cm
    out = []
    for ar in aspects:
        w = h_cm * ar
        out.append((x, w))
        x += w + gap_cm
    return out


def _apply_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": PANEL_TITLE_FONTSIZE,
            "axes.titlesize": PANEL_TITLE_FONTSIZE,
            "mathtext.fontset": "stix",
        }
    )


def _build_figure(
    imgs: list,
    aspects: list[float],
    titles: tuple[str, ...],
    fig_width_cm: float,
    gap_cm: float,
    left_cm: float,
    right_cm: float,
    *,
    title_row_cm: float = TITLE_ROW_CM,
    title_pad_pt: float = TITLE_PAD_PT,
):
    h_cm = _solve_height_cm(aspects, fig_width_cm, gap_cm, left_cm, right_cm)
    total_h_cm = title_row_cm + h_cm + BOTTOM_PAD_CM
    fig = plt.figure(figsize=(fig_width_cm / CM_PER_IN, total_h_cm / CM_PER_IN))
    y_frac = BOTTOM_PAD_CM / total_h_cm
    h_frac = h_cm / total_h_cm
    for img, title, (x_cm, w_cm) in zip(
        imgs, titles, _panel_x_widths_cm(aspects, h_cm, gap_cm, left_cm), strict=True
    ):
        ax = fig.add_axes([x_cm / fig_width_cm, y_frac, w_cm / fig_width_cm, h_frac])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(
            title,
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=title_pad_pt,
            x=_axes_box_center_x(img),
            ha="center",
            linespacing=1.1,
        )
    return fig, h_cm, total_h_cm


def save_three_panel(
    preset: str = "fig6",
    *,
    fig_dir: Path | None = None,
    gap_cm: float | None = None,
    left_cm: float | None = None,
    right_cm: float | None = None,
    out_stem: str | None = None,
) -> Path:
    """Save a combined three-panel figure using preset (or overridden) layout cm."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {list(PRESETS)}")

    fig_dir = Path(fig_dir) if fig_dir is not None else JOURNAL_PLOTS
    cfg = PRESETS[preset]
    layout = cfg["layout_cm"]
    gap_cm = layout["gap"] if gap_cm is None else gap_cm
    left_cm = layout["left"] if left_cm is None else left_cm
    right_cm = layout["right"] if right_cm is None else right_cm
    out_stem = cfg["out_stem"] if out_stem is None else out_stem
    fig_width_cm = cfg["width_cm"]

    imgs, aspects = [], []
    for stem in cfg["stems"]:
        img, ar, path = _load_png(stem, fig_dir)
        imgs.append(img)
        aspects.append(ar)
        print(f"Loaded {path.name}  (aspect w/h = {ar:.3f})")

    _apply_rcparams()
    fig, h_cm, total_h_cm = _build_figure(
        imgs,
        aspects,
        cfg["titles"],
        fig_width_cm,
        gap_cm,
        left_cm,
        right_cm,
        title_row_cm=cfg.get("title_row_cm", TITLE_ROW_CM),
        title_pad_pt=cfg.get("title_pad_pt", TITLE_PAD_PT),
    )
    print(
        f"{preset}: gap={gap_cm:.2f}cm, left={left_cm:.2f}cm, right={right_cm:.2f}cm  "
        f"-> panel_height={h_cm:.2f}cm, figure={fig_width_cm:.2f}x{total_h_cm:.2f}cm"
    )
    for ext in SAVE_EXTS:
        out = fig_dir / f"{out_stem}{ext}"
        fig.savefig(out, dpi=300, facecolor="white", format=ext.lstrip("."), bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Saved {out_stem}{', '.join(SAVE_EXTS)} -> {fig_dir}")
    return fig_dir / f"{out_stem}.png"


def compose_three_panel(preset: str = "fig6", *, fig_dir: Path | None = None) -> None:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {list(PRESETS)}")

    fig_dir = Path(fig_dir) if fig_dir is not None else JOURNAL_PLOTS
    cfg = PRESETS[preset]
    stems = cfg["stems"]
    titles = cfg["titles"]
    fig_width_cm = cfg["width_cm"]
    layout = cfg["layout_cm"]
    title_row_cm = cfg.get("title_row_cm", TITLE_ROW_CM)
    title_pad_pt = cfg.get("title_pad_pt", TITLE_PAD_PT)

    imgs, aspects = [], []
    for stem in stems:
        img, ar, path = _load_png(stem, fig_dir)
        imgs.append(img)
        aspects.append(ar)
        print(f"Loaded {path.name}  (aspect w/h = {ar:.3f})")

    _apply_rcparams()

    # --- Control window (fixed size; never resizes) ---
    fig_ctrl = plt.figure(figsize=(5.5, 2.2))
    fig_ctrl.canvas.manager.set_window_title(f"{preset} — controls (cm)")
    ax_gap = fig_ctrl.add_axes([0.22, 0.62, 0.68, 0.18])
    ax_left = fig_ctrl.add_axes([0.22, 0.36, 0.68, 0.18])
    ax_right = fig_ctrl.add_axes([0.22, 0.10, 0.68, 0.18])

    s_gap = Slider(ax_gap, "gap (cm)", -3.0, 3.0, valinit=layout["gap"], valstep=0.02)
    s_left = Slider(ax_left, "left (cm)", -3.0, 3.0, valinit=layout["left"], valstep=0.02)
    s_right = Slider(ax_right, "right (cm)", -3.0, 3.0, valinit=layout["right"], valstep=0.02)

    # --- Preview window (resizes live) ---
    fig_prev = plt.figure()
    fig_prev.canvas.manager.set_window_title(f"{preset} — preview")
    axes = [fig_prev.add_axes([0, 0, 1, 1]) for _ in stems]  # positions set in _update
    for ax, img, title in zip(axes, imgs, titles, strict=True):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(
            title,
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=title_pad_pt,
            x=_axes_box_center_x(img),
            ha="center",
            linespacing=1.1,
        )

    def _print_vals(h_cm: float, total_h_cm: float, prefix: str = "") -> None:
        print(
            f"{prefix}{preset}: gap={s_gap.val:.2f}cm, left={s_left.val:.2f}cm, right={s_right.val:.2f}cm  "
            f"-> panel_height={h_cm:.2f}cm, figure={fig_width_cm:.2f}x{total_h_cm:.2f}cm"
        )

    def _update(_=None, *, prefix: str = "") -> None:
        gap_cm, left_cm, right_cm = s_gap.val, s_left.val, s_right.val
        h_cm = _solve_height_cm(aspects, fig_width_cm, gap_cm, left_cm, right_cm)
        total_h_cm = title_row_cm + h_cm + BOTTOM_PAD_CM

        try:
            fig_prev.set_size_inches(fig_width_cm / CM_PER_IN, total_h_cm / CM_PER_IN, forward=True)
        except Exception:
            # Backend may already have torn down the window mid-drag.
            return

        y_frac = BOTTOM_PAD_CM / total_h_cm
        h_frac = h_cm / total_h_cm
        for ax, (x_cm, w_cm) in zip(axes, _panel_x_widths_cm(aspects, h_cm, gap_cm, left_cm), strict=True):
            ax.set_position([x_cm / fig_width_cm, y_frac, w_cm / fig_width_cm, h_frac])
        fig_prev.canvas.draw_idle()
        _print_vals(h_cm, total_h_cm, prefix)

    for s in (s_gap, s_left, s_right):
        s.on_changed(_update)

    _update(prefix="initial ")
    plt.show()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Live three-panel figure layout tuner")
    p.add_argument(
        "preset",
        nargs="?",
        default="fig6",
        choices=sorted(PRESETS),
        help="Which figure set to load (default: fig6)",
    )
    p.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=f"Image directory (default: {JOURNAL_PLOTS})",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save combined figure non-interactively using preset layout_cm (no sliders)",
    )
    p.add_argument("--gap", type=float, default=None, help="Override gap (cm) when using --save")
    p.add_argument("--left", type=float, default=None, help="Override left margin (cm) when using --save")
    p.add_argument("--right", type=float, default=None, help="Override right margin (cm) when using --save")
    args = p.parse_args(argv)
    try:
        if args.save:
            save_three_panel(
                args.preset,
                fig_dir=args.dir,
                gap_cm=args.gap,
                left_cm=args.left,
                right_cm=args.right,
            )
        else:
            compose_three_panel(args.preset, fig_dir=args.dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
