"""Output paths under Figs_current/ for paper and presentation figures."""

from __future__ import annotations

from pathlib import Path

PAPER_FIGURES_DIR = Path(__file__).resolve().parent
FIGS_CURRENT = PAPER_FIGURES_DIR / "Figs_current"
FINAL_CONF_PAPER = FIGS_CURRENT / "final_conf_paper"
EXPLORE_IDEAS = FIGS_CURRENT / "explore_ideas"
WHITTLE_ASME_PRACTICE = FIGS_CURRENT / "whittle_asme_practice"
CONF_PPT_PLOTS = FIGS_CURRENT / "conf_ppt_plots"
JOURNAL_PLOTS = FIGS_CURRENT / "final_journal_paper"


def ensure_fig_dirs() -> None:
    for path in (
        FINAL_CONF_PAPER,
        EXPLORE_IDEAS,
        WHITTLE_ASME_PRACTICE,
        CONF_PPT_PLOTS,
        JOURNAL_PLOTS,
    ):
        path.mkdir(parents=True, exist_ok=True)
