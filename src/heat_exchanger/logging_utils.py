from __future__ import annotations

import logging

DEFAULT_FORMAT = "%(levelname)s:%(name)s:%(message)s"


def configure_logging(
    level: int = logging.INFO,
    *,
    format: str = DEFAULT_FORMAT,
    force: bool | None = None,
) -> None:
    """Configure root logging once.

    Call from scripts/notebooks before importing modules that emit log messages at
    import time. `force` is forwarded to ``logging.basicConfig`` (Python ≥3.8) to
    allow reconfiguration when running interactively.
    """

    kwargs: dict[str, object] = {"level": level, "format": format}
    if force is not None:
        kwargs["force"] = force
    logging.basicConfig(**kwargs)


__all__ = ["configure_logging", "DEFAULT_FORMAT"]
