"""
common/logger.py
================
Shared logging setup. All modules call get_logger(__name__).
Logs go to stdout and to logs/pipeline.log.
"""

import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        logging.Logger instance with stdout + file handlers.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers if get_logger is called multiple times
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_DIR / "pipeline.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
