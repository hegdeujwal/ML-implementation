"""
common/logger.py
================
Shared logging setup. All modules call get_logger(__name__).

Logs to stdout with format: [TIMESTAMP] [LEVEL] [module_name] message
Also writes DEBUG+ to logs/pipeline.log.

Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
Set LOG_LEVEL=DEBUG in your .env to enable verbose output.
"""

import logging
import os
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Read log level from environment; config.py documents this constant.
# Reading from os.environ directly avoids a circular import with config.py.
_CONSOLE_LEVEL = getattr(
    logging,
    os.environ.get("LOG_LEVEL", "INFO").upper(),
    logging.INFO,
)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        logging.Logger instance with stdout + file handlers.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(_CONSOLE_LEVEL)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_DIR / "pipeline.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
