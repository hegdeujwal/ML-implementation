"""
Shared logger configuration for the project.

Provides a reusable logger instance
for consistent pipeline logging and debugging.
"""

import logging


def get_logger(name: str):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(name)