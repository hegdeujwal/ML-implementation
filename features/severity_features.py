"""
Severity-based feature extraction.

Maps log severity levels to numeric weights
for downstream ML scoring.
"""




from common.config import (
    SEVERITY_WEIGHTS,
    DEFAULT_SEVERITY_WEIGHT,
)

from common.logger import get_logger

logger = get_logger(__name__)


def add_severity_weight(df):

    unknown = (
        ~df["log_level"].isin(SEVERITY_WEIGHTS)
    )

    if unknown.any():
        logger.warning(
            "Unknown severity levels found. Using default weight."
        )

    df["severity_weight"] = (
        df["log_level"]
        .map(SEVERITY_WEIGHTS)
        .fillna(DEFAULT_SEVERITY_WEIGHT)
    )

    return df