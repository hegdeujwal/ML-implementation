"""
Severity-based feature extraction.

event_weight is computed in parsing/sessionizer.py from SEVERITY_WEIGHTS in config
and is already present in the input parquet.  Nothing to compute here.
This module is retained for future severity-related features.
"""


def add_severity_weight(df):
    # event_weight is computed in parsing/sessionizer.py from SEVERITY_WEIGHTS in config.
    # It is already present in the input parquet. Nothing to compute here.
    # This module is retained for future severity-related features.
    return df
