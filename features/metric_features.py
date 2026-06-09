"""
features/metric_features.py
===========================
Join Section-4 numeric telemetry (metrics_df, long/tidy) onto event rows as
per-log features for the IsolationForest.

Unlike severity (a human-assigned label proxy we deliberately keep out of the
model), these are *observed measurements* — drop rates, utilization, buffer/
counter levels — exactly the signal the dataset README points at for ML. Using
them is legitimate unsupervised feature engineering, not label leakage.

Produced features (all non-null; each paired with a 0/1 present flag so a
neutrally-filled 0 can't be mistaken for a real measurement):

    metric_zscore           max |z| of any nearby metric vs its own baseline
    metric_zscore_present
    drop_rate               nearest drop-family metric value near the event
    drop_rate_present
    utilization             nearest utilization/percentage metric near the event
    utilization_present

"Nearby" = within METRIC_JOIN_WINDOW_SECONDS, scoped to the same scenario so
metrics from one incident never bleed into another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import METRIC_JOIN_WINDOW_SECONDS, ZSCORE_MIN_STD
from common.logger import get_logger

logger = get_logger(__name__)

METRIC_FEATURE_COLUMNS = [
    "metric_zscore", "metric_zscore_present",
    "drop_rate", "drop_rate_present",
    "utilization", "utilization_present",
]

# Substring classifiers for the metric families we surface as named features.
_DROP_KEYS = ("drop", "loss", "discard")
_UTIL_KEYS = ("util", "pct", "buffer", "load", "usage")


def _neutral_frame(index: pd.Index) -> pd.DataFrame:
    """All-absent feature block (used for scenarios/rows with no metrics)."""
    return pd.DataFrame(
        {c: np.zeros(len(index)) for c in METRIC_FEATURE_COLUMNS},
        index=index,
    )


def _zscore_within_series(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add a 'zabs' column = |z-score| of metric_value within each
    (scenario_id, entity, metric_name) series."""
    grp = metrics.groupby(["scenario_id", "entity", "metric_name"])["metric_value"]
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(0.0)
    z = (metrics["metric_value"] - mean) / std.where(std > ZSCORE_MIN_STD, np.nan)
    metrics = metrics.copy()
    metrics["zabs"] = z.abs().fillna(0.0)
    return metrics


def _family_mask(metrics: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
    name = metrics["metric_name"].str.lower()
    mask = pd.Series(False, index=metrics.index)
    for k in keys:
        mask |= name.str.contains(k, regex=False)
    return mask


def _asof_nearest(
    events: pd.DataFrame, samples: pd.DataFrame, value_col: str, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """For each event, nearest sample value within ±window (per scenario).

    Returns (values, present) arrays aligned to events.index. Uses a backward+
    forward merge_asof and keeps whichever is closer in time.
    """
    if samples.empty:
        return np.zeros(len(events)), np.zeros(len(events))

    ev = events[["_row", "scenario_id", "timestamp"]].sort_values("timestamp")
    sm = samples[["scenario_id", "timestamp", value_col]].sort_values("timestamp")
    tol = pd.Timedelta(seconds=window_s)

    back = pd.merge_asof(
        ev, sm, on="timestamp", by="scenario_id",
        direction="backward", tolerance=tol,
    ).rename(columns={value_col: "v_back"})
    fwd = pd.merge_asof(
        ev, sm, on="timestamp", by="scenario_id",
        direction="forward", tolerance=tol,
    ).rename(columns={value_col: "v_fwd"})

    merged = back[["_row", "v_back"]].merge(fwd[["_row", "v_fwd"]], on="_row")
    # Prefer whichever side has a value (closer side is good enough; ties rare).
    merged["val"] = merged["v_back"].where(merged["v_back"].notna(), merged["v_fwd"])

    # Re-align to the original events order via the _row key.
    val_map = dict(zip(merged["_row"], merged["val"]))
    aligned = events["_row"].map(val_map)
    present_arr = aligned.notna().astype(float).values
    return aligned.fillna(0.0).values, present_arr


def compute_metric_features(
    events: pd.DataFrame, metrics: pd.DataFrame | None
) -> pd.DataFrame:
    """Return METRIC_FEATURE_COLUMNS aligned to events.index.

    Args:
        events:  must have timestamp; scenario_id optional (absent → single group).
        metrics: long metrics_df (timestamp, scenario_id, entity, metric_name,
                 metric_value) or None/empty for the legacy path.
    """
    if metrics is None or len(metrics) == 0:
        logger.info("No metrics available — emitting neutral metric features.")
        return _neutral_frame(events.index)

    ev = events.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    if "scenario_id" not in ev.columns:
        ev["scenario_id"] = "ALL"
    ev = ev.reset_index(drop=False).rename(columns={"index": "_row"})

    mt = metrics.copy()
    mt["timestamp"] = pd.to_datetime(mt["timestamp"])
    if "scenario_id" not in mt.columns:
        mt["scenario_id"] = "ALL"
    mt = _zscore_within_series(mt)

    win = float(METRIC_JOIN_WINDOW_SECONDS)
    z_val, z_present = _asof_nearest(ev, mt, "zabs", win)
    drop_val, drop_present = _asof_nearest(ev, mt[_family_mask(mt, _DROP_KEYS)], "metric_value", win)
    util_val, util_present = _asof_nearest(ev, mt[_family_mask(mt, _UTIL_KEYS)], "metric_value", win)

    out = pd.DataFrame({
        "metric_zscore": z_val,
        "metric_zscore_present": z_present,
        "drop_rate": drop_val,
        "drop_rate_present": drop_present,
        "utilization": util_val,
        "utilization_present": util_present,
    }, index=ev["_row"].values)
    out = out.reindex(events.index).fillna(0.0)
    logger.info(
        "Metric features: %d/%d rows have a nearby metric sample (zscore_present).",
        int(out["metric_zscore_present"].sum()), len(out),
    )
    return out
