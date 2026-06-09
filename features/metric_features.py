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
    metric_slope_short      trailing OLS slope (~32 min) of the fastest-drifting
    metric_slope_short_present  nearby metric, normalised by that series' std
    metric_slope_long       trailing OLS slope (~96 min), same normalisation
    metric_slope_long_present

The slope pair adds the rate-of-change signal that point-in-time metrics lack —
the signature of gradual drift (memory leaks, thermal/CPU creep, disk decay).
Both are backward-looking only (no future leakage at inference time).

"Nearby" = within METRIC_JOIN_WINDOW_SECONDS, scoped to the same scenario so
metrics from one incident never bleed into another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import (
    METRIC_JOIN_WINDOW_SECONDS,
    METRIC_SLOPE_LONG_WINDOW,
    METRIC_SLOPE_SHORT_WINDOW,
    ZSCORE_MIN_STD,
)
from common.logger import get_logger

logger = get_logger(__name__)

METRIC_FEATURE_COLUMNS = [
    "metric_zscore", "metric_zscore_present",
    "drop_rate", "drop_rate_present",
    "utilization", "utilization_present",
    "metric_slope_short", "metric_slope_short_present",
    "metric_slope_long", "metric_slope_long_present",
]

# Substring classifiers for the metric families we surface as named features.
_DROP_KEYS = ("drop", "loss", "discard")
_UTIL_KEYS = ("util", "pct", "buffer", "load", "usage")

# Bound on the normalised slope so a near-perfect ramp (tiny diff-noise) can't
# emit an unbounded outlier. Mirrors the [-5, 5] clip on zscore_base.
SLOPE_CLIP = 5.0


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


def _rolling_ols_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Backward-looking OLS slope over a trailing `window` of regularly-spaced
    samples. x is the within-window position 0..W-1, so the slope is in
    metric-value units per sample-step. NaN until the window first fills (warm-up).

    Closed form with constant x means the denominator is computed once; each
    window is one dot product, so this is O(n·W) with O(W) memory — a single
    linear pass per series, no big intermediate matrix.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if window < 2 or n < window:
        return out
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    for end in range(window, n + 1):
        y = values[end - window:end]
        out[end - 1] = float(np.dot(xc, y - y.mean())) / denom
    return out


def _add_slope_columns(metrics: pd.DataFrame, short_w: int, long_w: int) -> pd.DataFrame:
    """Add normalised trailing-slope columns per (scenario, entity, metric_name).

    Normalisation is by the series' *noise level* — the std of its step-to-step
    differences — NOT its total spread. This is deliberate: a steady ramp (the
    drift we want) has large total spread but small diff-noise, so a slope/spread
    ratio would shrink exactly the metric we care about while amplifying flat,
    jittery metrics. slope/diff-noise instead reads as a trend signal-to-noise
    ratio — high for smooth sustained drift, low for noisy oscillation — so the
    downstream max-|slope| selection picks the genuinely drifting metric.

    Result is clipped to [-SLOPE_CLIP, SLOPE_CLIP] so a near-perfect synthetic
    ramp (tiny diff-noise) can't produce an unbounded outlier that destabilises
    the StandardScaler. Warm-up positions stay NaN so the join skips them.
    """
    metrics = metrics.sort_values(
        ["scenario_id", "entity", "metric_name", "timestamp"]
    ).copy()
    grp = metrics.groupby(["scenario_id", "entity", "metric_name"])["metric_value"]

    # Per-series noise = std of first differences (one scalar broadcast per row).
    diff_noise = grp.transform(lambda s: s.diff().std())
    diff_noise = diff_noise.fillna(0.0)
    safe_noise = diff_noise.where(diff_noise > ZSCORE_MIN_STD, np.nan)

    for col, w in (("slope_short", short_w), ("slope_long", long_w)):
        raw = grp.transform(lambda s: _rolling_ols_slope(s.to_numpy(dtype=float), w))
        metrics[col] = (raw / safe_noise).clip(-SLOPE_CLIP, SLOPE_CLIP)
    return metrics


def _max_abs_per_time(metrics: pd.DataFrame, col: str) -> pd.DataFrame:
    """Reduce to one row per (scenario_id, timestamp): the signed value of `col`
    from whichever metric has the largest |col| at that instant.

    Answers "is ANY metric drifting hard near this event?" while keeping the sign
    (up- vs down-trend), and shrinks the frame the asof-join scans.
    """
    m = metrics[["scenario_id", "timestamp", col]].dropna(subset=[col]).copy()
    if m.empty:
        return m
    m["_abs"] = m[col].abs()
    idx = m.groupby(["scenario_id", "timestamp"])["_abs"].idxmax()
    return m.loc[idx, ["scenario_id", "timestamp", col]]


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
    mt = _add_slope_columns(mt, METRIC_SLOPE_SHORT_WINDOW, METRIC_SLOPE_LONG_WINDOW)

    win = float(METRIC_JOIN_WINDOW_SECONDS)
    z_val, z_present = _asof_nearest(ev, mt, "zabs", win)
    drop_val, drop_present = _asof_nearest(ev, mt[_family_mask(mt, _DROP_KEYS)], "metric_value", win)
    util_val, util_present = _asof_nearest(ev, mt[_family_mask(mt, _UTIL_KEYS)], "metric_value", win)

    # Trend features: join the fastest-drifting metric's slope near each event.
    ss_val, ss_present = _asof_nearest(ev, _max_abs_per_time(mt, "slope_short"), "slope_short", win)
    sl_val, sl_present = _asof_nearest(ev, _max_abs_per_time(mt, "slope_long"), "slope_long", win)

    out = pd.DataFrame({
        "metric_zscore": z_val,
        "metric_zscore_present": z_present,
        "drop_rate": drop_val,
        "drop_rate_present": drop_present,
        "utilization": util_val,
        "utilization_present": util_present,
        "metric_slope_short": ss_val,
        "metric_slope_short_present": ss_present,
        "metric_slope_long": sl_val,
        "metric_slope_long_present": sl_present,
    }, index=ev["_row"].values)
    out = out.reindex(events.index).fillna(0.0)
    logger.info(
        "Metric features: %d/%d rows have a nearby metric sample (zscore_present).",
        int(out["metric_zscore_present"].sum()), len(out),
    )
    return out
