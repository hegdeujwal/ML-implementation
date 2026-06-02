"""
correlation/fingerprint.py

Template fingerprint extraction and Jaccard similarity for cross-run
incident correlation (P5.5).

A fingerprint is an immutable frozenset of the unique template_id values
present in all logs that belong to a single incident.  Two incidents are
considered "related" when their Jaccard similarity meets or exceeds the
configured threshold (CROSS_RUN_SIMILARITY_THRESHOLD).

Public API
----------
fingerprint_from_df(incident_logs_df) -> frozenset[str]
    Extract fingerprint from a subset of scored_logs joined to sessionized_logs.

fingerprint_from_list(template_list) -> frozenset[str]
    Deserialise a fingerprint stored as a JSON string or plain Python list.

jaccard(a, b) -> float
    Compute Jaccard similarity between two frozensets.
"""

from __future__ import annotations

import json


def fingerprint_from_df(incident_logs_df) -> frozenset:
    """Return frozenset of unique non-null template_ids in the given rows.

    Parameters
    ----------
    incident_logs_df : pd.DataFrame
        Rows belonging to one incident.  Must contain a 'template_id' column.
    """
    return frozenset(
        str(t) for t in incident_logs_df["template_id"].dropna().unique().tolist()
    )


def fingerprint_from_list(template_list) -> frozenset:
    """Deserialise a fingerprint from its stored form.

    Accepts:
    - A JSON string (as stored in parquet / Postgres): '[\"T001\", \"T002\"]'
    - A Python list of strings: ['T001', 'T002']
    - None / empty: returns empty frozenset.
    """
    if template_list is None:
        return frozenset()
    if isinstance(template_list, str):
        try:
            template_list = json.loads(template_list)
        except (json.JSONDecodeError, ValueError):
            return frozenset()
    return frozenset(str(t) for t in template_list if t is not None)


def fingerprint_to_json(fp: frozenset) -> str:
    """Serialise a fingerprint frozenset to a JSON string for storage."""
    return json.dumps(sorted(fp))


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity = |A ∩ B| / |A ∪ B|.

    Returns 0.0 when both sets are empty (no shared information to exploit).
    Returns 1.0 when both sets are identical.
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def overlap_coefficient(a: frozenset, b: frozenset) -> float:
    """Szymkiewicz-Simpson overlap coefficient = |A ∩ B| / min(|A|, |B|).

    Better than Jaccard for precursor-detection because the precursor incident
    typically has fewer templates than the downstream failure incident (it shows
    early warning signs, not the full failure cascade).  This metric answers:
    "what fraction of the *smaller* incident's templates appear in the larger?"

    Returns 0.0 when either set is empty.
    Returns 1.0 when the smaller set is a complete subset of the larger.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))
