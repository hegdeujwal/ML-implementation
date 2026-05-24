"""
scripts/generate_real_logs.py

Generates a realistic synthetic sessionized log dataset for Phase 3 testing.

Output: parsing/processed/sessionized_logs.parquet

Schema
------
log_id          : str    -- unique identifier per log event
session_id      : str    -- groups related events into a session
timestamp       : float  -- Unix epoch seconds (float for sub-second precision)
template_id     : str    -- normalized log template (simulates Drain output)
log_level       : str    -- severity level: INFO / WARN / ERROR / CRITICAL
is_anomaly      : bool   -- True for ~5% of rows
anomaly_label   : str    -- non-empty only when is_anomaly=True

Design choices
--------------
- 50 sessions, ~100 events each -> ~5 000 rows (manageable for single-host dev)
- 20 distinct templates representing real HPE network-log categories
- Sessions are contiguous time ranges spread across a 24-hour window
- Inter-arrival times vary per template to mimic realistic bursty patterns
- Anomaly injection: 3 templates have elevated error rates; flagged rows get
  an anomaly_label so graph_builder can create dedicated anomaly nodes
- Sequences are deliberately seeded: certain template triples appear together
  in order within 30 s so sequence_engine has real patterns to discover
"""

from __future__ import annotations

import os
import random
import time

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
NUM_SESSIONS = 50
EVENTS_PER_SESSION_MEAN = 100
EVENTS_PER_SESSION_STD = 20
ANOMALY_RATE = 0.05          # ~5% of rows flagged is_anomaly=True
SEQUENCE_INJECT_PROB = 0.4   # probability a session gets a seeded sequence
SEQUENCE_WINDOW_S = 25       # seeded sequences packed within 25 s

# 20 normalized log templates (simulate Drain output)
TEMPLATES = [
    "IF_DOWN",
    "IF_UP",
    "BGP_PEER_RESET",
    "BGP_PEER_ESTABLISHED",
    "CPU_HIGH",
    "CPU_NORMAL",
    "MEM_THRESHOLD_EXCEEDED",
    "MEM_OK",
    "LOGIN_FAIL",
    "LOGIN_SUCCESS",
    "OSPF_ADJACENCY_LOST",
    "OSPF_ADJACENCY_UP",
    "LINK_FLAP",
    "PACKET_DROP_HIGH",
    "PACKET_DROP_NORMAL",
    "DISK_WRITE_SLOW",
    "DISK_OK",
    "FAN_SPEED_HIGH",
    "TEMP_THRESHOLD_EXCEEDED",
    "CONFIG_CHANGE",
]

# Templates that are inherently anomalous when they appear
ANOMALY_TEMPLATES = {
    "IF_DOWN": "anomaly:if_down",
    "BGP_PEER_RESET": "anomaly:bgp_reset",
    "OSPF_ADJACENCY_LOST": "anomaly:ospf_loss",
    "LINK_FLAP": "anomaly:link_flap",
    "TEMP_THRESHOLD_EXCEEDED": "anomaly:temp_high",
}

# Seeded sequences: ordered triples that the sequence engine should detect
SEEDED_SEQUENCES = [
    ["IF_DOWN", "BGP_PEER_RESET", "OSPF_ADJACENCY_LOST"],
    ["CPU_HIGH", "MEM_THRESHOLD_EXCEEDED", "DISK_WRITE_SLOW"],
    ["LOGIN_FAIL", "LOGIN_FAIL", "CONFIG_CHANGE"],
    ["LINK_FLAP", "IF_DOWN", "BGP_PEER_RESET"],
]

SEVERITY_MAP: dict[str, str] = {
    "IF_DOWN": "ERROR",
    "BGP_PEER_RESET": "CRITICAL",
    "OSPF_ADJACENCY_LOST": "CRITICAL",
    "LINK_FLAP": "ERROR",
    "TEMP_THRESHOLD_EXCEEDED": "CRITICAL",
    "CPU_HIGH": "WARN",
    "MEM_THRESHOLD_EXCEEDED": "WARN",
    "DISK_WRITE_SLOW": "WARN",
    "LOGIN_FAIL": "ERROR",
    "CONFIG_CHANGE": "WARN",
}

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _random_template(rng: random.Random) -> str:
    return rng.choice(TEMPLATES)


def _severity(template: str, rng: random.Random) -> str:
    base = SEVERITY_MAP.get(template, "INFO")
    # Occasionally downgrade severity for realism
    if base == "INFO" and rng.random() < 0.05:
        return "WARN"
    return base


def generate_session(
    session_id: str,
    start_ts: float,
    n_events: int,
    rng: random.Random,
    log_id_counter: list[int],
) -> list[dict]:
    rows: list[dict] = []
    ts = start_ts

    # Optionally inject a seeded sequence early in the session
    inject_seq = rng.random() < SEQUENCE_INJECT_PROB
    seq_templates: list[str] = []
    seq_start_idx = rng.randint(5, 15) if inject_seq else -1
    if inject_seq:
        seq_templates = rng.choice(SEEDED_SEQUENCES)

    for i in range(n_events):
        inter_arrival = rng.expovariate(1 / 3)  # mean ~3 s
        ts += inter_arrival

        # Inject seeded sequence events tightly packed
        if inject_seq and i == seq_start_idx:
            for j, tmpl in enumerate(seq_templates):
                log_id_counter[0] += 1
                lid = f"log_{log_id_counter[0]:06d}"
                level = _severity(tmpl, rng)
                is_anom = tmpl in ANOMALY_TEMPLATES
                rows.append({
                    "log_id": lid,
                    "session_id": session_id,
                    "timestamp": ts + j * rng.uniform(1, 8),
                    "template_id": tmpl,
                    "log_level": level,
                    "is_anomaly": is_anom,
                    "anomaly_label": ANOMALY_TEMPLATES.get(tmpl, ""),
                })
            ts += SEQUENCE_WINDOW_S
            inject_seq = False  # only inject once per session
            continue

        tmpl = _random_template(rng)
        is_anom = (
            tmpl in ANOMALY_TEMPLATES and rng.random() < ANOMALY_RATE * 4
        ) or rng.random() < ANOMALY_RATE * 0.2

        log_id_counter[0] += 1
        lid = f"log_{log_id_counter[0]:06d}"
        rows.append({
            "log_id": lid,
            "session_id": session_id,
            "timestamp": ts,
            "template_id": tmpl,
            "log_level": _severity(tmpl, rng),
            "is_anomaly": is_anom,
            "anomaly_label": ANOMALY_TEMPLATES.get(tmpl, "") if is_anom else "",
        })

    return sorted(rows, key=lambda r: r["timestamp"])


def generate_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    all_rows: list[dict] = []
    counter = [0]  # mutable container so sessions can share state

    # Spread sessions across a 24-hour window
    base_ts = time.time() - 86_400
    session_gap = 86_400 / NUM_SESSIONS  # even spacing

    for s in range(NUM_SESSIONS):
        sid = f"session_{s:03d}"
        start = base_ts + s * session_gap + rng.uniform(0, session_gap * 0.5)
        n = max(20, int(rng.gauss(EVENTS_PER_SESSION_MEAN, EVENTS_PER_SESSION_STD)))
        all_rows.extend(generate_session(sid, start, n, rng, counter))

    df = pd.DataFrame(all_rows)
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_path = "data/processed/sessionized_logs.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Generating synthetic sessionized log data...")
    df = generate_dataset()
    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df):,} rows to {output_path}")
    print(f"  Sessions  : {df['session_id'].nunique()}")
    print(f"  Templates : {df['template_id'].nunique()}")
    print(f"  Anomalies : {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"  Time span : {df['timestamp'].max() - df['timestamp'].min():.0f} s")
