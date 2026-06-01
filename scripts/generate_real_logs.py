"""
scripts/generate_real_logs.py

Generates synthetic syslog data by creating realistic raw BSD syslog lines
and routing them through the real parsing pipeline (normalizer → Drain →
sessionizer).  This exercises all parsing logic rather than bypassing it.

Side effects
------------
- Writes raw lines to   data/raw/synthetic_sample.log
- Runs parsing pipeline → data/processed/sessionized_logs.parquet
- Writes ground truth   → data/synthetic/ground_truth.parquet
  Columns: [sequence_number, true_label]  (true_label: "anomaly" | "normal")

Public API
----------
generate_dataset(seed=42) -> pd.DataFrame
    Returns the sessionized_logs DataFrame produced by the real parser.
    Same function name kept for backward compatibility with pipeline.py and
    correlation/run_correlation.py.
"""
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta

import pandas as pd

from common.utils import save_parquet

RANDOM_SEED = 42

RAW_LOG_PATH = "data/raw/synthetic_sample.log"
SESSIONIZED_PATH = "data/processed/sessionized_logs.parquet"
GROUND_TRUTH_PATH = "data/synthetic/ground_truth.parquet"

# HPE CX switch hostnames — 3 tiers, realistic naming
_HOSTS = (
    [f"sw-core-{i:02d}" for i in range(1, 6)]
    + [f"sw-edge-{i:02d}" for i in range(1, 6)]
    + [f"sw-dist-{i:02d}" for i in range(1, 4)]
)

# (service_name, message_template, pri_tag)
# PRI = local0 (facility 16) × 8 + severity: INFO=134, WARN=132, ERROR=131, CRIT=130
_NORMAL_PATTERNS = [
    ("ospf",        "Neighbor {ip}/0 changed state to Full",              "<134>"),
    ("bgp",         "BGP neighbor {ip} session established",              "<134>"),
    ("ifmgrd",      "Interface 1/1/{port} changed state to up",           "<134>"),
    ("lldpd",       "LLDP neighbor {mac} discovered on port 1/1/{port}",  "<134>"),
    ("kernel",      "CPU utilization {pct}%",                             "<134>"),
    ("cfgd",        "Configuration saved successfully",                   "<134>"),
    ("snmpd",       "SNMP query from {ip} OID sysUpTime",                 "<134>"),
    ("sshd",        "login successful for user admin from {ip}",          "<134>"),
    ("eventmgr",    "System health check passed all subsystems",          "<134>"),
    ("hpe-routing", "Route {ip}/24 added to RIB via OSPF",               "<134>"),
    ("ifmgrd",      "VLAN {vlan} added to interface 1/1/{port}",         "<134>"),
    ("ospf",        "OSPF adjacency established on vlan{vlan}",           "<134>"),
    ("bgp",         "BGP neighbor {ip} prefix count {pct}",              "<134>"),
    ("kernel",      "Memory usage {pct}% normal",                        "<134>"),
]

_ANOMALY_PATTERNS = [
    ("ospf",    "Neighbor {ip}/0 changed state to Down",              "<131>"),
    ("bgp",     "BGP neighbor {ip} connection reset hold timer expired", "<130>"),
    ("ifmgrd",  "Interface 1/1/{port} changed state to down",          "<131>"),
    ("ospf",    "OSPF adjacency lost on vlan{vlan}",                   "<131>"),
    ("ifmgrd",  "Interface 1/1/{port} link flap detected on uplink",   "<130>"),
    ("kernel",  "CPU utilization {pct}% exceeded threshold alert",     "<132>"),
    ("kernel",  "Memory usage {pct}% threshold exceeded critical",     "<132>"),
    ("ifmgrd",  "packet drop rate high on interface 1/1/{port}",       "<132>"),
    ("sshd",    "login failed authentication failure from {ip}",        "<131>"),
    ("bgp",     "BGP neighbor {ip} session reset by peer",             "<130>"),
]


def _rand_ip(rng: random.Random) -> str:
    return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _rand_mac(rng: random.Random) -> str:
    return ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6))


def _generate_raw_lines(seed: int = RANDOM_SEED) -> list[str]:
    """Return a list of raw RFC 3164 syslog lines for synthetic testing."""
    rng = random.Random(seed)
    lines: list[str] = []

    base_dt = datetime(2026, 5, 30, 0, 0, 0)

    for host in _HOSTS:
        host_ts = base_dt + timedelta(seconds=rng.uniform(0, 3600))
        n_events = rng.randint(150, 300)
        pid_base = rng.randint(1000, 9000)

        for _ in range(n_events):
            host_ts += timedelta(seconds=rng.expovariate(0.3))  # mean ~3 s

            ts_str = host_ts.strftime("%b %d %H:%M:%S")
            pid = pid_base + rng.randint(0, 50)

            if rng.random() < 0.15:
                svc, msg_tmpl, pri = rng.choice(_ANOMALY_PATTERNS)
            else:
                svc, msg_tmpl, pri = rng.choice(_NORMAL_PATTERNS)

            msg = msg_tmpl.format(
                ip=_rand_ip(rng),
                mac=_rand_mac(rng),
                port=rng.randint(1, 48),
                vlan=rng.randint(1, 4094),
                pct=rng.randint(70, 99),
            )
            lines.append(f"{pri}{ts_str} {host} {svc}[{pid}]: {msg}")

    return lines


def generate_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic data through the real parsing pipeline.

    Writes raw syslog lines to data/raw/synthetic_sample.log, runs them
    through parsing.sessionizer.run(), then writes a ground truth file to
    data/synthetic/ground_truth.parquet.

    Returns the sessionized_logs DataFrame (canonical schema) produced by
    the real parser.
    """
    from parsing.sessionizer import run as sessionize

    lines = _generate_raw_lines(seed)

    os.makedirs(os.path.dirname(RAW_LOG_PATH), exist_ok=True)
    with open(RAW_LOG_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    os.makedirs(os.path.dirname(SESSIONIZED_PATH), exist_ok=True)
    df = sessionize(RAW_LOG_PATH, output_path=SESSIONIZED_PATH)

    # Ground truth: rows the parser classifies as ERROR or CRITICAL are anomalies.
    # This is grounded in the actual parser output (keyword overrides applied).
    gt_df = pd.DataFrame({
        "sequence_number": df["sequence_number"],
        "true_label": df["log_level"].map(
            lambda lvl: "anomaly" if lvl in ("CRITICAL", "ERROR") else "normal"
        ),
    })
    os.makedirs(os.path.dirname(GROUND_TRUTH_PATH), exist_ok=True)
    save_parquet(gt_df, GROUND_TRUTH_PATH)

    return df


if __name__ == "__main__":
    print("Generating synthetic data via real parsing pipeline...")
    df = generate_dataset()
    print(f"Saved {len(df):,} rows to {SESSIONIZED_PATH}")
    print(f"  Sessions  : {df['session_id'].nunique()}")
    print(f"  Templates : {df['template_id'].nunique()}")
    print(f"  Hosts     : {df['host'].nunique()}")
    anomaly_rate = df["log_level"].isin(["CRITICAL", "ERROR"]).mean()
    print(f"  Anomaly rate (ERROR+CRITICAL): {anomaly_rate:.1%}")
    print(f"  Ground truth written → {GROUND_TRUTH_PATH}")
