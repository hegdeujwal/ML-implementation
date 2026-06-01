import time
from typing import Callable

import numpy as np
import pandas as pd


def _make_logs(n: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-05-01", periods=n, freq="s")
    return pd.DataFrame(
        {
            "log_id": [f"log_{i:07d}" for i in range(n)],
            "raw_text": [f"event {i}" for i in range(n)],
            "timestamp": timestamps,
            "severity_weight": np.random.random(n),
            "isolation_score": np.random.random(n),
            "zscore": np.random.random(n),
        }
    )


def _mock_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["final_score"] = (
        0.4 * out["isolation_score"] + 0.35 * out["zscore"] + 0.25 * out["severity_weight"]
    )
    out["label"] = pd.cut(
        out["final_score"],
        bins=[-0.01, 0.2, 0.5, 0.75, 1.0],
        labels=["ignore", "low", "medium", "critical"],
    ).astype(str)
    out["incident_id"] = [f"inc_{i//25:05d}" for i in range(len(out))]
    return out


def benchmark(run_fn: Callable[[pd.DataFrame], pd.DataFrame], n_rows: int) -> dict:
    df = _make_logs(n_rows)
    start = time.perf_counter()
    out = run_fn(df)
    elapsed = time.perf_counter() - start
    return {
        "rows": n_rows,
        "seconds": round(elapsed, 4),
        "rows_per_second": round(n_rows / elapsed, 2) if elapsed > 0 else float("inf"),
        "output_rows": len(out),
    }


def run_benchmarks() -> list[dict]:
    sizes = [1000, 10000]
    results = [benchmark(_mock_pipeline, n) for n in sizes]
    for r in results:
        print(
            f"rows={r['rows']}, seconds={r['seconds']}, "
            f"rows_per_second={r['rows_per_second']}, output_rows={r['output_rows']}"
        )
    return results


if __name__ == "__main__":
    run_benchmarks()
