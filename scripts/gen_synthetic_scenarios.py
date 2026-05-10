import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame({
    "log_id": [f"syn_{i}" for i in range(n)],
    "session_id": [f"sess_{i // 30}" for i in range(n)],
    "frequency_score": rng.normal(1.0, 0.3, n).clip(0),
    "burstiness_score": rng.uniform(0.0, 0.5, n),
    "zscore_base": rng.normal(0.0, 1.0, n),
    "time_delta_prev": rng.exponential(2.0, n),
    "time_delta_session_start": rng.uniform(0.0, 300.0, n),
    "inter_arrival_rate": rng.exponential(5.0, n),
    "severity_weight": rng.uniform(0.1, 0.5, n),
    "counter_proximity": rng.uniform(0.0, 0.2, n),
    "log_template": rng.choice(["CPU_HIGH", "IF_DOWN", "LOGIN_FAIL", "PORT_SCAN", "NORMAL"], n),
    "ground_truth_anomaly": False,
})

df.loc[270:, ["frequency_score", "severity_weight", "counter_proximity"]] = [50.0, 1.0, 1.0]
df.loc[270:, "ground_truth_anomaly"] = True

Path("data/synthetic").mkdir(parents=True, exist_ok=True)
df.to_parquet("data/synthetic/scenario_001.parquet", index=False)
print("Done — scenario_001.parquet written to data/synthetic/")