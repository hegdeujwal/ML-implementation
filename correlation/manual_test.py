"""
correlation/manual_test.py

Run this from the project root to inspect the correlation graph interactively:

    python -m correlation.manual_test

Loads the current sessionized_logs.parquet and prints graph statistics.
Not a pytest test -- kept as a developer diagnostic script.
"""

import pandas as pd

from correlation.graph_builder import build_graph
from correlation.centrality import compute_centrality


def _main() -> None:
    try:
        df = pd.read_parquet("data/processed/sessionized_logs.parquet")
    except FileNotFoundError:
        print("sessionized_logs.parquet not found. Run the pipeline first.")
        return

    g = build_graph(df)

    print("=" * 60)
    print(f"NODES  ({len(g.nodes)} total)")
    print("=" * 60)
    for node, attrs in sorted(g.nodes(data=True), key=lambda x: -x[1].get("frequency", 0))[:20]:
        print(f"  {node:<40} freq={attrs.get('frequency', 0):>5}  cluster={attrs.get('cluster_id', '?')}")

    print()
    print("=" * 60)
    print(f"EDGES  ({len(g.edges)} total)")
    print("=" * 60)
    for u, v, data in sorted(g.edges(data=True), key=lambda e: -e[2].get("weight", 0))[:20]:
        print(f"  {u:<25} -- {v:<25}  weight={data.get('weight', 0):.4f}  pmi={data.get('pmi', 0):.4f}")

    print()
    scores_df = compute_centrality(g, df)
    print("=" * 60)
    print("TOP 10 by centrality_score")
    print("=" * 60)
    top = (
        scores_df[["sequence_number", "centrality_score", "cluster_id", "in_graph"]]
        .sort_values("centrality_score", ascending=False)
        .drop_duplicates("centrality_score")
        .head(10)
    )
    print(top.to_string(index=False))
    print()
    print("Done.")


if __name__ == "__main__":
    _main()
