"""
parsing/log_parser.py
=====================
Drain3-inspired online log parser that clusters raw log messages into templates.

Algorithm summary (Drain)
--------------------------
1. Tokenise the message by whitespace.
2. Index by (token_count, first_token) in a prefix tree.
3. For each candidate cluster, compute token-match ratio.
4. Assign to the best-matching cluster (ratio > SIM_THRESHOLD) or create new.
5. Replace differing tokens in the cluster template with the wildcard <*>.

Output
------
add_log_message(msg) -> (template_id: str, template: str)
    template_id  -- stable slug derived from the template tokens, e.g. "IF_DOWN"
    template     -- human-readable template string with <*> placeholders

Known limitations
-----------------
- Numeric-only tokens are always treated as wildcards regardless of threshold.
- Template IDs are generated from the first non-wildcard, non-numeric token
  run; they may collide for very similar templates.  Use template strings as
  the canonical key if uniqueness matters.
- This implementation is single-threaded and in-memory; suitable for batch
  parsing of up to ~1M lines on a typical developer machine.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_THRESHOLD: float = 0.5   # minimum fraction of matching tokens to merge
MAX_CHILDREN: int = 100      # max clusters per (length, first_token) bucket
WILDCARD: str = "<*>"

_NUMERIC_RE = re.compile(r"^\d+(\.\d+)?$")
_WORD_RE = re.compile(r"[A-Z_]{2,}")   # candidate template keyword


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LogCluster:
    """A single log template cluster."""
    template: List[str]       # list of tokens; variable positions hold WILDCARD
    log_ids: List[str] = field(default_factory=list)
    count: int = 0

    def template_str(self) -> str:
        return " ".join(self.template)

    def template_id(self) -> str:
        """Derive a short slug from the first meaningful keyword token."""
        keywords = [
            t for t in self.template
            if t != WILDCARD and not _NUMERIC_RE.match(t) and len(t) > 1
        ]
        if keywords:
            # Use up to 3 keywords joined by underscore
            slug = "_".join(keywords[:3]).upper()
            # Truncate to 40 chars
            if len(slug) > 40:
                slug = slug[:40]
            return slug
        # Fallback: hash of template string
        h = hashlib.md5(self.template_str().encode()).hexdigest()[:8]
        return f"TMPL_{h.upper()}"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class DrainParser:
    """Streaming Drain log parser.

    Usage
    -----
    parser = DrainParser()
    for line in log_lines:
        template_id, template = parser.add_log_message(line)
    """

    def __init__(
        self,
        sim_threshold: float = SIM_THRESHOLD,
        max_children: int = MAX_CHILDREN,
    ) -> None:
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        # prefix_tree[length][first_token] -> list[LogCluster]
        self._tree: Dict[int, Dict[str, List[LogCluster]]] = defaultdict(
            lambda: defaultdict(list)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_log_message(
        self, message: str, log_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """Parse one log message and return its template.

        Args:
            message: The raw log message text (no timestamp/hostname prefix).
            log_id:  Optional identifier stored in the cluster for tracing.

        Returns:
            (template_id, template_string) tuple.
        """
        tokens = self._tokenize(message)
        if not tokens:
            return "EMPTY", "<empty>"

        cluster = self._match_or_create(tokens, log_id)
        return cluster.template_id(), cluster.template_str()

    def all_templates(self) -> List[Tuple[str, str, int]]:
        """Return list of (template_id, template_string, count) for all clusters."""
        result = []
        for length_map in self._tree.values():
            for clusters in length_map.values():
                for c in clusters:
                    result.append((c.template_id(), c.template_str(), c.count))
        return sorted(result, key=lambda x: -x[2])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(message: str) -> List[str]:
        """Split message into tokens, normalising numeric-only tokens."""
        raw = message.split()
        return [
            WILDCARD if _NUMERIC_RE.match(t) else t
            for t in raw
        ]

    def _match_or_create(
        self, tokens: List[str], log_id: Optional[str]
    ) -> LogCluster:
        length = len(tokens)
        first = tokens[0] if tokens[0] != WILDCARD else "<*>"

        candidates = self._tree[length][first]

        best_cluster: Optional[LogCluster] = None
        best_sim: float = -1.0

        for cluster in candidates:
            sim = self._similarity(tokens, cluster.template)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_sim >= self.sim_threshold and best_cluster is not None:
            self._update_template(best_cluster, tokens)
            best_cluster.count += 1
            if log_id:
                best_cluster.log_ids.append(log_id)
            return best_cluster

        # Create new cluster
        new_cluster = LogCluster(template=list(tokens), count=1)
        if log_id:
            new_cluster.log_ids.append(log_id)

        if len(candidates) < self.max_children:
            candidates.append(new_cluster)

        return new_cluster

    @staticmethod
    def _similarity(tokens: List[str], template: List[str]) -> float:
        """Fraction of positions where tokens[i] == template[i] (wildcards always match)."""
        if len(tokens) != len(template):
            return 0.0
        matches = sum(
            1 for t, tmpl in zip(tokens, template)
            if tmpl == WILDCARD or t == tmpl
        )
        return matches / len(template)

    @staticmethod
    def _update_template(cluster: LogCluster, tokens: List[str]) -> None:
        """Replace positions where tokens differ from current template with WILDCARD."""
        for i, (tok, tmpl) in enumerate(zip(tokens, cluster.template)):
            if tmpl != WILDCARD and tok != tmpl:
                cluster.template[i] = WILDCARD
