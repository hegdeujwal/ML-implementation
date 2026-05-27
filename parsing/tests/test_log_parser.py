"""
Tests for parsing/log_parser.py — DrainParser, template_id stability, collision detection.
"""

from __future__ import annotations

from parsing.log_parser import DrainParser, LogCluster, WILDCARD


# ---------------------------------------------------------------------------
# Template ID generation
# ---------------------------------------------------------------------------

def test_template_id_uses_keyword_tokens():
    c = LogCluster(template=["OSPF", "neighbor", "state", "changed"])
    assert c.template_id() == "OSPF_NEIGHBOR_STATE"


def test_template_id_skips_wildcards():
    c = LogCluster(template=[WILDCARD, "PORT", "DOWN", WILDCARD])
    assert c.template_id() == "PORT_DOWN"


def test_template_id_hash_fallback_when_no_keywords():
    c = LogCluster(template=[WILDCARD, WILDCARD])
    tid = c.template_id()
    assert tid.startswith("TMPL_")


# ---------------------------------------------------------------------------
# Fix 2 — resolve_template_id collision detection
# ---------------------------------------------------------------------------

def test_resolve_template_id_no_collision():
    parser = DrainParser()
    c = LogCluster(template=["OSPF", "NEIGHBOR", "DOWN"])
    result = parser.resolve_template_id(c)
    assert result == "OSPF_NEIGHBOR_DOWN"


def test_resolve_template_id_same_cluster_idempotent():
    parser = DrainParser()
    c = LogCluster(template=["OSPF", "NEIGHBOR", "DOWN"])
    r1 = parser.resolve_template_id(c)
    r2 = parser.resolve_template_id(c)
    assert r1 == r2


def test_resolve_template_id_collision_appends_suffix():
    parser = DrainParser()
    c1 = LogCluster(template=["OSPF", "NEIGHBOR", "STATE", "DOWN"])
    c2 = LogCluster(template=["OSPF", "NEIGHBOR", "STATE", "UP"])
    # Both produce base slug "OSPF_NEIGHBOR_STATE"
    assert c1.template_id() == c2.template_id() == "OSPF_NEIGHBOR_STATE"

    r1 = parser.resolve_template_id(c1)
    r2 = parser.resolve_template_id(c2)

    assert r1 == "OSPF_NEIGHBOR_STATE"          # first claimant — no suffix
    assert r2.startswith("OSPF_NEIGHBOR_STATE_") # collision — gets suffix
    assert r1 != r2


def test_resolve_template_id_suffix_is_4_char_hex():
    parser = DrainParser()
    c1 = LogCluster(template=["OSPF", "NEIGHBOR", "STATE", "DOWN"])
    c2 = LogCluster(template=["OSPF", "NEIGHBOR", "STATE", "UP"])
    parser.resolve_template_id(c1)
    r2 = parser.resolve_template_id(c2)
    suffix = r2.split("_")[-1]
    assert len(suffix) == 4
    assert all(ch in "0123456789abcdef" for ch in suffix)


# ---------------------------------------------------------------------------
# Fix 3 — add_log_message_cluster + two-pass stability
# ---------------------------------------------------------------------------

def test_add_log_message_cluster_returns_cluster():
    parser = DrainParser()
    cluster = parser.add_log_message_cluster("OSPF neighbor state changed to DOWN")
    assert isinstance(cluster, LogCluster)


def test_two_pass_template_id_is_stable():
    """All rows matched to the same cluster must get the same final slug."""
    parser = DrainParser()
    messages = [
        "OSPF neighbor state changed to DOWN",
        "OSPF neighbor state changed to UP",    # same cluster after wildcard merge
        "OSPF neighbor state changed to INIT",
    ]
    clusters = [parser.add_log_message_cluster(m) for m in messages]
    # After all messages parsed, resolve — every row that shares a cluster object
    # must get the same slug.
    slugs = [parser.resolve_template_id(c) for c in clusters]
    # All three messages merge into one cluster
    assert len(set(id(c) for c in clusters)) == 1, "expected single cluster"
    assert slugs[0] == slugs[1] == slugs[2]


def test_add_log_message_cluster_empty_message_returns_sentinel():
    parser = DrainParser()
    c1 = parser.add_log_message_cluster("")
    c2 = parser.add_log_message_cluster("")
    # Both calls return the same sentinel cluster object
    assert c1 is c2


def test_add_log_message_unchanged():
    """Original add_log_message API must still work correctly."""
    parser = DrainParser()
    tid, tstr = parser.add_log_message("PORT link state changed to DOWN")
    assert isinstance(tid, str)
    assert isinstance(tstr, str)
    assert len(tid) > 0
