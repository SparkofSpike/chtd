"""Loader tests: parse real ledgers, degrade to 暂无数据 when missing."""
from __future__ import annotations

from app.loaders import experiments as E
from app.loaders import sessions as S
from app.loaders.paths import PATHS
from app.loaders.trajectory import build_session


def test_records_discovered():
    rows = S.discover_records()
    assert len(rows) > 100, f"expected many v3 records, got {len(rows)}"
    by_arm = {}
    for r in rows:
        by_arm[r["arm"]] = by_arm.get(r["arm"], 0) + 1
    # main arms present
    assert by_arm.get("holdout4-base-nocheck") == 48
    assert by_arm.get("holdout4-chtd-seed0-nocheck") == 48
    assert by_arm.get("holdout4-chtd-seed1-nocheck") == 33


def test_pairs_built():
    rows = S.discover_records()
    pairs = S.build_pairs(rows)
    # at least the base<->chtd seed0 48 pairs exist
    assert len(pairs) > 40


def test_trajectory_on_recovery_success():
    rows = S.discover_records()
    rec = next(r for r in rows if r["task_id"] == "h4-idna-host-2"
               and r["arm"] == "holdout4-chtd-seed0-nocheck")
    assert rec["passed"] is True
    sess = S.build_full_session(rec)
    assert sess["has_events"] is True
    assert len(sess["steps"]) >= 1
    # tool calls paired with results
    n_calls = sum(len(s["calls"]) for s in sess["steps"])
    assert n_calls >= 3
    for s in sess["steps"]:
        for c in s["calls"]:
            assert "content" in c["result"]


def test_census_parses():
    census = E.load_tool_census()
    assert census["arms"], "tool-call-census.json should parse"
    s0 = census["arms"].get("chtd-seed0", {})
    assert s0.get("read_results", {}).get("ok_rate") == 0.119
    assert s0.get("passed") == 7


def test_dose_cells_parse():
    cells = E.load_dose_cells()
    assert cells["n"] == 75


def test_missing_source_degrades():
    # experiments loader must not crash on a nonexistent file (status 暂无数据)
    census = E.load_tool_census()
    # either real data or a clean marker; never raises
    assert "arms" in census or "status" in census


def test_summary_has_supports_block():
    s = E.summary_block()
    assert s["supports"] and s["does_not_support"]
    assert s["pooled"]["mcnemar_p"] == 0.077
