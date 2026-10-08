"""The ledger reader's job is to be trustworthy and to stay read-only. Both are tested."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from harness.ledger import LedgerError, open_ledger
from tests.ledger_fixture import build_ledger


def _one_session(tmp_path: Path):
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("s1").start("s1").user("s1", "hello").tool_call("s1", "read", {"path": "a.txt"})
    builder.close()
    return tmp_path / "sessions.db"


def test_opens_read_only_and_refuses_to_write(tmp_path: Path) -> None:
    path = _one_session(tmp_path)
    with open_ledger(path) as ledger:
        with pytest.raises(sqlite3.OperationalError):
            ledger._connection.execute("INSERT INTO events (session_key, seq) VALUES ('x', 1)")


def test_query_only_is_enforced_independently_of_the_uri(tmp_path: Path) -> None:
    # The URI and the pragma are two locks on the same door. This checks the second one exists, so
    # that a future refactor which rebuilt the URI would still be refused by SQLite.
    path = _one_session(tmp_path)
    with open_ledger(path) as ledger:
        assert ledger._connection.execute("PRAGMA query_only").fetchone() == (1,)


def test_missing_ledger_is_an_error_not_a_new_database(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(LedgerError, match="no ledger at"):
        open_ledger(missing)
    # The distinction that matters: creating the file would have been a write.
    assert not missing.exists()


def test_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LedgerError, match="is a directory"):
        open_ledger(tmp_path)


def test_reads_events_back_with_data_and_actor_decoded(tmp_path: Path) -> None:
    path = _one_session(tmp_path)
    with open_ledger(path) as ledger:
        events = ledger.events("s1")
        assert [e.type for e in events] == ["session/start", "user/message", "tool/call"]
        call = events[-1]
        assert call.data["name"] == "read"
        assert call.data["args"] == {"path": "a.txt"}
        assert call.actor["role"] == "owner"
        assert call.lane == "main"
        assert call.origin == "model"


def test_filters_mirror_the_harness_own_scan(tmp_path: Path) -> None:
    path = _one_session(tmp_path)
    with open_ledger(path) as ledger:
        assert len(ledger.events("s1", types=["tool/call"])) == 1
        assert len(ledger.events("s1", from_seq=2)) == 2
        assert len(ledger.events("s1", to_seq=1)) == 1
        assert len(ledger.events("s1", lane="main")) == 3
        assert ledger.events("s1", lane="other") == ()


def test_session_rows_expose_fork_ancestry(tmp_path: Path) -> None:
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("parent", created_at="2026-10-01T00:00:00Z").start("parent")
    builder.session("child", created_at="2026-10-01T00:01:00Z", parent_key="parent", boundary_seq=2)
    builder.start("child", parent={"key": "parent", "boundarySeq": 2})
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        assert ledger.session_keys() == ("parent", "child")
        rows = {row.session_key: row for row in ledger.sessions()}
        assert rows["child"].parent_key == "parent"
        assert rows["child"].boundary_seq == 2
        assert rows["parent"].parent_key is None


def test_event_types_and_counts_come_from_the_file(tmp_path: Path) -> None:
    path = _one_session(tmp_path)
    with open_ledger(path) as ledger:
        assert ledger.event_types("s1") == ("session/start", "tool/call", "user/message")
        assert ledger.row_count("s1") == 3
        assert ledger.last_seq("s1") == 3
        assert ledger.has_session("s1")
        assert not ledger.has_session("nope")


def test_unknown_event_types_are_read_rather_than_refused(tmp_path: Path) -> None:
    # Agnes Harness is pre-alpha and grows its vocabulary. A reader that refused an unknown type
    # would stop producing trajectories on an upgrade, and the experiment's own record would be the
    # thing that broke.
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("s1").start("s1")
    builder.append("s1", "x/future-event", {"something": ["new"]})
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        events = ledger.events("s1")
        assert events[-1].type == "x/future-event"
        assert events[-1].data == {"something": ["new"]}


def test_a_non_json_data_column_is_returned_as_text_rather_than_raising(tmp_path: Path) -> None:
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("s1").start("s1")
    builder._connection.execute(
        "INSERT INTO events (session_key, seq, ts, id, type, lane, actor, origin, trust, data)"
        " VALUES ('s1', 99, 't', 'i', 'odd', ?, ?, 'system', 'trusted', 'not json')",
        (b"main", "{}"),
    )
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        assert ledger.events("s1")[-1].data == "not json"


def test_a_session_with_no_sessions_row_still_lists_from_events(tmp_path: Path) -> None:
    # A fixture, or a ledger written before the sessions table existed. Reporting no sessions at all
    # would be less useful than reporting the ones the events prove.
    builder = build_ledger(tmp_path / "sessions.db")
    builder.start("orphan")
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        assert ledger.session_keys() == ("orphan",)
