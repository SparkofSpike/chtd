"""Build a synthetic Agnes Harness ledger for tests.

The tables below are copied from the harness's own DDL (`packages/host/src/adapters/ddl.ts`), not
invented. A fixture that invented its own schema would let this repository's reader pass tests
against a ledger the harness never writes, which is the one failure a test suite cannot catch by
itself.

Only the tables CHTD reads are created. A ledger missing `budget_scopes` is a ledger, and the
reader must not require tables it never queries.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

__all__ = ["LedgerBuilder", "build_ledger"]

DDL = (
    """CREATE TABLE IF NOT EXISTS events (
     session_key TEXT NOT NULL, seq INTEGER NOT NULL, ts TEXT NOT NULL, id TEXT NOT NULL, type TEXT NOT NULL,
     lane BLOB NOT NULL DEFAULT x'6d61696e', v INTEGER NOT NULL DEFAULT 1,
     actor TEXT NOT NULL, origin TEXT NOT NULL, trust TEXT NOT NULL,
     register TEXT, ignorable INTEGER, surface_op TEXT, source_event_seqs TEXT, data TEXT NOT NULL,
     integrity_mode TEXT, integrity_prev TEXT, integrity_digest TEXT,
     PRIMARY KEY (session_key, seq))""",
    "CREATE INDEX IF NOT EXISTS events_type ON events (session_key, type, seq)",
    """CREATE TABLE IF NOT EXISTS sessions (
     session_key TEXT PRIMARY KEY, format_version INTEGER NOT NULL DEFAULT 1, parent_key TEXT,
     boundary_seq INTEGER, created_at TEXT NOT NULL)""",
)

_ACTOR = {"id": "u", "org": "local", "role": "owner", "deptPath": [], "attrs": {}}


class LedgerBuilder:
    """Appends events to a ledger the way the harness does, for tests to read back."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._seq: dict[str, int] = {}
        self._ordinal = 0

    def session(
        self,
        key: str,
        *,
        created_at: str = "2026-10-01T00:00:00Z",
        parent_key: str | None = None,
        boundary_seq: int | None = None,
    ) -> "LedgerBuilder":
        self._connection.execute(
            "INSERT OR REPLACE INTO sessions (session_key, format_version, parent_key, boundary_seq,"
            " created_at) VALUES (?, 1, ?, ?, ?)",
            (key, parent_key, boundary_seq, created_at),
        )
        return self

    def fork_from(
        self,
        parent_key: str,
        child_key: str,
        boundary_seq: int,
        *,
        created_at: str = "2026-10-01T00:01:00Z",
    ) -> "LedgerBuilder":
        """Give the child the parent's prefix, the way the harness's `forkInto` does.

        The harness copies the parent's rows up to the boundary into the child's ledger and starts
        the child's own sequence there (storage.createChild). A fixture that gave the child only its
        own rows would model a session that shared no state with its parent — which is not a fork,
        and would let a broken prefix digest pass a test named after forks.
        """
        columns = (
            "seq, ts, id, type, lane, actor, origin, trust, register, ignorable, surface_op,"
            " source_event_seqs, data, integrity_mode, integrity_prev, integrity_digest"
        )
        rows = self._connection.execute(
            f"SELECT {columns} FROM events WHERE session_key = ? AND seq <= ? ORDER BY seq",
            (parent_key, boundary_seq),
        ).fetchall()
        if not rows:
            raise ValueError(f"{parent_key} has no events up to seq {boundary_seq} to fork from")
        for row in rows:
            placeholders = ",".join("?" * (len(row) + 1))
            self._connection.execute(
                f"INSERT INTO events (session_key, {columns}) VALUES ({placeholders})",
                (child_key, *row),
            )
        self._seq[child_key] = boundary_seq
        return self.session(child_key, created_at=created_at, parent_key=parent_key, boundary_seq=boundary_seq)

    def start(
        self,
        key: str,
        *,
        preset: str | None = "chtd",
        resolved_profile_hash: str | None = "sha256-" + "a" * 64,
        model_settings: Sequence[dict[str, Any]] | None = None,
        parent: dict[str, Any] | None = None,
        agnes_version: str = "0.1.0",
    ) -> "LedgerBuilder":
        data: dict[str, Any] = {
            "key": key,
            "resolvedProfileHash": resolved_profile_hash,
            "preset": preset,
            "agnesVersion": agnes_version,
        }
        if model_settings is not None:
            data["modelSettings"] = list(model_settings)
        if parent is not None:
            data["parent"] = parent
        return self.append(key, "session/start", data, origin="system")

    def user(self, key: str, text: str) -> "LedgerBuilder":
        return self.append(key, "user/message", {"content": [{"type": "text", "text": text}]})

    def turn_start(self, key: str) -> "LedgerBuilder":
        return self.append(key, "turn/start", {"trigger": "user"}, origin="system")

    def step_start(self, key: str) -> "LedgerBuilder":
        return self.append(key, "step/start", {"step": 1}, origin="system")

    def tool_call(
        self,
        key: str,
        name: str,
        args: Any,
        *,
        tool_use_id: str | None = None,
    ) -> "LedgerBuilder":
        self._ordinal += 1
        return self.append(
            key,
            "tool/call",
            {
                "toolUseId": tool_use_id or f"call-{self._ordinal}",
                "name": name,
                "args": args,
                "ordinal": self._ordinal,
            },
            origin="model",
        )

    def tool_result(
        self,
        key: str,
        tool_use_id: str,
        *,
        is_error: bool = False,
        text: str = "ok",
    ) -> "LedgerBuilder":
        return self.append(
            key,
            "tool/result",
            {
                "toolUseId": tool_use_id,
                "isError": is_error,
                "content": [{"type": "text", "text": text}],
            },
            origin="system",
        )

    def assistant(self, key: str, text: str) -> "LedgerBuilder":
        return self.append(key, "assistant/message", {"content": [{"type": "text", "text": text}]}, origin="model")

    def close(self) -> None:
        """Release the write connection, the way the harness process would on exit."""
        self._connection.close()

    def append(
        self,
        key: str,
        event_type: str,
        data: Any,
        *,
        origin: str = "system",
        lane: str = "main",
        ts: str | None = None,
    ) -> "LedgerBuilder":
        self._seq[key] = self._seq.get(key, 0) + 1
        seq = self._seq[key]
        self._connection.execute(
            "INSERT INTO events (session_key, seq, ts, id, type, lane, actor, origin, trust, data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                seq,
                ts or f"2026-10-01T00:00:{seq:02d}Z",
                f"evt-{key}-{seq}",
                event_type,
                lane.encode("utf-8"),
                json.dumps(_ACTOR),
                origin,
                "trusted",
                json.dumps(data, ensure_ascii=False),
            ),
        )
        return self


def build_ledger(path: str | Path) -> LedgerBuilder:
    """Create a fresh ledger file and return a builder for it.

    The file is created through a writable connection here, which is the point: the fixture plays
    the harness's role — the thing that writes — so that the reader under test can be checked
    against a real file rather than a mock of one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Autocommit, because the harness's own storage adapter commits each append and a reader in
    # another connection only ever sees committed rows. A fixture left in Python's implicit
    # transaction would write nothing a reader could find, and every test would pass against an
    # empty ledger — which is the failure this suite exists to catch, so it must not be built in.
    connection = sqlite3.connect(target, isolation_level=None)
    for statement in DDL:
        connection.execute(statement)
    return LedgerBuilder(connection)
