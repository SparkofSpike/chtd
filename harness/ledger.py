"""Read an Agnes Harness session ledger.

The ledger is the harness's append-only record of what a session did: every model decision, every
tool result, every register write. CHTD's whole method rests on reading it — the student's
trajectory defines the states the teacher is dropped into, and the comparison between arms is a
comparison of ledgers.

Two properties are enforced here rather than promised:

* **Read-only.** The connection is opened with SQLite's ``mode=ro`` URI, so a bug in this module
  cannot corrupt the record the experiment rests on. Opening a ledger that does not exist is an
  error, not an empty database, because creating one would be a write.
* **No schema assumption.** Column and event-type sets are read from the file, not declared here.
  Agnes Harness is pre-alpha and adds event types; a reader that refused an unknown type would stop
  working the moment the harness grew, which is the opposite of what a frozen experiment needs.

The storage layout this reads is `packages/host/src/adapters/ddl.ts` in Agnes Harness.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

__all__ = ["Event", "Ledger", "LedgerError", "SessionRow", "open_ledger"]


class LedgerError(RuntimeError):
    """A ledger could not be read, or was asked to do something a ledger may not do."""


# The harness writes a lane as a BLOB. SQLite hands a TEXT column back through a NUL-terminated
# C string, so a lane stored as text would stop at its first NUL and two distinct lanes could read
# back identically; bytes round-trip, which is why the column is a BLOB in the harness's own DDL.
_DEFAULT_LANE = b"main"


def _decode_lane(raw: Any) -> str:
    if raw is None:
        return _DEFAULT_LANE.decode()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)


def _decode_json(raw: Any) -> Any:
    """Parse a JSON column, tolerating the NULL and the already-decoded value.

    A column this module does not expect to be JSON is never passed here. When a value claims to be
    JSON and is not, the raw text is returned rather than raising: a reader that dies on one
    surprising row cannot produce a partial trajectory, and a partial trajectory is worse than a
    visible oddity in the data it does produce.
    """
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", "replace")
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


@dataclass(frozen=True)
class Event:
    """One row of the ledger, decoded."""

    session_key: str
    seq: int
    ts: str
    id: str
    type: str
    lane: str
    origin: str
    trust: str
    register: str | None
    data: Any
    actor: Any
    source_event_seqs: tuple[int, ...]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"#{self.seq} {self.type}"


@dataclass(frozen=True)
class SessionRow:
    """One row of the ``sessions`` table: the session's identity and, if it is a fork, its origin."""

    session_key: str
    format_version: int
    parent_key: str | None
    boundary_seq: int | None
    created_at: str


class Ledger:
    """A read-only view of one session ledger.

    Construct through :func:`open_ledger`; the constructor takes an already-opened connection and
    does not check that it is read-only, because only this module opens connections.
    """

    def __init__(self, connection: sqlite3.Connection, path: Path) -> None:
        self._connection = connection
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reading ---------------------------------------------------------------------------------

    def session_keys(self) -> tuple[str, ...]:
        """Every session in the ledger, in creation order.

        A fork's child is a row here too, which is the point: the child is the teacher's run at the
        student's states, and telling the two apart is a query rather than a naming convention.
        """
        rows = self._connection.execute(
            "SELECT session_key FROM sessions ORDER BY created_at, session_key"
        ).fetchall()
        if rows:
            return tuple(row[0] for row in rows)
        # A ledger whose sessions table predates this read, or a fixture that writes only events:
        # fall back to the events themselves rather than reporting no sessions at all.
        rows = self._connection.execute(
            "SELECT DISTINCT session_key FROM events ORDER BY session_key"
        ).fetchall()
        return tuple(row[0] for row in rows)

    def sessions(self) -> tuple[SessionRow, ...]:
        rows = self._connection.execute(
            "SELECT session_key, format_version, parent_key, boundary_seq, created_at "
            "FROM sessions ORDER BY created_at, session_key"
        ).fetchall()
        return tuple(
            SessionRow(
                session_key=row[0],
                format_version=int(row[1]),
                parent_key=row[2],
                boundary_seq=row[3],
                created_at=row[4],
            )
            for row in rows
        )

    def has_session(self, session_key: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM events WHERE session_key = ? LIMIT 1", (session_key,)
        ).fetchone()
        return row is not None

    def events(
        self,
        session_key: str,
        *,
        from_seq: int | None = None,
        to_seq: int | None = None,
        types: Sequence[str] | None = None,
        lane: str | None = None,
    ) -> tuple[Event, ...]:
        """Events for one session, ordered by seq.

        The filters mirror the harness's own scan, so a caller can ask the same question of the
        ledger that the harness asks of itself.
        """
        sql = [
            "SELECT session_key, seq, ts, id, type, lane, origin, trust, register, data, actor,"
            " source_event_seqs FROM events WHERE session_key = ?"
        ]
        params: list[Any] = [session_key]
        if from_seq is not None:
            sql.append("AND seq >= ?")
            params.append(from_seq)
        if to_seq is not None:
            sql.append("AND seq <= ?")
            params.append(to_seq)
        if types:
            sql.append(f"AND type IN ({','.join('?' * len(types))})")
            params.extend(types)
        if lane is not None:
            sql.append("AND lane = ?")
            params.append(lane.encode("utf-8"))
        sql.append("ORDER BY seq")
        rows = self._connection.execute(" ".join(sql), params).fetchall()
        return tuple(self._event(row) for row in rows)

    def iter_events(self, session_key: str, **kwargs: Any) -> Iterator[Event]:
        yield from self.events(session_key, **kwargs)

    def event_types(self, session_key: str) -> tuple[str, ...]:
        """The distinct event types this session used, so a reader can see what it is working with."""
        rows = self._connection.execute(
            "SELECT DISTINCT type FROM events WHERE session_key = ? ORDER BY type", (session_key,)
        ).fetchall()
        return tuple(row[0] for row in rows)

    def row_count(self, session_key: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM events WHERE session_key = ?", (session_key,)
        ).fetchone()
        return int(row[0])

    def last_seq(self, session_key: str) -> int:
        row = self._connection.execute(
            "SELECT MAX(seq) FROM events WHERE session_key = ?", (session_key,)
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def has_table(self, name: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        return row is not None

    def _event(self, row: Sequence[Any]) -> Event:
        source_seqs = _decode_json(row[11])
        return Event(
            session_key=row[0],
            seq=int(row[1]),
            ts=row[2],
            id=row[3],
            type=row[4],
            lane=_decode_lane(row[5]),
            origin=row[6],
            trust=row[7],
            register=row[8],
            data=_decode_json(row[9]),
            actor=_decode_json(row[10]),
            source_event_seqs=tuple(int(x) for x in source_seqs) if isinstance(source_seqs, list) else (),
        )


def open_ledger(path: str | Path) -> Ledger:
    """Open a ledger read-only.

    ``mode=ro`` is the mechanism, not a convention: a write attempted through this connection fails
    in SQLite rather than in a code review. The file must already exist — a missing ledger is an
    error, because an empty database at that path would be indistinguishable from a session that
    recorded nothing, and the two mean very different things to an experiment.
    """
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise LedgerError(f"no ledger at {resolved}")
    if resolved.is_dir():
        raise LedgerError(f"{resolved} is a directory, not a ledger file")
    uri = f"file:{resolved.as_posix()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    except sqlite3.Error as error:  # pragma: no cover - platform-dependent
        raise LedgerError(f"cannot open {resolved} read-only: {error}") from error
    try:
        # Two independent locks on the same door, both enforced by SQLite rather than by a check
        # written here. `mode=ro` is a property of the handle: a write through it fails with
        # "attempt to write a readonly database". `query_only` is a property of the connection: it
        # refuses any statement that would change data even if the handle were writable. A later
        # refactor that rebuilt the URI would lose the first without losing the second, and the
        # failure mode of a misread ledger is silent corruption of the record the experiment rests
        # on.
        #
        # There is deliberately no probe that attempts a write to find out. An earlier version
        # tried `BEGIN IMMEDIATE`; it turned out that a read-only connection accepts that lock, so
        # the probe refused every healthy ledger while a plain `INSERT` was in fact being refused
        # by the URI. What actually establishes the property is tests/test_ledger.py, which opens a
        # real ledger and asserts the write fails — testing it where testing is safe, instead of
        # touching the reader's own file to prove a point.
        connection.execute("PRAGMA query_only = ON")
        if connection.execute("PRAGMA query_only").fetchone() != (1,):
            raise LedgerError(f"could not put {resolved} into enforced read-only mode")
    except Exception:
        connection.close()
        raise
    return Ledger(connection, resolved)
