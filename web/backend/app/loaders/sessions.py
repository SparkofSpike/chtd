"""Read-only loading of records.jsonl ledgers and the SQLite event store."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .paths import PATHS
from .trajectory import build_session


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            out.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return out


# The seed-1 CHTD arm was collected in two shards (33 + 15 tasks, disjoint, same train => one arm);
# they are presented as one series. Keyed by directory name.
SHARD_OF: dict[str, str] = {
    "holdout4-chtd-seed1-resume": "holdout4-chtd-seed1-nocheck",
}

# Stable group ids the frontend pairs on: an arm and its shards share one group, so a task that
# only exists in the resume shard still pairs against the base run of the same task.
def arm_group(arm_dir: str) -> str:
    canonical = SHARD_OF.get(arm_dir, arm_dir)
    for prefix in ("holdout4-",):
        if canonical.startswith(prefix):
            canonical = canonical[len(prefix):]
    return canonical.removesuffix("-nocheck")


def discover_records(paths=PATHS) -> list[dict[str, Any]]:
    """Scan v3 rollouts + demo, return flattened record index entries."""
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    # v3 rollouts: one records.jsonl per arm directory. A canonical `*-nocheck` directory is read
    # before any scratch/shard directory of the same arm, so when a task appears in both (an
    # aborted attempt alongside the real run) the recorded run is the one kept.
    roll_root = paths.v3_rollouts
    if roll_root.is_dir():
        arm_dirs = sorted(
            (d for d in roll_root.iterdir() if (d / "records.jsonl").is_file()),
            key=lambda d: (0 if "nocheck" in d.name else 1, d.name),
        )
        for arm_dir in arm_dirs:
            rfile = arm_dir / "records.jsonl"
            group = arm_group(arm_dir.name)
            for rec in _read_jsonl(rfile):
                key = (group, rec.get("task_id"))
                if key in seen:
                    continue  # a shard repeating a task never overrides the first read
                seen.add(key)
                entries.append({
                    "batch": "holdout-4-20261014-v3",
                    "arm": arm_dir.name,
                    "arm_group": group,
                    "shard": SHARD_OF.get(arm_dir.name),
                    "arm_kind": rec.get("arm"),
                    "task_id": rec.get("task_id"),
                    "passed": rec.get("passed"),
                    "seconds": rec.get("seconds"),
                    "exit": rec.get("exit"),
                    "timed_out": rec.get("timed_out"),
                    "session_key": rec.get("session_key"),
                    "track_sha256": rec.get("track_sha256"),
                    "preset": rec.get("preset"),
                    "profile_hash": rec.get("resolved_profile_hash"),
                    "answer": rec.get("answer"),
                    "check_output": rec.get("check_output"),
                    "stderr_tail": rec.get("stderr_tail"),
                    "db": "v3",
                    "source_file": str(rfile),
                })

    # demo rollout
    if paths.demo_rollout.exists():
        for rec in _read_jsonl(paths.demo_rollout):
            entries.append({
                "batch": "demo-20261015",
                "arm": "chtd-demo",
                "arm_group": "chtd-demo",
                "arm_kind": rec.get("arm"),
                "task_id": rec.get("task_id"),
                "passed": rec.get("passed"),
                "seconds": rec.get("seconds"),
                "exit": rec.get("exit"),
                "timed_out": rec.get("timed_out"),
                "session_key": rec.get("session_key"),
                "track_sha256": rec.get("track_sha256"),
                "preset": rec.get("preset"),
                "profile_hash": rec.get("resolved_profile_hash"),
                "answer": rec.get("answer"),
                "check_output": rec.get("check_output"),
                "stderr_tail": rec.get("stderr_tail"),
                "db": "v3",  # best-effort; may not be present
                "source_file": str(paths.demo_rollout),
            })
    return entries


def _db_path(which: str, paths=PATHS) -> Path | None:
    if which == "v3":
        return paths.v3_events_db if paths.v3_events_db.exists() else None
    if which == "baseline":
        return paths.baseline_events_db if paths.baseline_events_db.exists() else None
    return None


def fetch_events(session_key: str, db: str = "v3", paths=PATHS) -> list[dict[str, Any]]:
    """Return parsed events for a session, ordered by seq. Empty list if absent."""
    db_path = _db_path(db, paths)
    if db_path is None or not session_key:
        return []
    try:
        conn = sqlite3.connect(
            "file:" + str(db_path).replace("\\", "/") + "?mode=ro", uri=True
        )
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "SELECT seq, type, data FROM events WHERE session_key=? ORDER BY seq",
            (session_key,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for seq, etype, data in rows:
        try:
            payload = json.loads(data) if isinstance(data, str) else (
                json.loads(data.decode("utf-8", "replace") if isinstance(data, bytes) else "{}")
            )
        except (json.JSONDecodeError, ValueError):
            payload = {}
        out.append({"seq": seq, "type": etype, "data": payload})
    return out


def build_full_session(record: dict[str, Any], paths=PATHS) -> dict[str, Any]:
    """Attach reconstructed steps to a record index entry."""
    events = fetch_events(record.get("session_key", ""), record.get("db", "v3"), paths)
    session = build_session(record, events)
    session["has_events"] = bool(events)
    return session


# The pairings the UI offers first. `chtd-seed1` is the arm whose behaviour matches the training
# material; `chtd-seed0` is kept selectable because its exploration regression on this task set is
# itself a recorded finding (runs/holdout-4-20261014/README.md), not because it flatters the method.
CANON_BASE_GROUP = "base"
CANON_CHTD_GROUP = "chtd-seed1"
CANON_CHTD_FALLBACK = "chtd-seed0"


def build_pairs(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Map each session to the other arms that ran the same task.

    Grouped by `arm_group` so a sharded arm (seed-1 ran in two disjoint halves) pairs as one arm,
    and a task present only in a resume shard still finds its base counterpart. Returns
    session_key -> {arm_group: sibling session_key}.
    """
    by_bt: dict[tuple, dict[str, dict[str, Any]]] = {}
    for r in records:
        group = r.get("arm_group") or r.get("arm")
        by_bt.setdefault((r.get("batch"), r.get("task_id")), {})[group] = r

    pairs: dict[str, dict[str, str]] = {}
    for group_map in by_bt.values():
        keys = {g: r.get("session_key") for g, r in group_map.items() if r.get("session_key")}
        for g, r in group_map.items():
            sk = r.get("session_key")
            if not sk:
                continue
            pairs[sk] = {other: key for other, key in keys.items() if other != g}
    return pairs


def canonical_sort_key(r: dict[str, Any]) -> tuple:
    """Canonical arms first (avoid smoke / baseline-control noise), then group/task."""
    group = r.get("arm_group") or r.get("arm") or ""
    canonical = 0 if ("nocheck" in (r.get("arm") or "") and "smoke" not in (r.get("arm") or "")) else 1
    return (canonical, group, r.get("task_id") or "")
