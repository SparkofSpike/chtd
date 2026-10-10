"""Drive harness sessions and collect their trajectories.

One task set, one session per task, one ledger row per run. The module's job is to make a rollout
*recorded* rather than remembered: every run writes what track it ran, on which model, and what it
decided, so that the dataset step can refuse to compare runs that were not comparable and the
write-up can reconstruct which run produced which row.

**The model is a seam.** `Runner` is the interface; `CliRunner` drives a built harness through its
own CLI, and `ReplayRunner` serves a prepared ledger so that everything downstream of the driver can
be tested without a model account, a GPU, or a network. A pipeline that could only be exercised
against a live endpoint would be a pipeline nobody could check.

**Untested path, stated plainly.** `CliRunner` has not driven a rollout against a live model. The
harness does build here — `pnpm --filter @agnes/cli build:local` succeeds once `AGNES_NODE_HEADERS`
points at a Node header cache, and `packages/cli/dist/local/agnes.mjs` runs — and its exact argv was
checked against that binary. Two things came out of doing so: the model slot is `primary` and not
`main` (a lane is where rows land, a slot is which model answers), and `--standalone` is required
because the daemon does not come up on this machine.

With both fixed, a real session was created and its ledger written; `harness/ledger.py` read it
back and classified every row. The turn itself ended in `AUTH route=...` because the probe pointed
at a closed port — but by then the session, its `session/start`, its user message and its full turn
lifecycle were already durable, which is what the reader consumes. Everything downstream of the
runner is tested against ledgers like that one.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from distill.gate import Outcome
from harness.ledger import open_ledger
from harness.trajectory import Trajectory, read_trajectory

__all__ = [
    "CliRunner",
    "ReplayRunner",
    "RolloutRecord",
    "Runner",
    "RunResult",
    "Task",
    "collect",
    "load_tasks",
    "read_track_digest",
    "write_records",
]


@dataclass(frozen=True)
class Task:
    """One thing to ask an agent to do."""

    id: str
    prompt: str
    #: Optional per-task workspace; a rollout that needs files to work on gets its own directory.
    cwd: str | None = None

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> "Task":
        task_id = row.get("id")
        prompt = row.get("prompt")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("a task needs a non-empty string id")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"task {task_id} needs a non-empty string prompt")
        cwd = row.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError(f"task {task_id} has a non-string cwd")
        return cls(id=task_id, prompt=prompt, cwd=cwd)


def load_tasks(path: str | Path) -> tuple[Task, ...]:
    """Read a JSONL task set.

    Ids must be unique: a duplicate would let two runs collide on one output file, and the later
    one would silently replace the earlier — which looks exactly like a task that never ran.
    """
    source = Path(path)
    tasks: list[Task] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                row = json.loads(text)
            except ValueError as error:
                raise ValueError(f"{source}:{number} is not JSON: {error}") from error
            task = Task.from_json(row)
            if task.id in seen:
                raise ValueError(f"{source}:{number} repeats task id {task.id}")
            seen.add(task.id)
            tasks.append(task)
    if not tasks:
        raise ValueError(f"{source} has no tasks")
    return tuple(tasks)


@dataclass(frozen=True)
class RunResult:
    """What a runner hands back: where the run's record landed."""

    session_key: str
    ledger_path: str


class Runner(Protocol):
    """The model seam. Everything downstream of this is model-independent."""

    def run(self, task: Task, *, arm: str, model: str, route: str, home: str,
            cwd: str | None = None) -> RunResult: ...


@dataclass(frozen=True)
class RolloutRecord:
    """One run, as recorded. This is the row the rest of the pipeline reads.

    ``track_sha256`` is on every record because the comparison depends on it: two records that
    disagree on this field describe two different experiments and the dataset step refuses to mix
    them.
    """

    task_id: str
    arm: str
    session_key: str
    ledger_path: str
    track_sha256: str
    preset: str | None
    route: str | None
    model: str | None
    outcome: str = Outcome.UNKNOWN.value
    decisions: int = 0
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "RolloutRecord":
        row = json.loads(text)
        return cls(
            task_id=row["task_id"],
            arm=row["arm"],
            session_key=row["session_key"],
            ledger_path=row["ledger_path"],
            track_sha256=row["track_sha256"],
            preset=row.get("preset"),
            route=row.get("route"),
            model=row.get("model"),
            outcome=row.get("outcome", Outcome.UNKNOWN.value),
            decisions=int(row.get("decisions", 0)),
            notes=row.get("notes", ""),
        )


def read_track_digest(path: str | Path = "packages/chtd/presets/chtd.sha256") -> str:
    """The frozen track's recorded digest, read from the file the package's gate reads.

    Read rather than recomputed. Recomputing would let a rollout record a digest that no gate ever
    checks, and the whole point of the digest is that it is the one the package refuses to load
    without.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no recorded track digest at {source}")
    digest = source.read_text(encoding="utf-8").strip()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{source} does not hold a sha256 hex digest")
    return digest


def write_records(records: Iterable[RolloutRecord], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
    return target


def read_records(path: str | Path) -> tuple[RolloutRecord, ...]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return tuple(
            RolloutRecord.from_json(line)
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )


@dataclass
class CliRunner:
    """Drive the harness's own CLI, one session per task.

    The command is ``agnez -p --cwd <workspace> --model <slot>=<route>/<model> <prompt>``
    against a shared ``AGH_HOME``, with the profile named by ``AGNES_PROFILE``.

    **Where the session key comes from.** A key is
    ``agnes:local:<profile>:<surface>:workspace:<hash of the working directory>``, so two
    runs are told apart by *where they ran*, not by which home they used. Giving each run
    its own workspace therefore separates them on one shared ledger — which is the shape
    the design asks for: *rollouts land on one shared, append-only ledger, so a teacher
    step and a student step are directly comparable.* One home per run would instead
    require a copy of the installed packages and the lockfile for every task, because a
    non-builtin package is resolved through the lockfile and nothing else.

    **`--standalone` is not used, and that is a finding rather than a preference.** The
    flag runs the host embedded in the CLI process, and that path does not consult the
    runtime-snapshot store, so a profile naming a non-builtin package fails with
    ``E_EXT_LOAD: trusted package has no immutable runtime snapshot`` no matter how
    healthy the store is. Driven through the daemon, the same profile and the same track
    load. Measured both ways on this machine; see ``docs/harness-integration.md``.

    The model slot is `primary`, not `main`: the accepted slots are `primary`,
    `escalation`, `fast`, `compaction`, `verifier`, `image` and `video`. `main` is the name
    of a *lane* in the ledger, which is where the confusion came from; a lane is where rows
    land, a slot is which model answers.

    ``env`` carries whatever the deployment's routes need — a credential for a hosted
    teacher, for instance. It is merged over the ambient environment rather than replacing
    it, and it is what keeps a route's API key out of the profile and the lockfile.
    """

    entry: str
    profile: str = "local-dev"
    timeout_seconds: int = 900
    #: Which model slot the session runs on. A field so a deployment can point a different slot
    #: without rewriting the command.
    slot: str = "primary"
    #: Extra environment for the run — credentials, chiefly. Merged over `os.environ`.
    env: Mapping[str, str] | None = None
    #: Run through the daemon (the default) rather than embedded. See the class docstring for why
    #: the embedded path cannot carry a packaged track.
    standalone: bool = False

    def run(self, task: Task, *, arm: str, model: str, route: str, home: str,
            cwd: str | None = None) -> RunResult:
        home_path = Path(home)
        home_path.mkdir(parents=True, exist_ok=True)
        workspace = Path(cwd or task.cwd or (home_path / "work"))
        workspace.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)

        argv = [
            "node",
            self.entry,
            "-p",
            *(["--standalone"] if self.standalone else []),
            "--cwd",
            str(workspace),
            "--model",
            f"{self.slot}={route}/{model}",
            task.prompt,
        ]
        # Inherited, not replaced. An environment holding only PATH and AGH_HOME is missing
        # SystemRoot, TEMP and whatever else the platform needs — and a route's credential lives in
        # the environment too, so throwing it away would drop the very thing being configured.
        child_env = dict(os.environ)
        child_env.update({
            "AGH_HOME": str(home_path),
            "AGNES_PROFILE": self.profile,
        })
        if self.env:
            child_env.update(self.env)

        completed = subprocess.run(
            argv,
            cwd=str(workspace),
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"task {task.id} ({arm}) exited {completed.returncode}: {completed.stderr.strip()[:2000]}"
            )

        ledger = home_path / "data" / "sessions.db"
        # Bounded to this run. A workspace that already held a session produces no new one —
        # the harness reuses it — and reading whatever is newest would then hand back another
        # run's trajectory as if it were this one's.
        session_key = _newest_session(ledger, since=started)
        return RunResult(session_key=session_key, ledger_path=str(ledger))


def _newest_session(ledger_path: Path, *, since: datetime | None = None) -> str:
    """The session the run just created.

    Ordered by the session's own `created_at` and not by its highest sequence number. Sequence would
    be wrong wherever a fork is present: a fork's child starts at its parent's boundary, so a child
    that ran on can hold a lower last sequence than the parent it forked from, and picking the
    larger number would then name the parent.

    ``since`` bounds the search to sessions created at or after the run started, and a run that
    created none is a **failure** rather than a lookup that falls through to an older row. That
    fall-through is not hypothetical: the harness derives a session key from the working directory,
    so re-running a task in the same workspace reuses the existing session instead of creating one,
    and the caller would then be handed a previous run's trajectory — one arm's answers recorded
    under another's name, with nothing to notice it by. Asking for a session the run did not create
    is an error the caller can fix (a fresh workspace); a silent substitution is not.

    The sequence-based ordering is kept as the fallback for a ledger whose `sessions` table is empty
    — a fixture, or a ledger written before that table existed. There it cannot be wrong, because
    with no `sessions` rows there are no forks to confuse it with.
    """
    with open_ledger(ledger_path) as ledger:
        rows = ledger.sessions()
        if rows:
            if since is not None:
                fresh = [row for row in rows if not _predates(row.created_at, since)]
                if not fresh:
                    raise RuntimeError(
                        f"no session was created in {ledger_path} after {since.isoformat()}. The "
                        "harness reuses a session when the working directory already has one, so "
                        "this run's rows went into a previous session — give each run its own "
                        "workspace rather than reading a session that was not created here."
                    )
                newest = max(fresh, key=lambda row: (row.created_at, row.session_key))
            else:
                newest = max(rows, key=lambda row: (row.created_at, row.session_key))
            return newest.session_key
        keys = ledger.session_keys()
        if not keys:
            raise RuntimeError(f"the run wrote no session to {ledger_path}")
        return max(keys, key=lambda key: (ledger.last_seq(key), key))


def _predates(created_at: str, since: datetime) -> bool:
    """Whether an ISO timestamp is older than ``since``, tolerating an unparseable one.

    A timestamp that cannot be read is treated as NOT older, so an odd format errs toward
    accepting the row rather than rejecting every row and reporting a missing session.
    """
    try:
        stamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp < since


def _merge_ledger(source: Path, target: Path) -> list[str]:
    """Copy every session and event from one ledger into another, returning the new keys.

    Runs share one home, so a second task that replaced the file would erase the first
    task's session — and the rollout record already written would name a file that no
    longer held what it described. The keys added are returned so the caller names the
    run by what it actually contributed rather than by whatever is newest by timestamp.
    """
    before = set(_session_keys(target))
    con = sqlite3.connect(str(target))
    try:
        con.execute('ATTACH DATABASE ? AS incoming', (str(source),))
        for table in ('sessions', 'events'):
            columns = [row[1] for row in con.execute(f'PRAGMA incoming.table_info({table})')]
            if not columns:
                continue
            listing = ', '.join(f'"{c}"' for c in columns)
            con.execute(
                f'INSERT OR REPLACE INTO main."{table}" ({listing}) '
                f'SELECT {listing} FROM incoming."{table}"'
            )
        con.commit()
        con.execute('DETACH DATABASE incoming')
    finally:
        con.close()
    return sorted(set(_session_keys(target)) - before)


def _session_keys(ledger_path: Path) -> list[str]:
    con = sqlite3.connect(str(ledger_path))
    try:
        return [row[0] for row in con.execute('SELECT session_key FROM sessions')]
    except sqlite3.Error:
        return []
    finally:
        con.close()


@dataclass
class ReplayRunner:
    """Serve prepared ledgers instead of running a model.

    Maps a task id to a directory holding a ``sessions.db``. This is what lets the dataset and
    evaluation code be tested end to end — through real ledgers, real trajectories and real
    divergence extraction — without a model endpoint.

    The prepared ledger is **merged** into the home's own rather than written over it, because
    runs share a home; see :func:`_merge_ledger`.
    """

    sources: dict[str, str]

    def run(self, task: Task, *, arm: str, model: str, route: str, home: str,
            cwd: str | None = None) -> RunResult:
        source = self.sources.get(task.id)
        if source is None:
            raise KeyError(f"no prepared ledger for task {task.id}")
        home_path = Path(home)
        data = home_path / "data"
        data.mkdir(parents=True, exist_ok=True)
        target = data / "sessions.db"
        incoming = Path(source) / "sessions.db"
        # Which session this run *is*: the prepared ledger's own newest, decided before the
        # merge. After it, the target holds every replayed run's sessions and "newest" there
        # would depend on how the fixtures happened to stamp their clocks.
        session_key = _newest_session(incoming)
        if not target.exists():
            shutil.copyfile(incoming, target)
        else:
            _merge_ledger(incoming, target)
        return RunResult(session_key=session_key, ledger_path=str(target))


@dataclass
class CollectionOutcome:
    records: tuple[RolloutRecord, ...]
    trajectories: dict[str, Trajectory] = field(default_factory=dict)


def collect(
    tasks: Sequence[Task],
    runner: Runner,
    *,
    arm: str,
    model: str,
    route: str,
    home_root: str | Path,
    track_sha256: str,
) -> CollectionOutcome:
    """Run every task once and record what happened.

    Every run gets its own **workspace**, and they share one home. The workspace is what
    makes a run identifiable — a session key is derived from the working directory — and
    sharing the home is what lets a packaged track be installed once rather than copied
    per task. The design asks for exactly this: one shared, append-only ledger, so a
    teacher step and a student step are directly comparable.
    """
    home_root_path = Path(home_root)
    home_root_path.mkdir(parents=True, exist_ok=True)
    records: list[RolloutRecord] = []
    trajectories: dict[str, Trajectory] = {}

    for task in tasks:
        # A task that brings its own workspace keeps it. Clearing the directory here would
        # delete the very files the task is about — the fixtures it was written to work on —
        # and the run would then fail for a reason that looks like the model's fault. Only a
        # directory this function invents is this function's to clear.
        if task.cwd:
            workspace = Path(task.cwd)
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            # The name carries the start time because a session key is derived from the working
            # directory: two runs in the same directory are one session to the harness, so the
            # second would append to the first rather than produce a trajectory of its own.
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
            workspace = home_root_path / f"{arm}-{task.id}-{stamp}"
            workspace.mkdir(parents=True, exist_ok=True)
        result = runner.run(task, arm=arm, model=model, route=route,
                            home=str(home_root_path), cwd=str(workspace))

        with open_ledger(result.ledger_path) as ledger:
            trajectory = read_trajectory(ledger, result.session_key)
        key = f"{arm}/{task.id}"
        trajectories[key] = trajectory

        records.append(
            RolloutRecord(
                task_id=task.id,
                arm=arm,
                session_key=result.session_key,
                ledger_path=result.ledger_path,
                track_sha256=track_sha256,
                preset=trajectory.preset,
                route=route,
                model=model,
                outcome=Outcome.UNKNOWN.value,
                decisions=len(trajectory.points),
            )
        )

    return CollectionOutcome(records=tuple(records), trajectories=trajectories)
