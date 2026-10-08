"""Drive harness sessions and collect their trajectories.

One task set, one session per task, one ledger row per run. The module's job is to make a rollout
*recorded* rather than remembered: every run writes what track it ran, on which model, and what it
decided, so that the dataset step can refuse to compare runs that were not comparable and the
write-up can reconstruct which run produced which row.

**The model is a seam.** `Runner` is the interface; `CliRunner` drives a built harness through its
own CLI, and `ReplayRunner` serves a prepared ledger so that everything downstream of the driver can
be tested without a model account, a GPU, or a network. A pipeline that could only be exercised
against a live endpoint would be a pipeline nobody could check.

**Untested path, stated plainly.** `CliRunner` has not been run: building the harness on this
machine fails in its own native step (`packages/system-node/scripts/build-native.mjs` wants Node
headers that are not installed), so the command it builds is written from the CLI's documented
arguments and verified only by construction. Everything it feeds — ledger reading, trajectory
extraction, divergence, gating — is tested against real ledgers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
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

    def run(self, task: Task, *, arm: str, model: str, route: str, home: str) -> RunResult: ...


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

    The command is ``agnes -p --model main=<route>/<model> <prompt>`` against an isolated
    ``AGH_HOME``, with the profile named by ``AGNES_PROFILE``. Every part of that is the documented
    interface; none of it is exercised here (see the module docstring).
    """

    entry: str
    profile: str = "chtd"
    timeout_seconds: int = 900

    def run(self, task: Task, *, arm: str, model: str, route: str, home: str) -> RunResult:
        home_path = Path(home)
        home_path.mkdir(parents=True, exist_ok=True)
        cwd = Path(task.cwd) if task.cwd else home_path / "work"
        cwd.mkdir(parents=True, exist_ok=True)

        completed = subprocess.run(
            [
                "node",
                self.entry,
                "-p",
                "--model",
                f"main={route}/{model}",
                task.prompt,
            ],
            cwd=str(cwd),
            env={
                "PATH": _path_of_environment(),
                "AGH_HOME": str(home_path),
                "AGNES_PROFILE": self.profile,
            },
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"task {task.id} ({arm}) exited {completed.returncode}: {completed.stderr.strip()[:2000]}"
            )

        ledger = home_path / "data" / "sessions.db"
        session_key = _newest_session(ledger)
        return RunResult(session_key=session_key, ledger_path=str(ledger))


def _path_of_environment() -> str:
    import os

    return os.environ.get("PATH", "")


def _newest_session(ledger_path: Path) -> str:
    """The session the run just created: the one with the highest last sequence number.

    Found rather than parsed out of the CLI's output, because the ledger is the record that matters
    and stdout is a convenience that a future version may change shape.
    """
    with open_ledger(ledger_path) as ledger:
        keys = ledger.session_keys()
        if not keys:
            raise RuntimeError(f"the run wrote no session to {ledger_path}")
        return max(keys, key=lambda key: (ledger.last_seq(key), key))


@dataclass
class ReplayRunner:
    """Serve prepared ledgers instead of running a model.

    Maps a task id to a directory holding a ``sessions.db``. This is what lets the dataset and
    evaluation code be tested end to end — through real ledgers, real trajectories and real
    divergence extraction — without a model endpoint.
    """

    sources: dict[str, str]

    def run(self, task: Task, *, arm: str, model: str, route: str, home: str) -> RunResult:
        source = self.sources.get(task.id)
        if source is None:
            raise KeyError(f"no prepared ledger for task {task.id}")
        home_path = Path(home)
        data = home_path / "data"
        data.mkdir(parents=True, exist_ok=True)
        target = data / "sessions.db"
        shutil.copyfile(Path(source) / "sessions.db", target)
        session_key = _newest_session(target)
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

    Every run gets its own home. Sharing one would put two sessions in one ledger, and the ledger is
    how this pipeline tells runs apart.
    """
    home_root_path = Path(home_root)
    records: list[RolloutRecord] = []
    trajectories: dict[str, Trajectory] = {}

    for task in tasks:
        home = home_root_path / f"{arm}-{task.id}"
        if home.exists():
            shutil.rmtree(home)
        result = runner.run(task, arm=arm, model=model, route=route, home=str(home))

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
