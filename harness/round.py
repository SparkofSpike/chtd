"""A collection round: every task, both arms, each from an identical initial workspace.

Two things make a round comparable rather than merely long.

**Both arms start from the same bytes.** Each run gets its own copy of the task's fixture, and the
copies are made from the same source before either run touches anything. If the teacher ran first in
the live directory, the student would inherit the teacher's repairs and the comparison would measure
the order the runs happened in.

**Every run's outcome is observed, not asserted.** After a run the task's own check is executed in
that run's workspace. The exit code is the outcome the signal gate validates against — which is
what lets a divergence be admitted or refused at all. A round that recorded trajectories without
outcomes would produce a dataset with nothing to gate on.

What this module does not do is decide which models run. That arrives as :class:`Arm` values, so no
model id or endpoint is a constant here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from harness.ledger import open_ledger
from harness.rollout import RolloutRecord, Runner, Task, collect

__all__ = [
    "Arm",
    "RoundTask",
    "RunOutcome",
    "load_round_tasks",
    "observed_track",
    "run_round",
    "write_outcomes",
]


@dataclass(frozen=True)
class RoundTask:
    """One task: what to ask, what state to start from, and how success is observed."""

    id: str
    prompt: str
    fixture: str
    check: str


@dataclass(frozen=True)
class Arm:
    """One side of the comparison: a name, a route, and the model that route runs."""

    name: str
    route: str
    model: str


@dataclass
class RunOutcome:
    task_id: str
    arm: str
    workspace: str
    session_key: str
    ledger_path: str
    check_passed: bool
    check_output: str
    record: RolloutRecord | None = None


def load_round_tasks(path: str | Path) -> tuple[RoundTask, ...]:
    """Read a task set: one JSON object per line with ``id``, ``prompt`` and ``check``."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no task set at {source}")
    tasks: list[RoundTask] = []
    seen: set[str] = set()
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{source}:{number} has no id")
        if task_id in seen:
            # A duplicate would let two runs share one output directory, and the later would
            # silently replace the earlier — which looks exactly like a task that never ran.
            raise ValueError(f"{source}:{number} repeats the id {task_id!r}")
        seen.add(task_id)
        for field_name in ("prompt", "check"):
            if not isinstance(row.get(field_name), str) or not row[field_name]:
                raise ValueError(f"{source}:{number} ({task_id}) has no {field_name}")
        # The workspace is resolved from the id: the task set records no absolute path, because the
        # fixtures are generated into a gitignored directory and a recorded path would be one
        # machine's. Three levels up from `harness/tasks/<file>.jsonl` is the repository root;
        # `harness/tasks/build_workspaces.py` produces exactly this layout.
        fixture = row.get("cwd") or str(
            source.resolve().parent.parent.parent / "data" / "tasks" / task_id
        )
        tasks.append(RoundTask(id=task_id, prompt=row["prompt"], fixture=fixture, check=row["check"]))
    if not tasks:
        raise ValueError(f"{source} holds no tasks")
    return tuple(tasks)


def _run_check(
    workspace: Path,
    check: str,
    *,
    timeout: int = 300,
    env: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Run a task's own check in a run's workspace. The exit code is the outcome.

    ``env`` is merged over the ambient environment. It exists because a task's result is not always
    something in the workspace: a read-only task produces an answer, and the answer lives in the
    ledger. A check that has to read one is given the session key and the ledger path rather than
    guessing them from the directory, since the directory is what does not identify a run.
    """
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    try:
        completed = subprocess.run(
            check,
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return False, f"check timed out after {timeout}s"
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


def _prepare(round_root: Path, task: RoundTask, arm: Arm) -> Path:
    """A fresh copy of the fixture, made before either arm has touched anything."""
    source = Path(task.fixture)
    if not source.is_dir():
        raise FileNotFoundError(f"task {task.id} has no fixture at {source}")
    workspace = round_root / task.id / arm.name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, workspace)
    return workspace


def observed_track(ledger_path: str | Path, session_key: str) -> tuple[str, str | None]:
    """The track a session actually ran, read out of the session's own record.

    Why this rather than a value the caller supplies. A track digest passed in by the caller records
    the caller's *intention*; the comparison the dataset rests on is between what two runs actually
    did. Those diverge exactly when something is wrong — a profile pointing at a different preset, a
    run that resolved to a different recipe — and a supplied digest is blind to that while looking
    perfectly well-formed.

    So the digest is derived from the fields the harness writes at session start: the preset that
    resolved, and the profile hash it resolved against. Two sessions on one track produce one digest;
    two sessions on different tracks produce two, and `distill.dataset` then refuses to mix them,
    which is the behaviour that makes "the same track" a fact rather than a claim.

    The harness does not record a preset byte hash in this version, so this is the strongest
    identifier available from the ledger. It is weaker than a byte hash by exactly one thing: two
    different builds of a builtin preset under the same name and profile would hash alike. That is
    stated here rather than left for a reader to discover.
    """
    with open_ledger(ledger_path) as ledger:
        for event in ledger.events(session_key):
            if event.type != "session/start":
                continue
            data = event.data if isinstance(event.data, dict) else {}
            preset = data.get("preset") if isinstance(data.get("preset"), str) else None
            profile = (
                data.get("resolvedProfileHash")
                if isinstance(data.get("resolvedProfileHash"), str)
                else None
            )
            digest = hashlib.sha256(
                f"chtd/track/v1\npreset:{preset}\nprofile:{profile}\n".encode("utf-8")
            ).hexdigest()
            return digest, preset
    return "", None


def run_round(
    *,
    tasks: Sequence[RoundTask],
    arms: Sequence[Arm],
    runner: Runner,
    home_root: str | Path,
    out_dir: str | Path,
    throttle_seconds: float = 0.0,
    expected_track: str | None = None,
    log=print,
) -> tuple[RunOutcome, ...]:
    """Run every task under every arm, then observe what each run left behind.

    ``throttle_seconds`` is a pause between runs. It exists because a hosted endpoint can rate-limit
    a burst, and a round that is refused halfway is worse than a round that is slow: the refusals
    look like task failures.

    ``expected_track`` is optional and is a *check*, not a source: when given, a run whose observed
    track digest differs from it is an error rather than a row. The digest written to the record is
    always the observed one — see :func:`observed_track`.
    """
    import time  # noqa: PLC0415 - only this function sleeps

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    outcomes: list[RunOutcome] = []

    for task in tasks:
        for arm in arms:
            workspace = _prepare(root, task, arm)
            log(f"{task.id}  [{arm.name}]  {arm.route}/{arm.model}")
            started = time.time()
            try:
                collected = collect(
                    [Task(id=task.id, prompt=task.prompt, cwd=str(workspace))],
                    runner,
                    arm=arm.name,
                    model=arm.model,
                    route=arm.route,
                    home_root=home_root,
                    track_sha256="pending",
                )
            except Exception as error:  # noqa: BLE001 - a failed run is a datum, not a stop
                log(f"    RUN FAILED: {type(error).__name__}: {str(error)[:200]}")
                outcomes.append(
                    RunOutcome(
                        task_id=task.id,
                        arm=arm.name,
                        workspace=str(workspace),
                        session_key="",
                        ledger_path="",
                        check_passed=False,
                        check_output=f"run failed: {type(error).__name__}",
                    )
                )
                if throttle_seconds:
                    time.sleep(throttle_seconds)
                continue

            record = collected.records[0]
            digest, preset = observed_track(record.ledger_path, record.session_key)
            if expected_track and digest != expected_track:
                log(f"    TRACK MISMATCH: ran {preset!r} ({digest[:12]}), expected {expected_track[:12]}")
                outcomes.append(
                    RunOutcome(
                        task_id=task.id,
                        arm=arm.name,
                        workspace=str(workspace),
                        session_key=record.session_key,
                        ledger_path=record.ledger_path,
                        check_passed=False,
                        check_output=f"track mismatch: ran {preset!r}",
                    )
                )
                if throttle_seconds:
                    time.sleep(throttle_seconds)
                continue
            record = replace(record, track_sha256=digest, preset=preset)

            passed, output = _run_check(
                workspace,
                task.check,
                env={
                    "CHTD_TASK": task.id,
                    "CHTD_ARM": arm.name,
                    "CHTD_SESSION_KEY": record.session_key,
                    "CHTD_LEDGER": record.ledger_path,
                    "CHTD_WORKSPACE": str(workspace),
                },
            )
            log(f"    session {record.session_key[-18:]}  preset={preset}  "
                f"check={'pass' if passed else 'FAIL'}  {time.time() - started:.0f}s")
            outcomes.append(
                RunOutcome(
                    task_id=task.id,
                    arm=arm.name,
                    workspace=str(workspace),
                    session_key=record.session_key,
                    ledger_path=record.ledger_path,
                    check_passed=passed,
                    check_output=output,
                    record=record,
                )
            )
            if throttle_seconds:
                time.sleep(throttle_seconds)

    return tuple(outcomes)


def write_outcomes(
    outcomes: Sequence[RunOutcome],
    *,
    records_path: str | Path,
    outcomes_path: str | Path,
) -> tuple[Path, Path]:
    """Write ``records.jsonl`` for the pipeline and ``outcomes.jsonl`` for the gate.

    The outcome rows carry the session key and the prefix digest the divergence is paired on, so a
    refusal can be traced to the run it came from. The check's own output is kept beside them in
    ``checks.jsonl`` — not in the outcomes file, because that file is read by a machine.
    """
    records_target = Path(records_path)
    records_target.parent.mkdir(parents=True, exist_ok=True)
    with records_target.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            if outcome.record is not None:
                handle.write(outcome.record.to_json() + "\n")

    outcomes_target = Path(outcomes_path)
    outcomes_target.parent.mkdir(parents=True, exist_ok=True)
    with outcomes_target.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            if not outcome.session_key:
                continue
            handle.write(
                json.dumps(
                    {
                        "session_key": outcome.session_key,
                        "task_id": outcome.task_id,
                        "arm": outcome.arm,
                        "passed": outcome.check_passed,
                    }
                )
                + "\n"
            )

    checks_target = outcomes_target.with_name("checks.jsonl")
    with checks_target.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(
                json.dumps(
                    {
                        "task_id": outcome.task_id,
                        "arm": outcome.arm,
                        "passed": outcome.check_passed,
                        "output": outcome.check_output[:2000],
                        "workspace": outcome.workspace,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return records_target, outcomes_target


def main(argv: Sequence[str] | None = None) -> int:
    """Print what a round would do, without a model. The real driver supplies the credentials."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Inspect a task set.")
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args(argv)

    tasks = load_round_tasks(args.tasks)
    print(f"{len(tasks)} tasks")
    for task in tasks:
        fixture = Path(task.fixture)
        files = sorted(p.name for p in fixture.rglob("*") if p.is_file()) if fixture.is_dir() else []
        print(f"  {task.id:26} fixture={'ok' if fixture.is_dir() else 'MISSING'} "
              f"files={len(files)}  check={task.check}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main(sys.argv[1:]))
