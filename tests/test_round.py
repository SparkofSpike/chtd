"""A collection round, tested without a model.

The round is where two claims become facts rather than intentions: that both arms started from the
same bytes, and that the track recorded on a row is the track the session actually ran. Both are
tested here against real ledgers, through `ReplayRunner` — no endpoint, no GPU, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.round import (
    Arm,
    RoundTask,
    load_round_tasks,
    observed_track,
    run_round,
    write_outcomes,
)
from harness.rollout import ReplayRunner
from harness.tasks.build_workspaces import CHECK_SCRIPT
from tests.ledger_fixture import build_ledger

SESSION = "agnes:local:local-dev:cli:workspace:" + "c" * 16

#: A check that succeeds, on any platform. `true` would be the Unix spelling and is not a program on
#: Windows — which is not a hypothetical: an earlier version of this file used it, and the tests that
#: assert a passing run failed here for that reason rather than for the one they were testing.
PASSES = 'python -c "pass"'
FAILS = 'python -c "import sys; sys.exit(3)"'


# --------------------------------------------------------------- task sets


def _task_set(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "\n".join(
            line if isinstance(line, str) else json.dumps(line, ensure_ascii=False)
            for line in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_a_task_set_round_trips(tmp_path) -> None:
    path = _task_set(tmp_path, [
        {"id": "a", "prompt": "fix it", "check": "python check.py"},
        {"id": "b", "prompt": "and this", "check": "python check.py"},
    ])
    tasks = load_round_tasks(path)
    assert [t.id for t in tasks] == ["a", "b"]
    assert tasks[0].check == "python check.py"


def test_committed_holdout_checks_use_this_checkout() -> None:
    """The committed holdout set: portably addressed, disjoint from training, and large enough to
    resolve a difference.

    The size assertion is deliberately a floor rather than an exact count. It was `== 5` until
    2026-10-11, and five is the number this test was written around: with five tasks one task is
    twenty points of the score, and the base model's own run-to-run range measured wider than any
    gap between arms. A set that cannot see an effect cannot support a claim, so the floor records
    the reason rather than the current number — adding tasks should not require editing a test, and
    shrinking the set back should fail.
    """
    task_set = Path(__file__).resolve().parent.parent / "harness" / "tasks" / "final-holdout.jsonl"
    tasks = load_round_tasks(task_set)
    assert len(tasks) >= 15, (
        f"the held-out set holds {len(tasks)} tasks; below 15 one task is worth more than 6 points "
        "and the measurement cannot separate two arms"
    )
    assert CHECK_SCRIPT.is_file()
    assert all(str(CHECK_SCRIPT) in task.check for task in tasks)

    training = load_round_tasks(
        Path(__file__).resolve().parent.parent / "harness" / "tasks" / "error-recovery.jsonl"
    )
    overlap = {task.id for task in tasks} & {task.id for task in training}
    assert not overlap, f"a task is in both sets: {sorted(overlap)}"


def test_comments_and_blank_lines_are_skipped(tmp_path) -> None:
    path = _task_set(tmp_path, [
        "# a note to a reader",
        "",
        {"id": "a", "prompt": "fix it", "check": "python check.py"},
    ])
    assert [t.id for t in load_round_tasks(path)] == ["a"]


def test_a_repeated_id_is_refused(tmp_path) -> None:
    """Two runs sharing one output directory: the later silently replaces the earlier, which looks
    exactly like a task that never ran."""
    path = _task_set(tmp_path, [
        {"id": "a", "prompt": "one", "check": "true"},
        {"id": "a", "prompt": "two", "check": "true"},
    ])
    with pytest.raises(ValueError, match="repeats the id"):
        load_round_tasks(path)


def test_a_task_without_a_check_is_refused(tmp_path) -> None:
    """Without one there is no outcome, and a dataset with nothing to gate on is not a dataset."""
    path = _task_set(tmp_path, [{"id": "a", "prompt": "fix it"}])
    with pytest.raises(ValueError, match="check"):
        load_round_tasks(path)


def test_the_workspace_is_resolved_from_the_id(tmp_path) -> None:
    """No absolute path in the task set: the fixtures are generated into a gitignored directory, and
    a recorded path would be one machine's.

    Asserted against the shape the generator actually produces — a real file at
    ``<root>/harness/tasks/<name>.jsonl`` resolving to ``<root>/data/tasks/<id>`` — because a test
    that only checked the tail of the string passed while the resolution was off by a directory.
    """
    root = tmp_path / "repo"
    (root / "harness" / "tasks").mkdir(parents=True)
    (root / "data" / "tasks" / "fix-assertion").mkdir(parents=True)
    path = root / "harness" / "tasks" / "error-recovery.jsonl"
    path.write_text(
        json.dumps({"id": "fix-assertion", "prompt": "p", "check": PASSES}) + "\n",
        encoding="utf-8",
    )

    task = load_round_tasks(path)[0]
    assert Path(task.fixture) == root / "data" / "tasks" / "fix-assertion"
    assert Path(task.fixture).is_dir(), "the resolved path must be the one the generator writes"


# --------------------------------------------------------- the observed track


def _ledger_with_start(tmp_path: Path, key: str, preset: str | None, profile: str | None,
                       *, suffix: str = "") -> Path:
    path = tmp_path / f"{key[-8:]}{suffix}.db"
    builder = build_ledger(path)
    builder.session(key).start(key, preset=preset, resolved_profile_hash=profile)
    builder.close()
    return path


def test_the_track_is_read_from_the_session(tmp_path) -> None:
    key = "agnes:local:local-dev:cli:workspace:" + "d" * 16
    path = _ledger_with_start(tmp_path, key, "standard-windows", "sha256-" + "a" * 64)
    digest, preset = observed_track(path, key)
    assert preset == "standard-windows"
    assert len(digest) == 64


def test_two_sessions_on_one_track_produce_one_digest(tmp_path) -> None:
    """The property the dataset's refusal to mix tracks rests on."""
    profile = "sha256-" + "a" * 64
    first = _ledger_with_start(tmp_path, "agnes:local:local-dev:cli:workspace:" + "1" * 16,
                               "standard-windows", profile)
    second = _ledger_with_start(tmp_path, "agnes:local:local-dev:cli:workspace:" + "2" * 16,
                                "standard-windows", profile)
    assert observed_track(first, "agnes:local:local-dev:cli:workspace:" + "1" * 16)[0] == \
        observed_track(second, "agnes:local:local-dev:cli:workspace:" + "2" * 16)[0]


def test_two_different_tracks_produce_two_digests(tmp_path) -> None:
    """And therefore that a round accidentally spanning two tracks is caught downstream."""
    profile = "sha256-" + "a" * 64
    first = _ledger_with_start(tmp_path, "agnes:local:local-dev:cli:workspace:" + "1" * 16,
                               "standard-windows", profile)
    second = _ledger_with_start(tmp_path, "agnes:local:local-dev:cli:workspace:" + "2" * 16,
                                "chtd", profile)
    assert observed_track(first, "agnes:local:local-dev:cli:workspace:" + "1" * 16)[0] != \
        observed_track(second, "agnes:local:local-dev:cli:workspace:" + "2" * 16)[0]


def test_a_different_profile_is_a_different_track(tmp_path) -> None:
    """One track is a preset *and* the profile it resolved against."""
    key = "agnes:local:local-dev:cli:workspace:" + "3" * 16
    one = _ledger_with_start(tmp_path, key, "standard-windows", "sha256-" + "a" * 64, suffix="-1")
    two = _ledger_with_start(tmp_path, key, "standard-windows", "sha256-" + "b" * 64, suffix="-2")
    assert observed_track(one, key)[0] != observed_track(two, key)[0]


def test_a_session_that_never_started_yields_nothing(tmp_path) -> None:
    """An empty digest rather than a fabricated one: the caller can see it did not get one."""
    path = tmp_path / "empty.db"
    builder = build_ledger(path)
    builder.session("some-other-key")
    builder.close()
    digest, preset = observed_track(path, "some-other-key")
    assert digest == ""
    assert preset is None


# ------------------------------------------------------------------ the round


def _fixture_workspace(tmp_path: Path, name: str, check_command: str) -> Path:
    workspace = tmp_path / "fixtures" / name
    workspace.mkdir(parents=True)
    (workspace / "value.txt").write_text("broken\n", encoding="utf-8")
    return workspace


def test_both_arms_start_from_the_same_bytes(tmp_path) -> None:
    """If the teacher ran first in the live directory, the student would inherit its repairs and the
    comparison would measure the order the runs happened in."""
    fixture = _fixture_workspace(tmp_path, "t1", PASSES)
    source = tmp_path / "prepared"
    source.mkdir()
    ledger = _ledger_with_start(source, SESSION, "standard-windows", "sha256-" + "a" * 64)
    (source / "sessions.db").write_bytes(ledger.read_bytes())

    runner = ReplayRunner(sources={"t1": str(source)})
    outcomes = run_round(
        tasks=[RoundTask(id="t1", prompt="p", fixture=str(fixture), check=PASSES)],
        arms=[Arm(name="student", route="s", model="m1"),
              Arm(name="teacher", route="t", model="m2")],
        runner=runner,
        home_root=tmp_path / "home",
        out_dir=tmp_path / "round",
        log=lambda *_: None,
    )

    assert len(outcomes) == 2
    student = Path(outcomes[0].workspace)
    teacher = Path(outcomes[1].workspace)
    assert student != teacher, "each run needs its own workspace, or one session key covers both"
    assert (student / "value.txt").read_text(encoding="utf-8") == "broken\n"
    assert (teacher / "value.txt").read_text(encoding="utf-8") == "broken\n"


def test_the_recorded_track_is_the_observed_one(tmp_path) -> None:
    """Not the value the caller passed. `run_round` takes no track argument at all now: a caller
    who could set it could set it wrong, and a wrong digest looks perfectly well-formed."""
    fixture = _fixture_workspace(tmp_path, "t1", "true")
    source = tmp_path / "prepared"
    source.mkdir()
    ledger = _ledger_with_start(source, SESSION, "standard-windows", "sha256-" + "a" * 64)
    (source / "sessions.db").write_bytes(ledger.read_bytes())

    runner = ReplayRunner(sources={"t1": str(source)})
    outcomes = run_round(
        tasks=[RoundTask(id="t1", prompt="p", fixture=str(fixture), check=PASSES)],
        arms=[Arm(name="student", route="s", model="m")],
        runner=runner,
        home_root=tmp_path / "home",
        out_dir=tmp_path / "round",
        log=lambda *_: None,
    )

    record = outcomes[0].record
    assert record is not None
    assert record.track_sha256 == observed_track(record.ledger_path, record.session_key)[0]
    assert record.track_sha256 != "pending"
    assert record.preset == "standard-windows"


def test_an_unexpected_track_is_refused(tmp_path) -> None:
    """`expected_track` is a check, not a source: a run that landed elsewhere is an error rather
    than a row, because a row would be gated and trained on as though nothing had happened."""
    fixture = _fixture_workspace(tmp_path, "t1", "true")
    source = tmp_path / "prepared"
    source.mkdir()
    ledger = _ledger_with_start(source, SESSION, "standard-windows", "sha256-" + "a" * 64)
    (source / "sessions.db").write_bytes(ledger.read_bytes())

    runner = ReplayRunner(sources={"t1": str(source)})
    outcomes = run_round(
        tasks=[RoundTask(id="t1", prompt="p", fixture=str(fixture), check=PASSES)],
        arms=[Arm(name="student", route="s", model="m")],
        runner=runner,
        home_root=tmp_path / "home",
        out_dir=tmp_path / "round",
        expected_track="f" * 64,
        log=lambda *_: None,
    )

    assert outcomes[0].record is None
    assert outcomes[0].check_passed is False
    assert "track mismatch" in outcomes[0].check_output


def test_the_outcome_comes_from_the_check_not_from_the_run(tmp_path) -> None:
    """The task's own check runs in the run's workspace afterwards. A run that reported success
    while leaving the task broken is recorded as broken."""
    fixture = _fixture_workspace(tmp_path, "t1", "true")
    source = tmp_path / "prepared"
    source.mkdir()
    ledger = _ledger_with_start(source, SESSION, "standard-windows", "sha256-" + "a" * 64)
    (source / "sessions.db").write_bytes(ledger.read_bytes())

    runner = ReplayRunner(sources={"t1": str(source)})
    outcomes = run_round(
        tasks=[RoundTask(id="t1", prompt="p", fixture=str(fixture), check=FAILS)],
        arms=[Arm(name="student", route="s", model="m")],
        runner=runner,
        home_root=tmp_path / "home",
        out_dir=tmp_path / "round",
        log=lambda *_: None,
    )
    assert outcomes[0].check_passed is False


def test_a_failed_run_is_a_row_not_a_stop(tmp_path) -> None:
    """One endpoint refusing must not lose the other tasks' data."""
    runner = ReplayRunner(sources={})  # raises KeyError for every task
    outcomes = run_round(
        tasks=[RoundTask(id="t1", prompt="p", fixture=str(_fixture_workspace(tmp_path, "t1", PASSES)),
                         check=PASSES)],
        arms=[Arm(name="student", route="s", model="m")],
        runner=runner,
        home_root=tmp_path / "home",
        out_dir=tmp_path / "round",
        log=lambda *_: None,
    )
    assert len(outcomes) == 1
    assert outcomes[0].record is None
    assert "run failed" in outcomes[0].check_output


def test_outcomes_are_written_in_three_files(tmp_path) -> None:
    """The pipeline's records, the gate's outcome rows, and the checks' own output — apart, because
    only the first two are read by a machine."""
    fixture = _fixture_workspace(tmp_path, "t1", PASSES)
    source = tmp_path / "prepared"
    source.mkdir()
    ledger = _ledger_with_start(source, SESSION, "standard-windows", "sha256-" + "a" * 64)
    (source / "sessions.db").write_bytes(ledger.read_bytes())

    runner = ReplayRunner(sources={"t1": str(source)})
    outcomes = run_round(
        tasks=[RoundTask(id="t1", prompt="p", fixture=str(fixture), check=PASSES)],
        arms=[Arm(name="student", route="s", model="m")],
        runner=runner,
        home_root=tmp_path / "home",
        out_dir=tmp_path / "round",
        log=lambda *_: None,
    )
    records, outcome_rows = write_outcomes(
        outcomes,
        records_path=tmp_path / "out" / "records.jsonl",
        outcomes_path=tmp_path / "out" / "outcomes.jsonl",
    )

    assert len(records.read_text(encoding="utf-8").strip().splitlines()) == 1
    row = json.loads(outcome_rows.read_text(encoding="utf-8").strip())
    assert row["passed"] is True
    assert row["arm"] == "student"
    checks = (tmp_path / "out" / "checks.jsonl").read_text(encoding="utf-8").strip()
    assert "workspace" in json.loads(checks)
