"""The task set: its fixtures are really broken, its repairs really work, and its checks judge a
diagnosis rather than a changed directory.

The first two are what `build_workspaces.self_check` establishes and what these tests re-establish
against the committed task set; the third is what `check_diagnosis` decides, and it is tested here
because a check that always passes would make every run look like a success.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.tasks.build_workspaces import EXPECT, TASKS, check_command, self_check
from harness.tasks.check_diagnosis import answer_text, main
from harness.ledger import open_ledger
from tests.ledger_fixture import build_ledger

SESSION = "agnes:local:local-dev:cli:workspace:" + "e" * 16


def test_every_task_has_expected_phrases() -> None:
    """A task with no phrases is a task a round could not judge."""
    for task in TASKS:
        assert task["id"] in EXPECT, task["id"]
        assert EXPECT[task["id"]], task["id"]


def test_every_task_has_a_verify_command() -> None:
    for task in TASKS:
        assert task["check"], task["id"]


def test_the_expected_phrases_are_not_in_the_prompt() -> None:
    """The prompt must not name what the check looks for, or the task becomes a transcription."""
    from harness.tasks.build_workspaces import BUG_PROMPT

    for phrases in EXPECT.values():
        for phrase in phrases:
            assert phrase.lower() not in BUG_PROMPT.lower(), phrase


def test_the_check_command_names_the_phrases() -> None:
    command = check_command("fix-assertion")
    assert "check_diagnosis.py" in command
    for phrase in EXPECT["fix-assertion"]:
        assert phrase in command


def test_the_fixtures_are_broken_and_repairable() -> None:
    """The set's own self-check, run as a test.

    Twelve processes, so it is slower than the rest of the suite — and it is the one check that
    catches a fixture which passes before the fix, which is a task that measures nothing while
    looking exactly like one the model solved.
    """
    assert self_check(verbose=False) == 0


# ------------------------------------------------------------- the diagnosis check


def _ledger_with_answer(tmp_path: Path, text: str) -> tuple[str, str]:
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "what is wrong?")
    if text:
        builder.assistant(SESSION, text)
    builder.close()
    return str(path), SESSION


def test_a_correct_diagnosis_passes(tmp_path, monkeypatch) -> None:
    ledger, session = _ledger_with_answer(
        tmp_path, "The assert in test_average expects 3 but mean([1, 2, 3]) is 2."
    )
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "assert", "mean"])
    assert main() == 0


def test_a_run_that_did_not_find_it_fails(tmp_path, monkeypatch) -> None:
    """The property that makes the check an outcome rather than decoration."""
    ledger, session = _ledger_with_answer(tmp_path, "Hello! How can I help you today?")
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "assert", "mean"])
    assert main() == 1


def test_a_partial_diagnosis_fails(tmp_path, monkeypatch) -> None:
    """Every phrase is required. One of several is a run that noticed some of it."""
    ledger, session = _ledger_with_answer(tmp_path, "The assert is wrong.")
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "assert", "mean"])
    assert main() == 1


def test_matching_is_case_insensitive(tmp_path, monkeypatch) -> None:
    ledger, session = _ledger_with_answer(tmp_path, "ASSERT the MEAN is off")
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "assert", "mean"])
    assert main() == 0


def test_a_silent_session_fails(tmp_path, monkeypatch) -> None:
    ledger, session = _ledger_with_answer(tmp_path, "")
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "assert"])
    assert main() == 1


def test_no_phrases_is_a_usage_error(tmp_path, monkeypatch) -> None:
    ledger, session = _ledger_with_answer(tmp_path, "something")
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.delenv("CHTD_EXPECT", raising=False)
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py"])
    assert main() == 2


def test_missing_session_is_a_usage_error(monkeypatch) -> None:
    monkeypatch.delenv("CHTD_LEDGER", raising=False)
    monkeypatch.delenv("CHTD_SESSION_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "assert"])
    assert main() == 2


def test_every_assistant_message_counts(tmp_path) -> None:
    """Not just the last: a run that states the diagnosis and restates it should not be judged on
    the order it happened to use."""
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "what is wrong?")
    builder.assistant(SESSION, "The mean is computed correctly but the assert is wrong.")
    builder.assistant(SESSION, "In short: fix the expected value.")
    builder.close()
    text = answer_text(str(path), SESSION)
    assert "assert" in text.lower()
    assert "expected value" in text.lower()
