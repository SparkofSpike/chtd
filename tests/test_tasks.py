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
    """The prompt must not name what the check looks for, or the task becomes a transcription.

    Every alternative is checked, not just the phrase as written: `copy|in.?place` is two ways to
    say the same thing, and either one appearing in the prompt makes the task a transcription. The
    `.` wildcard is dropped before comparing, so `in.?place` is tested as both `inplace` and
    `in.place` — a spelling the prompt could contain even when the literal phrase does not.
    """
    from harness.tasks.build_workspaces import BUG_PROMPT

    prompt = BUG_PROMPT.lower()
    for task_id, phrases in EXPECT.items():
        for phrase in phrases:
            for alternative in phrase.split("|"):
                word = alternative.strip().lower()
                if not word:
                    continue
                assert word not in prompt, f"{task_id}: {word!r} is in the prompt"
                collapsed = word.replace(".", "")
                assert collapsed not in prompt, f"{task_id}: {collapsed!r} is in the prompt"


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


# ------------------------------------- the refusals a phrase list cannot express
#
# A phrase list asks whether the answer says the right words. It cannot ask whether the answer says
# anything untrue, and the 2026-10-10 control sweep is why that matters: four of five answers that
# invented files absent from the fixture matched the expected phrases. Both refusals below were
# added for that, and both are tested here so they cannot quietly stop working.


def test_a_phrase_list_alone_cannot_reject_an_invented_file(tmp_path, monkeypatch) -> None:
    """The gap, stated as a test: without a workspace the answer passes on its keywords.

    This is the behaviour that produced the wrong score, kept as a test so the fix below is a
    measured difference rather than a claim.
    """
    ledger, session = _ledger_with_answer(
        tmp_path,
        "the defect is a mismatch between the contract and the `src/contract.md` file, which is "
        "missing; the parse function in names.py is otherwise fine.",
    )
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.delenv("CHTD_WORKSPACE", raising=False)
    monkeypatch.delenv("CHTD_TASK_DIR", raising=False)
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "names.py", "parse"])
    assert main() == 0


def test_a_cited_file_the_fixture_does_not_hold_is_refused(tmp_path, monkeypatch) -> None:
    """The same answer, judged with the workspace known: named every phrase, described a directory
    it never read."""
    ledger, session = _ledger_with_answer(
        tmp_path,
        "the defect is a mismatch between the contract and the `src/contract.md` file, which is "
        "missing; the parse function in names.py is otherwise fine.",
    )
    fixture = tmp_path / "workspace"
    fixture.mkdir()
    (fixture / "names.py").write_text("def parse(x):\n    return x\n", encoding="utf-8")
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.setenv("CHTD_WORKSPACE", str(fixture))
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "names.py", "parse"])
    assert main() == 1


def test_an_answer_about_files_that_are_there_is_not_refused(tmp_path, monkeypatch) -> None:
    """The other direction, and the one that decides whether the check is usable: naming real files
    must stay a pass."""
    ledger, session = _ledger_with_answer(
        tmp_path, "names.py parses the wrong way; check.py asserts the right thing."
    )
    fixture = tmp_path / "workspace"
    fixture.mkdir()
    (fixture / "names.py").write_text("def parse(x):\n    return x\n", encoding="utf-8")
    (fixture / "check.py").write_text("from names import parse\n", encoding="utf-8")
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.setenv("CHTD_WORKSPACE", str(fixture))
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "names.py", "parse"])
    assert main() == 0


def test_ordinary_prose_is_not_mistaken_for_a_file_reference() -> None:
    """A check that cries wolf on `sorted()` or `==` is a check nobody keeps."""
    from harness.tasks.check_diagnosis import invented_files

    prose = (
        "`sorted()` is called with a key function; values are compared with `==` and the "
        "exception is an `IndexError`. the module is imported by `check.py`."
    )
    cited = invented_files(prose, {"check.py"})
    assert cited == [], cited


def test_a_forbidden_phrase_is_refused_with_every_expected_phrase_present(
    tmp_path, monkeypatch
) -> None:
    """A task can say what a confused answer reaches for, and that refusal is independent of the
    expected phrases."""
    ledger, session = _ledger_with_answer(
        tmp_path, "it is a 1-based index, so the empty string raises IndexError"
    )
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.setenv("CHTD_FORBID", json.dumps(["1-based"]))
    monkeypatch.setattr("sys.argv", ["check_diagnosis.py", "index", "empty"])
    assert main() == 1


def test_the_forbidden_list_can_arrive_on_the_command_line(tmp_path, monkeypatch) -> None:
    """A task's stored `check` string is its only channel, so the forbidden list has to fit there."""
    ledger, session = _ledger_with_answer(tmp_path, "the index is off and the input is empty")
    monkeypatch.setenv("CHTD_LEDGER", ledger)
    monkeypatch.setenv("CHTD_SESSION_KEY", session)
    monkeypatch.delenv("CHTD_FORBID", raising=False)
    monkeypatch.delenv("CHTD_WORKSPACE", raising=False)
    monkeypatch.setattr(
        "sys.argv", ["check_diagnosis.py", "index", "empty", "--forbid", "1-based", "off-by-one"]
    )
    assert main() == 0

    monkeypatch.setattr(
        "sys.argv",
        ["check_diagnosis.py", "index", "empty", "--forbid", "empty"],
    )
    assert main() == 1


def test_the_check_command_carries_the_forbidden_phrases() -> None:
    """`check_command` is what a round actually runs, so a forbid list that never reaches it is a
    refusal that never happens."""
    from harness.tasks.build_workspaces import FORBID, check_command as build_command

    task_id = "fix-ordinal-index"
    assert FORBID.get(task_id), "this task is expected to carry a forbidden list"
    command = build_command(task_id)
    assert "--forbid" in command
    for phrase in FORBID[task_id]:
        assert phrase in command


def test_a_task_without_forbidden_phrases_has_an_unchanged_command() -> None:
    """No forbid list means no flag: an existing command keeps its exact shape."""
    from harness.tasks.build_workspaces import FORBID, check_command as build_command

    task_id = "fix-assertion"
    assert task_id not in FORBID
    assert "--forbid" not in build_command(task_id)


# ------------------------------------------------------------- phrase collisions


def test_no_task_s_phrases_are_satisfied_by_another_task_s_answer() -> None:
    """No task's phrase set may be a subset of another's.

    The reason is the shape of the check: every phrase is required, and beyond that the answer is
    free. So an answer satisfying task A also satisfies task B whenever B's phrases are all present
    in A's — which makes the two fixtures interchangeable and lets one reading of one directory
    score on both.

    Sharing individual words is fine and often deliberate: `index` appears in three tasks, each
    paired with a different second phrase that only its own defect produces. Sharing a whole set is
    not, and this test is here because it happened — `fix-missing-dict-default` and
    `fix-dict-get-chain` both carried `("keyerror", "get")` when the second tranche was written,
    which is exactly equivalent, and nothing would have caught it by eye.
    """
    sets = {task_id: {phrase.lower() for phrase in phrases} for task_id, phrases in EXPECT.items()}
    clashes: list[str] = []
    for a_id, a in sets.items():
        for b_id, b in sets.items():
            if a_id != b_id and b <= a:
                clashes.append(f"{b_id} {sorted(b)} is satisfied by {a_id} {sorted(a)}")
    assert not clashes, "interchangeable tasks:\n  " + "\n  ".join(clashes)


def test_alternation_accepts_every_spelling_of_one_mechanism() -> None:
    """`copy|in.?place|mutat` is one concept with three spellings, and all three must pass.

    The measurement behind this: on 2026-10-11 four arms read the same fixture and three diagnosed
    the same in-place mutation in three different words — "changes the original `table` directly",
    "modifies the original table in place", "mutates each dict in the original list". The single
    phrase `mutat` passed one of the three, and the score it produced was quoted as a result. A
    phrasing that is correct and scored wrong corrupts the comparison more quietly than a wrong
    answer scored right, because nothing about it looks anomalous.
    """
    from harness.tasks.check_diagnosis import _spells

    # The three real answers, as they were written.
    direct = "this changes the original table directly"
    in_place = "modifies the original table in place instead of creating a new copy"
    mutated = "mutates each dict in the original list"
    hyphenated = "modifies the existing rows in-place"

    for text in (direct, in_place, mutated, hyphenated):
        assert _spells("copy|in.?place|mutat|changes the original|modifies the original", text), text

    # And the wildcard is exactly one character: two words are not one spelling of `in?place`.
    assert not _spells("in.?place", "a thing in some place")


def test_a_phrase_matches_itself_literally() -> None:
    """Regex metacharacters in a phrase are literal, so `key=len` and `list.sort` match themselves."""
    from harness.tasks.check_diagnosis import _spells

    assert _spells("key=len", "the call is key=len(words)")
    assert _spells("list.sort", "list.sort returns None")
    assert not _spells("key=len", "the key is length")


def test_a_task_never_requires_and_forbids_the_same_phrase() -> None:
    """A phrase in both lists can never be satisfied, so the task would fail every run."""
    from harness.tasks.build_workspaces import FORBID

    for task_id, banned in FORBID.items():
        expected = {phrase.lower() for phrase in EXPECT.get(task_id, ())}
        clash = expected & {phrase.lower() for phrase in banned}
        assert not clash, f"{task_id} both requires and forbids {clash}"
