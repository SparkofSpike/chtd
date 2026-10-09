"""Decision points and prefix digests: the units the whole comparison rests on."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.ledger import open_ledger
from harness.trajectory import (
    BOOKKEEPING_TYPES,
    EXTENSION_EVENT_PREFIX,
    Action,
    DecisionPoint,
    Trajectory,
    decision_points,
    is_conversation,
    prefix_chain,
    read_trajectory,
)
from tests.ledger_fixture import build_ledger

MODEL_SETTINGS = [{"slot": "primary", "route": "student", "model": "qwen-small"}]


def _student(tmp_path: Path):
    """A short run: read a file, then call a tool that fails, then answer in text."""
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("student").start("student", model_settings=MODEL_SETTINGS)
    builder.user("student", "fix the failing test")
    builder.turn_start("student")
    builder.step_start("student")
    builder.tool_call("student", "read", {"path": "a.txt"}, tool_use_id="c1")
    builder.tool_result("student", "c1", is_error=False, text="contents")
    builder.tool_call("student", "shell", {"cmd": "pytest"}, tool_use_id="c2")
    builder.tool_result("student", "c2", is_error=True, text="1 failed")
    builder.assistant("student", "the test fails because the fixture is stale")
    builder.close()
    return tmp_path / "sessions.db"


def test_decision_points_are_extracted_in_order(tmp_path: Path) -> None:
    with open_ledger(_student(tmp_path)) as ledger:
        points = read_trajectory(ledger, "student").points
    assert [p.action.kind for p in points] == ["tool_call", "tool_call", "message"]
    assert [p.action.name for p in points] == ["read", "shell", None]
    assert [p.ordinal for p in points] == [0, 1, 2]
    assert points[0].action.payload == {"path": "a.txt"}


def test_tool_outcomes_are_attached_to_the_decision_that_caused_them(tmp_path: Path) -> None:
    # "The student chose badly" and "the tool was broken" are different events. A gate that
    # conflated them would teach the teacher's caution about a broken tool.
    with open_ledger(_student(tmp_path)) as ledger:
        points = read_trajectory(ledger, "student").points
    assert points[0].errored is False
    assert points[1].errored is True
    assert points[2].errored is None  # a message has no result to report


def test_session_start_provenance_is_read_from_the_ledger(tmp_path: Path) -> None:
    with open_ledger(_student(tmp_path)) as ledger:
        trajectory = read_trajectory(ledger, "student")
    assert trajectory.preset == "chtd"
    assert trajectory.resolved_profile_hash == "sha256-" + "a" * 64
    assert trajectory.model_settings == tuple(MODEL_SETTINGS)
    assert trajectory.parent_key is None
    assert trajectory.last_seq == 9


def test_a_prefix_digest_is_a_function_of_content_not_of_sequence(tmp_path: Path) -> None:
    # A fork starts its child at the parent's sequence numbers, so sequence would work there and
    # break for every other case: a trajectory replayed into a fresh ledger, or recorded on a
    # machine whose ledger began elsewhere. Content is what makes two states the same state.
    #
    # The two sessions also carry different keys and different session/start rows; those are
    # bookkeeping, so they must not reach the digest either.
    builder = build_ledger(tmp_path / "a.db")
    builder.session("a").start("a").user("a", "same question").tool_call("a", "read", {"path": "x"})
    builder.session("b").start("b").user("b", "same question").tool_call("b", "read", {"path": "x"})
    builder.close()

    with open_ledger(tmp_path / "a.db") as ledger:
        a = read_trajectory(ledger, "a")
        b = read_trajectory(ledger, "b")
    assert a.points[0].prefix_digest == b.points[0].prefix_digest


def test_different_conversations_do_not_share_a_prefix(tmp_path: Path) -> None:
    builder = build_ledger(tmp_path / "a.db")
    builder.session("a").start("a").user("a", "one question").tool_call("a", "read", {})
    builder.session("b").start("b").user("b", "another question").tool_call("b", "read", {})
    builder.close()

    with open_ledger(tmp_path / "a.db") as ledger:
        assert (
            read_trajectory(ledger, "a").points[0].prefix_digest
            != read_trajectory(ledger, "b").points[0].prefix_digest
        )


def test_a_forked_child_meets_its_parent_in_the_same_state(tmp_path: Path) -> None:
    # The design's central mechanism: the teacher is dropped into the student's exact prefix. The
    # child writes its own session/start (and the harness its budget.state, inbox and one
    # x/core/model-switch per pinned model slot) at the boundary, so this is also the test that
    # those rows stay out of the digest — with them in it, nothing would ever align and the
    # pipeline would find no divergences at all.
    #
    # The model-switch row is the one that bites: the host resolves a preset's `default` sentinel
    # before core sees it, so `preset.model.id` is non-empty in the ordinary case and `forkInto`
    # therefore writes one of these on every fork.
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("student").start("student")
    builder.user("student", "run the suite")
    # The fork boundary is the state *before* the student decides: this is where the teacher is
    # dropped in, and it is where its own rows begin.
    builder.fork_from(
        "student",
        "teacher",
        2,
        model_selections=[{"slot": "primary", "route": "teacher", "model": "qwen-large"}],
    )
    builder.start("teacher", parent={"key": "student", "boundarySeq": 2})
    builder.append("teacher", "budget.state", None)
    builder.append("teacher", "inbox", None)
    builder.tool_call("teacher", "shell", {"cmd": "pytest -x --lf"}, tool_use_id="t1")
    # The student decides in the same state, differently.
    builder.tool_call("student", "shell", {"cmd": "pytest"}, tool_use_id="c1")
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        student = read_trajectory(ledger, "student")
        teacher = read_trajectory(ledger, "teacher")

    assert student.points[0].prefix_digest == teacher.points[0].prefix_digest
    assert student.points[0].action.fingerprint() != teacher.points[0].action.fingerprint()


def test_extension_events_never_move_the_chain() -> None:
    # `x/` is the harness's extension namespace — core's own audit rows and ordinary plugin rows
    # alike. A plugin names its own events (`x/plugin/<hash>/<name>`), so the set is not
    # enumerable; a reader that had to list them would silently stop aligning the moment somebody
    # installed a plugin.
    for event_type in (
        "x/core/model-switch",
        "x/core/preset-switch",
        "x/core/context-breakdown",
        "x/core/compaction-begin",
        "x/plugin/6a629743672f8122/my-event",
        "x/agnes/whatever/next",
    ):
        assert is_conversation(_event(event_type)) is False, event_type
    # And the fixed conversation vocabulary is still state.
    for event_type in ("user/message", "assistant/message", "tool/call", "tool/result"):
        assert is_conversation(_event(event_type)) is True, event_type


def test_register_rows_never_move_the_chain() -> None:
    # The reducer's OBJECT_REGISTERS set: rows that maintain a register rather than carry a
    # message. A fork rewrites some of them at the boundary, which is reason enough.
    for event_type in ("budget.state", "inbox", "plan.items", "artifact/job", "op.state"):
        assert is_conversation(_event(event_type)) is False, event_type


def test_a_child_that_inherited_a_different_prefix_does_not_align(tmp_path: Path) -> None:
    # The other half of the same property: alignment has to be able to fail, or "they aligned" would
    # mean nothing.
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("student").start("student")
    builder.user("student", "run the suite")
    builder.tool_call("student", "shell", {"cmd": "pytest"}, tool_use_id="c1")
    builder.fork_from("student", "teacher", 2)
    builder.start("teacher", parent={"key": "student", "boundarySeq": 2})
    builder.user("teacher", "a different question entirely")
    builder.tool_call("teacher", "shell", {"cmd": "pytest"}, tool_use_id="t1")
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        student = read_trajectory(ledger, "student")
        teacher = read_trajectory(ledger, "teacher")
    assert student.points[0].prefix_digest != teacher.points[0].prefix_digest


def test_the_known_bookkeeping_rows_are_not_state() -> None:
    for event_type in ("session/start", "turn/start", "budget.state", "inbox", "step/start"):
        assert event_type in BOOKKEEPING_TYPES
        assert not is_conversation(_event(event_type))
    for event_type in ("user/message", "assistant/message", "tool/call", "tool/result"):
        assert is_conversation(_event(event_type))


def test_a_type_nobody_has_seen_counts_as_state() -> None:
    # The default outside the extension namespace. An unlisted bookkeeping row makes two identical
    # states look different, which shows up at once as a pipeline that finds nothing; the opposite
    # default would align two genuinely different states and teach from a comparison that was never
    # true. So an unknown type is state until somebody classifies it.
    assert is_conversation(_event("some-new-type-nobody-has-seen"))
    assert is_conversation(_event("session/something-added-later"))


def test_the_extension_namespace_is_the_one_open_set() -> None:
    # `x/` inverts the default, because there the set is open by construction: a plugin chooses its
    # own event name, so no list can be complete and a reader that waited to learn one would stop
    # aligning the moment somebody installed a plugin.
    assert EXTENSION_EVENT_PREFIX == "x/"
    assert not is_conversation(_event("x/anything/at/all"))
    assert not is_conversation(_event("x/plugin/a/b/c"))


def test_the_chain_does_not_move_across_bookkeeping_rows() -> None:
    rows = [
        _event("user/message", {"content": [{"type": "text", "text": "hi"}]}, seq=1),
        _event("turn/start", {}, seq=2),
        _event("budget.state", None, seq=3),
        _event("inbox", None, seq=4),
    ]
    chain = prefix_chain(rows)
    # The user message is state and moves the chain; the three rows after it are not, so a decision
    # made after them stands in the same state as one made immediately after the message.
    assert chain[1] != chain[2]
    assert chain[2] == chain[3] == chain[4]


def test_the_index_refuses_two_different_decisions_in_one_state() -> None:
    # An invariant rather than a case the ledger can produce: because a decision is itself
    # conversation, the next decision has it in its prefix and cannot share the digest. The check
    # guards what its failure would mean — a digest that stopped discriminating would align states
    # that were never the same, and the corruption would be silent. Raised directly for that reason.
    shared = "d" * 64
    trajectory = Trajectory(
        session_key="s",
        lane="main",
        preset="chtd",
        resolved_profile_hash=None,
        model_settings=(),
        points=(
            DecisionPoint(seq=1, ordinal=0, prefix_digest=shared, action=Action("tool_call", "read", {})),
            DecisionPoint(seq=2, ordinal=1, prefix_digest=shared, action=Action("tool_call", "write", {})),
        ),
        last_seq=2,
        event_types=(),
    )
    with pytest.raises(ValueError, match="decided twice in one state"):
        trajectory.by_prefix()


def test_consecutive_calls_are_in_different_states(tmp_path: Path) -> None:
    # The first call is part of the state the second was made in, so a lane of consecutive calls is
    # a lane of distinct states — which is what makes the index above an invariant.
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("s").start("s")
    builder.tool_call("s", "read", {"path": "a"})
    builder.tool_call("s", "read", {"path": "a"})
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        index = read_trajectory(ledger, "s").by_prefix()
    assert len(index) == 2


def test_lanes_are_kept_apart(tmp_path: Path) -> None:
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("s").start("s")
    builder.tool_call("s", "read", {"path": "main-lane"})
    builder.append("s", "tool/call", {"name": "read", "args": {"path": "side"}}, origin="model", lane="side")
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        assert len(decision_points(ledger.events("s"), lane="main")) == 1
        assert len(decision_points(ledger.events("s"), lane="side")) == 1


def test_an_unknown_event_type_is_not_a_decision_and_is_not_dropped_from_the_record(tmp_path: Path) -> None:
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("s").start("s")
    builder.append("s", "x/future-decision", {"name": "read", "args": {}}, origin="model")
    builder.tool_call("s", "read", {})
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        trajectory = read_trajectory(ledger, "s")
    assert len(trajectory.points) == 1
    assert "x/future-decision" in trajectory.event_types


def test_a_message_with_no_text_is_not_a_decision(tmp_path: Path) -> None:
    # An empty assistant turn is not a choice; recording it would put a decision point where the
    # model decided nothing and skew every divergence count.
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("s").start("s")
    builder.append("s", "assistant/message", {"content": []}, origin="model")
    builder.append("s", "assistant/message", {"content": [{"type": "text", "text": ""}]}, origin="model")
    builder.close()

    with open_ledger(tmp_path / "sessions.db") as ledger:
        assert read_trajectory(ledger, "s").points == ()


def test_an_action_fingerprint_ignores_the_per_call_id(tmp_path: Path) -> None:
    # Two runs that made the identical choice carry different toolUseIds. Hashing the id would make
    # every choice look like a divergence.
    a = Action(kind="tool_call", name="read", payload={"path": "x"})
    b = Action(kind="tool_call", name="read", payload={"path": "x"})
    c = Action(kind="tool_call", name="read", payload={"path": "y"})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_argument_key_order_does_not_change_the_fingerprint() -> None:
    a = Action(kind="tool_call", name="edit", payload={"path": "x", "text": "y"})
    b = Action(kind="tool_call", name="edit", payload={"text": "y", "path": "x"})
    assert a.fingerprint() == b.fingerprint()


def test_a_empty_lane_has_no_prefix_entries() -> None:
    assert prefix_chain([]) == {}


def _event(event_type: str, data: object = None, *, seq: int = 1):
    from harness.ledger import Event

    return Event(
        session_key="s",
        seq=seq,
        ts="t",
        id=f"e{seq}",
        type=event_type,
        lane="main",
        origin="system",
        trust="trusted",
        register=None,
        data=data,
        actor={},
        source_event_seqs=(),
    )


def test_reading_a_session_with_no_events_names_the_ledger(tmp_path: Path) -> None:
    builder = build_ledger(tmp_path / "sessions.db")
    builder.session("empty")
    builder.close()
    with open_ledger(tmp_path / "sessions.db") as ledger:
        with pytest.raises(ValueError, match="no events"):
            read_trajectory(ledger, "empty")
