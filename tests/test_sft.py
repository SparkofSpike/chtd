"""The dataset's rows, expanded into training samples.

The property under test is fidelity: a sample must be the conversation that ran, plus the teacher's
answer, and nothing else. Two failure modes are worth more than the rest and are tested directly —
a label that is not what the teacher did, and a state that is not what the student was in.

No model, no GPU, no network: the harness prefix is a fixture and the ledger is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.dataset import Dataset, DatasetEntry, DatasetManifest
from distill.sft import (
    HarnessPrefix,
    PrefixMissing,
    StateMismatch,
    UnknownAction,
    UnknownState,
    build_samples,
    conversation_before,
    label_message,
    write_samples,
)
from harness.ledger import open_ledger
from tests.ledger_fixture import build_ledger

SESSION = "agnes:local:local-dev:cli:workspace:" + "a" * 16


def _entry(**overrides: object) -> DatasetEntry:
    fields: dict[str, object] = {
        "task_id": "t1",
        "prefix_digest": "935ef0d446090af60b5748a48b5cb847fa5440ff945ad705ad0e40f3546b0e1f",
        "student_session": SESSION,
        "teacher_session": "agnes:local:local-dev:cli:workspace:" + "b" * 16,
        "student_seq": 9,
        "teacher_seq": 9,
        "student_action": json.dumps({"kind": "message", "name": None, "payload": "Hello!"}),
        "teacher_action": json.dumps(
            {"kind": "tool_call", "name": "read", "payload": {"path": "notes.txt"}}
        ),
        "teacher_action_fingerprint": '{"args":{"path":"notes.txt"},"tool":"read"}',
    }
    fields.update(overrides)
    return DatasetEntry(**fields)  # type: ignore[arg-type]


def _prefix(tmp_path, *, system: str = "You are on a track.", tools: list | None = None) -> HarnessPrefix:
    path = tmp_path / "capture.json"
    path.write_text(
        json.dumps(
            {
                "provenance": {"captured_from": "a test"},
                "system": system,
                "tools": tools if tools is not None else [{"name": "read", "description": "d", "parameters": {}}],
            }
        ),
        encoding="utf-8",
    )
    return HarnessPrefix.load(path)


# ------------------------------------------------------------------ the prefix


def test_prefix_loads_both_halves(tmp_path) -> None:
    prefix = _prefix(tmp_path)
    assert prefix.system == "You are on a track."
    assert prefix.tools[0]["name"] == "read"


def test_prefix_digest_covers_the_tools_too(tmp_path) -> None:
    """Two runs that differed only in the disclosed tools are two different inputs."""
    a = _prefix(tmp_path, tools=[{"name": "read", "description": "d", "parameters": {}}])
    b = _prefix(tmp_path, tools=[{"name": "write", "description": "d", "parameters": {}}])
    assert a.digest() != b.digest()


def test_missing_prefix_is_named_not_defaulted(tmp_path) -> None:
    with pytest.raises(PrefixMissing, match="no harness prefix"):
        HarnessPrefix.load(tmp_path / "absent.json")


def test_prefix_without_a_system_prompt_is_refused(tmp_path) -> None:
    path = tmp_path / "capture.json"
    path.write_text(json.dumps({"tools": []}), encoding="utf-8")
    with pytest.raises(PrefixMissing, match="system prompt"):
        HarnessPrefix.load(path)


# ------------------------------------------------- the conversation, rebuilt


def test_conversation_is_the_shape_the_harness_sends(tmp_path) -> None:
    """The measured wire shape, not an idealised one.

    A user message carries content blocks; an assistant message carries a plain string; a tool call
    lives inside the assistant turn that emitted it; a tool result is its own role naming the call.
    Each of those was read off a captured request, so a change here is a change of contract with the
    harness rather than a style choice.
    """
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "Read notes.txt")
    builder.append(SESSION, "assistant/message", {"content": [], "stopReason": "tool_use"}, origin="model")
    builder.tool_call(SESSION, "read", {"path": "notes.txt"}, tool_use_id="t0-abc")
    builder.tool_result(SESSION, "t0-abc", text="1\talpha\n2\tbravo\n")
    builder.append(SESSION, "assistant/message", {"content": [{"type": "text", "text": "bravo"}]}, origin="model")
    builder.close()

    with open_ledger(path) as ledger:
        messages = conversation_before(ledger, SESSION, before_seq=99)

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[0]["content"] == [{"type": "text", "text": "Read notes.txt"}]
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": '{"path":"notes.txt"}',
    }
    assert messages[1]["tool_calls"][0]["id"] == "t0-abc"
    assert messages[2] == {
        "role": "tool",
        # The trailing newline is kept: the wire carries what the ledger holds.
        "content": "1\talpha\n2\tbravo\n",
        "tool_call_id": "t0-abc",
    }
    assert messages[3]["content"] == "bravo"


def test_the_range_ends_before_the_decision(tmp_path) -> None:
    """The state a decision was made *in*, so the decision is not part of it."""
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "first")
    builder.assistant(SESSION, "answer one")
    decision_seq = builder._seq[SESSION] + 1
    builder.assistant(SESSION, "the decision itself")
    builder.close()

    with open_ledger(path) as ledger:
        messages = conversation_before(ledger, SESSION, before_seq=decision_seq)

    assert [m["role"] for m in messages] == ["user", "assistant"]
    # A user message carries content blocks; an assistant message carries a plain string.
    assert messages[0]["content"] == [{"type": "text", "text": "first"}]
    assert messages[1]["content"] == "answer one"


def test_an_empty_assistant_turn_is_dropped(tmp_path) -> None:
    """A row that reached the reader but never reached the model would be a different conversation.

    The provider layer skips an assistant message with neither content nor tool calls, so a sample
    that kept one would be one message longer than the run it claims to be.
    """
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "hi")
    builder.append(SESSION, "assistant/message", {"content": [], "stopReason": "aborted"}, origin="model")
    builder.close()

    with open_ledger(path) as ledger:
        messages = conversation_before(ledger, SESSION, before_seq=99)

    assert [m["role"] for m in messages] == ["user"]


def test_bookkeeping_rows_are_not_messages(tmp_path) -> None:
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "hi")
    builder.append(SESSION, "budget.state", {"slot": "primary"}, origin="system")
    builder.append(SESSION, "x/core/op-mark", {"phase": "inference"}, origin="system")
    builder.close()

    with open_ledger(path) as ledger:
        messages = conversation_before(ledger, SESSION, before_seq=99)

    assert [m["role"] for m in messages] == ["user"]


def test_a_tool_result_is_not_repeated(tmp_path) -> None:
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "hi")
    builder.append(SESSION, "assistant/message", {"content": [], "stopReason": "tool_use"}, origin="model")
    builder.tool_call(SESSION, "read", {"path": "x"}, tool_use_id="t0-a")
    builder.tool_result(SESSION, "t0-a", text="one")
    builder.tool_result(SESSION, "t0-a", text="duplicate")
    builder.close()

    with open_ledger(path) as ledger:
        messages = conversation_before(ledger, SESSION, before_seq=99)

    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages[2]["content"] == "one"


# ---------------------------------------------------------------- the label


def test_a_tool_call_label_is_an_assistant_turn_with_the_call_in_it() -> None:
    label = label_message(_entry())
    assert label["role"] == "assistant"
    assert label["content"] is None
    call = label["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"path": "notes.txt"}


def test_a_message_label_keeps_the_teacher_text_whole() -> None:
    """Not the rendering: `Action.describe` truncates at 200 characters."""
    long_text = "x" * 400
    entry = _entry(
        teacher_action=json.dumps({"kind": "message", "name": None, "payload": long_text}),
    )
    assert label_message(entry)["content"] == long_text


def test_the_same_row_always_yields_the_same_call_id() -> None:
    """So that rebuilding the dataset reproduces the sample byte for byte."""
    assert label_message(_entry()) == label_message(_entry())


def test_different_rows_get_different_call_ids() -> None:
    a = label_message(_entry())
    b = label_message(_entry(prefix_digest="0" * 64))
    assert a["tool_calls"][0]["id"] != b["tool_calls"][0]["id"]


def test_an_action_with_no_message_form_is_refused() -> None:
    """Refused rather than skipped: a row silently dropped is a dataset smaller than it says."""
    entry = _entry(teacher_action=json.dumps({"kind": "thinking", "name": None, "payload": "hmm"}))
    with pytest.raises(UnknownAction, match="no message form"):
        label_message(entry)


# ------------------------------------------------------------ the whole row


def _real_digest(path, session: str, seq: int) -> str:
    """The digest the harness gives the state at ``seq``, computed from the ledger.

    A sample row is checked against this, so a test that hard-coded a digest would be testing its
    own literal rather than the join — and would pass while the join was broken. These tests build
    the real one instead, which is also how they caught the check being added.
    """
    from harness.trajectory import prefix_chain

    with open_ledger(path) as ledger:
        return prefix_chain(ledger.events(session))[seq]


def _dataset(entries: list[DatasetEntry]) -> Dataset:
    return Dataset(
        entries=tuple(entries),
        refusals=(),
        manifest=DatasetManifest(
            track_sha256="f" * 64,
            dataset_sha256="f" * 64,
            entries=len(entries),
            refusals=0,
            tasks=("t1",),
            arms=("baseline", "chtd"),
            created_at="2026-10-09T00:00:00Z",
        ),
    )


def test_a_sample_is_the_prefix_the_conversation_and_the_label(tmp_path) -> None:
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "Read the file notes.txt")
    decision_seq = builder._seq[SESSION] + 1
    builder.assistant(SESSION, "Hello! How can I help?")
    builder.close()

    prefix = _prefix(tmp_path)
    samples = build_samples(
        dataset=_dataset([_entry(prefix_digest=_real_digest(path, SESSION, decision_seq),
                                student_seq=decision_seq)]),
        student_ledger=path,
        prefix=prefix,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert [m["role"] for m in sample.messages] == ["system", "user", "assistant"]
    assert sample.messages[0]["content"] == prefix.system
    assert sample.messages[1]["content"][0]["text"] == "Read the file notes.txt"
    # The label is the teacher's call, not the student's "Hello!" that was recorded in the state.
    assert sample.messages[2]["tool_calls"][0]["function"]["name"] == "read"
    assert sample.tools == prefix.tools
    assert sample.harness_prefix_digest == prefix.digest()


def test_a_sample_carries_no_gate_evidence(tmp_path) -> None:
    """The dataset's guarantee, kept: privileged information decided admission, not content."""
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "Read the file")
    decision_seq = builder._seq[SESSION] + 1
    builder.assistant(SESSION, "Hello!")
    builder.close()

    sample = build_samples(
        dataset=_dataset([_entry(prefix_digest=_real_digest(path, SESSION, decision_seq),
                                student_seq=decision_seq)]),
        student_ledger=path,
        prefix=_prefix(tmp_path),
    )[0]

    assert set(json.loads(sample.to_json())) == {
        "task_id",
        "prefix_digest",
        "harness_prefix_digest",
        "messages",
        "tools",
    }
    assert "outcome" not in sample.to_json()
    assert "succeeded" not in sample.to_json()


def test_samples_are_written_one_per_line(tmp_path) -> None:
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "Read the file")
    decision_seq = builder._seq[SESSION] + 1
    builder.assistant(SESSION, "Hello!")
    builder.close()

    samples = build_samples(
        dataset=_dataset([_entry(prefix_digest=_real_digest(path, SESSION, decision_seq),
                                student_seq=decision_seq)]),
        student_ledger=path,
        prefix=_prefix(tmp_path),
    )
    target = write_samples(samples, tmp_path / "out" / "train.jsonl")

    lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["task_id"] == "t1"


def test_a_tool_result_is_delivered_byte_for_byte(tmp_path) -> None:
    """No tidying on the way through.

    A rebuild that stripped whitespace would differ from the endpoint by a byte or two per tool
    result — small, invisible in a diff of the messages, and a real difference between the
    conversation the model read and the one being trained on.
    """
    raw = "1\talpha\r\n2\tbravo\r\n3\tcharlie\r"
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "Read the file")
    builder.append(SESSION, "assistant/message", {"content": [], "stopReason": "tool_use"},
                   origin="model")
    builder.tool_call(SESSION, "read", {"path": "notes.txt"}, tool_use_id="t0-a")
    builder.tool_result(SESSION, "t0-a", text=raw)
    builder.close()

    with open_ledger(path) as ledger:
        messages = conversation_before(ledger, SESSION, before_seq=99)

    assert messages[2]["content"] == raw
    assert messages[2]["content"].endswith("\r")


def test_a_missing_ledger_is_named(tmp_path) -> None:
    with pytest.raises(PrefixMissing, match="no student ledger"):
        build_samples(
            dataset=_dataset([_entry()]),
            student_ledger=tmp_path / "absent.db",
            prefix=_prefix(tmp_path),
        )


# ------------------------------------------------- the state is checked, not claimed


def test_a_row_whose_state_cannot_be_rebuilt_is_refused(tmp_path) -> None:
    """The check that makes `prefix_digest` a measurement rather than a label.

    The dataset copies a digest the harness computed; this module rebuilds the conversation
    independently. If the two disagree the sample is stamped with a state it does not hold, and
    every claim about which state the teacher answered in becomes false while looking normal. Here
    the row names a digest nothing in the ledger produces, and the build must stop.
    """
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "Read the file")
    decision_seq = builder._seq[SESSION] + 1
    builder.assistant(SESSION, "Hello!")
    builder.close()

    with pytest.raises(StateMismatch, match="hashes to"):
        build_samples(
            dataset=_dataset([_entry(student_seq=decision_seq, prefix_digest="0" * 64)]),
            student_ledger=path,
            prefix=_prefix(tmp_path),
        )


def test_a_row_naming_a_sequence_that_is_not_there_is_refused(tmp_path) -> None:
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "Read the file")
    builder.close()

    with pytest.raises(StateMismatch, match="no event at seq"):
        build_samples(
            dataset=_dataset([_entry(student_seq=9999)]),
            student_ledger=path,
            prefix=_prefix(tmp_path),
        )


def test_the_rebuild_matches_the_captured_wire_shape(tmp_path) -> None:
    """The shape comes from a captured request, and this is what makes that a fact rather than a claim.

    `harness/prefix/capture-1.json` carries the assistant and tool turns of a request the harness
    actually sent, dumped by the endpoint that received it. This test feeds the same events into a
    synthetic ledger and requires the rebuild to produce the same messages field for field. If the
    harness changes how it renders a tool result — a different key, a different role — this fails
    here rather than producing a fine-tune trained on a conversation the endpoint never sends.
    """
    fixture = json.loads(
        (Path(__file__).resolve().parent.parent / "harness" / "prefix" / "capture-1.json")
        .read_text(encoding="utf-8")
    )
    captured = fixture["tool_turn"]
    assistant_turn = captured["assistant_turn"]
    tool_turn = captured["tool_result_turn"]

    # The captured role order, with the system prompt and the user message the harness also sent.
    assert captured["message_roles"] == ["system", "user", "assistant", "tool"]

    call = assistant_turn["tool_calls"][0]
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "Read the file notes.txt")
    builder.append(SESSION, "assistant/message", {"content": [], "stopReason": "tool_use"},
                   origin="model")
    builder.tool_call(SESSION, call["function"]["name"],
                      json.loads(call["function"]["arguments"]), tool_use_id=call["id"])
    builder.tool_result(SESSION, call["id"], text=tool_turn["content"])
    builder.close()

    with open_ledger(path) as ledger:
        messages = conversation_before(ledger, SESSION, before_seq=99)

    rebuilt_call = messages[1]
    assert rebuilt_call["role"] == assistant_turn["role"]
    assert rebuilt_call["content"] == assistant_turn["content"]
    assert rebuilt_call["tool_calls"][0]["type"] == call["type"]
    assert rebuilt_call["tool_calls"][0]["id"] == call["id"]
    assert rebuilt_call["tool_calls"][0]["function"]["name"] == call["function"]["name"]
    assert (rebuilt_call["tool_calls"][0]["function"]["arguments"]
            == json.dumps(json.loads(call["function"]["arguments"]), separators=(",", ":")))

    rebuilt_result = messages[2]
    assert rebuilt_result["role"] == tool_turn["role"]
    assert rebuilt_result["tool_call_id"] == tool_turn["tool_call_id"]
    # Byte for byte, including the trailing carriage return the capture holds. An earlier version of
    # the rebuild stripped it; that is the difference this test exists to catch.
    assert rebuilt_result["content"] == tool_turn["content"]


def test_an_unrecognised_state_row_is_refused_not_skipped(tmp_path) -> None:
    """The direction that matters. `is_conversation` treats an unknown type as state on purpose —
    the harness's own note says the opposite default would align two genuinely different states. A
    reader that silently dropped such a row would build a conversation missing a message the model
    read, and the sample would look entirely well-formed.
    """
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "Read the file")
    # Not in BOOKKEEPING_TYPES and not under the `x/` prefix, so the harness counts it as state.
    builder.append(SESSION, "assistant/annotation", {"note": "something the model read"}, origin="model")
    # One more row, so the decision the row names is a sequence that exists: this test is about the
    # unknown type being refused, not about a missing one.
    builder.user(SESSION, "carry on")
    builder.close()

    with pytest.raises(UnknownState, match="has no message form"):
        build_samples(
            dataset=_dataset([_entry(prefix_digest=_real_digest(path, SESSION, 4), student_seq=4)]),
            student_ledger=path,
            prefix=_prefix(tmp_path),
        )


def test_an_extension_row_is_not_state(tmp_path) -> None:
    """The other side of the same rule: an `x/` row is about the run, not part of it."""
    path = tmp_path / "sessions.db"
    builder = build_ledger(path)
    builder.session(SESSION).start(SESSION)
    builder.user(SESSION, "hi")
    builder.append(SESSION, "x/plugin/whatever", {"anything": True}, origin="system")
    builder.append(SESSION, "assistant/message", {"content": [{"type": "text", "text": "ok"}]},
                   origin="model")
    builder.close()

    with open_ledger(path) as ledger:
        messages = conversation_before(ledger, SESSION, before_seq=99)

    assert [m["role"] for m in messages] == ["user", "assistant"]
