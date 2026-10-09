"""From the dataset to the rows a supervised fine-tune consumes.

`distill/dataset.py` records *which* decision the teacher made in a student's state. This module
expands each of those rows into the thing a trainer eats: the conversation the student was in, the
tool disclosure it was given, and the teacher's answer as the assistant turn to imitate.

Three things decide whether a row is faithful, and each is a decision rather than a detail.

**The harness prefix is a capture, not a reconstruction.** The system prompt is assembled from
several packages, and the disclosed tools come from the loaded assembly rather than from the preset
— the preset names two tools, a real run discloses twenty-two. Rebuilding either from source would
produce a sample that is subtly *not* what the model saw, and a fine-tune trained against a
mismatched prefix measures the mismatch. `harness/prefix/capture-1.json` holds the bytes the
harness actually sent, with the digest of the request they came from.

**The prefix is left in.** It is tempting to drop the untrusted-content rules and an
`<available_skills>` block for a task about reading a file, because they look like noise. They are
left in anyway: the claim under test is that a capability transfers *within the harness*, and the
harness is what the student will meet at inference. A sample cleaned of the prefix would be a
sample of a different harness.

**The label carries only what the teacher did.** The dataset guarantees no gate evidence reaches a
row, and this module preserves that: a sample's keys are fixed by :meth:`TrainingSample.to_json`,
and nothing here reads an outcome or a verdict. Privileged information decided admission; it does
not appear in what the student is trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distill.dataset import Dataset, DatasetEntry, read_dataset
from harness.ledger import Ledger, open_ledger
from harness.trajectory import _message_text

__all__ = [
    "HarnessPrefix",
    "PrefixMissing",
    "TrainingSample",
    "UnknownAction",
    "build_samples",
    "conversation_before",
    "main",
    "samples_digest",
    "write_samples",
]

#: The event types that become messages. Deliberately narrower than
#: `harness.trajectory.is_conversation`: that function answers "is this state?" for prefix
#: alignment, where an `assistant/message` with no text counts as nothing. Here the question is
#: "what did the model read?", and a tool call belongs to the assistant turn that emitted it even
#: when that turn carried no prose.
_MESSAGE_EVENTS = ("user/message", "assistant/message", "tool/call", "tool/result")


class PrefixMissing(RuntimeError):
    """The capture the samples must be built against is not there or is unreadable."""


class UnknownAction(RuntimeError):
    """A teacher action whose kind this module cannot turn into a message."""


@dataclass(frozen=True)
class HarnessPrefix:
    """What the harness says before any conversation: the system prompt and the tool disclosure.

    Both halves are needed. A sample carrying the system prompt but not the tools shows the model a
    conversation about tools it was never told it had; a sample carrying the tools but not the
    system prompt drops the rules that say how tool output is delimited.
    """

    system: str
    tools: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> HarnessPrefix:
        source = Path(path)
        if not source.exists():
            raise PrefixMissing(
                f"no harness prefix at {source}. The prefix cannot be reconstructed from this "
                "repository: the system prompt is assembled from several packages and the disclosed "
                "tools come from the loaded assembly. Capture one from a live run and store it "
                "under harness/prefix/."
            )
        payload = json.loads(source.read_text(encoding="utf-8"))
        system = payload.get("system")
        tools = payload.get("tools")
        if not isinstance(system, str) or not system:
            raise PrefixMissing(f"{source} carries no system prompt")
        if not isinstance(tools, list):
            raise PrefixMissing(f"{source} carries no tool disclosure")
        return cls(
            system=system,
            tools=tuple(tools),
            provenance=payload.get("provenance") or {},
        )

    def digest(self) -> str:
        """A digest over both halves, so a sample can name the prefix it was built against.

        Both halves, because two runs that differed only in the disclosed tools are two different
        inputs, and a digest over the system prompt alone would call them the same.
        """
        hasher = hashlib.sha256()
        hasher.update(b"chtd/harness-prefix/v1\n")
        hasher.update(self.system.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(
            json.dumps(self.tools, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return hasher.hexdigest()


@dataclass(frozen=True)
class TrainingSample:
    """One row: the state as the model saw it, and the teacher's answer as the turn to imitate.

    ``messages`` already ends with the label, which is how every supervised fine-tune format on the
    market spells "train on this turn and not the ones before it". The alternative — a separate
    label field — would need a trainer willing to read it, and no trainer is.
    """

    task_id: str
    prefix_digest: str
    harness_prefix_digest: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "task_id": self.task_id,
                "prefix_digest": self.prefix_digest,
                "harness_prefix_digest": self.harness_prefix_digest,
                "messages": list(self.messages),
                "tools": list(self.tools),
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    def label(self) -> Mapping[str, Any]:
        """The last message, which is the teacher's turn."""
        return self.messages[-1]


def _blocks_text(content: Any) -> str:
    """The text of a content block list, joined the way the harness joins it."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts).strip()


def conversation_before(
    ledger: Ledger,
    session_key: str,
    *,
    before_seq: int,
) -> tuple[Mapping[str, Any], ...]:
    """The messages a model read, up to but not including one decision.

    ``before_seq`` is the decision's own sequence, so the range is exclusive and the state rebuilt
    here is the state that decision was made *in* — the same state
    :func:`harness.trajectory.prefix_chain` digests, read out as messages rather than hashed.

    The output is in the shape the harness actually puts on the wire, which was measured rather
    than assumed (``harness/prefix/capture-1.json``): a user message carries content blocks, an
    assistant message carries a plain string or ``null``, a tool call lives inside the assistant
    turn that emitted it, and a tool result is its own ``tool`` role naming the call it answers.

    An assistant message that ends up with neither prose nor a tool call is dropped, because the
    provider layer drops it: a row that reached the trainer but never reached the model would make
    the sample a different conversation from the one that ran.
    """
    messages: list[dict[str, Any]] = []
    #: The index of the assistant message a following tool call belongs to, or None when the last
    #: assistant turn is already closed by a tool result.
    open_assistant: int | None = None
    answered: set[str] = set()

    for event in ledger.events(session_key):
        if event.seq >= before_seq:
            break
        if event.type not in _MESSAGE_EVENTS:
            continue
        data = event.data if isinstance(event.data, dict) else {}

        if event.type == "user/message":
            text = _blocks_text(data.get("content")) or _message_text(data)
            if text:
                messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
            open_assistant = None
        elif event.type == "assistant/message":
            text = _message_text(data)
            messages.append({"role": "assistant", "content": text or None})
            open_assistant = len(messages) - 1
        elif event.type == "tool/call":
            call_id = data.get("toolUseId")
            name = data.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                continue
            if open_assistant is None:
                # The turn that asked for it is not in range. Refusing is worse than skipping: a
                # prefix cut mid-turn is a real state, and the call still belongs to the turn that
                # made it, so an assistant message is opened for it rather than the call dropped.
                messages.append({"role": "assistant", "content": None})
                open_assistant = len(messages) - 1
            target = messages[open_assistant]
            calls = target.setdefault("tool_calls", [])
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            data.get("args"), ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            )
        elif event.type == "tool/result":
            call_id = data.get("toolUseId")
            if not isinstance(call_id, str) or call_id in answered:
                continue
            answered.add(call_id)
            messages.append(
                {
                    "role": "tool",
                    "content": _blocks_text(data.get("content")),
                    "tool_call_id": call_id,
                }
            )
            open_assistant = None

    return tuple(m for m in messages if m.get("content") is not None or m.get("tool_calls"))


def _tool_call_id(entry: DatasetEntry) -> str:
    """A stable id for the call the teacher made.

    Not the teacher's own ``toolUseId``: that value is minted per call in the teacher's session and
    means nothing in the student's, where the harness will mint a fresh one at inference. What the
    student has to learn is that a call is made and with which arguments. The id is derived from
    the row so that rebuilding the dataset reproduces the sample byte for byte.
    """
    hasher = hashlib.sha256()
    hasher.update(b"chtd/tool-call-id/v1\n")
    hasher.update(entry.prefix_digest.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(entry.teacher_action_fingerprint.encode("utf-8"))
    return "t0-" + hasher.hexdigest()[:32]


def label_message(entry: DatasetEntry) -> Mapping[str, Any]:
    """The teacher's decision, as the assistant turn the student imitates.

    The action is stored canonically by the dataset, so it is parsed rather than re-derived from a
    rendering: `Action.describe` truncates long messages, and a label built from it would be a
    different string from what the teacher said.
    """
    action = json.loads(entry.teacher_action)
    kind = action.get("kind")
    if kind == "message":
        text = action.get("payload")
        if not isinstance(text, str):
            raise UnknownAction(f"teacher message payload is not text: {type(text).__name__}")
        return {"role": "assistant", "content": text}
    if kind == "tool_call":
        name = action.get("name")
        if not isinstance(name, str):
            raise UnknownAction("teacher tool call has no name")
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": _tool_call_id(entry),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            action.get("payload"), ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
            ],
        }
    raise UnknownAction(
        f"teacher action kind {kind!r} has no message form. A sample built from it would show the "
        "student an assistant turn the teacher never produced."
    )


def build_samples(
    *,
    dataset: Dataset,
    student_ledger: str | Path,
    prefix: HarnessPrefix,
    limit: int | None = None,
) -> tuple[TrainingSample, ...]:
    """Expand every admitted row into a training sample.

    The student's ledger is the source of the states because the dataset names a session and a
    sequence, not a conversation: storing the conversation in the dataset would let it drift from
    the ledger it came from, and the ledger is the record the experiment rests on.
    """
    ledger_path = Path(student_ledger)
    if not ledger_path.exists():
        raise PrefixMissing(f"no student ledger at {ledger_path}")

    samples: list[TrainingSample] = []
    with open_ledger(ledger_path) as ledger:
        for entry in dataset.entries:
            conversation = conversation_before(
                ledger, entry.student_session, before_seq=entry.student_seq
            )
            messages = (
                {"role": "system", "content": prefix.system},
                *conversation,
                label_message(entry),
            )
            samples.append(
                TrainingSample(
                    task_id=entry.task_id,
                    prefix_digest=entry.prefix_digest,
                    harness_prefix_digest=prefix.digest(),
                    messages=messages,
                    tools=prefix.tools,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return tuple(samples)


def write_samples(samples: Iterable[TrainingSample], path: str | Path) -> Path:
    """One sample per line, in the order given.

    JSONL rather than a JSON array: a trainer streams it, and a run that dies partway leaves a file
    whose completed lines are still readable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json())
            handle.write("\n")
    return target


def sample_rows(samples: Sequence[TrainingSample]) -> tuple[str, ...]:
    """The rows as text, for a digest that a reader can recompute."""
    return tuple(sample.to_json() for sample in samples)


def samples_digest(rows: Sequence[str]) -> str:
    """A digest over the rows in order, so a run can name the sample set it trained on.

    Takes the row text — what ``write_samples`` wrote — rather than the objects, because the digest
    has to be over the file a reader can rebuild. Re-serialising a parsed object can differ from the
    original by a space or a float spelling, and a run whose recorded digest matches no buildable
    file is a run nobody can check.

    The construction matches :func:`distill.dataset.dataset_digest`: canonical JSON per row, joined
    with newlines, so the digest moves if a row is added, removed, reordered or altered.
    """
    hasher = hashlib.sha256()
    hasher.update(b"chtd/sft-samples/v1\n")
    for row in rows:
        hasher.update(row.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """The command behind the reproduction steps, so a reader can rebuild the rows themselves.

    It refuses to write a sample set built against a missing prefix or an empty dataset rather than
    emitting an empty file: an empty file is a fine-tune that trains on nothing, and it looks
    exactly like a successful step.
    """
    parser = argparse.ArgumentParser(
        description="Expand a gated dataset into the rows a supervised fine-tune consumes."
    )
    parser.add_argument("--dataset", required=True, help="a directory distill.pipeline wrote")
    parser.add_argument(
        "--student-ledger",
        required=True,
        help="the ledger holding the sessions the dataset's rows name",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="a captured request under harness/prefix/ (it cannot be reconstructed)",
    )
    parser.add_argument("--out", required=True, help="where the JSONL goes")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many samples")
    args = parser.parse_args(argv)

    dataset = read_dataset(args.dataset)
    manifest = dataset.manifest
    if manifest is None:  # read_dataset requires one; this is the type narrowing
        parser.error(f"{args.dataset} has no manifest")
    if not dataset.entries:
        parser.error(
            f"{args.dataset} holds no admitted entries; there is nothing to train on, and an "
            "empty sample set looks exactly like a successful build"
        )

    prefix = HarnessPrefix.load(args.prefix)
    samples = build_samples(
        dataset=dataset,
        student_ledger=args.student_ledger,
        prefix=prefix,
        limit=args.limit,
    )
    path = write_samples(samples, args.out)

    print(f"dataset: {manifest.dataset_sha256[:16]}  entries={manifest.entries}")
    print(f"prefix : {prefix.digest()[:16]}  system={len(prefix.system)} chars  "
          f"tools={len(prefix.tools)}")
    print(f"samples: {len(samples)}")
    for sample in samples:
        supervised = sample.label()
        kind = "tool_call" if supervised.get("tool_calls") else "message"
        print(f"  {sample.task_id:<24} state={sample.prefix_digest[:12]} label={kind} "
              f"messages={len(sample.messages)}")
    print(f"digest : {samples_digest(sample_rows(samples))}")
    print(f"wrote  : {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
