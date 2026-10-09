"""Turn a ledger into a trajectory of decision points.

A **decision point** is one place the model chose something: a tool call, or a message where a call
could have gone. That is the unit CHTD works in. The student's decision points define the states the
teacher is dropped into, and a divergence is two different choices at one state.

**State is a prefix digest.** The design says the teacher is dropped into the student's exact
prefix, so "the same state" has to be a fact rather than a resemblance. Each decision point
therefore carries a digest of the events that preceded it, computed as a chain over the lane's own
content. Two points with the same digest saw the same conversation; the harness's same-prefix fork
is what produces that, and this is what checks it.

The digest is over the **conversation**, not over the ledger. The harness records two kinds of row:
what the model said and saw, and what the machinery did about it. Only the first is state. The
second is excluded deliberately, because a same-prefix fork writes some of its own — a child's
`session/start`, its `budget.state`, its `inbox` — at the boundary, so a digest that covered them
would give the teacher a different prefix from the student *in the very state the fork exists to
reproduce*, and nothing would ever align.

The exclusion is by an explicit bookkeeping list rather than by an explicit content list, so the
default for a type nobody has seen before is to **include** it. That default is chosen for the
failure it produces: an unlisted bookkeeping row makes two identical states look different, which
shows up immediately as a pipeline that finds no divergences at all, while the opposite default —
silently ignoring a content row — would align two genuinely different states and quietly teach from
a comparison that was never true.

The digest is over content and not over `seq`. A fork starts its child at the parent's own sequence
numbers, so sequence would work for that case and break for every other — a trajectory replayed
into a fresh ledger, or recorded on a machine whose ledger began at a different point. Content is
what makes two states the same state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from harness.ledger import Event, Ledger

__all__ = [
    "Action",
    "BOOKKEEPING_TYPES",
    "EXTENSION_EVENT_PREFIX",
    "DecisionPoint",
    "Trajectory",
    "decision_points",
    "is_conversation",
    "prefix_chain",
    "read_trajectory",
]

ActionKind = Literal["tool_call", "message"]

#: Rows the harness writes about its own operation rather than about the conversation. The prefix
#: digest skips these: they are not state, and a same-prefix fork writes some of its own at the
#: boundary, so counting them would make the teacher's prefix differ from the student's in the very
#: state the fork reproduces.
#:
#: Enumerated from Agnes Harness rather than guessed. The conversation proper is a small, stable
#: vocabulary — `user/message`, `assistant/message`, `tool/call`, `tool/result` — while the
#: bookkeeping is large and grows with every feature. Every type below was found by searching the
#: harness source for written event types and deciding what each one is:
#:
#: - lifecycle and registers: `session/*`, `turn/*`, `step/*`, and the reducer's own register rows
#:   (`OBJECT_REGISTERS` in `packages/core/src/reduce/reducer.ts`).
#: - the model's streamed progress as opposed to its output: `assistant/output` carries `state`,
#:   `chars` and `estimatedTokens`, and is written several times per turn, so counting it would make
#:   two runs differ by how the stream was chunked.
#: - audit and accounting rows: approvals, cost, effects, deviations, verifier signals, subagent
#:   cost, runtime and tree convergence, repair decisions, user feedback.
#:
#: The `x/` namespace is handled separately, as a prefix, because it is not enumerable: an ordinary
#: plugin names its own events (`x/plugin/<hash>/<name>`, see `ctx.extension().events.append`) and
#: the harness writes about thirty `x/core/*` audit types of its own. A reader that had to list them
#: would stop aligning the moment somebody installed a plugin.
BOOKKEEPING_TYPES = frozenset(
    {
        # lifecycle
        "session/start",
        "session/opened",
        "turn/start",
        "turn/end",
        "step/start",
        "step/end",
        # registers (`OBJECT_REGISTERS` and the rows that maintain them)
        "budget.state",
        "inbox",
        "plan.items",
        "artifact/job",
        "op.state",
        "harness/entry",
        # request accounting
        "request/header",
        "request/sent",
        # the model's streamed progress, not its output
        "assistant/output",
        "assistant/chunk",
        # approvals
        "approval/asked",
        "approval/decided",
        "approval/guardian-decided",
        # audit and accounting
        "cost/ledger",
        "effect/intent",
        "effect/settled",
        "format/deviation",
        "harness/refine",
        "repair/decision",
        "runtime.converged",
        "runtime.other",
        "runtime.stale",
        "tree.stale",
        "subagent/cost",
        "verifier/signal",
        "feedback/implicit",
        "feedback/rating",
    }
)

#: The harness's extension namespace. Everything under it is about the run rather than part of it:
#: core's own audit rows (`x/core/*`) and plugin-declared rows (`x/plugin/*`) alike. Matched by
#: prefix because a plugin chooses its own event names, so no list can be complete.
EXTENSION_EVENT_PREFIX = "x/"


def is_conversation(event: Event) -> bool:
    """Whether a row is part of the state the model was in.

    Anything neither named below nor under the extension namespace counts as conversation. A type
    this module has never seen is therefore treated as state, which is the reading that surfaces a
    mistake: an unrecognised bookkeeping row makes two identical states look different, and that
    shows up at once as a pipeline finding no divergences — while the opposite default, silently
    ignoring a content row, would align two genuinely different states and teach from a comparison
    that was never true.

    The `x/` prefix is the one place the opposite default applies, because there the set is open by
    construction: a plugin's event name is its own, and a reader that waited to learn it would stop
    aligning the moment somebody installed one.
    """
    if event.type.startswith(EXTENSION_EVENT_PREFIX):
        return False
    return event.type not in BOOKKEEPING_TYPES


def _canonical(value: Any) -> str:
    """A stable spelling of a JSON-shaped value.

    ``sort_keys`` because two mappings that differ only in insertion order are the same value, and
    a digest that disagreed would report a divergence where the model did the same thing. Keys are
    sorted rather than the whole document being re-serialised by some other rule so the result is
    reproducible across Python versions.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True)
class Action:
    """What the model chose at one point."""

    kind: ActionKind
    name: str | None
    payload: Any

    def fingerprint(self) -> str:
        """The identity used to decide whether two decisions agree.

        For a tool call this is the tool's name and its arguments. The harness's own ``toolUseId``
        is deliberately excluded: it is minted per call, so two runs that made the identical choice
        would carry different ids and every choice would look like a divergence.
        """
        if self.kind == "tool_call":
            return _canonical({"tool": self.name, "args": self.payload})
        return _canonical({"message": self.payload})

    def canonical(self) -> str:
        """The whole action, losslessly, for anything that has to store or label it.

        :meth:`describe` truncates because it is for a reader's eye; a dataset row built from it is
        a label with the end cut off, so two different teacher messages sharing a long prefix would
        be stored as the same label. This is the form to persist.
        """
        return _canonical(
            {"kind": self.kind, "name": self.name, "payload": self.payload}
        )

    def describe(self) -> str:
        """A short human-readable form, for a report or a log line. Never a stored label."""
        if self.kind == "tool_call":
            return f"{self.name}({_canonical(self.payload)})"
        text = self.payload if isinstance(self.payload, str) else _canonical(self.payload)
        return text if len(text) <= 200 else f"{text[:197]}..."


@dataclass(frozen=True)
class DecisionPoint:
    """One model decision, and the state it was made in."""

    seq: int
    ordinal: int
    prefix_digest: str
    action: Action
    #: The harness's own id for this call, kept for provenance and for joining a result back on.
    tool_use_id: str | None = None
    #: ``True`` when the matching ``tool/result`` reported an error, ``False`` when it succeeded and
    #: ``None`` when the run ended before a result was written. Recorded here because "the student
    #: chose badly" and "the tool failed" are different events and a gate that conflated them would
    #: teach the teacher's caution about a broken tool.
    errored: bool | None = None

    def key(self) -> tuple[str, int]:
        return (self.prefix_digest, self.ordinal)


@dataclass(frozen=True)
class Trajectory:
    """A session's decisions, with the provenance the co-harness claim needs."""

    session_key: str
    lane: str
    preset: str | None
    resolved_profile_hash: str | None
    model_settings: tuple[Any, ...]
    points: tuple[DecisionPoint, ...]
    last_seq: int
    event_types: tuple[str, ...]
    parent_key: str | None = None
    boundary_seq: int | None = None

    def by_prefix(self) -> dict[str, DecisionPoint]:
        """Index by prefix digest.

        One digest maps to one point. With the digest taken over the conversation this holds by
        construction — a decision is itself conversation, so the next decision in the same lane has
        it in its prefix and cannot share the digest. The check stays as an invariant because that
        is what it would be if it failed: a digest that stopped discriminating would align states
        that were never the same, and the corruption would be silent.
        """
        index: dict[str, DecisionPoint] = {}
        for point in self.points:
            existing = index.get(point.prefix_digest)
            if existing is not None and existing.action.fingerprint() != point.action.fingerprint():
                raise ValueError(
                    f"session {self.session_key} decided twice in one state "
                    f"(seq {existing.seq} and {point.seq}); refusing to index it"
                )
            index.setdefault(point.prefix_digest, point)
        return index

    def tool_calls(self) -> tuple[DecisionPoint, ...]:
        return tuple(p for p in self.points if p.action.kind == "tool_call")

    def prefix_digests(self) -> tuple[str, ...]:
        return tuple(p.prefix_digest for p in self.points)


def _start_event(events: Sequence[Event]) -> Event | None:
    for event in events:
        if event.type == "session/start":
            return event
    return None


def _action_of(event: Event) -> Action | None:
    """Read one event as a decision, or return ``None`` when it is not one.

    The payload shapes are read leniently on purpose. Agnes Harness is pre-alpha and has already
    changed its event vocabulary once; a reader that required an exact key set would stop producing
    trajectories on an upgrade, and the experiment's own record would be the thing that broke. A
    tool call missing a name is still reported — as a nameless call — because dropping it would
    make the two arms disagree about how many decisions a run contained.
    """
    data = event.data if isinstance(event.data, dict) else {}

    if event.type == "tool/call":
        return Action(kind="tool_call", name=data.get("name"), payload=data.get("args"))

    if event.type == "assistant/message":
        if "content" in data:
            blocks = data.get("content")
            texts = []
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        texts.append(block["text"])
            joined = "\n".join(texts).strip()
            if joined:
                return Action(kind="message", name=None, payload=joined)
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            return Action(kind="message", name=None, payload=text)

    return None


def prefix_chain(events: Iterable[Event]) -> dict[int, str]:
    """The prefix digest in force *before* each event, keyed by that event's seq.

    A chain rather than a hash of the whole prefix: the harness's ledger is append-only and a run
    can hold thousands of rows, so rehashing a growing prefix per decision would make reading a
    long trajectory quadratic for no benefit.

    Every row gets an entry — a decision point needs the digest in force at its own seq — but only a
    conversation row advances the chain. A batch of bookkeeping rows therefore leaves the digest
    where it was, which is what lets a forked child meet its parent at the same state.

    ``sha256`` of the previous digest plus the row's own type and data. The row's ``ts`` and ``id``
    stay out: both are per-run values that carry no state, and including them would make every
    prefix unique so that nothing ever aligned.
    """
    digests: dict[int, str] = {}
    running = hashlib.sha256(b"chtd/prefix/v2").digest()
    for event in events:
        digests[event.seq] = running.hex()
        if not is_conversation(event):
            continue
        running = hashlib.sha256(
            running + event.type.encode("utf-8") + b"\x00" + _canonical(event.data).encode("utf-8")
        ).digest()
    return digests


def decision_points(events: Sequence[Event], *, lane: str = "main") -> tuple[DecisionPoint, ...]:
    """Every decision the model made in one lane, in order."""
    lane_events = [e for e in events if e.lane == lane]
    prefixes = prefix_chain(lane_events)

    errored_by_call: dict[str, bool] = {}
    for event in lane_events:
        if event.type == "tool/result" and isinstance(event.data, dict):
            call_id = event.data.get("toolUseId")
            if isinstance(call_id, str):
                errored_by_call[call_id] = bool(event.data.get("isError"))

    points: list[DecisionPoint] = []
    ordinal = 0
    for event in lane_events:
        action = _action_of(event)
        if action is None:
            continue
        data = event.data if isinstance(event.data, dict) else {}
        call_id = data.get("toolUseId") if isinstance(data.get("toolUseId"), str) else None
        points.append(
            DecisionPoint(
                seq=event.seq,
                ordinal=ordinal,
                prefix_digest=prefixes[event.seq],
                action=action,
                tool_use_id=call_id,
                errored=errored_by_call.get(call_id) if call_id is not None else None,
            )
        )
        ordinal += 1
    return tuple(points)


def read_trajectory(ledger: Ledger, session_key: str, *, lane: str = "main") -> Trajectory:
    """Read one session as a trajectory.

    The provenance fields come from the session's own ``session/start`` row, read from the ledger
    rather than supplied by the caller: a rollout that was described from outside could describe
    itself as anything, and the comparison the experiment rests on is between what two ledgers say.
    """
    events = ledger.events(session_key)
    if not events:
        raise ValueError(f"session {session_key} has no events in {ledger.path}")

    start = _start_event(events)
    start_data = start.data if start is not None and isinstance(start.data, dict) else {}
    parent = start_data.get("parent") if isinstance(start_data.get("parent"), dict) else {}
    settings = start_data.get("modelSettings")

    sessions = {row.session_key: row for row in ledger.sessions()}
    row = sessions.get(session_key)

    return Trajectory(
        session_key=session_key,
        lane=lane,
        preset=start_data.get("preset") if isinstance(start_data.get("preset"), str) else None,
        resolved_profile_hash=(
            start_data.get("resolvedProfileHash")
            if isinstance(start_data.get("resolvedProfileHash"), str)
            else None
        ),
        model_settings=tuple(settings) if isinstance(settings, list) else (),
        points=decision_points(events, lane=lane),
        last_seq=ledger.last_seq(session_key),
        event_types=ledger.event_types(session_key),
        parent_key=(row.parent_key if row is not None else None) or parent.get("key"),
        boundary_seq=(row.boundary_seq if row is not None else None) or parent.get("boundarySeq"),
    )
