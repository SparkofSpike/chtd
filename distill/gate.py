"""The signal-quality gate: a divergence is not automatically a place where the teacher was right.

Without this step the dataset teaches the teacher's quirks along with its capability, and nothing
downstream can tell which it learned. The gate's job is to admit only the divergences whose
teacher-side choice the outcome validated, and — just as importantly — to say *why* each row it
refused was refused.

Every validator is fail-closed. An unknown outcome refuses rather than admits, because the cost of
the two mistakes is not symmetric: a refused row loses a little signal that the write-up can count,
while an admitted row that should not have been there is indistinguishable, afterwards, from one
that should.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from distill.divergence import Divergence

__all__ = [
    "Admission",
    "Outcome",
    "OutcomeTable",
    "OutcomeValidator",
    "RecordedVerdicts",
    "SignalGate",
    "Validator",
    "Verdict",
]


class Outcome(str, Enum):
    """What happened on a branch that started from a state."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _as_outcome(value: object) -> Outcome:
    """Coerce a stored outcome, refusing anything outside the vocabulary.

    A value that is not one of the three is an error and not a fourth state. Treating it as
    ``UNKNOWN`` would be fail-closed and still wrong: an unrecognised word usually means the judge
    and the gate disagree about spelling, and folding it into "no recorded outcome" would send the
    reader to look for a missing row instead of a mis-spelled one.
    """
    if isinstance(value, Outcome):
        return value
    try:
        return Outcome(value)
    except ValueError as error:
        raise ValueError(
            f"{value!r} is not an outcome; expected one of "
            f"{', '.join(repr(o.value) for o in Outcome)}"
        ) from error


@dataclass(frozen=True)
class Verdict:
    admitted: bool
    reason: str
    by: str


@dataclass(frozen=True)
class Admission:
    """One divergence and the verdicts passed on it."""

    divergence: Divergence
    admitted: bool
    verdicts: tuple[Verdict, ...]

    @property
    def reason(self) -> str:
        """The first refusal, which is the one that decided it."""
        for verdict in self.verdicts:
            if not verdict.admitted:
                return f"{verdict.by}: {verdict.reason}"
        return "; ".join(f"{v.by}: {v.reason}" for v in self.verdicts)


class Validator(Protocol):
    """One reason to admit or refuse a divergence."""

    name: str

    def assess(self, divergence: Divergence) -> Verdict: ...


@dataclass
class OutcomeTable:
    """Branch outcomes, keyed by the run they belong to and the state they started from.

    A key is ``(session_key, prefix_digest)`` and not the state alone: the same state reached in two
    runs is two branches, and the student's branch failing says nothing about the teacher's.

    Values are coerced to :class:`Outcome` on the way in. They used to be stored as handed over,
    and the validator identified them with ``is`` — so a table holding plain strings (which is what
    a rollout record carries, and what this module's own ``Outcome.SUCCEEDED.value`` produces)
    matched none of the three cases and fell through to the admitting default. Every divergence was
    admitted, including the two the validator exists to refuse.
    """

    outcomes: Mapping[tuple[str, str], Outcome] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.outcomes = {
            (session_key, prefix_digest): _as_outcome(value)
            for (session_key, prefix_digest), value in self.outcomes.items()
        }

    def get(self, session_key: str, prefix_digest: str) -> Outcome:
        return self.outcomes.get((session_key, prefix_digest), Outcome.UNKNOWN)

    def record(self, session_key: str, prefix_digest: str, outcome: Outcome) -> None:
        if not isinstance(self.outcomes, dict):
            raise TypeError("this table was built from a read-only mapping")
        self.outcomes[(session_key, prefix_digest)] = _as_outcome(outcome)  # type: ignore[index]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> "OutcomeTable":
        """Build from rollout records carrying ``session_key``, ``prefix_digest`` and ``outcome``."""
        table: dict[tuple[str, str], Outcome] = {}
        for record in records:
            session_key = record.get("session_key")
            prefix_digest = record.get("prefix_digest")
            outcome = record.get("outcome")
            if not isinstance(session_key, str) or not isinstance(prefix_digest, str):
                raise ValueError("an outcome record needs session_key and prefix_digest as strings")
            table[(session_key, prefix_digest)] = _as_outcome(outcome)
        return cls(table)


@dataclass(frozen=True)
class OutcomeValidator:
    """Admit when the teacher's branch succeeded and the student's did not.

    Both halves are needed. The teacher succeeding says its choice worked; it does not say the
    student needed help there, and a divergence where both succeeded is a difference of style. The
    student failing is what makes the state one where supervision is worth spending.
    """

    outcomes: OutcomeTable
    name: str = "outcome"

    def assess(self, divergence: Divergence) -> Verdict:
        teacher = self.outcomes.get(divergence.teacher_session, divergence.prefix_digest)
        student = self.outcomes.get(divergence.student_session, divergence.prefix_digest)

        if teacher is Outcome.UNKNOWN or student is Outcome.UNKNOWN:
            missing = [
                side
                for side, value in (("teacher", teacher), ("student", student))
                if value is Outcome.UNKNOWN
            ]
            return Verdict(
                False,
                f"no recorded outcome for the {' and '.join(missing)} branch; "
                "an unvalidated divergence is not evidence",
                self.name,
            )
        if teacher is Outcome.FAILED:
            return Verdict(
                False,
                "the teacher's own choice did not succeed from this state; "
                "there is nothing here to teach",
                self.name,
            )
        if student is Outcome.SUCCEEDED:
            return Verdict(
                False,
                "the student's choice succeeded from this state; "
                "this is a difference of style, not a failure to correct",
                self.name,
            )
        return Verdict(True, "the teacher's choice succeeded where the student's did not", self.name)


@dataclass(frozen=True)
class RecordedVerdicts:
    """Judgements recorded elsewhere — by a human reading the pair, or by a verifier.

    Kept as a separate validator rather than folded into the outcome one because they answer
    different questions: the outcome asks whether the choice worked, and a reader or a verifier can
    admit a divergence whose outcome was never measured. A row with no recorded judgement is
    refused, so adding this validator can only ever narrow the set.

    Values must be booleans. ``from_path`` has always checked that; the dataclass did not, so a
    hand-built instance holding the string ``"false"`` admitted what it was meant to refuse —
    truthiness is not a verdict.
    """

    verdicts: Mapping[str, bool]
    name: str = "recorded"

    def __post_init__(self) -> None:
        for digest, admitted in self.verdicts.items():
            if not isinstance(admitted, bool):
                raise ValueError(
                    f"verdict for {digest[:12]} is {admitted!r}, not a boolean; "
                    "a truthy string is not an admission"
                )

    def assess(self, divergence: Divergence) -> Verdict:
        if divergence.prefix_digest not in self.verdicts:
            return Verdict(False, "no recorded judgement for this state", self.name)
        if self.verdicts[divergence.prefix_digest] is True:
            return Verdict(True, "a reader or verifier admitted this pair", self.name)
        return Verdict(False, "a reader or verifier refused this pair", self.name)

    @classmethod
    def from_path(cls, path: str | Path) -> "RecordedVerdicts":
        """Read a JSONL file of ``{"prefix_digest": ..., "admitted": true|false}`` rows."""
        verdicts: dict[str, bool] = {}
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                try:
                    row = json.loads(text)
                except ValueError as error:
                    raise ValueError(f"{source}:{number} is not JSON: {error}") from error
                digest = row.get("prefix_digest")
                admitted = row.get("admitted")
                if not isinstance(digest, str) or not isinstance(admitted, bool):
                    raise ValueError(
                        f"{source}:{number} needs a string prefix_digest and a boolean admitted"
                    )
                verdicts[digest] = admitted
        return cls(verdicts)


@dataclass(frozen=True)
class SignalGate:
    """Every validator must admit a divergence before the dataset will use it.

    Conjunction rather than a first-that-decides chain: a set of checks that could override each
    other would make the pipeline's behaviour depend on the order they happened to be listed in.

    The validators are materialised in ``__post_init__``. They used to be kept as handed over, and
    the emptiness check was a truthiness test — so an iterator was always truthy, an empty one was
    accepted, and a one-shot iterator was consumed by the first ``assess``, after which every later
    call admitted with no verdicts at all. A gate that stops checking after one use is worse than no
    gate, because it reports zero refusals.
    """

    validators: Sequence[Validator]

    def __post_init__(self) -> None:
        # `object.__setattr__` because the dataclass is frozen; the alternative is a mutable gate.
        materialised = tuple(self.validators)
        if not materialised:
            raise ValueError(
                "a gate with no validators admits everything, which is the behaviour it exists to "
                "prevent; pass at least one"
            )
        object.__setattr__(self, "validators", materialised)

    def assess(self, divergence: Divergence) -> Admission:
        verdicts = tuple(validator.assess(divergence) for validator in self.validators)
        return Admission(
            divergence=divergence,
            admitted=all(verdict.admitted for verdict in verdicts),
            verdicts=verdicts,
        )

    def filter(self, divergences: Iterable[Divergence]) -> tuple[tuple[Divergence, ...], tuple[Admission, ...]]:
        """Split into what the gate admitted and every refusal, with its reason."""
        admitted: list[Divergence] = []
        refused: list[Admission] = []
        for divergence in divergences:
            assessment = self.assess(divergence)
            if assessment.admitted:
                admitted.append(divergence)
            else:
                refused.append(assessment)
        return tuple(admitted), tuple(refused)
