"""Find the places the student and the teacher chose differently in one state.

A divergence is a pair of decision points with the same prefix digest and different actions. That
definition is short because the work went into the prefix digest (`harness.trajectory`): once two
decisions can be shown to stand in the same state, comparing them is trivial, and if they cannot be
shown to, the comparison is not worth making.

Extraction reports what it could not pair as well as what it could. A student decision the teacher
never answered and a teacher decision in a state the student never visited are different facts, and
a pipeline that folded them into "no divergence" would overstate how much of the run it covered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from harness.trajectory import DecisionPoint, Trajectory

__all__ = ["Alignment", "Divergence", "align", "divergence_rate"]


@dataclass(frozen=True)
class Divergence:
    """One state, two choices."""

    prefix_digest: str
    student: DecisionPoint
    teacher: DecisionPoint
    #: Which run each side came from. Carried on the divergence rather than looked up by the
    #: caller, because a branch outcome is a fact about one run and a gate that had to guess the
    #: session key would eventually guess wrong.
    student_session: str
    teacher_session: str

    def describe(self) -> str:
        return f"{self.student.action.describe()}  ->  {self.teacher.action.describe()}"


@dataclass(frozen=True)
class Alignment:
    """The full result of comparing two trajectories, including what did not line up."""

    student_session: str
    teacher_session: str
    #: Same state, different choice. The signal.
    divergences: tuple[Divergence, ...]
    #: Same state, same choice. Not signal, and not nothing: a run that agrees everywhere is a
    #: statement about the student, and its size belongs next to the divergence count.
    agreements: tuple[tuple[DecisionPoint, DecisionPoint], ...]
    #: The student decided and the teacher did not answer there. Coverage gap, reported rather than
    #: hidden: silently treating these as agreement is how a thin dataset looks complete.
    unanswered: tuple[DecisionPoint, ...]
    #: The teacher decided in a state the student never reached. Expected at the tail of a fork —
    #: the teacher's own run goes on past the boundary — and not signal either.
    novel: tuple[DecisionPoint, ...]

    @property
    def student_decisions(self) -> int:
        return len(self.divergences) + len(self.agreements) + len(self.unanswered)


def align(student: Trajectory, teacher: Trajectory) -> Alignment:
    """Pair two trajectories by state and report every way they match, differ, or fail to meet.

    The teacher trajectory is normally a fork of the student one, so it carries the student's prefix
    and then its own continuation. Pairing on the prefix digest rather than on position is what lets
    the two be compared at all: the fork writes rows of its own at the boundary, and a positional
    comparison would slide out of step there and stay out.
    """
    teacher_by_prefix = teacher.by_prefix()

    divergences: list[Divergence] = []
    agreements: list[tuple[DecisionPoint, DecisionPoint]] = []
    unanswered: list[DecisionPoint] = []
    matched: set[str] = set()

    for point in student.points:
        answer = teacher_by_prefix.get(point.prefix_digest)
        if answer is None:
            unanswered.append(point)
            continue
        matched.add(point.prefix_digest)
        if point.action.fingerprint() == answer.action.fingerprint():
            agreements.append((point, answer))
        else:
            divergences.append(
                Divergence(
                    prefix_digest=point.prefix_digest,
                    student=point,
                    teacher=answer,
                    student_session=student.session_key,
                    teacher_session=teacher.session_key,
                )
            )

    novel = tuple(
        point for digest, point in teacher_by_prefix.items() if digest not in matched
    )
    return Alignment(
        student_session=student.session_key,
        teacher_session=teacher.session_key,
        divergences=tuple(divergences),
        agreements=tuple(agreements),
        unanswered=tuple(unanswered),
        novel=novel,
    )


def align_all(student: Trajectory, teachers: Iterable[Trajectory]) -> Alignment:
    """Align one student run against several teacher forks and merge the results.

    A DAgger round forks once per decision point, so the teacher's answers arrive as many small
    sessions rather than one long one. Merging them here keeps the "one trajectory per side" shape
    the rest of the pipeline uses.

    Raises if two teacher sessions answer the same state differently: that is two labels for one
    row, and a dataset that picked one silently would be teaching a coin flip.
    """
    merged = Alignment(
        student_session=student.session_key,
        teacher_session="+".join(t.session_key for t in teachers),
        divergences=(),
        agreements=(),
        unanswered=(),
        novel=(),
    )
    by_state: dict[str, Divergence] = {}
    agreements: list[tuple[DecisionPoint, DecisionPoint]] = []
    answered: set[str] = set()
    novel: dict[str, DecisionPoint] = {}

    for teacher in teachers:
        part = align(student, teacher)
        for divergence in part.divergences:
            existing = by_state.get(divergence.prefix_digest)
            if existing is not None:
                if existing.teacher.action.fingerprint() != divergence.teacher.action.fingerprint():
                    raise ValueError(
                        f"two teacher sessions gave different answers in state "
                        f"{divergence.prefix_digest[:12]}; refusing to choose one"
                    )
                continue
            by_state[divergence.prefix_digest] = divergence
        agreements.extend(part.agreements)
        answered.update(d.prefix_digest for d in part.divergences)
        answered.update(p.prefix_digest for p, _ in part.agreements)
        for point in part.novel:
            novel.setdefault(point.prefix_digest, point)

    unanswered = tuple(p for p in student.points if p.prefix_digest not in answered)
    return Alignment(
        student_session=merged.student_session,
        teacher_session=merged.teacher_session,
        divergences=tuple(by_state[d] for d in sorted(by_state)),
        agreements=tuple(agreements),
        unanswered=unanswered,
        novel=tuple(novel[d] for d in sorted(novel)),
    )


def divergence_rate(alignment: Alignment) -> float:
    """Divergences as a share of the student decisions the teacher answered.

    Against answered decisions rather than against all of them: a fork that covered only part of a
    run would otherwise look like a well-behaved student, and the coverage gap is already reported
    separately as ``unanswered``.
    """
    answered = len(alignment.divergences) + len(alignment.agreements)
    if answered == 0:
        return 0.0
    return len(alignment.divergences) / answered
