"""Turn gated divergences into the rows a fine-tune reads, with the provenance to re-derive them.

Two properties are enforced here rather than documented and hoped for:

**One track.** Every rollout record that feeds a dataset must carry the same ``track_sha256``. Two
runs on different tracks are two experiments, and a dataset that mixed them would be measuring
something nobody chose. The check refuses rather than warns.

**Labels, not privileges.** The design keeps the teacher's extra information out of the student's
input: the teacher may see outcomes and verifier results, and those may decide *whether* a row is
admitted, but they must never become part of what the student is asked to predict. That is enforced
structurally — an entry has no field that could carry them — and tested.

The state a row refers to is a prefix digest, not a copy of the conversation. The training step
resolves it against the ledger, which keeps the transcript in one place and makes every row
traceable to the run it came from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from distill.divergence import Divergence
from distill.gate import Admission, SignalGate
from harness.rollout import RolloutRecord

__all__ = [
    "DatasetEntry",
    "DatasetManifest",
    "DatasetRefusal",
    "build_dataset",
    "dataset_digest",
    "mixed_tracks",
    "write_dataset",
]


class MixedTracksError(ValueError):
    """Rollouts from more than one track were offered to one dataset."""

    def __init__(self, tracks: Sequence[str]) -> None:
        self.tracks = tuple(tracks)
        super().__init__(
            "refusing to build a dataset from more than one track: "
            + ", ".join(t[:12] for t in self.tracks)
            + ". Two tracks are two experiments; compare them, do not merge them."
        )


@dataclass(frozen=True)
class DatasetEntry:
    """One training row: a state, and the decision the teacher made in it.

    Deliberately narrow. There is no field for an outcome, a verifier verdict or a reference
    solution: those are gate evidence, they decide admission, and a field that could hold them would
    eventually hold them into the student's input.
    """

    task_id: str
    prefix_digest: str
    student_session: str
    teacher_session: str
    student_seq: int
    teacher_seq: int
    student_action: str
    teacher_action: str
    teacher_action_fingerprint: str

    @classmethod
    def from_divergence(cls, task_id: str, divergence: Divergence) -> "DatasetEntry":
        return cls(
            task_id=task_id,
            prefix_digest=divergence.prefix_digest,
            student_session=divergence.student_session,
            teacher_session=divergence.teacher_session,
            student_seq=divergence.student.seq,
            teacher_seq=divergence.teacher.seq,
            student_action=divergence.student.action.describe(),
            teacher_action=divergence.teacher.action.describe(),
            teacher_action_fingerprint=divergence.teacher.action.fingerprint(),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class DatasetRefusal:
    """A divergence the gate kept out, with the reason — kept so the write-up can count them."""

    task_id: str
    prefix_digest: str
    reason: str


@dataclass(frozen=True)
class DatasetManifest:
    """What the dataset is, so a reader can tell whether a later run reproduced it."""

    track_sha256: str
    dataset_sha256: str
    entries: int
    refusals: int
    tasks: tuple[str, ...]
    arms: tuple[str, ...]
    created_at: str
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, indent=2)


@dataclass
class Dataset:
    entries: tuple[DatasetEntry, ...] = ()
    refusals: tuple[DatasetRefusal, ...] = ()
    manifest: DatasetManifest | None = None
    reasons: dict[str, int] = field(default_factory=dict)


def mixed_tracks(records: Iterable[RolloutRecord]) -> tuple[str, ...]:
    """The distinct track digests among these records, sorted. Empty or one means comparable."""
    return tuple(sorted({record.track_sha256 for record in records}))


def dataset_digest(entries: Sequence[DatasetEntry]) -> str:
    """A digest over the rows in order.

    Over the canonical JSON of each entry, joined with newlines, so that the digest changes if a row
    is added, removed, reordered or altered. The manifest records it and a reproduction compares it.
    """
    hasher = hashlib.sha256()
    hasher.update(b"chtd/dataset/v1\n")
    for entry in entries:
        hasher.update(entry.to_json().encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def build_dataset(
    *,
    task_ids: Sequence[str],
    alignments: Mapping[str, object],
    gate: SignalGate,
    records: Sequence[RolloutRecord],
    arms: Sequence[str],
    now: datetime | None = None,
) -> Dataset:
    """Build the dataset, refusing to mix tracks.

    ``alignments`` maps a task id to an ``Alignment``; it is typed loosely here to keep this module
    from importing the trajectory model for a type annotation alone.
    """
    tracks = mixed_tracks(records)
    if len(tracks) > 1:
        raise MixedTracksError(tracks)
    if not tracks:
        raise ValueError("no rollout records; a dataset cannot say what track it came from")

    entries: list[DatasetEntry] = []
    refusals: list[DatasetRefusal] = []
    reasons: dict[str, int] = {}

    for task_id in task_ids:
        alignment = alignments.get(task_id)
        if alignment is None:
            continue
        admitted, refused = gate.filter(getattr(alignment, "divergences"))
        for divergence in admitted:
            entries.append(DatasetEntry.from_divergence(task_id, divergence))
        for admission in refused:
            refusals.append(
                DatasetRefusal(
                    task_id=task_id,
                    prefix_digest=admission.divergence.prefix_digest,
                    reason=admission.reason,
                )
            )
            head = admission.reason.split(":", 1)[0]
            reasons[head] = reasons.get(head, 0) + 1

    created = (now or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z")
    manifest = DatasetManifest(
        track_sha256=tracks[0],
        dataset_sha256=dataset_digest(entries),
        entries=len(entries),
        refusals=len(refusals),
        tasks=tuple(task_ids),
        arms=tuple(arms),
        created_at=created,
    )
    return Dataset(entries=tuple(entries), refusals=tuple(refusals), manifest=manifest, reasons=reasons)


def write_dataset(dataset: Dataset, directory: str | Path) -> dict[str, Path]:
    """Write ``entries.jsonl``, ``refusals.jsonl`` and ``manifest.json``.

    The refusals are written out, not dropped. A dataset handed to a reviewer without them cannot be
    checked for whether it is small because the method is selective or small because the gate
    rejected everything.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    entries_path = target / "entries.jsonl"
    with entries_path.open("w", encoding="utf-8") as handle:
        for entry in dataset.entries:
            handle.write(entry.to_json() + "\n")

    refusals_path = target / "refusals.jsonl"
    with refusals_path.open("w", encoding="utf-8") as handle:
        for refusal in dataset.refusals:
            handle.write(json.dumps(asdict(refusal), sort_keys=True, ensure_ascii=False) + "\n")

    manifest_path = target / "manifest.json"
    if dataset.manifest is None:
        raise ValueError("dataset has no manifest; build it before writing it")
    manifest_path.write_text(dataset.manifest.to_json() + "\n", encoding="utf-8")

    return {"entries": entries_path, "refusals": refusals_path, "manifest": manifest_path}
