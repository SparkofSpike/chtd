"""The CHTD arm must be able to start from the control collection as shipped.

`--expected-track` was introduced after the first control collection, so the
committed manifest carries the track only inside its per-task records. These
tests pin the fallback the readers use, the refusal when records disagree, and
the shipped collection itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.collect_baseline import expected_track, observed_tracks

TRACK = "d403e1e12186a4021cc9d20bbc2c8770cf8366e9fccfc67621a64fde307967ba"


def _task(run: Path, task_id: str, *, track: str | None) -> None:
    (run / "tasks").mkdir(parents=True, exist_ok=True)
    record = {"track_sha256": track} if track is not None else None
    (run / "tasks" / f"{task_id}.json").write_text(
        json.dumps({"task_id": task_id, "record": record}), encoding="utf-8"
    )


def test_expected_track_prefers_the_manifest_record() -> None:
    manifest = {"expected_track_sha256": "a" * 64}
    assert expected_track(manifest, ["b" * 64]) == "a" * 64


def test_expected_track_falls_back_to_the_records(tmp_path: Path) -> None:
    run = tmp_path / "collection"
    _task(run, "one", track="b" * 64)
    _task(run, "two", track="b" * 64)
    _task(run, "three", track=None)
    tracks = observed_tracks(run, ["one", "two", "three"])
    assert tracks == {"b" * 64}
    assert expected_track({}, tracks) == "b" * 64


def test_expected_track_refuses_records_that_disagree() -> None:
    with pytest.raises(ValueError, match="distinct tracks"):
        expected_track({}, ["b" * 64, "c" * 64])


def test_expected_track_refuses_when_no_record_names_a_track(tmp_path: Path) -> None:
    run = tmp_path / "collection"
    _task(run, "one", track=None)
    with pytest.raises(ValueError, match="no expected track"):
        expected_track({}, observed_tracks(run, ["one"]))


def test_the_shipped_control_collection_yields_its_track() -> None:
    """The committed collection predates `--expected-track`; the readers must
    still resolve its track from the records, or the CHTD arm cannot start."""
    run = Path(__file__).resolve().parent.parent / "runs" / "baseline-20261010-a" / "collection"
    manifest = json.loads((run / "collection-manifest.json").read_text(encoding="utf-8"))
    assert "expected_track_sha256" not in manifest
    tracks = observed_tracks(run, manifest["task_ids"])
    assert tracks == {TRACK}
    assert expected_track(manifest, tracks) == TRACK
