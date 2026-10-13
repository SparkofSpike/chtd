"""The matched split must be the reviewed split, not whatever rows happen to be on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.select_matched import select
from distill.sft import samples_digest


def _write_arm(root: Path, arm: str, *, holdout: list[str] | None = None) -> tuple[Path, Path]:
    rows = [{
        "task_id": "train-1",
        "prefix_digest": f"{arm}-state",
        "harness_prefix_digest": "harness-prefix",
        "messages": [{"role": "assistant", "content": f"{arm} answer"}],
    }]
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    sample_path = root / f"{arm}.jsonl"
    sample_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    manifest_path = root / f"{arm}-manifest.json"
    manifest_path.write_text(json.dumps({
        "track_sha256": "a" * 64,
        "task_ids": ["train-1"],
        "holdout_ids": holdout if holdout is not None else ["hold-1"],
        "harness_prefix_digest": "harness-prefix",
        "samples_sha256": samples_digest(lines),
    }), encoding="utf-8")
    return sample_path, manifest_path


def test_select_refuses_sample_file_that_disagrees_with_manifest(tmp_path: Path) -> None:
    baseline, base_manifest = _write_arm(tmp_path, "baseline")
    chtd, chtd_manifest = _write_arm(tmp_path, "chtd")
    chtd.write_text(chtd.read_text(encoding="utf-8").replace("chtd answer", "altered answer"),
                    encoding="utf-8")

    with pytest.raises(ValueError, match="CHTD samples do not match"):
        select(baseline, base_manifest, chtd, chtd_manifest, tmp_path / "out")


def test_select_refuses_different_holdout_splits(tmp_path: Path) -> None:
    baseline, base_manifest = _write_arm(tmp_path, "baseline")
    chtd, chtd_manifest = _write_arm(tmp_path, "chtd", holdout=["other-holdout"])

    with pytest.raises(ValueError, match="different train or holdout splits"):
        select(baseline, base_manifest, chtd, chtd_manifest, tmp_path / "out")


def test_select_records_the_requested_output_root(tmp_path: Path) -> None:
    baseline, base_manifest = _write_arm(tmp_path, "baseline")
    chtd, chtd_manifest = _write_arm(tmp_path, "chtd")
    output_root = "/mnt/proj/chtd-experiments/unique-run"

    report = select(baseline, base_manifest, chtd, chtd_manifest, tmp_path / "out",
                    output_root=output_root)

    assert report["task_ids"] == ["train-1"]
    for arm in ("baseline", "chtd"):
        plan = json.loads((tmp_path / "out" / f"{arm}-plan.json").read_text(encoding="utf-8"))
        assert plan["output_dir"] == f"{output_root}/{arm}-adapter"
