"""The sweep must apply the same gates as the task runner and retain raw evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import holdout_sweep


def test_score_passes_forbidden_phrases_and_workspace(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="passed")

    monkeypatch.setattr(holdout_sweep.subprocess, "run", fake_run)
    ledger = tmp_path / "sessions.db"
    workspace = tmp_path / "student"
    assert holdout_sweep.score(ledger, "session", ["expected"], ["wrong"], workspace) == (
        True, "passed"
    )
    argv, kwargs = calls[0]
    assert argv[-1] == "expected"
    assert kwargs["env"]["CHTD_LEDGER"] == str(ledger)
    assert kwargs["env"]["CHTD_SESSION_KEY"] == "session"
    assert json.loads(kwargs["env"]["CHTD_FORBID"]) == ["wrong"]
    assert kwargs["env"]["CHTD_WORKSPACE"] == str(workspace)


def test_rerun_refuses_to_delete_an_existing_ledger(tmp_path: Path) -> None:
    root = tmp_path / "rollouts"
    base = root / "base" / "fix-example"
    base.mkdir(parents=True)
    ledger = base / "sessions.db"
    ledger.write_bytes(b"raw evidence")

    with pytest.raises(FileExistsError, match="preserve its raw ledger"):
        holdout_sweep.run_one("base", 8000, {"id": "fix-example"}, root)

    assert ledger.read_bytes() == b"raw evidence"
