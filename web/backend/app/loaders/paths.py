"""Unified path resolution for CHTD data and run ledgers (read-only)."""
from __future__ import annotations

from pathlib import Path

from .. import config


class Paths:
    """Resolved locations under CHTD_ROOT. All paths are treated read-only."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else config.CHTD_ROOT

    # -- convenience roots -------------------------------------------------
    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def tasks_dir(self) -> Path:
        return self.root / "harness" / "tasks"

    # -- holdout-4 v3 trajectory ledger ------------------------------------
    # The Replay data ships with the demo under web/demo-data/ so a fresh clone runs without the
    # full (gitignored) data/ tree; a working copy that has the original layout still resolves.
    @property
    def v3_dir(self) -> Path:
        bundled = self.root / "web" / "demo-data" / "holdout-4-20261014-v3"
        if bundled.is_dir():
            return bundled
        return self.data / "holdout-4-20261014-v3"

    @property
    def v3_rollouts(self) -> Path:
        return self.v3_dir / "rollouts"

    @property
    def v3_events_db(self) -> Path:
        return self.v3_dir / "agh_home" / "data" / "sessions.db"

    # -- demo rollout ------------------------------------------------------
    @property
    def demo_rollout(self) -> Path:
        return self.data / "rollouts" / "demo-20261015" / "records.jsonl"

    # -- baseline collection (26 sessions, second event DB) ----------------
    @property
    def baseline_events_db(self) -> Path:
        return self.runs / "baseline-20261010-a" / "collection" / "sessions.db"

    # -- runs json / md sources --------------------------------------------
    def holdout4(self, name: str) -> Path:
        return self.runs / "holdout-4-20261014" / name

    def holdout3(self, name: str) -> Path:
        return self.runs / "holdout-3-20261014" / name

    def machine_score(self, name: str) -> Path:
        return self.runs / "holdout-3-20261014" / "machine-scores" / name

    def teacher_state(self, name: str) -> Path:
        return self.runs / "teacher-state-20261014" / name

    def chtd_early(self, name: str) -> Path:
        return self.runs / "chtd-20261011-a" / name

    # -- task sets ---------------------------------------------------------
    TASK_SETS = ["error-recovery", "final-holdout", "holdout-3", "example"]

    def task_file(self, name: str) -> Path:
        return self.tasks_dir / f"{name}.jsonl"


PATHS = Paths()
