"""Build ``RunMetrics`` rows for tests without going through a ledger every time."""

from __future__ import annotations

from eval.metrics import FAILED, SUCCEEDED, UNKNOWN_OUTCOME, RunMetrics

__all__ = ["run", "FAILED", "SUCCEEDED", "UNKNOWN_OUTCOME"]


def run(
    task_id: str,
    arm: str,
    *,
    outcome: str = UNKNOWN_OUTCOME,
    decisions: int = 3,
    tool_calls: int = 2,
    messages: int = 1,
    failed_calls: int = 0,
    events: int = 8,
    session_key: str | None = None,
) -> RunMetrics:
    return RunMetrics(
        task_id=task_id,
        arm=arm,
        session_key=session_key or f"{arm}-{task_id}",
        outcome=outcome,
        decisions=decisions,
        tool_calls=tool_calls,
        messages=messages,
        failed_calls=failed_calls,
        events=events,
    )
