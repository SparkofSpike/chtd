"""Recheck saved answer text with the current automatic diagnosis gate.

This does not replace a raw-ledger rerun or human correctness review. It makes discrepancies
between a published manual rescore and the executable scorer visible by task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.tasks.check_diagnosis import _spells, fixture_files, invented_files


def audit(artifact: dict, tasks: dict[str, dict], workspaces: Path) -> dict:
    arms = {}
    for arm, entry in artifact["arms"].items():
        rows = []
        for task_id, result in entry["per_task"].items():
            task = tasks[task_id]
            answer = result["answer"].lower()
            missing = [phrase for phrase in task["expect"] if not _spells(phrase, answer)]
            forbidden = [phrase for phrase in task.get("forbid", [])
                         if _spells(phrase, answer)]
            invented = invented_files(answer, fixture_files(str(workspaces / task_id)))
            automatic_pass = not (missing or forbidden or invented)
            rows.append({
                "task_id": task_id,
                "automatic_pass": automatic_pass,
                "reported_rescore": result["passed_rescored"],
                "missing": missing,
                "forbidden": forbidden,
                "invented_files": invented,
            })
        arms[arm] = {
            "automatic_passes": sum(row["automatic_pass"] for row in rows),
            "reported_passes": entry["passed_rescored"],
            "rows": rows,
        }
    return {"scope": "saved answers, current automatic gate, not human truth", "arms": arms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--tasks", type=Path, default=Path("harness/tasks/final-holdout.jsonl"))
    parser.add_argument("--workspaces", type=Path, default=Path("data/tasks"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tasks = {row["id"]: row for line in args.tasks.read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#") for row in [json.loads(line)]}
    missing_workspaces = [task_id for task_id in tasks
                          if not (args.workspaces / task_id).is_dir()]
    if missing_workspaces:
        parser.error(f"missing task workspaces: {missing_workspaces}")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    report = audit(artifact, tasks, args.workspaces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    for arm, result in report["arms"].items():
        mismatches = [row["task_id"] for row in result["rows"]
                      if row["automatic_pass"] != row["reported_rescore"]]
        print(f"{arm}: automatic={result['automatic_passes']} "
              f"reported={result['reported_passes']} mismatches={mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
