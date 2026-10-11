"""Build an exploratory, reviewed CHTD run with a shared first-tool anchor.

In first mode each accepted task contributes two last-turn SFT rows: the student's first ``ls``
call and the teacher's reviewed diagnosis at the student-reached state. In all-tools mode the
student's actual tool calls are separate labels. Both modes explicitly record provenance; neither
is an all-teacher trajectory or a controlled comparison against a teacher-own arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from distill.sft import samples_digest
from harness.tasks.build_workspaces import FINAL_HOLDOUT


def build_rows(source: list[dict[str, Any]], accepted: list[str],
               anchor_mode: str = "first") -> list[dict[str, Any]]:
    if anchor_mode not in ("first", "all-tools"):
        raise ValueError(f"unknown anchor mode: {anchor_mode}")
    by_task = {row["task_id"]: row for row in source}
    if len(by_task) != len(source):
        raise ValueError("source has duplicate task ids")
    if not accepted or len(set(accepted)) != len(accepted):
        raise ValueError("accepted tasks must be a nonempty list without duplicates")
    if set(accepted) & FINAL_HOLDOUT:
        raise ValueError(f"held-out tasks in training set: {sorted(set(accepted) & FINAL_HOLDOUT)}")
    if set(accepted) - by_task.keys():
        raise ValueError(f"review names absent tasks: {sorted(set(accepted) - by_task.keys())}")

    built: list[dict[str, Any]] = []
    prefix_digest = None
    for task_id in accepted:
        row = by_task[task_id]
        messages = row["messages"]
        if len(messages) < 4 or [m["role"] for m in messages[:3]] != [
            "system", "user", "assistant"
        ]:
            raise ValueError(f"{task_id}: no first assistant tool decision")
        calls = messages[2].get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != "ls":
            raise ValueError(f"{task_id}: first student action is not exactly one ls call")
        if messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
            raise ValueError(f"{task_id}: final teacher label is not a diagnosis")
        if not messages[-1].get("content"):
            raise ValueError(f"{task_id}: final teacher diagnosis is empty")
        if prefix_digest is None:
            prefix_digest = row["harness_prefix_digest"]
        elif row["harness_prefix_digest"] != prefix_digest:
            raise ValueError(f"{task_id}: harness prefix differs from earlier rows")

        tool_turns = [i for i, message in enumerate(messages[:-1])
                      if message.get("role") == "assistant" and message.get("tool_calls")]
        if not tool_turns or tool_turns[0] != 2:
            raise ValueError(f"{task_id}: no student first tool call")
        if anchor_mode == "first":
            tool_turns = tool_turns[:1]
        for ordinal, index in enumerate(tool_turns, start=1):
            if len(messages[index]["tool_calls"]) != 1:
                raise ValueError(f"{task_id}: tool turn {index} has multiple calls")
            built.append({
                "task_id": (f"{task_id}::student-first-ls" if anchor_mode == "first"
                            else f"{task_id}::student-tool-{ordinal}"),
                "source_task_id": task_id,
                "label_source": ("student_shared_tool_anchor" if anchor_mode == "first"
                                 else "student_tool_anchor"),
                "harness_prefix_digest": row["harness_prefix_digest"],
                ("source_prefix_digest" if anchor_mode == "first"
                 else "source_final_prefix_digest"): row["prefix_digest"],
                "messages": messages[:index + 1],
                "tools": row["tools"],
                "teacher_turn_indices": [],
            })
        teacher = dict(row)
        teacher["task_id"] = f"{task_id}::teacher-at-student-state"
        teacher["source_task_id"] = task_id
        teacher["label_source"] = "teacher_on_student_state"
        teacher["teacher_turn_indices"] = [len(messages) - 1]
        built.append(teacher)
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--plan-template", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--anchor-mode", choices=("first", "all-tools"), default="first")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    source_bytes = args.source.read_bytes()
    review_bytes = args.review.read_bytes()
    review = json.loads(review_bytes)
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != review["source_samples_sha256"]:
        raise ValueError("review names a different source sample file")
    source = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()
              if line.strip()]
    if set(review["accepted_task_ids"]) | set(review["rejected"]) != {
        row["task_id"] for row in source
    }:
        raise ValueError("review must account for every source task")
    rows = build_rows(source, review["accepted_task_ids"], args.anchor_mode)
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    samples_path = args.out_dir / "samples.jsonl"
    samples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plan = json.loads(args.plan_template.read_text(encoding="utf-8"))
    plan["arm"] = ("reviewed-tool-anchor-exploratory" if args.anchor_mode == "first"
                   else "reviewed-all-tool-anchors-exploratory")
    plan["entries"] = len(rows)
    plan["recipe"]["epochs"] = args.epochs
    plan["output_dir"] = args.model_output
    plan["dataset_sha256"] = hashlib.sha256(
        (b"chtd/aux-first-tool/v1\n" if args.anchor_mode == "first"
         else b"chtd/aux-all-tools/v1\n") + source_sha.encode() + b"\n" +
        hashlib.sha256(review_bytes).hexdigest().encode()
    ).hexdigest()
    plan["notes"] = [
        ("Exploratory model-improvement arm. Each task contributes one student-sourced first ls "
         "action and one reviewed teacher diagnosis at a student-reached state."
         if args.anchor_mode == "first" else
         "Exploratory model-improvement arm. Each task contributes separate student-sourced "
         "tool actions and one reviewed teacher diagnosis at a student-reached state."),
        "This is not an all-teacher trajectory or a controlled CHTD-versus-teacher-own comparison.",
    ]
    (args.out_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_samples_sha256": source_sha,
        "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "samples_sha256": samples_digest(lines),
        "entries": len(rows),
        "accepted_task_ids": review["accepted_task_ids"],
        "label_sources": {
            "student_shared_tool_anchor" if args.anchor_mode == "first"
            else "student_tool_anchor": len(rows) - len(review["accepted_task_ids"]),
            "teacher_on_student_state": len(review["accepted_task_ids"]),
        },
        "plan_dataset_sha256": plan["dataset_sha256"],
    }
    if args.anchor_mode != "first":
        manifest["anchor_mode"] = args.anchor_mode
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows for {len(review['accepted_task_ids'])} reviewed tasks "
          f"to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
