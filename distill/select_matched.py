"""Freeze equal-count, same-task samples for a baseline versus after-turn CHTD pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from distill.sft import samples_digest
from distill.train import LoraRecipe, TrainingPlan
from harness.digest import text_sha256


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_rows(path: Path, rows: list[dict]) -> str:
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    return samples_digest(lines)


def select(baseline_samples: Path, baseline_manifest: Path, chtd_samples: Path,
           chtd_manifest: Path, output: Path, *, epochs: int = 30,
           output_root: str = "/mnt/proj/chtd-experiments/node4-recollection-20261012/matched-v1",
           base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
           teacher_model: str = "agnes-3.0-flash") -> dict:
    base = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    chtd = json.loads(chtd_manifest.read_text(encoding="utf-8"))
    if base["track_sha256"] != chtd["track_sha256"]:
        raise ValueError("arm tracks differ")
    baseline_rows = _rows(baseline_samples)
    chtd_rows = _rows(chtd_samples)
    if samples_digest([line for line in baseline_samples.read_text(encoding="utf-8").splitlines()
                       if line.strip()]) != base["samples_sha256"]:
        raise ValueError("baseline samples do not match their source manifest digest")
    if samples_digest([line for line in chtd_samples.read_text(encoding="utf-8").splitlines()
                       if line.strip()]) != chtd["samples_sha256"]:
        raise ValueError("CHTD samples do not match their source manifest digest")
    base_tasks, chtd_tasks = set(base["task_ids"]), set(chtd["task_ids"])
    base_holdout, chtd_holdout = set(base["holdout_ids"]), set(chtd["holdout_ids"])
    if base_tasks != chtd_tasks or base_holdout != chtd_holdout:
        raise ValueError("arm source manifests have different train or holdout splits")
    if base_tasks & base_holdout:
        raise ValueError("source manifest overlaps its training and holdout tasks")
    if any(row["task_id"] not in base_tasks for row in baseline_rows + chtd_rows):
        raise ValueError("a sample names a task outside the frozen training split")
    prefix_digests = {row["harness_prefix_digest"] for row in baseline_rows + chtd_rows}
    if len(prefix_digests) != 1 or base["harness_prefix_digest"] not in prefix_digests:
        raise ValueError("captured harness prefixes differ")
    by_baseline: dict[str, list[tuple[int, dict]]] = {}
    by_chtd: dict[str, list[tuple[int, dict]]] = {}
    for index, row in enumerate(baseline_rows):
        by_baseline.setdefault(row["task_id"], []).append((index, row))
    for index, row in enumerate(chtd_rows):
        by_chtd.setdefault(row["task_id"], []).append((index, row))
    common = sorted(set(by_baseline) & set(by_chtd))
    if not common:
        raise ValueError("no reviewed task has an eligible sample in both arms")
    chosen_baseline, chosen_chtd, selections = [], [], []
    for task_id in common:
        # Use the last observed teacher-own final message, rather than choosing
        # by content or by the held-out score. The CHTD side has one aligned label.
        finals = [(index, row) for index, row in by_baseline[task_id]
                  if row["messages"][-1]["role"] == "assistant" and
                  not row["messages"][-1].get("tool_calls")]
        if not finals or len(by_chtd[task_id]) != 1:
            raise ValueError(f"{task_id}: expected one CHTD label and a baseline final message")
        baseline_index, baseline = finals[-1]
        chtd_index, experimental = by_chtd[task_id][0]
        if experimental["messages"][-1]["role"] != "assistant" or (
                experimental["messages"][-1].get("tool_calls")):
            raise ValueError(f"{task_id}: label kinds differ")
        chosen_baseline.append(baseline)
        chosen_chtd.append(experimental)
        selections.append({"task_id": task_id, "baseline_source_index": baseline_index,
                           "chtd_source_index": chtd_index,
                           "baseline_state": baseline["prefix_digest"],
                           "chtd_state": experimental["prefix_digest"]})
    output.mkdir(parents=True, exist_ok=True)
    baseline_digest = _write_rows(output / "baseline-samples.jsonl", chosen_baseline)
    chtd_digest = _write_rows(output / "chtd-samples.jsonl", chosen_chtd)
    recipe = LoraRecipe(epochs=epochs)
    root = output_root.rstrip("/")
    for arm, digest in (("baseline", baseline_digest), ("chtd", chtd_digest)):
        plan = TrainingPlan(
            arm=arm, base_model=base_model,
            base_model_revision=None, teacher_model=teacher_model,
            track_sha256=base["track_sha256"], dataset_sha256=digest,
            entries=len(common), recipe=recipe,
            output_dir=f"{root}/{arm}-adapter",
            notes=("After-turn recovery pilot with one final-message label per common task.",
                   "Holdout remains untouched until both adapters are frozen.",
                   "A one-task intersection cannot support a method-lift claim."),
        )
        (output / f"{arm}-plan.json").write_text(plan.to_json() + "\n", encoding="utf-8")
    report = {
        "status": "frozen_before_holdout",
        "definition": "one final assistant label from each arm on each common reviewed task",
        "selection_rule": "last teacher-own assistant message without tool calls; unique aligned CHTD label",
        "task_ids": common,
        "holdout_ids": chtd["holdout_ids"],
        "track_sha256": base["track_sha256"],
        "harness_prefix_digest": base["harness_prefix_digest"],
        "baseline_source_manifest_sha256": text_sha256(baseline_manifest),
        "chtd_source_manifest_sha256": text_sha256(chtd_manifest),
        "baseline_samples_sha256": baseline_digest,
        "chtd_samples_sha256": chtd_digest,
        "recipe": json.loads(recipe.fingerprint()),
        "selections": selections,
    }
    (output / "selection-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-samples", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--chtd-samples", type=Path, required=True)
    parser.add_argument("--chtd-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-root",
                        default="/mnt/proj/chtd-experiments/node4-recollection-20261012/matched-v1")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="recorded in the generated plans; set it when you train a "
                             "different student")
    parser.add_argument("--teacher-model", default="agnes-3.0-flash",
                        help="recorded in the generated plans; set it when your labels came "
                             "from a different teacher")
    args = parser.parse_args()
    report = select(args.baseline_samples, args.baseline_manifest, args.chtd_samples,
                    args.chtd_manifest, args.out, epochs=args.epochs,
                    output_root=args.output_root, base_model=args.base_model,
                    teacher_model=args.teacher_model)
    print(f"frozen {len(report['task_ids'])} common tasks, one label per arm per task")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
