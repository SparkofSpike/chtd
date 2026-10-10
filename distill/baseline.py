"""Turn observed teacher-own trajectories into auditable control-arm candidates.

The collector writes one result per task. This reader admits decisions only from a
teacher run whose task check passed, then reconstructs the teacher's own state from
its ledger. Selection to a matched training budget happens later, once both arms'
coverage is known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from distill.sft import HarnessPrefix, TrainingSample, conversation_before, samples_digest
from harness.ledger import open_ledger
from harness.trajectory import prefix_chain, read_trajectory


def _label(action: Any, prefix_digest: str) -> dict[str, Any]:
    if action.kind == "message":
        return {"role": "assistant", "content": action.payload}
    if action.kind != "tool_call" or not isinstance(action.name, str):
        raise ValueError(f"unsupported teacher action {action.kind!r}")
    hasher = hashlib.sha256()
    hasher.update(b"chtd/tool-call-id/v1\n")
    hasher.update(prefix_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(action.fingerprint().encode("utf-8"))
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "t0-" + hasher.hexdigest()[:32],
            "type": "function",
            "function": {"name": action.name, "arguments": json.dumps(
                action.payload, ensure_ascii=False, separators=(",", ":")
            )},
        }],
    }


def _path_refusal(action: Any) -> str | None:
    if action.kind != "tool_call" or action.name not in {"read", "write", "edit", "ls", "grep", "find"}:
        return None
    if not isinstance(action.payload, dict):
        return None
    path = action.payload.get("path")
    if not isinstance(path, str) or not path:
        return None
    if path.startswith(("/", "\\")) or (len(path) >= 2 and path[1] == ":" and path[0].isalpha()):
        return "non-portable absolute path"
    return None


def build_candidates(collection: Path, prefix_file: Path, review_file: Path, output: Path) -> dict[str, Any]:
    collection_manifest = json.loads((collection / "collection-manifest.json").read_text(encoding="utf-8"))
    task_ids = collection_manifest["task_ids"]
    if set(task_ids) & set(collection_manifest["holdout_ids"]):
        raise ValueError("train/holdout task overlap in the collection manifest")
    review = json.loads(review_file.read_text(encoding="utf-8"))
    collection_manifest_sha = hashlib.sha256(
        (collection / "collection-manifest.json").read_bytes()
    ).hexdigest()
    if review.get("collection_manifest_sha256") != collection_manifest_sha:
        raise ValueError("review is for a different collection manifest")
    accepted = review.get("accepted_task_ids")
    rejected = review.get("rejected")
    if (not isinstance(accepted, list) or not all(isinstance(x, str) for x in accepted)
            or len(accepted) != len(set(accepted)) or not isinstance(rejected, dict)
            or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in rejected.items())
            or set(accepted) & set(rejected) or set(accepted) | set(rejected) != set(task_ids)):
        raise ValueError("review must partition every training task into accepted or rejected")
    accepted_ids = set(accepted)
    prefix = HarnessPrefix.load(prefix_file)
    if prefix.provenance.get("preset") != "minimal-rl":
        raise ValueError("the prefix capture is not from minimal-rl")

    decisions: list[dict[str, Any]] = []
    samples: list[TrainingSample] = []
    refusals: list[dict[str, str]] = []
    tracks: set[str] = set()
    for task_id in task_ids:
        if task_id not in accepted_ids:
            refusals.append({"task_id": task_id, "reason": "manual review: " + rejected[task_id]})
            continue
        result = json.loads((collection / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        record = result.get("record")
        if not result.get("passed") or not record:
            refusals.append({"task_id": task_id, "reason": "teacher task did not pass its observed check"})
            continue
        if record["task_id"] != task_id or record["arm"] != "teacher":
            raise ValueError(f"wrong rollout record for {task_id}")
        tracks.add(record["track_sha256"])
        ledger_path = Path(record["ledger_path"])
        if not ledger_path.is_absolute():
            ledger_path = collection / ledger_path
        with open_ledger(ledger_path) as ledger:
            trajectory = read_trajectory(ledger, record["session_key"])
            if trajectory.preset != "minimal-rl":
                raise ValueError(f"{task_id} ran on {trajectory.preset!r}")
            prefixes = prefix_chain(ledger.events(record["session_key"]))
            for point in trajectory.points:
                if prefixes.get(point.seq) != point.prefix_digest:
                    raise ValueError(f"{task_id} decision seq {point.seq} has a mismatched prefix")
                reason = _path_refusal(point.action)
                if reason:
                    refusals.append({"task_id": task_id, "reason": reason,
                                     "prefix_digest": point.prefix_digest})
                    continue
                conversation = conversation_before(ledger, record["session_key"], before_seq=point.seq)
                sample = TrainingSample(
                    task_id=task_id,
                    prefix_digest=point.prefix_digest,
                    harness_prefix_digest=prefix.digest(),
                    messages=({"role": "system", "content": prefix.system}, *conversation,
                              _label(point.action, point.prefix_digest)),
                    tools=prefix.tools,
                )
                samples.append(sample)
                decisions.append({
                    "task_id": task_id,
                    "teacher_session": record["session_key"],
                    "teacher_seq": point.seq,
                    "prefix_digest": point.prefix_digest,
                    "teacher_action": point.action.canonical(),
                    "teacher_action_fingerprint": point.action.fingerprint(),
                })
    if len(tracks) != 1 or not samples:
        raise ValueError(f"baseline needs one observed track and nonempty samples, got {sorted(tracks)}")

    output.mkdir(parents=True, exist_ok=True)
    sample_rows = [sample.to_json() for sample in samples]
    (output / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in decisions),
        encoding="utf-8",
    )
    (output / "samples.jsonl").write_text("".join(row + "\n" for row in sample_rows), encoding="utf-8")
    (output / "refusals.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in refusals),
        encoding="utf-8",
    )
    manifest = {
        "arm": "baseline",
        "definition": "teacher decisions on the teacher's own trajectory",
        "track_sha256": next(iter(tracks)),
        "harness_prefix_digest": prefix.digest(),
        "harness_prefix_file_sha256": hashlib.sha256(prefix_file.read_bytes()).hexdigest(),
        "collection_manifest_sha256": collection_manifest_sha,
        "review_sha256": hashlib.sha256(review_file.read_bytes()).hexdigest(),
        "samples_sha256": samples_digest(sample_rows),
        "candidate_count": len(decisions),
        "refusal_count": len(refusals),
        "task_ids": task_ids,
        "holdout_ids": collection_manifest["holdout_ids"],
        "selection_status": "unmatched; compare against the CHTD arm before training",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_candidates(args.collection, args.prefix, args.review, args.out)
    print(f"baseline candidates={manifest['candidate_count']} refused={manifest['refusal_count']} "
          f"samples_sha256={manifest['samples_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
