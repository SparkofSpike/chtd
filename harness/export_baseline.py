"""Publish a portable, secret-checked snapshot of an observed control collection."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from distill.baseline import build_candidates
from harness.digest import raw_sha256, text_sha256


def export_collection(
    source: Path, destination: Path, prefix: Path, review: Path, secret_file: Path
) -> dict:
    secret = secret_file.read_text(encoding="utf-8-sig").strip().encode("utf-8")
    if len(secret) < 16:
        raise ValueError("secret file is missing a credential")
    manifest_bytes = (source / "collection-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    task_ids = manifest["task_ids"]
    raw = [(task_id, (source / "tasks" / f"{task_id}.json").read_bytes())
           for task_id in task_ids]
    if secret in manifest_bytes or any(secret in content for _, content in raw):
        raise ValueError("credential found in collection metadata; refusing export")
    ledgers = {json.loads(content)["record"]["ledger_path"] for _, content in raw
               if json.loads(content).get("record")}
    if len(ledgers) != 1:
        raise ValueError(f"expected one source ledger, found {len(ledgers)}")
    ledger = Path(next(iter(ledgers)))
    if not ledger.is_file():
        raise FileNotFoundError(f"source ledger missing: {ledger}")
    if secret in ledger.read_bytes():
        raise ValueError("credential found in source ledger; refusing export")
    if destination.exists():
        raise FileExistsError(f"export destination already exists: {destination}")
    collection = destination / "collection"
    task_dir = collection / "tasks"
    task_dir.mkdir(parents=True)
    (collection / "collection-manifest.json").write_bytes(manifest_bytes)
    with sqlite3.connect(ledger) as source_db, sqlite3.connect(collection / "sessions.db") as copy_db:
        source_db.backup(copy_db)
    for task_id, content in raw:
        row = json.loads(content)
        if row.get("record"):
            row["record"]["ledger_path"] = "sessions.db"
            row["ledger_path"] = "sessions.db"
        row["workspace"] = f"workspaces/{task_id}/teacher"
        (task_dir / f"{task_id}.json").write_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n")
    shutil.copyfile(prefix, destination / "prefix.json")
    shutil.copyfile(review, destination / "review.json")
    sample_manifest = build_candidates(
        collection, destination / "prefix.json", destination / "review.json", destination / "sft"
    )
    exported = destination / "collection/sessions.db"
    if secret in exported.read_bytes():
        raise ValueError("credential found in exported ledger; refusing publication")
    summary = {
        "collection_manifest_sha256": text_sha256(source / "collection-manifest.json"),
        "ledger_sha256": raw_sha256(exported),
        "raw_task_count": len(raw),
        "accepted_task_count": len(json.loads(review.read_text(encoding="utf-8"))["accepted_task_ids"]),
        "candidate_count": sample_manifest["candidate_count"],
        "samples_sha256": sample_manifest["samples_sha256"],
        "training_status": "unmatched; no adapter was trained from this snapshot",
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    args = parser.parse_args()
    summary = export_collection(args.source, args.destination, args.prefix, args.review,
                                args.secret_file)
    print(f"exported {summary['raw_task_count']} tasks and {summary['candidate_count']} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
