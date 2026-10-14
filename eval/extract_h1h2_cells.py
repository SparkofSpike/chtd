"""Extract h1 + h2 answers into judge cells, for the 57-task ordinal re-analysis.

h1 records carry the answer text inline; h2 records carry only the session key and its ledger,
so the answer is read back from the session ledger's ``assistant/message`` events.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

CHTD = Path(r"E:\Coding\chtd")
OUT = CHTD / "runs" / "holdout-4-20261014"
OUT.mkdir(parents=True, exist_ok=True)

PROMPT = ("Something in this directory is broken. Read the files and tell me what the defect is. "
          "Do not run any commands and do not change any files - I want to know what is wrong first.")

H1_ARMS = {
    "base-h1":  "data/rollouts/base-h1-20261013/records.jsonl",
    "v5-h1":    "data/rollouts/v5-h1-20261013/records.jsonl",
    "v5b-h1":   "data/rollouts/v5-h1b-20261014/records.jsonl",
    "chtd-h1":  "data/rollouts/chtd-arm-h1-20261014/records.jsonl",
    "ctrl-h1":  "data/rollouts/ctrl-assistant-h1-20261014/records.jsonl",
    "ctrl1-h1": "data/rollouts/ctrl-seed1-h1-20261014/records.jsonl",
}

H2_ARMS = {
    "base-h2a":  "data/holdout-2-20261013/records-base.json",
    "base2-h2a": "data/holdout-2-20261013/records-base2.json",
    "base-h2b":  "data/holdout-2-20261013/records-base-h2b.json",
    "chtd-h2a":  "data/holdout-2-20261013/records-chtd-arm-h2a.json",
    "ctrl-h2a":  "data/holdout-2-20261013/records-ctrl-h2a.json",
    "ctrl1-h2a": "data/holdout-2-20261013/records-ctrl-seed1-h2a.json",
    "v4-h2a":    "data/holdout-2-20261013/records-v4.json",
    "v5-h2a":    "data/holdout-2-20261013/records-v5.json",
    "v5b-h2a":   "data/holdout-2-20261013/records-v5b.json",
    "v5-h2b":    "data/holdout-2-20261013/records-v5-h2b.json",
}


def final_answer(ledger: Path, key: str) -> str:
    """The last assistant text in the session, exactly as the record would carry it."""
    db = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT data FROM events WHERE session_key=? AND type='assistant/message' ORDER BY seq",
            (key,),
        ).fetchall()
    finally:
        db.close()
    for (data,) in reversed(rows):
        msg = json.loads(data)
        parts = [c.get("text", "") for c in msg.get("content", []) if c.get("type") == "text"]
        text = "\n".join(p for p in parts if p).strip()
        if text:
            return text
    return ""


rows: list[dict] = []

for arm, rel in H1_ARMS.items():
    p = CHTD / rel
    if not p.exists():
        print(f"  MISSING {p}")
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "set": "h1", "arm": arm, "task": r["task_id"],
            "answer": (r.get("answer") or "").strip(),
            "fixture_dir": str(Path(r["workspace"])),
            "prompt": PROMPT,
            "passed_gate": bool(r.get("passed")),
        })

for arm, rel in H2_ARMS.items():
    p = CHTD / rel
    if not p.exists():
        print(f"  MISSING {p}")
        continue
    for r in json.loads(p.read_text(encoding="utf-8")):
        answer = ""
        try:
            answer = final_answer(Path(r["ledger"]), r["session"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [{arm}/{r['task_id']}] ledger read failed: {type(exc).__name__}: {exc}")
        rows.append({
            "set": "h2", "arm": arm, "task": r["task_id"],
            "answer": answer,
            "fixture_dir": str(Path(r["workspace"])),
            "prompt": PROMPT,
            "passed_gate": r.get("passed"),
        })

print(f"total cells: {len(rows)}  (h1 {sum(1 for r in rows if r['set'] == 'h1')}, "
      f"h2 {sum(1 for r in rows if r['set'] == 'h2')})")
empty = [r for r in rows if not r["answer"]]
print(f"cells without an answer: {len(empty)}")
for r in empty[:10]:
    print(f"   {r['set']}/{r['arm']}/{r['task']}")
missing = [r for r in rows if not Path(r["fixture_dir"]).is_dir()]
print(f"cells whose fixture dir is missing: {len(missing)}")
for r in missing[:5]:
    print(f"   {r['set']}/{r['arm']}/{r['task']}: {r['fixture_dir']}")

(OUT / "h1h2-cells.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("wrote", OUT / "h1h2-cells.json")
