"""Grade a batch of diagnosis answers with an OpenAI-compatible judge, one cell per call.

Why this exists: the phrase gate (`harness/tasks/check_diagnosis.py`) agrees with a two-human
panel on 75% of cells; the remaining error is answers with the right words and a wrong
conclusion, which no phrase list can catch. Reading every answer by hand does not scale past a
few hundred. This script is the measured middle path: a model grades every cell blind, and the
humans read only the cells where it disagrees with them.

Measured on 2026-10-14 (deepseek-flash, 74-cell human-agreed panel): 85% three-way / 89% binary
agreement, against 84% between the two human readers themselves. See
`data/holdout-3-20261014/llm-judge-deepseek-flash.json` for the pilot, its per-cell detail, and
the third-vote split on human-disagreement cells.

Two operational facts baked in below, both measured:
  * the judge must be given room to think — at max_tokens 1200 a reasoning model returned empty
    content on 3/74 cells with the budget spent on reasoning; 4000 cleared it.
  * grade extraction must survive a truncated JSON tail — a lenient regex recovers cells a strict
    JSON parse misses.

Usage:
    python eval/llm_judge_pilot.py --cells <cells.json> --out <graded.json>

`cells.json` is a list of {"arm", "task", "answer", "fixture_dir", "prompt", "human"} — the
caller decides where answers come from (a ledger, a records file). Environment:

    JUDGE_BASE_URL   e.g. https://api.deepseek.com/v1
    JUDGE_API_KEY    bearer token
    JUDGE_MODEL      e.g. deepseek-flash
    JUDGE_MAX_TOKENS default 4000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM = """You are grading one answer to a code-diagnosis task. The fixtures are single-defect
Python workspaces; the prompt asked the model to read the files and say what the defect is.

Grade strictly against the fixture source, on this rubric:
- correct: the answer points at the defective code, and its account of the failure matches what
  the code actually does; any fix it proposes would make check.py pass. Minor mis-statements of
  the failure mode (e.g. calling an expression a "syntax error" when it is a value error) are
  NOT demotions.
- partial: right area, but the account of the failure does not explain the observed failure, OR
  the proposed fix is wrong or hedged toward editing check.py instead of the module.
- failed: the answer concludes there is no defect, blames the test, or names a mechanism that
  does not exist in the fixture.

Output ONLY a JSON object: {"grade": "correct"|"partial"|"failed", "why": "<one sentence>"}"""

GRADE_RE = re.compile(r'"grade"\s*:\s*"(\w+)"')


def judge_one(cell: dict, base_url: str, api_key: str, model: str, max_tokens: int,
              timeout: int = 300, attempts: int = 3) -> tuple[str | None, str]:
    fixture = Path(cell["fixture_dir"])
    files = "\n\n".join(
        f"--- {f.name} ---\n{f.read_text(encoding='utf-8')}"
        for f in sorted(fixture.glob("*.py"))
    )
    user = (f"# Task prompt\n{cell['prompt']}\n\n# Fixture files\n{files}\n\n"
            f"# The answer to grade\n{cell['answer'][:6000]}\n\n{SYSTEM}")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    last = ""
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
            content = payload["choices"][0]["message"].get("content") or ""
            if not content:
                # reasoning model spent the whole budget thinking; retry with more room
                max_tokens = max(max_tokens * 2, 4000)
                body = json.loads(body)
                body["max_tokens"] = max_tokens
                body = json.dumps(body).encode("utf-8")
                continue
            m = GRADE_RE.search(content)
            return (m.group(1) if m else None), content[:400]
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code} {exc.reason}"
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    return None, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base_url = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.environ.get("JUDGE_API_KEY")
    model = os.environ.get("JUDGE_MODEL", "deepseek-flash")
    max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", "4000"))
    if not api_key:
        raise SystemExit("JUDGE_API_KEY is not set")

    cells = json.loads(args.cells.read_text(encoding="utf-8"))
    rows = []
    began = time.time()
    for i, cell in enumerate(cells, 1):
        grade, raw = judge_one(cell, base_url, api_key, model, max_tokens)
        rows.append({"arm": cell.get("arm"), "task": cell.get("task"),
                     "human": cell.get("human"), "llm": grade, "raw": raw})
        mark = "=" if grade == cell.get("human") else "!"
        print(f"[{i}/{len(cells)}] {mark} {cell.get('task', '?'):30s} "
              f"human={cell.get('human')} llm={grade}", flush=True)

    graded = [r for r in rows if r["llm"]]
    agree = sum(1 for r in graded if r["llm"] == r["human"])
    binary = sum(1 for r in graded if (r["llm"] == "correct") == (r["human"] == "correct"))
    summary = {
        "model": model,
        "n": len(rows),
        "graded": len(graded),
        "three_way_agreement": f"{agree}/{len(graded)}" if graded else "0/0",
        "binary_agreement": f"{binary}/{len(graded)}" if graded else "0/0",
        "seconds": round(time.time() - began, 1),
    }
    args.out.write_text(json.dumps({"summary": summary, "cells": rows},
                                   ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print("\n" + json.dumps(summary, ensure_ascii=False))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
