"""Grade h1+h2 answers with the validated deepseek-flash judge, in parallel.

Same system prompt, temperature, token budget and lenient extraction as
``eval/llm_judge_pilot.py`` (the pilot validated 85%/89% against two human readers); the only
change is a small thread pool, because this batch is 210 cells instead of 74.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_judge_pilot import GRADE_RE, SYSTEM  # noqa: E402  the validated rubric, reused

DEFAULT_MAX_TOKENS = 4000


def judge_one(cell: dict, base_url: str, api_key: str, model: str, max_tokens: int,
              timeout: int = 300, attempts: int = 4) -> tuple[str | None, str]:
    fixture = Path(cell["fixture_dir"])
    files = "\n\n".join(
        f"--- {f.name} ---\n{f.read_text(encoding='utf-8', errors='replace')}"
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
    for _ in range(attempts):
        try:
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
            content = payload["choices"][0]["message"].get("content") or ""
            if not content:
                max_tokens = max(max_tokens * 2, DEFAULT_MAX_TOKENS)
                b = json.loads(body)
                b["max_tokens"] = max_tokens
                body = json.dumps(b).encode("utf-8")
                continue
            m = GRADE_RE.search(content)
            return (m.group(1) if m else None), content[:400]
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code} {exc.reason}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    return None, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    base_url = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.environ["JUDGE_API_KEY"]
    model = os.environ.get("JUDGE_MODEL", "deepseek-flash")
    max_tokens = int(os.environ.get("JUDGE_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))

    cells = json.loads(args.cells.read_text(encoding="utf-8"))
    rows: list[dict] = [None] * len(cells)  # type: ignore[list-item]
    began = time.time()
    done = 0

    def work(i: int) -> tuple[int, dict]:
        cell = cells[i]
        grade, raw = judge_one(cell, base_url, api_key, model, max_tokens)
        return i, {"set": cell["set"], "arm": cell["arm"], "task": cell["task"],
                   "llm": grade, "raw": raw, "passed_gate": cell.get("passed_gate")}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, i): i for i in range(len(cells))}
        for fut in as_completed(futures):
            i, row = fut.result()
            rows[i] = row
            done += 1
            print(f"[{done}/{len(cells)}] {row['set']}/{row['arm']:12s} {row['task']:24s} llm={row['llm']}",
                  flush=True)

    graded = [r for r in rows if r["llm"]]
    print(f"\ngraded {len(graded)}/{len(rows)} in {time.time() - began:.0f}s "
          f"({(time.time() - began) / max(1, len(rows)):.1f}s per cell)")
    args.out.write_text(json.dumps({"cells": rows}, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
