"""Decide whether a run diagnosed the defect it was shown.

Why a ledger reader rather than a workspace check. `minimal-rl` — the one builtin track whose
`prompt_sections` is empty, and therefore the one track two arms in two directories can share —
gates `edit`, `write` and `shell` behind an approval nobody can answer. `read` is the only tool that
runs. So a task cannot ask for a repair, and its outcome cannot be "the workspace is now correct".

What it can ask for is a diagnosis, and a diagnosis is checkable: the run's own words either name the
defect or they do not. This script reads the run's assistant messages out of the ledger it was given
and requires every phrase in ``CHTD_EXPECT`` to appear. The phrases are chosen to be things only a
run that found the defect would say, and the task's prompt never names them.

Inputs arrive in the environment, set by `harness.round`, because a check has no other way to learn
which run it is judging:

    CHTD_LEDGER        the ledger holding the run
    CHTD_SESSION_KEY   the session to read
    CHTD_EXPECT        a JSON array of phrases, all of which must appear

Exit 0 when every phrase is present, 1 otherwise, with the reason on stdout so `checks.jsonl`
records why a run was judged as it was.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from harness.ledger import open_ledger  # noqa: E402


def answer_text(ledger_path: str, session_key: str) -> str:
    """Every assistant message in the session, joined.

    All of them, not just the last: a run that states the diagnosis and then restates it in a
    closing sentence should not be judged on the order it happened to use.
    """
    parts: list[str] = []
    with open_ledger(ledger_path) as ledger:
        for event in ledger.events(session_key):
            if event.type != "assistant/message":
                continue
            data = event.data if isinstance(event.data, dict) else {}
            blocks = data.get("content")
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
            elif isinstance(data.get("text"), str):
                parts.append(data["text"])
    return "\n".join(parts)


def main() -> int:
    ledger_path = os.environ.get("CHTD_LEDGER")
    session_key = os.environ.get("CHTD_SESSION_KEY")
    if not ledger_path or not session_key:
        print("check_diagnosis: CHTD_LEDGER and CHTD_SESSION_KEY are required")
        return 2

    # Phrases from the command line, or from CHTD_EXPECT when a caller prefers the environment.
    expected = list(sys.argv[1:])
    if not expected:
        raw = os.environ.get("CHTD_EXPECT")
        if raw:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                expected = [str(item) for item in loaded]
    if not expected:
        print("check_diagnosis: at least one expected phrase is required")
        return 2

    text = answer_text(ledger_path, session_key).lower()
    if not text.strip():
        print("no assistant message in the session")
        return 1

    missing = [phrase for phrase in expected if phrase.lower() not in text]
    if missing:
        print(f"answer does not name: {missing}")
        print(f"answer was: {text[:400]!r}")
        return 1

    print(f"answer names every expected phrase: {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
