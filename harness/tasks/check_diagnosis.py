"""Decide whether a run diagnosed the defect it was shown.

Why a ledger reader rather than a workspace check. `minimal-rl` — the one builtin track whose
`prompt_sections` is empty, and therefore the one track two arms in two directories can share —
gates `edit`, `write` and `shell` behind an approval nobody can answer. `read` is the only tool that
runs. So a task cannot ask for a repair, and its outcome cannot be "the workspace is now correct".

What it can ask for is a diagnosis, and a diagnosis is checkable: the run's own words either name the
defect or they do not. This script reads the run's assistant messages out of the ledger it was given
and requires every phrase in ``CHTD_EXPECT`` to appear. The phrases are chosen to be things only a
run that found the defect would say, and the task's prompt never names them.

**What requiring phrases cannot do, and the two checks that cover it.** A phrase list is a
one-sided test: it asks whether the answer says the right *words*, never whether it says anything
that is not true. Measured on the 2026-10-10 control sweep, that gap was not hypothetical — the
machine check scored an arm 4/5 while a human reviewer scored the same answers 0/5, because the
answers **invented files that do not exist in the fixture** and the keywords were present anyway:

    "Invents JS paths absent from the Python fixture."
    "Invents a JS contract absent from the Python fixture."

An arm that confabulates scores like an arm that diagnoses, which is the one failure a scorer for
this method cannot have. So two refusals were added, both of which a phrase list cannot express:

* ``CHTD_FORBID`` — phrases that must **not** appear. A task uses it to name what a confused answer
  reaches for (a mechanism from the wrong language, a file that is not there).
* ``CHTD_WORKSPACE`` (or ``CHTD_TASK_DIR``) — the fixture workspace. When given, every file name
  the answer cites (``foo.py``, ``data/rows.csv``) is checked against what the fixture actually
  holds, and an answer citing a file that does not exist is refused **even if every expected phrase
  is present**. This is the check that catches confabulation without anyone having to predict, per
  task, what will be invented. `harness.round` already exports ``CHTD_WORKSPACE`` for every check it
  runs, so a round gets this without changing its call site.

Inputs arrive in the environment, set by `harness.tasks.build_workspaces.check_command` and
`harness.round`, because a check has no other way to learn which run it is judging:

    CHTD_LEDGER        the ledger holding the run
    CHTD_SESSION_KEY   the session to read
    CHTD_EXPECT        a JSON array of phrases, all of which must appear
    CHTD_FORBID        a JSON array of phrases, none of which may appear (optional)
    CHTD_WORKSPACE     the fixture workspace, enabling the invented-file check (optional)
    CHTD_TASK_DIR      the same thing under an older name; either is accepted

Exit 0 when every phrase is present and no forbidden phrase or invented file is, 1 otherwise, with
the reason on stdout so `checks.jsonl` records why a run was judged as it was.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from harness.ledger import open_ledger  # noqa: E402

#: A file name as an answer writes one: a bare name, or a path, with an extension a fixture uses.
#: Deliberately narrow. Matching every token that contains a dot would flag `sorted()` — a function,
#: not a file — and a check that cries wolf on ordinary prose is a check nobody keeps.
FILE_REFERENCE = re.compile(r"(?<![\w./-])([\w-]+(?:/[\w.-]+)*\.(?:py|json|csv|txt|md|ya?ml|ts|js|toml|ini|cfg))\b")


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


def fixture_files(task_dir: str) -> set[str]:
    """Every file the fixture holds, as both a bare name and a relative path.

    Both spellings, because an answer is as likely to say `table.py` as `student/table.py` and either
    is correct when the file exists. A fixture that cannot be read is not treated as empty: an
    unreadable directory would make every cited name look invented, which is a false accusation
    rather than a check.
    """
    root = Path(task_dir)
    names: set[str] = set()
    if not root.is_dir():
        return names
    for path in root.rglob("*"):
        if path.is_file():
            names.add(path.name)
            names.add(path.relative_to(root).as_posix())
    return names


def invented_files(text: str, present: set[str]) -> list[str]:
    """Names the answer cites that the fixture does not hold, in the order first cited.

    Why this is a refusal and not a note. The task asks *what is wrong in this directory*, so a file
    the directory does not contain cannot be part of the right answer — and when an answer names one
    it is describing a workspace it imagined. The 2026-10-10 control arm did exactly that five times
    out of five and still matched the expected keywords four times, so this is the difference between
    a scorer that can rank arms and one that cannot.

    A name is accepted if it matches a file anywhere in the fixture, by bare name or by relative
    path: an answer that says `check.py` is right whether or not it also says which directory that
    sits in.
    """
    if not present:
        return []
    seen: list[str] = []
    for match in FILE_REFERENCE.finditer(text):
        cited = match.group(1)
        if cited in present or cited.rsplit("/", 1)[-1] in present:
            continue
        # An answer quoting the harness's own output may carry a path from outside the fixture;
        # only bare names and fixture-relative paths are held to the fixture.
        if cited.startswith("/") or re.match(r"^[A-Za-z]:", cited):
            continue
        if cited not in seen:
            seen.append(cited)
    return seen


def main() -> int:
    ledger_path = os.environ.get("CHTD_LEDGER")
    session_key = os.environ.get("CHTD_SESSION_KEY")
    if not ledger_path or not session_key:
        print("check_diagnosis: CHTD_LEDGER and CHTD_SESSION_KEY are required")
        return 2

    # Phrases from the command line, or from CHTD_EXPECT when a caller prefers the environment.
    # A `--forbid` marker splits the two lists on one command line, so a caller that cannot set
    # environment variables (a task's stored `check` string) still gets both refusals.
    argv = list(sys.argv[1:])
    forbid_flag = argv.index("--forbid") if "--forbid" in argv else None
    if forbid_flag is None:
        expected = argv
        forbid_from_argv: list[str] = []
    else:
        expected = argv[:forbid_flag]
        forbid_from_argv = argv[forbid_flag + 1:]

    if not expected:
        raw = os.environ.get("CHTD_EXPECT")
        if raw:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                expected = [str(item) for item in loaded]
    if not expected:
        print("check_diagnosis: at least one expected phrase is required")
        return 2

    forbidden: list[str] = list(forbid_from_argv)
    raw_forbid = os.environ.get("CHTD_FORBID")
    if raw_forbid:
        loaded = json.loads(raw_forbid)
        if isinstance(loaded, list):
            forbidden.extend(str(item) for item in loaded)

    text = answer_text(ledger_path, session_key).lower()
    if not text.strip():
        print("no assistant message in the session")
        return 1

    missing = [phrase for phrase in expected if phrase.lower() not in text]
    if missing:
        print(f"answer does not name: {missing}")
        print(f"answer was: {text[:400]!r}")
        return 1

    # Refusals that a phrase list cannot express. Both are reported with the phrase that is
    # missing from the passing answer, so `checks.jsonl` shows why an answer that named every
    # expected word was still judged as not having diagnosed anything.
    present_phrases = [phrase for phrase in forbidden if phrase.lower() in text]
    if present_phrases:
        print(f"answer names every expected phrase, but reaches for {present_phrases}, which this "
              f"defect does not involve")
        print(f"answer was: {text[:400]!r}")
        return 1

    task_dir = os.environ.get("CHTD_WORKSPACE") or os.environ.get("CHTD_TASK_DIR")
    if task_dir:
        invented = invented_files(text, fixture_files(task_dir))
        if invented:
            print(f"answer names every expected phrase, but cites files the workspace does not "
                  f"hold: {invented} — it is describing a directory it did not read")
            print(f"answer was: {text[:400]!r}")
            return 1

    print(f"answer names every expected phrase: {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
