"""Run the five held-out diagnosis tasks against both arms and score them.

Why this is hand-driven rather than a pipeline command: the held-out measurement is a
*serving* comparison, not a training step. Each arm is an endpoint -- the adapter and the
unadapted base -- and each task is one session in its own workspace, on the same frozen
`minimal-rl` track with the same prompt. The only difference between the columns is which
port the student route points at.

Layout produced:

    data/rollouts/<run>/<arm>/<task_id>/student/    the workspace that session ran in
    data/rollouts/<run>/<arm>/<task_id>/_agh_home/  the ledger holding that one session
    data/rollouts/<run>/records.jsonl                every run, with its score
    data/rollouts/<run>/run.json                    the sweep's own record

The workspace sits under the arm because two arms sharing a path share a session key, and a
session key is derived from the working directory. The ledger is per run rather than per arm
because `sessions()[-1]` is only "the session just run" when the ledger holds exactly one:
both were got wrong once, and the first sweep scored the wrong session four times out of five
while reporting "no assistant message in the session" instead of failing loudly.

Scoring calls `harness/tasks/check_diagnosis.py`, the same reader the training gate uses,
so a pass here means the same thing it means there.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHTD = Path(__file__).resolve().parent.parent
AGNES = r"E:\Coding\agnes-harness\packages\cli\dist\local\agnes.mjs"
TASKS_JSONL = CHTD / "harness" / "tasks" / "final-holdout.jsonl"
PROFILE_SRC = CHTD / "data" / "agh_home" / "profiles" / "local-dev" / "profile.yaml"
TASKS_DIR = CHTD / "data" / "tasks"

HOST = os.environ.get("CHTD_EVAL_HOST", "100.84.151.62")
ARMS = [("adapter", 8000), ("base", 8001)]
# `CHTD_ARMS=name:port,name:port` overrides the pair, so the same sweep can run a different
# serving configuration without editing the file that produced the recorded numbers.
if os.environ.get("CHTD_ARMS"):
    ARMS = [(part.split(":")[0], int(part.split(":")[1]))
            for part in os.environ["CHTD_ARMS"].split(",") if part.strip()]
RUN_ROOT = CHTD / "data" / "rollouts"
STUDENT_MODEL = "primary=student/qwen3-4b-instruct-2507"
PRESET = "minimal-rl"
# A held-out diagnosis is three or four tool calls; a healthy run finishes in ~20 s. A run
# that has not finished in three minutes is not slow, it is looping, and the sweep should
# record that and move on.
TASK_TIMEOUT = 180


def load_tasks() -> list[dict]:
    rows = []
    for line in TASKS_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def profile_for(port: int) -> str:
    """The same profile with the student route pointed at this arm's port."""
    text = PROFILE_SRC.read_text(encoding="utf-8")
    old = "http://100.113.249.5:8000/v1"
    new = f"http://{HOST}:{port}/v1"
    if old not in text:
        raise SystemExit(f"profile does not contain the expected student baseUrl {old!r}; "
                         "refusing to guess which line to repoint")
    return text.replace(old, new)


def session_key(ledger: Path) -> str:
    sys.path.insert(0, str(CHTD))
    from harness.ledger import open_ledger
    with open_ledger(str(ledger)) as book:
        sessions = book.sessions()
    if not sessions:
        raise RuntimeError(f"no session in {ledger}")
    return sessions[-1].session_key


def score(ledger: Path, key: str, expect: list[str]) -> tuple[bool, str]:
    env = dict(os.environ)
    env["CHTD_LEDGER"] = str(ledger)
    env["CHTD_SESSION_KEY"] = key
    r = subprocess.run(
        [sys.executable, str(CHTD / "harness" / "tasks" / "check_diagnosis.py"), *expect],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=120,
    )
    return r.returncode == 0, (r.stdout or "").strip()


def run_one(arm: str, port: int, task: dict, root: Path) -> dict:
    """One task, one arm, its own workspace and its own ledger.

    Both directory choices matter and both were got wrong once:

    * the workspace sits under the arm, so the two arms never share a path -- a session key
      is derived from the working directory, and two arms in one directory are one session;
    * the ledger is per run, not per arm, because `sessions()[-1]` is only *the session just
      run* when the ledger holds exactly one. Sharing one across five tasks scored the wrong
      session four times out of five and reported it as "no assistant message".
    """
    task_id = task["id"]
    base_dir = root / arm / task_id
    work = base_dir / "student"
    home = base_dir / "_agh_home"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    work.mkdir(parents=True)
    (home / "profiles" / "local-dev").mkdir(parents=True)
    (home / "profiles" / "local-dev" / "profile.yaml").write_text(
        profile_for(port), encoding="utf-8")
    for src in (TASKS_DIR / task_id).iterdir():
        if src.is_file():
            shutil.copy2(src, work / src.name)

    env = dict(os.environ)
    env["AGH_HOME"] = str(home)
    env["AGNES_PROFILE"] = "local-dev"
    env["AGNES_SECRET_STUDENT_DEFAULT"] = "local-no-auth"
    env["AGNES_SECRET_TEACHER_DEFAULT"] = "unused-for-this-measurement"
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(var, None)

    argv = ["node", AGNES, "-p", "--standalone", "--preset", PRESET, "--model", STUDENT_MODEL,
            task["prompt"]]
    started = time.time()
    timed_out = False
    try:
        r = subprocess.run(argv, cwd=str(work), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=TASK_TIMEOUT)
        exit_code, out, err = r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as exc:
        # A run that never finishes is a result too, and a more interesting one than an
        # error: it means the loop did not terminate. Record it as such instead of
        # letting it take the whole sweep down with it.
        timed_out = True
        exit_code = -1
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = f"timed out after {TASK_TIMEOUT}s"
    elapsed = time.time() - started
    ledger = home / "data" / "sessions.db"

    record = {
        "arm": arm, "task_id": task_id, "exit": exit_code, "timed_out": timed_out,
        "seconds": round(elapsed, 1),
        "answer": out.strip()[-2000:],
        "stderr_tail": err.strip()[-600:],
        "workspace": str(work),
    }
    try:
        key = session_key(ledger)
        record["session_key"] = key
        passed, output = score(ledger, key, task["expect"])
        record["passed"] = passed
        record["check_output"] = output
    except Exception as exc:  # a run that produced no session is a failed run, recorded as one
        record["session_key"] = None
        record["passed"] = False
        record["check_output"] = f"no scoreable session: {type(exc).__name__}: {exc}"
    print(f"[{arm:8s}] {task_id:22s} exit={exit_code} {elapsed:6.1f}s "
          f"{'TIMEOUT' if timed_out else ''} passed={record['passed']}", flush=True)
    return record


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--only")]
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = a.split("=", 1)[1]
    run_name = argv[0] if argv else "chtd-20261011-a"
    root = RUN_ROOT / run_name
    root.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    if only:
        tasks = [t for t in tasks if t["id"] == only]
        if not tasks:
            raise SystemExit(f"no task {only!r} in {TASKS_JSONL}")
    arms = ARMS if not only else [ARMS[0]]
    print(f"run root : {root}")
    print(f"tasks    : {[t['id'] for t in tasks]}")
    print(f"arms     : {arms}")
    print()

    records = []
    for arm, port in arms:
        print(f"--- arm {arm} (port {port})")
        for task in tasks:
            records.append(run_one(arm, port, task, root))

    print()
    print("=" * 74)
    for arm, _ in arms:
        rows = [r for r in records if r["arm"] == arm]
        n = sum(1 for r in rows if r["passed"])
        print(f"{arm:8s} {n}/{len(rows)}  " +
              " ".join(f"{r['task_id'].replace('fix-','')}={'OK' if r['passed'] else 'no'}"
                       for r in rows))
    print("=" * 74)

    (root / "records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    (root / "run.json").write_text(json.dumps({
        "run": run_name,
        "host": HOST,
        "arms": [{"arm": a, "port": p} for a, p in ARMS],
        "tasks": [t["id"] for t in tasks],
        "expect": {t["id"]: t["expect"] for t in tasks},
        "preset": PRESET,
        "student_model": STUDENT_MODEL,
        "results": {a: {"passed": sum(1 for r in records if r["arm"] == a and r["passed"]),
                        "of": len(tasks)} for a, _ in arms},
        "records": [{"arm": r["arm"], "task_id": r["task_id"], "exit": r["exit"],
                     "passed": r["passed"], "seconds": r["seconds"],
                     "session_key": r["session_key"]} for r in records],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {root / 'records.jsonl'} and {root / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
