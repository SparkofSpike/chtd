"""Run the selected held-out diagnosis tasks against serving arms and score them.

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
AGNES = os.environ.get("CHTD_AGNES_CLI", r"E:\Coding\agnes-harness\packages\cli\dist\local\agnes.mjs")
NODE = os.environ.get("CHTD_NODE", "node")
TASKS_JSONL = Path(os.environ.get("CHTD_TASKS_JSONL", str(CHTD / "harness" / "tasks" / "final-holdout.jsonl")))
COLLECTION_MANIFEST = Path(os.environ.get(
    "CHTD_COLLECTION_MANIFEST",
    str(CHTD / "data" / "chtd-student-20261012" / "prepared" / "matched" / "selection-manifest.json"),
))
PROFILE_SRC = Path(os.environ.get("CHTD_PROFILE_SRC", str(
    CHTD / "data" / "agh_home" / "profiles" / "local-dev" / "profile.yaml")))
TASKS_DIR = Path(os.environ.get("CHTD_TASKS_DIR", str(CHTD / "data" / "tasks")))

HOST = os.environ.get("CHTD_EVAL_HOST", "127.0.0.1")
PORT = int(os.environ.get("CHTD_EVAL_PORT", "8006"))
ARM = os.environ.get("CHTD_EVAL_ARM", "base")
RUN_ROOT = Path(os.environ.get("CHTD_RUN_ROOT", str(CHTD / "data" / "rollouts")))
SHARED_HOME = Path(os.environ.get("CHTD_SHARED_AGH_HOME", str(RUN_ROOT / "_paired-holdout-home")))
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
    old = os.environ.get("CHTD_PROFILE_STUDENT_URL", "http://100.113.249.5:8000/v1")
    new = f"http://{HOST}:{port}/v1"
    if old not in text:
        raise SystemExit(f"profile does not contain the expected student baseUrl {old!r}; "
                         "refusing to guess which line to repoint")
    return text.replace(old, new)


def session_keys(ledger: Path) -> set[str]:
    sys.path.insert(0, str(CHTD))
    from harness.ledger import open_ledger
    if not ledger.exists():
        return set()
    with open_ledger(str(ledger)) as book:
        return {session.session_key for session in book.sessions()}


def new_session_key(ledger: Path, before: set[str]) -> str:
    created = session_keys(ledger) - before
    if len(created) != 1:
        raise RuntimeError(f"expected exactly one new session in {ledger}, found {len(created)}")
    return created.pop()


def score(ledger: Path, key: str, expect: list[str], forbid: list[str],
          workspace: Path) -> tuple[bool, str]:
    env = dict(os.environ)
    env["CHTD_LEDGER"] = str(ledger)
    env["CHTD_SESSION_KEY"] = key
    env["CHTD_FORBID"] = json.dumps(forbid)
    env["CHTD_WORKSPACE"] = str(workspace)
    r = subprocess.run(
        [sys.executable, str(CHTD / "harness" / "tasks" / "check_diagnosis.py"), *expect],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=120,
    )
    return r.returncode == 0, (r.stdout or "").strip()


def run_one(arm: str, port: int, task: dict, root: Path,
            shared_home: Path | None = None) -> dict:
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
    home = shared_home or SHARED_HOME
    if base_dir.exists():
        raise FileExistsError(
            f"{base_dir} already exists; choose a new run name to preserve its raw ledger"
        )
    work.mkdir(parents=True)
    profile_dir = home / "profiles" / "local-dev"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.yaml"
    intended_profile = profile_for(port)
    if profile_path.exists():
        if profile_path.read_text(encoding="utf-8") != intended_profile:
            raise ValueError(f"shared AGH profile {profile_path} differs; refusing a track change")
    else:
        profile_path.write_text(intended_profile, encoding="utf-8", newline="\n")
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

    argv = [NODE, AGNES, "-p", "--standalone", "--preset", PRESET, "--model", STUDENT_MODEL,
            task["prompt"]]
    ledger = home / "data" / "sessions.db"
    before = session_keys(ledger)
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
    record = {
        "arm": arm, "task_id": task_id, "exit": exit_code, "timed_out": timed_out,
        "seconds": round(elapsed, 1),
        "answer": out.strip()[-2000:],
        "stderr_tail": err.strip()[-600:],
        "workspace": str(work),
    }
    try:
        key = new_session_key(ledger, before)
        record["session_key"] = key
        sys.path.insert(0, str(CHTD))
        from harness.ledger import open_ledger
        from harness.round import observed_track
        from harness.trajectory import read_trajectory
        track, preset = observed_track(ledger, key)
        with open_ledger(ledger) as book:
            trajectory = read_trajectory(book, key)
        record["track_sha256"] = track
        record["preset"] = preset
        record["resolved_profile_hash"] = trajectory.resolved_profile_hash
        expected_track = os.environ.get("CHTD_EXPECTED_TRACK")
        if expected_track and track != expected_track:
            raise ValueError(f"observed track {track} differs from expected {expected_track}")
        from harness.tasks.build_workspaces import FORBID
        forbidden = task.get("forbid", list(FORBID.get(task_id, ())))
        passed, output = score(ledger, key, task["expect"], forbidden, work)
        record["passed"] = exit_code == 0 and not timed_out and passed
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
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} already contains a run; choose a new name")
    root.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    task_ids = [t["id"] for t in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("holdout task list contains duplicate ids")
    manifest = json.loads(COLLECTION_MANIFEST.read_text(encoding="utf-8"))
    expected_holdout = set(manifest["holdout_ids"])
    if set(task_ids) != expected_holdout:
        raise ValueError(
            "holdout tasks differ from the frozen selection manifest "
            f"(missing={sorted(expected_holdout - set(task_ids))}, "
            f"unexpected={sorted(set(task_ids) - expected_holdout)})"
        )
    train_ids = set(manifest["task_ids"])
    if train_ids & set(task_ids):
        raise ValueError(f"training/holdout leakage: {sorted(train_ids & set(task_ids))}")
    if only:
        tasks = [t for t in tasks if t["id"] == only]
        if not tasks:
            raise SystemExit(f"no task {only!r} in {TASKS_JSONL}")
    arms = [(ARM, PORT)]
    print(f"run root : {root}")
    print(f"tasks    : {[t['id'] for t in tasks]}")
    print(f"arm      : {arms[0]}")
    print(f"AGH home : {SHARED_HOME}")
    print()

    records = []
    with (root / "records.jsonl").open("x", encoding="utf-8") as journal:
        for arm, port in arms:
            print(f"--- arm {arm} (port {port})")
            for task in tasks:
                record = run_one(arm, port, task, root, SHARED_HOME)
                records.append(record)
                journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                journal.flush()

    print()
    print("=" * 74)
    for arm, _ in arms:
        rows = [r for r in records if r["arm"] == arm]
        n = sum(1 for r in rows if r["passed"])
        print(f"{arm:8s} {n}/{len(rows)}  " +
              " ".join(f"{r['task_id'].replace('fix-','')}={'OK' if r['passed'] else 'no'}"
                       for r in rows))
    print("=" * 74)

    tracks = {r.get("track_sha256") for r in records if r.get("track_sha256")}
    if len(tracks) != 1:
        raise RuntimeError(f"evaluation arm has {len(tracks)} observed tracks; refusing to summarize")
    track = tracks.pop()

    (root / "run.json").write_text(json.dumps({
        "run": run_name,
        "host": HOST,
        "arms": [{"arm": a, "port": p} for a, p in arms],
        "tasks": [t["id"] for t in tasks],
        "expect": {t["id"]: t["expect"] for t in tasks},
        "preset": PRESET,
        "track_sha256": track,
        "shared_agh_home": str(SHARED_HOME),
        "profile_sha256": __import__("hashlib").sha256(
            (SHARED_HOME / "profiles" / "local-dev" / "profile.yaml").read_bytes()
        ).hexdigest(),
        "student_model": STUDENT_MODEL,
        "artifact_id": os.environ.get("CHTD_MODEL_ARTIFACT_ID"),
        "artifact_sha256": os.environ.get("CHTD_MODEL_ARTIFACT_SHA256"),
        "holdout_task_count": len(tasks),
        "score_type": "automatic_expected_forbidden_phrase_proxy",
        "collection_manifest_sha256": __import__("hashlib").sha256(
            COLLECTION_MANIFEST.read_bytes()
        ).hexdigest(),
        "results": {a: {"passed": sum(1 for r in records if r["arm"] == a and r["passed"]),
                        "of": len(tasks)} for a, _ in arms},
        "records": [{"arm": r["arm"], "task_id": r["task_id"], "exit": r["exit"],
                     "passed": r["passed"], "seconds": r["seconds"],
                     "session_key": r["session_key"], "track_sha256": r.get("track_sha256"),
                     "resolved_profile_hash": r.get("resolved_profile_hash")} for r in records],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {root / 'records.jsonl'} and {root / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
