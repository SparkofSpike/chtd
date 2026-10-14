"""Live mode: probe whether a real AGH rollout can run here, and drive one when it can.

Two jobs live in this module. The first is a **capability probe**: `capabilities()` reports
whether the agnes CLI, node, and at least one usable route credential are present, and the
arena router falls back to replaying recorded ledgers when they are not — a demo that dies
without saying which piece is missing is worse than one that says so.

The second job is the **real driver**. `stream_run` builds a scratch AGH home under
`backend/.live/runs/<id>/`, spawns the CLI in it, and tails the ledger the run is writing, so
events reach the browser while the model is still working. The event vocabulary is the one the
replay lane already speaks (`begin`/`step`/`tool_call`/`tool_result`/`final`/`end`), so the
frontend renders a live run through the code path it has.

Three shape notes. AGH's `-p` surface answers one task and exits, so a run is bounded by a
timeout rather than by a conversation. The ledger is opened read-only while another process
appends to it, so a locked read is retried rather than raised — a poll returning nothing and a
step that has not happened yet are indistinguishable, and both mean "try again shortly". And a
fixture is copied into the run's own workspace: the checked-in task seeds are never written to.

Nothing here trains, and nothing writes outside `.live/`.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from . import config
from .loaders import trajectory

#: How often the ledger is polled for new rows while a run is in flight.
POLL_SECONDS = 0.4
#: A single poll may lose a race with the writer; this is how long it waits before giving up.
SQLITE_TIMEOUT = 1.0
#: After the run's own `turn/end` lands, how long the process gets to exit on its own before the
#: run is ended for it. Measured: an endpoint that keeps its connection open (HTTP/1.1 without a
#: close, which is what a vLLM-style server does) left the harness process alive after its turn
#: was complete — the ledger held the answer and the process did not exit, so the page sat on
#: "running" until the whole timeout elapsed. The work was done; only the exit was missing. A few
#: seconds covers the harness's own wrapping-up writes (title, cost rows) that follow `turn/end`.
FINISH_GRACE_SECONDS = 5.0
#: Capabilities answer in a few hundred milliseconds when a route is up and time out when it is
#: not, and the top bar asks every five seconds. The cache keeps that polling from spending the
#: whole interval inside `node --version` and a connect attempt.
CAPS_CACHE_SECONDS = 15.0
#: An endpoint's own answer is cached separately and for longer when it is bad news. A local server
#: that is not running does not come up five seconds later, while one an operator just started
#: should turn green promptly — so a refusal is remembered longer than an answer.
ENDPOINT_CACHE_OK_SECONDS = 10.0
ENDPOINT_CACHE_FAIL_SECONDS = 45.0
_ENDPOINT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


# --------------------------------------------------------------------------- capability probe


def probe_endpoint(base_url: str, *, fresh: bool = False) -> dict[str, Any]:
    """Probe a model endpoint. `/v1/models` first: it is the OpenAI-compatible fact.

    A route pointed at a hosted API (the teacher) is *not* probed here — this is only for the
    local student endpoint, which is the piece an operator serves themselves. `trust_env=False`
    keeps an ambient proxy setting from answering for the endpoint.

    Cached per URL: this runs inside `/api/health`, which the top bar polls every five seconds, and
    on this machine a connection to a closed local port takes ~2 seconds to fail. A failed probe
    is remembered longer than a successful one — see `ENDPOINT_CACHE_FAIL_SECONDS`.
    """
    now = time.time()
    if not fresh:
        cached = _ENDPOINT_CACHE.get(base_url)
        if cached is not None:
            age, hit = cached
            ttl = ENDPOINT_CACHE_OK_SECONDS if hit["ok"] else ENDPOINT_CACHE_FAIL_SECONDS
            if now - age < ttl:
                return hit
    info = _probe_endpoint_uncached(base_url)
    _ENDPOINT_CACHE[base_url] = (now, info)
    return info


def _probe_endpoint_uncached(base_url: str) -> dict[str, Any]:
    info: dict[str, Any] = {"url": base_url, "ok": False, "model": None, "detail": ""}
    base = normalize_base_url(base_url)
    info["url"] = base
    # `/models` hangs off the version segment; `/health` sits on the service root. Building the
    # first path as `base + "/v1/models"` produced `/v1/v1/models` — measured against this
    # project's own `serve_openai.py`, which matches its paths exactly and therefore answered 404.
    # A real endpoint would then read as unreachable while being perfectly usable.
    origin = base[: base.rindex("/")] if "/" in base else base
    for url, label in ((f"{base}/models", "models"), (f"{origin}/health", "health")):
        try:
            response = httpx.get(url, timeout=config.ENDPOINT_TIMEOUT, trust_env=False)
        except Exception as error:  # network / DNS / refused -> try the next shape
            info["detail"] = f"{label} unreachable: {type(error).__name__}"
            continue
        if response.status_code != 200:
            info["detail"] = f"{label} HTTP {response.status_code}"
            continue
        info["ok"] = True
        info["detail"] = url
        try:
            payload = response.json()
            if label == "models":
                rows = payload.get("data") or []
                if rows and isinstance(rows[0], dict):
                    info["model"] = rows[0].get("id")
            else:
                info["model"] = payload.get("model")
        except Exception:
            pass
        break
    return info


def agnes_cli_status() -> dict[str, Any]:
    p = os.environ.get("CHTD_AGNES_CLI", config.AGNES_CLI)
    return {"path": p, "exists": Path(p).exists()}


def node_status() -> dict[str, Any]:
    node = os.environ.get("CHTD_NODE", config.NODE_BIN)
    try:
        done = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=15)
    except Exception as error:
        return {"bin": node, "ok": False, "version": "", "detail": type(error).__name__}
    version = (done.stdout or "").strip()
    return {"bin": node, "ok": done.returncode == 0, "version": version,
            "detail": "" if done.returncode == 0 else (done.stderr or "").strip()[:200]}


def credential_status(route: str) -> dict[str, Any]:
    """Where a route's key would come from. The value itself is never returned."""
    env_name = f"AGNES_SECRET_{route.upper().replace('-', '_')}_DEFAULT"
    if os.environ.get(env_name, "").strip():
        return {"source": "env", "env": env_name, "ok": True}
    key_file = Path(config.LIVE_KEY_FILE)
    try:
        if key_file.is_file() and key_file.read_text(encoding="utf-8").strip():
            return {"source": "file", "path": str(key_file), "ok": True, "env": env_name}
    except OSError:
        pass
    return {"source": "none", "env": env_name, "ok": False}


_CAPS_CACHE: tuple[float, dict[str, Any]] | None = None


def capabilities(*, fresh: bool = False) -> dict[str, Any]:
    """What Live needs, and which of those pieces is missing.

    The teacher route is a hosted API, so it needs a credential and nothing else; the student
    route needs an endpoint that answers. A machine with neither can still run the whole demo
    through replay, which is why this reports rather than refuses.

    Cached: the probe shells out to `node --version` and opens a socket, and `/api/health` is
    polled on a timer. `fresh=True` recomputes, which is what the Live page wants after an
    operator edits the model settings.
    """
    global _CAPS_CACHE
    now = time.time()
    if not fresh and _CAPS_CACHE and now - _CAPS_CACHE[0] < CAPS_CACHE_SECONDS:
        return _CAPS_CACHE[1]
    result = _capabilities_uncached(fresh=fresh)
    _CAPS_CACHE = (now, result)
    return result


def _capabilities_uncached(*, fresh: bool = False) -> dict[str, Any]:
    cli = agnes_cli_status()
    node = node_status()
    route = config.LIVE_ROUTE
    teacher_cred = credential_status("teacher")
    student_cred = credential_status("student")
    student_raw = config.STUDENT_URL
    student_endpoint = probe_endpoint(normalize_base_url(student_raw), fresh=fresh)

    routes = [
        {"route": "teacher", "kind": "hosted",
         "base_url": config.LIVE_BASE_URL, "model": config.LIVE_MODEL,
         "credential": teacher_cred, "reachable": None,
         "ready": bool(teacher_cred["ok"]),
         "note": "托管 API；有凭据即可用，网络在运行时才真正打通"},
        {"route": "student", "kind": "local",
         "base_url": normalize_base_url(student_raw),
         "model": config.STUDENT_MODEL_ID,
         "credential": student_cred, "reachable": bool(student_endpoint["ok"]),
         "endpoint": student_endpoint,
         "ready": bool(student_endpoint["ok"]),
         "note": "你自己服务的 OpenAI 兼容端点，点右侧「配置」填地址"},
    ]

    reasons: list[str] = []
    if not cli["exists"]:
        reasons.append(f"agnes CLI not found at {cli['path']} (CHTD_AGNES_CLI)")
    if not node["ok"]:
        reasons.append(f"node not runnable as {node['bin']} (CHTD_NODE)")

    runnable_routes = [r["route"] for r in routes if r["ready"]]
    if not reasons and not runnable_routes:
        reasons.append("no route is ready: teacher needs a credential "
                       f"({teacher_cred['env']} or CHTD_LIVE_KEY_FILE), "
                       "student needs an endpoint answering /v1/models")

    return {
        "mode": "live" if (not reasons and runnable_routes) else "replay",
        "agnes_cli": cli,
        "node": node,
        "routes": routes,
        "runnable_routes": runnable_routes,
        "reasons": reasons,
        "defaults": {
            "route": route, "model": config.LIVE_MODEL, "preset": config.LIVE_PRESET,
            "timeout": config.LIVE_TIMEOUT,
        },
        "scratch": str(config.LIVE_RUNS_DIR),
    }


def health() -> dict[str, Any]:
    """The top bar's poll: can Live run here at all, and on which route.

    The endpoint probe inside `capabilities()` is reused rather than repeated — this runs on a
    five-second timer, and probing twice would spend the interval waiting on a socket.
    """
    caps = capabilities()
    student = next((r for r in caps["routes"] if r["route"] == "student"), None)
    endpoint = (student or {}).get("endpoint") or {"url": "", "ok": False, "model": None,
                                                   "detail": "not probed"}
    return {
        "mode": caps["mode"],
        # Kept for clients that predate the route list; the Live page reads `routes`, which
        # names which model each one would run.
        "endpoints": {"base": endpoint, "adapter": endpoint},
        "routes": [{"route": r["route"], "ready": r["ready"], "kind": r["kind"],
                    "model": r["model"], "base_url": r["base_url"]} for r in caps["routes"]],
        "agnes_cli": caps["agnes_cli"],
        "chtd_root": str(config.CHTD_ROOT),
        "runnable_routes": caps["runnable_routes"],
        "reasons": caps["reasons"],
    }


FALLBACK_MESSAGE = "Live 暂不可用（未检测到本地模型端点或 Agnes CLI），已切换到 Replay 回放真实记录。"


# --------------------------------------------------------------------------- task sets


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            row = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("id"):
            rows.append(row)
    return rows


def _fixture_dir(task_id: str, candidates: tuple[Path | None, ...]) -> Path | None:
    """The directory holding a task's files, if this machine has one.

    Checked in priority order: a built workspace tree first (what the recorded sweeps ran
    against), then the checked-in seed directory. Missing everywhere is reported rather than
    guessed — a run without its fixture fails in a way that looks like the model's fault.
    """
    if not config.LIVE_TASK_ID_OK.match(task_id):
        return None
    for candidate in candidates:
        if candidate is None:
            continue
        directory = candidate / task_id
        if directory.is_dir():
            return directory
    return None


def list_tasks() -> list[dict[str, Any]]:
    """Task sets the Live panel can run, with the fixtures each one would use."""
    out: list[dict[str, Any]] = []
    for jsonl, primary, fallback in config.LIVE_TASK_SETS:
        if not jsonl.is_file():
            continue
        for row in _read_jsonl(jsonl):
            task_id = row["id"]
            if not config.LIVE_TASK_ID_OK.match(task_id):
                continue
            fixture = _fixture_dir(task_id, (primary, fallback))
            files: list[str] = []
            if fixture is not None:
                try:
                    files = sorted(p.name for p in fixture.iterdir() if p.is_file())
                except OSError:
                    files = []
            out.append({
                "task_id": task_id,
                "set": jsonl.stem,
                "prompt": row.get("prompt") or "",
                "check": row.get("check") or row.get("verify"),
                "expect": row.get("expect") or [],
                "forbid": row.get("forbid") or [],
                "why": row.get("why") or "",
                "fixture_ready": fixture is not None,
                "fixture_path": str(fixture) if fixture is not None else None,
                "files": files,
            })
    return out


def find_task(task_id: str) -> dict[str, Any] | None:
    for row in list_tasks():
        if row["task_id"] == task_id:
            return row
    return None


# --------------------------------------------------------------------------- driving a run


@dataclass
class LiveSpec:
    """One live run, fully resolved: nothing downstream reads the environment again."""

    task_id: str
    prompt: str
    route: str
    model: str
    base_url: str
    preset: str
    timeout: int
    api_key: str
    expect: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    check: str | None = None
    fixture: Path | None = None

    @property
    def secret_env(self) -> str:
        return f"AGNES_SECRET_{self.route.upper().replace('-', '_')}_DEFAULT"


def normalize_base_url(url: str) -> str:
    """The endpoint root the harness appends `/chat/completions` to.

    The suffix is not cosmetic. Measured on this machine: a profile whose student route read
    `http://127.0.0.1:8000` made the harness ask for `http://127.0.0.1:8000/chat/completions`,
    and the endpoint answered 404 — the harness appends the path rather than replacing one. Every
    recorded student profile writes it as `.../v1`, so that is the form this produces; an operator
    who typed `http://localhost:8000` meant the same server and now gets the same URL.

    A URL that already carries a version segment is left alone, so `https://openrouter.ai/api/v1`
    and `http://host:8000/v2` survive unchanged.
    """
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return cleaned
    if not re.search(r"/v\d+(?:/|$)", cleaned):
        cleaned += "/v1"
    return cleaned


def build_profile(spec: LiveSpec) -> str:
    """A one-route AGH profile for this run.

    `name` has to be a builtin template name — the resolver passes it straight through and
    refuses anything else — so the directory is `profiles/local-dev/` even though this profile
    is generated, not checked in.
    """
    base_url = normalize_base_url(spec.base_url)
    return f"""# Generated by the CHTD console for one live run; not a checked-in profile.
name: local-dev
schemaVersion: 1
provider:
  package: "@agnes/ai"
  adapters: ["@agnes/ai"]
  routes:
    - route: {spec.route}
      api: openai-completions
      baseUrl: "{base_url}"
      credentialRef: "secret://{spec.route}/default"
      models:
        - id: "{spec.model}"
          name: "{spec.model}"
          api: openai-completions
          route: {spec.route}
          baseUrl: "{base_url}"
          reasoning: false
          input: ["text"]
          cost: {{ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }}
          contextWindow: 32768
          maxTokens: 8192
          toolCallFormats: ["native"]
          thinkingReplay: "native"
          contract_id: null
adapters:
  secrets: {{ kind: env }}
presets: {{ default: {spec.preset}, allowed: [{spec.preset}] }}
"""


def _newest_session_key(ledger: Path) -> str | None:
    """The session the run is writing. Read-only, and forgiving while the writer holds a lock."""
    if not ledger.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=SQLITE_TIMEOUT)
    except sqlite3.Error:
        return None
    try:
        row = con.execute(
            "SELECT session_key FROM sessions ORDER BY created_at DESC LIMIT 1").fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None
    finally:
        con.close()


def _read_events(ledger: Path, session_key: str) -> list[dict[str, Any]]:
    """Every event of a session, in seq order. A locked read returns what it managed to get."""
    try:
        con = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=SQLITE_TIMEOUT)
    except sqlite3.Error:
        return []
    try:
        rows = con.execute(
            "SELECT seq, type, data FROM events WHERE session_key = ? ORDER BY seq",
            (session_key,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out: list[dict[str, Any]] = []
    for seq, etype, data in rows:
        try:
            parsed = json.loads(data) if isinstance(data, str) else {}
        except ValueError:
            parsed = {}
        out.append({"seq": seq, "type": etype, "data": parsed})
    return out


class _Cursor:
    """Turns successive whole-session rebuilds into frames that are each sent exactly once.

    The trajectory builder is a pure function over the *whole* event list, so the tail is
    rebuilt on every poll and this keeps track of what the browser has already been told. A
    step is announced once and re-announced when its plan text appears; a call is announced
    only once its result is in, so the two arrive together rather than as a call that dangles.
    """

    def __init__(self) -> None:
        self.steps = 0
        self.calls: dict[int, int] = {}
        self.plans: dict[int, str] = {}

    def drain(self, session: dict[str, Any], *, force: bool = False) -> list[tuple[str, dict]]:
        frames: list[tuple[str, dict]] = []
        for index, step in enumerate(session.get("steps") or []):
            plan = step.get("plan") or ""
            if index >= self.steps:
                frames.append(("step", {"step_index": index, "turn": step.get("turn"),
                                        "plan": plan,
                                        "recovery_kind": step.get("recovery_kind")}))
                self.steps = index + 1
                self.plans[index] = plan
            elif plan and plan != self.plans.get(index):
                # The step was announced before its prose landed; send the fuller version rather
                # than leaving the browser holding an empty plan for a step that had one.
                frames.append(("step", {"step_index": index, "turn": step.get("turn"),
                                        "plan": plan, "update": True,
                                        "recovery_kind": step.get("recovery_kind")}))
                self.plans[index] = plan
            done = self.calls.get(index, 0)
            for call in (step.get("calls") or [])[done:]:
                result = call.get("result") or {}
                if not force and not result.get("content") and not result.get("isError"):
                    break
                frames.append(("tool_call", {
                    "step_index": index, "name": call.get("name"),
                    "args": call.get("args"), "ordinal": call.get("ordinal"),
                    "policy": call.get("policy")}))
                frames.append(("tool_result", {
                    "step_index": index, "name": call.get("name"),
                    "content": trajectory.sanitize_text(result.get("content") or ""),
                    "isError": bool(result.get("isError")),
                    "result_kind": result.get("kind") or ("model" if result.get("isError") else "ok"),
                    "duration_ms": call.get("duration_ms")}))
                done += 1
            self.calls[index] = done
        return frames


def _run_diagnosis(ledger: Path, session_key: str, expect: list[str], forbid: list[str],
                   workspace: Path) -> dict[str, Any]:
    """Judge a diagnosis with the project's own scorer.

    `harness/tasks/check_diagnosis.py` is what produced every recorded holdout-3 score, so
    pointing at it rather than reimplementing its phrase semantics keeps a live run comparable
    with the recorded ones: `|` offers spellings of one mechanism and `.?` covers `in place` /
    `in-place` / `inplace`. It also refuses an answer that cites a file the fixture does not
    hold, which a phrase list alone cannot express.

    A task's stored `check` string is *not* this: on the diagnosis sets it names a file inside
    the fixture (`python check.py`), which is evidence for the model to read rather than the
    command that judges the run. Running it would fail every correct diagnosis.
    """
    script = config.CHTD_ROOT / "harness" / "tasks" / "check_diagnosis.py"
    if not script.is_file():
        return {"ran": False, "detail": f"scorer not found at {script}"}
    argv = [sys.executable, str(script), *expect]
    if forbid:
        argv.append("--forbid")
        argv.extend(forbid)
    env = dict(os.environ)
    env.update({"CHTD_LEDGER": str(ledger), "CHTD_SESSION_KEY": session_key,
                "CHTD_WORKSPACE": str(workspace)})
    try:
        done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=60, env=env)
    except Exception as error:
        return {"ran": False, "detail": type(error).__name__}
    output = ((done.stdout or "") + (done.stderr or "")).strip()
    return {"ran": True, "ok": done.returncode == 0, "exit": done.returncode,
            "output": output[:4000], "scorer": "check_diagnosis"}


def _result_counts(session: dict[str, Any]) -> dict[str, int]:
    """How the run's tool calls ended, by kind.

    A run in this project routinely contains refusals: the workspace boundary stops a path outside
    the sandbox, and the read-only preset's approval gate stops `edit`/`write`/`shell`. Reported
    as plain numbers so the page can say "the boundary stopped 2 calls and the model adapted"
    instead of leaving a reader to count red chips and conclude something is broken.
    """
    counts = {"ok": 0, "guardrail": 0, "approval": 0, "model": 0}
    for step in session.get("steps") or []:
        for call in step.get("calls") or []:
            result = call.get("result") or {}
            kind = result.get("kind") or ("model" if result.get("isError") else "ok")
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _tail_of(path: Path, limit: int = 4000) -> str:
    """The end of a run's output, for the case where it did not finish cleanly.

    Without this the page can only say `exit 4`, and a reader has no way to tell an approval
    refusal from a transport failure from a model that simply wandered off. The harness writes
    both to the same place, so the page shows what it wrote.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return trajectory.sanitize_text(text[-limit:].strip())


def _run_check(work: Path, command: str) -> dict[str, Any]:
    """Run a task's own check script inside the workspace it just ran in.

    Only ever called with a command that came from the task set, never with one from a request.
    """
    argv = command.split()
    if argv and argv[0] in ("python", "python3"):
        argv[0] = sys.executable
    try:
        done = subprocess.run(argv, cwd=str(work), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60)
    except Exception as error:
        return {"ran": True, "ok": False, "detail": type(error).__name__}
    output = ((done.stdout or "") + (done.stderr or "")).strip()
    return {"ran": True, "ok": done.returncode == 0, "exit": done.returncode,
            "output": trajectory.sanitize_text(output[:4000])}


async def stream_run(spec: LiveSpec) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Drive one run and yield `(event, payload)` as the ledger grows."""
    caps = capabilities()
    if spec.route not in caps["runnable_routes"]:
        # The capabilities answer is cached for 15 s, so it can be stale in exactly the case that
        # matters: an operator started the endpoint and clicked Run without clicking re-check.
        # Refusing here would be a false negative on a route that is up. Ask once more, for real,
        # before saying no — this costs a probe only on the path that was about to fail anyway.
        caps = capabilities(fresh=True)
    if spec.route not in caps["runnable_routes"]:
        yield ("live_error", {"message": f"route {spec.route!r} is not ready",
                              "reasons": caps["reasons"]})
        return

    cli = Path(caps["agnes_cli"]["path"])
    node = caps["node"]["bin"]
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    run_dir = config.LIVE_RUNS_DIR / run_id
    home = run_dir / "home"
    work = run_dir / "work"
    (home / "profiles" / "local-dev").mkdir(parents=True, exist_ok=True)
    (home / "data").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    if spec.fixture is not None and spec.fixture.is_dir():
        for item in sorted(spec.fixture.iterdir()):
            target = work / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
            copied.append(item.name)

    (home / "profiles" / "local-dev" / "profile.yaml").write_text(
        build_profile(spec), encoding="utf-8", newline="\n")
    ledger = home / "data" / "sessions.db"
    log_path = run_dir / "stdout.log"

    argv = [node, str(cli), "-p", "--standalone", "--preset", spec.preset,
            "--cwd", str(work), "--model", f"primary={spec.route}/{spec.model}", spec.prompt]

    child_env = dict(os.environ)
    child_env.update({"AGH_HOME": str(home), "AGNES_PROFILE": "local-dev",
                      spec.secret_env: spec.api_key})

    yield ("begin", {
        "mode": "live", "effective_mode": "live", "lanes": [f"{spec.route}/{spec.model}"],
        "task_id": spec.task_id, "run_id": run_id, "route": spec.route, "model": spec.model,
        "preset": spec.preset, "base_url": normalize_base_url(spec.base_url),
        "workspace": str(work), "files": copied, "replay_note": "",
    })

    started = time.time()
    cursor = _Cursor()
    session_key: str | None = None
    timed_out = False
    exit_code: int | None = None
    ended_by_ledger = False
    turn_done_at: float | None = None
    check_result: dict[str, Any] | None = None
    process: subprocess.Popen | None = None

    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=str(work), env=child_env,
                                       stdout=log, stderr=subprocess.STDOUT, text=True)
            while True:
                session_key = session_key or _newest_session_key(ledger)
                if session_key:
                    events = _read_events(ledger, session_key)
                    if events:
                        session = trajectory.build_session(
                            {"task_id": spec.task_id, "arm": f"{spec.route}/{spec.model}",
                             "session_key": session_key, "preset": spec.preset}, events)
                        for frame in cursor.drain(session):
                            yield frame
                        if turn_done_at is None and any(e["type"] == "turn/end" for e in events):
                            turn_done_at = time.time()
                exit_code = process.poll()
                if exit_code is not None:
                    break
                if turn_done_at is not None and time.time() - turn_done_at > FINISH_GRACE_SECONDS:
                    # The turn is complete and the process is still here only because something is
                    # holding a connection open. End the run; the answer is already in the ledger.
                    ended_by_ledger = True
                    process.kill()
                    process.wait(timeout=15)
                    break
                if time.time() - started > spec.timeout:
                    timed_out = True
                    process.kill()
                    process.wait(timeout=15)
                    exit_code = -1
                    break
                await asyncio.sleep(POLL_SECONDS)

        # Drain what the last poll could not see: the process is gone, so nothing more is coming.
        if session_key:
            events = _read_events(ledger, session_key)
            session = trajectory.build_session(
                {"task_id": spec.task_id, "arm": f"{spec.route}/{spec.model}",
                 "session_key": session_key, "preset": spec.preset}, events)
            for frame in cursor.drain(session, force=True):
                yield frame
        else:
            session = {"steps": [], "final_answer": "", "user_prompt": spec.prompt}

        # A run the ledger finished counts as finished: the process was only still alive because a
        # connection was. Judging it is the point, so both endings qualify.
        completed = exit_code == 0 or ended_by_ledger
        if spec.expect and not timed_out and completed and session_key:
            check_result = _run_diagnosis(ledger, session_key, spec.expect, spec.forbid, work)
        elif spec.check and not timed_out and completed:
            check_result = _run_check(work, spec.check)

        seconds = round(time.time() - started, 1)
        passed = check_result.get("ok") if check_result else None
        yield ("final", {
            "lane": f"{spec.route}/{spec.model}",
            "answer": trajectory.sanitize_text(session.get("final_answer") or ""),
            "verdict": {"passed": passed},
            "model_settings": session.get("model_settings") or [],
            "steps": len(session.get("steps") or []),
        })
        yield ("end", {
            "lane": f"{spec.route}/{spec.model}",
            "session": {"session_key": session_key, "exit": exit_code,
                        "ended_by": "ledger" if ended_by_ledger else "process",
                        "timed_out": timed_out, "seconds": seconds, "passed": passed,
                        "check": check_result, "run_id": run_id,
                        "result_counts": _result_counts(session),
                        "ledger": str(ledger) if session_key else None,
                        "stdout": str(log_path),
                        "stdout_tail": _tail_of(log_path)},
        })
    finally:
        if process is not None and process.poll() is None:  # pragma: no cover - safety net
            process.kill()
    yield ("done", {"lanes": [f"{spec.route}/{spec.model}"], "run_id": run_id})
