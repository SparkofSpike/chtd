"""Live driver tests: what can be checked without a model, a network, or a CLI.

The parts of `live.py` that decide things — how a profile is built, when an event is sent, which
tasks are runnable, where a credential comes from — are pure or nearly so, and those are what this
file pins. The part that spawns a process is exercised by running the page, not here.
"""
from __future__ import annotations

import json

from app import live


def _spec(**overrides):
    fields = dict(task_id="t", prompt="p", route="teacher", model="agnes-3.0-flash",
                  base_url="https://example.invalid/v1", preset="minimal-rl",
                  timeout=30, api_key="sk-test-value")
    fields.update(overrides)
    return live.LiveSpec(**fields)


# --------------------------------------------------------------------------- the profile


def test_profile_names_the_route_and_the_preset():
    text = live.build_profile(_spec())
    assert "route: teacher" in text
    assert "secret://teacher/default" in text
    assert 'baseUrl: "https://example.invalid/v1"' in text
    assert "presets: { default: minimal-rl, allowed: [minimal-rl] }" in text


def test_a_key_never_reaches_the_profile_file():
    # The key travels in the child's environment and nowhere else: a profile written to disk with
    # a key in it is a key committed to a repository the first time somebody runs the page.
    text = live.build_profile(_spec(api_key="sk-must-not-appear"))
    assert "sk-must-not-appear" not in text


def test_a_base_url_without_a_version_segment_gets_one():
    """Measured: a profile reading `http://127.0.0.1:8000` made the harness ask for
    `http://127.0.0.1:8000/chat/completions` and the endpoint answered 404 — the harness appends
    the path rather than replacing one. Every recorded student profile writes `.../v1`."""
    assert live.normalize_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
    assert live.normalize_base_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000/v1"
    assert live.normalize_base_url("  http://127.0.0.1:8000  ") == "http://127.0.0.1:8000/v1"


def test_a_base_url_that_already_has_a_version_segment_is_left_alone():
    for url in ("http://127.0.0.1:8000/v1",
                "https://api.agnes-ai.cn/v1",
                "https://openrouter.ai/api/v1",
                "http://host:9000/v2"):
        assert live.normalize_base_url(url) == url


def test_the_profile_gets_the_normalized_url():
    # The spec keeps what the operator typed; only the generated profile is normalized, so the
    # page can still show the raw value while the run uses the working one.
    spec = _spec(base_url="http://127.0.0.1:8000")
    text = live.build_profile(spec)
    assert 'baseUrl: "http://127.0.0.1:8000/v1"' in text
    assert 'baseUrl: "http://127.0.0.1:8000"' not in text
    assert spec.base_url == "http://127.0.0.1:8000"


def test_the_route_name_chooses_the_secret_variable():
    assert _spec(route="teacher").secret_env == "AGNES_SECRET_TEACHER_DEFAULT"
    assert _spec(route="student").secret_env == "AGNES_SECRET_STUDENT_DEFAULT"


# --------------------------------------------------------------------------- the frame cursor


def _session(calls, plan=""):
    return {"steps": [{"index": 0, "turn": 1, "plan": plan, "calls": calls}]}


def _call(name="ls", result=None, ordinal=0):
    return {"name": name, "args": {}, "ordinal": ordinal,
            "result": result if result is not None else {}, "duration_ms": 1.0}


def test_a_call_is_announced_only_once_its_result_is_in():
    cursor = live._Cursor()
    pending = _session([_call()])
    assert [n for n, _ in cursor.drain(pending)] == ["step"]
    pending["steps"][0]["calls"][0]["result"] = {"content": "a.py", "isError": False}
    assert [n for n, _ in cursor.drain(pending)] == ["tool_call", "tool_result"]


def test_nothing_is_announced_twice():
    cursor = live._Cursor()
    session = _session([_call(result={"content": "a.py", "isError": False})])
    assert [n for n, _ in cursor.drain(session)] == ["step", "tool_call", "tool_result"]
    assert cursor.drain(session) == []


def test_a_step_whose_plan_arrives_late_is_re_sent():
    cursor = live._Cursor()
    session = _session([])
    cursor.drain(session)
    session["steps"][0]["plan"] = "先读文件"
    assert cursor.drain(session) == [
        ("step", {"step_index": 0, "turn": 1, "plan": "先读文件", "update": True})]


def test_force_drains_a_call_whose_result_never_arrived():
    # A process that dies mid-tool leaves a call with no result; the browser should still see it
    # rather than losing the step that explains what the run was doing when it stopped.
    cursor = live._Cursor()
    session = _session([_call()])
    assert [n for n, _ in cursor.drain(session, force=True)] == ["step", "tool_call", "tool_result"]


# --------------------------------------------------------------------------- task sets


def test_jsonl_reader_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text('# a comment\n\n{"id": "t1", "prompt": "x"}\nnot json\n',
                    encoding="utf-8")
    assert [row["id"] for row in live._read_jsonl(path)] == ["t1"]


def _task_set(tmp_path, ids):
    jsonl = tmp_path / "set.jsonl"
    jsonl.write_text("\n".join(json.dumps({"id": i, "prompt": "p"}) for i in ids),
                     encoding="utf-8")
    fixtures = tmp_path / "ws"
    (fixtures / "with-fix").mkdir(parents=True)
    (fixtures / "with-fix" / "check.py").write_text("print('ok')", encoding="utf-8")
    return jsonl, fixtures


def test_task_list_says_which_tasks_a_machine_can_actually_run(tmp_path, monkeypatch):
    jsonl, fixtures = _task_set(tmp_path, ["with-fix", "no-fix"])
    monkeypatch.setattr(live.config, "LIVE_TASK_SETS", [(jsonl, fixtures, None)])
    rows = {row["task_id"]: row for row in live.list_tasks()}
    assert rows["with-fix"]["fixture_ready"] is True
    assert rows["with-fix"]["files"] == ["check.py"]
    assert rows["no-fix"]["fixture_ready"] is False
    assert rows["no-fix"]["fixture_path"] is None


def test_a_task_id_cannot_walk_out_of_its_fixture_root(tmp_path, monkeypatch):
    jsonl = tmp_path / "set.jsonl"
    jsonl.write_text(json.dumps({"id": "../../../etc/passwd", "prompt": "p"}), encoding="utf-8")
    monkeypatch.setattr(live.config, "LIVE_TASK_SETS", [(jsonl, tmp_path, None)])
    assert live.list_tasks() == []


# --------------------------------------------------------------------------- the capabilities API


def test_the_capabilities_endpoint_can_be_asked_to_probe_again(monkeypatch):
    """An operator who just started an endpoint should not wait out the failure cache.

    A failed probe is remembered for 45 s so a five-second poll does not sit on a closed port;
    `?fresh=true` is the page's way of saying "I changed something, look now".
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from app.routers import live as live_router

    seen: list[bool] = []

    def fake_capabilities(**kwargs):
        seen.append(bool(kwargs.get("fresh")))
        return {"mode": "replay", "agnes_cli": {"path": "x", "exists": False},
                "node": {"bin": "node", "ok": False, "version": "", "detail": ""},
                "routes": [], "runnable_routes": [], "reasons": ["stub"],
                "defaults": {"route": "teacher", "model": "m", "preset": "minimal-rl",
                             "timeout": 240}, "scratch": "", "endpoints": {}}

    monkeypatch.setattr(live_router.live, "capabilities", fake_capabilities)
    client = TestClient(app)
    assert client.get("/api/live/capabilities").status_code == 200
    assert client.get("/api/live/capabilities?fresh=true").status_code == 200
    assert seen == [False, True]


# --------------------------------------------------------------------------- credentials


def test_credential_prefers_the_environment_over_the_file(tmp_path, monkeypatch):
    key_file = tmp_path / "teacher.key"
    key_file.write_text("sk-from-file\n", encoding="utf-8")
    monkeypatch.setattr(live.config, "LIVE_KEY_FILE", str(key_file))
    monkeypatch.delenv("AGNES_SECRET_TEACHER_DEFAULT", raising=False)
    assert live.credential_status("teacher")["source"] == "file"
    monkeypatch.setenv("AGNES_SECRET_TEACHER_DEFAULT", "sk-from-env")
    assert live.credential_status("teacher")["source"] == "env"


def test_credential_status_does_not_carry_the_secret(tmp_path, monkeypatch):
    key_file = tmp_path / "teacher.key"
    key_file.write_text("sk-must-not-be-returned", encoding="utf-8")
    monkeypatch.setattr(live.config, "LIVE_KEY_FILE", str(key_file))
    monkeypatch.delenv("AGNES_SECRET_TEACHER_DEFAULT", raising=False)
    assert "sk-must-not-be-returned" not in json.dumps(live.credential_status("teacher"))


# --------------------------------------------------------------------------- capability probe


def test_capabilities_says_exactly_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(live.config, "LIVE_KEY_FILE", str(tmp_path / "absent.key"))
    monkeypatch.delenv("AGNES_SECRET_TEACHER_DEFAULT", raising=False)
    monkeypatch.setenv("CHTD_AGNES_CLI", str(tmp_path / "absent.mjs"))
    monkeypatch.setattr(live, "node_status",
                        lambda: {"bin": "node", "ok": True, "version": "v0.0.0", "detail": ""})
    monkeypatch.setattr(live, "probe_endpoint",
                        lambda url, **_: {"url": url, "ok": False, "model": None,
                                          "detail": "test"})
    monkeypatch.setattr(live, "_ENDPOINT_CACHE", {})
    caps = live.capabilities(fresh=True)
    assert caps["mode"] == "replay"
    assert caps["runnable_routes"] == []
    assert any("agnes CLI not found" in reason for reason in caps["reasons"])


# --------------------------------------------------------------------------- endpoint probe cache


def test_the_probe_asks_for_the_models_path_once(monkeypatch):
    """`base + "/v1/models"` produced `/v1/v1/models` — measured against this project's own
    serve_openai.py, which matches its paths exactly and therefore answered 404. A real endpoint
    would have read as unreachable while being perfectly usable."""
    seen: list[str] = []

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"data": [{"id": "m"}]}

    def fake_get(url, **kwargs):
        seen.append(url)
        return FakeResponse()

    monkeypatch.setattr(live.httpx, "get", fake_get)
    live._probe_endpoint_uncached("http://127.0.0.1:8000")
    assert seen == ["http://127.0.0.1:8000/v1/models"]
    assert not any("/v1/v1/" in url for url in seen)


def test_the_health_fallback_sits_on_the_service_root(monkeypatch):
    """`/health` is not under the version segment — `serve_openai.py` serves it at `/health`."""
    seen: list[str] = []

    class NotFound:
        status_code = 404
        def json(self):
            return {}

    def fake_get(url, **kwargs):
        seen.append(url)
        return NotFound()

    monkeypatch.setattr(live.httpx, "get", fake_get)
    live._probe_endpoint_uncached("http://127.0.0.1:8000/v1")
    assert seen == ["http://127.0.0.1:8000/v1/models", "http://127.0.0.1:8000/health"]


def test_a_failed_probe_is_not_repeated_on_every_poll(monkeypatch):
    """`/api/health` is polled every five seconds and a probe of a closed local port costs ~2 s here.

    The counter is the point: without the cache this fires on every poll.
    """
    monkeypatch.setattr(live, "_ENDPOINT_CACHE", {})
    calls = []

    def counting(url):
        calls.append(url)
        return {"url": url, "ok": False, "model": None, "detail": "refused"}

    monkeypatch.setattr(live, "_probe_endpoint_uncached", counting)
    for _ in range(5):
        live.probe_endpoint("http://127.0.0.1:9")
    assert len(calls) == 1


def test_fresh_forces_the_probe_again(monkeypatch):
    monkeypatch.setattr(live, "_ENDPOINT_CACHE", {})
    calls = []
    monkeypatch.setattr(live, "_probe_endpoint_uncached",
                        lambda url: (calls.append(url),
                                     {"url": url, "ok": False, "model": None, "detail": ""})[1])
    live.probe_endpoint("http://127.0.0.1:9")
    live.probe_endpoint("http://127.0.0.1:9", fresh=True)
    assert len(calls) == 2


def test_a_reachable_endpoint_is_re_probed_sooner_than_a_dead_one():
    """A server an operator starts should turn green quickly; one that is down can wait."""
    assert live.ENDPOINT_CACHE_OK_SECONDS < live.ENDPOINT_CACHE_FAIL_SECONDS


# --------------------------------------------------------------------------- what a clone has


def test_the_fixture_falls_back_to_the_checked_in_seeds(tmp_path, monkeypatch):
    """`data/` is generated and untracked; a clone only has `harness/tasks/<set>/<task_id>/`."""
    jsonl = tmp_path / "holdout-3.jsonl"
    jsonl.write_text(json.dumps({"id": "seeded", "prompt": "p"}), encoding="utf-8")
    seeds = tmp_path / "harness" / "tasks" / "holdout-3" / "seeded"
    seeds.mkdir(parents=True)
    (seeds / "check.py").write_text("print('ok')", encoding="utf-8")
    # The primary root is the one a full checkout has and a clone does not.
    monkeypatch.setattr(
        live.config, "LIVE_TASK_SETS",
        [(jsonl, tmp_path / "data" / "workspaces", tmp_path / "harness" / "tasks" / "holdout-3")])
    row = live.list_tasks()[0]
    assert row["fixture_ready"] is True
    assert row["files"] == ["check.py"]


def test_capabilities_marks_a_route_ready_when_its_pieces_are_present(tmp_path, monkeypatch):
    key_file = tmp_path / "teacher.key"
    key_file.write_text("sk-present", encoding="utf-8")
    cli = tmp_path / "agnes.mjs"
    cli.write_text("// cli", encoding="utf-8")
    monkeypatch.setattr(live.config, "LIVE_KEY_FILE", str(key_file))
    monkeypatch.setenv("CHTD_AGNES_CLI", str(cli))
    monkeypatch.delenv("AGNES_SECRET_TEACHER_DEFAULT", raising=False)
    monkeypatch.setattr(live, "node_status",
                        lambda: {"bin": "node", "ok": True, "version": "v0.0.0", "detail": ""})
    monkeypatch.setattr(live, "probe_endpoint",
                        lambda url, **_: {"url": url, "ok": False, "model": None, "detail": "test"})
    monkeypatch.setattr(live, "_ENDPOINT_CACHE", {})
    caps = live.capabilities(fresh=True)
    assert caps["mode"] == "live"
    assert caps["runnable_routes"] == ["teacher"]
    assert caps["reasons"] == []
