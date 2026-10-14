"""Live mode: endpoint health probing and graceful degradation.

MVP: no subprocess orchestration. If the model endpoints are unreachable or the
Agnes CLI is absent (true on this machine), live requests immediately emit a
`mode_fallback` event and the arena router replays the real recorded trajectory.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from . import config


def probe_endpoint(base_url: str) -> dict[str, Any]:
    """Probe /health and /v1/models. Returns status dict."""
    info: dict[str, Any] = {"url": base_url, "ok": False, "model": None, "detail": ""}
    try:
        r = httpx.get(base_url + "/health", timeout=config.ENDPOINT_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            info["model"] = data.get("model")
            info["ok"] = True
        else:
            info["detail"] = f"/health HTTP {r.status_code}"
    except Exception as e:  # network / parse errors -> not ok
        info["detail"] = f"/health unreachable: {type(e).__name__}"
    if info["ok"]:
        try:
            r2 = httpx.get(base_url + "/v1/models", timeout=config.ENDPOINT_TIMEOUT)
            if r2.status_code != 200:
                info["ok"] = False
                info["detail"] = f"/v1/models HTTP {r2.status_code}"
        except Exception as e:
            info["ok"] = False
            info["detail"] = f"/v1/models unreachable: {type(e).__name__}"
    return info


def agnes_cli_status() -> dict[str, Any]:
    p = os.environ.get("CHTD_AGNES_CLI", config.AGNES_CLI)
    return {"path": p, "exists": Path(p).exists()}


def health() -> dict[str, Any]:
    base = probe_endpoint(config.BASE_URL)
    adapter = probe_endpoint(config.ADAPTER_URL)
    cli = agnes_cli_status()
    live_ready = (base["ok"] or adapter["ok"]) and cli["exists"]
    return {
        "mode": "live" if live_ready else "replay",
        "endpoints": {"base": base, "adapter": adapter},
        "agnes_cli": cli,
        "chtd_root": str(config.CHTD_ROOT),
    }


FALLBACK_MESSAGE = "Live 暂不可用（未检测到本地模型端点或 Agnes CLI），已切换到 Replay 回放真实记录。"
