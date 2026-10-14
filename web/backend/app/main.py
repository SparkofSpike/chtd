"""FastAPI entrypoint: routes + static hosting of the built frontend (SPA).

The frontend dist may not exist yet (frontend builds separately). When absent
we still start and serve a harmless placeholder page at /; missing dist must
NOT break startup.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, state
from .routers import arena, experiments, health, live, sessions, tasks

app = FastAPI(title="CHTD Web Console", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    state.load_all()


# Routers
app.include_router(health.router)
app.include_router(experiments.router)
app.include_router(sessions.router)
app.include_router(tasks.router)
app.include_router(arena.router)
app.include_router(live.router)


@app.get("/api")
def api_root() -> dict:
    return {"service": "CHTD Web Console", "docs": "/docs",
            "routes": ["/api/health", "/api/tasks", "/api/experiments",
                       "/api/sessions", "/api/arena/run",
                       "/api/live/capabilities", "/api/live/tasks", "/api/live/run"]}


def _dist_ready() -> bool:
    return config.FRONTEND_DIST.is_dir() and (config.FRONTEND_DIST / "index.html").exists()


if _dist_ready():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    def spa_root() -> FileResponse:
        return FileResponse(config.FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "not found"}, status_code=404)
        candidate = config.FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(config.FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def placeholder() -> dict:
        return {
            "service": "CHTD Web Console (backend up)",
            "frontend": "frontend/dist not built yet; API is live at /api/*",
            "health": "/api/health",
        }


def main() -> None:  # pragma: no cover
    import uvicorn
    uvicorn.run("app.main:app", host=config.SERVER_HOST, port=config.SERVER_PORT)


if __name__ == "__main__":  # pragma: no cover
    main()
