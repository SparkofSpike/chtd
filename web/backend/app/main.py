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


@app.middleware("http")
async def _cache_headers(request, call_next):
    """Tell the browser which files may be reused and which may not.

    Without this, a response carries no `Cache-Control` and the browser falls back to heuristic
    freshness — which is how a rebuilt console keeps serving the previous page from cache after a
    plain reload (observed: the server sent `index-BL769I1j.css` while the page still ran
    `index-BoRCJhHC.css`). The two halves have opposite answers:

    * `/assets/*` carries a content hash in its name, so a changed file is a different URL and the
      old one can be kept forever.
    * Everything else that is not the API is SPA entry or an unhashed asset, and its **name** does
      not change when its content does — so it must be revalidated on every load. That includes
      `index.html`, which is the file that names the hashed URLs: leaving it cacheable defeats the
      hashing entirely, because the pointer to the new hash is itself stale.
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif not path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


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


#: Assets a browser asks for by convention rather than because a page named them. Answering these
#: with `index.html` is what makes a console show an error on every navigation: the browser gets HTML
#: where it expected an icon and fails to decode it. A 404 is the honest answer, and a quiet one.
_ASSET_LIKE = ("favicon.ico", "apple-touch-icon.png", "apple-touch-icon-precomposed.png",
               "robots.txt", "sitemap.xml")


if _dist_ready():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    def spa_root() -> FileResponse:
        return FileResponse(config.FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "not found"}, status_code=404)
        if full_path in _ASSET_LIKE:
            return JSONResponse({"detail": f"no {full_path} shipped"}, status_code=404)
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
