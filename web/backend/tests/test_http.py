"""HTTP-level checks on the static hosting.

The console serves its own built frontend, and the two ways that goes wrong are quiet:

* A browser that keeps the previous `index.html` keeps running the previous console. That is not a
  hypothetical — after a rebuild the page still requested `index-BoRCJhHC.css` while the server was
  sending `index-BL769I1j.css`, because nothing said how long the HTML could be reused. The rule
  that fixes it is one line in a middleware and one assertion here.
* A browser asks for `/favicon.ico` by convention even when the page declares an SVG icon, and the
  SPA fallback answered with `index.html`. The browser then fails to decode its own icon on every
  navigation, which showed up as console errors on a page where nothing was actually wrong.

These run against the real `dist/` when it exists, and skip the parts that need it otherwise.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, _ASSET_LIKE

client = TestClient(app)


def _dist_ready() -> bool:
    from app import config
    return (config.FRONTEND_DIST / "index.html").is_file()


needs_dist = pytest.mark.skipif(not _dist_ready(), reason="frontend dist not built")


@needs_dist
def test_the_entry_document_must_be_revalidated():
    # `index.html` names the hashed asset URLs. If it may be reused, so may the names inside it,
    # and a rebuilt console keeps serving the old bundle.
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"


@needs_dist
def test_an_unhashed_static_file_must_be_revalidated():
    # The icon's name does not change when its content does, so the same rule applies.
    response = client.get("/favicon.svg")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"


@needs_dist
def test_hashed_assets_may_be_kept_forever():
    from app import config
    assets = sorted((config.FRONTEND_DIST / "assets").glob("*"))
    if not assets:
        pytest.skip("no built assets")
    response = client.get(f"/assets/{assets[0].name}")
    assert response.status_code == 200
    assert "immutable" in response.headers["cache-control"]


@needs_dist
def test_a_conventional_icon_path_is_not_answered_with_html():
    # Answering this with the SPA document is what put decode errors in the console.
    response = client.get("/favicon.ico")
    assert response.status_code == 404
    assert not response.headers["content-type"].startswith("text/html")


@needs_dist
def test_every_conventional_asset_path_is_a_404_rather_than_the_spa_document():
    for path in _ASSET_LIKE:
        response = client.get("/" + path)
        assert response.status_code in (200, 404), path
        if response.status_code == 404:
            assert not response.headers["content-type"].startswith("text/html"), path


@needs_dist
def test_a_client_side_route_still_gets_the_spa_document():
    # The rule above must not swallow real routes: /arena is a page, not an asset.
    response = client.get("/some/deep/route")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_the_api_is_never_answered_with_the_spa_document():
    response = client.get("/api/definitely-not-a-route")
    assert response.status_code == 404
    assert response.json() == {"detail": "not found"}
