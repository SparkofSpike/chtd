"""The frozen tracks, checked against git as well as against the worktree.

`harness/freeze.py` and `packages/chtd/tools/freeze.mjs` both compare the file on disk with its
recorded hash. That catches an edit. It does not catch the other half of the claim: that the bytes
git stores are the same bytes, which is what makes the track survive a clone onto a machine with a
different line-ending policy. `M1` in `docs/plan.md` asserts both, so both are tested here.

The check is over the committed blob rather than over `git diff`, because `.gitattributes` marks
these paths `-text` precisely so that the stored bytes are the worktree bytes; a repository where
that attribute was lost would pass a diff-based check and fail this one.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: `(track file, its recorded hash)`. Both frozen tracks, so neither can drift unnoticed.
TRACKS = (
    (Path("harness/presets/chtd.yaml"), Path("harness/presets/chtd.sha256")),
    (Path("packages/chtd/presets/chtd.mjs"), Path("packages/chtd/presets/chtd.sha256")),
)


def _git(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"git {args[0]} unavailable here: {result.stderr.decode('utf-8', 'replace')[:200]}")
    return result.stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(("track", "recorded"), TRACKS, ids=lambda p: str(p))
def test_the_recorded_hash_matches_the_worktree(track: Path, recorded: Path) -> None:
    expected = (REPO / recorded).read_text(encoding="utf-8").strip()
    assert _sha256((REPO / track).read_bytes()) == expected


@pytest.mark.parametrize(("track", "recorded"), TRACKS, ids=lambda p: str(p))
def test_the_recorded_hash_matches_the_committed_blob(track: Path, recorded: Path) -> None:
    # M1's claim, and the one a worktree check cannot make: the bytes git stores are the bytes the
    # hash was taken over. Without this, a `.gitattributes` that stopped applying would let a clone
    # rewrite line endings and hand the loader a track whose hash its own gate refuses.
    expected = (REPO / recorded).read_text(encoding="utf-8").strip()
    blob = _git("cat-file", "blob", f"HEAD:{track.as_posix()}")
    assert _sha256(blob) == expected, (
        f"{track} as committed does not match its recorded hash. If the track was just re-signed, "
        "the new bytes have to be committed; if it was not, something rewrote them."
    )


def _attributes(path: Path) -> dict[str, str]:
    """What git says the attributes are for one path.

    Asked of git rather than parsed out of `.gitattributes`: that file uses wildcards, so a reader
    matching literal paths would find nothing and report the attribute missing, and git's answer is
    the one that decides what happens to the bytes.
    """
    output = _git("check-attr", "-a", "--", path.as_posix()).decode("utf-8", "replace")
    fields: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split(": ")
        if len(parts) == 3:
            fields[parts[1].strip()] = parts[2].strip()
    return fields


@pytest.mark.parametrize(("track", "recorded"), TRACKS, ids=lambda p: str(p))
def test_the_paths_are_marked_non_text(track: Path, recorded: Path) -> None:
    # The attribute is the mechanism behind the two checks above, and losing it is silent: the
    # hashes stay right until somebody clones on a machine that rewrites line endings.
    for path in (track, recorded):
        fields = _attributes(path)
        assert fields, f"git reports no attributes at all for {path}"
        assert fields.get("text") == "unset", (
            f"{path} is not marked -text (git reports text={fields.get('text')!r}); "
            "git may rewrite its bytes"
        )
        assert fields.get("diff") == "unset", f"{path} should be marked -diff too"


def test_the_track_files_are_not_empty() -> None:
    # A hash of nothing matches a file of nothing; this is the one failure the hashes alone cannot
    # see, because both sides would agree.
    for track, _ in TRACKS:
        assert (REPO / track).stat().st_size > 0
