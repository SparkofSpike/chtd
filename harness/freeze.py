#!/usr/bin/env python3
"""Freeze and verify CHTD's co-harness track.

The track is only a control if its bytes cannot drift. This script records the
sha256 of a preset file and verifies it later, mirroring how Agnes Harness freezes
its own `minimal-rl` baseline: the comparison is over raw file bytes, not a parse,
so a reordered key, a changed comment and a changed value all fail alike.

Usage:
    python harness/freeze.py check     # verify the recorded hash matches
    python harness/freeze.py write     # record the hash of the current bytes

A drift means the experiment changed. Re-recording is a deliberate act: do it in
its own commit, and say in the message why the track changed.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PRESET_DIR = Path(__file__).resolve().parent / "presets"
TRACK = "chtd"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> int:
    """Verify every frozen preset. Returns a process exit code."""
    preset = PRESET_DIR / f"{TRACK}.yaml"
    recorded = PRESET_DIR / f"{TRACK}.sha256"

    if not preset.exists():
        print(f"FAIL: preset missing: {preset}", file=sys.stderr)
        return 1
    if not recorded.exists():
        print(
            f"FAIL: no recorded hash for {TRACK}; run `python harness/freeze.py write`",
            file=sys.stderr,
        )
        return 1

    expected = recorded.read_text(encoding="utf-8").strip()
    actual = digest(preset)
    if actual != expected:
        print(
            f"FAIL: {preset.name} has drifted.\n"
            f"  recorded: {expected}\n"
            f"  actual:   {actual}\n"
            "The track changed, so the experiment changed. If that is intended,\n"
            "re-record in its own commit and state why.",
            file=sys.stderr,
        )
        return 1

    print(f"ok: {preset.name} matches its recorded hash ({actual[:12]}...)")
    return 0


def write() -> int:
    """Record the current hash. Refuses to silently bless a missing preset."""
    preset = PRESET_DIR / f"{TRACK}.yaml"
    if not preset.exists():
        print(f"FAIL: preset missing: {preset}", file=sys.stderr)
        return 1
    value = digest(preset)
    (PRESET_DIR / f"{TRACK}.sha256").write_text(value + "\n", encoding="utf-8")
    print(f"recorded {preset.name} -> {value}")
    return 0


def main(argv: list[str]) -> int:
    action = argv[1] if len(argv) > 1 else "check"
    if action == "check":
        return check()
    if action == "write":
        return write()
    print(f"usage: {argv[0]} [check|write]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
