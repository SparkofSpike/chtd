"""Hashing for recorded artifacts, so a record is a property of its content.

A digest a reader cannot reproduce is not evidence. The first control-arm
snapshot recorded its text artifacts over the bytes of one machine's working
tree — CRLF, because that machine had `core.autocrlf=true` — while git stores
LF. The rebuild command failed on every checkout except the one that produced
it, and the same commit hashed differently on Windows and on Linux.

So the rule here is: **text artifacts are hashed over their bytes with CRLF
normalised to LF**, and binary artifacts are hashed raw. A `\\r\\n` inside the
SQLite ledger is data; a `\\r\\n` between two JSON lines is a line ending, and
which one a checkout hands you is a property of the checkout, not of the
record. Normalising on the read side makes every checkout reproduce the same
digest, whatever `core.autocrlf` and git's text filters did on the way in.

New products should still be written with real LF newlines (``newline="\\n"``)
so the physical bytes and the canonical bytes agree; the normalisation here is
what makes the old, CRLF-era recordings checkable, not a licence to write them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def canonical_bytes(path: Path) -> bytes:
    """A text file's bytes with CRLF normalised to LF."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def text_sha256(path: Path) -> str:
    """sha256 of a text file, over its newline-canonicalised bytes."""
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def raw_sha256(path: Path) -> str:
    """sha256 of a binary file, over its bytes exactly as stored."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
