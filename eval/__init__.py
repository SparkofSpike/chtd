"""Evaluation: measure the arms, keep the comparison honest, and report what survived.

The design's deciding comparison is not "does the student improve" but "does sharing the harness
cause the improvement", so this package is built around two things a single-arm evaluation cannot
do: comparing a baseline arm against the CHTD arm under identical conditions, and reporting a gain
that only exists on the training track as scaffold overfit rather than as a capability.
"""

from __future__ import annotations

__all__: list[str] = []
