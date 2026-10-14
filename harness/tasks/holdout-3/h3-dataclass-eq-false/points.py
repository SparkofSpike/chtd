"""Points."""
from dataclasses import dataclass


@dataclass(eq=False)
class Point:
    x: int
    y: int
