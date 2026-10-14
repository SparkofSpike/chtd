"""Status codes."""
from enum import Enum


class Status(Enum):
    OK = "ok"
    FAILED = "failed"


def describe(status):
    """A line describing the status."""
    return "status: " + status
