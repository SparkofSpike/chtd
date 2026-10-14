"""Bounds checking."""


def within(low, value, high):
    """Whether the value lies strictly inside the bounds."""
    return low < value > high
