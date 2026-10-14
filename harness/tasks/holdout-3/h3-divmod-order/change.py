"""Making change."""


def split(total, unit):
    """How many whole units fit, and what remains."""
    remainder, count = divmod(total, unit)
    return count, remainder
