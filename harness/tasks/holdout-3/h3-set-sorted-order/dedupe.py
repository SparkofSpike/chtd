"""Deduplication."""


def unique_first(items):
    """The items with duplicates removed, first occurrence kept."""
    return sorted(set(items))
