"""Queue configuration."""


def first_entry(mapping):
    """The entry that was added first."""
    key = sorted(mapping)[0]
    return key, mapping[key]
