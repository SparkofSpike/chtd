"""Report formatting."""


def describe(record):
    """One line describing the record, with a placeholder for a missing status."""
    return "{name}: {status}".format(**record)
