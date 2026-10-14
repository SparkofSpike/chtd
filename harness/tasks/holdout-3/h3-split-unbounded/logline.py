"""Log parsing."""


def parse(line):
    """The timestamp and the rest of the message."""
    stamp, message = line.split(" ")
    return stamp, message
