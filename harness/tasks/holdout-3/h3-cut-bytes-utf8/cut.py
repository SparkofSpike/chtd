"""Text truncation."""


def cut_bytes(text, limit):
    """The text cut to at most the given number of bytes."""
    return text.encode("utf-8")[:limit].decode("utf-8")
