"""File name helpers."""


def stem(filename):
    """The file name without its extension."""
    return filename[:filename.find(".")]
