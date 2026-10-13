"""Letter tallying."""
from collections import Counter


def most_common_letter(text):
    """The most common letter, or None for empty text."""
    counts = Counter(text)
    return counts.most_common(1)
