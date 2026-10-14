"""Leaderboard ordering."""


def by_score_then_name(players):
    """Players sorted by score descending, then name ascending."""
    return sorted(players, key=lambda p: (p["score"], p["name"]))
