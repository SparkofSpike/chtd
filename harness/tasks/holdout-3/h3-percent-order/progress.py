"""Progress display."""


def percent(done, total):
    """The completion percentage, rounded down, as an integer."""
    return done // total * 100
