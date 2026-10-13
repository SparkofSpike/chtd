"""Grid totals."""


def column_totals(grid):
    """The sum of each column."""
    return [sum(row[i] for row in grid) for i in range(len(grid))]
