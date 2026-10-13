"""Temperature readings."""


class Reading:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        """The reading in Celsius."""
        return self._celsius


def bump(reading, delta):
    """Increase the reading by delta degrees."""
    reading.celsius += delta
    return reading.celsius
