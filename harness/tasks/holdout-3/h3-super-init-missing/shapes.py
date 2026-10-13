"""Shapes."""


class Shape:
    def __init__(self, name):
        self.name = name


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius


def describe(shape):
    """'name: radius' for the shape."""
    return f"{shape.name}: {shape.radius}"
