from nearest import nearest

assert nearest(2.4) == 2
assert isinstance(nearest(2.4), int), "the result should be an int, got " + type(nearest(2.4)).__name__
print("ok")
