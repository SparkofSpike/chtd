from csvline import line

assert line(["a", "b"]) == "a,b"
assert line(["a", 3]) == "a,3", line(["a", 3])
print("ok")
