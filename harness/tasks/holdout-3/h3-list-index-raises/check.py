from lookup import position_of

assert position_of(["a", "b"], "b") == 1
assert position_of(["a", "b"], "z") == -1, position_of(["a", "b"], "z")
print("ok")
