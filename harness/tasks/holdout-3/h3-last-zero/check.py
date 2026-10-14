from windows import last

assert last(["a", "b", "c"], 2) == ["b", "c"], last(["a", "b", "c"], 2)
assert last(["a", "b", "c"], 0) == [], last(["a", "b", "c"], 0)
print("ok")
