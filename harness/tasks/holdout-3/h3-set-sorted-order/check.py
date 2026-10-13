from dedupe import unique_first

assert unique_first(["b", "a", "b"]) == ["b", "a"], unique_first(["b", "a", "b"])
print("ok")
