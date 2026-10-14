from scores import best

assert best([3, 9, 4]) == 9
assert best([]) is None, "no scores has no best"
print("ok")
