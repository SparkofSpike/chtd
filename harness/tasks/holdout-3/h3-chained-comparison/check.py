from bounds import within

assert within(0, 5, 10), "inside the range"
assert not within(0, 15, 10), "above the range"
print("ok")
