from cut import cut_bytes

assert cut_bytes("hello", 3) == "hel", cut_bytes("hello", 3)
assert cut_bytes("h\u00e9llo", 2) == "h", cut_bytes("h\u00e9llo", 2)
print("ok")
