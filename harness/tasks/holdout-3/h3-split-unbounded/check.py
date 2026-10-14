from logline import parse

assert parse("2026-01-01 boot") == ("2026-01-01", "boot")
assert parse("2026-01-01 disk almost full") == ("2026-01-01", "disk almost full"), "message keeps its words"
print("ok")
