from tokens import MAGIC, is_magic

assert is_magic(MAGIC), "the module's own constant is the same object"
assert is_magic(int(str(MAGIC))), "the same value rebuilt"
print("ok")
