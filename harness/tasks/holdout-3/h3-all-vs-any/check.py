from policy import allowed

assert allowed(["admin"], ["admin", "user"])
assert allowed(["guest", "admin"], ["admin"]), "one allowed role is enough"
assert not allowed([], ["admin"]), "no roles granted"
print("ok")
