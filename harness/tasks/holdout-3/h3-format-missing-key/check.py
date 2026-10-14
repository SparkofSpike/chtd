from report import describe

assert describe({"name": "api", "status": "up"}) == "api: up"
assert describe({"name": "db"}) == "db: unknown", describe({"name": "db"})
print("ok")
