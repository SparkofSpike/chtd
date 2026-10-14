from queue_config import first_entry

settings = {}
settings["zeta"] = 1
settings["alpha"] = 2
assert first_entry(settings) == ("zeta", 1), first_entry(settings)
print("ok")
