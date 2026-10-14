from status import Status, describe

assert describe(Status.OK) == "status: ok", describe(Status.OK)
print("ok")
