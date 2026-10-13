from temperature import Reading, bump

reading = Reading(20)
assert bump(reading, 5) == 25, bump(reading, 5)
print("ok")
