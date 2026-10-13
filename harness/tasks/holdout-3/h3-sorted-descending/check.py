from leaderboard import by_score_then_name

players = [
    {"score": 10, "name": "zoe"},
    {"score": 10, "name": "ann"},
    {"score": 12, "name": "bob"},
]
want = [
    {"score": 12, "name": "bob"},
    {"score": 10, "name": "ann"},
    {"score": 10, "name": "zoe"},
]
assert by_score_then_name(players) == want, by_score_then_name(players)
print("ok")
