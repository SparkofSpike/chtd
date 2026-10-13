from settings import add_tag

config = {"name": "app", "tags": ["a"]}
updated = add_tag(config, "b")

assert config["tags"] == ["a"], config["tags"]
assert updated["tags"] == ["a", "b"], updated["tags"]
print("ok")
