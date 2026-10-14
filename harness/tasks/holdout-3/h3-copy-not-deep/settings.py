"""Settings helpers."""
import copy


def add_tag(config, tag):
    """The config with the tag appended, without changing the original."""
    result = copy.copy(config)
    result["tags"].append(tag)
    return result
