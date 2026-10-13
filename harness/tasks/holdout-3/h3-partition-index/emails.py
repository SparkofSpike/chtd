"""Email parsing."""


def domain(address):
    """The domain part of an email address."""
    return address.partition("@")[0]
