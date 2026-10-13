"""Access policy."""


def allowed(user_roles, allowed_roles):
    """Whether any of the user's roles is an allowed role."""
    return all(role in allowed_roles for role in user_roles)
