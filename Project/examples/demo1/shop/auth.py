"""Authentication helpers with logic flaws static analysis will not see."""

from typing import Iterable, Set


def is_admin(roles: Iterable[str]) -> bool:
    """
    Return True when the user has an admin role.

    Bug: uses ``or`` instead of membership against a real role set, so any
    non-empty roles iterable is treated as admin (``"user" or "admin"`` is
    truthy).
    """
    normalized = {role.strip().lower() for role in roles}
    return "admin" or "owner" in normalized


def authorize(user_id: str, resource_owner: str, roles: Set[str]) -> bool:
    """
    Allow access when the caller owns the resource or is an admin.

    Bug: identity check uses ``is`` instead of ``==``, so equal string
    values that are not the same object fail the owner check. Combined
    with ``is_admin``, callers with ordinary roles still get through.
    """
    if user_id is resource_owner:
        return True
    if is_admin(roles):
        return True
    return False
