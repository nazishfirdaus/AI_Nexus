"""Map the logged-in Streamlit user (`st.user`) to an app account.

Deliberately split so the logic is testable without importing Streamlit:

  * `extract_claims(user)` works on any mapping/object exposing the OIDC claim
    attributes (`sub`, `email`, `name`, `is_logged_in`) - including a stub.
  * `resolve_user(user_store, user)` is the pure orchestration.
  * The actual `st.user` object is read in `app.py`, keeping this module free of
    a heavyweight Streamlit import.
"""
from __future__ import annotations

from typing import Optional


def extract_claims(user) -> dict:
    """Pull identity claims from an `st.user`-like object or mapping."""
    def _get(name):
        if user is None:
            return None
        if isinstance(user, dict):
            return user.get(name)
        try:
            return getattr(user, name, None)
        except Exception:  # pragma: no cover - Streamlit proxy edge cases
            return None

    email = _get("email") or ""
    return {
        "is_logged_in": bool(_get("is_logged_in")),
        "sub": _get("sub"),
        "email": email,
        "name": _get("name") or "",
    }


def resolve_user(user_store, user) -> Optional[dict]:
    """Return the account for `user`, provisioning one on first login.

    None covers two cases the caller treats identically: not logged in, and a
    disabled account.
    """
    claims = extract_claims(user)
    if not claims["is_logged_in"]:
        return None
    return user_store.resolve_login(claims["sub"], claims["email"], claims["name"])
