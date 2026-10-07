from __future__ import annotations


class Blocked(ValueError):
    """A guard failed: the forecast is withheld, never fabricated."""
