"""Dependency probing helpers.

``importlib.util.find_spec`` is the cheap way to ask "is this installed?"
without paying for the import or triggering its side effects. It is not
totally safe on its own: it raises if a *parent* package cannot be imported,
which happens for real on robots (a half-installed ROS overlay, a compiled
extension built against the wrong numpy). A probe that throws defeats the
purpose of guarding the import in the first place.
"""

from __future__ import annotations

import importlib.util

__all__ = ["module_available"]


def module_available(name: str) -> bool:
    """True if ``name`` can be located by the import system.

    Never raises: any failure to even *look* for the module is reported as
    "not available", which is the answer the caller needs anyway.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False
