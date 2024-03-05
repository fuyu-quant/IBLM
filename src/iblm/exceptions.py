from __future__ import annotations


__all__ = [
    "InvalidModelObjectiveError",
    "InvalidCodeModelError",
    "UndefinedCodeModelError",
    "InvalidAPIType",
    "InvalidAPIOption",
]


class InvalidModelObjectiveError(Exception):
    pass


class InvalidCodeModelError(Exception):
    pass


class UndefinedCodeModelError(Exception):
    pass


class InvalidAPIType(Exception):
    pass


class InvalidAPIOption(Exception):
    pass
