"""
Filter classes for search operations.

These filters allow for filtering memory search results.
"""

from datetime import datetime

from pydantic import BaseModel


class BaseFilter(BaseModel):
    """Base class for all filters"""

    pass


class SessionId(BaseFilter):
    """Filter by session ID"""

    eq: str | None = None
    in_: list[str] | None = None
    not_eq: str | None = None
    not_in: list[str] | None = None


class Namespace(BaseFilter):
    """Filter by namespace"""

    eq: str | None = None
    in_: list[str] | None = None
    not_eq: str | None = None
    not_in: list[str] | None = None


class UserId(BaseFilter):
    """Filter by user ID"""

    eq: str | None = None
    in_: list[str] | None = None
    not_eq: str | None = None
    not_in: list[str] | None = None


class Topics(BaseFilter):
    """Filter by topics"""

    any: list[str] | None = None
    all: list[str] | None = None
    none: list[str] | None = None


class Entities(BaseFilter):
    """Filter by entities"""

    any: list[str] | None = None
    all: list[str] | None = None
    none: list[str] | None = None


class CreatedAt(BaseFilter):
    """Filter by creation date"""

    gte: datetime | None = None
    lte: datetime | None = None
    eq: datetime | None = None


class LastAccessed(BaseFilter):
    """Filter by last accessed date"""

    gte: datetime | None = None
    lte: datetime | None = None
    eq: datetime | None = None


class EventDate(BaseFilter):
    """Filter by event date"""

    gte: datetime | None = None
    lte: datetime | None = None
    eq: datetime | None = None


class MemoryType(BaseFilter):
    """Filter by memory type"""

    eq: str | None = None
    in_: list[str] | None = None
    not_eq: str | None = None
    not_in: list[str] | None = None
