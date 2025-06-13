"""
Exception classes for the Agent Memory Client.
"""


class MemoryClientError(Exception):
    """Base exception for all memory client errors."""

    pass


class MemoryValidationError(MemoryClientError, ValueError):
    """Raised when memory record or filter validation fails.

    Subclassing ``ValueError`` ensures that client code (and our test suite)
    can catch validation issues using the built-in exception while still
    signaling a distinct, library-specific error type when desired.
    """

    pass


class MemoryNotFoundError(MemoryClientError):
    """Raised when a requested memory or session is not found."""

    pass


class MemoryServerError(MemoryClientError):
    """Raised when the memory server returns an error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
