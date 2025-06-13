"""
Agent Memory Client

A Python client for the Agent Memory Server REST API providing comprehensive
memory management capabilities for AI agents and applications.
"""

__version__ = "0.9.0"

from .client import MemoryAPIClient, MemoryClientConfig, create_memory_client
from .exceptions import (
    MemoryClientError,
    MemoryNotFoundError,
    MemoryServerError,
    MemoryValidationError,
)
from .models import (
    # Re-export essential models for convenience
    ModelNameLiteral,
)

__all__ = [
    # Client classes
    "MemoryAPIClient",
    "MemoryClientConfig",
    "create_memory_client",
    # Exceptions
    "MemoryClientError",
    "MemoryValidationError",
    "MemoryNotFoundError",
    "MemoryServerError",
    # Types
    "ModelNameLiteral",
]
