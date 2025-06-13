"""
Basic tests for the Agent Memory Client package.
"""

import pytest

from agent_memory_client import (
    MemoryAPIClient,
    MemoryClientConfig,
    MemoryClientError,
    MemoryValidationError,
    create_memory_client,
)
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum


def test_imports():
    """Test that all essential imports work."""
    assert MemoryAPIClient is not None
    assert MemoryClientConfig is not None
    assert create_memory_client is not None
    assert MemoryClientError is not None
    assert MemoryValidationError is not None


def test_client_config():
    """Test client configuration."""
    config = MemoryClientConfig(
        base_url="http://localhost:8000", timeout=30.0, default_namespace="test"
    )

    assert config.base_url == "http://localhost:8000"
    assert config.timeout == 30.0
    assert config.default_namespace == "test"


def test_client_creation():
    """Test client creation."""
    config = MemoryClientConfig(base_url="http://localhost:8000")
    client = MemoryAPIClient(config)

    assert client.config == config
    assert client._client is not None


def test_memory_record_creation():
    """Test creating memory records."""
    memory = ClientMemoryRecord(
        text="Test memory",
        memory_type=MemoryTypeEnum.SEMANTIC,
        topics=["test"],
        user_id="test-user",
    )

    assert memory.text == "Test memory"
    assert memory.memory_type == MemoryTypeEnum.SEMANTIC
    assert memory.topics == ["test"]
    assert memory.user_id == "test-user"
    assert memory.id is not None  # Should auto-generate


def test_validation_methods():
    """Test validation methods exist."""
    config = MemoryClientConfig(base_url="http://localhost:8000")
    client = MemoryAPIClient(config)

    # Test that validation methods exist
    assert hasattr(client, "validate_memory_record")
    assert hasattr(client, "validate_search_filters")
    assert hasattr(client, "_is_valid_ulid")


def test_enhanced_methods():
    """Test that enhanced methods exist."""
    config = MemoryClientConfig(base_url="http://localhost:8000")
    client = MemoryAPIClient(config)

    # Test lifecycle management
    assert hasattr(client, "promote_working_memories_to_long_term")

    # Test batch operations
    assert hasattr(client, "bulk_create_long_term_memories")

    # Test pagination
    assert hasattr(client, "search_all_long_term_memories")
    assert hasattr(client, "search_all_memories")

    # Test enhanced convenience methods
    assert hasattr(client, "update_working_memory_data")
    assert hasattr(client, "append_messages_to_working_memory")


@pytest.mark.asyncio
async def test_create_memory_client_function():
    """Test the create_memory_client helper function."""
    # This will fail to connect, but we can test that it creates the client
    with pytest.raises(MemoryClientError):
        await create_memory_client("http://nonexistent:8000")
