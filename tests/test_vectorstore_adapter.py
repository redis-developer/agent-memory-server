"""Tests for the VectorStore adapter functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.models import MemoryRecord, MemoryTypeEnum
from agent_memory_server.vectorstore_adapter import (
    LangChainVectorStoreAdapter,
    RedisVectorStoreAdapter,
    VectorStoreAdapter,
)
from agent_memory_server.vectorstore_factory import create_vectorstore_adapter


class TestVectorStoreAdapter:
    """Test cases for VectorStore adapter functionality."""

    def test_memory_to_document_conversion(self):
        """Test converting MemoryRecord to LangChain Document."""
        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Create a sample memory
        memory = MemoryRecord(
            text="This is a test memory",
            id="test-123",
            session_id="session-456",
            user_id="user-789",
            namespace="test",
            topics=["testing", "memory"],
            entities=["test"],
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # Convert to document
        doc = adapter.memory_to_document(memory)

        # Verify conversion
        assert doc.page_content == "This is a test memory"
        assert doc.metadata["id_"] == "test-123"
        assert doc.metadata["id"] == "test-123"
        assert doc.metadata["session_id"] == "session-456"
        assert doc.metadata["user_id"] == "user-789"
        assert doc.metadata["namespace"] == "test"
        assert doc.metadata["topics"] == ["testing", "memory"]
        assert doc.metadata["entities"] == ["test"]
        assert doc.metadata["memory_type"] == "semantic"

    def test_document_to_memory_conversion(self):
        """Test converting LangChain Document to MemoryRecordResult."""
        from langchain_core.documents import Document

        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Create a sample document
        doc = Document(
            page_content="This is a test memory",
            metadata={
                "id": "test-123",
                "session_id": "session-456",
                "user_id": "user-789",
                "namespace": "test",
                "topics": ["testing", "memory"],
                "entities": ["test"],
                "memory_type": "semantic",
                "created_at": "2024-01-01T00:00:00Z",
                "last_accessed": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
        )

        # Convert to memory
        memory_result = adapter.document_to_memory(doc, score=0.8)

        # Verify conversion
        assert memory_result.text == "This is a test memory"
        assert memory_result.id == "test-123"
        assert memory_result.session_id == "session-456"
        assert memory_result.user_id == "user-789"
        assert memory_result.namespace == "test"
        assert memory_result.topics == ["testing", "memory"]
        assert memory_result.entities == ["test"]
        assert memory_result.memory_type == "semantic"
        assert memory_result.dist == 0.8

    @pytest.mark.asyncio
    async def test_add_memories_with_mock_vectorstore(self):
        """Test adding memories to a mock vector store."""
        # Create a mock VectorStore with proper async mocking
        mock_vectorstore = MagicMock()
        mock_vectorstore.aadd_documents = AsyncMock(return_value=["doc1", "doc2"])
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Create sample memories
        memories = [
            MemoryRecord(
                text="Memory 1",
                id="mem1",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
            MemoryRecord(
                text="Memory 2",
                id="mem2",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
        ]

        # Add memories
        ids = await adapter.add_memories(memories)

        # Verify
        assert ids == ["doc1", "doc2"]
        mock_vectorstore.aadd_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_vectorstore_factory_creates_adapter(self):
        """Integration test: verify that the factory can create an adapter."""
        # Clear the global adapter to force recreation
        import agent_memory_server.vectorstore_factory

        agent_memory_server.vectorstore_factory._adapter = None

        # Test with Redis backend (default factory) - this uses actual settings
        adapter = create_vectorstore_adapter()

        # For Redis backend, we should get RedisVectorStoreAdapter (not LangChainVectorStoreAdapter)
        assert isinstance(adapter, RedisVectorStoreAdapter)

        # Reset the global adapter
        agent_memory_server.vectorstore_factory._adapter = None

        # Test with custom factory function that returns a VectorStore
        with (
            patch(
                "agent_memory_server.vectorstore_factory.create_embeddings"
            ) as mock_create_embeddings,
            patch(
                "agent_memory_server.vectorstore_factory._import_and_call_factory"
            ) as mock_import_factory,
            patch("agent_memory_server.vectorstore_factory.settings") as mock_settings,
        ):
            # Mock the embeddings
            mock_embeddings = MagicMock()
            mock_create_embeddings.return_value = mock_embeddings

            # Mock settings to use a non-Redis factory path
            mock_settings.vectorstore_factory = "my_module.create_custom_vectorstore"

            # Create a proper mock VectorStore that actually inherits from VectorStore
            from langchain_core.vectorstores import VectorStore

            class MockVectorStore(VectorStore):
                def add_texts(self, texts, metadatas=None, **kwargs):
                    return []

                def similarity_search(self, query, k=4, **kwargs):
                    return []

                @classmethod
                def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
                    return cls()

            mock_vectorstore = MockVectorStore()
            mock_import_factory.return_value = mock_vectorstore

            # Create adapter with mocked factory
            adapter = create_vectorstore_adapter()

            # For non-Redis backends, we should get LangChainVectorStoreAdapter
            assert isinstance(adapter, LangChainVectorStoreAdapter)
            assert adapter.vectorstore == mock_vectorstore
            assert adapter.embeddings == mock_embeddings

        # Test that factory function can also return VectorStoreAdapter directly
        agent_memory_server.vectorstore_factory._adapter = None

        with (
            patch(
                "agent_memory_server.vectorstore_factory.create_embeddings"
            ) as mock_create_embeddings,
            patch(
                "agent_memory_server.vectorstore_factory._import_and_call_factory"
            ) as mock_import_factory,
        ):
            # Mock the embeddings and custom adapter
            mock_embeddings = MagicMock()

            # Create a proper mock VectorStoreAdapter that actually inherits from VectorStoreAdapter
            class MockVectorStoreAdapter(VectorStoreAdapter):
                def __init__(self):
                    pass  # Skip parent constructor for test

                # Add minimal required methods for test
                async def add_memories(self, memories):
                    return []

                async def search_memories(self, query, **kwargs):
                    return []

                async def count_memories(self, **kwargs):
                    return 0

                async def delete_memories(self, memory_ids):
                    return 0

            mock_custom_adapter = MockVectorStoreAdapter()

            mock_create_embeddings.return_value = mock_embeddings
            mock_import_factory.return_value = mock_custom_adapter

            # Create adapter with mocked factory that returns adapter directly
            adapter = create_vectorstore_adapter()

            # Should get the custom adapter returned directly
            assert adapter == mock_custom_adapter

    def test_memory_hash_generation(self):
        """Test memory hash generation."""
        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Create a sample memory
        memory = MemoryRecord(
            text="This is a test memory",
            id="test-hash-123",
            user_id="user-123",
            session_id="session-456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # Generate hash
        hash1 = adapter.generate_memory_hash(memory)
        hash2 = adapter.generate_memory_hash(memory)

        # Verify hash is stable
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

        # Verify different memories produce different hashes
        different_memory = MemoryRecord(
            text="This is a different memory",
            id="test-hash-456",
            user_id="user-123",
            session_id="session-456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )
        different_hash = adapter.generate_memory_hash(different_memory)
        assert hash1 != different_hash

    @pytest.mark.asyncio
    async def test_empty_memories_handling(self):
        """Test handling of empty memory lists."""
        # Create a mock VectorStore
        mock_vectorstore = MagicMock()
        mock_embeddings = MagicMock()

        # Create adapter
        adapter = LangChainVectorStoreAdapter(mock_vectorstore, mock_embeddings)

        # Test adding empty list
        ids = await adapter.add_memories([])
        assert ids == []

        # Test deleting empty list
        deleted = await adapter.delete_memories([])
        assert deleted == 0
