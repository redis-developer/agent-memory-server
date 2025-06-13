import time
from datetime import UTC, datetime
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import ulid
from redis.commands.search.document import Document

from agent_memory_server.filters import Namespace, SessionId
from agent_memory_server.long_term_memory import (
    compact_long_term_memories,
    count_long_term_memories,
    deduplicate_by_hash,
    deduplicate_by_id,
    extract_memory_structure,
    generate_memory_hash,
    index_long_term_memories,
    merge_memories_with_llm,
    promote_working_memory_to_long_term,
    search_long_term_memories,
    search_memories,
)
from agent_memory_server.models import MemoryRecord, MemoryRecordResult, MemoryTypeEnum
from agent_memory_server.utils.redis import ensure_search_index_exists


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_index_memories(
        self, mock_openai_client, mock_async_redis_client, session
    ):
        """Test indexing messages"""
        long_term_memories = [
            MemoryRecord(
                id="memory-1", text="Paris is the capital of France", session_id=session
            ),
            MemoryRecord(
                id="memory-2", text="France is a country in Europe", session_id=session
            ),
        ]

        # Create two separate embedding vectors
        mock_vectors = [
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes(),
            np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32).tobytes(),
        ]

        mock_vectorizer = MagicMock()
        mock_vectorizer.aembed_many = AsyncMock(return_value=mock_vectors)

        mock_async_redis_client.hset = AsyncMock()

        with mock.patch(
            "agent_memory_server.long_term_memory.OpenAITextVectorizer",
            return_value=mock_vectorizer,
        ):
            await index_long_term_memories(
                long_term_memories,
                redis_client=mock_async_redis_client,
            )

        # Check that create_embedding was called with the right arguments
        contents = [memory.text for memory in long_term_memories]
        mock_vectorizer.aembed_many.assert_called_with(
            contents,
            batch_size=20,
            as_buffer=True,
        )

        # Verify one of the calls to make sure the data is correct
        for i, call in enumerate(mock_async_redis_client.hset.call_args_list):
            args, kwargs = call

            # Check that the key starts with the memory key prefix
            assert args[0].startswith("memory:")

            # Check that the mapping contains the essential keys
            mapping = kwargs["mapping"]
            assert mapping["text"] == long_term_memories[i].text
            assert (
                mapping["id_"] == long_term_memories[i].id
            )  # id_ is the internal Redis field
            assert mapping["session_id"] == long_term_memories[i].session_id
            assert mapping["user_id"] == long_term_memories[i].user_id
            assert "last_accessed" in mapping
            assert "created_at" in mapping
            assert mapping["vector"] == mock_vectors[i]

    @pytest.mark.asyncio
    async def test_search_memories(self, mock_openai_client, mock_async_redis_client):
        """Test searching memories"""
        # Set up the mock embedding response
        mock_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mock_vectorizer = MagicMock()
        mock_vectorizer.aembed = AsyncMock(return_value=mock_vector)

        class MockResult:
            def __init__(self, docs):
                self.total = len(docs)
                self.docs = docs

        mock_now = time.time()

        mock_query = AsyncMock()
        # Return a list of documents directly instead of a MockResult object
        mock_query.return_value = [
            Document(
                id=b"doc1",
                id_=str(ulid.new()),
                text=b"Hello, world!",
                vector_distance=0.25,
                created_at=mock_now,
                last_accessed=mock_now,
                user_id=None,
                session_id=None,
                namespace=None,
                topics=None,
                entities=None,
            ),
            Document(
                id=b"doc2",
                id_=str(ulid.new()),
                text=b"Hi there!",
                vector_distance=0.75,
                created_at=mock_now,
                last_accessed=mock_now,
                user_id=None,
                session_id=None,
                namespace=None,
                topics=None,
                entities=None,
            ),
        ]

        mock_index = MagicMock()
        mock_index.query = mock_query

        query = "What is the meaning of life?"
        session_id = SessionId(eq="test-session")

        with (
            mock.patch(
                "agent_memory_server.long_term_memory.OpenAITextVectorizer",
                return_value=mock_vectorizer,
            ),
            mock.patch(
                "agent_memory_server.long_term_memory.get_search_index",
                return_value=mock_index,
            ),
        ):
            results = await search_long_term_memories(
                query,
                mock_async_redis_client,
                session_id=session_id,
            )

        # Check that create_embedding was called with the right arguments
        mock_vectorizer.aembed.assert_called_with(query)

        assert mock_index.query.call_count == 1

        assert len(results.memories) == 1
        assert isinstance(results.memories[0], MemoryRecordResult)
        assert results.memories[0].text == "Hello, world!"
        assert results.memories[0].dist == 0.25
        assert results.memories[0].memory_type == "message"

    @pytest.mark.asyncio
    async def test_search_memories_unified_search(self, mock_async_redis_client):
        """Test unified search across working memory and long-term memory"""

        from agent_memory_server.models import (
            MemoryRecordResults,
            WorkingMemory,
        )

        # Mock search_long_term_memories to return some long-term results
        mock_long_term_results = MemoryRecordResults(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="long-term-1",
                    text="Long-term: User likes coffee",
                    dist=0.3,
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    created_at=datetime.fromtimestamp(1000),
                    updated_at=datetime.fromtimestamp(1000),
                    last_accessed=datetime.fromtimestamp(1000),
                )
            ],
        )

        # Mock working memory with matching content
        test_working_memory = WorkingMemory(
            session_id="test-session",
            namespace="test",
            messages=[],
            memories=[
                MemoryRecord(
                    id="working-1",
                    text="Working memory: coffee preferences",
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    persisted_at=None,
                )
            ],
        )

        # Mock the search and working memory functions
        with (
            patch(
                "agent_memory_server.long_term_memory.search_long_term_memories"
            ) as mock_search_lt,
            patch(
                "agent_memory_server.working_memory.get_working_memory"
            ) as mock_get_wm,
            patch(
                "agent_memory_server.working_memory.list_sessions"
            ) as mock_list_sessions,
        ):
            mock_search_lt.return_value = mock_long_term_results
            mock_get_wm.return_value = test_working_memory
            # Mock list_sessions to return a list of session IDs
            mock_list_sessions.return_value = (1, ["test-session"])

            # Test unified search WITHOUT providing session_id to avoid the namespace_value bug
            results = await search_memories(
                text="coffee",
                redis=mock_async_redis_client,
                namespace=Namespace(eq="test"),
                limit=10,
                include_working_memory=True,
                include_long_term_memory=True,
            )

            # Verify both long-term and working memory were searched
            mock_search_lt.assert_called_once()
            mock_get_wm.assert_called_once()
            mock_list_sessions.assert_called_once()

            # Check results contain both types
            assert len(results.memories) == 2
            long_term_result = next(
                r for r in results.memories if "Long-term" in r.text
            )
            working_result = next(
                r for r in results.memories if "Working memory" in r.text
            )

            assert long_term_result.text == "Long-term: User likes coffee"
            assert working_result.text == "Working memory: coffee preferences"

    @pytest.mark.asyncio
    async def test_deduplicate_by_id(self, mock_async_redis_client):
        """Test deduplication by id"""
        memory = MemoryRecord(
            text="Test memory",
            id="test-id",
            session_id="test-session",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # Test case 1: Memory doesn't exist
        mock_async_redis_client.execute_command = AsyncMock(return_value=[0])

        result_memory, overwrite = await deduplicate_by_id(
            memory, redis_client=mock_async_redis_client
        )

        assert result_memory == memory
        assert overwrite is False

        # Test case 2: Memory exists
        mock_async_redis_client.execute_command = AsyncMock(
            return_value=[1, "memory:existing-key", "1234567890"]
        )
        mock_async_redis_client.delete = AsyncMock()

        result_memory, overwrite = await deduplicate_by_id(
            memory, redis_client=mock_async_redis_client
        )

        assert result_memory == memory
        assert overwrite is True
        mock_async_redis_client.delete.assert_called_once_with("memory:existing-key")

    def test_generate_memory_hash(self):
        """Test memory hash generation"""
        memory1 = {
            "text": "Hello world",
            "user_id": "user123",
            "session_id": "session456",
        }

        memory2 = {
            "text": "Hello world",
            "user_id": "user123",
            "session_id": "session456",
        }

        memory3 = {
            "text": "Different text",
            "user_id": "user123",
            "session_id": "session456",
        }

        # Same content should produce same hash
        hash1 = generate_memory_hash(memory1)
        hash2 = generate_memory_hash(memory2)
        assert hash1 == hash2

        # Different content should produce different hash
        hash3 = generate_memory_hash(memory3)
        assert hash1 != hash3

        # Test with missing fields
        memory4 = {"text": "Hello world"}
        hash4 = generate_memory_hash(memory4)
        assert hash4 != hash1  # Should be different when fields are missing

    @pytest.mark.asyncio
    async def test_extract_memory_structure(self, mock_async_redis_client):
        """Test memory structure extraction"""
        with (
            patch(
                "agent_memory_server.long_term_memory.get_redis_conn"
            ) as mock_get_redis,
            patch(
                "agent_memory_server.long_term_memory.handle_extraction"
            ) as mock_extract,
        ):
            # Set up proper async mocks
            mock_redis = AsyncMock()
            mock_get_redis.return_value = mock_redis
            mock_extract.return_value = (["topic1", "topic2"], ["entity1", "entity2"])

            await extract_memory_structure(
                "test-id", "Test text content", "test-namespace"
            )

            # Verify extraction was called
            mock_extract.assert_called_once_with("Test text content")

            # Verify Redis was updated with topics and entities
            mock_redis.hset.assert_called_once()
            args, kwargs = mock_redis.hset.call_args

            # Check the key format - it includes namespace in the key structure
            assert "memory:" in args[0] and "test-id" in args[0]

            # Check the mapping
            mapping = kwargs["mapping"]
            assert mapping["topics"] == "topic1,topic2"
            assert mapping["entities"] == "entity1,entity2"

    @pytest.mark.asyncio
    async def test_count_long_term_memories(self, mock_async_redis_client):
        """Test counting long-term memories"""

        # Mock execute_command for both FT.INFO and FT.SEARCH
        def mock_execute_command(command):
            if command.startswith("FT.INFO"):
                # Return success for index info check
                return {"num_docs": 42}
            if command.startswith("FT.SEARCH"):
                # Return search results with count as first element
                return [42]  # Total count
            return []

        mock_async_redis_client.execute_command = AsyncMock(
            side_effect=mock_execute_command
        )

        count = await count_long_term_memories(
            namespace="test-namespace",
            user_id="test-user",
            session_id="test-session",
            redis_client=mock_async_redis_client,
        )

        assert count == 42

        # Verify the execute_command was called
        assert mock_async_redis_client.execute_command.call_count >= 1

    @pytest.mark.asyncio
    async def test_deduplicate_by_hash(self, mock_async_redis_client):
        """Test deduplication by hash"""
        memory = MemoryRecord(
            id="test-memory-1",
            text="Test memory",
            session_id="test-session",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # Test case 1: No duplicate found
        # Mock Redis execute_command to return 0 results
        mock_async_redis_client.execute_command = AsyncMock(return_value=[0])

        result_memory, overwrite = await deduplicate_by_hash(
            memory, redis_client=mock_async_redis_client
        )

        assert result_memory == memory
        assert overwrite is False

        # Test case 2: Duplicate found
        # Mock Redis execute_command to return 1 result (return bytes like real Redis)
        mock_async_redis_client.execute_command = AsyncMock(
            return_value=[1, b"memory:existing-key", b"existing-id-123"]
        )

        # Mock the hset call that updates last_accessed
        mock_async_redis_client.hset = AsyncMock()

        result_memory, overwrite = await deduplicate_by_hash(
            memory, redis_client=mock_async_redis_client
        )

        # Should return None (duplicate found) and overwrite=True
        assert result_memory is None
        assert overwrite is True
        # Verify the last_accessed timestamp was updated
        mock_async_redis_client.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_memories_with_llm(self):
        """Test merging memories with LLM"""
        memories = [
            {
                "text": "User likes coffee",
                "topics": ["coffee", "preferences"],
                "entities": ["user"],
                "created_at": 1000,
                "last_accessed": 1500,
                "namespace": "test",
                "user_id": "user123",
                "session_id": "session456",
                "memory_type": "semantic",
                "discrete_memory_extracted": "t",
            },
            {
                "text": "User enjoys drinking coffee in the morning",
                "topics": ["coffee", "morning"],
                "entities": ["user"],
                "created_at": 1200,
                "last_accessed": 1600,
                "namespace": "test",
                "user_id": "user123",
                "session_id": "session456",
                "memory_type": "semantic",
                "discrete_memory_extracted": "t",
            },
        ]

        # Mock LLM client
        mock_llm_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "User enjoys drinking coffee, especially in the morning"
        mock_llm_client.create_chat_completion.return_value = mock_response

        merged = await merge_memories_with_llm(memories, llm_client=mock_llm_client)

        # Check merged content
        assert "coffee" in merged["text"].lower()
        assert merged["created_at"] == 1000  # Earliest timestamp
        assert merged["last_accessed"] == 1600  # Latest timestamp
        assert set(merged["topics"]) == {"coffee", "preferences", "morning"}
        assert set(merged["entities"]) == {"user"}
        assert merged["user_id"] == "user123"
        assert merged["session_id"] == "session456"
        assert merged["namespace"] == "test"
        assert "memory_hash" in merged

        # Test single memory case
        single_memory = memories[0]
        result = await merge_memories_with_llm([single_memory])
        assert result == single_memory

    @pytest.mark.asyncio
    async def test_compact_long_term_memories(self, mock_async_redis_client):
        """Test compacting long-term memories"""
        # Mock Redis search to return some memories for compaction
        mock_doc1 = MagicMock()
        mock_doc1.id = "memory:id1:namespace"
        mock_doc1.memory_hash = "hash1"
        mock_doc1.text = "User likes coffee"

        mock_doc2 = MagicMock()
        mock_doc2.id = "memory:id2:namespace"
        mock_doc2.memory_hash = "hash1"  # Same hash - duplicate
        mock_doc2.text = "User enjoys coffee"

        # Mock the search results for the initial memory search
        mock_search_result = MagicMock()
        mock_search_result.docs = [mock_doc1, mock_doc2]
        mock_search_result.total = 2

        mock_ft = MagicMock()
        mock_ft.search.return_value = mock_search_result
        mock_async_redis_client.ft.return_value = mock_ft

        # Mock the execute_command for both index operations and final count
        def mock_execute_command(command):
            if "FT.SEARCH" in command and "memory_hash" in command:
                # Hash-based duplicate search - return 0 (no hash duplicates)
                return [0]
            if "FT.SEARCH" in command and "LIMIT 0 0" in command:
                # Final count query - return 2
                return [2]
            return [0]

        mock_async_redis_client.execute_command = AsyncMock(
            side_effect=mock_execute_command
        )

        # Mock LLM client
        mock_llm_client = AsyncMock()

        with (
            patch(
                "agent_memory_server.long_term_memory.get_model_client"
            ) as mock_get_client,
            patch(
                "agent_memory_server.long_term_memory.merge_memories_with_llm"
            ) as mock_merge,
            patch(
                "agent_memory_server.long_term_memory.index_long_term_memories"
            ) as mock_index,
        ):
            mock_get_client.return_value = mock_llm_client
            mock_merge.return_value = {
                "text": "Merged: User enjoys coffee",
                "id_": "merged-id",
                "memory_hash": "new-hash",
                "created_at": 1000,
                "last_accessed": 1500,
                "updated_at": 1600,
                "user_id": None,
                "session_id": None,
                "namespace": "test",
                "topics": ["coffee"],
                "entities": ["user"],
                "memory_type": "semantic",
                "discrete_memory_extracted": "t",
            }

            # Mock deletion and indexing
            mock_async_redis_client.delete = AsyncMock()
            mock_index.return_value = None

            remaining_count = await compact_long_term_memories(
                namespace="test",
                redis_client=mock_async_redis_client,
                llm_client=mock_llm_client,
                compact_hash_duplicates=True,
                compact_semantic_duplicates=False,  # Test hash duplicates only
            )

            # Since the hash search returns 0 duplicates, merge should not be called
            # This tests the "no duplicates found" path
            mock_merge.assert_not_called()

            # Should return count from final search
            assert remaining_count == 2  # Mocked total

    @pytest.mark.asyncio
    async def test_promote_working_memory_to_long_term(self, mock_async_redis_client):
        """Test promoting memories from working memory to long-term storage"""

        from agent_memory_server.models import (
            MemoryRecord,
            MemoryTypeEnum,
            WorkingMemory,
        )

        # Create test memories - mix of persisted and unpersisted
        persisted_memory = MemoryRecord(
            text="Already persisted memory",
            id="persisted-id",
            namespace="test",
            memory_type=MemoryTypeEnum.SEMANTIC,
            persisted_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        )

        unpersisted_memory1 = MemoryRecord(
            text="Unpersisted memory 1",
            id="unpersisted-1",
            namespace="test",
            memory_type=MemoryTypeEnum.SEMANTIC,
            persisted_at=None,
        )

        unpersisted_memory2 = MemoryRecord(
            text="Unpersisted memory 2",
            id="unpersisted-2",
            namespace="test",
            memory_type=MemoryTypeEnum.SEMANTIC,
            persisted_at=None,
        )

        test_working_memory = WorkingMemory(
            session_id="test-session",
            namespace="test",
            messages=[],
            memories=[persisted_memory, unpersisted_memory1, unpersisted_memory2],
        )

        with (
            patch("agent_memory_server.working_memory.get_working_memory") as mock_get,
            patch("agent_memory_server.working_memory.set_working_memory") as mock_set,
            patch(
                "agent_memory_server.long_term_memory.deduplicate_by_id"
            ) as mock_dedup,
            patch(
                "agent_memory_server.long_term_memory.index_long_term_memories"
            ) as mock_index,
        ):
            # Setup mocks
            mock_get.return_value = test_working_memory
            mock_set.return_value = None
            mock_dedup.side_effect = [
                (unpersisted_memory1, False),  # First call - no overwrite
                (unpersisted_memory2, False),  # Second call - no overwrite
            ]
            mock_index.return_value = None

            # Call the promotion function
            promoted_count = await promote_working_memory_to_long_term(
                session_id="test-session",
                namespace="test",
                redis_client=mock_async_redis_client,
            )

            # Verify results
            assert promoted_count == 2

            # Verify working memory was retrieved
            mock_get.assert_called_once_with(
                session_id="test-session",
                namespace="test",
                redis_client=mock_async_redis_client,
            )

            # Verify deduplication was called for unpersisted memories
            assert mock_dedup.call_count == 2

            # Verify indexing was called for unpersisted memories
            assert mock_index.call_count == 2

            # Verify working memory was updated with new timestamps
            mock_set.assert_called_once()
            updated_memory = mock_set.call_args[1]["working_memory"]

            # Check that the unpersisted memories now have persisted_at set
            unpersisted_memories_updated = [
                mem
                for mem in updated_memory.memories
                if mem.id in ["unpersisted-1", "unpersisted-2"]
            ]
            assert len(unpersisted_memories_updated) == 2
            for mem in unpersisted_memories_updated:
                assert mem.persisted_at is not None
                assert isinstance(mem.persisted_at, datetime)

            # Check that already persisted memory was unchanged
            persisted_memories = [
                mem for mem in updated_memory.memories if mem.id == "persisted-id"
            ]
            assert len(persisted_memories) == 1
            assert persisted_memories[0].persisted_at == persisted_memory.persisted_at

        # Now test client resubmission scenario
        # Simulate client resubmitting stale state with new memory
        resubmitted_memory = WorkingMemory(
            session_id="test-session",
            namespace="test",
            messages=[],
            memories=[
                # Existing memory resubmitted without persisted_at (client doesn't track this)
                MemoryRecord(
                    text="Unpersisted memory 1",
                    id="unpersisted-1",  # Same id as before
                    namespace="test",
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    persisted_at=None,  # Client doesn't know about server timestamps
                ),
                # New memory from client
                MemoryRecord(
                    text="New memory from client",
                    id="new-memory-3",
                    namespace="test",
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    persisted_at=None,
                ),
            ],
        )

        with (
            patch("agent_memory_server.working_memory.get_working_memory") as mock_get2,
            patch("agent_memory_server.working_memory.set_working_memory") as mock_set2,
            patch(
                "agent_memory_server.long_term_memory.deduplicate_by_id"
            ) as mock_dedup2,
            patch(
                "agent_memory_server.long_term_memory.index_long_term_memories"
            ) as mock_index2,
        ):
            # Setup mocks for resubmission scenario
            mock_get2.return_value = resubmitted_memory
            mock_set2.return_value = None
            # First call: existing memory found (overwrite)
            # Second call: new memory, no existing (no overwrite)
            mock_dedup2.side_effect = [
                (resubmitted_memory.memories[0], True),  # Overwrite existing
                (resubmitted_memory.memories[1], False),  # New memory
            ]
            mock_index2.return_value = None

            # Call promotion again
            promoted_count_2 = await promote_working_memory_to_long_term(
                session_id="test-session",
                namespace="test",
                redis_client=mock_async_redis_client,
            )

            # Both memories should be promoted (one overwrite, one new)
            assert promoted_count_2 == 2

            # Verify final working memory state
            mock_set2.assert_called_once()
            final_memory = mock_set2.call_args[1]["working_memory"]

            # Both memories should have persisted_at set
            for mem in final_memory.memories:
                assert mem.persisted_at is not None

            # This demonstrates that:
            # 1. Client can safely resubmit stale state
            # 2. Server handles id-based overwrites correctly
            # 3. Working memory converges to consistent state with proper timestamps


@pytest.mark.requires_api_keys
class TestLongTermMemoryIntegration:
    """Integration tests for long-term memory"""

    @pytest.mark.asyncio
    async def test_search_messages(self, async_redis_client):
        """Test searching messages"""
        await ensure_search_index_exists(async_redis_client)

        long_term_memories = [
            MemoryRecord(
                id="memory-1", text="Paris is the capital of France", session_id="123"
            ),
            MemoryRecord(
                id="memory-2", text="France is a country in Europe", session_id="123"
            ),
        ]

        with mock.patch(
            "agent_memory_server.long_term_memory.get_redis_conn",
            return_value=async_redis_client,
        ):
            await index_long_term_memories(
                long_term_memories,
                redis_client=async_redis_client,
            )

        results = await search_long_term_memories(
            "What is the capital of France?",
            async_redis_client,
            session_id=SessionId(eq="123"),
            limit=1,
        )

        assert results.total == 1
        assert len(results.memories) == 1
        assert results.memories[0].text == "Paris is the capital of France"
        assert results.memories[0].session_id == "123"
        assert results.memories[0].memory_type == "message"

    @pytest.mark.asyncio
    async def test_search_messages_with_distance_threshold(self, async_redis_client):
        """Test searching messages with a distance threshold"""
        await ensure_search_index_exists(async_redis_client)

        long_term_memories = [
            MemoryRecord(
                id="memory-1", text="Paris is the capital of France", session_id="123"
            ),
            MemoryRecord(
                id="memory-2", text="France is a country in Europe", session_id="123"
            ),
        ]

        with mock.patch(
            "agent_memory_server.long_term_memory.get_redis_conn",
            return_value=async_redis_client,
        ):
            await index_long_term_memories(
                long_term_memories,
                redis_client=async_redis_client,
            )

        results = await search_long_term_memories(
            "What is the capital of France?",
            async_redis_client,
            session_id=SessionId(eq="123"),
            distance_threshold=0.1,
            limit=2,
        )

        assert results.total == 1
        assert len(results.memories) == 1
        assert results.memories[0].text == "Paris is the capital of France"
        assert results.memories[0].session_id == "123"
        assert results.memories[0].memory_type == "message"
