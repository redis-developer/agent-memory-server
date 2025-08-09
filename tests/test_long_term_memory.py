from datetime import UTC, datetime
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from agent_memory_server.filters import SessionId
from agent_memory_server.long_term_memory import (
    compact_long_term_memories,
    count_long_term_memories,
    deduplicate_by_hash,
    deduplicate_by_id,
    delete_long_term_memories,
    extract_memory_structure,
    generate_memory_hash,
    index_long_term_memories,
    merge_memories_with_llm,
    promote_working_memory_to_long_term,
    search_long_term_memories,
)
from agent_memory_server.models import (
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
)


# from agent_memory_server.utils.redis import ensure_search_index_exists  # Not used currently


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_index_memories(
        self, mock_openai_client, mock_async_redis_client, session
    ):
        """Test indexing memories using vectorstore adapter"""
        long_term_memories = [
            MemoryRecord(
                id="memory-1", text="Paris is the capital of France", session_id=session
            ),
            MemoryRecord(
                id="memory-2", text="France is a country in Europe", session_id=session
            ),
        ]

        # Mock the vectorstore adapter add_memories method
        mock_adapter = AsyncMock()
        mock_adapter.add_memories.return_value = ["memory-1", "memory-2"]

        with mock.patch(
            "agent_memory_server.long_term_memory.get_vectorstore_adapter",
            return_value=mock_adapter,
        ):
            await index_long_term_memories(
                long_term_memories,
                redis_client=mock_async_redis_client,
            )

        # Check that the adapter add_memories was called with the right arguments
        mock_adapter.add_memories.assert_called_once()
        call_args = mock_adapter.add_memories.call_args

        # Verify the memories passed to the adapter
        memories_arg = call_args[0][0]  # First positional argument
        assert len(memories_arg) == 2
        assert memories_arg[0].id == "memory-1"
        assert memories_arg[0].text == "Paris is the capital of France"
        assert memories_arg[1].id == "memory-2"
        assert memories_arg[1].text == "France is a country in Europe"

    @pytest.mark.asyncio
    async def test_search_memories(self, mock_openai_client, mock_async_redis_client):
        """Test searching memories using vectorstore adapter"""
        from agent_memory_server.models import MemoryRecordResult, MemoryRecordResults

        # Mock the vectorstore adapter search_memories method
        mock_adapter = AsyncMock()

        # Create mock search results in the expected format
        mock_memory_result = MemoryRecordResult(
            id="test-id",
            text="Hello, world!",
            dist=0.25,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
            user_id="test-user",
            session_id="test-session",
            namespace="test-namespace",
            topics=["greeting"],
            entities=["world"],
            memory_hash="test-hash",
            memory_type=MemoryTypeEnum.MESSAGE,
        )

        mock_search_results = MemoryRecordResults(
            memories=[mock_memory_result],
            total=1,
            next_offset=None,
        )

        mock_adapter.search_memories.return_value = mock_search_results

        query = "What is the meaning of life?"
        session_id = SessionId(eq="test-session")

        with mock.patch(
            "agent_memory_server.long_term_memory.get_vectorstore_adapter",
            return_value=mock_adapter,
        ):
            results = await search_long_term_memories(
                query,
                session_id=session_id,
                optimize_query=False,  # Disable query optimization for this unit test
            )

        # Check that the adapter search_memories was called with the right arguments
        mock_adapter.search_memories.assert_called_once()
        call_args = mock_adapter.search_memories.call_args
        assert call_args[1]["query"] == query  # Check query parameter
        assert call_args[1]["session_id"] == session_id  # Check session_id filter

        assert len(results.memories) == 1
        assert isinstance(results.memories[0], MemoryRecordResult)
        assert results.memories[0].text == "Hello, world!"
        assert results.memories[0].dist == 0.25
        assert results.memories[0].memory_type == "message"

    @pytest.mark.asyncio
    async def test_deduplicate_by_id(self, mock_async_redis_client):
        """Test deduplication by id using vectorstore adapter"""
        memory = MemoryRecord(
            text="Test memory",
            id="test-id",
            session_id="test-session",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        with patch(
            "agent_memory_server.long_term_memory.get_vectorstore_adapter"
        ) as mock_get_adapter:
            mock_adapter = AsyncMock()
            mock_get_adapter.return_value = mock_adapter

            # Test case 1: Memory doesn't exist
            mock_search_result = Mock()
            mock_search_result.memories = []  # No existing memories
            mock_adapter.search_memories.return_value = mock_search_result

            result_memory, overwrite = await deduplicate_by_id(
                memory, redis_client=mock_async_redis_client
            )

            assert result_memory == memory
            assert overwrite is False

            # Verify search was called with correct filters
            mock_adapter.search_memories.assert_called_once()
            call_kwargs = mock_adapter.search_memories.call_args[1]
            assert call_kwargs["query"] == ""
            assert call_kwargs["limit"] == 1

            # Test case 2: Memory exists
            existing_memory = MemoryRecordResult(
                id="test-id",
                text="Existing memory",
                session_id="test-session",
                dist=0.0,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                last_accessed=datetime.now(UTC),
                persisted_at=datetime.now(UTC),
                memory_type="semantic",
                memory_hash="",
                user_id=None,
                namespace=None,
                topics=[],
                entities=[],
            )

            mock_search_result.memories = [existing_memory]
            mock_adapter.search_memories.return_value = mock_search_result
            mock_adapter.delete_memories = AsyncMock()

            result_memory, overwrite = await deduplicate_by_id(
                memory, redis_client=mock_async_redis_client
            )

            assert result_memory == memory
            assert overwrite is True
            mock_adapter.delete_memories.assert_called_once_with(["test-id"])

    def test_generate_memory_hash(self):
        """Test memory hash generation"""
        memory1 = MemoryRecord(
            id="test-id-1",
            text="Hello world",
            user_id="user123",
            session_id="session456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        memory2 = MemoryRecord(
            id="test-id-2",
            text="Hello world",
            user_id="user123",
            session_id="session456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        memory3 = MemoryRecord(
            id="test-id-3",
            text="Different text",
            user_id="user123",
            session_id="session456",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # MemoryRecord objects with same content produce same hash (content-based hashing)
        # IDs and timestamps don't affect the hash
        hash1 = generate_memory_hash(memory1)
        hash2 = generate_memory_hash(memory2)
        hash3 = generate_memory_hash(memory3)

        # Same content should produce same hash
        assert hash1 == hash2  # Same content, different IDs
        assert hash1 != hash3  # Different text
        assert hash2 != hash3  # Different text

        # Test with missing user_id field
        memory4 = MemoryRecord(
            id="test-id-4",
            text="Hello world",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )
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

            # Create a proper MemoryRecord
            memory = MemoryRecord(
                id="test-id",
                text="Test text content",
                namespace="test-namespace",
                memory_type=MemoryTypeEnum.SEMANTIC,
            )

            await extract_memory_structure(memory)

            # Verify extraction was called
            mock_extract.assert_called_once_with("Test text content")

            # Verify Redis was updated with topics and entities
            mock_redis.hset.assert_called_once()
            args, kwargs = mock_redis.hset.call_args

            # Check the key format - it includes the memory ID in the key structure
            assert "memory_idx:" in args[0] and "test-id" in args[0]

            # Check the mapping
            mapping = kwargs["mapping"]
            assert mapping["topics"] == "topic1,topic2"
            assert mapping["entities"] == "entity1,entity2"

    @pytest.mark.asyncio
    async def test_count_long_term_memories(self, mock_async_redis_client):
        """Test counting long-term memories using vectorstore adapter"""

        # Mock the vectorstore adapter count_memories method
        mock_adapter = AsyncMock()
        mock_adapter.count_memories.return_value = 42

        with mock.patch(
            "agent_memory_server.long_term_memory.get_vectorstore_adapter",
            return_value=mock_adapter,
        ):
            count = await count_long_term_memories(
                namespace="test-namespace",
                user_id="test-user",
                session_id="test-session",
                redis_client=mock_async_redis_client,
            )

        assert count == 42

        # Verify the adapter count_memories was called with the right arguments
        mock_adapter.count_memories.assert_called_once_with(
            namespace="test-namespace",
            user_id="test-user",
            session_id="test-session",
        )

    @pytest.mark.asyncio
    async def test_deduplicate_by_hash(self, mock_async_redis_client):
        """Test deduplication by hash using vectorstore adapter"""
        memory = MemoryRecord(
            id="test-memory-1",
            text="Test memory",
            session_id="test-session",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # Test case 1: No duplicate found
        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = MemoryRecordResults(
            total=0, memories=[]
        )

        with mock.patch(
            "agent_memory_server.long_term_memory.get_vectorstore_adapter",
            return_value=mock_adapter,
        ):
            result_memory, overwrite = await deduplicate_by_hash(
                memory, redis_client=mock_async_redis_client
            )

        assert result_memory == memory
        assert overwrite is False

        # Test case 2: Duplicate found
        existing_memory = MemoryRecordResult(
            id="existing-memory-id",
            text="Test memory",
            dist=0.0,
            memory_type=MemoryTypeEnum.SEMANTIC,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
        )

        mock_adapter.search_memories.return_value = MemoryRecordResults(
            total=1, memories=[existing_memory]
        )

        # Mock the hset call that updates last_accessed
        mock_async_redis_client.hset = AsyncMock()

        with mock.patch(
            "agent_memory_server.long_term_memory.get_vectorstore_adapter",
            return_value=mock_adapter,
        ):
            result_memory, overwrite = await deduplicate_by_hash(
                memory, redis_client=mock_async_redis_client
            )

        # Should return None (duplicate found) and overwrite=True
        assert result_memory is None
        assert overwrite is True

        # Verify that last_accessed was updated
        mock_async_redis_client.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_memories_with_llm(self):
        """Test merging memories with LLM"""
        from datetime import UTC, datetime

        memories = [
            MemoryRecord(
                id="test-id-1",
                text="User likes coffee",
                topics=["coffee", "preferences"],
                entities=["user"],
                created_at=datetime.fromtimestamp(1000, UTC),
                last_accessed=datetime.fromtimestamp(1500, UTC),
                namespace="test",
                user_id="user123",
                session_id="session456",
                memory_type=MemoryTypeEnum.SEMANTIC,
                discrete_memory_extracted="t",
            ),
            MemoryRecord(
                id="test-id-2",
                text="User enjoys drinking coffee in the morning",
                topics=["coffee", "morning"],
                entities=["user"],
                created_at=datetime.fromtimestamp(1200, UTC),
                last_accessed=datetime.fromtimestamp(1600, UTC),
                namespace="test",
                user_id="user123",
                session_id="session456",
                memory_type=MemoryTypeEnum.SEMANTIC,
                discrete_memory_extracted="t",
            ),
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
        assert "coffee" in merged.text.lower()
        assert merged.created_at == datetime.fromtimestamp(
            1000, UTC
        )  # Earliest timestamp
        assert merged.last_accessed == datetime.fromtimestamp(
            1600, UTC
        )  # Latest timestamp
        assert set(merged.topics) == {"coffee", "preferences", "morning"}
        assert set(merged.entities) == {"user"}
        assert merged.user_id == "user123"
        assert merged.session_id == "session456"
        assert merged.namespace == "test"
        assert merged.memory_hash is not None

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
            patch(
                "agent_memory_server.long_term_memory.count_long_term_memories"
            ) as mock_count,
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
            mock_count.return_value = 2  # Return expected count

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
                user_id=None,
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

    @pytest.mark.asyncio
    async def test_delete_long_term_memories(self, mock_async_redis_client):
        """Test deleting long-term memories by ID"""

        # Test IDs to delete
        memory_ids = ["memory-1", "memory-2", "memory-3"]

        # Mock the vectorstore adapter delete_memories method
        mock_adapter = AsyncMock()
        mock_adapter.delete_memories.return_value = 3  # 3 memories deleted

        with mock.patch(
            "agent_memory_server.long_term_memory.get_vectorstore_adapter",
            return_value=mock_adapter,
        ):
            deleted_count = await delete_long_term_memories(
                ids=memory_ids,
            )

        # Verify the adapter was called with the correct IDs
        mock_adapter.delete_memories.assert_called_once_with(memory_ids)

        # Verify the correct count was returned
        assert deleted_count == 3

    @pytest.mark.asyncio
    async def test_delete_long_term_memories_empty_list(self, mock_async_redis_client):
        """Test deleting long-term memories with empty ID list"""

        # Mock the vectorstore adapter delete_memories method
        mock_adapter = AsyncMock()
        mock_adapter.delete_memories.return_value = 0  # No memories deleted

        with mock.patch(
            "agent_memory_server.long_term_memory.get_vectorstore_adapter",
            return_value=mock_adapter,
        ):
            deleted_count = await delete_long_term_memories(
                ids=[],
            )

        # Verify the adapter was called with empty list
        mock_adapter.delete_memories.assert_called_once_with([])

        # Verify zero count was returned
        assert deleted_count == 0


@pytest.mark.requires_api_keys
class TestLongTermMemoryIntegration:
    """Integration tests for long-term memory"""

    @pytest.mark.asyncio
    async def test_search_messages(self, async_redis_client):
        """Test searching messages"""
        # await ensure_search_index_exists(async_redis_client)  # Let LangChain handle index

        long_term_memories = [
            MemoryRecord(
                id="memory-1", text="Paris is the capital of France", session_id="123"
            ),
            MemoryRecord(
                id="memory-2", text="France is a country in Europe", session_id="123"
            ),
        ]

        # Index memories using the test Redis connection (already patched by conftest)
        await index_long_term_memories(
            long_term_memories,
            redis_client=async_redis_client,
        )

        # Search using the same connection (should be patched by conftest)
        results = await search_long_term_memories(
            "What is the capital of France?",
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
        # await ensure_search_index_exists(async_redis_client)  # Let LangChain handle index

        long_term_memories = [
            MemoryRecord(
                id="memory-1", text="Paris is the capital of France", session_id="123"
            ),
            MemoryRecord(
                id="memory-2", text="France is a country in Europe", session_id="123"
            ),
        ]

        # Index memories using the test Redis connection (already patched by conftest)
        await index_long_term_memories(
            long_term_memories,
            redis_client=async_redis_client,
        )

        # Search using the same connection (should be patched by conftest)
        results = await search_long_term_memories(
            "What is the capital of France?",
            session_id=SessionId(eq="123"),
            distance_threshold=0.3,
            limit=2,
        )

        # At least one memory should pass the threshold, and the most relevant one should be first
        assert results.total >= 1
        assert len(results.memories) >= 1

        # Verify that the first result is the more directly relevant one
        assert results.memories[0].text == "Paris is the capital of France"
        assert results.memories[0].session_id == "123"
        assert results.memories[0].memory_type == "message"

        # Test with a very strict threshold that should filter out results
        strict_results = await search_long_term_memories(
            "What is the capital of France?",
            session_id=SessionId(eq="123"),
            distance_threshold=0.05,  # Very strict threshold
            limit=2,
        )

        # With strict threshold, we should get fewer or equal results
        assert strict_results.total <= results.total

    @pytest.mark.asyncio
    async def test_deduplicate_by_id_with_user_id_real_redis_error(
        self, async_redis_client
    ):
        """Test to reproduce the actual Redis error with user_id using real Redis connection"""

        # First, create the index by indexing some memories
        initial_memories = [
            MemoryRecord(
                id="setup-memory-1",
                text="Setup memory to create index",
                session_id="setup-session",
            ),
        ]

        # Index memories to create the Redis search index
        await index_long_term_memories(
            initial_memories,
            redis_client=async_redis_client,
        )

        # Now create a memory with user_id that causes the staging error
        memory = MemoryRecord(
            text="Test memory with user ID",
            id="test-memory-with-user-id",
            session_id="test-session",
            user_id="U08TTULBA1F",  # This causes the error in staging
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        # This should reproduce the actual Redis error from staging
        try:
            result_memory, overwrite = await deduplicate_by_id(
                memory, redis_client=async_redis_client
            )

            # If we get here without error, the test environment has proper schema
            print("SUCCESS: No error occurred - Redis index supports user_id field")

        except Exception as e:
            print(f"ERROR REPRODUCED: {type(e).__name__}: {e}")

            # Check if this is the specific error we're trying to reproduce
            if "Unknown argument" in str(e) and "@user_id:" in str(e):
                print("✅ Successfully reproduced the staging error!")
                print("The Redis search index doesn't have user_id field indexed")
            else:
                print("❌ Different error occurred")

            # Re-raise to see the full traceback
            raise


@pytest.mark.asyncio
class TestSearchQueryOptimization:
    """Test query optimization in search_long_term_memories function."""

    @patch("agent_memory_server.long_term_memory.get_vectorstore_adapter")
    @patch("agent_memory_server.long_term_memory.optimize_query_for_vector_search")
    async def test_search_with_query_optimization_enabled(
        self, mock_optimize, mock_get_adapter
    ):
        """Test that query optimization is applied when optimize_query=True."""
        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = MemoryRecordResults(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="test-id",
                    text="Test memory",
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    dist=0.1,
                )
            ],
        )
        mock_get_adapter.return_value = mock_adapter

        # Mock query optimization
        mock_optimize.return_value = "optimized search query"

        # Call search with optimization enabled
        result = await search_long_term_memories(
            text="tell me about my preferences", optimize_query=True, limit=10
        )

        # Verify optimization was called
        mock_optimize.assert_called_once_with("tell me about my preferences")

        # Verify adapter was called with optimized query
        mock_adapter.search_memories.assert_called_once()
        call_kwargs = mock_adapter.search_memories.call_args[1]
        assert call_kwargs["query"] == "optimized search query"

        # Verify results
        assert result.total == 1
        assert len(result.memories) == 1

    @patch("agent_memory_server.long_term_memory.get_vectorstore_adapter")
    @patch("agent_memory_server.long_term_memory.optimize_query_for_vector_search")
    async def test_search_with_query_optimization_disabled(
        self, mock_optimize, mock_get_adapter
    ):
        """Test that query optimization is skipped when optimize_query=False."""
        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = MemoryRecordResults(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="test-id",
                    text="Test memory",
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    dist=0.1,
                )
            ],
        )
        mock_get_adapter.return_value = mock_adapter

        # Call search with optimization disabled
        result = await search_long_term_memories(
            text="tell me about my preferences", optimize_query=False, limit=10
        )

        # Verify optimization was NOT called
        mock_optimize.assert_not_called()

        # Verify adapter was called with original query
        mock_adapter.search_memories.assert_called_once()
        call_kwargs = mock_adapter.search_memories.call_args[1]
        assert call_kwargs["query"] == "tell me about my preferences"

        # Verify results
        assert result.total == 1
        assert len(result.memories) == 1

    @patch("agent_memory_server.long_term_memory.get_vectorstore_adapter")
    @patch("agent_memory_server.long_term_memory.optimize_query_for_vector_search")
    async def test_search_with_empty_query_skips_optimization(
        self, mock_optimize, mock_get_adapter
    ):
        """Test that empty queries skip optimization."""
        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = MemoryRecordResults(
            total=0, memories=[]
        )
        mock_get_adapter.return_value = mock_adapter

        # Call search with empty query
        await search_long_term_memories(text="", optimize_query=True, limit=10)

        # Verify optimization was NOT called for empty query
        mock_optimize.assert_not_called()

        # Verify adapter was called with empty query
        mock_adapter.search_memories.assert_called_once()
        call_kwargs = mock_adapter.search_memories.call_args[1]
        assert call_kwargs["query"] == ""

    @patch("agent_memory_server.long_term_memory.get_vectorstore_adapter")
    @patch("agent_memory_server.long_term_memory.optimize_query_for_vector_search")
    async def test_search_optimization_failure_fallback(
        self, mock_optimize, mock_get_adapter
    ):
        """Test that search continues with original query if optimization fails."""
        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = MemoryRecordResults(
            total=0, memories=[]
        )
        mock_get_adapter.return_value = mock_adapter

        # Mock optimization to return original query (simulating internal error handling)
        mock_optimize.return_value = (
            "test query"  # Returns original query after internal error handling
        )

        # Call search - this should not raise an exception
        await search_long_term_memories(
            text="test query", optimize_query=True, limit=10
        )

        # Verify optimization was attempted
        mock_optimize.assert_called_once_with("test query")

        # Verify search proceeded with the query (original after fallback)
        mock_adapter.search_memories.assert_called_once()
        call_kwargs = mock_adapter.search_memories.call_args[1]
        assert call_kwargs["query"] == "test query"

    @patch("agent_memory_server.long_term_memory.get_vectorstore_adapter")
    @patch("agent_memory_server.long_term_memory.optimize_query_for_vector_search")
    async def test_search_passes_all_parameters_correctly(
        self, mock_optimize, mock_get_adapter
    ):
        """Test that all search parameters are passed correctly to the adapter."""
        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = MemoryRecordResults(
            total=0, memories=[]
        )
        mock_get_adapter.return_value = mock_adapter

        # Mock query optimization
        mock_optimize.return_value = "optimized query"

        # Create filter objects for testing
        session_filter = SessionId(eq="test-session")

        # Call search with various parameters
        await search_long_term_memories(
            text="test query",
            session_id=session_filter,
            limit=20,
            offset=10,
            distance_threshold=0.3,
            optimize_query=True,
        )

        # Verify optimization was called
        mock_optimize.assert_called_once_with("test query")

        # Verify all parameters were passed to adapter
        mock_adapter.search_memories.assert_called_once()
        call_kwargs = mock_adapter.search_memories.call_args[1]
        assert call_kwargs["query"] == "optimized query"
        assert call_kwargs["session_id"] == session_filter
        assert call_kwargs["limit"] == 20
        assert call_kwargs["offset"] == 10
        assert call_kwargs["distance_threshold"] == 0.3
