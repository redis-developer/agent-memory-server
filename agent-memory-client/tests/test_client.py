"""
Test file for the enhanced Memory API Client functionality.

Tests for new features like lifecycle management, batch operations,
pagination utilities, validation, and enhanced convenience methods.
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.models import (
    AckResponse,
    ClientMemoryRecord,
    MemoryMessage,
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
    WorkingMemoryResponse,
)


@pytest.fixture
async def enhanced_test_client() -> AsyncGenerator[MemoryAPIClient, None]:
    """Create a memory client for testing with mocked HTTP client."""
    config = MemoryClientConfig(
        base_url="http://test", default_namespace="test-namespace"
    )
    client = MemoryAPIClient(config)

    # Mock the HTTP client to avoid actual network calls
    client._client = AsyncMock(spec=httpx.AsyncClient)

    yield client

    await client.close()


class TestMemoryLifecycleManagement:
    """Tests for memory lifecycle management methods."""

    @pytest.mark.asyncio
    async def test_promote_working_memories_to_long_term(self, enhanced_test_client):
        """Test promoting specific working memories to long-term storage."""
        session_id = "test-session"

        # Create test memories
        memories = [
            ClientMemoryRecord(
                id="memory-1",
                text="User prefers dark mode",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
            ClientMemoryRecord(
                id="memory-2",
                text="User completed project setup",
                memory_type=MemoryTypeEnum.EPISODIC,
            ),
        ]

        # Mock working memory response
        working_memory_response = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=memories,
            data={},
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_session_memory") as mock_get,
            patch.object(
                enhanced_test_client, "create_long_term_memory"
            ) as mock_create,
        ):
            mock_get.return_value = working_memory_response
            mock_create.return_value = AckResponse(status="ok")

            # Test promoting all memories
            result = await enhanced_test_client.promote_working_memories_to_long_term(
                session_id=session_id
            )

            assert result.status == "ok"
            mock_get.assert_called_once_with(session_id=session_id, namespace=None)
            mock_create.assert_called_once_with(memories)

    @pytest.mark.asyncio
    async def test_promote_specific_memory_ids(self, enhanced_test_client):
        """Test promoting only specific memory IDs."""
        session_id = "test-session"

        memories = [
            ClientMemoryRecord(
                id="memory-1",
                text="User prefers dark mode",
                memory_type=MemoryTypeEnum.SEMANTIC,
            ),
            ClientMemoryRecord(
                id="memory-2",
                text="User completed project setup",
                memory_type=MemoryTypeEnum.EPISODIC,
            ),
        ]

        working_memory_response = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=memories,
            data={},
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_session_memory") as mock_get,
            patch.object(
                enhanced_test_client, "create_long_term_memory"
            ) as mock_create,
        ):
            mock_get.return_value = working_memory_response
            mock_create.return_value = AckResponse(status="ok")

            # Test promoting only specific memory
            result = await enhanced_test_client.promote_working_memories_to_long_term(
                session_id=session_id, memory_ids=["memory-1"]
            )

            assert result.status == "ok"
            # Should only promote memory-1
            mock_create.assert_called_once()
            promoted_memories = mock_create.call_args[0][0]
            assert len(promoted_memories) == 1
            assert promoted_memories[0].id == "memory-1"

    @pytest.mark.asyncio
    async def test_promote_no_memories(self, enhanced_test_client):
        """Test promoting when no memories exist."""
        session_id = "test-session"

        working_memory_response = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={},
            context=None,
            user_id=None,
        )

        with patch.object(enhanced_test_client, "get_session_memory") as mock_get:
            mock_get.return_value = working_memory_response

            result = await enhanced_test_client.promote_working_memories_to_long_term(
                session_id=session_id
            )

            assert result.status == "ok"


class TestBatchOperations:
    """Tests for batch operations."""

    @pytest.mark.asyncio
    async def test_bulk_create_long_term_memories(self, enhanced_test_client):
        """Test bulk creation of long-term memories with batching."""
        # Create test memory batches
        batch1 = [
            ClientMemoryRecord(
                text=f"Memory {i}",
                memory_type=MemoryTypeEnum.SEMANTIC,
            )
            for i in range(50)
        ]
        batch2 = [
            ClientMemoryRecord(
                text=f"Memory {i}",
                memory_type=MemoryTypeEnum.EPISODIC,
            )
            for i in range(30)
        ]

        memory_batches = [batch1, batch2]

        with patch.object(
            enhanced_test_client, "create_long_term_memory"
        ) as mock_create:
            mock_create.return_value = AckResponse(status="ok")

            # Test with default batch size
            results = await enhanced_test_client.bulk_create_long_term_memories(
                memory_batches=memory_batches,
                batch_size=25,
                delay_between_batches=0,  # No delay for test speed
            )

            # Should have created 4 batches: 25+25 for batch1, 25+5 for batch2
            assert len(results) == 4
            assert all(result.status == "ok" for result in results)
            assert mock_create.call_count == 4

    @pytest.mark.asyncio
    async def test_bulk_create_with_delay(self, enhanced_test_client):
        """Test bulk creation with rate limiting delay."""
        batch = [
            ClientMemoryRecord(
                text="Test memory",
                memory_type=MemoryTypeEnum.SEMANTIC,
            )
        ]

        with (
            patch.object(
                enhanced_test_client, "create_long_term_memory"
            ) as mock_create,
            patch("asyncio.sleep") as mock_sleep,
        ):
            mock_create.return_value = AckResponse(status="ok")

            asyncio.get_event_loop().time()
            await enhanced_test_client.bulk_create_long_term_memories(
                memory_batches=[batch],
                delay_between_batches=0.1,
            )

            # Should have called sleep (though mocked)
            mock_sleep.assert_called_with(0.1)


class TestPaginationUtilities:
    """Tests for pagination utilities."""

    @pytest.mark.asyncio
    async def test_search_all_long_term_memories(self, enhanced_test_client):
        """Test auto-paginating search for long-term memories."""
        # Mock responses for pagination
        first_response = MemoryRecordResults(
            total=150,
            memories=[
                MemoryRecordResult(
                    id=f"memory-{i}",
                    text=f"Memory text {i}",
                    dist=0.1,
                )
                for i in range(50)
            ],
            next_offset=50,
        )

        second_response = MemoryRecordResults(
            total=150,
            memories=[
                MemoryRecordResult(
                    id=f"memory-{i}",
                    text=f"Memory text {i}",
                    dist=0.1,
                )
                for i in range(50, 100)
            ],
            next_offset=100,
        )

        third_response = MemoryRecordResults(
            total=150,
            memories=[
                MemoryRecordResult(
                    id=f"memory-{i}",
                    text=f"Memory text {i}",
                    dist=0.1,
                )
                for i in range(100, 130)  # Less than batch_size, indicating end
            ],
            next_offset=None,
        )

        with patch.object(
            enhanced_test_client, "search_long_term_memory"
        ) as mock_search:
            mock_search.side_effect = [first_response, second_response, third_response]

            # Collect all results
            all_memories = []
            async for memory in enhanced_test_client.search_all_long_term_memories(
                text="test query", batch_size=50
            ):
                all_memories.append(memory)

            # Should have retrieved all 130 memories
            assert len(all_memories) == 130
            assert all_memories[0].id == "memory-0"
            assert all_memories[-1].id == "memory-129"

            # Should have made 3 API calls
            assert mock_search.call_count == 3

    @pytest.mark.asyncio
    async def test_search_all_memories(self, enhanced_test_client):
        """Test auto-paginating unified memory search."""
        # Similar test for unified search
        response = MemoryRecordResults(
            total=25,
            memories=[
                MemoryRecordResult(
                    id=f"memory-{i}",
                    text=f"Memory text {i}",
                    dist=0.1,
                )
                for i in range(25)
            ],
            next_offset=None,
        )

        with patch.object(enhanced_test_client, "search_memories") as mock_search:
            mock_search.return_value = response

            all_memories = []
            async for memory in enhanced_test_client.search_all_memories(
                text="test query", batch_size=50
            ):
                all_memories.append(memory)

            assert len(all_memories) == 25
            assert mock_search.call_count == 1


class TestClientSideValidation:
    """Tests for client-side validation methods."""

    def test_validate_memory_record_success(self, enhanced_test_client):
        """Test successful memory record validation."""
        memory = ClientMemoryRecord(
            text="Valid memory text",
            memory_type=MemoryTypeEnum.SEMANTIC,
            id="01HN0000000000000000000000",  # Valid ULID
        )

        # Should not raise
        enhanced_test_client.validate_memory_record(memory)

    def test_validate_memory_record_empty_text(self, enhanced_test_client):
        """Test validation failure for empty text."""
        memory = ClientMemoryRecord(
            text="",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )

        with pytest.raises(ValueError, match="Memory text cannot be empty"):
            enhanced_test_client.validate_memory_record(memory)

    def test_validate_memory_record_invalid_type(self, enhanced_test_client):
        """Test validation failure for invalid memory type."""
        # Test with a valid memory but manually set invalid type
        memory = ClientMemoryRecord(
            text="Valid text",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )
        # Manually override the memory type to test validation
        memory.memory_type = "invalid_type"  # type: ignore

        with pytest.raises(ValueError, match="Invalid memory type"):
            enhanced_test_client.validate_memory_record(memory)

    def test_validate_memory_record_invalid_id(self, enhanced_test_client):
        """Test validation failure for invalid ID format."""
        memory = ClientMemoryRecord(
            text="Valid text",
            memory_type=MemoryTypeEnum.SEMANTIC,
            id="invalid-id-format",
        )

        with pytest.raises(ValueError, match="Invalid ID format"):
            enhanced_test_client.validate_memory_record(memory)

    def test_validate_search_filters_success(self, enhanced_test_client):
        """Test successful search filter validation."""
        filters = {
            "limit": 10,
            "offset": 0,
            "distance_threshold": 0.5,
            "session_id": "test-session",
        }

        # Should not raise
        enhanced_test_client.validate_search_filters(**filters)

    def test_validate_search_filters_invalid_key(self, enhanced_test_client):
        """Test validation failure for invalid filter key."""
        filters = {"invalid_key": "value"}

        with pytest.raises(ValueError, match="Invalid filter key"):
            enhanced_test_client.validate_search_filters(**filters)

    def test_validate_search_filters_invalid_limit(self, enhanced_test_client):
        """Test validation failure for invalid limit."""
        filters = {"limit": -1}

        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            enhanced_test_client.validate_search_filters(**filters)

    def test_validate_search_filters_invalid_offset(self, enhanced_test_client):
        """Test validation failure for invalid offset."""
        filters = {"offset": -1}

        with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
            enhanced_test_client.validate_search_filters(**filters)

    def test_validate_search_filters_invalid_distance(self, enhanced_test_client):
        """Test validation failure for invalid distance threshold."""
        filters = {"distance_threshold": -0.5}

        with pytest.raises(
            ValueError, match="Distance threshold must be a non-negative number"
        ):
            enhanced_test_client.validate_search_filters(**filters)


class TestEnhancedConvenienceMethods:
    """Tests for enhanced convenience methods."""

    @pytest.mark.asyncio
    async def test_update_working_memory_data_merge(self, enhanced_test_client):
        """Test updating working memory data with merge strategy."""
        session_id = "test-session"

        existing_memory = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={"existing_key": "existing_value", "shared_key": "old_value"},
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_session_memory") as mock_get,
            patch.object(enhanced_test_client, "put_session_memory") as mock_put,
        ):
            mock_get.return_value = existing_memory
            mock_put.return_value = existing_memory

            updates = {"new_key": "new_value", "shared_key": "new_value"}

            await enhanced_test_client.update_working_memory_data(
                session_id=session_id,
                data_updates=updates,
                merge_strategy="merge",
            )

            # Check that put was called with merged data
            mock_put.assert_called_once()
            working_memory_arg = mock_put.call_args[0][1]
            expected_data = {
                "existing_key": "existing_value",
                "shared_key": "new_value",
                "new_key": "new_value",
            }
            assert working_memory_arg.data == expected_data

    @pytest.mark.asyncio
    async def test_update_working_memory_data_replace(self, enhanced_test_client):
        """Test updating working memory data with replace strategy."""
        session_id = "test-session"

        existing_memory = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={"existing_key": "existing_value"},
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_session_memory") as mock_get,
            patch.object(enhanced_test_client, "put_session_memory") as mock_put,
        ):
            mock_get.return_value = existing_memory
            mock_put.return_value = existing_memory

            updates = {"new_key": "new_value"}

            await enhanced_test_client.update_working_memory_data(
                session_id=session_id,
                data_updates=updates,
                merge_strategy="replace",
            )

            # Check that put was called with replaced data
            working_memory_arg = mock_put.call_args[0][1]
            assert working_memory_arg.data == updates

    @pytest.mark.asyncio
    async def test_update_working_memory_data_deep_merge(self, enhanced_test_client):
        """Test updating working memory data with deep merge strategy."""
        session_id = "test-session"

        existing_memory = WorkingMemoryResponse(
            session_id=session_id,
            messages=[],
            memories=[],
            data={
                "nested": {"existing": "value", "shared": "old"},
                "top_level": "existing",
            },
            context=None,
            user_id=None,
        )

        with (
            patch.object(enhanced_test_client, "get_session_memory") as mock_get,
            patch.object(enhanced_test_client, "put_session_memory") as mock_put,
        ):
            mock_get.return_value = existing_memory
            mock_put.return_value = existing_memory

            updates = {
                "nested": {"new": "value", "shared": "new"},
                "new_top": "new",
            }

            await enhanced_test_client.update_working_memory_data(
                session_id=session_id,
                data_updates=updates,
                merge_strategy="deep_merge",
            )

            # Check deep merge result
            working_memory_arg = mock_put.call_args[0][1]
            expected_data = {
                "nested": {"existing": "value", "shared": "new", "new": "value"},
                "top_level": "existing",
                "new_top": "new",
            }
            assert working_memory_arg.data == expected_data

    @pytest.mark.asyncio
    async def test_append_messages_to_working_memory(self, enhanced_test_client):
        """Test appending messages to existing working memory."""
        session_id = "test-session"

        existing_messages = [
            MemoryMessage(role="user", content="First message"),
        ]

        existing_memory = WorkingMemoryResponse(
            session_id=session_id,
            messages=existing_messages,
            memories=[],
            data={},
            context=None,
            user_id=None,
        )

        new_messages = [
            MemoryMessage(role="assistant", content="Second message"),
            MemoryMessage(role="user", content="Third message"),
        ]

        with (
            patch.object(enhanced_test_client, "get_session_memory") as mock_get,
            patch.object(enhanced_test_client, "put_session_memory") as mock_put,
        ):
            mock_get.return_value = existing_memory
            mock_put.return_value = existing_memory

            await enhanced_test_client.append_messages_to_working_memory(
                session_id=session_id,
                messages=new_messages,
            )

            # Check that messages were appended
            working_memory_arg = mock_put.call_args[0][1]
            assert len(working_memory_arg.messages) == 3
            assert working_memory_arg.messages[0].content == "First message"
            assert working_memory_arg.messages[1].content == "Second message"
            assert working_memory_arg.messages[2].content == "Third message"

    def test_deep_merge_dicts(self, enhanced_test_client):
        """Test the deep merge dictionary utility method."""
        base = {
            "a": {"nested": {"deep": "value1", "shared": "old"}},
            "b": "simple",
        }

        updates = {
            "a": {"nested": {"shared": "new", "additional": "value2"}},
            "c": "new_simple",
        }

        result = enhanced_test_client._deep_merge_dicts(base, updates)

        expected = {
            "a": {
                "nested": {
                    "deep": "value1",
                    "shared": "new",
                    "additional": "value2",
                }
            },
            "b": "simple",
            "c": "new_simple",
        }

        assert result == expected

    def test_is_valid_ulid(self, enhanced_test_client):
        """Test ULID validation utility method."""
        # Valid ULID
        assert enhanced_test_client._is_valid_ulid("01HN0000000000000000000000")

        # Invalid ULID
        assert not enhanced_test_client._is_valid_ulid("invalid-id")
        assert not enhanced_test_client._is_valid_ulid("")
        assert not enhanced_test_client._is_valid_ulid("too-short")


class TestErrorHandling:
    """Tests for error handling in new methods."""

    @pytest.mark.asyncio
    async def test_bulk_create_handles_failures(self, enhanced_test_client):
        """Test that bulk create handles individual batch failures."""
        batch = [
            ClientMemoryRecord(
                text="Test memory",
                memory_type=MemoryTypeEnum.SEMANTIC,
            )
        ]

        with patch.object(
            enhanced_test_client, "create_long_term_memory"
        ) as mock_create:
            # First call succeeds, second fails, third succeeds
            mock_create.side_effect = [
                AckResponse(status="ok"),
                Exception("API Error"),
                AckResponse(status="ok"),
            ]

            # Should raise the exception from the second batch
            with pytest.raises(Exception, match="API Error"):
                await enhanced_test_client.bulk_create_long_term_memories(
                    memory_batches=[batch, batch, batch],
                    delay_between_batches=0,
                )

    @pytest.mark.asyncio
    async def test_pagination_handles_empty_results(self, enhanced_test_client):
        """Test pagination utilities handle empty result sets."""
        empty_response = MemoryRecordResults(
            total=0,
            memories=[],
            next_offset=None,
        )

        with patch.object(
            enhanced_test_client, "search_long_term_memory"
        ) as mock_search:
            mock_search.return_value = empty_response

            # Should handle empty results gracefully
            all_memories = []
            async for memory in enhanced_test_client.search_all_long_term_memories(
                text="test query"
            ):
                all_memories.append(memory)

            assert len(all_memories) == 0
            assert mock_search.call_count == 1

    def test_validation_with_none_values(self, enhanced_test_client):
        """Test validation handles None values appropriately."""
        memory = ClientMemoryRecord(
            text="Valid text",
            memory_type=MemoryTypeEnum.SEMANTIC,
        )
        # ClientMemoryRecord generates a ULID ID by default, so this should pass

        # Should not raise
        enhanced_test_client.validate_memory_record(memory)
