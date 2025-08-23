"""Tests for thread-aware contextual grounding functionality."""

from datetime import UTC, datetime

import pytest
import ulid

from agent_memory_server.long_term_memory import (
    extract_memories_from_session_thread,
    should_extract_session_thread,
)
from agent_memory_server.models import MemoryMessage, WorkingMemory
from agent_memory_server.working_memory import set_working_memory


@pytest.mark.asyncio
class TestThreadAwareContextualGrounding:
    """Test thread-aware contextual grounding with full conversation context."""

    async def create_test_conversation(self, session_id: str) -> WorkingMemory:
        """Create a test conversation with cross-message pronoun references."""
        messages = [
            MemoryMessage(
                id=str(ulid.ULID()),
                role="user",
                content="John is our new backend developer.",
                timestamp=datetime.now(UTC).isoformat(),
                discrete_memory_extracted="f",
            ),
            MemoryMessage(
                id=str(ulid.ULID()),
                role="assistant",
                content="That's great! What technologies does he work with?",
                timestamp=datetime.now(UTC).isoformat(),
                discrete_memory_extracted="f",
            ),
            MemoryMessage(
                id=str(ulid.ULID()),
                role="user",
                content="He specializes in Python and PostgreSQL. His experience with microservices is excellent.",
                timestamp=datetime.now(UTC).isoformat(),
                discrete_memory_extracted="f",
            ),
        ]

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id="test-user",
            namespace="test-namespace",
            messages=messages,
            memories=[],
        )

        # Store in working memory
        await set_working_memory(working_memory)
        return working_memory

    @pytest.mark.requires_api_keys
    async def test_thread_aware_pronoun_resolution(self):
        """Test that thread-aware extraction properly resolves pronouns across messages."""

        session_id = f"test-thread-{ulid.ULID()}"

        # Create conversation with cross-message pronoun references
        await self.create_test_conversation(session_id)

        # Extract memories using thread-aware approach
        extracted_memories = await extract_memories_from_session_thread(
            session_id=session_id,
            namespace="test-namespace",
            user_id="test-user",
        )

        # Should have extracted some memories
        assert len(extracted_memories) > 0

        # Combine all extracted memory text
        all_memory_text = " ".join([mem.text for mem in extracted_memories])

        print(f"\nExtracted memories: {len(extracted_memories)}")
        for i, mem in enumerate(extracted_memories):
            print(f"{i + 1}. [{mem.memory_type}] {mem.text}")

        print(f"\nCombined memory text: {all_memory_text}")

        # Check that pronouns were properly grounded
        # The memories should mention "John" instead of leaving "he/his" unresolved
        assert (
            "john" in all_memory_text.lower()
        ), "Memories should contain the grounded name 'John'"

        # Ideally, there should be minimal or no ungrounded pronouns
        ungrounded_pronouns = [
            "he ",
            "his ",
            "him ",
        ]  # Note: spaces to avoid false positives
        ungrounded_count = sum(
            all_memory_text.lower().count(pronoun) for pronoun in ungrounded_pronouns
        )

        print(f"Ungrounded pronouns found: {ungrounded_count}")

        # This is a softer assertion since full grounding is still being improved
        # But we should see significant improvement over per-message extraction
        assert (
            ungrounded_count <= 2
        ), f"Should have minimal ungrounded pronouns, found {ungrounded_count}"

    async def test_debounce_mechanism(self, redis_url):
        """Test that the debounce mechanism prevents frequent re-extraction."""
        from redis.asyncio import Redis

        # Use testcontainer Redis instead of localhost:6379
        redis = Redis.from_url(redis_url)
        session_id = f"test-debounce-{ulid.ULID()}"
        print(f"Testing debounce with Redis URL: {redis_url}")

        # First call should allow extraction
        should_extract_1 = await should_extract_session_thread(session_id, redis)
        assert should_extract_1 is True, "First extraction attempt should be allowed"

        # Immediate second call should be debounced
        should_extract_2 = await should_extract_session_thread(session_id, redis)
        assert (
            should_extract_2 is False
        ), "Second extraction attempt should be debounced"

        # Clean up
        debounce_key = f"extraction_debounce:{session_id}"
        await redis.delete(debounce_key)

    @pytest.mark.requires_api_keys
    async def test_empty_conversation_handling(self):
        """Test that empty or non-existent conversations are handled gracefully."""

        session_id = f"test-empty-{ulid.ULID()}"

        # Try to extract from non-existent session
        extracted_memories = await extract_memories_from_session_thread(
            session_id=session_id,
            namespace="test-namespace",
            user_id="test-user",
        )

        # Should return empty list without errors
        assert extracted_memories == []

    @pytest.mark.requires_api_keys
    async def test_multi_entity_conversation(self):
        """Test contextual grounding with multiple entities in conversation."""

        session_id = f"test-multi-entity-{ulid.ULID()}"

        # Create conversation with multiple people
        messages = [
            MemoryMessage(
                id=str(ulid.ULID()),
                role="user",
                content="John and Sarah are working on the API redesign project.",
                timestamp=datetime.now(UTC).isoformat(),
                discrete_memory_extracted="f",
            ),
            MemoryMessage(
                id=str(ulid.ULID()),
                role="user",
                content="He's handling the backend while she focuses on the frontend integration.",
                timestamp=datetime.now(UTC).isoformat(),
                discrete_memory_extracted="f",
            ),
            MemoryMessage(
                id=str(ulid.ULID()),
                role="user",
                content="Their collaboration has been very effective. His Python skills complement her React expertise.",
                timestamp=datetime.now(UTC).isoformat(),
                discrete_memory_extracted="f",
            ),
        ]

        working_memory = WorkingMemory(
            session_id=session_id,
            user_id="test-user",
            namespace="test-namespace",
            messages=messages,
            memories=[],
        )

        await set_working_memory(working_memory)

        # Extract memories
        extracted_memories = await extract_memories_from_session_thread(
            session_id=session_id,
            namespace="test-namespace",
            user_id="test-user",
        )

        assert len(extracted_memories) > 0

        all_memory_text = " ".join([mem.text for mem in extracted_memories])

        print(f"\nMulti-entity extracted memories: {len(extracted_memories)}")
        for i, mem in enumerate(extracted_memories):
            print(f"{i + 1}. [{mem.memory_type}] {mem.text}")

        # Should mention both John and Sarah by name
        assert "john" in all_memory_text.lower(), "Should mention John by name"
        assert "sarah" in all_memory_text.lower(), "Should mention Sarah by name"

        # Check for reduced pronoun usage
        pronouns = ["he ", "she ", "his ", "her ", "him "]
        pronoun_count = sum(all_memory_text.lower().count(p) for p in pronouns)
        print(f"Remaining pronouns: {pronoun_count}")

        # Allow some remaining pronouns since this is a complex multi-entity case
        # This is still a significant improvement over per-message extraction
        assert (
            pronoun_count <= 5
        ), f"Should have reduced pronoun usage, found {pronoun_count}"
