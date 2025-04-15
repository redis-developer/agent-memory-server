import json
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from agent_memory_server.long_term_memory import (
    index_long_term_memories,
)
from agent_memory_server.messages import (
    delete_session_memory,
    get_session_memory,
    list_sessions,
    set_session_memory,
)
from agent_memory_server.models import LongTermMemory, MemoryMessage, SessionMemory
from agent_memory_server.summarization import summarize_session


@pytest.mark.asyncio
class TestListSessions:
    async def test_list_sessions_empty(self, mock_async_redis_client):
        """Test listing sessions when none exist"""
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.zcard = MagicMock(return_value=mock_pipeline)
        mock_pipeline.zrange = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(return_value=(0, []))
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        total, sessions = await list_sessions(mock_async_redis_client)

        assert total == 0
        assert sessions == []
        mock_pipeline.zcard.assert_called_once()
        mock_pipeline.zrange.assert_called_once()

    async def test_list_sessions_with_sessions(self, mock_async_redis_client):
        """Test listing sessions when some exist"""
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.zcard = MagicMock(return_value=mock_pipeline)
        mock_pipeline.zrange = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(return_value=(2, [b"session1", b"session2"]))
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        total, sessions = await list_sessions(mock_async_redis_client)

        assert total == 2
        assert sessions == ["session1", "session2"]

    async def test_list_sessions_with_namespace(self, mock_async_redis_client):
        """Test listing sessions with a namespace"""
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.zcard = MagicMock(return_value=mock_pipeline)
        mock_pipeline.zrange = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(return_value=(1, [b"session1"]))
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        total, sessions = await list_sessions(mock_async_redis_client, namespace="test")

        assert total == 1
        assert sessions == ["session1"]
        mock_pipeline.zcard.assert_called_with("sessions:test")


@pytest.mark.asyncio
class TestGetSessionMemory:
    async def test_get_nonexistent_session(self, mock_async_redis_client):
        """Test getting a session that doesn't exist"""
        mock_async_redis_client.zscore = AsyncMock(return_value=None)

        result = await get_session_memory(mock_async_redis_client, "nonexistent")

        assert result is None
        mock_async_redis_client.zscore.assert_called_once()

    async def test_get_existing_session(self, mock_async_redis_client):
        """Test getting an existing session"""
        mock_async_redis_client.zscore = AsyncMock(return_value=time.time())

        # Mock messages and metadata
        message = {"role": "user", "content": "Hello"}
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.lrange = MagicMock(return_value=mock_pipeline)
        mock_pipeline.hgetall = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(
            return_value=(
                [json.dumps(message).encode()],
                {b"context": b"test context"},
            )
        )
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        result = await get_session_memory(mock_async_redis_client, "test-session")

        assert result is not None
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"
        assert result.context == "test context"


@pytest.mark.asyncio
class TestSetSessionMemory:
    async def test_set_session_memory_basic(self, mock_async_redis_client):
        """Test basic session memory setting"""
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.zadd = MagicMock(return_value=mock_pipeline)
        mock_pipeline.rpush = MagicMock(return_value=mock_pipeline)
        mock_pipeline.hset = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(return_value=None)
        mock_pipeline.multi = MagicMock(return_value=mock_pipeline)
        mock_async_redis_client.llen = AsyncMock(return_value=5)  # Below window size
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        mock_background_tasks = MagicMock()

        memory = SessionMemory(
            messages=[MemoryMessage(role="user", content="Hello")],
            context="test context",
        )

        settings_patch = patch.multiple(
            "agent_memory_server.messages.settings",
            window_size=20,
            long_term_memory=False,
            generation_model="gpt-4o-mini",
        )

        with settings_patch:
            await set_session_memory(
                mock_async_redis_client, "test-session", memory, mock_background_tasks
            )

        # Verify Redis calls
        mock_pipeline.zadd.assert_called_once()
        mock_pipeline.rpush.assert_called_once()
        mock_pipeline.hset.assert_called_once()
        mock_pipeline.execute.assert_called_once()

        # Verify no background tasks were added (window size not exceeded)
        mock_background_tasks.add_task.assert_not_called()

    async def test_set_session_memory_window_size_exceeded(
        self, mock_async_redis_client
    ):
        """Test session memory setting when window size is exceeded"""
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.zadd = MagicMock(return_value=mock_pipeline)
        mock_pipeline.rpush = MagicMock(return_value=mock_pipeline)
        mock_pipeline.hset = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(return_value=None)
        mock_pipeline.multi = MagicMock(return_value=mock_pipeline)
        mock_async_redis_client.llen = AsyncMock(return_value=21)  # Exceed window size
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        mock_background_tasks = MagicMock()
        mock_background_tasks.add_task = AsyncMock()

        memory = SessionMemory(
            messages=[MemoryMessage(role="user", content="Hello")],
            context="test context",
        )

        settings_patch = patch.multiple(
            "agent_memory_server.messages.settings",
            window_size=20,
            long_term_memory=False,
            generation_model="gpt-4o-mini",
        )

        with settings_patch:
            await set_session_memory(
                mock_async_redis_client, "test-session", memory, mock_background_tasks
            )

        # Verify summarization task was added
        mock_background_tasks.add_task.assert_called_with(
            summarize_session,
            "test-session",
            "gpt-4o-mini",
            20,
        )

        # Verify long-term memory indexing task was not added
        assert mock_background_tasks.add_task.call_count == 1

    async def test_set_session_memory_with_long_term_memory(
        self, mock_async_redis_client
    ):
        """Test session memory setting with long-term memory enabled"""
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.zadd = MagicMock(return_value=mock_pipeline)
        mock_pipeline.rpush = MagicMock(return_value=mock_pipeline)
        mock_pipeline.hset = MagicMock(return_value=mock_pipeline)
        mock_pipeline.execute = AsyncMock(return_value=None)
        mock_pipeline.multi = MagicMock(return_value=mock_pipeline)
        mock_async_redis_client.llen = AsyncMock(return_value=5)  # Below window size
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        mock_background_tasks = MagicMock()
        mock_background_tasks.add_task = AsyncMock()

        memory = SessionMemory(
            messages=[MemoryMessage(role="user", content="Hello")],
            context="test context",
        )

        settings_patch = patch.multiple(
            "agent_memory_server.messages.settings",
            window_size=20,
            long_term_memory=True,
            generation_model="gpt-4o-mini",
        )

        with settings_patch:
            await set_session_memory(
                mock_async_redis_client, "test-session", memory, mock_background_tasks
            )

        # Verify long-term memory indexing task was added
        assert mock_background_tasks.add_task.call_count == 1
        assert mock_background_tasks.add_task.call_args_list == [
            call(
                index_long_term_memories,
                [LongTermMemory(session_id="test-session", text="user: Hello")],
            ),
        ]


@pytest.mark.asyncio
class TestDeleteSessionMemory:
    async def test_delete_session_memory(self, mock_async_redis_client):
        """Test deleting session memory"""
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.delete = MagicMock()
        mock_pipeline.zrem = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=None)
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        await delete_session_memory(mock_async_redis_client, "test-session")

        # Verify Redis pipeline calls
        mock_pipeline.delete.assert_called_once()
        mock_pipeline.zrem.assert_called_once()
        mock_pipeline.execute.assert_called_once()

    async def test_delete_session_memory_with_namespace(self, mock_async_redis_client):
        """Test deleting session memory with namespace"""
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.delete = MagicMock()
        mock_pipeline.zrem = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=None)
        mock_async_redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        await delete_session_memory(
            mock_async_redis_client, "test-session", namespace="test"
        )

        # Verify Redis pipeline calls with namespace
        mock_pipeline.delete.assert_called_once()
        mock_pipeline.zrem.assert_called_with("sessions:test", "test-session")
        mock_pipeline.execute.assert_called_once()
