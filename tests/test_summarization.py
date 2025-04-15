import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.summarization import (
    _incremental_summary,
    summarize_session,
)
from agent_memory_server.utils import Keys


@pytest.mark.asyncio
class TestIncrementalSummarization:
    async def test_incremental_summarization_no_context(self, mock_openai_client):
        """Test incremental summarization without previous context"""
        model = "gpt-3.5-turbo"
        context = None
        messages = [
            json.dumps({"role": "user", "content": "Hello, world!"}),
            json.dumps({"role": "assistant", "content": "How are you?"}),
        ]

        mock_response = MagicMock()
        mock_response.choices = [{"message": {"content": "This is a summary"}}]
        mock_response.total_tokens = 150

        mock_openai_client.create_chat_completion.return_value = mock_response

        summary, tokens_used = await _incremental_summary(
            model, mock_openai_client, context, messages
        )

        assert summary == "This is a summary"
        assert tokens_used == 150

        mock_openai_client.create_chat_completion.assert_called_once()
        args = mock_openai_client.create_chat_completion.call_args[0]

        assert args[0] == model
        assert "How are you?" in args[1]
        assert "Hello, world!" in args[1]

    async def test_incremental_summarization_with_context(self, mock_openai_client):
        """Test incremental summarization with previous context"""
        model = "gpt-3.5-turbo"
        context = "Previous summary"
        messages = [
            json.dumps({"role": "user", "content": "Hello, world!"}),
            json.dumps({"role": "assistant", "content": "How are you?"}),
        ]

        # Create a response that matches our new ChatResponse format
        mock_response = MagicMock()
        mock_response.choices = [{"message": {"content": "Updated summary"}}]
        mock_response.total_tokens = 200

        mock_openai_client.create_chat_completion.return_value = mock_response

        summary, tokens_used = await _incremental_summary(
            model, mock_openai_client, context, messages
        )

        assert summary == "Updated summary"
        assert tokens_used == 200

        mock_openai_client.create_chat_completion.assert_called_once()
        args = mock_openai_client.create_chat_completion.call_args[0]

        assert args[0] == model
        assert "Previous summary" in args[1]
        assert "How are you?" in args[1]
        assert "Hello, world!" in args[1]


class TestSummarizeSession:
    @pytest.mark.asyncio
    @patch("agent_memory_server.summarization._incremental_summary")
    async def test_summarize_session(
        self, mock_summarization, mock_openai_client, mock_async_redis_client
    ):
        """Test summarize_session with mocked summarization"""
        session_id = "test-session"
        model = "gpt-3.5-turbo"
        window_size = 4

        pipeline_mock = MagicMock()  # pipeline is not a coroutine
        pipeline_mock.__aenter__ = AsyncMock(return_value=pipeline_mock)
        pipeline_mock.watch = AsyncMock()
        mock_async_redis_client.pipeline = MagicMock(return_value=pipeline_mock)

        # This needs to match the window size
        messages_raw = [
            json.dumps({"role": "user", "content": "Message 1"}),
            json.dumps({"role": "assistant", "content": "Message 2"}),
            json.dumps({"role": "user", "content": "Message 3"}),
            json.dumps({"role": "assistant", "content": "Message 4"}),
        ]

        pipeline_mock.lrange = AsyncMock(return_value=messages_raw)
        pipeline_mock.hgetall = AsyncMock(
            return_value={
                "context": "Previous summary",
                "tokens": "100",
            }
        )
        pipeline_mock.hmset = MagicMock(return_value=True)
        pipeline_mock.ltrim = MagicMock(return_value=True)
        pipeline_mock.execute = AsyncMock(return_value=True)
        pipeline_mock.llen = AsyncMock(return_value=window_size)

        mock_summarization.return_value = ("New summary", 300)

        with (
            patch(
                "agent_memory_server.summarization.get_model_client"
            ) as mock_get_model_client,
            patch(
                "agent_memory_server.summarization.get_redis_conn",
                return_value=mock_async_redis_client,
            ),
        ):
            mock_get_model_client.return_value = mock_openai_client

            await summarize_session(
                session_id,
                model,
                window_size,
            )

        assert pipeline_mock.lrange.call_count == 1
        assert pipeline_mock.lrange.call_args[0][0] == Keys.messages_key(session_id)
        assert pipeline_mock.lrange.call_args[0][1] == 0
        assert pipeline_mock.lrange.call_args[0][2] == window_size - 1

        assert pipeline_mock.hgetall.call_count == 1
        assert pipeline_mock.hgetall.call_args[0][0] == Keys.metadata_key(session_id)

        assert pipeline_mock.hmset.call_count == 1
        assert pipeline_mock.hmset.call_args[0][0] == Keys.metadata_key(session_id)
        assert pipeline_mock.hmset.call_args.kwargs["mapping"] == {
            "context": "New summary",
            "tokens": "320",
        }

        assert pipeline_mock.ltrim.call_count == 1
        assert pipeline_mock.ltrim.call_args[0][0] == Keys.messages_key(session_id)
        assert pipeline_mock.ltrim.call_args[0][1] == 0
        assert pipeline_mock.ltrim.call_args[0][2] == window_size - 1

        assert pipeline_mock.execute.call_count == 1

        mock_summarization.assert_called_once()
        assert mock_summarization.call_args[0][0] == model
        assert mock_summarization.call_args[0][1] == mock_openai_client
        assert mock_summarization.call_args[0][2] == "Previous summary"
        assert mock_summarization.call_args[0][3] == [
            "user: Message 1",
            "assistant: Message 2",
            "user: Message 3",
            "assistant: Message 4",
        ]

    @pytest.mark.asyncio
    @patch("agent_memory_server.summarization._incremental_summary")
    async def test_handle_summarization_no_messages(
        self, mock_summarization, mock_openai_client, mock_async_redis_client
    ):
        """Test summarize_session when no messages need summarization"""
        session_id = "test-session"
        model = "gpt-3.5-turbo"
        window_size = 12

        pipeline_mock = MagicMock()  # pipeline is not a coroutine
        pipeline_mock.__aenter__ = AsyncMock(return_value=pipeline_mock)
        pipeline_mock.watch = AsyncMock()
        mock_async_redis_client.pipeline = MagicMock(return_value=pipeline_mock)

        pipeline_mock.llen = AsyncMock(return_value=0)
        pipeline_mock.lrange = AsyncMock(return_value=[])
        pipeline_mock.hgetall = AsyncMock(return_value={})
        pipeline_mock.hmset = AsyncMock(return_value=True)
        pipeline_mock.lpop = AsyncMock(return_value=True)
        pipeline_mock.execute = AsyncMock(return_value=True)

        with patch(
            "agent_memory_server.summarization.get_redis_conn",
            return_value=mock_async_redis_client,
        ):
            await summarize_session(
                session_id,
                model,
                window_size,
            )

        assert mock_summarization.call_count == 0
        assert pipeline_mock.lrange.call_count == 0
        assert pipeline_mock.hgetall.call_count == 0
        assert pipeline_mock.hmset.call_count == 0
        assert pipeline_mock.lpop.call_count == 0
        assert pipeline_mock.execute.call_count == 0
