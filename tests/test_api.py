from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agent_memory_server.config import Settings
from agent_memory_server.long_term_memory import index_long_term_memories
from agent_memory_server.models import (
    LongTermMemoryResult,
    LongTermMemoryResultsResponse,
    MemoryMessage,
    SessionListResponse,
    SessionMemoryResponse,
)
from agent_memory_server.summarization import summarize_session


@pytest.fixture
def mock_openai_client_wrapper():
    """Create a mock OpenAIClientWrapper that doesn't need an API key"""
    with patch("agent_memory_server.models.OpenAIClientWrapper") as mock_wrapper:
        # Create a mock instance
        mock_instance = AsyncMock()
        mock_wrapper.return_value = mock_instance

        # Mock the create_embedding and create_chat_completion methods
        mock_instance.create_embedding.return_value = np.array(
            [[0.1] * 1536], dtype=np.float32
        )
        mock_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 100},
        }

        yield mock_wrapper


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test the health endpoint"""
        response = await client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert "now" in data
        assert isinstance(data["now"], int)


class TestMemoryEndpoints:
    async def test_list_sessions_empty(self, client):
        """Test the list_sessions endpoint with no sessions"""
        response = await client.get("/sessions/?offset=0&limit=10")

        assert response.status_code == 200

        data = response.json()
        response = SessionListResponse(**data)
        assert response.sessions == []
        assert response.total == 0

    async def test_list_sessions_with_sessions(self, client, session):
        """Test the list_sessions endpoint with a session"""
        response = await client.get(
            "/sessions/?offset=0&limit=10&namespace=test-namespace"
        )
        assert response.status_code == 200

        data = response.json()
        response = SessionListResponse(**data)
        assert response.sessions == [session]
        assert response.total == 1

    async def test_get_memory(self, client, session):
        """Test the get_memory endpoint"""
        session_id = session

        response = await client.get(
            f"/sessions/{session_id}/memory?namespace=test-namespace"
        )

        assert response.status_code == 200

        data = response.json()
        response = SessionMemoryResponse(**data)
        assert response.messages == [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        roles = [msg["role"] for msg in data["messages"]]
        contents = [msg["content"] for msg in data["messages"]]
        assert "user" in roles
        assert "assistant" in roles
        assert "Hello" in contents
        assert "Hi there" in contents

        # Check context and tokens
        assert data["context"] == "Sample context"
        assert int(data["tokens"]) == 150  # Convert string to int for comparison

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_put_memory(self, client):
        """Test the post_memory endpoint"""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "context": "Previous context",
            "namespace": "test-namespace",
        }

        response = await client.put("/sessions/test-session/memory", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_put_memory_stores_messages_in_long_term_memory(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test the put_memory endpoint"""
        client = client_with_mock_background_tasks
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "context": "Previous context",
        }
        mock_settings = Settings(long_term_memory=True)

        with patch("agent_memory_server.api.settings", mock_settings):
            response = await client.put("/sessions/test-session/memory", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

        # Check that background tasks were called
        assert mock_background_tasks.add_task.call_count == 1

        # Check that the last call was for long-term memory indexing
        assert (
            mock_background_tasks.add_task.call_args_list[-1][0][0]
            == index_long_term_memories
        )

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_post_memory_compacts_long_conversation(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test the post_memory endpoint"""
        client = client_with_mock_background_tasks
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "context": "Previous context",
        }
        mock_settings = Settings(window_size=1, long_term_memory=False)
        MagicMock()

        with patch("agent_memory_server.api.messages.settings", mock_settings):
            response = await client.put("/sessions/test-session/memory", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

        # Check that background tasks were called
        assert mock_background_tasks.add_task.call_count == 1

        # Check that the last call was for compaction
        assert (
            mock_background_tasks.add_task.call_args_list[-1][0][0] == summarize_session
        )

    @pytest.mark.asyncio
    async def test_delete_memory(self, client, session):
        """Test the delete_memory endpoint"""
        session_id = session

        response = await client.get(
            f"/sessions/{session_id}/memory?namespace=test-namespace"
        )

        assert response.status_code == 200

        data = response.json()
        assert len(data["messages"]) == 2

        response = await client.delete(
            f"/sessions/{session_id}/memory?namespace=test-namespace"
        )

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

        response = await client.get(
            f"/sessions/{session_id}/memory?namespace=test-namespace"
        )
        assert response.status_code == 404


@pytest.mark.requires_api_keys
class TestSearchEndpoint:
    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_search(self, mock_search, client):
        """Test the search endpoint"""
        mock_search.return_value = LongTermMemoryResultsResponse(
            memories=[
                LongTermMemoryResult(id_="1", text="User: Hello, world!", dist=0.25),
                LongTermMemoryResult(id_="2", text="Assistant: Hi there!", dist=0.75),
            ],
            total=2,
        )

        # Create payload
        payload = {"text": "What is the capital of France?"}

        # Call endpoint with the correct URL format (matching the router definition)
        response = await client.post("/long-term-memory/search", json=payload)

        # Check status code
        assert response.status_code == 200, response.text

        # Check response structure
        data = response.json()
        assert "memories" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["memories"]) == 2

        # Check first result
        assert data["memories"][0]["id_"] == "1"
        assert data["memories"][0]["text"] == "User: Hello, world!"
        assert data["memories"][0]["dist"] == 0.25

        # Check second result
        assert data["memories"][1]["id_"] == "2"
        assert data["memories"][1]["text"] == "Assistant: Hi there!"
        assert data["memories"][1]["dist"] == 0.75
