from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agent_memory_server.config import Settings
from agent_memory_server.long_term_memory import (
    promote_working_memory_to_long_term,
)
from agent_memory_server.models import (
    MemoryMessage,
    MemoryRecordResult,
    MemoryRecordResultsResponse,
    SessionListResponse,
    WorkingMemory,
    WorkingMemoryResponse,
)


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
        response = await client.get("/v1/health")

        assert response.status_code == 200

        data = response.json()
        assert "now" in data
        assert isinstance(data["now"], int)


class TestMemoryEndpoints:
    async def test_list_sessions_empty(self, client):
        """Test the list_sessions endpoint with no sessions"""
        response = await client.get("/v1/working-memory/?offset=0&limit=10")

        assert response.status_code == 200

        data = response.json()
        response = SessionListResponse(**data)
        assert response.sessions == []
        assert response.total == 0

    async def test_list_sessions_with_sessions(self, client, session):
        """Test the list_sessions endpoint with a session"""
        response = await client.get(
            "/v1/working-memory/?offset=0&limit=10&namespace=test-namespace"
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
            f"/v1/working-memory/{session_id}?namespace=test-namespace&user_id=test-user"
        )

        assert response.status_code == 200

        data = response.json()
        response = WorkingMemoryResponse(**data)

        # Check that we have 2 messages with correct roles and content
        assert len(response.messages) == 2

        # Check message content and roles (IDs are auto-generated so we can't compare directly)
        message_contents = [msg.content for msg in response.messages]
        message_roles = [msg.role for msg in response.messages]
        assert "Hello" in message_contents
        assert "Hi there" in message_contents
        assert "user" in message_roles
        assert "assistant" in message_roles

        # Check that all messages have IDs (auto-generated)
        for msg in response.messages:
            assert msg.id is not None
            assert len(msg.id) > 0

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
            "memories": [],
            "context": "Previous context",
            "namespace": "test-namespace",
            "session_id": "test-session",
        }

        response = await client.put("/v1/working-memory/test-session", json=payload)

        assert response.status_code == 200

        data = response.json()
        # Should return the working memory, not just a status
        assert "messages" in data
        assert "context" in data
        assert "namespace" in data
        assert data["context"] == "Previous context"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "Hi there"

        # Verify we can still retrieve the session memory
        updated_session = await client.get(
            "/v1/working-memory/test-session?namespace=test-namespace"
        )
        assert updated_session.status_code == 200
        retrieved_messages = updated_session.json()["messages"]
        assert len(retrieved_messages) == len(payload["messages"])
        for i, msg in enumerate(retrieved_messages):
            assert msg["role"] == payload["messages"][i]["role"]
            assert msg["content"] == payload["messages"][i]["content"]

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
            "memories": [],
            "context": "Previous context",
            "namespace": "test-namespace",
            "session_id": "test-session",
        }
        mock_settings = Settings(long_term_memory=True)

        with patch("agent_memory_server.api.settings", mock_settings):
            response = await client.put("/v1/working-memory/test-session", json=payload)

        assert response.status_code == 200

        data = response.json()
        # Should return the working memory, not just a status
        assert "messages" in data
        assert "context" in data
        assert data["context"] == "Previous context"

        # Check that background tasks were called
        assert mock_background_tasks.add_task.call_count == 1

        # Check that the last call was for long-term memory promotion
        assert (
            mock_background_tasks.add_task.call_args_list[-1][0][0]
            == promote_working_memory_to_long_term
        )

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_put_memory_with_structured_memories_triggers_promotion(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test that structured memories trigger background promotion task"""
        client = client_with_mock_background_tasks
        payload = {
            "messages": [],
            "memories": [
                {
                    "text": "User prefers dark mode",
                    "id": "test-memory-1",
                    "memory_type": "semantic",
                    "namespace": "test-namespace",
                }
            ],
            "context": "Previous context",
            "namespace": "test-namespace",
            "session_id": "test-session",
        }
        mock_settings = Settings(long_term_memory=True)

        with patch("agent_memory_server.api.settings", mock_settings):
            response = await client.put("/v1/working-memory/test-session", json=payload)

        assert response.status_code == 200

        data = response.json()
        assert "memories" in data
        assert len(data["memories"]) == 1
        assert data["memories"][0]["text"] == "User prefers dark mode"

        # Check that promotion background task was called
        assert mock_background_tasks.add_task.call_count == 1

        # Check that it was the promotion task, not indexing
        assert (
            mock_background_tasks.add_task.call_args_list[0][0][0]
            == promote_working_memory_to_long_term
        )

        # Check the arguments passed to the promotion task
        task_call = mock_background_tasks.add_task.call_args_list[0]
        task_kwargs = task_call[1]
        assert task_kwargs["session_id"] == "test-session"
        assert task_kwargs["namespace"] == "test-namespace"

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_post_memory_compacts_long_conversation(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test the post_memory endpoint with window size exceeded"""
        client = client_with_mock_background_tasks
        payload = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "memories": [],
            "context": "Previous context",
            "namespace": "test-namespace",
            "session_id": "test-session",
        }
        mock_settings = Settings(long_term_memory=False)

        with (
            patch("agent_memory_server.api.settings", mock_settings),
            patch(
                "agent_memory_server.api._summarize_working_memory"
            ) as mock_summarize,
        ):
            # Mock the summarization to return the working memory with updated context
            mock_summarized_memory = WorkingMemory(
                messages=[
                    MemoryMessage(role="assistant", content="Hi there")
                ],  # Only keep last message
                memories=[],
                context="Summary: User greeted and assistant responded.",
                session_id="test-session",
                namespace="test-namespace",
            )
            mock_summarize.return_value = mock_summarized_memory

            response = await client.put("/v1/working-memory/test-session", json=payload)

        assert response.status_code == 200

        data = response.json()
        # Should return the summarized working memory
        assert "messages" in data
        assert "context" in data
        # Should have been summarized (token-based summarization in _summarize_working_memory)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Hi there"
        assert "Summary:" in data["context"]

        # Verify summarization was called
        mock_summarize.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_memory(self, client, session):
        """Test the delete_memory endpoint"""
        session_id = session

        response = await client.get(
            f"/v1/working-memory/{session_id}?namespace=test-namespace&user_id=test-user"
        )

        assert response.status_code == 200

        data = response.json()
        assert len(data["messages"]) == 2

        response = await client.delete(
            f"/v1/working-memory/{session_id}?namespace=test-namespace&user_id=test-user"
        )

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

        response = await client.get(
            f"/v1/working-memory/{session_id}?namespace=test-namespace&user_id=test-user"
        )
        assert response.status_code == 200

        # Should return empty working memory after deletion
        data = response.json()
        assert len(data["messages"]) == 0


@pytest.mark.requires_api_keys
class TestSearchEndpoint:
    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_search(self, mock_search, client):
        """Test the search endpoint"""
        mock_search.return_value = MemoryRecordResultsResponse(
            total=2,
            memories=[
                MemoryRecordResult(id="1", text="User: Hello, world!", dist=0.25),
                MemoryRecordResult(id="2", text="Assistant: Hi there!", dist=0.75),
            ],
            next_offset=None,
        )

        # Create payload
        payload = {"text": "What is the capital of France?"}

        # Call endpoint with the correct URL format (matching the router definition)
        response = await client.post("/v1/long-term-memory/search", json=payload)

        # Check status code
        assert response.status_code == 200, response.text

        # Check response structure
        data = response.json()
        assert "memories" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["memories"]) == 2

    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_search_with_optimize_query_true(self, mock_search, client):
        """Test search endpoint with optimize_query=True (default)."""
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(id="1", text="Optimized result", dist=0.1),
            ],
            next_offset=None,
        )

        payload = {"text": "tell me about my preferences"}

        # Call endpoint without optimize_query parameter (should default to True)
        response = await client.post("/v1/long-term-memory/search", json=payload)

        assert response.status_code == 200

        # Verify search was called with optimize_query=True (default)
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("optimize_query") is True

    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_search_with_optimize_query_false(self, mock_search, client):
        """Test search endpoint with optimize_query=False."""
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(id="1", text="Non-optimized result", dist=0.1),
            ],
            next_offset=None,
        )

        payload = {"text": "tell me about my preferences"}

        # Call endpoint with optimize_query=False as query parameter
        response = await client.post(
            "/v1/long-term-memory/search",
            json=payload,
            params={"optimize_query": "false"},
        )

        assert response.status_code == 200

        # Verify search was called with optimize_query=False
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("optimize_query") is False

    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_search_with_optimize_query_explicit_true(self, mock_search, client):
        """Test search endpoint with explicit optimize_query=True."""
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(id="1", text="Optimized result", dist=0.1),
            ],
            next_offset=None,
        )

        payload = {"text": "what are my UI settings"}

        # Call endpoint with explicit optimize_query=True
        response = await client.post(
            "/v1/long-term-memory/search",
            json=payload,
            params={"optimize_query": "true"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify search was called with optimize_query=True
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("optimize_query") is True

        # Check response structure
        assert "memories" in data
        assert len(data["memories"]) == 1
        assert data["memories"][0]["id"] == "1"
        assert data["memories"][0]["text"] == "Optimized result"


@pytest.mark.requires_api_keys
class TestMemoryPromptEndpoint:
    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_session_id(self, mock_get_working_memory, client):
        """Test the memory_prompt endpoint with only session_id provided"""
        # Mock the session memory
        mock_working_memory = WorkingMemoryResponse(
            messages=[
                MemoryMessage(role="user", content="Hello"),
                MemoryMessage(role="assistant", content="Hi there"),
            ],
            memories=[],
            session_id="test-session",
            context="Previous conversation context",
            tokens=150,
        )
        mock_get_working_memory.return_value = mock_working_memory

        # Call the endpoint
        query = "What's the weather like?"
        response = await client.post(
            "/v1/memory/prompt",
            json={
                "query": query,
                "session": {
                    "session_id": "test-session",
                    "namespace": "test-namespace",
                    "model_name": "gpt-4o",
                    "context_window_max": 1000,
                },
            },
        )

        # Check status code
        assert response.status_code == 200

        # Check response data
        data = response.json()
        assert isinstance(data, dict)
        assert (
            len(data["messages"]) == 4
        )  # Context message + 2 session messages + query

        # Verify the messages content
        assert data["messages"][0]["role"] == "system"
        assert "Previous conversation context" in data["messages"][0]["content"]["text"]
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"]["text"] == "Hello"
        assert data["messages"][2]["role"] == "assistant"
        assert data["messages"][2]["content"]["text"] == "Hi there"
        assert data["messages"][3]["role"] == "user"
        assert data["messages"][3]["content"]["text"] == query

    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_long_term_memory(self, mock_search, client):
        """Test the memory_prompt endpoint with only long_term_search_payload provided"""
        # Mock the long-term memory search
        mock_search.return_value = MemoryRecordResultsResponse(
            total=2,
            memories=[
                MemoryRecordResult(id="1", text="User likes coffee", dist=0.25),
                MemoryRecordResult(
                    id="2", text="User is allergic to peanuts", dist=0.35
                ),
            ],
            next_offset=None,
        )

        # Prepare the payload
        payload = {
            "query": "What should I eat?",
            "long_term_search": {
                "text": "food preferences allergies",
            },
        }

        # Call the endpoint
        response = await client.post("/v1/memory/prompt", json=payload)

        # Check status code
        assert response.status_code == 200

        # Check response data
        data = response.json()
        assert isinstance(data, dict)
        assert len(data["messages"]) == 2  # Long-term memory message + query

        # Verify the messages content
        assert data["messages"][0]["role"] == "system"
        assert "Long term memories" in data["messages"][0]["content"]["text"]
        assert "User likes coffee" in data["messages"][0]["content"]["text"]
        assert "User is allergic to peanuts" in data["messages"][0]["content"]["text"]
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"]["text"] == "What should I eat?"

    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_both_sources(
        self, mock_search, mock_get_working_memory, client
    ):
        """Test the memory_prompt endpoint with both session_id and long_term_search_payload"""
        # Mock session memory
        mock_working_memory = WorkingMemoryResponse(
            messages=[
                MemoryMessage(role="user", content="How do you make pasta?"),
                MemoryMessage(
                    role="assistant",
                    content="Boil water, add pasta, cook until al dente.",
                ),
            ],
            memories=[],
            session_id="test-session",
            context="Cooking conversation",
            tokens=200,
        )
        mock_get_working_memory.return_value = mock_working_memory

        # Mock the long-term memory search
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="1", text="User prefers gluten-free pasta", dist=0.3
                ),
            ],
            next_offset=None,
        )

        # Prepare the payload
        payload = {
            "query": "What pasta should I buy?",
            "session": {
                "session_id": "test-session",
                "namespace": "test-namespace",
            },
            "long_term_search": {
                "text": "pasta preferences",
            },
        }

        # Call the endpoint
        response = await client.post("/v1/memory/prompt", json=payload)

        # Check status code
        assert response.status_code == 200

        # Check response data
        data = response.json()
        assert isinstance(data, dict)
        assert (
            len(data["messages"]) == 5
        )  # Context + 2 session messages + long-term memory + query

        # Verify the messages content (order matters)
        assert data["messages"][0]["role"] == "system"
        assert "Cooking conversation" in data["messages"][0]["content"]["text"]
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"]["text"] == "How do you make pasta?"
        assert data["messages"][2]["role"] == "assistant"
        assert (
            data["messages"][2]["content"]["text"]
            == "Boil water, add pasta, cook until al dente."
        )
        assert data["messages"][3]["role"] == "system"
        assert "Long term memories" in data["messages"][3]["content"]["text"]
        assert (
            "User prefers gluten-free pasta" in data["messages"][3]["content"]["text"]
        )
        assert data["messages"][4]["role"] == "user"
        assert data["messages"][4]["content"]["text"] == "What pasta should I buy?"

    @pytest.mark.asyncio
    async def test_memory_prompt_without_required_params(self, client):
        """Test the memory_prompt endpoint without required parameters"""
        # Call the endpoint without session or long_term_search
        response = await client.post("/v1/memory/prompt", json={"query": "test"})

        # Check status code (should be 400 Bad Request)
        assert response.status_code == 400

        # Check error message
        data = response.json()
        assert "detail" in data
        assert "Either session or long_term_search must be provided" in data["detail"]

    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @pytest.mark.asyncio
    async def test_memory_prompt_session_not_found(
        self, mock_get_working_memory, client
    ):
        """Test the memory_prompt endpoint when session is not found"""
        # Mock the session memory to return None (session not found)
        mock_get_working_memory.return_value = None

        # Call the endpoint
        query = "What's the weather like?"
        response = await client.post(
            "/v1/memory/prompt",
            json={
                "query": query,
                "session": {
                    "session_id": "nonexistent-session",
                    "namespace": "test-namespace",
                },
            },
        )

        # Check status code (should be successful)
        assert response.status_code == 200

        # Check response data (should only contain the query)
        data = response.json()
        assert isinstance(data, dict)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"]["text"] == query

    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @patch("agent_memory_server.api.get_model_config")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_model_name(
        self, mock_get_model_config, mock_get_working_memory, client
    ):
        """Test the memory_prompt endpoint with model_name parameter"""
        # Mock the model config
        model_config = MagicMock()
        model_config.max_tokens = 4000
        mock_get_model_config.return_value = model_config

        # Mock the session memory
        mock_working_memory = WorkingMemoryResponse(
            messages=[
                MemoryMessage(role="user", content="Hello"),
                MemoryMessage(role="assistant", content="Hi there"),
            ],
            memories=[],
            session_id="test-session",
            context="Previous context",
            tokens=150,
        )
        mock_get_working_memory.return_value = mock_working_memory

        # Call the endpoint with model_name
        query = "What's the weather like?"
        response = await client.post(
            "/v1/memory/prompt",
            json={
                "query": query,
                "session": {
                    "session_id": "test-session",
                    "model_name": "gpt-4o",
                },
            },
        )

        # Check the model config was used
        mock_get_model_config.assert_called_once_with("gpt-4o")

        # Check status code
        assert response.status_code == 200

        # Verify the working memory function was called
        mock_get_working_memory.assert_called_once()

    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_optimize_query_default_true(
        self, mock_get_working_memory, mock_search, client
    ):
        """Test memory prompt endpoint with default optimize_query=True."""
        # Mock working memory
        mock_get_working_memory.return_value = WorkingMemoryResponse(
            session_id="test-session",
            messages=[
                MemoryMessage(role="user", content="Hello"),
                MemoryMessage(role="assistant", content="Hi there"),
            ],
            memories=[],
            context=None,
        )

        # Mock search for long-term memory
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(id="1", text="User preferences about UI", dist=0.1),
            ],
            next_offset=None,
        )

        payload = {
            "query": "what are my preferences?",
            "session": {"session_id": "test-session"},
            "long_term_search": {"text": "preferences"},
        }

        # Call endpoint without optimize_query parameter (should default to True)
        response = await client.post("/v1/memory/prompt", json=payload)

        assert response.status_code == 200

        # Verify search was called with optimize_query=True (default)
        mock_search.assert_called_once()
        # The search is called indirectly through the API's search_long_term_memory function
        # which should have optimize_query=True by default

    @patch("agent_memory_server.api.long_term_memory.search_long_term_memories")
    @patch("agent_memory_server.api.working_memory.get_working_memory")
    @pytest.mark.asyncio
    async def test_memory_prompt_with_optimize_query_false(
        self, mock_get_working_memory, mock_search, client
    ):
        """Test memory prompt endpoint with optimize_query=False."""
        # Mock working memory
        mock_get_working_memory.return_value = WorkingMemoryResponse(
            session_id="test-session",
            messages=[
                MemoryMessage(role="user", content="Hello"),
                MemoryMessage(role="assistant", content="Hi there"),
            ],
            memories=[],
            context=None,
        )

        # Mock search for long-term memory
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(id="1", text="User preferences about UI", dist=0.1),
            ],
            next_offset=None,
        )

        payload = {
            "query": "what are my preferences?",
            "session": {"session_id": "test-session"},
            "long_term_search": {"text": "preferences"},
        }

        # Call endpoint with optimize_query=False as query parameter
        response = await client.post(
            "/v1/memory/prompt", json=payload, params={"optimize_query": "false"}
        )

        assert response.status_code == 200


@pytest.mark.requires_api_keys
class TestLongTermMemoryEndpoint:
    @pytest.mark.asyncio
    async def test_create_long_term_memory_with_valid_id(self, client):
        """Test creating long-term memory with valid id"""
        payload = {
            "memories": [
                {
                    "text": "User prefers dark mode",
                    "user_id": "user123",
                    "session_id": "session123",
                    "namespace": "test",
                    "memory_type": "semantic",
                    "id": "test-client-123",
                }
            ]
        }

        response = await client.post("/v1/long-term-memory/", json=payload)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_create_long_term_memory_missing_id(self, client):
        """Test creating long-term memory without id should fail"""
        payload = {
            "memories": [
                {
                    "text": "User prefers dark mode",
                    "user_id": "user123",
                    "session_id": "session123",
                    "namespace": "test",
                    "memory_type": "semantic",
                    # Missing id field
                }
            ]
        }

        response = await client.post("/v1/long-term-memory/", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "Field required" in str(data["detail"])

    @pytest.mark.requires_api_keys
    @pytest.mark.asyncio
    async def test_create_long_term_memory_persisted_at_ignored(self, client):
        """Test that client-provided persisted_at is ignored"""
        payload = {
            "memories": [
                {
                    "text": "User prefers dark mode",
                    "id": "test-client-456",
                    "memory_type": "semantic",
                    "persisted_at": "2023-01-01T00:00:00Z",  # Use ISO string instead of datetime object
                }
            ]
        }

        response = await client.post("/v1/long-term-memory/", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_delete_long_term_memory_success(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test successfully deleting long-term memories"""
        client = client_with_mock_background_tasks

        memory_ids = ["memory-1", "memory-2", "memory-3"]

        mock_settings = Settings(long_term_memory=True)

        # Mock the delete_long_term_memories function to return a count
        with (
            patch("agent_memory_server.api.settings", mock_settings),
            patch(
                "agent_memory_server.api.long_term_memory.delete_long_term_memories"
            ) as mock_delete,
        ):
            mock_delete.return_value = 3  # 3 memories deleted

            response = await client.delete(
                "/v1/long-term-memory", params={"memory_ids": memory_ids}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok, deleted 3 memories"

        # Verify delete function was called with correct arguments
        mock_delete.assert_called_once_with(ids=["memory-1", "memory-2", "memory-3"])

    @pytest.mark.asyncio
    async def test_delete_long_term_memory_empty_list(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test deleting long-term memories with empty ID list"""
        client = client_with_mock_background_tasks

        memory_ids = []

        mock_settings = Settings(long_term_memory=True)

        # Mock the delete_long_term_memories function to return zero count
        with (
            patch("agent_memory_server.api.settings", mock_settings),
            patch(
                "agent_memory_server.api.long_term_memory.delete_long_term_memories"
            ) as mock_delete,
        ):
            mock_delete.return_value = 0  # No memories deleted

            response = await client.delete(
                "/v1/long-term-memory", params={"memory_ids": memory_ids}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok, deleted 0 memories"

        # Verify delete function was called
        mock_delete.assert_called_once_with(ids=[])

    @pytest.mark.asyncio
    async def test_delete_long_term_memory_disabled(self, client):
        """Test deleting long-term memories when long-term memory is disabled"""
        memory_ids = ["memory-1", "memory-2"]

        mock_settings = Settings(long_term_memory=False)

        with patch("agent_memory_server.api.settings", mock_settings):
            response = await client.delete(
                "/v1/long-term-memory", params={"memory_ids": memory_ids}
            )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"] == "Long-term memory is disabled"

    @pytest.mark.asyncio
    async def test_delete_long_term_memory_no_parameters(
        self, client_with_mock_background_tasks, mock_background_tasks
    ):
        """Test deleting long-term memories with no parameters (defaults to empty list)"""
        client = client_with_mock_background_tasks

        mock_settings = Settings(long_term_memory=True)

        # Mock the delete_long_term_memories function to return zero count for empty list
        with (
            patch("agent_memory_server.api.settings", mock_settings),
            patch(
                "agent_memory_server.api.long_term_memory.delete_long_term_memories"
            ) as mock_delete,
        ):
            mock_delete.return_value = 0  # No memories to delete

            response = await client.delete("/v1/long-term-memory")

        # Should succeed with 0 deletions (empty list is valid)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok, deleted 0 memories"

        # Verify delete function was called with empty list
        mock_delete.assert_called_once_with(ids=[])
