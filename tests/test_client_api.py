"""
Test file for the Redis Memory Server API Client.

This file contains tests that demonstrate how to use the Memory API client.
"""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from agent_memory_client.filters import Namespace, SessionId, Topics, UserId
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from agent_memory_server.api import router as memory_router
from agent_memory_server.healthcheck import router as health_router
from agent_memory_server.models import (
    MemoryMessage,
    MemoryPromptResponse,
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResultsResponse,
    MemoryTypeEnum,
    SystemMessage,
    WorkingMemory,
    WorkingMemoryResponse,
)


class MockMessage:
    """Mock message class to simulate MCP message objects for testing"""

    def __init__(self, message_dict):
        self.content = MockContent(message_dict.get("content", {}))
        self.role = message_dict.get("role", "user")


class MockContent:
    """Mock content class to simulate TextContent for testing"""

    def __init__(self, content_dict):
        self.text = content_dict.get("text", "")
        self.type = content_dict.get("type", "text")


class MockMemoryPromptResponse:
    """Mock response class to simulate MemoryPromptResponse for testing"""

    def __init__(self, response_dict):
        self.messages = [MockMessage(msg) for msg in response_dict.get("messages", [])]


@pytest.fixture
def memory_app() -> FastAPI:
    """Create a test FastAPI app with memory routers for testing the client."""
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(memory_router)
    return app


@pytest.fixture
async def memory_test_client(
    memory_app: FastAPI,
) -> AsyncGenerator[MemoryAPIClient, None]:
    """Create a memory client that uses the test FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=memory_app),
        base_url="http://test",
    ) as http_client:
        # Create the memory client with our test http client
        config = MemoryClientConfig(
            base_url="http://test", default_namespace="test-namespace"
        )
        client = MemoryAPIClient(config)

        # Replace the internal http client with our test client
        client._client = http_client

        yield client


@pytest.mark.asyncio
async def test_health_check(memory_test_client: MemoryAPIClient):
    """Test the health check endpoint"""
    # Mock the response from the health endpoint
    response = await memory_test_client.health_check()
    assert response.now > 0


@pytest.mark.asyncio
async def test_session_lifecycle(memory_test_client: MemoryAPIClient):
    """Test the complete lifecycle of a session"""
    # For this test, we need to set up mocks for all the API calls
    session_id = "test-client-session"

    # Mock memory data
    memory = WorkingMemory(
        messages=[
            MemoryMessage(role="user", content="Hello from the client!"),
            MemoryMessage(role="assistant", content="Hi there, I'm the memory server!"),
        ],
        memories=[],
        context="This is a test session created by the API client.",
        session_id=session_id,
    )

    # First, mock PUT response for creating a session
    with patch(
        "agent_memory_server.working_memory.set_working_memory"
    ) as mock_set_memory:
        mock_set_memory.return_value = None

        # Step 1: Create new session memory
        response = await memory_test_client.put_working_memory(session_id, memory)
        assert response.messages[0].content == "Hello from the client!"
        assert response.messages[1].content == "Hi there, I'm the memory server!"
        assert response.context == "This is a test session created by the API client."

    # Next, mock GET response for retrieving session memory
    with patch(
        "agent_memory_server.working_memory.get_working_memory"
    ) as mock_get_memory:
        # Get memory data and explicitly exclude session_id to avoid duplicate parameter
        memory_data = memory.model_dump(exclude={"session_id"})
        mock_response = WorkingMemoryResponse(**memory_data, session_id=session_id)
        mock_get_memory.return_value = mock_response

        # Step 2: Retrieve the session memory
        session = await memory_test_client.get_working_memory(session_id)
        assert len(session.messages) == 2
        assert session.messages[0].content == "Hello from the client!"
        assert session.messages[1].content == "Hi there, I'm the memory server!"
        assert session.context == "This is a test session created by the API client."

    # Mock list sessions
    with patch(
        "agent_memory_server.working_memory.list_sessions"
    ) as mock_list_sessions:
        mock_list_sessions.return_value = (1, [session_id])

        # Step 3: List sessions and verify our test session is included
        sessions = await memory_test_client.list_sessions()
        assert session_id in sessions.sessions

    # Mock delete session
    with patch(
        "agent_memory_server.working_memory.delete_working_memory"
    ) as mock_delete:
        mock_delete.return_value = None

        # Step 4: Delete the session
        response = await memory_test_client.delete_working_memory(session_id)
        assert response.status == "ok"

    # Verify it's gone by mocking a 404 response
    with patch(
        "agent_memory_server.working_memory.get_working_memory"
    ) as mock_get_memory:
        mock_get_memory.return_value = None

        # This should not raise an error anymore since the unified API returns empty working memory instead of 404
        session = await memory_test_client.get_working_memory(session_id)
        assert len(session.messages) == 0  # Should return empty working memory


@pytest.mark.asyncio
async def test_long_term_memory(memory_test_client: MemoryAPIClient):
    """Test long-term memory creation and search"""
    # Create some test memories
    memories = [
        MemoryRecord(
            text="User prefers dark mode",
            id="test-client-1",
            memory_type=MemoryTypeEnum.SEMANTIC,
            user_id="user123",
        ),
        MemoryRecord(
            text="User is working on a Python project",
            id="test-client-2",
            memory_type=MemoryTypeEnum.EPISODIC,
            user_id="user123",
        ),
    ]

    # Mock the memory creation
    with patch(
        "agent_memory_server.long_term_memory.index_long_term_memories"
    ) as mock_index:
        mock_index.return_value = None

        # Store the memories
        with patch("agent_memory_server.api.settings.long_term_memory", True):
            response = await memory_test_client.create_long_term_memory(memories)
            assert response.status == "ok"

    # Mock the search results
    with patch(
        "agent_memory_server.long_term_memory.search_long_term_memories"
    ) as mock_search:
        mock_search.return_value = MemoryRecordResultsResponse(
            total=2,
            memories=[
                MemoryRecordResult(
                    id="result-1",
                    text="User prefers dark mode",
                    dist=0.1,
                    user_id="user123",
                    namespace="preferences",
                ),
                MemoryRecordResult(
                    id="result-2",
                    text="User likes coffee",
                    dist=0.2,
                    user_id="user123",
                    namespace="preferences",
                ),
            ],
            next_offset=None,
        )

        # Search with various filters
        with patch("agent_memory_server.api.settings.long_term_memory", True):
            results = await memory_test_client.search_long_term_memory(
                text="What color does the user prefer?",
                user_id=UserId(eq="test-user"),
                topics=Topics(any=["colors", "preferences"]),
            )

            assert results.total == 2
            # Check that we got the memories we created
            assert any(
                "dark mode" in memory.text.lower() for memory in results.memories
            )

            # Try another search using filter objects instead of dictionaries
            results = await memory_test_client.search_long_term_memory(
                text="dark mode",
                namespace=Namespace(eq="test-namespace"),
            )

            assert results.total == 2
            assert any(
                "dark mode" in memory.text.lower() for memory in results.memories
            )


@pytest.mark.asyncio
async def test_client_with_context_manager(memory_app: FastAPI):
    """Test using the client with a context manager"""
    async with (
        AsyncClient(
            transport=ASGITransport(app=memory_app),
            base_url="http://test",
        ) as http_client,
        MemoryAPIClient(MemoryClientConfig(base_url="http://test")) as client,
    ):
        # Replace the internal client
        client._client = http_client

        # Perform a simple health check
        response = await client.health_check()
        assert response.now > 0

        # The client will be automatically closed when the context block exits


@pytest.mark.asyncio
async def test_memory_prompt(memory_test_client: MemoryAPIClient):
    """Test the memory_prompt method"""
    session_id = "test-client-session"
    query = "What was my favorite color?"

    # Create expected response
    expected_messages = [
        base.UserMessage(
            content=TextContent(type="text", text="What is your favorite color?"),
        ),
        base.AssistantMessage(
            content=TextContent(type="text", text="I like blue, how about you?"),
        ),
        base.UserMessage(
            content=TextContent(type="text", text=query),
        ),
    ]

    # Create expected response payload
    expected_response = MemoryPromptResponse(messages=expected_messages)

    # Mock the HTTP client's post method directly
    with patch.object(memory_test_client._client, "post") as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_response.json = MagicMock(return_value=expected_response.model_dump())
        mock_post.return_value = mock_response

        # Test the client method
        response = await memory_test_client.memory_prompt(
            query=query,
            session_id=session_id,
            namespace="test-namespace",
            model_name="gpt-4o",
            context_window_max=4000,
        )

        # Convert raw dict response to mock object for testing
        response = MockMemoryPromptResponse(response)

        # Verify the response
        assert len(response.messages) == 3
        assert isinstance(response.messages[0].content, MockContent)
        assert response.messages[0].content.text.startswith(
            "What is your favorite color?"
        )
        assert isinstance(response.messages[-1].content, MockContent)
        assert response.messages[-1].content.text == query

        # Test without session_id (only semantic search)
        mock_post.reset_mock()
        mock_post.return_value = mock_response

        response = await memory_test_client.memory_prompt(
            query=query,
        )

        # Convert raw dict response to mock object for testing
        response = MockMemoryPromptResponse(response)

        # Verify the response is the same (it's mocked)
        assert len(response.messages) == 3


@pytest.mark.asyncio
async def test_hydrate_memory_prompt(memory_test_client: MemoryAPIClient):
    """Test the hydrate_memory_prompt method with filters"""
    query = "What was my favorite color?"

    # Create expected response
    expected_messages = [
        base.AssistantMessage(
            content=TextContent(
                type="text",
                text="The user's favorite color is blue",
            ),
        ),
        base.UserMessage(
            content=TextContent(type="text", text=query),
        ),
    ]

    # Create expected response payload
    expected_response = MemoryPromptResponse(messages=expected_messages)

    # Mock the HTTP client's post method directly
    with patch.object(memory_test_client._client, "post") as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_response.json = MagicMock(return_value=expected_response.model_dump())
        mock_post.return_value = mock_response

        # Test with filter dictionaries
        response = await memory_test_client.hydrate_memory_prompt(
            query=query,
            session_id={"eq": "test-session"},
            namespace={"eq": "test-namespace"},
            topics={"any": ["preferences", "colors"]},
            limit=5,
        )

        # Convert raw dict response to mock object for testing
        response = MockMemoryPromptResponse(response)

        # Verify the response
        assert len(response.messages) == 2
        assert isinstance(response.messages[0].content, MockContent)
        assert "favorite color" in response.messages[0].content.text
        assert isinstance(response.messages[1].content, MockContent)
        assert response.messages[1].content.text == query

        # Test with filter objects
        mock_post.reset_mock()
        mock_post.return_value = mock_response

        response = await memory_test_client.hydrate_memory_prompt(
            query=query,
            session_id=SessionId(eq="test-session"),
            namespace=Namespace(eq="test-namespace"),
            topics=Topics(any=["preferences"]),
            limit=5,
        )

        # Convert raw dict response to mock object for testing
        response = MockMemoryPromptResponse(response)

        # Response should be the same because it's mocked
        assert len(response.messages) == 2

        # Test with no filters (just query)
        mock_post.reset_mock()
        mock_post.return_value = mock_response

        response = await memory_test_client.hydrate_memory_prompt(
            query=query,
        )

        # Convert raw dict response to mock object for testing
        response = MockMemoryPromptResponse(response)

        # Response should still be the same (mocked)
        assert len(response.messages) == 2


@pytest.mark.asyncio
async def test_memory_prompt_integration(memory_test_client: MemoryAPIClient):
    """Test the memory_prompt method with both session and long-term search"""
    session_id = "test-client-session"
    query = "What was my favorite color?"

    # Create expected response with both session and LTM content
    expected_messages = [
        SystemMessage(
            content=TextContent(
                type="text",
                text="## A summary of the conversation so far\nPrevious conversation about website design preferences.",
            ),
        ),
        base.UserMessage(
            content=TextContent(
                type="text", text="What is a good color for a website?"
            ),
        ),
        base.AssistantMessage(
            content=TextContent(
                type="text",
                text="It depends on the website's purpose. Blue is often used for professional sites.",
            ),
        ),
        SystemMessage(
            content=TextContent(
                type="text",
                text="## Long term memories related to the user's query\n - The user's favorite color is blue",
            ),
        ),
        base.UserMessage(
            content=TextContent(type="text", text=query),
        ),
    ]

    # Create expected response payload
    expected_response = MemoryPromptResponse(messages=expected_messages)

    # Mock the HTTP client's post method directly
    with patch.object(memory_test_client._client, "post") as mock_post:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock(return_value=None)
        mock_response.json = MagicMock(return_value=expected_response.model_dump())
        mock_post.return_value = mock_response

        # Let the client method run with our mocked response
        response = await memory_test_client.memory_prompt(
            query=query,
            session_id=session_id,
            namespace="test-namespace",
        )

        # Convert raw dict response to mock object for testing
        response = MockMemoryPromptResponse(response)

        # Check that both session memory and LTM are in the response
        assert len(response.messages) == 5

        # Extract text from contents
        message_texts = []
        for m in response.messages:
            if isinstance(m.content, MockContent):
                message_texts.append(m.content.text)

        # The messages should include at least one from the session
        assert any("website" in text for text in message_texts)
        # And at least one from LTM
        assert any("favorite color is blue" in text for text in message_texts)
        # And the query itself
        assert query in message_texts[-1]


@pytest.mark.asyncio
async def test_search_long_term_memory_with_optimize_query_default_true(
    memory_test_client: MemoryAPIClient,
):
    """Test that client search_long_term_memory uses optimize_query=True by default."""
    with patch(
        "agent_memory_server.long_term_memory.search_long_term_memories"
    ) as mock_search:
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="test-1",
                    text="User preferences about UI",
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    dist=0.1,
                )
            ],
            next_offset=None,
        )

        # Call search without optimize_query parameter (should default to True)
        results = await memory_test_client.search_long_term_memory(
            text="tell me about my preferences"
        )

        # Verify search was called with optimize_query=True (default)
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("optimize_query") is True

        # Verify results
        assert results.total == 1
        assert len(results.memories) == 1


@pytest.mark.asyncio
async def test_search_long_term_memory_with_optimize_query_false_explicit(
    memory_test_client: MemoryAPIClient,
):
    """Test that client search_long_term_memory can use optimize_query=False when explicitly set."""
    with patch(
        "agent_memory_server.long_term_memory.search_long_term_memories"
    ) as mock_search:
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="test-1",
                    text="User preferences about UI",
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    dist=0.1,
                )
            ],
            next_offset=None,
        )

        # Call search with explicit optimize_query=False
        await memory_test_client.search_long_term_memory(
            text="tell me about my preferences", optimize_query=False
        )

        # Verify search was called with optimize_query=False
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("optimize_query") is False


@pytest.mark.asyncio
async def test_search_memory_tool_with_optimize_query_false_default(
    memory_test_client: MemoryAPIClient,
):
    """Test that client search_memory_tool uses optimize_query=False by default (for LLM tool use)."""
    with patch(
        "agent_memory_server.long_term_memory.search_long_term_memories"
    ) as mock_search:
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="test-1",
                    text="User preferences about UI",
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    dist=0.1,
                )
            ],
            next_offset=None,
        )

        # Call search_memory_tool without optimize_query parameter (should default to False for LLM tools)
        results = await memory_test_client.search_memory_tool(
            query="tell me about my preferences"
        )

        # Verify search was called with optimize_query=False (default for LLM tools)
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("optimize_query") is False

        # Verify results format is suitable for LLM consumption
        assert "memories" in results
        assert "summary" in results


@pytest.mark.asyncio
async def test_search_memory_tool_with_optimize_query_true_explicit(
    memory_test_client: MemoryAPIClient,
):
    """Test that client search_memory_tool can use optimize_query=True when explicitly set."""
    with patch(
        "agent_memory_server.long_term_memory.search_long_term_memories"
    ) as mock_search:
        mock_search.return_value = MemoryRecordResultsResponse(
            total=1,
            memories=[
                MemoryRecordResult(
                    id="test-1",
                    text="User preferences about UI",
                    memory_type=MemoryTypeEnum.SEMANTIC,
                    dist=0.1,
                )
            ],
            next_offset=None,
        )

        # Call search_memory_tool with explicit optimize_query=True
        await memory_test_client.search_memory_tool(
            query="tell me about my preferences", optimize_query=True
        )

        # Verify search was called with optimize_query=True
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("optimize_query") is True


@pytest.mark.asyncio
async def test_memory_prompt_with_optimize_query_default_true(
    memory_test_client: MemoryAPIClient,
):
    """Test that client memory_prompt uses optimize_query=True by default."""
    with patch(
        "agent_memory_server.long_term_memory.search_long_term_memories"
    ) as mock_search:
        mock_search.return_value = MemoryRecordResultsResponse(
            total=0, memories=[], next_offset=None
        )

        # Call memory_prompt without optimize_query parameter (should default to True)
        result = await memory_test_client.memory_prompt(
            query="what are my preferences?", long_term_search={"text": "preferences"}
        )

        # Verify search was called with optimize_query=True (default)
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("optimize_query") is True
        assert result is not None


@pytest.mark.asyncio
async def test_memory_prompt_with_optimize_query_false_explicit(
    memory_test_client: MemoryAPIClient,
):
    """Test that client memory_prompt can use optimize_query=False when explicitly set."""
    with patch(
        "agent_memory_server.long_term_memory.search_long_term_memories"
    ) as mock_search:
        mock_search.return_value = MemoryRecordResultsResponse(
            total=0, memories=[], next_offset=None
        )

        # Call memory_prompt with explicit optimize_query=False
        result = await memory_test_client.memory_prompt(
            query="what are my preferences?",
            long_term_search={"text": "preferences"},
            optimize_query=False,
        )

        # Verify search was called with optimize_query=False
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("optimize_query") is False
        assert result is not None
