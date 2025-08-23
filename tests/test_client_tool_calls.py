"""
Test file for the Memory API Client tool call functionality.

Tests for multi-provider tool call parsing, resolution, and schema generation.
"""

import json
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
from agent_memory_client import MemoryAPIClient, MemoryClientConfig
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agent_memory_server.api import router as memory_router
from agent_memory_server.healthcheck import router as health_router


@pytest.fixture
def memory_app() -> FastAPI:
    """Create a test FastAPI app with memory routers for testing the client."""
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(memory_router)
    return app


@pytest.fixture
async def tool_call_test_client(
    memory_app: FastAPI,
) -> AsyncGenerator[MemoryAPIClient, None]:
    """Create a memory client that uses the test FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=memory_app),
        base_url="http://test",
    ) as http_client:
        config = MemoryClientConfig(
            base_url="http://test", default_namespace="test-namespace"
        )
        client = MemoryAPIClient(config)
        client._client = http_client
        yield client


class TestToolCallParsing:
    """Tests for tool call parsing across different provider formats."""

    def test_parse_openai_function_call(self):
        """Test parsing OpenAI legacy function call format."""
        function_call = {
            "name": "search_memory",
            "arguments": json.dumps({"query": "user preferences", "max_results": 5}),
        }

        result = MemoryAPIClient.parse_openai_function_call(function_call)

        assert result["id"] is None
        assert result["name"] == "search_memory"
        assert result["arguments"]["query"] == "user preferences"
        assert result["arguments"]["max_results"] == 5
        assert result["provider"] == "openai"

    def test_parse_openai_function_call_invalid_json(self):
        """Test parsing OpenAI function call with invalid JSON arguments."""
        function_call = {"name": "search_memory", "arguments": "invalid json"}

        result = MemoryAPIClient.parse_openai_function_call(function_call)

        assert result["name"] == "search_memory"
        assert result["arguments"] == {}
        assert result["provider"] == "openai"

    def test_parse_openai_tool_call(self):
        """Test parsing OpenAI current tool call format."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "add_memory_to_working_memory",
                "arguments": json.dumps(
                    {"text": "User likes pizza", "memory_type": "semantic"}
                ),
            },
        }

        result = MemoryAPIClient.parse_openai_tool_call(tool_call)

        assert result["id"] == "call_123"
        assert result["name"] == "add_memory_to_working_memory"
        assert result["arguments"]["text"] == "User likes pizza"
        assert result["arguments"]["memory_type"] == "semantic"
        assert result["provider"] == "openai"

    def test_parse_openai_tool_call_dict_arguments(self):
        """Test parsing OpenAI tool call with dict arguments (not JSON string)."""
        tool_call = {
            "id": "call_456",
            "type": "function",
            "function": {
                "name": "get_working_memory",
                "arguments": {"session_id": "test"},
            },
        }

        result = MemoryAPIClient.parse_openai_tool_call(tool_call)

        assert result["id"] == "call_456"
        assert result["name"] == "get_working_memory"
        assert result["arguments"]["session_id"] == "test"
        assert result["provider"] == "openai"

    def test_parse_anthropic_tool_use(self):
        """Test parsing Anthropic tool use format."""
        tool_use = {
            "type": "tool_use",
            "id": "tool_789",
            "name": "update_working_memory_data",
            "input": {
                "data": {"preferences": {"theme": "dark"}},
                "merge_strategy": "merge",
            },
        }

        result = MemoryAPIClient.parse_anthropic_tool_use(tool_use)

        assert result["id"] == "tool_789"
        assert result["name"] == "update_working_memory_data"
        assert result["arguments"]["data"]["preferences"]["theme"] == "dark"
        assert result["arguments"]["merge_strategy"] == "merge"
        assert result["provider"] == "anthropic"

    def test_parse_tool_call_auto_detect_formats(self):
        """Test automatic detection of different formats."""
        # Anthropic format
        anthropic_call = {
            "type": "tool_use",
            "id": "tool_auto",
            "name": "search_memory",
            "input": {"query": "test"},
        }
        result = MemoryAPIClient.parse_tool_call(anthropic_call)
        assert result["provider"] == "anthropic"

        # OpenAI current format
        openai_current = {
            "id": "call_auto",
            "type": "function",
            "function": {
                "name": "search_memory",
                "arguments": json.dumps({"query": "test"}),
            },
        }
        result = MemoryAPIClient.parse_tool_call(openai_current)
        assert result["provider"] == "openai"

        # OpenAI legacy format
        openai_legacy = {
            "name": "search_memory",
            "arguments": json.dumps({"query": "test"}),
        }
        result = MemoryAPIClient.parse_tool_call(openai_legacy)
        assert result["provider"] == "openai"


class TestToolCallResolution:
    """Tests for tool call resolution functionality."""

    @pytest.mark.asyncio
    async def test_resolve_function_call_search_memory(self, tool_call_test_client):
        """Test resolving search_memory function call."""
        mock_result = {
            "memories": [{"text": "test memory", "memory_type": "semantic"}],
            "total_found": 1,
            "query": "test",
            "summary": "Found 1 relevant memories for: test",
        }

        with patch.object(
            tool_call_test_client, "search_memory_tool", return_value=mock_result
        ):
            result = await tool_call_test_client.resolve_function_call(
                function_name="search_memory",
                function_arguments={"query": "test", "max_results": 3},
                session_id="test_session",
            )

            assert result["success"] is True
            assert result["function_name"] == "search_memory"
            assert result["result"] == mock_result
            assert result["error"] is None
            assert "Found 1 relevant memories" in result["formatted_response"]

    @pytest.mark.asyncio
    async def test_resolve_function_call_get_working_memory(
        self, tool_call_test_client
    ):
        """Test resolving get_working_memory function call."""
        mock_result = {
            "session_id": "test_session",
            "message_count": 5,
            "memory_count": 2,
            "summary": "Session has 5 messages, 2 stored memories, and 0 data entries",
        }

        with patch.object(
            tool_call_test_client, "get_working_memory_tool", return_value=mock_result
        ):
            result = await tool_call_test_client.resolve_function_call(
                function_name="get_working_memory",
                function_arguments={},
                session_id="test_session",
            )

            assert result["success"] is True
            assert result["function_name"] == "get_working_memory"
            assert result["result"] == mock_result
            assert "Session has 5 messages" in result["formatted_response"]

    @pytest.mark.asyncio
    async def test_resolve_function_call_add_memory(self, tool_call_test_client):
        """Test resolving add_memory_to_working_memory function call."""
        mock_result = {
            "success": True,
            "memory_type": "semantic",
            "text_preview": "User prefers dark mode...",
            "summary": "Successfully stored semantic memory: User prefers dark mode...",
        }

        with patch.object(
            tool_call_test_client, "add_memory_tool", return_value=mock_result
        ):
            result = await tool_call_test_client.resolve_function_call(
                function_name="add_memory_to_working_memory",
                function_arguments={
                    "text": "User prefers dark mode",
                    "memory_type": "semantic",
                    "topics": ["preferences"],
                },
                session_id="test_session",
            )

            assert result["success"] is True
            assert result["function_name"] == "add_memory_to_working_memory"
            assert result["result"] == mock_result
            assert "Successfully stored semantic memory" in result["formatted_response"]

    @pytest.mark.asyncio
    async def test_resolve_function_call_update_data(self, tool_call_test_client):
        """Test resolving update_working_memory_data function call."""
        mock_result = {
            "success": True,
            "updated_keys": ["user_settings"],
            "merge_strategy": "merge",
            "summary": "Successfully updated 1 data entries using merge strategy",
        }

        with patch.object(
            tool_call_test_client, "update_memory_data_tool", return_value=mock_result
        ):
            result = await tool_call_test_client.resolve_function_call(
                function_name="update_working_memory_data",
                function_arguments={
                    "data": {"user_settings": {"theme": "dark"}},
                    "merge_strategy": "merge",
                },
                session_id="test_session",
            )

            assert result["success"] is True
            assert result["function_name"] == "update_working_memory_data"
            assert result["result"] == mock_result
            assert "Successfully updated 1 data entries" in result["formatted_response"]

    @pytest.mark.asyncio
    async def test_resolve_function_call_unknown_function(self, tool_call_test_client):
        """Test resolving unknown function call."""
        result = await tool_call_test_client.resolve_function_call(
            function_name="unknown_function",
            function_arguments={},
            session_id="test_session",
        )

        assert result["success"] is False
        assert result["function_name"] == "unknown_function"
        assert result["result"] is None
        assert "Unknown function: unknown_function" in result["error"]
        assert "don't know how to handle" in result["formatted_response"]

    @pytest.mark.asyncio
    async def test_resolve_function_call_invalid_json_arguments(
        self, tool_call_test_client
    ):
        """Test resolving function call with invalid JSON arguments."""
        result = await tool_call_test_client.resolve_function_call(
            function_name="search_memory",
            function_arguments="invalid json",
            session_id="test_session",
        )

        assert result["success"] is False
        assert result["function_name"] == "search_memory"
        assert result["result"] is None
        assert "JSON decode error" in result["error"]
        assert "error parsing the function arguments" in result["formatted_response"]

    @pytest.mark.asyncio
    async def test_resolve_function_call_missing_required_args(
        self, tool_call_test_client
    ):
        """Test resolving function call with missing required arguments."""
        result = await tool_call_test_client.resolve_function_call(
            function_name="search_memory",
            function_arguments={},  # Missing required 'query' parameter
            session_id="test_session",
        )

        assert result["success"] is False
        assert result["function_name"] == "search_memory"
        assert result["result"] is None
        assert "Query parameter is required" in result["error"]

    @pytest.mark.asyncio
    async def test_resolve_tool_call_openai_legacy(self, tool_call_test_client):
        """Test resolving OpenAI legacy format tool call."""
        tool_call = {
            "name": "search_memory",
            "arguments": json.dumps({"query": "test"}),
        }

        mock_result = {
            "memories": [],
            "total_found": 0,
            "query": "test",
            "summary": "Found 0 relevant memories for: test",
        }

        with patch.object(
            tool_call_test_client, "search_memory_tool", return_value=mock_result
        ):
            result = await tool_call_test_client.resolve_tool_call(
                tool_call=tool_call, session_id="test_session"
            )

            assert result["success"] is True
            assert result["function_name"] == "search_memory"
            assert "Found 0 relevant memories" in result["formatted_response"]

    @pytest.mark.asyncio
    async def test_resolve_tool_call_anthropic(self, tool_call_test_client):
        """Test resolving Anthropic format tool call."""
        tool_call = {
            "type": "tool_use",
            "id": "tool_123",
            "name": "get_working_memory",
            "input": {},
        }

        mock_result = {
            "session_id": "test_session",
            "summary": "Session has 0 messages, 0 stored memories, and 0 data entries",
        }

        with patch.object(
            tool_call_test_client, "get_working_memory_tool", return_value=mock_result
        ):
            result = await tool_call_test_client.resolve_tool_call(
                tool_call=tool_call, session_id="test_session"
            )

            assert result["success"] is True
            assert result["function_name"] == "get_working_memory"
            assert "Session has 0 messages" in result["formatted_response"]

    @pytest.mark.asyncio
    async def test_resolve_tool_calls_batch(self, tool_call_test_client):
        """Test resolving multiple tool calls in batch."""
        tool_calls = [
            {"name": "search_memory", "arguments": json.dumps({"query": "test1"})},
            {
                "type": "tool_use",
                "id": "tool_2",
                "name": "get_working_memory",
                "input": {},
            },
        ]

        mock_search_result = {"summary": "Search result"}
        mock_memory_result = {"summary": "Memory state"}

        with (
            patch.object(
                tool_call_test_client,
                "search_memory_tool",
                return_value=mock_search_result,
            ),
            patch.object(
                tool_call_test_client,
                "get_working_memory_tool",
                return_value=mock_memory_result,
            ),
        ):
            results = await tool_call_test_client.resolve_tool_calls(
                tool_calls=tool_calls, session_id="test_session"
            )

            assert len(results) == 2
            assert results[0]["success"] is True
            assert results[0]["function_name"] == "search_memory"
            assert results[1]["success"] is True
            assert results[1]["function_name"] == "get_working_memory"


class TestToolSchemaGeneration:
    """Tests for tool schema generation in different formats."""

    def test_get_memory_search_tool_schema(self):
        """Test getting memory search tool schema in OpenAI format."""
        schema = MemoryAPIClient.get_memory_search_tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_memory"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]

    def test_get_memory_search_tool_schema_anthropic(self):
        """Test getting memory search tool schema in Anthropic format."""
        schema = MemoryAPIClient.get_memory_search_tool_schema_anthropic()

        assert schema["name"] == "search_memory"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "query" in schema["input_schema"]["properties"]
        assert "query" in schema["input_schema"]["required"]

    def test_get_all_memory_tool_schemas(self):
        """Test getting all memory tool schemas in OpenAI format."""
        schemas = MemoryAPIClient.get_all_memory_tool_schemas()

        # We now expose additional tools (get_current_datetime, long-term tools)
        # So just assert that required core tools are present
        function_names = {schema["function"]["name"] for schema in schemas}
        required = {
            "search_memory",
            "get_working_memory",
            "add_memory_to_working_memory",
            "update_working_memory_data",
            "get_current_datetime",
        }
        assert required.issubset(function_names)

    def test_get_all_memory_tool_schemas_anthropic(self):
        """Test getting all memory tool schemas in Anthropic format."""
        schemas = MemoryAPIClient.get_all_memory_tool_schemas_anthropic()

        # We now expose additional tools; assert required core tools are present
        function_names = {schema["name"] for schema in schemas}
        required = {
            "search_memory",
            "get_working_memory",
            "add_memory_to_working_memory",
            "update_working_memory_data",
            "get_current_datetime",
        }
        assert required.issubset(function_names)

    def test_convert_openai_to_anthropic_schema(self):
        """Test converting OpenAI schema to Anthropic format."""
        openai_schema = {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "integer"},
                    },
                    "required": ["param1"],
                },
            },
        }

        anthropic_schema = MemoryAPIClient._convert_openai_to_anthropic_schema(
            openai_schema
        )

        assert anthropic_schema["name"] == "test_function"
        assert anthropic_schema["description"] == "Test function description"
        assert anthropic_schema["input_schema"]["type"] == "object"
        assert (
            anthropic_schema["input_schema"]["properties"]["param1"]["type"] == "string"
        )
        assert anthropic_schema["input_schema"]["required"] == ["param1"]


class TestToolCallErrorHandling:
    """Tests for tool call error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_resolve_tool_call_parse_error(self, tool_call_test_client):
        """Test resolving tool call that causes parse error."""
        # Malformed tool call that should cause parsing issues
        tool_call = {"malformed": "data"}

        result = await tool_call_test_client.resolve_tool_call(
            tool_call=tool_call, session_id="test_session"
        )

        # Should still work with generic parsing but with empty name
        assert result["success"] is False
        # The function name will be empty string, not "unknown"
        assert result["function_name"] == ""

    @pytest.mark.asyncio
    async def test_resolve_function_call_exception_handling(
        self, tool_call_test_client
    ):
        """Test that exceptions in tool methods are properly handled."""
        with patch.object(
            tool_call_test_client,
            "search_memory_tool",
            side_effect=Exception("Tool error"),
        ):
            result = await tool_call_test_client.resolve_function_call(
                function_name="search_memory",
                function_arguments={"query": "test"},
                session_id="test_session",
            )

            assert result["success"] is False
            assert result["function_name"] == "search_memory"
            assert result["result"] is None
            assert "Tool error" in result["error"]
            assert "error while executing search_memory" in result["formatted_response"]

    def test_parse_tool_call_edge_cases(self):
        """Test tool call parsing with edge cases."""
        # Empty tool call
        empty_call = {}
        result = MemoryAPIClient.parse_tool_call(empty_call)
        assert result["name"] == ""
        assert result["arguments"] == {}
        assert result["provider"] == "generic"

        # Tool call with None values
        none_call = {"name": None, "arguments": None}
        result = MemoryAPIClient.parse_tool_call(none_call)
        # The actual parsing returns None for name and arguments, not empty values
        assert result["name"] is None
        assert result["arguments"] is None

    @pytest.mark.asyncio
    async def test_resolve_function_calls_legacy_method(self, tool_call_test_client):
        """Test the legacy resolve_function_calls method still works."""
        function_calls = [
            {"name": "search_memory", "arguments": {"query": "test1"}},
            {"name": "get_working_memory", "arguments": {}},
        ]

        mock_search_result = {"summary": "Search result"}
        mock_memory_result = {"summary": "Memory state"}

        with (
            patch.object(
                tool_call_test_client,
                "search_memory_tool",
                return_value=mock_search_result,
            ),
            patch.object(
                tool_call_test_client,
                "get_working_memory_tool",
                return_value=mock_memory_result,
            ),
        ):
            results = await tool_call_test_client.resolve_function_calls(
                function_calls=function_calls, session_id="test_session"
            )

            assert len(results) == 2
            assert all(result["success"] for result in results)
