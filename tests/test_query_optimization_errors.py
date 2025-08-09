"""
Test error handling and edge cases for query optimization feature.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_memory_server.llms import optimize_query_for_vector_search
from agent_memory_server.long_term_memory import search_long_term_memories
from agent_memory_server.models import MemoryRecordResults


@pytest.mark.asyncio
class TestQueryOptimizationErrorHandling:
    """Test error handling scenarios for query optimization."""

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_with_network_timeout(self, mock_get_client):
        """Test graceful fallback when model API times out."""
        # Simulate network timeout
        mock_client = AsyncMock()
        mock_client.create_chat_completion.side_effect = TimeoutError(
            "Request timed out"
        )
        mock_get_client.return_value = mock_client

        original_query = "Can you tell me about my settings?"
        result = await optimize_query_for_vector_search(original_query)

        # Should fall back to original query
        assert result == original_query
        mock_get_client.assert_called_once()

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_with_invalid_api_key(self, mock_get_client):
        """Test fallback when API key is invalid."""
        # Simulate authentication error
        mock_get_client.side_effect = Exception("Invalid API key")

        original_query = "What are my preferences?"
        result = await optimize_query_for_vector_search(original_query)

        # Should fall back to original query
        assert result == original_query

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_with_malformed_response(self, mock_get_client):
        """Test handling of malformed model responses."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        # Malformed response - no choices attribute
        del mock_response.choices
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        original_query = "Find my user settings"
        result = await optimize_query_for_vector_search(original_query)

        # Should fall back to original query
        assert result == original_query

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_with_none_response(self, mock_get_client):
        """Test handling when model returns None."""
        mock_client = AsyncMock()
        mock_client.create_chat_completion.return_value = None
        mock_get_client.return_value = mock_client

        original_query = "Show my preferences"
        result = await optimize_query_for_vector_search(original_query)

        # Should fall back to original query
        assert result == original_query

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_with_unicode_query(self, mock_get_client):
        """Test optimization with unicode and special characters."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "préférences utilisateur émojis 🎉"
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        unicode_query = "Mes préférences avec émojis 🎉 et caractères spéciaux"
        result = await optimize_query_for_vector_search(unicode_query)

        assert result == "préférences utilisateur émojis 🎉"
        mock_get_client.assert_called_once()

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_with_very_long_query(self, mock_get_client):
        """Test optimization with extremely long queries."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "long query optimized"
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Create a very long query (10,000 characters)
        long_query = "Tell me about " + "preferences " * 1000 + "settings"
        result = await optimize_query_for_vector_search(long_query)

        assert result == "long query optimized"
        mock_get_client.assert_called_once()

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_preserves_query_intent(self, mock_get_client):
        """Test that optimization preserves the core intent of queries."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Mock an optimization that maintains intent
        mock_response.choices[0].message.content = "user interface dark mode settings"
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        original_query = (
            "Can you please tell me about my dark mode settings for the UI?"
        )
        result = await optimize_query_for_vector_search(original_query)

        assert result == "user interface dark mode settings"
        # Verify the prompt includes the original query
        call_args = mock_client.create_chat_completion.call_args
        prompt = call_args[1]["prompt"]
        assert original_query in prompt

    @patch("agent_memory_server.long_term_memory.get_vectorstore_adapter")
    @patch("agent_memory_server.long_term_memory.optimize_query_for_vector_search")
    async def test_search_continues_when_optimization_fails(
        self, mock_optimize, mock_get_adapter
    ):
        """Test that search continues even if optimization completely fails."""
        # Mock optimization to return original query (simulating internal error handling)
        mock_optimize.return_value = (
            "test query"  # The function handles errors internally
        )

        # Mock the vectorstore adapter
        mock_adapter = AsyncMock()
        mock_adapter.search_memories.return_value = MemoryRecordResults(
            total=0, memories=[]
        )
        mock_get_adapter.return_value = mock_adapter

        # This should not raise an exception
        await search_long_term_memories(
            text="test query", optimize_query=True, limit=10
        )

        # Verify optimization was attempted
        mock_optimize.assert_called_once()
        # Verify search still proceeded
        mock_adapter.search_memories.assert_called_once()

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_handles_special_characters_in_response(
        self, mock_get_client
    ):
        """Test handling of special characters and formatting in model responses."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Response with various formatting that should be cleaned
        mock_response.choices[
            0
        ].message.content = "\n\n  **user preferences settings**  \n\n"
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = await optimize_query_for_vector_search("What are my settings?")

        # Should strip whitespace but preserve the content
        assert result == "**user preferences settings**"

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_with_model_rate_limit(self, mock_get_client):
        """Test fallback when model API is rate limited."""
        # Simulate rate limit error
        mock_get_client.side_effect = Exception("Rate limit exceeded")

        original_query = "Find my account settings"
        result = await optimize_query_for_vector_search(original_query)

        # Should fall back to original query
        assert result == original_query

    @patch("agent_memory_server.llms.settings")
    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimization_with_invalid_model_name(
        self, mock_get_client, mock_settings
    ):
        """Test handling of invalid/unavailable model names."""
        # Set an invalid model name
        mock_settings.fast_model = "invalid-model-name"
        mock_get_client.side_effect = Exception("Model not found")

        original_query = "Show user preferences"
        result = await optimize_query_for_vector_search(original_query)

        # Should fall back to original query
        assert result == original_query
        mock_get_client.assert_called_once_with("invalid-model-name")
