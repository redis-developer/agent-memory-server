import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agent_memory_server.llms import (
    ModelProvider,
    OpenAIClientWrapper,
    get_model_client,
    get_model_config,
    optimize_query_for_vector_search,
)


@pytest.mark.asyncio
class TestOpenAIClientWrapper:
    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-key",
        },
    )
    @patch("agent_memory_server.llms.AsyncOpenAI")
    async def test_init_regular_openai(self, mock_openai):
        """Test initializing with regular OpenAI"""
        # Set up the mock to return an AsyncMock
        mock_openai.return_value = AsyncMock()

        OpenAIClientWrapper()

        # Verify the client was created
        assert mock_openai.called

    @patch.object(OpenAIClientWrapper, "__init__", return_value=None)
    async def test_create_embedding(self, mock_init):
        """Test creating embeddings"""
        # Create a client with mocked init
        client = OpenAIClientWrapper()

        # Mock the embedding client and response
        mock_response = AsyncMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]

        client.embedding_client = AsyncMock()
        client.embedding_client.embeddings.create = AsyncMock(
            return_value=mock_response
        )

        # Test creating embeddings
        query_vec = ["Hello, world!", "How are you?"]
        embeddings = await client.create_embedding(query_vec)

        # Verify embeddings were created correctly
        assert len(embeddings) == 2
        # Convert NumPy array to list or use np.array_equal for comparison
        assert np.array_equal(
            embeddings[0], np.array([0.1, 0.2, 0.3], dtype=np.float32)
        )
        assert np.array_equal(
            embeddings[1], np.array([0.4, 0.5, 0.6], dtype=np.float32)
        )

        # Verify the client was called with correct parameters
        client.embedding_client.embeddings.create.assert_called_with(
            model="text-embedding-ada-002", input=query_vec
        )

    @patch.object(OpenAIClientWrapper, "__init__", return_value=None)
    async def test_create_chat_completion(self, mock_init):
        """Test creating chat completions"""
        # Create a client with mocked init
        client = OpenAIClientWrapper()

        # Mock the completion client and response
        # Create a response structure that matches our new ChatResponse format
        mock_response = AsyncMock()
        mock_response.choices = [{"message": {"content": "Test response"}}]
        mock_response.usage = {"total_tokens": 100}

        client.completion_client = AsyncMock()
        client.completion_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Test creating chat completion
        model = "gpt-3.5-turbo"
        prompt = "Hello, world!"
        response = await client.create_chat_completion(model, prompt)

        # Verify the response contains the expected structure
        assert response.choices[0]["message"]["content"] == "Test response"
        assert response.total_tokens == 100

        # Verify the client was called with correct parameters
        client.completion_client.chat.completions.create.assert_called_with(
            model=model, messages=[{"role": "user", "content": prompt}]
        )


@pytest.mark.parametrize(
    ("model_name", "expected_provider", "expected_max_tokens"),
    [
        ("gpt-4o", "openai", 128000),
        ("claude-3-sonnet-20240229", "anthropic", 200000),
        ("nonexistent-model", "openai", 128000),  # Should default to GPT-4o-mini
    ],
)
def test_get_model_config(model_name, expected_provider, expected_max_tokens):
    """Test the get_model_config function"""
    # Get the model config
    config = get_model_config(model_name)

    # Check the provider
    if expected_provider == "openai":
        assert config.provider == ModelProvider.OPENAI
    else:
        assert config.provider == ModelProvider.ANTHROPIC

    # Check the max tokens
    assert config.max_tokens == expected_max_tokens


@pytest.mark.asyncio
async def test_get_model_client():
    """Test the get_model_client function"""
    # Test with OpenAI model
    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        patch("agent_memory_server.llms.OpenAIClientWrapper") as mock_openai,
    ):
        mock_openai.return_value = "openai-client"
        client = await get_model_client("gpt-4")
        assert client == "openai-client"

    # Test with Anthropic model
    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
        patch("agent_memory_server.llms.AnthropicClientWrapper") as mock_anthropic,
    ):
        mock_anthropic.return_value = "anthropic-client"
        client = await get_model_client("claude-3-sonnet-20240229")
        assert client == "anthropic-client"


@pytest.mark.asyncio
class TestQueryOptimization:
    """Test query optimization functionality."""

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_success(self, mock_get_client):
        """Test successful query optimization."""
        # Mock the model client and response
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = "user interface preferences dark mode"
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = await optimize_query_for_vector_search(
            "Can you tell me about my UI preferences for dark mode?"
        )

        assert result == "user interface preferences dark mode"
        mock_get_client.assert_called_once()
        mock_client.create_chat_completion.assert_called_once()

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_with_custom_model(self, mock_get_client):
        """Test query optimization with custom model."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "optimized query"
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = await optimize_query_for_vector_search(
            "original query", model_name="custom-model"
        )

        assert result == "optimized query"
        mock_client.create_chat_completion.assert_called_once()
        # Verify the model name was passed to create_chat_completion
        call_kwargs = mock_client.create_chat_completion.call_args[1]
        assert call_kwargs["model"] == "custom-model"

    @patch("agent_memory_server.llms.settings")
    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_uses_fast_model_default(
        self, mock_get_client, mock_settings
    ):
        """Test that optimization uses fast_model by default."""
        mock_settings.fast_model = "gpt-4o-mini"
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "optimized"
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        await optimize_query_for_vector_search("test query")

        mock_get_client.assert_called_once_with("gpt-4o-mini")

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_empty_input(self, mock_get_client):
        """Test optimization with empty or None input."""
        # Test empty string
        result = await optimize_query_for_vector_search("")
        assert result == ""
        mock_get_client.assert_not_called()

        # Test None
        result = await optimize_query_for_vector_search(None)
        assert result is None
        mock_get_client.assert_not_called()

        # Test whitespace only
        result = await optimize_query_for_vector_search("   ")
        assert result == "   "
        mock_get_client.assert_not_called()

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_client_error_fallback(self, mock_get_client):
        """Test fallback to original query when client fails."""
        mock_get_client.side_effect = Exception("Model client error")

        original_query = "What are my preferences?"
        result = await optimize_query_for_vector_search(original_query)

        assert result == original_query
        mock_get_client.assert_called_once()

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_empty_response_fallback(self, mock_get_client):
        """Test fallback when model returns empty response."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""  # Empty response
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        original_query = "What are my preferences?"
        result = await optimize_query_for_vector_search(original_query)

        assert result == original_query

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_short_response_fallback(self, mock_get_client):
        """Test fallback when model returns very short response."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "a"  # Too short
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        original_query = "What are my preferences?"
        result = await optimize_query_for_vector_search(original_query)

        assert result == original_query

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_no_choices_fallback(self, mock_get_client):
        """Test fallback when model response has no choices."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = []  # No choices
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        original_query = "What are my preferences?"
        result = await optimize_query_for_vector_search(original_query)

        assert result == original_query

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_different_response_formats(self, mock_get_client):
        """Test handling different response formats (text vs message)."""
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        # Test with 'text' attribute
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        del mock_response.choices[0].message  # Remove message attribute
        mock_response.choices[0].text = "optimized via text"
        mock_client.create_chat_completion.return_value = mock_response

        result = await optimize_query_for_vector_search("test query")
        assert result == "optimized via text"

    @patch("agent_memory_server.llms.get_model_client")
    async def test_optimize_query_strips_whitespace(self, mock_get_client):
        """Test that optimization strips whitespace from response."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  optimized query  \n"
        mock_client.create_chat_completion.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = await optimize_query_for_vector_search("test query")
        assert result == "optimized query"

    async def test_optimize_query_prompt_format(self):
        """Test that the optimization prompt is correctly formatted."""
        with patch("agent_memory_server.llms.get_model_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "optimized"
            mock_client.create_chat_completion.return_value = mock_response
            mock_get_client.return_value = mock_client

            test_query = "Can you tell me about user preferences?"
            await optimize_query_for_vector_search(test_query)

            # Check that the prompt contains our test query
            call_args = mock_client.create_chat_completion.call_args
            prompt = call_args[1]["prompt"]
            assert test_query in prompt
            assert "semantic search" in prompt
            assert "Guidelines:" in prompt
            assert "Optimized query:" in prompt
