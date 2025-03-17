import logging
import os
from enum import Enum
from typing import Any

import anthropic
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


# Setup logging
logger = logging.getLogger(__name__)


class MemoryMessage(BaseModel):
    """A message in the memory system"""

    role: str
    content: str


class MemoryMessagesAndContext(BaseModel):
    """Request payload for adding messages to memory"""

    messages: list[MemoryMessage]
    context: str | None = None


class MemoryResponse(BaseModel):
    """Response containing messages and context"""

    messages: list[MemoryMessage]
    context: str | None = None
    tokens: int | None = None


class SearchPayload(BaseModel):
    """Payload for semantic search"""

    text: str


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""

    now: int


class AckResponse(BaseModel):
    """Generic acknowledgement response"""

    status: str


class RedisearchResult(BaseModel):
    """Result from a redisearch query"""

    role: str
    content: str
    dist: float


class SearchResults(BaseModel):
    """Results from a redisearch query"""

    docs: list[RedisearchResult]
    total: int


class NamespaceQuery(BaseModel):
    """Query parameters for namespace"""

    namespace: str | None = None


class GetSessionsQuery(BaseModel):
    """Query parameters for getting sessions"""

    page: int = Field(default=1)
    size: int = Field(default=20)
    namespace: str | None = None


class ModelProvider(str, Enum):
    """Type of model provider"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelConfig(BaseModel):
    """Configuration for a model"""

    provider: ModelProvider
    name: str
    max_tokens: int
    embedding_dimensions: int = 1536  # Default for OpenAI ada-002


# Model configurations
MODEL_CONFIGS = {
    # OpenAI Models
    "gpt-3.5-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-3.5-turbo",
        max_tokens=4096,
        embedding_dimensions=1536,
    ),
    "gpt-3.5-turbo-16k": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-3.5-turbo-16k",
        max_tokens=16384,
        embedding_dimensions=1536,
    ),
    "gpt-4": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4",
        max_tokens=8192,
        embedding_dimensions=1536,
    ),
    "gpt-4-32k": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4-32k",
        max_tokens=32768,
        embedding_dimensions=1536,
    ),
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4o",
        max_tokens=128000,
        embedding_dimensions=1536,
    ),
    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="gpt-4o-mini",
        max_tokens=128000,
        embedding_dimensions=1536,
    ),
    # Newer reasoning models
    "o1": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="o1",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "o1-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="o1-mini",
        max_tokens=128000,
        embedding_dimensions=1536,
    ),
    "o3-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="o3-mini",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # Embedding models
    "text-embedding-ada-002": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="text-embedding-ada-002",
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
    "text-embedding-3-small": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="text-embedding-3-small",
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
    "text-embedding-3-large": ModelConfig(
        provider=ModelProvider.OPENAI,
        name="text-embedding-3-large",
        max_tokens=8191,
        embedding_dimensions=3072,
    ),
    # Anthropic Models
    "claude-3-opus-20240229": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-opus-20240229",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-sonnet-20240229": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-sonnet-20240229",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-haiku-20240307": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-haiku-20240307",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-sonnet-20240620": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-sonnet-20240620",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # Latest Anthropic Models
    "claude-3-7-sonnet-20250219": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-7-sonnet-20250219",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-sonnet-20241022": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-sonnet-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-haiku-20241022": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-haiku-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    # Convenience aliases
    "claude-3-7-sonnet-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-7-sonnet-20250219",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-sonnet-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-sonnet-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-5-haiku-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-5-haiku-20241022",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
    "claude-3-opus-latest": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        name="claude-3-opus-20240229",
        max_tokens=200000,
        embedding_dimensions=1536,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a model"""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # Default to GPT-4o-mini if model not found
    logger.warning(f"Model {model_name} not found in configuration, using gpt-4o-mini")
    return MODEL_CONFIGS["gpt-4o-mini"]


class ChatResponse:
    """Unified wrapper for chat responses from different providers"""

    def __init__(self, choices: list[Any], usage: dict[str, int]):
        self.choices = choices or []
        self.usage = usage or {"total_tokens": 0}

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class AnthropicClientWrapper:
    """Wrapper for Anthropic client"""

    def __init__(self, api_key: str | None = None):
        """Initialize the Anthropic client"""
        anthropic_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required")

        self.client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

    async def create_chat_completion(self, model: str, prompt: str) -> ChatResponse:
        """Create a chat completion using the Anthropic API"""
        try:
            response = await self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            # Convert to a unified format - safely extract content
            content = ""
            if (
                hasattr(response, "content")
                and response.content
                and len(response.content) > 0
                and hasattr(response.content[0], "text")
            ):
                content = response.content[0].text

            choices = [{"message": {"content": content}}]

            # Handle both object and dictionary usage formats for testing
            input_tokens = output_tokens = 0
            if hasattr(response, "usage"):
                if isinstance(response.usage, dict):
                    input_tokens = response.usage.get("input_tokens", 0)
                    output_tokens = response.usage.get("output_tokens", 0)
                else:
                    input_tokens = getattr(response.usage, "input_tokens", 0)
                    output_tokens = getattr(response.usage, "output_tokens", 0)

            usage = {"total_tokens": input_tokens + output_tokens}

            return ChatResponse(choices=choices, usage=usage)
        except Exception as e:
            logger.error(f"Error creating chat completion with Anthropic: {e}")
            raise

    async def create_embedding(self, query_vec: list[str]) -> np.ndarray:
        """
        Create embeddings for the given texts
        Note: Anthropic doesn't offer an embedding API, so we'll use OpenAI's
        embeddings or raise an error if needed
        """
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. Please use OpenAI for embeddings."
        )


class OpenAIClientWrapper:
    """Wrapper for OpenAI client"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the OpenAI client based on environment variables"""

        # Regular OpenAI setup
        openai_api_base = base_url or os.environ.get("OPENAI_API_BASE")
        openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OpenAI API key is required")

        if openai_api_base:
            self.completion_client = AsyncOpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            self.embedding_client = AsyncOpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            self.completion_client = AsyncOpenAI(api_key=openai_api_key)
            self.embedding_client = AsyncOpenAI(api_key=openai_api_key)

    async def create_chat_completion(
        self,
        model: str,
        progressive_prompt: str,
    ) -> ChatResponse:
        """Create a chat completion using the OpenAI API"""
        try:
            response = await self.completion_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": progressive_prompt}],
            )

            # Convert to unified format
            # Handle both object and dictionary usage formats for testing
            total_tokens = 0
            if hasattr(response, "usage"):
                if isinstance(response.usage, dict):
                    total_tokens = response.usage.get("total_tokens", 0)
                else:
                    total_tokens = getattr(response.usage, "total_tokens", 0)

            return ChatResponse(
                choices=response.choices,
                usage={"total_tokens": total_tokens},
            )
        except Exception as e:
            logger.error(f"Error creating chat completion with OpenAI: {e}")
            raise

    async def create_embedding(self, query_vec: list[str]) -> np.ndarray:
        """Create embeddings for the given texts"""
        try:
            embeddings = []
            embedding_model = "text-embedding-ada-002"

            # Process in batches of 20 to avoid rate limits
            batch_size = 20
            for i in range(0, len(query_vec), batch_size):
                batch = query_vec[i : i + batch_size]
                response = await self.embedding_client.embeddings.create(
                    model=embedding_model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise


class ModelClientFactory:
    """Factory for creating model clients"""

    @staticmethod
    async def get_client(
        model_name: str,
    ) -> OpenAIClientWrapper | AnthropicClientWrapper:
        """Get the appropriate client for a model"""
        model_config = get_model_config(model_name)

        if model_config.provider == ModelProvider.OPENAI:
            return OpenAIClientWrapper(api_key=os.environ.get("OPENAI_API_KEY"))
        if model_config.provider == ModelProvider.ANTHROPIC:
            return AnthropicClientWrapper(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        raise ValueError(f"Unsupported model provider: {model_config.provider}")
