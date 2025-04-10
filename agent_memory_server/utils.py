import logging
from typing import Any

from redis.asyncio import ConnectionPool, Redis
from redis.commands.search.document import Document
from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema

# Replace previous class with a redisvl imported symbol
from agent_memory_server.config import settings
from agent_memory_server.llms import (
    AnthropicClientWrapper,
    ModelClientFactory,
    OpenAIClientWrapper,
)


logger = logging.getLogger(__name__)
_redis_pool = None
_openai_client = None
_anthropic_client = None
_model_clients = {}  # TODO: Use WeakRefDict
_index = None


def get_search_index(
    redis: Redis,
    index_name: str = settings.redisvl_index_name,
    vector_dimensions: int = settings.redisvl_vector_dimensions,
    distance_metric: str = settings.redisvl_distance_metric,
) -> AsyncSearchIndex:
    global _index
    if _index is None:
        schema = {
            "index": {
                "name": index_name,
                "prefix": f"{index_name}:",
                "key_separator": ":",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "text", "type": "text"},
                {"name": "id_", "type": "tag"},
                {"name": "session_id", "type": "tag"},
                {"name": "user_id", "type": "tag"},
                {"name": "namespace", "type": "tag"},
                {"name": "topics", "type": "tag"},
                {"name": "entities", "type": "tag"},
                {"name": "created_at", "type": "numeric"},
                {"name": "last_accessed", "type": "numeric"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "HNSW",
                        "dims": vector_dimensions,
                        "distance_metric": distance_metric,
                        "datatype": "float32",
                    },
                },
            ],
        }
        index_schema = IndexSchema.from_dict(schema)
        _index = AsyncSearchIndex(index_schema, redis_client=redis)
    return _index


def get_redis_conn(url: str | None = settings.redis_url, **kwargs) -> Redis:
    """Get Redis connection"""
    global _redis_pool
    if _redis_pool is None:
        if url:
            _redis_pool = ConnectionPool.from_url(url, **kwargs)
            return Redis(connection_pool=_redis_pool)
        _redis_pool = ConnectionPool(**kwargs)
    return Redis(connection_pool=_redis_pool)


async def ensure_redisearch_index(
    redis: Redis,
    index_name: str = settings.redisvl_index_name,
    vector_dimensions: int = settings.redisvl_vector_dimensions,
    distance_metric: str = settings.redisvl_distance_metric,
) -> None:
    """
    Ensure that the async search index exists, create it if it doesn't.
    Uses RedisVL's AsyncSearchIndex.

    Args:
        redis: A Redis client instance
        vector_dimensions: Dimensions of the embedding vectors
        distance_metric: Distance metric to use (default: COSINE)
        index_name: The name of the index
    """
    index = get_search_index(redis, index_name, vector_dimensions, distance_metric)
    if await index.exists():
        logger.info("Async search index already exists")
        return

    logger.info("Async search index doesn't exist, creating...")
    await index.create()

    logger.info(
        f"Created async search index with {vector_dimensions} dimensions and {distance_metric} metric"
    )


async def get_openai_client(**kwargs) -> OpenAIClientWrapper:
    """Get OpenAI client (legacy function, use get_model_client instead)"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClientWrapper(api_key=settings.openai_api_key, **kwargs)
    return _openai_client


async def get_anthropic_client(**kwargs) -> AnthropicClientWrapper:
    """Get Anthropic client (legacy function, use get_model_client instead)"""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AnthropicClientWrapper(
            api_key=settings.anthropic_api_key, **kwargs
        )
    return _anthropic_client


async def get_model_client(
    model_name: str,
) -> OpenAIClientWrapper | AnthropicClientWrapper:
    """Get the appropriate client for a model using the factory"""
    global _model_clients

    if model_name not in _model_clients:
        _model_clients[model_name] = await ModelClientFactory.get_client(model_name)

    return _model_clients[model_name]


class Keys:
    """Keys for Redis"""

    @staticmethod
    def context_key(session_id: str, namespace: str | None = None) -> str:
        return (
            f"context:{namespace}:{session_id}"
            if namespace
            else f"context:{session_id}"
        )

    @staticmethod
    def token_count_key(session_id: str, namespace: str | None = None) -> str:
        return (
            f"tokens:{namespace}:{session_id}" if namespace else f"tokens:{session_id}"
        )

    @staticmethod
    def messages_key(session_id: str, namespace: str | None = None) -> str:
        return (
            f"messages:{namespace}:{session_id}"
            if namespace
            else f"messages:{session_id}"
        )

    @staticmethod
    def sessions_key(namespace: str | None = None) -> str:
        return f"sessions:{namespace}" if namespace else "sessions"

    @staticmethod
    def memory_key(id: str, namespace: str | None = None) -> str:
        return f"memory:{namespace}:{id}" if namespace else f"memory:{id}"

    @staticmethod
    def metadata_key(session_id: str, namespace: str | None = None) -> str:
        return (
            f"metadata:{namespace}:{session_id}"
            if namespace
            else f"metadata:{session_id}"
        )


def safe_get(doc: Document, key: str, default: Any | None = None) -> Any:
    """Get a value from a Document, returning a default if the key is not present"""
    try:
        return getattr(doc, key)
    except AttributeError:
        return default
