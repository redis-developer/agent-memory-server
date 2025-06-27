"""Redis utility functions."""

import logging
from typing import Any

from redis.asyncio import Redis
from redisvl.index import AsyncSearchIndex

from agent_memory_server.config import settings


logger = logging.getLogger(__name__)
_redis_pool: Redis | None = None
_index: AsyncSearchIndex | None = None


async def get_redis_conn(url: str = settings.redis_url, **kwargs) -> Redis:
    """Get a Redis connection.

    Args:
        url: Redis connection URL, or None to use settings.redis_url
        **kwargs: Additional arguments to pass to Redis.from_url

    Returns:
        A Redis client instance
    """
    global _redis_pool

    # Always use the existing _redis_pool if it's not None, regardless of the URL parameter
    # This ensures that the patched _redis_pool from the test fixture is used
    if _redis_pool is None:
        _redis_pool = Redis.from_url(url, **kwargs)
    return _redis_pool


async def ensure_search_index_exists(
    redis: Redis,
    index_name: str = settings.redisvl_index_name,
    vector_dimensions: str = settings.redisvl_vector_dimensions,
    distance_metric: str = settings.redisvl_distance_metric,
    overwrite: bool = True,
) -> None:
    """
    Ensure that the async search index exists, create it if it doesn't.
    This function is deprecated and only exists for compatibility.
    The VectorStore adapter now handles index creation automatically.

    Args:
        redis: A Redis client instance
        vector_dimensions: Dimensions of the embedding vectors
        distance_metric: Distance metric to use (default: COSINE)
        index_name: The name of the index
    """
    logger.warning(
        "ensure_search_index_exists is deprecated. "
        "Index creation is now handled by the VectorStore adapter."
    )


def safe_get(doc: Any, key: str, default: Any | None = None) -> Any:
    """Get a value from a Document, returning a default if the key is not present.

    Args:
        doc: Document or object to get a value from
        key: Key to get
        default: Default value to return if key is not found

    Returns:
        The value if found, or the default
    """
    if isinstance(doc, dict):
        return doc.get(key, default)
    try:
        return getattr(doc, key)
    except (AttributeError, KeyError):
        return default
