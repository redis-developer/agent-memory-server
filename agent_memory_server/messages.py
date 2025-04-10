import json
import logging
import time

from fastapi import BackgroundTasks
from redis import WatchError
from redis.asyncio import Redis

from agent_memory_server.config import settings
from agent_memory_server.long_term_memory import index_long_term_memories
from agent_memory_server.models import (
    LongTermMemory,
    MemoryMessage,
    SessionMemory,
)
from agent_memory_server.summarization import summarize_session
from agent_memory_server.utils import (
    Keys,
)


logger = logging.getLogger(__name__)


async def list_sessions(
    redis: Redis,
    limit: int = 20,
    offset: int = 0,
    namespace: str | None = None,
) -> tuple[int, list[str]]:
    """List sessions"""
    # Calculate start and end indices (0-indexed start, inclusive end)
    start = offset
    end = offset + limit - 1

    sessions_key = Keys.sessions_key(namespace=namespace)
    async with redis.pipeline() as pipe:
        pipe.zcard(sessions_key)
        pipe.zrange(sessions_key, start, end)
        total, session_ids = await pipe.execute()

    return total, [
        s.decode("utf-8") if isinstance(s, bytes) else s for s in session_ids
    ]


async def get_session_memory(
    redis: Redis,
    session_id: str,
    window_size: int = settings.window_size,
    namespace: str | None = None,
) -> SessionMemory | None:
    """Get a session's memory"""
    sessions_key = Keys.sessions_key(namespace=namespace)
    messages_key = Keys.messages_key(session_id, namespace=namespace)
    metadata_key = Keys.metadata_key(session_id, namespace=namespace)

    session_exists = await redis.zscore(sessions_key, session_id)
    if not session_exists:
        return None

    pipe = redis.pipeline()
    pipe.lrange(messages_key, 0, window_size - 1)  # Get messages
    pipe.hgetall(metadata_key)  # Get metadata

    messages_raw, metadata_raw = await pipe.execute()

    memory_messages = []
    for msg_raw in messages_raw:
        # Decode if needed
        if isinstance(msg_raw, bytes):
            msg_raw = msg_raw.decode("utf-8")

        # Parse JSON
        msg_dict = json.loads(msg_raw)

        # Convert comma-separated strings back to lists for topics and entities
        if "topics" in msg_dict:
            msg_dict["topics"] = (
                msg_dict["topics"].split(",") if msg_dict["topics"] else []
            )
        if "entities" in msg_dict:
            msg_dict["entities"] = (
                msg_dict["entities"].split(",") if msg_dict["entities"] else []
            )

        memory_messages.append(MemoryMessage(**msg_dict))

    kwargs = {
        "messages": memory_messages,
        **{k.decode("utf-8"): v.decode("utf-8") for k, v in metadata_raw.items()},
    }

    return SessionMemory(**kwargs)


async def set_session_memory(
    redis: Redis,
    session_id: str,
    memory: SessionMemory,
    background_tasks: BackgroundTasks,
):
    """
    Create or update a session's memory

    TODO: This shouldn't need BackgroundTasks.

    Args:
        redis: The Redis client
        session_id: The session ID
        memory: The session memory to set
        background_tasks: The background tasks to add the summarization task to
    """
    sessions_key = Keys.sessions_key(namespace=memory.namespace)
    messages_key = Keys.messages_key(session_id, namespace=memory.namespace)
    metadata_key = Keys.metadata_key(session_id, namespace=memory.namespace)
    messages_json = [json.dumps(msg.model_dump()) for msg in memory.messages]
    metadata = memory.model_dump(
        exclude_none=True,
        exclude={"messages"},
    )

    async with redis.pipeline(transaction=True) as pipe:
        await pipe.watch(messages_key, metadata_key)
        pipe.multi()

        while True:
            try:
                current_time = int(time.time())
                pipe.zadd(sessions_key, {session_id: current_time})
                pipe.rpush(messages_key, *messages_json)  # type: ignore
                pipe.hset(metadata_key, mapping=metadata)  # type: ignore
                await pipe.execute()
            except WatchError:
                continue
            break

    # Check if window size is exceeded
    current_size = await redis.llen(messages_key)  # type: ignore
    if current_size > settings.window_size:
        # Handle summarization in background
        background_tasks.add_task(
            summarize_session,
            redis,
            session_id,
            settings.generation_model,
            settings.window_size,
        )

    # If long-term memory is enabled, index messages
    # TODO: Use a distributed task queue
    # TODO: Allow strategies for long-term memory: indexing
    #       messages vs. extracting memories from messages, etc.
    if settings.long_term_memory:
        background_tasks.add_task(
            index_long_term_memories,
            redis,
            [
                LongTermMemory(
                    session_id=session_id,
                    text=f"{msg.role}: {msg.content}",
                    namespace=memory.namespace,
                )
                for msg in memory.messages
            ],
            background_tasks,
        )


async def delete_session_memory(
    redis: Redis,
    session_id: str,
    namespace: str | None = None,
):
    """Delete a session's memory"""
    # Define keys
    messages_key = Keys.messages_key(session_id)
    sessions_key = f"sessions:{namespace}" if namespace else "sessions"
    metadata_key = Keys.metadata_key(session_id, namespace=namespace)

    # Create pipeline for deletion
    pipe = redis.pipeline()
    pipe.delete(messages_key, metadata_key)
    pipe.zrem(sessions_key, session_id)
    await pipe.execute()
