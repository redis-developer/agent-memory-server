from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from agent_memory_server import long_term_memory, messages
from agent_memory_server.config import settings
from agent_memory_server.logging import get_logger
from agent_memory_server.models import (
    AckResponse,
    CreateLongTermMemoryPayload,
    GetSessionsQuery,
    LongTermMemoryResultsResponse,
    SearchPayload,
    SessionListResponse,
    SessionMemory,
    SessionMemoryResponse,
)
from agent_memory_server.utils import get_redis_conn


logger = get_logger(__name__)

router = APIRouter()


@router.get("/sessions/", response_model=SessionListResponse)
async def list_sessions(
    options: GetSessionsQuery = Depends(),
):
    """
    Get a list of session IDs, with optional pagination.

    Args:
        options: Query parameters (page, size, namespace)

    Returns:
        List of session IDs
    """
    redis = get_redis_conn()

    total, session_ids = await messages.list_sessions(
        redis=redis,
        limit=options.limit,
        offset=options.offset,
        namespace=options.namespace,
    )

    return SessionListResponse(
        sessions=session_ids,
        total=total,
    )


@router.get("/sessions/{session_id}/memory", response_model=SessionMemoryResponse)
async def get_session_memory(
    session_id: str,
    namespace: str | None = None,
    window_size: int = settings.window_size,
):
    """
    Get memory for a session.

    This includes stored conversation history and context.

    Args:
        session_id: The session ID
        window_size: The number of messages to include in the response
        namespace: The namespace to use for the session

    Returns:
        Conversation history and context
    """
    redis = get_redis_conn()

    session = await messages.get_session_memory(
        redis=redis,
        session_id=session_id,
        window_size=window_size,
        namespace=namespace,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.put("/sessions/{session_id}/memory", response_model=AckResponse)
async def put_session_memory(
    session_id: str,
    memory: SessionMemory,
    background_tasks: BackgroundTasks,
):
    """
    Set session memory. Replaces existing session memory.

    Args:
        session_id: The session ID
        memory: Messages and context to save

    Returns:
        Acknowledgement response
    """
    redis = get_redis_conn()

    await messages.set_session_memory(
        redis=redis,
        session_id=session_id,
        memory=memory,
        background_tasks=background_tasks,
    )
    return AckResponse(status="ok")


@router.delete("/sessions/{session_id}/memory", response_model=AckResponse)
async def delete_session_memory(
    session_id: str,
    namespace: str | None = None,
):
    """
    Delete a session's memory

    Args:
        session_id: The session ID
        namespace: Optional namespace for the session

    Returns:
        Acknowledgement response
    """
    redis = get_redis_conn()
    await messages.delete_session_memory(
        redis=redis,
        session_id=session_id,
        namespace=namespace,
    )
    return AckResponse(status="ok")


@router.post("/long-term-memory", response_model=AckResponse)
async def create_long_term_memory(
    payload: CreateLongTermMemoryPayload, background_tasks: BackgroundTasks
):
    """
    Create a long-term memory

    Args:
        payload: Long-term memory payload

    Returns:
        Acknowledgement response
    """
    redis = get_redis_conn()

    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    await long_term_memory.index_long_term_memories(
        redis=redis,
        memories=payload.memories,
        background_tasks=background_tasks,
    )
    return AckResponse(status="ok")


@router.post("/long-term-memory/search", response_model=LongTermMemoryResultsResponse)
async def search_long_term_memory(payload: SearchPayload):
    """
    Run a semantic search on long-term memory with filtering options.

    Args:
        payload: Search payload with filter objects for precise queries

    Returns:
        List of search results
    """
    redis = get_redis_conn()

    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    # Extract filter objects from the payload
    filters = payload.get_filters()

    # Pass text, redis, and filter objects to the search function
    return await long_term_memory.search_long_term_memories(
        redis=redis,
        text=payload.text,
        distance_threshold=payload.distance_threshold,
        limit=payload.limit,
        offset=payload.offset,
        **filters,
    )
