from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from agent_memory_server import long_term_memory, messages
from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.llms import get_model_config
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

ModelNameLiteral = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o3-mini",
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus-latest",
]

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
    model_name: ModelNameLiteral | None = None,
    context_window_max: int | None = None,
):
    """
    Get memory for a session.

    This includes stored conversation history and context.

    Args:
        session_id: The session ID
        namespace: The namespace to use for the session
        window_size: The number of messages to include in the response
        model_name: The client's LLM model name (will determine context window size if provided)
        context_window_max: Direct specification of the context window max tokens (overrides model_name)

    Returns:
        Conversation history and context
    """
    redis = get_redis_conn()

    # If context_window_max is explicitly provided, use that
    if context_window_max is not None:
        effective_window_size = min(window_size, context_window_max)
    # If model_name is provided, get its max_tokens from our config
    elif model_name is not None:
        model_config = get_model_config(model_name)
        effective_window_size = min(window_size, model_config.max_tokens)
    # Otherwise use the default window_size
    else:
        effective_window_size = window_size

    session = await messages.get_session_memory(
        redis=redis,
        session_id=session_id,
        window_size=effective_window_size,
        namespace=namespace,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.put("/sessions/{session_id}/memory", response_model=AckResponse)
async def put_session_memory(
    session_id: str,
    memory: SessionMemory,
    background_tasks=Depends(get_background_tasks),
):
    """
    Set session memory. Replaces existing session memory.

    Args:
        session_id: The session ID
        memory: Messages and context to save
        background_tasks: DocketBackgroundTasks instance (injected automatically)

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
    payload: CreateLongTermMemoryPayload,
    background_tasks=Depends(get_background_tasks),
):
    """
    Create a long-term memory

    Args:
        payload: Long-term memory payload
        background_tasks: DocketBackgroundTasks instance (injected automatically)

    Returns:
        Acknowledgement response
    """
    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    await background_tasks.add_task(
        long_term_memory.index_long_term_memories,
        memories=payload.memories,
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
