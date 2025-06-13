import tiktoken
import ulid
from fastapi import APIRouter, Depends, HTTPException
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from agent_memory_server import long_term_memory, working_memory
from agent_memory_server.auth import UserInfo, get_current_user
from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.llms import get_model_client, get_model_config
from agent_memory_server.logging import get_logger
from agent_memory_server.models import (
    AckResponse,
    CreateMemoryRecordRequest,
    GetSessionsQuery,
    MemoryPromptRequest,
    MemoryPromptResponse,
    MemoryRecordResultsResponse,
    MemoryTypeEnum,
    ModelNameLiteral,
    SearchRequest,
    SessionListResponse,
    SystemMessage,
    WorkingMemory,
    WorkingMemoryResponse,
)
from agent_memory_server.summarization import _incremental_summary
from agent_memory_server.utils.redis import get_redis_conn


logger = get_logger(__name__)

router = APIRouter()


def _get_effective_window_size(
    window_size: int,
    context_window_max: int | None,
    model_name: ModelNameLiteral | None,
) -> int:
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
    return effective_window_size


async def _summarize_working_memory(
    memory: WorkingMemory,
    window_size: int,
    model: str = settings.generation_model,
) -> WorkingMemory:
    """
    Summarize working memory when it exceeds the window size.

    Args:
        memory: The working memory to potentially summarize
        window_size: Maximum number of messages to keep
        model: The model to use for summarization

    Returns:
        Updated working memory with summary and trimmed messages
    """
    if len(memory.messages) <= window_size:
        return memory

    # Get model client for summarization
    client = await get_model_client(model)
    model_config = get_model_config(model)
    max_tokens = model_config.max_tokens

    # Token allocation (same logic as original summarize_session)
    if max_tokens < 10000:
        summary_max_tokens = max(512, max_tokens // 8)  # 12.5%
    elif max_tokens < 50000:
        summary_max_tokens = max(1024, max_tokens // 10)  # 10%
    else:
        summary_max_tokens = max(2048, max_tokens // 20)  # 5%

    buffer_tokens = min(max(230, max_tokens // 100), 1000)
    max_message_tokens = max_tokens - summary_max_tokens - buffer_tokens

    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    messages_to_summarize = []

    # Calculate how many messages from the beginning we should summarize
    # Keep the most recent messages within window_size
    messages_to_check = (
        memory.messages[:-window_size] if len(memory.messages) > window_size else []
    )

    for msg in messages_to_check:
        msg_str = f"{msg.role}: {msg.content}"
        msg_tokens = len(encoding.encode(msg_str))

        # Handle oversized messages
        if msg_tokens > max_message_tokens:
            msg_str = msg_str[: max_message_tokens // 2]
            msg_tokens = len(encoding.encode(msg_str))

        if total_tokens + msg_tokens <= max_message_tokens:
            total_tokens += msg_tokens
            messages_to_summarize.append(msg_str)
        else:
            break

    if not messages_to_summarize:
        # No messages to summarize, just return original memory
        return memory

    # Generate summary
    summary, summary_tokens_used = await _incremental_summary(
        model,
        client,
        memory.context,  # Use existing context as base
        messages_to_summarize,
    )

    # Update working memory with new summary and trimmed messages
    # Keep only the most recent messages within window_size
    updated_memory = memory.model_copy(deep=True)
    updated_memory.context = summary
    updated_memory.messages = memory.messages[
        -window_size:
    ]  # Keep most recent messages
    updated_memory.tokens = memory.tokens + summary_tokens_used

    return updated_memory


@router.get("/v1/working-memory/", response_model=SessionListResponse)
async def list_sessions(
    options: GetSessionsQuery = Depends(),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Get a list of session IDs, with optional pagination.

    Args:
        options: Query parameters (page, size, namespace)

    Returns:
        List of session IDs
    """
    redis = await get_redis_conn()

    total, session_ids = await working_memory.list_sessions(
        redis=redis,
        limit=options.limit,
        offset=options.offset,
        namespace=options.namespace,
    )

    return SessionListResponse(
        sessions=session_ids,
        total=total,
    )


@router.get("/v1/working-memory/{session_id}", response_model=WorkingMemoryResponse)
async def get_session_memory(
    session_id: str,
    namespace: str | None = None,
    window_size: int = settings.window_size,
    model_name: ModelNameLiteral | None = None,
    context_window_max: int | None = None,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Get working memory for a session.

    This includes stored conversation messages, context, and structured memory records.

    Args:
        session_id: The session ID
        namespace: The namespace to use for the session
        window_size: The number of messages to include in the response
        model_name: The client's LLM model name (will determine context window size if provided)
        context_window_max: Direct specification of the context window max tokens (overrides model_name)

    Returns:
        Working memory containing messages, context, and structured memory records
    """
    redis = await get_redis_conn()
    effective_window_size = _get_effective_window_size(
        window_size=window_size,
        context_window_max=context_window_max,
        model_name=model_name,
    )

    # Get unified working memory
    working_mem = await working_memory.get_working_memory(
        session_id=session_id,
        namespace=namespace,
        redis_client=redis,
    )

    if not working_mem:
        # Return empty working memory if none exists
        working_mem = WorkingMemory(
            messages=[],
            memories=[],
            session_id=session_id,
            namespace=namespace,
        )

    # Apply window size to messages if needed
    if len(working_mem.messages) > effective_window_size:
        working_mem.messages = working_mem.messages[-effective_window_size:]

    return working_mem


@router.put("/v1/working-memory/{session_id}", response_model=WorkingMemoryResponse)
async def put_session_memory(
    session_id: str,
    memory: WorkingMemory,
    background_tasks=Depends(get_background_tasks),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Set working memory for a session. Replaces existing working memory.

    If the message count exceeds the window size, messages will be summarized
    immediately and the updated memory state returned to the client.

    Args:
        session_id: The session ID
        memory: Working memory to save
        background_tasks: DocketBackgroundTasks instance (injected automatically)

    Returns:
        Updated working memory (potentially with summary if messages were condensed)
    """
    redis = await get_redis_conn()

    # Ensure session_id matches
    memory.session_id = session_id

    # Validate that all structured memories have id (if any)
    for mem in memory.memories:
        if not mem.id:
            raise HTTPException(
                status_code=400,
                detail="All memory records in working memory must have an ID",
            )

    # Handle summarization if needed (before storing)
    updated_memory = memory
    if memory.messages and len(memory.messages) > settings.window_size:
        updated_memory = await _summarize_working_memory(memory, settings.window_size)

    await working_memory.set_working_memory(
        working_memory=updated_memory,
        redis_client=redis,
    )

    # Background tasks for long-term memory promotion and indexing (if enabled)
    if settings.long_term_memory:
        # Promote structured memories from working memory to long-term storage
        if updated_memory.memories:
            await background_tasks.add_task(
                long_term_memory.promote_working_memory_to_long_term,
                session_id,
                updated_memory.namespace,
            )

        # Index message-based memories (existing logic)
        if updated_memory.messages:
            from agent_memory_server.models import MemoryRecord

            memories = [
                MemoryRecord(
                    id=str(ulid.new()),
                    session_id=session_id,
                    text=f"{msg.role}: {msg.content}",
                    namespace=updated_memory.namespace,
                    memory_type=MemoryTypeEnum.MESSAGE,
                )
                for msg in updated_memory.messages
            ]

            await background_tasks.add_task(
                long_term_memory.index_long_term_memories,
                memories,
            )

    return updated_memory


@router.delete("/v1/working-memory/{session_id}", response_model=AckResponse)
async def delete_session_memory(
    session_id: str,
    namespace: str | None = None,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Delete working memory for a session.

    This deletes all stored memory (messages, context, structured memories) for a session.

    Args:
        session_id: The session ID
        namespace: Optional namespace for the session

    Returns:
        Acknowledgement response
    """
    redis = await get_redis_conn()

    # Delete unified working memory
    await working_memory.delete_working_memory(
        session_id=session_id,
        namespace=namespace,
        redis_client=redis,
    )

    return AckResponse(status="ok")


@router.post("/v1/long-term-memory/", response_model=AckResponse)
async def create_long_term_memory(
    payload: CreateMemoryRecordRequest,
    background_tasks=Depends(get_background_tasks),
    current_user: UserInfo = Depends(get_current_user),
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

    # Validate and process memories according to Stage 2 requirements
    for memory in payload.memories:
        # Enforce that id is required on memory sent from clients
        if not memory.id:
            raise HTTPException(
                status_code=400, detail="id is required for all memory records"
            )

        # Ensure persisted_at is server-assigned and read-only for clients
        # Clear any client-provided persisted_at value
        memory.persisted_at = None

    await background_tasks.add_task(
        long_term_memory.index_long_term_memories,
        memories=payload.memories,
    )
    return AckResponse(status="ok")


@router.post("/v1/long-term-memory/search", response_model=MemoryRecordResultsResponse)
async def search_long_term_memory(
    payload: SearchRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Run a semantic search on long-term memory with filtering options.

    Args:
        payload: Search payload with filter objects for precise queries

    Returns:
        List of search results
    """
    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    redis = await get_redis_conn()

    # Extract filter objects from the payload
    filters = payload.get_filters()

    kwargs = {
        "redis": redis,
        "distance_threshold": payload.distance_threshold,
        "limit": payload.limit,
        "offset": payload.offset,
        **filters,
    }

    if payload.text:
        kwargs["text"] = payload.text

    # Pass text, redis, and filter objects to the search function
    return await long_term_memory.search_long_term_memories(**kwargs)


@router.post("/v1/memory/search", response_model=MemoryRecordResultsResponse)
async def search_memory(
    payload: SearchRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Run a search across all memory types (working memory and long-term memory).

    This endpoint searches both working memory (ephemeral, session-scoped) and
    long-term memory (persistent, indexed) to provide comprehensive results.

    For working memory:
    - Uses simple text matching
    - Searches across all sessions (unless session_id filter is provided)
    - Returns memories that haven't been promoted to long-term storage

    For long-term memory:
    - Uses semantic vector search
    - Includes promoted memories from working memory
    - Supports advanced filtering by topics, entities, etc.

    Args:
        payload: Search payload with filter objects for precise queries

    Returns:
        Search results from both memory types, sorted by relevance
    """
    redis = await get_redis_conn()

    # Extract filter objects from the payload
    filters = payload.get_filters()

    kwargs = {
        "redis": redis,
        "distance_threshold": payload.distance_threshold,
        "limit": payload.limit,
        "offset": payload.offset,
        **filters,
    }

    if payload.text:
        kwargs["text"] = payload.text

    # Use the search function
    return await long_term_memory.search_memories(**kwargs)


@router.post("/v1/memory/prompt", response_model=MemoryPromptResponse)
async def memory_prompt(
    params: MemoryPromptRequest,
    current_user: UserInfo = Depends(get_current_user),
) -> MemoryPromptResponse:
    """
    Hydrate a user query with memory context and return a prompt
    ready to send to an LLM.

    `query` is the input text that the caller of this API wants to use to find
    relevant context. If `session_id` is provided and matches an existing
    session, the resulting prompt will include those messages as the immediate
    history of messages leading to a message containing `query`.

    If `long_term_search_payload` is provided, the resulting prompt will include
    relevant long-term memories found via semantic search with the options
    provided in the payload.

    Args:
        params: MemoryPromptRequest

    Returns:
        List of messages to send to an LLM, hydrated with relevant memory context
    """
    if not params.session and not params.long_term_search:
        raise HTTPException(
            status_code=400,
            detail="Either session or long_term_search must be provided",
        )

    redis = await get_redis_conn()
    _messages = []

    if params.session:
        effective_window_size = _get_effective_window_size(
            window_size=params.session.window_size,
            context_window_max=params.session.context_window_max,
            model_name=params.session.model_name,
        )
        working_mem = await working_memory.get_working_memory(
            session_id=params.session.session_id,
            namespace=params.session.namespace,
            redis_client=redis,
        )

        if working_mem:
            if working_mem.context:
                # TODO: Weird to use MCP types here?
                _messages.append(
                    SystemMessage(
                        content=TextContent(
                            type="text",
                            text=f"## A summary of the conversation so far:\n{working_mem.context}",
                        ),
                    )
                )
            # Apply window size and ignore past system messages as the latest context may have changed
            recent_messages = (
                working_mem.messages[-effective_window_size:]
                if len(working_mem.messages) > effective_window_size
                else working_mem.messages
            )
            for msg in recent_messages:
                if msg.role == "user":
                    msg_class = base.UserMessage
                else:
                    msg_class = base.AssistantMessage
                _messages.append(
                    msg_class(
                        content=TextContent(type="text", text=msg.content),
                    )
                )

    if params.long_term_search:
        # TODO: Exclude session messages if we already included them from session memory
        long_term_memories = await search_long_term_memory(
            params.long_term_search,
        )

        if long_term_memories.total > 0:
            long_term_memories_text = "\n".join(
                [f"- {m.text}" for m in long_term_memories.memories]
            )
            _messages.append(
                SystemMessage(
                    content=TextContent(
                        type="text",
                        text=f"## Long term memories related to the user's query\n {long_term_memories_text}",
                    ),
                )
            )

    _messages.append(
        base.UserMessage(
            content=TextContent(type="text", text=params.query),
        )
    )

    return MemoryPromptResponse(messages=_messages)
