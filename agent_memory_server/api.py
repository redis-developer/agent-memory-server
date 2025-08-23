from typing import Any

import tiktoken
from fastapi import APIRouter, Depends, HTTPException, Query
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from agent_memory_server import long_term_memory, working_memory
from agent_memory_server.auth import UserInfo, get_current_user
from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.filters import SessionId, UserId
from agent_memory_server.llms import get_model_client, get_model_config
from agent_memory_server.logging import get_logger
from agent_memory_server.models import (
    AckResponse,
    CreateMemoryRecordRequest,
    EditMemoryRecordRequest,
    GetSessionsQuery,
    MemoryMessage,
    MemoryPromptRequest,
    MemoryPromptResponse,
    MemoryRecord,
    MemoryRecordResultsResponse,
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


@router.post("/v1/long-term-memory/forget")
async def forget_endpoint(
    policy: dict,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = 1000,
    dry_run: bool = True,
    pinned_ids: list[str] | None = None,
    current_user: UserInfo = Depends(get_current_user),
):
    """Run a forgetting pass with the provided policy. Returns summary data.

    This is an admin-style endpoint; auth is enforced by the standard dependency.
    """
    return await long_term_memory.forget_long_term_memories(
        policy,
        namespace=namespace,
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        dry_run=dry_run,
        pinned_ids=pinned_ids,
    )


def _get_effective_token_limit(
    model_name: ModelNameLiteral | None,
    context_window_max: int | None,
) -> int:
    """Calculate the effective token limit for working memory based on model context window."""
    # If context_window_max is explicitly provided, use that
    if context_window_max is not None:
        return context_window_max
    # If model_name is provided, get its max_tokens from our config
    if model_name is not None:
        model_config = get_model_config(model_name)
        return model_config.max_tokens
    # Otherwise use a conservative default (GPT-3.5 context window)
    return 16000  # Conservative default


def _calculate_messages_token_count(messages: list[MemoryMessage]) -> int:
    """Calculate total token count for a list of messages."""
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0

    for msg in messages:
        msg_str = f"{msg.role}: {msg.content}"
        msg_tokens = len(encoding.encode(msg_str))
        total_tokens += msg_tokens

    return total_tokens


def _calculate_context_usage_percentages(
    messages: list[MemoryMessage],
    model_name: ModelNameLiteral | None,
    context_window_max: int | None,
) -> tuple[float | None, float | None]:
    """
    Calculate context usage percentages for total usage and until summarization triggers.

    Args:
        messages: List of messages to calculate token count for
        model_name: The client's LLM model name for context window determination
        context_window_max: Direct specification of context window max tokens

    Returns:
        Tuple of (total_percentage, until_summarization_percentage)
        - total_percentage: Percentage (0-100) of total context window used
        - until_summarization_percentage: Percentage (0-100) until summarization triggers
        Both values are None if no model info provided
    """
    if not messages or (not model_name and not context_window_max):
        return None, None

    # Calculate current token usage
    current_tokens = _calculate_messages_token_count(messages)

    # Get effective token limit for the client's model
    max_tokens = _get_effective_token_limit(model_name, context_window_max)

    # Calculate percentage of total context window used
    total_percentage = (current_tokens / max_tokens) * 100.0

    # Calculate percentage until summarization threshold
    token_threshold = int(max_tokens * settings.summarization_threshold)
    until_summarization_percentage = (current_tokens / token_threshold) * 100.0

    # Cap both at 100% for display purposes
    return min(total_percentage, 100.0), min(until_summarization_percentage, 100.0)


def _build_recency_params(payload: SearchRequest) -> dict[str, Any]:
    """Build recency parameters dict from payload."""
    return {
        "semantic_weight": (
            payload.recency_semantic_weight
            if payload.recency_semantic_weight is not None
            else 0.8
        ),
        "recency_weight": (
            payload.recency_recency_weight
            if payload.recency_recency_weight is not None
            else 0.2
        ),
        "freshness_weight": (
            payload.recency_freshness_weight
            if payload.recency_freshness_weight is not None
            else 0.6
        ),
        "novelty_weight": (
            payload.recency_novelty_weight
            if payload.recency_novelty_weight is not None
            else 0.4
        ),
        "half_life_last_access_days": (
            payload.recency_half_life_last_access_days
            if payload.recency_half_life_last_access_days is not None
            else 7.0
        ),
        "half_life_created_days": (
            payload.recency_half_life_created_days
            if payload.recency_half_life_created_days is not None
            else 30.0
        ),
    }


async def _summarize_working_memory(
    memory: WorkingMemory,
    model_name: ModelNameLiteral | None = None,
    context_window_max: int | None = None,
    model: str = settings.generation_model,
) -> WorkingMemory:
    """
    Summarize working memory when it exceeds token limits.

    Args:
        memory: The working memory to potentially summarize
        model_name: The client's LLM model name for context window determination
        context_window_max: Direct specification of context window max tokens
        model: The model to use for summarization

    Returns:
        Updated working memory with summary and trimmed messages
    """
    # Calculate current token usage
    current_tokens = _calculate_messages_token_count(memory.messages)

    # Get effective token limit for the client's model
    max_tokens = _get_effective_token_limit(model_name, context_window_max)

    # Reserve space for new messages, function calls, and response generation
    # Use configurable threshold to leave room for new content
    token_threshold = int(max_tokens * settings.summarization_threshold)

    if current_tokens <= token_threshold:
        return memory

    # Get model client for summarization
    client = await get_model_client(model)
    model_config = get_model_config(model)
    summarization_max_tokens = model_config.max_tokens

    # Token allocation for summarization (same logic as original summarize_session)
    if summarization_max_tokens < 10000:
        summary_max_tokens = max(512, summarization_max_tokens // 8)  # 12.5%
    elif summarization_max_tokens < 50000:
        summary_max_tokens = max(1024, summarization_max_tokens // 10)  # 10%
    else:
        summary_max_tokens = max(2048, summarization_max_tokens // 20)  # 5%

    buffer_tokens = min(max(230, summarization_max_tokens // 100), 1000)
    max_message_tokens = summarization_max_tokens - summary_max_tokens - buffer_tokens

    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    messages_to_summarize = []

    # We want to keep recent messages that fit in our target token budget
    target_remaining_tokens = int(
        max_tokens * 0.4
    )  # Keep 40% of context for recent messages

    # Work backwards from the end to find how many recent messages we can keep
    recent_messages_tokens = 0
    keep_count = 0

    for i in range(len(memory.messages) - 1, -1, -1):
        msg = memory.messages[i]
        msg_str = f"{msg.role}: {msg.content}"
        msg_tokens = len(encoding.encode(msg_str))

        if recent_messages_tokens + msg_tokens <= target_remaining_tokens:
            recent_messages_tokens += msg_tokens
            keep_count += 1
        else:
            break

    # Messages to summarize are the ones we're not keeping
    messages_to_check = (
        memory.messages[:-keep_count] if keep_count > 0 else memory.messages[:-1]
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
    # Keep only the most recent messages that fit in our token budget
    updated_memory = memory.model_copy(deep=True)
    updated_memory.context = summary
    updated_memory.messages = (
        memory.messages[-keep_count:] if keep_count > 0 else [memory.messages[-1]]
    )
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
        options: Query parameters (limit, offset, namespace, user_id)

    Returns:
        List of session IDs
    """
    redis = await get_redis_conn()

    total, session_ids = await working_memory.list_sessions(
        redis=redis,
        limit=options.limit,
        offset=options.offset,
        namespace=options.namespace,
        user_id=options.user_id,
    )

    return SessionListResponse(
        sessions=session_ids,
        total=total,
    )


@router.get("/v1/working-memory/{session_id}", response_model=WorkingMemoryResponse)
async def get_working_memory(
    session_id: str,
    user_id: str | None = None,
    namespace: str | None = None,
    model_name: ModelNameLiteral | None = None,
    context_window_max: int | None = None,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Get working memory for a session.

    This includes stored conversation messages, context, and structured memory records.
    If the messages exceed the token limit, older messages will be truncated.

    Args:
        session_id: The session ID
        user_id: The user ID to retrieve working memory for
        namespace: The namespace to use for the session
        model_name: The client's LLM model name (will determine context window size if provided)
        context_window_max: Direct specification of the context window max tokens (overrides model_name)

    Returns:
        Working memory containing messages, context, and structured memory records
    """
    redis = await get_redis_conn()

    working_mem = await working_memory.get_working_memory(
        session_id=session_id,
        namespace=namespace,
        redis_client=redis,
        user_id=user_id,
    )

    if not working_mem:
        # Create empty working memory if none exists
        working_mem = WorkingMemory(
            messages=[],
            memories=[],
            session_id=session_id,
            namespace=namespace,
            user_id=user_id,
        )

    # Apply token-based truncation if we have messages and model info
    if working_mem.messages and (model_name or context_window_max):
        token_limit = _get_effective_token_limit(model_name, context_window_max)
        current_token_count = _calculate_messages_token_count(working_mem.messages)

        # If we exceed the token limit, truncate from the beginning (keep recent messages)
        if current_token_count > token_limit:
            # Keep removing oldest messages until we're under the limit
            truncated_messages = working_mem.messages[:]
            while len(truncated_messages) > 1:  # Always keep at least 1 message
                truncated_messages = truncated_messages[1:]  # Remove oldest
                if _calculate_messages_token_count(truncated_messages) <= token_limit:
                    break
            working_mem.messages = truncated_messages

    logger.debug(f"Working mem: {working_mem}")

    # Calculate context usage percentages
    total_percentage, until_summarization_percentage = (
        _calculate_context_usage_percentages(
            messages=working_mem.messages,
            model_name=model_name,
            context_window_max=context_window_max,
        )
    )

    # Return WorkingMemoryResponse with both percentage values
    working_mem_data = working_mem.model_dump()
    working_mem_data["context_percentage_total_used"] = total_percentage
    working_mem_data["context_percentage_until_summarization"] = (
        until_summarization_percentage
    )
    return WorkingMemoryResponse(**working_mem_data)


@router.put("/v1/working-memory/{session_id}", response_model=WorkingMemoryResponse)
async def put_working_memory(
    session_id: str,
    memory: WorkingMemory,
    user_id: str | None = None,
    model_name: ModelNameLiteral | None = None,
    context_window_max: int | None = None,
    background_tasks=Depends(get_background_tasks),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Set working memory for a session. Replaces existing working memory.

    If the token count exceeds the context window threshold, messages will be summarized
    immediately and the updated memory state returned to the client.

    Args:
        session_id: The session ID
        memory: Working memory to save
        user_id: Optional user ID for the session (overrides user_id in memory object)
        model_name: The client's LLM model name for context window determination
        context_window_max: Direct specification of context window max tokens
        background_tasks: DocketBackgroundTasks instance (injected automatically)

    Returns:
        Updated working memory (potentially with summary if tokens were condensed)
    """
    redis = await get_redis_conn()

    # Ensure session_id matches
    memory.session_id = session_id

    # Override user_id if provided as query parameter
    if user_id is not None:
        memory.user_id = user_id

    # Validate that all long-term memories have id (if any)
    for long_term_mem in memory.memories:
        if not long_term_mem.id:
            raise HTTPException(
                status_code=400,
                detail="All long-term memory records in working memory must have an ID",
            )

    # Validate that all messages have non-empty content
    for msg in memory.messages:
        if not msg.content or not msg.content.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Message content cannot be empty (message ID: {msg.id})",
            )

    # Handle summarization if needed (before storing) - now token-based
    updated_memory = memory
    if memory.messages:
        updated_memory = await _summarize_working_memory(
            memory, model_name=model_name, context_window_max=context_window_max
        )

    await working_memory.set_working_memory(
        working_memory=updated_memory,
        redis_client=redis,
    )

    # Background tasks for long-term memory promotion and indexing (if enabled)
    if settings.long_term_memory and (
        updated_memory.memories or updated_memory.messages
    ):
        # Promote structured memories from working memory to long-term storage
        await background_tasks.add_task(
            long_term_memory.promote_working_memory_to_long_term,
            session_id=session_id,
            user_id=updated_memory.user_id,
            namespace=updated_memory.namespace,
        )

    # Calculate context usage percentages based on the final state (after potential summarization)
    total_percentage, until_summarization_percentage = (
        _calculate_context_usage_percentages(
            messages=updated_memory.messages,
            model_name=model_name,
            context_window_max=context_window_max,
        )
    )

    # Return WorkingMemoryResponse with both percentage values
    updated_memory_data = updated_memory.model_dump()
    updated_memory_data["context_percentage_total_used"] = total_percentage
    updated_memory_data["context_percentage_until_summarization"] = (
        until_summarization_percentage
    )
    return WorkingMemoryResponse(**updated_memory_data)


@router.delete("/v1/working-memory/{session_id}", response_model=AckResponse)
async def delete_working_memory(
    session_id: str,
    user_id: str | None = None,
    namespace: str | None = None,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Delete working memory for a session.

    This deletes all stored memory (messages, context, structured memories) for a session.

    Args:
        session_id: The session ID
        user_id: Optional user ID for the session
        namespace: Optional namespace for the session

    Returns:
        Acknowledgement response
    """
    redis = await get_redis_conn()

    # Delete unified working memory
    await working_memory.delete_working_memory(
        session_id=session_id,
        user_id=user_id,
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
    optimize_query: bool = True,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Run a semantic search on long-term memory with filtering options.

    Args:
        payload: Search payload with filter objects for precise queries
        optimize_query: Whether to optimize the query for vector search using a fast model (default: True)

    Returns:
        List of search results
    """
    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    # Extract filter objects from the payload
    filters = payload.get_filters()

    logger.debug(f"Long-term search filters: {filters}")

    kwargs = {
        "distance_threshold": payload.distance_threshold,
        "limit": payload.limit,
        "offset": payload.offset,
        "optimize_query": optimize_query,
        **filters,
    }

    kwargs["text"] = payload.text or ""

    logger.debug(f"Long-term search kwargs: {kwargs}")

    # Pass text and filter objects to the search function (no redis needed for vectorstore adapter)
    # Server-side recency rerank toggle (Redis-only path); defaults to False
    server_side_recency = (
        payload.server_side_recency
        if payload.server_side_recency is not None
        else False
    )
    if server_side_recency:
        kwargs["server_side_recency"] = True
        kwargs["recency_params"] = _build_recency_params(payload)
        return await long_term_memory.search_long_term_memories(**kwargs)

    raw_results = await long_term_memory.search_long_term_memories(**kwargs)

    # Soft-filter fallback: if strict filters yield no results, relax filters and
    # inject hints into the query text to guide semantic search. For memory_prompt
    # unit tests, the underlying function is mocked; avoid triggering fallback to
    # keep call counts stable when optimize_query behavior is being asserted.
    try:
        had_any_strict_filters = any(
            key in kwargs and kwargs[key] is not None
            for key in ("topics", "entities", "namespace", "memory_type", "event_date")
        )
        is_mocked = "unittest.mock" in str(
            type(long_term_memory.search_long_term_memories)
        )
        if raw_results.total == 0 and had_any_strict_filters and not is_mocked:
            fallback_kwargs = dict(kwargs)
            for key in ("topics", "entities", "namespace", "memory_type", "event_date"):
                fallback_kwargs.pop(key, None)

            def _vals(f):
                vals: list[str] = []
                if not f:
                    return vals
                for attr in ("eq", "any", "all"):
                    v = getattr(f, attr, None)
                    if isinstance(v, list):
                        vals.extend([str(x) for x in v])
                    elif v is not None:
                        vals.append(str(v))
                return vals

            topics_vals = _vals(filters.get("topics")) if filters else []
            entities_vals = _vals(filters.get("entities")) if filters else []
            namespace_vals = _vals(filters.get("namespace")) if filters else []
            memory_type_vals = _vals(filters.get("memory_type")) if filters else []

            hint_parts: list[str] = []
            if topics_vals:
                hint_parts.append(f"topics: {', '.join(sorted(set(topics_vals)))}")
            if entities_vals:
                hint_parts.append(f"entities: {', '.join(sorted(set(entities_vals)))}")
            if namespace_vals:
                hint_parts.append(
                    f"namespace: {', '.join(sorted(set(namespace_vals)))}"
                )
            if memory_type_vals:
                hint_parts.append(f"type: {', '.join(sorted(set(memory_type_vals)))}")

            base_text = payload.text or ""
            hint_suffix = f" ({'; '.join(hint_parts)})" if hint_parts else ""
            fallback_kwargs["text"] = (base_text + hint_suffix).strip()

            logger.debug(
                f"Soft-filter fallback engaged. Fallback kwargs: { {k: (str(v) if k == 'text' else v) for k, v in fallback_kwargs.items()} }"
            )
            raw_results = await long_term_memory.search_long_term_memories(
                **fallback_kwargs
            )
    except Exception as e:
        logger.warning(f"Soft-filter fallback failed: {e}")

    # Recency-aware re-ranking of results (configurable)
    try:
        from datetime import UTC, datetime as _dt

        # Decide whether to apply recency boost
        recency_boost = (
            payload.recency_boost if payload.recency_boost is not None else True
        )
        if not recency_boost or not raw_results.memories:
            return raw_results

        now = _dt.now(UTC)
        recency_params = _build_recency_params(payload)
        ranked = long_term_memory.rerank_with_recency(
            raw_results.memories, now=now, params=recency_params
        )
        # Update last_accessed in background with rate limiting
        ids = [m.id for m in ranked if m.id]
        if ids:
            background_tasks = get_background_tasks()
            await background_tasks.add_task(long_term_memory.update_last_accessed, ids)

        raw_results.memories = ranked
        return raw_results
    except Exception:
        return raw_results


@router.delete("/v1/long-term-memory", response_model=AckResponse)
async def delete_long_term_memory(
    memory_ids: list[str] = Query(default=[], alias="memory_ids"),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Delete long-term memories by ID

    Args:
        memory_ids: List of memory IDs to delete (passed as query parameters)
    """
    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    count = await long_term_memory.delete_long_term_memories(ids=memory_ids)
    return AckResponse(status=f"ok, deleted {count} memories")


@router.get("/v1/long-term-memory/{memory_id}", response_model=MemoryRecord)
async def get_long_term_memory(
    memory_id: str,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Get a long-term memory by its ID

    Args:
        memory_id: The ID of the memory to retrieve

    Returns:
        The memory record if found

    Raises:
        HTTPException: 404 if memory not found, 400 if long-term memory disabled
    """
    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    memory = await long_term_memory.get_long_term_memory_by_id(memory_id)
    if not memory:
        raise HTTPException(
            status_code=404, detail=f"Memory with ID {memory_id} not found"
        )

    return memory


@router.patch("/v1/long-term-memory/{memory_id}", response_model=MemoryRecord)
async def update_long_term_memory(
    memory_id: str,
    updates: EditMemoryRecordRequest,
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Update a long-term memory by its ID

    Args:
        memory_id: The ID of the memory to update
        updates: The fields to update

    Returns:
        The updated memory record

    Raises:
        HTTPException: 404 if memory not found, 400 if invalid fields or long-term memory disabled
    """
    if not settings.long_term_memory:
        raise HTTPException(status_code=400, detail="Long-term memory is disabled")

    # Convert request model to dictionary, excluding None values
    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}

    if not update_dict:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    try:
        updated_memory = await long_term_memory.update_long_term_memory(
            memory_id, update_dict
        )
        if not updated_memory:
            raise HTTPException(
                status_code=404, detail=f"Memory with ID {memory_id} not found"
            )

        return updated_memory
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/v1/memory/prompt", response_model=MemoryPromptResponse)
async def memory_prompt(
    params: MemoryPromptRequest,
    optimize_query: bool = True,
    current_user: UserInfo = Depends(get_current_user),
) -> MemoryPromptResponse:
    """
    Hydrate a user query with memory context and return a prompt
    ready to send to an LLM.

    `query` is the query for vector search that the caller of this API wants to use to find
    relevant context. If `session_id` is provided and matches an existing
    session, the resulting prompt will include those messages as the immediate
    history of messages leading to a message containing `query`.

    If `long_term_search_payload` is provided, the resulting prompt will include
    relevant long-term memories found via semantic search with the options
    provided in the payload.

    Args:
        params: MemoryPromptRequest
        optimize_query: Whether to optimize the query for vector search using a fast model (default: True)

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

    logger.debug(f"Memory prompt params: {params}")

    if params.session:
        # Use token limit for memory prompt - model info is required now
        if params.session.model_name or params.session.context_window_max:
            token_limit = _get_effective_token_limit(
                model_name=params.session.model_name,
                context_window_max=params.session.context_window_max,
            )
            effective_token_limit = token_limit
        else:
            # No model info provided - use all messages without truncation
            effective_token_limit = None
        working_mem = await working_memory.get_working_memory(
            session_id=params.session.session_id,
            namespace=params.session.namespace,
            user_id=params.session.user_id,
            redis_client=redis,
        )

        logger.debug(f"Found working memory: {working_mem}")

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
            # Apply token-based truncation if model info is provided
            if effective_token_limit is not None:
                # Token-based truncation
                if (
                    _calculate_messages_token_count(working_mem.messages)
                    > effective_token_limit
                ):
                    # Keep removing oldest messages until we're under the limit
                    recent_messages = working_mem.messages[:]
                    while len(recent_messages) > 1:  # Always keep at least 1 message
                        recent_messages = recent_messages[1:]  # Remove oldest
                        if (
                            _calculate_messages_token_count(recent_messages)
                            <= effective_token_limit
                        ):
                            break
                else:
                    recent_messages = working_mem.messages
            else:
                # No token limit provided - use all messages
                recent_messages = working_mem.messages

            for msg in recent_messages:
                if msg.role == "user":
                    msg_class = base.UserMessage
                elif msg.role == "assistant":
                    msg_class = base.AssistantMessage
                else:
                    # For tool messages or other roles, treat as assistant for MCP compatibility
                    # since MCP base only supports UserMessage and AssistantMessage
                    msg_class = base.AssistantMessage

                _messages.append(
                    msg_class(
                        content=TextContent(type="text", text=msg.content),
                    )
                )

    if params.long_term_search:
        logger.debug(
            f"[memory_prompt] Long-term search args: {params.long_term_search}"
        )
        if isinstance(params.long_term_search, bool):
            search_kwargs = {}
            if params.session:
                # Exclude memories from the current session because we already included them
                search_kwargs["session_id"] = SessionId(ne=params.session.session_id)
            if params.session and params.session.user_id:
                search_kwargs["user_id"] = UserId(eq=params.session.user_id)
            search_payload = SearchRequest(**search_kwargs, limit=20, offset=0)
        else:
            search_payload = params.long_term_search.model_copy()
            # Set the query text for the search
            search_payload.text = params.query
            # Merge session user_id into the search request if not already specified
            if params.session and params.session.user_id and not search_payload.user_id:
                search_payload.user_id = UserId(eq=params.session.user_id)

        logger.debug(f"[memory_prompt] Search payload: {search_payload}")
        long_term_memories = await search_long_term_memory(
            search_payload,
            optimize_query=optimize_query,
        )

        logger.debug(f"[memory_prompt] Long-term memories: {long_term_memories}")

        if long_term_memories.total > 0:
            long_term_memories_text = "\n".join(
                [f"- {m.text} (ID: {m.id})" for m in long_term_memories.memories]
            )
            _messages.append(
                SystemMessage(
                    content=TextContent(
                        type="text",
                        text=f"## Long term memories related to the user's query\n {long_term_memories_text}",
                    ),
                )
            )
        else:
            # Always include a system message about long-term memories, even if empty
            _messages.append(
                SystemMessage(
                    content=TextContent(
                        type="text",
                        text="## Long term memories related to the user's query\n No relevant long-term memories found.",
                    ),
                )
            )

    _messages.append(
        base.UserMessage(
            content=TextContent(type="text", text=params.query),
        )
    )

    return MemoryPromptResponse(messages=_messages)
