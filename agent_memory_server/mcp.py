import logging
import sys

from fastapi import BackgroundTasks, HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from agent_memory_server.api import (
    create_long_term_memory as core_create_long_term_memory,
    get_session_memory as core_get_session_memory,
    search_long_term_memory as core_search_long_term_memory,
)
from agent_memory_server.config import settings
from agent_memory_server.models import (
    AckResponse,
    CreateLongTermMemoryPayload,
    LongTermMemory,
    LongTermMemoryResults,
    SearchPayload,
)


logger = logging.getLogger(__name__)
mcp_app = FastMCP("Redis Agent Memory Server", port=settings.mcp_port)


@mcp_app.tool()
async def create_long_term_memories(
    memories: list[LongTermMemory],
) -> AckResponse:
    """
    Create long-term memories that can be searched later.

    Use this tool to store information that should be retrievable in future conversations.
    Each memory can be associated with a session_id, user_id, and namespace for organization.

    IMPORTANT NOTES ON SESSION IDs:
    - When including a session_id, use the EXACT session identifier from the current conversation
    - NEVER invent or guess a session ID - if you don't know it, omit the field or ask the user
    - If you want memories accessible across all sessions, omit the session_id field
    - Session IDs from examples will NOT work with real data

    Each memory should include:
    - text: The content of the memory (required)
    - id_: Optional unique identifier (will be auto-generated if not provided)
    - session_id: Optional conversation session identifier
    - user_id: Optional user identifier
    - namespace: Optional grouping namespace
    - topics: Optional list of topics for better searchability
    - entities: Optional list of entities mentioned in the text

    Example:
    ```python
    create_long_term_memories(memories=[
        {
            "text": "The user prefers dark mode in all applications",
            "session_id": "session_12345",
            "user_id": "user_789"
        }
    ])
    ```

    Args:
        memories: A list of long-term memories to create.

    Returns:
        An acknowledgement of the creation.
    """
    payload = CreateLongTermMemoryPayload(memories=memories)
    return await core_create_long_term_memory(
        payload, background_tasks=BackgroundTasks()
    )


@mcp_app.tool()
async def search_long_term_memory(
    query: str,
    session_id: str | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    # Extended filter options
    # Session ID filters
    session_id_ne: str | None = None,
    session_ids_any: list[str] | None = None,
    session_ids_all: list[str] | None = None,
    # Namespace filters
    namespace_ne: str | None = None,
    namespaces_any: list[str] | None = None,
    namespaces_all: list[str] | None = None,
    # Topics filters
    topics_ne: str | None = None,
    topics_any: list[str] | None = None,
    topics_all: list[str] | None = None,
    # Entities filters
    entities_ne: str | None = None,
    entities_any: list[str] | None = None,
    entities_all: list[str] | None = None,
    # User ID filters
    user_id_ne: str | None = None,
    user_ids_any: list[str] | None = None,
    user_ids_all: list[str] | None = None,
    # Time range filters
    created_at_gt: int | None = None,
    created_at_lt: int | None = None,
    created_at_gte: int | None = None,
    created_at_lte: int | None = None,
    created_at_eq: int | None = None,
    created_at_ne: int | None = None,
    created_at_between: list[float] | None = None,
    last_accessed_gt: int | None = None,
    last_accessed_lt: int | None = None,
    last_accessed_gte: int | None = None,
    last_accessed_lte: int | None = None,
    last_accessed_eq: int | None = None,
    last_accessed_ne: int | None = None,
    last_accessed_between: list[float] | None = None,
    # Pagination and other options
    distance_threshold: float | None = None,
    limit: int = 10,
    offset: int = 0,
) -> LongTermMemoryResults:
    """
    Search for long-term memories using semantic similarity and filters.

    This tool performs a semantic search on stored memories that match the query text
    and any provided filters. The search returns memories ranked by relevance.

    IMPORTANT NOTES ON SESSION IDs:
    - When filtering by session_id, you must provide the EXACT session identifier
    - NEVER invent, guess, or make up a session ID - only use one that was explicitly provided
    - If no session_id has been provided to you, OMIT this parameter entirely to search across all sessions
    - Do NOT copy session IDs from examples - they will not work with the actual data
    - Current session's ID is usually provided explicitly in the conversation

    COMMON USAGE PATTERNS:

    1. Basic search with just a query:
    ```python
    search_long_term_memory(query="user's favorite color")
    ```

    2. Search within current session:
    ```python
    search_long_term_memory(
        query="user's favorite color",
        session_id="session_12345"  # Current session ID
    )
    ```

    3. Search with topic filters:
    ```python
    search_long_term_memory(
        query="color preferences",
        topics=["preferences", "ui"]
    )
    ```

    4. Advanced filtering:
    ```python
    search_long_term_memory(
        query="user preferences",
        created_at_gt=1640995200,  # After Jan 1, 2022
        topics_any=["preferences", "settings"],
        limit=5
    )
    ```

    Args:
        query: The semantic search query text (required)

        # Basic filters (most common)
        session_id: Filter to exact session ID only
        namespace: Filter to exact namespace only
        user_id: Filter to exact user ID only
        topics: Filter to memories containing any of these topics
        entities: Filter to memories containing any of these entities

        # Advanced session filters
        session_id_ne: Exclude this specific session ID
        session_ids_any: Include memories from any of these session IDs
        session_ids_all: Include only memories that have all these session IDs

        # Advanced namespace filters
        namespace_ne: Exclude this namespace
        namespaces_any: Include memories from any of these namespaces
        namespaces_all: Include only memories that have all these namespaces

        # Advanced topic filters
        topics_ne: Exclude memories with this topic
        topics_any: Include memories with any of these topics
        topics_all: Include only memories with all these topics

        # Advanced entity filters
        entities_ne: Exclude memories with this entity
        entities_any: Include memories with any of these entities
        entities_all: Include only memories with all these entities

        # Advanced user filters
        user_id_ne: Exclude this user ID
        user_ids_any: Include memories from any of these user IDs
        user_ids_all: Include only memories that have all these user IDs

        # Creation time filters
        created_at_gt: After this Unix timestamp
        created_at_lt: Before this Unix timestamp
        created_at_gte: At or after this Unix timestamp
        created_at_lte: At or before this Unix timestamp
        created_at_eq: At exactly this Unix timestamp
        created_at_ne: Not at this Unix timestamp
        created_at_between: Between these two Unix timestamps [start, end]

        # Last accessed time filters
        last_accessed_gt: Last accessed after this Unix timestamp
        last_accessed_lt: Last accessed before this Unix timestamp
        last_accessed_gte: Last accessed at or after this Unix timestamp
        last_accessed_lte: Last accessed at or before this Unix timestamp
        last_accessed_eq: Last accessed at exactly this Unix timestamp
        last_accessed_ne: Not last accessed at this Unix timestamp
        last_accessed_between: Last accessed between these Unix timestamps [start, end]

        # Results options
        distance_threshold: Maximum semantic distance (0.0-1.0, lower is more similar)
        limit: Maximum number of results to return (default: 10)
        offset: Number of results to skip for pagination (default: 0)

    Returns:
        Object containing matched memories sorted by relevance, with pagination info
    """
    # Create a payload with proper filter objects
    payload = SearchPayload.create_with_primitives(
        text=query,
        # Session ID filters
        session_id=session_id,
        session_id_ne=session_id_ne,
        session_ids_any=session_ids_any,
        session_ids_all=session_ids_all,
        # Namespace filters
        namespace=namespace,
        namespace_ne=namespace_ne,
        namespaces_any=namespaces_any,
        namespaces_all=namespaces_all,
        # Topics filters
        topics=topics,
        topics_ne=topics_ne,
        topics_any=topics_any,
        topics_all=topics_all,
        # Entities filters
        entities=entities,
        entities_ne=entities_ne,
        entities_any=entities_any,
        entities_all=entities_all,
        # User ID filters
        user_id=user_id,
        user_id_ne=user_id_ne,
        user_ids_any=user_ids_any,
        user_ids_all=user_ids_all,
        # Time range filters
        created_at_gt=created_at_gt,
        created_at_lt=created_at_lt,
        created_at_gte=created_at_gte,
        created_at_lte=created_at_lte,
        created_at_eq=created_at_eq,
        created_at_ne=created_at_ne,
        created_at_between=created_at_between,
        last_accessed_gt=last_accessed_gt,
        last_accessed_lt=last_accessed_lt,
        last_accessed_gte=last_accessed_gte,
        last_accessed_lte=last_accessed_lte,
        last_accessed_eq=last_accessed_eq,
        last_accessed_ne=last_accessed_ne,
        last_accessed_between=last_accessed_between,
        # Other options
        distance_threshold=distance_threshold,
        limit=limit,
        offset=offset,
    )
    return await core_search_long_term_memory(payload)


@mcp_app.prompt()
async def memory_prompt(
    query: str,
    session_id: str | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    # Extended filter options
    # Session ID filters
    session_id_ne: str | None = None,
    session_ids_any: list[str] | None = None,
    session_ids_all: list[str] | None = None,
    # Namespace filters
    namespace_ne: str | None = None,
    namespaces_any: list[str] | None = None,
    namespaces_all: list[str] | None = None,
    # Topics filters
    topics_ne: str | None = None,
    topics_any: list[str] | None = None,
    topics_all: list[str] | None = None,
    # Entities filters
    entities_ne: str | None = None,
    entities_any: list[str] | None = None,
    entities_all: list[str] | None = None,
    # User ID filters
    user_id_ne: str | None = None,
    user_ids_any: list[str] | None = None,
    user_ids_all: list[str] | None = None,
    # Pagination and other options
    distance_threshold: float | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[base.Message]:
    """
    Create a prompt that includes relevant session history and long-term memories.

    This prompt generator enriches the user's query with:
    1. Context from the current conversation session (if available)
    2. Relevant long-term memories related to the query

    IMPORTANT NOTES ON SESSION IDs:
    - When filtering by session_id, you must provide the EXACT session identifier
    - NEVER invent, guess, or make up a session ID - only use one that was explicitly provided
    - If no session_id has been provided to you, OMIT this parameter entirely to search across all sessions
    - Do NOT copy session IDs from examples - they will not work with the actual data
    - Current session's ID is usually provided explicitly in the conversation

    Example usage:
    ```python
    memory_prompt(
        session_id="session_12345",
        query="What was my favorite color again?",
        namespace="user_preferences"  # Optional namespace to search in
    )
    ```

    Args:
        query: The user's query/message to enhance with context (required)
        session_id: Optional session identifier
        namespace: Optional namespace to filter memories by

    Returns:
        A list of messages that form a prompt, including context and the user's query
    """
    messages = []
    if session_id:
        try:
            session_memory = await core_get_session_memory(session_id)
        except HTTPException:
            session_memory = None

    if session_memory:
        if session_memory.context:
            messages.append(
                base.AssistantMessage(
                    content=TextContent(
                        type="text",
                        text=f"## A summary of the conversation so far\n{session_memory.context}",
                    ),
                )
            )
        for msg in session_memory.messages:
            if msg.role == "user":
                msg_class = base.UserMessage
            else:
                msg_class = base.AssistantMessage
            messages.append(
                msg_class(
                    content=TextContent(type="text", text=msg.content),
                )
            )

    try:
        # Create a long-term memory search payload with proper filter objects
        search_payload = SearchPayload.create_with_primitives(
            text=query,
            # Session ID filters
            session_id=session_id,
            session_id_ne=session_id_ne,
            session_ids_any=session_ids_any,
            session_ids_all=session_ids_all,
            # Namespace filters
            namespace=namespace,
            namespace_ne=namespace_ne,
            namespaces_any=namespaces_any,
            namespaces_all=namespaces_all,
            # Topics filters
            topics=topics,
            topics_ne=topics_ne,
            topics_any=topics_any,
            topics_all=topics_all,
            # Entities filters
            entities=entities,
            entities_ne=entities_ne,
            entities_any=entities_any,
            entities_all=entities_all,
            # User ID filters
            user_id=user_id,
            user_id_ne=user_id_ne,
            user_ids_any=user_ids_any,
            user_ids_all=user_ids_all,
            # Other options
            distance_threshold=distance_threshold,
            limit=limit,
            offset=offset,
        )
        long_term_memories = await core_search_long_term_memory(search_payload)
        if long_term_memories.total > 0:
            long_term_memories_text = "\n".join(
                [f"- {m.text}" for m in long_term_memories.memories]
            )
            messages.append(
                base.AssistantMessage(
                    content=TextContent(
                        type="text",
                        text=f"## Long term memories related to the user's query\n {long_term_memories_text}",
                    ),
                )
            )
    except Exception as e:
        logger.error(f"Error searching long-term memory: {e}")

    messages.append(
        base.UserMessage(
            content=TextContent(type="text", text=query),
        )
    )

    return messages


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        mcp_app.run(transport="sse")
    else:
        mcp_app.run(transport="stdio")
