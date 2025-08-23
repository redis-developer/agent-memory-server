import logging
from datetime import datetime
from typing import Any

import ulid
from mcp.server.fastmcp import FastMCP as _FastMCPBase

from agent_memory_server.api import (
    create_long_term_memory as core_create_long_term_memory,
    delete_long_term_memory as core_delete_long_term_memory,
    get_long_term_memory as core_get_long_term_memory,
    get_working_memory as core_get_working_memory,
    memory_prompt as core_memory_prompt,
    put_working_memory as core_put_working_memory,
    search_long_term_memory as core_search_long_term_memory,
    update_long_term_memory as core_update_long_term_memory,
)
from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    LastAccessed,
    MemoryType,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from agent_memory_server.models import (
    AckResponse,
    CreateMemoryRecordRequest,
    EditMemoryRecordRequest,
    LenientMemoryRecord,
    MemoryMessage,
    MemoryPromptRequest,
    MemoryPromptResponse,
    MemoryRecord,
    MemoryRecordResults,
    MemoryTypeEnum,
    ModelNameLiteral,
    SearchRequest,
    WorkingMemory,
    WorkingMemoryRequest,
    WorkingMemoryResponse,
)


logger = logging.getLogger(__name__)


def _parse_iso8601_datetime(event_date: str) -> datetime:
    """
    Parse ISO 8601 datetime string with robust handling of different timezone formats.

    Args:
        event_date: ISO 8601 formatted datetime string

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If the datetime format is invalid
    """
    try:
        # Handle 'Z' suffix (UTC indicator)
        if event_date.endswith("Z"):
            return datetime.fromisoformat(event_date.replace("Z", "+00:00"))
        # Let fromisoformat handle other timezone formats like +05:00, -08:00, etc.
        return datetime.fromisoformat(event_date)
    except ValueError as e:
        raise ValueError(f"Invalid ISO 8601 datetime format '{event_date}': {e}") from e


class FastMCP(_FastMCPBase):
    """Extend FastMCP to support optional URL namespace and default STDIO namespace."""

    def __init__(self, *args, default_namespace=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_namespace = default_namespace
        self._current_request = None  # Initialize the attribute

    def sse_app(self):
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.routing import Mount, Route

        sse = SseServerTransport(self.settings.message_path)

        async def handle_sse(request: Request) -> None:
            # Store the request in the FastMCP instance so call_tool can access it
            self._current_request = request

            try:
                async with sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send,  # type: ignore
                ) as (read_stream, write_stream):
                    await self._mcp_server.run(
                        read_stream,
                        write_stream,
                        self._mcp_server.create_initialization_options(),
                    )
            finally:
                # Clean up request reference
                self._current_request = None

        return Starlette(
            debug=self.settings.debug,
            routes=[
                Route(self.settings.sse_path, endpoint=handle_sse),
                Route(f"/{{namespace}}{self.settings.sse_path}", endpoint=handle_sse),
                Mount(self.settings.message_path, app=sse.handle_post_message),
                Mount(
                    f"/{{namespace}}{self.settings.message_path}",
                    app=sse.handle_post_message,
                ),
            ],
        )

    async def call_tool(self, name, arguments):
        # Get the namespace from the request context
        namespace = None
        try:
            # RequestContext doesn't expose the path_params directly
            # We use a ThreadLocal or context variable pattern instead
            from starlette.requests import Request

            request = getattr(self, "_current_request", None)
            if isinstance(request, Request):
                namespace = request.path_params.get("namespace")
        except Exception:
            # Silently continue if we can't get namespace from request
            pass

        # Inject namespace only for tools that accept it
        if name in ("search_long_term_memory", "hydrate_memory_prompt"):
            if namespace and "namespace" not in arguments:
                arguments["namespace"] = Namespace(eq=namespace)
            elif (
                not namespace
                and self.default_namespace
                and "namespace" not in arguments
            ):
                arguments["namespace"] = Namespace(eq=self.default_namespace)
        elif name in ("set_working_memory",):
            if namespace and "namespace" not in arguments:
                arguments["namespace"] = namespace
            elif (
                not namespace
                and self.default_namespace
                and "namespace" not in arguments
            ):
                arguments["namespace"] = self.default_namespace

        return await super().call_tool(name, arguments)

    async def run_sse_async(self):
        """Ensure Redis search index exists before starting SSE server."""
        from agent_memory_server.utils.redis import (
            ensure_search_index_exists,
            get_redis_conn,
        )

        redis = await get_redis_conn()
        await ensure_search_index_exists(redis)

        # Run the SSE server using our custom implementation
        import uvicorn

        app = self.sse_app()
        await uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=int(self.settings.port))
        ).serve()

    async def run_stdio_async(self):
        """Ensure Redis search index exists before starting STDIO MCP server."""
        from agent_memory_server.utils.redis import (
            ensure_search_index_exists,
            get_redis_conn,
        )

        redis = await get_redis_conn()
        await ensure_search_index_exists(redis)
        return await super().run_stdio_async()


INSTRUCTIONS = """
    When responding to user queries, ALWAYS check memory first before answering
    questions about user preferences, history, or personal information.
"""


mcp_app = FastMCP(
    "Redis Agent Memory Server",
    port=settings.mcp_port,
    instructions=INSTRUCTIONS,
    default_namespace=settings.default_mcp_namespace,
)


@mcp_app.tool()
async def get_current_datetime() -> dict[str, str | int]:
    """
    Get the current datetime in UTC for grounding relative time expressions.

    Use this tool whenever the user provides a relative time (e.g., "today",
    "yesterday", "last week") or when you need to include a concrete date in
    text. Always combine this with setting the structured `event_date` field on
    episodic memories.

    Returns:
        - iso_utc: Current time in ISO 8601 format with Z suffix, e.g.,
          "2025-08-14T23:59:59Z"
        - unix_ts: Current Unix timestamp (seconds)

    Example:
        1. User: "I was promoted today"
           - Call get_current_datetime → use `iso_utc` to set `event_date`
           - Update text to include a grounded, human-readable date
             (e.g., "User was promoted to Principal Engineer on August 14, 2025.")
    """
    now = datetime.utcnow()
    # Produce a Z-suffixed ISO 8601 string
    iso_utc = now.replace(microsecond=0).isoformat() + "Z"
    return {"iso_utc": iso_utc, "unix_ts": int(now.timestamp())}


@mcp_app.tool()
async def create_long_term_memories(
    memories: list[LenientMemoryRecord],
) -> AckResponse:
    """
    Create long-term memories that can be searched later.

    This tool saves memories contained in the payload for future retrieval.

    CONTEXTUAL GROUNDING REQUIREMENTS:
    When creating memories, you MUST resolve all contextual references to their concrete referents:

    1. PRONOUNS: Replace ALL pronouns (he/she/they/him/her/them/his/hers/theirs) with actual person names
       - "He prefers Python" → "User prefers Python" (if "he" refers to the user)
       - "Her expertise is valuable" → "User's expertise is valuable" (if "her" refers to the user)

    2. TEMPORAL REFERENCES: Convert relative time expressions to absolute dates/times
       - "yesterday" → "2024-03-15" (if today is March 16, 2024)
       - "last week" → "March 4-10, 2024" (if current week is March 11-17, 2024)

    3. SPATIAL REFERENCES: Resolve place references to specific locations
       - "there" → "San Francisco office" (if referring to SF office)
       - "here" → "the main conference room" (if referring to specific room)

    4. DEFINITE REFERENCES: Resolve definite articles to specific entities
       - "the project" → "the customer portal redesign project"
       - "the bug" → "the authentication timeout issue"

    MANDATORY: Never create memories with unresolved pronouns, vague time references, or unclear spatial references. Always ground contextual references using the full conversation context.

    MEMORY TYPES - SEMANTIC vs EPISODIC:

    There are two main types of long-term memories you can create:

    1. **SEMANTIC MEMORIES** (memory_type="semantic"):
       - General facts, knowledge, and user preferences that are timeless
       - Information that remains relevant across multiple conversations
       - User preferences, settings, and general knowledge
       - Examples:
         * "User prefers dark mode in all applications"
         * "User is a data scientist working with Python"
         * "User dislikes spicy food"
         * "The company's API rate limit is 1000 requests per hour"

    2. **EPISODIC MEMORIES** (memory_type="episodic"):
       - Specific events, experiences, or time-bound information
       - Things that happened at a particular time or in a specific context
       - MUST have a time dimension to be truly episodic
       - Should include an event_date when the event occurred
       - Examples:
         * "User visited Paris last month and had trouble with the metro"
         * "User reported a login bug on January 15th, 2024"
         * "User completed the onboarding process yesterday"
         * "User mentioned they're traveling to Tokyo next week"

    WHEN TO USE EACH TYPE:

    Use SEMANTIC for:
    - User preferences and settings
    - Skills, roles, and background information
    - General facts and knowledge
    - Persistent user characteristics
    - System configuration and rules

    Use EPISODIC for:
    - Specific events and experiences
    - Time-bound activities and plans
    - Historical interactions and outcomes
    - Contextual information tied to specific moments

    IMPORTANT NOTES ON SESSION IDs:
    - When including a session_id, use the EXACT session identifier from the current conversation
    - NEVER invent or guess a session ID - if you don't know it, omit the field
    - If you want memories accessible across all sessions, omit the session_id field

    COMMON USAGE PATTERNS:

    1. Create semantic memories (user preferences):
    ```python
    create_long_term_memories(
        memories=[
            {
                "text": "User prefers dark mode in all applications",
                "memory_type": "semantic",
                "user_id": "user_789",
                "namespace": "user_preferences",
                "topics": ["preferences", "ui", "theme"]
            }
        ]
    )
    ```

    2. Create episodic memories (specific events):
    ```python
    create_long_term_memories(
        memories=[
            {
                "text": "User reported login issues during morning session",
                "memory_type": "episodic",
                "event_date": "2024-01-15T09:30:00Z",  # Semantic memories must have an event_date!
                "user_id": "user_789",
                "topics": ["bug_report", "authentication"],
                "entities": ["login", "authentication_system"]
            }
        ]
    )
    ```

    3. Create multiple memories of different types:
    ```python
    create_long_term_memories(
        memories=[
            {
                "text": "User is a Python developer",
                "memory_type": "semantic",
                "topics": ["skills", "programming"]
            },
            {
                "text": "User completed Python certification course last week",
                "memory_type": "episodic",
                "event_date": "2024-01-10T00:00:00Z",
                "topics": ["education", "achievement"]
            }
        ]
    )
    ```

    4. Create memories with different namespaces:
    ```python
    create_long_term_memories(
        memories=[
            {
                "text": "User prefers email notifications",
                "memory_type": "semantic",
                "namespace": "user_preferences"
            },
            {
                "text": "System maintenance scheduled for next weekend",
                "memory_type": "episodic",
                "namespace": "system_events",
                "event_date": "2024-01-20T02:00:00Z"
            }
        ]
    )
    ```

    Args:
        memories: A list of MemoryRecord objects to create

    Returns:
        An acknowledgement response indicating success
    """
    # Apply default namespace for STDIO if not provided in memory entries
    for mem in memories:
        if mem.namespace is None and settings.default_mcp_namespace:
            mem.namespace = settings.default_mcp_namespace
        if mem.user_id is None and settings.default_mcp_user_id:
            mem.user_id = settings.default_mcp_user_id

    payload = CreateMemoryRecordRequest(memories=memories)
    return await core_create_long_term_memory(
        payload, background_tasks=get_background_tasks()
    )


@mcp_app.tool()
async def search_long_term_memory(
    text: str | None,
    session_id: SessionId | None = None,
    namespace: Namespace | None = None,
    topics: Topics | None = None,
    entities: Entities | None = None,
    created_at: CreatedAt | None = None,
    last_accessed: LastAccessed | None = None,
    user_id: UserId | None = None,
    memory_type: MemoryType | None = None,
    distance_threshold: float | None = None,
    limit: int = 10,
    offset: int = 0,
    optimize_query: bool = False,
) -> MemoryRecordResults:
    """
    Search for memories related to a query for vector search.

    Finds memories based on a combination of semantic similarity and input filters.

    This tool performs a semantic search on stored memories using the query for vector search and filters
    in the payload. Results are ranked by relevance.

    DATETIME INPUT FORMAT:
    - All datetime filters accept ISO 8601 formatted strings (e.g., "2023-01-01T00:00:00Z")
    - Timezone-aware datetimes are recommended (use "Z" for UTC or "+HH:MM" for other timezones)
    - Supported operations: gt, gte, lt, lte, eq, ne, between
    - Example: {"gt": "2023-01-01T00:00:00Z", "lt": "2024-01-01T00:00:00Z"}

    IMPORTANT NOTES ON SESSION IDs:
    - When including a session_id filter, use the EXACT session identifier
    - NEVER invent or guess a session ID - if you don't know it, omit this filter
    - If you want to search across all sessions, don't include a session_id filter
    - Session IDs from examples will NOT work with real data

    COMMON USAGE PATTERNS:

    1. Basic search with just query text:
    ```python
    search_long_term_memory(text="user's favorite color")
    ```

    2. Get ALL memories for a user (e.g., "what do you remember about me?"):
    ```python
    search_long_term_memory(
        text="",  # Empty string returns all memories for the user
        user_id={"eq": "user_123"},
        limit=50  # Adjust based on how many memories you want
    )
    ```

    3. Search with simple session filter:
    ```python
    search_long_term_memory(text="user's favorite color", session_id={
        "eq": "session_12345"
    })
    ```

    4. Search with complex filters:
    ```python
    search_long_term_memory(
        text="user preferences",
        topics={
            "any": ["preferences", "settings"]
        },
        created_at={
            "gt": "2023-01-01T00:00:00Z"
        },
        limit=5
    )
    ```

    5. Search with datetime range filters:
    ```python
    search_long_term_memory(
        text="recent conversations",
        created_at={
            "gte": "2024-01-01T00:00:00Z",
            "lt": "2024-02-01T00:00:00Z"
        },
        last_accessed={
            "gt": "2024-01-15T12:00:00Z"
        }
    )
    ```

    6. Search with between datetime filter:
    ```python
    search_long_term_memory(
        text="holiday discussions",
        created_at={
            "between": ["2023-12-20T00:00:00Z", "2023-12-31T23:59:59Z"]
        }
    )
    ```

    Args:
        text: The query for vector search (required). Use empty string "" to get all memories for a user.
        session_id: Filter by session ID
        namespace: Filter by namespace
        topics: Filter by topics
        entities: Filter by entities
        created_at: Filter by creation date
        last_accessed: Filter by last access date
        user_id: Filter by user ID
        memory_type: Filter by memory type
        distance_threshold: Distance threshold for semantic search
        limit: Maximum number of results
        offset: Offset for pagination
        optimize_query: Whether to optimize the query for vector search (default: False - LLMs typically provide already optimized queries)

    Returns:
        MemoryRecordResults containing matched memories sorted by relevance
    """
    if user_id is None and settings.default_mcp_user_id:
        user_id = UserId(eq=settings.default_mcp_user_id)
    if namespace is None and settings.default_mcp_namespace:
        namespace = Namespace(eq=settings.default_mcp_namespace)

    try:
        payload = SearchRequest(
            text=text,
            session_id=session_id,
            namespace=namespace,
            topics=topics,
            entities=entities,
            created_at=created_at,
            last_accessed=last_accessed,
            user_id=user_id,
            memory_type=memory_type,
            distance_threshold=distance_threshold,
            limit=limit,
            offset=offset,
        )
        results = await core_search_long_term_memory(
            payload, optimize_query=optimize_query
        )
        return MemoryRecordResults(
            total=results.total,
            memories=results.memories,
            next_offset=results.next_offset,
        )
    except Exception as e:
        logger.error(f"Error in search_long_term_memory tool: {e}")
        return MemoryRecordResults(total=0, memories=[], next_offset=None)


# Notes that exist outside of the docstring to avoid polluting the LLM prompt:
# 1. The "prompt" abstraction in FastAPI doesn't support search filters, so we use a tool.
# 2. Some applications, such as Cursor, get confused with nested objects in tool parameters,
#    so we use a flat set of parameters instead.
@mcp_app.tool()
async def memory_prompt(
    query: str,
    session_id: SessionId | None = None,
    namespace: Namespace | None = None,
    model_name: ModelNameLiteral | None = None,
    context_window_max: int | None = None,
    topics: Topics | None = None,
    entities: Entities | None = None,
    created_at: CreatedAt | None = None,
    last_accessed: LastAccessed | None = None,
    user_id: UserId | None = None,
    memory_type: MemoryType | None = None,
    distance_threshold: float | None = None,
    limit: int = 10,
    offset: int = 0,
    optimize_query: bool = False,
) -> MemoryPromptResponse:
    """
    Hydrate a query for vector search with relevant session history and long-term memories.

    This tool enriches the query by retrieving:
    1. Context from the current conversation session
    2. Relevant long-term memories related to the query

    The tool returns both the relevant memories AND the user's query in a format ready for
    generating comprehensive responses.

    The function uses the query field as the query for vector search,
    and any filters to retrieve relevant memories.

    DATETIME INPUT FORMAT:
    - All datetime filters accept ISO 8601 formatted strings (e.g., "2023-01-01T00:00:00Z")
    - Timezone-aware datetimes are recommended (use "Z" for UTC or "+HH:MM" for other timezones)
    - Supported operations: gt, gte, lt, lte, eq, ne, between
    - Example: {"gt": "2023-01-01T00:00:00Z", "lt": "2024-01-01T00:00:00Z"}

    IMPORTANT NOTES ON SESSION IDs:
    - When filtering by session_id, you must provide the EXACT session identifier
    - NEVER invent or guess a session ID - if you don't know it, omit this filter
    - Session IDs from examples will NOT work with real data

    COMMON USAGE PATTERNS:
    ```python
    1. Hydrate a user prompt with long-term memory search:
    memory_prompt(query="What was my favorite color?")
    ```

    2. Answer "what do you remember about me?" type questions:
    memory_prompt(
        query="What do you remember about me?",
        user_id={"eq": "user_123"},
        limit=50
    )
    ```

    3. Hydrate a user prompt with long-term memory search and session filter:
    memory_prompt(
        query="What is my favorite color?",
        session_id={
            "eq": "session_12345"
        },
        namespace={
            "eq": "user_preferences"
        }
    )

    4. Hydrate a user prompt with long-term memory search and complex filters:
    memory_prompt(
        query="What was my favorite color?",
        topics={
            "any": ["preferences", "settings"]
        },
        created_at={
            "gt": "2023-01-01T00:00:00Z"
        },
        limit=5
    )

    5. Search with datetime range filters:
    memory_prompt(
        query="What did we discuss recently?",
        created_at={
            "gte": "2024-01-01T00:00:00Z",
            "lt": "2024-02-01T00:00:00Z"
        },
        last_accessed={
            "gt": "2024-01-15T12:00:00Z"
        }
    )
    ```

    Args:
        - query: The query for vector search
        - session_id: Add conversation history from a working memory session
        - namespace: Filter session and long-term memory namespace
        - topics: Search for long-term memories matching topics
        - entities: Search for long-term memories matching entities
        - created_at: Search for long-term memories matching creation date
        - last_accessed: Search for long-term memories matching last access date
        - user_id: Search for long-term memories matching user ID
        - distance_threshold: Distance threshold for semantic search
        - limit: Maximum number of long-term memory results
        - offset: Offset for pagination of long-term memory results
        - optimize_query: Whether to optimize the query for vector search (default: False - LLMs typically provide already optimized queries)

    Returns:
        A list of messages, including memory context and the user's query
    """
    _session_id = session_id.eq if session_id and session_id.eq else None
    session = None

    if user_id is None and settings.default_mcp_user_id:
        user_id = UserId(eq=settings.default_mcp_user_id)

    if _session_id is not None:
        session = WorkingMemoryRequest(
            session_id=_session_id,
            namespace=namespace.eq if namespace and namespace.eq else None,
            user_id=user_id.eq if user_id and user_id.eq else None,
            model_name=model_name,
            context_window_max=context_window_max,
        )

    search_payload = SearchRequest(
        text=query,
        session_id=session_id,
        namespace=namespace,
        topics=topics,
        entities=entities,
        created_at=created_at,
        last_accessed=last_accessed,
        user_id=user_id,
        distance_threshold=distance_threshold,
        memory_type=memory_type,
        limit=limit,
        offset=offset,
    )
    _params = {}
    if session is not None:
        _params["session"] = session
    if search_payload is not None:
        _params["long_term_search"] = search_payload

    return await core_memory_prompt(
        params=MemoryPromptRequest(query=query, **_params),
        optimize_query=optimize_query,
    )


@mcp_app.tool()
async def set_working_memory(
    session_id: str,
    memories: list[LenientMemoryRecord] | None = None,
    messages: list[MemoryMessage] | None = None,
    context: str | None = None,
    data: dict[str, Any] | None = None,
    namespace: str | None = settings.default_mcp_namespace,
    user_id: str | None = settings.default_mcp_user_id,
    ttl_seconds: int = 3600,
) -> WorkingMemoryResponse:
    """
    Set working memory for a session. This works like the PUT /sessions/{id}/memory API endpoint.

    Replaces existing working memory with new content. Can store structured memory records
    and messages, but agents should primarily use this for memory records and JSON data,
    not conversation messages.

    USAGE PATTERNS:

    1. Store structured memory records:
    ```python
    set_working_memory(
        session_id="current_session",
        memories=[
            {
                "text": "User prefers dark mode",
                "id": "pref_dark_mode",
                "memory_type": "semantic",
                "topics": ["preferences", "ui"]
            }
        ]
    )
    ```

    2. Store arbitrary JSON data separately:
    ```python
    set_working_memory(
        session_id="current_session",
        data={
            "user_settings": {"theme": "dark", "lang": "en"},
            "preferences": {"notifications": True, "sound": False}
        }
    )
    ```

    3. Store both memories and JSON data:
    ```python
    set_working_memory(
        session_id="current_session",
        memories=[
            {
                "text": "User prefers dark mode",
                "id": "pref_dark_mode",
                "memory_type": "semantic",
                "topics": ["preferences", "ui"]
            }
        ],
        data={
            "current_settings": {"theme": "dark", "lang": "en"}
        }
    )
    ```

    4. Store conversation messages:
    ```python
    set_working_memory(
        session_id="current_session",
        messages=[
            {
                "role": "user",
                "content": "What is the weather like?",
                "id": "msg_001"  # Optional - auto-generated if not provided
            },
            {
                "role": "assistant",
                "content": "I'll check the weather for you."
            }
        ]
    )
    ```

    5. Replace entire working memory state:
    ```python
    set_working_memory(
        session_id="current_session",
        memories=[...],  # structured memories
        messages=[...],  # conversation history
        context="Summary of previous conversation",
        user_id="user123"
    )
    ```

    Args:
        session_id: The session ID to set memory for (required)
        memories: List of structured memory records (semantic, episodic, message types)
        messages: List of conversation messages (role/content pairs with optional id/persisted_at)
        context: Optional summary/context text
        data: Optional dictionary for storing arbitrary JSON data
        namespace: Optional namespace for scoping
        user_id: Optional user ID
        ttl_seconds: TTL for the working memory (default 1 hour)

    Returns:
        Updated working memory response (may include summarization if window exceeded)
    """
    # Auto-generate IDs for memories that don't have them
    processed_memories = []
    if memories:
        for memory in memories:
            # Handle both MemoryRecord objects and dict inputs
            if isinstance(memory, MemoryRecord):
                # Already a MemoryRecord object, ensure it has an ID
                memory_id = memory.id or str(ulid.ULID())
                processed_memory = memory.model_copy(
                    update={
                        "id": memory_id,
                        "user_id": user_id,
                        "persisted_at": None,  # Mark as pending promotion
                    }
                )
            else:
                # Dictionary input, convert to MemoryRecord
                memory_dict = dict(memory)
                if not memory_dict.get("id"):
                    memory_dict["id"] = str(ulid.ULID())
                memory_dict["persisted_at"] = None
                processed_memory = MemoryRecord(**memory_dict)

            processed_memories.append(processed_memory)

    # Process messages to ensure proper format
    processed_messages = []
    if messages:
        for message in messages:
            # Handle both MemoryMessage objects and dict inputs
            if isinstance(message, MemoryMessage):
                # Already a MemoryMessage object, ensure persisted_at is None for new messages
                processed_message = message.model_copy(
                    update={
                        "persisted_at": None,  # Mark as pending promotion
                    }
                )
            else:
                # Dictionary input, convert to MemoryMessage
                message_dict = dict(message)
                # Remove id=None to allow auto-generation
                if message_dict.get("id") is None:
                    message_dict.pop("id", None)
                message_dict["persisted_at"] = None
                processed_message = MemoryMessage(**message_dict)

            processed_messages.append(processed_message)

    # Create the working memory object
    working_memory_obj = WorkingMemory(
        session_id=session_id,
        namespace=namespace,
        memories=processed_memories,
        messages=processed_messages,
        context=context,
        data=data or {},
        user_id=user_id,
        ttl_seconds=ttl_seconds,
    )

    # Update working memory via the API - this handles summarization and background promotion
    result = await core_put_working_memory(
        session_id=session_id,
        memory=working_memory_obj,
        background_tasks=get_background_tasks(),
    )

    # Convert to WorkingMemoryResponse to satisfy return type
    return WorkingMemoryResponse(**result.model_dump())


@mcp_app.tool()
async def get_working_memory(
    session_id: str,
) -> WorkingMemory:
    """
    Get working memory for a session. This works like the GET /sessions/{id}/memory API endpoint.
    """
    return await core_get_working_memory(session_id=session_id)


@mcp_app.tool()
async def get_long_term_memory(
    memory_id: str,
) -> MemoryRecord:
    """
    Get a long-term memory by its ID.

    This tool retrieves a specific long-term memory record using its unique identifier.

    Args:
        memory_id: The unique ID of the memory to retrieve

    Returns:
        The memory record if found

    Raises:
        Exception: If memory not found or long-term memory is disabled

    Example:
    ```python
    get_long_term_memory(memory_id="01HXE2B1234567890ABCDEF")
    ```
    """
    return await core_get_long_term_memory(memory_id=memory_id)


@mcp_app.tool()
async def edit_long_term_memory(
    memory_id: str,
    text: str | None = None,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    memory_type: MemoryTypeEnum | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    event_date: str | None = None,
) -> MemoryRecord:
    """
    Edit an existing long-term memory by its ID.

    This tool allows you to update specific fields of a long-term memory record.
    Only the fields you provide will be updated; other fields remain unchanged.

    IMPORTANT: Use this tool whenever you need to update existing memories based on new information
    or corrections provided by the user. This is essential for maintaining accurate memory records.

    Args:
        memory_id: The unique ID of the memory to edit (required)
        text: Updated text content for the memory
        topics: Updated list of topics for the memory
        entities: Updated list of entities mentioned in the memory
        memory_type: Updated memory type ("semantic", "episodic", or "message")
        namespace: Updated namespace for organizing the memory
        user_id: Updated user ID associated with the memory
        session_id: Updated session ID where the memory originated
        event_date: Updated event date for episodic memories (ISO 8601 format: "2024-01-15T14:30:00Z")

    Returns:
        The updated memory record

    Raises:
        Exception: If memory not found, invalid fields, or long-term memory is disabled

    IMPORTANT DATE HANDLING RULES:
    - For time-bound updates (episodic), ALWAYS set `event_date`.
    - When users provide relative dates ("today", "yesterday", "last week"),
      call `get_current_datetime` to resolve the current date/time, then set
      `event_date` using the ISO value and include a grounded, human-readable
      date in the `text` (e.g., "on August 14, 2025").
    - Do not guess dates; if unsure, ask or omit the date phrase in `text`.

    COMMON USAGE PATTERNS:

    1. Update memory text content:
    ```python
    edit_long_term_memory(
        memory_id="01HXE2B1234567890ABCDEF",
        text="User prefers dark mode UI (updated preference)"
    )
    ```

    2. Update memory type and add event date:
    ```python
    edit_long_term_memory(
        memory_id="01HXE2B1234567890ABCDEF",
        memory_type="episodic",
        event_date="2024-01-15T14:30:00Z"
    )
    ```

    2b. Include grounded date in text AND set event_date:
    ```python
    # After resolving relative time with get_current_datetime
    edit_long_term_memory(
        memory_id="01HXE2B1234567890ABCDEF",
        text="User was promoted to Principal Engineer on January 15, 2024.",
        memory_type="episodic",
        event_date="2024-01-15T14:30:00Z"
    )
    ```

    3. Update topics and entities:
    ```python
    edit_long_term_memory(
        memory_id="01HXE2B1234567890ABCDEF",
        topics=["preferences", "ui", "accessibility"],
        entities=["dark_mode", "user_interface"]
    )
    ```

    4. Update multiple fields at once:
    ```python
    edit_long_term_memory(
        memory_id="01HXE2B1234567890ABCDEF",
        text="User completed Python certification course",
        memory_type="episodic",
        event_date="2024-01-10T00:00:00Z",
        topics=["education", "achievement", "python"],
        entities=["Python", "certification"]
    )
    ```

    5. Move memory to different namespace or user:
    ```python
    edit_long_term_memory(
        memory_id="01HXE2B1234567890ABCDEF",
        namespace="work_projects",
        user_id="user_456"
    )
    ```
    """
    # Build the update request dictionary, handling event_date parsing
    update_dict = {
        "text": text,
        "topics": topics,
        "entities": entities,
        "memory_type": memory_type,
        "namespace": namespace,
        "user_id": user_id,
        "session_id": session_id,
        "event_date": (
            _parse_iso8601_datetime(event_date) if event_date is not None else None
        ),
    }

    # Filter out None values to only include fields that should be updated
    update_dict = {k: v for k, v in update_dict.items() if v is not None}
    updates = EditMemoryRecordRequest(**update_dict)

    return await core_update_long_term_memory(memory_id=memory_id, updates=updates)


@mcp_app.tool()
async def delete_long_term_memories(
    memory_ids: list[str],
) -> AckResponse:
    """
    Delete long-term memories by their IDs.

    This tool permanently removes specified long-term memory records.
    Use with caution as this action cannot be undone.

    Args:
        memory_ids: List of memory IDs to delete

    Returns:
        Acknowledgment response with the count of deleted memories

    Raises:
        Exception: If long-term memory is disabled or deletion fails

    Example:
    ```python
    delete_long_term_memories(
        memory_ids=["01HXE2B1234567890ABCDEF", "01HXE2B9876543210FEDCBA"]
    )
    ```
    """
    if not settings.long_term_memory:
        raise ValueError("Long-term memory is disabled")

    return await core_delete_long_term_memory(memory_ids=memory_ids)
