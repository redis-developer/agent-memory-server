"""
Agent Memory Server API Client

This module provides a standalone client for the REST API of the Agent Memory Server.
"""

import asyncio
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from typing_extensions import Self

import httpx
from pydantic import BaseModel
from ulid import ULID

from .exceptions import MemoryClientError, MemoryServerError, MemoryValidationError
from .filters import (
    CreatedAt,
    Entities,
    LastAccessed,
    MemoryType,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from .models import (
    AckResponse,
    ClientMemoryRecord,
    HealthCheckResponse,
    MemoryRecord,
    MemoryRecordResults,
    MemoryTypeEnum,
    ModelNameLiteral,
    SessionListResponse,
    WorkingMemory,
    WorkingMemoryResponse,
)

# === Tool Call Type Definitions ===


class OpenAIFunctionCall(TypedDict):
    """OpenAI function call format (legacy)."""

    name: str
    arguments: str


class OpenAIToolCall(TypedDict):
    """OpenAI tool call format (current)."""

    id: str
    type: Literal["function"]
    function: OpenAIFunctionCall


class AnthropicToolUse(TypedDict):
    """Anthropic tool use format."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class UnifiedToolCall(TypedDict):
    """Unified tool call format for internal use."""

    id: str | None
    name: str
    arguments: dict[str, Any]
    provider: Literal["openai", "anthropic", "generic"]


class ToolCallResolutionResult(TypedDict):
    """Result of resolving a tool call."""

    success: bool
    function_name: str
    result: Any | None
    error: str | None
    formatted_response: str


# === Client Configuration ===


class MemoryClientConfig(BaseModel):
    """Configuration for the Memory API Client"""

    base_url: str
    timeout: float = 30.0
    default_namespace: str | None = None
    default_model_name: str | None = None
    default_context_window_max: int | None = None


class MemoryAPIClient:
    """
    Client for the Agent Memory Server REST API.

    This client provides methods to interact with all server endpoints:
    - Health check
    - Session management (list, get, put, delete)
    - Long-term memory (create, search)
    - Enhanced functionality (lifecycle, batch, pagination, validation)
    """

    def __init__(self, config: MemoryClientConfig):
        """
        Initialize the Memory API Client.

        Args:
            config: MemoryClientConfig instance with server connection details
        """
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "Self":
        """Support using the client as an async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the client when exiting the context manager."""
        await self.close()

    def _handle_http_error(self, response: httpx.Response) -> None:
        """Handle HTTP errors and convert to appropriate exceptions."""
        if response.status_code == 404:
            from .exceptions import MemoryNotFoundError

            raise MemoryNotFoundError(f"Resource not found: {response.url}")
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("detail", f"HTTP {response.status_code}")
            except Exception:
                message = f"HTTP {response.status_code}: {response.text}"
            raise MemoryServerError(message, response.status_code)

    async def health_check(self) -> HealthCheckResponse:
        """
        Check the health of the memory server.

        Returns:
            HealthCheckResponse with current server timestamp
        """
        try:
            response = await self._client.get("/v1/health")
            response.raise_for_status()
            return HealthCheckResponse(**response.json())
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e.response)
            raise

    async def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> SessionListResponse:
        """
        List available sessions with optional pagination and namespace filtering.

        Args:
            limit: Maximum number of sessions to return (default: 20)
            offset: Offset for pagination (default: 0)
            namespace: Optional namespace filter
            user_id: Optional user ID filter

        Returns:
            SessionListResponse containing session IDs and total count
        """
        params = {
            "limit": str(limit),
            "offset": str(offset),
        }
        if namespace is not None:
            params["namespace"] = namespace
        elif self.config.default_namespace is not None:
            params["namespace"] = self.config.default_namespace

        if user_id is not None:
            params["user_id"] = user_id

        try:
            response = await self._client.get("/v1/working-memory/", params=params)
            response.raise_for_status()
            return SessionListResponse(**response.json())
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e.response)
            raise

    async def get_working_memory(
        self,
        session_id: str,
        user_id: str | None = None,
        namespace: str | None = None,
        window_size: int | None = None,
        model_name: ModelNameLiteral | None = None,
        context_window_max: int | None = None,
    ) -> WorkingMemoryResponse:
        """
        Get working memory for a session, including messages and context.

        Args:
            session_id: The session ID to retrieve working memory for
            user_id: The user ID to retrieve working memory for
            namespace: Optional namespace for the session
            window_size: Optional number of messages to include
            model_name: Optional model name to determine context window size
            context_window_max: Optional direct specification of context window tokens

        Returns:
            WorkingMemoryResponse containing messages, context and metadata

        Raises:
            MemoryNotFoundError: If the session is not found
            MemoryServerError: For other server errors
        """
        params = {}

        if user_id is not None:
            params["user_id"] = user_id

        if namespace is not None:
            params["namespace"] = namespace
        elif self.config.default_namespace is not None:
            params["namespace"] = self.config.default_namespace

        if window_size is not None:
            params["window_size"] = str(window_size)

        # Use provided model_name or fall back to config default
        effective_model_name = model_name or self.config.default_model_name
        if effective_model_name is not None:
            params["model_name"] = effective_model_name

        # Use provided context_window_max or fall back to config default
        effective_context_window_max = (
            context_window_max or self.config.default_context_window_max
        )
        if effective_context_window_max is not None:
            params["context_window_max"] = str(effective_context_window_max)

        try:
            response = await self._client.get(
                f"/v1/working-memory/{session_id}", params=params
            )
            response.raise_for_status()

            # Get the raw JSON response
            response_data = response.json()

            # Messages from JSON parsing are already in the correct dict format
            return WorkingMemoryResponse(**response_data)
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e.response)
            raise

    async def put_working_memory(
        self,
        session_id: str,
        memory: WorkingMemory,
        user_id: str | None = None,
        model_name: str | None = None,
        context_window_max: int | None = None,
    ) -> WorkingMemoryResponse:
        """
        Store session memory. Replaces existing session memory if it exists.

        Args:
            session_id: The session ID to store memory for
            memory: WorkingMemory object with messages and optional context
            user_id: Optional user ID for the session (overrides user_id in memory object)
            model_name: Optional model name for context window management
            context_window_max: Optional direct specification of context window max tokens

        Returns:
            WorkingMemoryResponse with the updated memory (potentially summarized if token limit exceeded)
        """
        # If namespace not specified in memory but set in config, use config's namespace
        if memory.namespace is None and self.config.default_namespace is not None:
            memory.namespace = self.config.default_namespace

        # Build query parameters for model-aware summarization
        params = {}

        if user_id is not None:
            params["user_id"] = user_id

        # Use provided model_name or fall back to config default
        effective_model_name = model_name or self.config.default_model_name
        if effective_model_name is not None:
            params["model_name"] = effective_model_name

        # Use provided context_window_max or fall back to config default
        effective_context_window_max = (
            context_window_max or self.config.default_context_window_max
        )
        if effective_context_window_max is not None:
            params["context_window_max"] = str(effective_context_window_max)

        try:
            response = await self._client.put(
                f"/v1/working-memory/{session_id}",
                json=memory.model_dump(exclude_none=True, mode="json"),
                params=params,
            )
            response.raise_for_status()
            return WorkingMemoryResponse(**response.json())
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e.response)
            raise

    async def delete_working_memory(
        self, session_id: str, namespace: str | None = None, user_id: str | None = None
    ) -> AckResponse:
        """
        Delete working memory for a session.

        Args:
            session_id: The session ID to delete memory for
            namespace: Optional namespace for the session
            user_id: Optional user ID for the session

        Returns:
            AckResponse indicating success
        """
        params = {}
        if namespace is not None:
            params["namespace"] = namespace
        elif self.config.default_namespace is not None:
            params["namespace"] = self.config.default_namespace

        if user_id is not None:
            params["user_id"] = user_id

        try:
            response = await self._client.delete(
                f"/v1/working-memory/{session_id}", params=params
            )
            response.raise_for_status()
            return AckResponse(**response.json())
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e.response)
            raise

    async def set_working_memory_data(
        self,
        session_id: str,
        data: dict[str, Any],
        namespace: str | None = None,
        preserve_existing: bool = True,
    ) -> WorkingMemoryResponse:
        """
        Convenience method to set JSON data in working memory.

        This method allows you to easily store arbitrary JSON data in working memory
        without having to construct a full WorkingMemory object.

        Args:
            session_id: The session ID to set data for
            data: Dictionary of JSON data to store
            namespace: Optional namespace for the session
            preserve_existing: If True, preserve existing messages and memories (default: True)

        Returns:
            WorkingMemoryResponse with the updated memory

        Example:
            ```python
            # Store user preferences
            await client.set_working_memory_data(
                session_id="session123",
                data={
                    "user_settings": {"theme": "dark", "language": "en"},
                    "preferences": {"notifications": True}
                }
            )
            ```
        """
        # Get existing memory if preserving
        existing_memory = None
        if preserve_existing:
            existing_memory = await self.get_working_memory(
                session_id=session_id,
                namespace=namespace,
            )

        # Create new working memory with the data
        working_memory = WorkingMemory(
            session_id=session_id,
            namespace=namespace or self.config.default_namespace,
            messages=existing_memory.messages if existing_memory else [],
            memories=existing_memory.memories if existing_memory else [],
            data=data,
            context=existing_memory.context if existing_memory else None,
            user_id=existing_memory.user_id if existing_memory else None,
        )

        return await self.put_working_memory(session_id, working_memory)

    async def add_memories_to_working_memory(
        self,
        session_id: str,
        memories: list[ClientMemoryRecord | MemoryRecord],
        namespace: str | None = None,
        replace: bool = False,
    ) -> WorkingMemoryResponse:
        """
        Convenience method to add structured memories to working memory.

        This method allows you to easily add MemoryRecord objects to working memory
        without having to manually construct and manage the full WorkingMemory object.

        Args:
            session_id: The session ID to add memories to
            memories: List of MemoryRecord objects to add
            namespace: Optional namespace for the session
            replace: If True, replace all existing memories; if False, append to existing (default: False)

        Returns:
            WorkingMemoryResponse with the updated memory

        Example:
            ```python
            # Add a semantic memory
            await client.add_memories_to_working_memory(
                session_id="session123",
                memories=[
                    MemoryRecord(
                        text="User prefers dark mode",
                        memory_type="semantic",
                        topics=["preferences", "ui"],
                        id="pref_dark_mode"
                    )
                ]
            )
            ```
        """
        # Get existing memory
        existing_memory = await self.get_working_memory(
            session_id=session_id,
            namespace=namespace,
        )

        # Determine final memories list
        if replace or not existing_memory:
            final_memories = memories
        else:
            final_memories = existing_memory.memories + memories

        # Auto-generate IDs for memories that don't have them
        for memory in final_memories:
            if not memory.id:
                memory.id = str(ULID())

        # Create new working memory with the memories
        working_memory = WorkingMemory(
            session_id=session_id,
            namespace=namespace or self.config.default_namespace,
            messages=existing_memory.messages if existing_memory else [],
            memories=final_memories,
            data=existing_memory.data if existing_memory else {},
            context=existing_memory.context if existing_memory else None,
            user_id=existing_memory.user_id if existing_memory else None,
        )

        return await self.put_working_memory(session_id, working_memory)

    async def create_long_term_memory(
        self, memories: list[ClientMemoryRecord | MemoryRecord]
    ) -> AckResponse:
        """
        Create long-term memories for later retrieval.

        Args:
            memories: List of MemoryRecord objects to store

        Returns:
            AckResponse indicating success

        Raises:
            MemoryServerError: If long-term memory is disabled or other errors

        Example:
            ```python
            from .models import ClientMemoryRecord

            # Store user preferences as semantic memory
            memories = [
                ClientMemoryRecord(
                    text="User prefers dark mode interface",
                    memory_type="semantic",
                    topics=["preferences", "ui"],
                    entities=["dark_mode", "interface"]
                ),
                ClientMemoryRecord(
                    text="User mentioned they work late nights frequently",
                    memory_type="episodic",
                    topics=["work_habits", "schedule"],
                    entities=["work", "schedule"]
                )
            ]

            response = await client.create_long_term_memory(memories)
            print(f"Stored memories: {response.status}")
            ```
        """
        # Apply default namespace if needed
        if self.config.default_namespace is not None:
            for memory in memories:
                if memory.namespace is None:
                    memory.namespace = self.config.default_namespace

        payload = {
            "memories": [m.model_dump(exclude_none=True, mode="json") for m in memories]
        }

        try:
            response = await self._client.post(
                "/v1/long-term-memory/",
                json=payload,
            )
            response.raise_for_status()
            return AckResponse(**response.json())
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e.response)
            raise

    async def search_long_term_memory(
        self,
        text: str,
        session_id: SessionId | dict[str, Any] | None = None,
        namespace: Namespace | dict[str, Any] | None = None,
        topics: Topics | dict[str, Any] | None = None,
        entities: Entities | dict[str, Any] | None = None,
        created_at: CreatedAt | dict[str, Any] | None = None,
        last_accessed: LastAccessed | dict[str, Any] | None = None,
        user_id: UserId | dict[str, Any] | None = None,
        distance_threshold: float | None = None,
        memory_type: MemoryType | dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """
        Search long-term memories using semantic search and filters.

        Args:
            text: Search query text for semantic similarity
            session_id: Optional session ID filter
            namespace: Optional namespace filter
            topics: Optional topics filter
            entities: Optional entities filter
            created_at: Optional creation date filter
            last_accessed: Optional last accessed date filter
            user_id: Optional user ID filter
            distance_threshold: Optional distance threshold for search results
            memory_type: Optional memory type filter
            limit: Maximum number of results to return (default: 10)
            offset: Offset for pagination (default: 0)

        Returns:
            MemoryRecordResults with matching memories and metadata

        Raises:
            MemoryServerError: If long-term memory is disabled or other errors

        Example:
            ```python
            # Search with topic and entity filters
            from .filters import Topics, Entities

            results = await client.search_long_term_memory(
                text="meeting notes about project alpha",
                topics=Topics(all=["meetings", "projects"]),
                entities=Entities(any=["project_alpha", "team_meeting"]),
                limit=10,
                distance_threshold=0.3
            )

            print(f"Found {results.total} memories")
            for memory in results.memories:
                print(f"- {memory.text[:100]}... (distance: {memory.dist})")
            ```
        """
        # Convert dictionary filters to their proper filter objects if needed
        if isinstance(session_id, dict):
            session_id = SessionId(**session_id)
        if isinstance(namespace, dict):
            namespace = Namespace(**namespace)
        if isinstance(topics, dict):
            topics = Topics(**topics)
        if isinstance(entities, dict):
            entities = Entities(**entities)
        if isinstance(created_at, dict):
            created_at = CreatedAt(**created_at)
        if isinstance(last_accessed, dict):
            last_accessed = LastAccessed(**last_accessed)
        if isinstance(memory_type, dict):
            memory_type = MemoryType(**memory_type)

        # Apply default namespace if needed and no namespace filter specified
        if namespace is None and self.config.default_namespace is not None:
            namespace = Namespace(eq=self.config.default_namespace)

        payload = {
            "text": text,
            "limit": limit,
            "offset": offset,
        }

        # Add filters if provided
        if session_id:
            payload["session_id"] = session_id.model_dump(exclude_none=True)
        if namespace:
            payload["namespace"] = namespace.model_dump(exclude_none=True)
        if topics:
            payload["topics"] = topics.model_dump(exclude_none=True)
        if entities:
            payload["entities"] = entities.model_dump(exclude_none=True)
        if created_at:
            payload["created_at"] = created_at.model_dump(
                exclude_none=True, mode="json"
            )
        if last_accessed:
            payload["last_accessed"] = last_accessed.model_dump(
                exclude_none=True, mode="json"
            )
        if user_id:
            if isinstance(user_id, dict):
                payload["user_id"] = user_id
            else:
                payload["user_id"] = user_id.model_dump(exclude_none=True)
        if memory_type:
            payload["memory_type"] = memory_type.model_dump(exclude_none=True)
        if distance_threshold is not None:
            payload["distance_threshold"] = distance_threshold

        try:
            response = await self._client.post(
                "/v1/memory/search",
                json=payload,
            )
            response.raise_for_status()
            return MemoryRecordResults(**response.json())
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e.response)
            raise

    # === LLM Tool Integration ===

    async def search_memory_tool(
        self,
        query: str,
        topics: list[str] | None = None,
        entities: list[str] | None = None,
        memory_type: str | None = None,
        max_results: int = 5,
        min_relevance: float | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Simplified long-term memory search designed for LLM tool use.

        This method provides a streamlined interface for LLMs to search
        long-term memory with common parameters and user-friendly output.
        Perfect for exposing as a tool to LLM frameworks. Note: This only
        searches long-term memory, not working memory.

        Args:
            query: The search query text
            topics: Optional list of topic strings to filter by
            entities: Optional list of entity strings to filter by
            memory_type: Optional memory type ("episodic", "semantic", "message")
            max_results: Maximum results to return (default: 5)
            min_relevance: Optional minimum relevance score (0.0-1.0)
            user_id: Optional user ID to filter memories by

        Returns:
            Dict with 'memories' list and 'summary' for LLM consumption

        Example:
            ```python
            # Simple search for LLM tool use
            result = await client.search_memory_tool(
                query="user preferences about UI themes",
                topics=["preferences", "ui"],
                max_results=3,
                min_relevance=0.7
            )

            print(result["summary"])  # "Found 2 relevant memories for: user preferences about UI themes"
            for memory in result["memories"]:
                print(f"- {memory['text']} (score: {memory['relevance_score']})")
            ```

        LLM Framework Integration:
            ```python
            # Register as OpenAI tool
            tools = [MemoryAPIClient.get_memory_search_tool_schema()]

            # Handle tool calls
            if tool_call.function.name == "search_memory":
                args = json.loads(tool_call.function.arguments)
                result = await client.search_memory_tool(**args)
            ```
        """
        from .filters import Entities, MemoryType, Topics

        # Convert simple parameters to filter objects
        topics_filter = Topics(any=topics) if topics else None
        entities_filter = Entities(any=entities) if entities else None
        memory_type_filter = MemoryType(eq=memory_type) if memory_type else None
        user_id_filter = UserId(eq=user_id) if user_id else None

        # Convert min_relevance to distance_threshold (assuming 0-1 relevance maps to 1-0 distance)
        distance_threshold = (
            (1.0 - min_relevance) if min_relevance is not None else None
        )

        results = await self.search_long_term_memory(
            text=query,
            topics=topics_filter,
            entities=entities_filter,
            memory_type=memory_type_filter,
            distance_threshold=distance_threshold,
            limit=max_results,
            user_id=user_id_filter,
        )

        # Format for LLM consumption
        formatted_memories = []
        for memory in results.memories:
            formatted_memories.append(
                {
                    "text": memory.text,
                    "memory_type": memory.memory_type,
                    "topics": memory.topics or [],
                    "entities": memory.entities or [],
                    "created_at": memory.created_at.isoformat()
                    if memory.created_at
                    else None,
                    "relevance_score": 1.0 - memory.dist
                    if hasattr(memory, "dist") and memory.dist is not None
                    else None,
                }
            )

        return {
            "memories": formatted_memories,
            "total_found": results.total,
            "query": query,
            "summary": f"Found {len(formatted_memories)} relevant memories for: {query}",
        }

    @classmethod
    def get_memory_search_tool_schema(cls) -> dict[str, Any]:
        """
        Get OpenAI-compatible tool schema for memory search.

        Returns tool definition that can be passed to LLM frameworks
        like OpenAI, Anthropic Claude, etc. Use this to register
        memory search as a tool that LLMs can call.

        Returns:
            Tool schema dictionary compatible with OpenAI tool calling format

        Example:
            ```python
            # Register with OpenAI
            import openai

            tools = [MemoryAPIClient.get_memory_search_tool_schema()]

            response = await openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "What did I say about my preferences?"}],
                tools=tools,
                tool_choice="auto"
            )
            ```

        Tool Handler Example:
            ```python
            async def handle_tool_calls(client, tool_calls):
                for tool_call in tool_calls:
                    if tool_call.function.name == "search_memory":
                        args = json.loads(tool_call.function.arguments)
                        result = await client.search_memory_tool(**args)
                        # Process result and send back to LLM
                        yield result
            ```
        """
        return {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Search long-term memory for relevant information based on a query. Use this when you need to recall past conversations, user preferences, or previously stored information. Note: This searches only long-term memory, not current working memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query describing what information you're looking for",
                        },
                        "topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of topics to filter by (e.g., ['preferences', 'work', 'personal'])",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of entities to filter by (e.g., ['John', 'project_alpha', 'meetings'])",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "message"],
                            "description": "Optional filter by memory type: 'episodic' (events/experiences), 'semantic' (facts/knowledge), 'message' (conversation history)",
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                            "description": "Maximum number of results to return",
                        },
                        "min_relevance": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Optional minimum relevance score (0.0-1.0, higher = more relevant)",
                        },
                        "user_id": {
                            "type": "string",
                            "description": "Optional user ID to filter memories by (e.g., 'user123')",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    # === Working Memory Tool Integration ===

    async def get_working_memory_tool(
        self,
        session_id: str,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get current working memory state formatted for LLM consumption.

        This method provides a summary of the current working memory state
        that's easy for LLMs to understand and work with.

        Args:
            session_id: The session ID to get memory for
            namespace: Optional namespace for the session
            user_id: Optional user ID for the session

        Returns:
            Dict with formatted working memory information

        Example:
            ```python
            # Get working memory state for LLM
            memory_state = await client.get_working_memory_tool(
                session_id="current_session"
            )

            print(memory_state["summary"])  # Human-readable summary
            print(f"Messages: {memory_state['message_count']}")
            print(f"Memories: {len(memory_state['memories'])}")
            ```
        """
        try:
            result = await self.get_working_memory(
                session_id=session_id,
                namespace=namespace or self.config.default_namespace,
                user_id=user_id,
            )

            # Format for LLM consumption
            message_count = len(result.messages) if result.messages else 0
            memory_count = len(result.memories) if result.memories else 0
            data_keys = list(result.data.keys()) if result.data else []

            # Create formatted memories list
            formatted_memories = []
            if result.memories:
                for memory in result.memories:
                    formatted_memories.append(
                        {
                            "text": memory.text,
                            "memory_type": memory.memory_type,
                            "topics": memory.topics or [],
                            "entities": memory.entities or [],
                            "created_at": memory.created_at.isoformat()
                            if memory.created_at
                            else None,
                        }
                    )

            return {
                "session_id": session_id,
                "message_count": message_count,
                "memory_count": memory_count,
                "memories": formatted_memories,
                "data_keys": data_keys,
                "data": result.data or {},
                "context": result.context,
                "summary": f"Session has {message_count} messages, {memory_count} stored memories, and {len(data_keys)} data entries",
            }

        except Exception as e:
            return {
                "session_id": session_id,
                "error": str(e),
                "summary": f"Error retrieving working memory: {str(e)}",
            }

    async def add_memory_tool(
        self,
        session_id: str,
        text: str,
        memory_type: str,
        topics: list[str] | None = None,
        entities: list[str] | None = None,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a structured memory to working memory with LLM-friendly response.

        This method allows LLMs to store important information as structured
        memories that will be automatically managed by the memory server.

        Args:
            session_id: The session ID to add memory to
            text: The memory content to store
            memory_type: Type of memory ("episodic", "semantic", "message")
            topics: Optional topics for categorization
            entities: Optional entities mentioned
            namespace: Optional namespace for the session
            user_id: Optional user ID for the session

        Returns:
            Dict with success/failure information

        Example:
            ```python
            # Store user preference as semantic memory
            result = await client.add_memory_tool(
                session_id="current_session",
                text="User prefers vegetarian restaurants",
                memory_type="semantic",
                topics=["preferences", "dining"],
                entities=["vegetarian", "restaurants"]
            )

            print(result["summary"])  # "Successfully stored semantic memory"
            ```
        """
        try:
            # Create memory record
            memory = ClientMemoryRecord(
                text=text,
                memory_type=MemoryTypeEnum(memory_type),
                topics=topics,
                entities=entities,
                namespace=namespace or self.config.default_namespace,
                user_id=user_id,
            )

            # Add to working memory
            await self.add_memories_to_working_memory(
                session_id=session_id,
                memories=[memory],
                namespace=namespace or self.config.default_namespace,
                replace=False,
            )

            return {
                "success": True,
                "memory_type": memory_type,
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "topics": topics or [],
                "entities": entities or [],
                "summary": f"Successfully stored {memory_type} memory: {text[:50]}...",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": f"Error storing memory: {str(e)}",
            }

    async def update_memory_data_tool(
        self,
        session_id: str,
        data: dict[str, Any],
        merge_strategy: Literal["replace", "merge", "deep_merge"] = "merge",
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Update working memory data with LLM-friendly response.

        This method allows LLMs to store and update structured session data
        that persists throughout the conversation.

        Args:
            session_id: The session ID to update data for
            data: Dictionary of data to store/update
            merge_strategy: How to handle existing data ("replace", "merge", "deep_merge")
            namespace: Optional namespace for the session
            user_id: Optional user ID for the session

        Returns:
            Dict with success/failure information

        Example:
            ```python
            # Store current trip planning data
            result = await client.update_memory_data_tool(
                session_id="current_session",
                data={
                    "trip_destination": "Paris",
                    "travel_dates": {"start": "2024-06-01", "end": "2024-06-07"},
                    "budget": 2000
                }
            )

            print(result["summary"])  # "Successfully updated 3 data entries"
            ```
        """
        try:
            # Update working memory data
            await self.update_working_memory_data(
                session_id=session_id,
                data_updates=data,
                namespace=namespace or self.config.default_namespace,
                merge_strategy=merge_strategy,
            )

            data_summary = ", ".join(f"{k}: {str(v)[:50]}..." for k, v in data.items())

            return {
                "success": True,
                "updated_keys": list(data.keys()),
                "merge_strategy": merge_strategy,
                "data_preview": data_summary,
                "summary": f"Successfully updated {len(data)} data entries using {merge_strategy} strategy",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": f"Error updating working memory data: {str(e)}",
            }

    @classmethod
    def get_working_memory_tool_schema(cls) -> dict[str, Any]:
        """
        Get OpenAI-compatible tool schema for reading working memory.

        Returns:
            Tool schema dictionary compatible with OpenAI tool calling format
        """
        return {
            "type": "function",
            "function": {
                "name": "get_working_memory",
                "description": "Get the current working memory state including messages, stored memories, and session data. Use this to understand what information is already stored in the current session.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    @classmethod
    def get_add_memory_tool_schema(cls) -> dict[str, Any]:
        """
        Get OpenAI-compatible tool schema for adding memories to working memory.

        Returns:
            Tool schema dictionary compatible with OpenAI tool calling format
        """
        return {
            "type": "function",
            "function": {
                "name": "add_memory_to_working_memory",
                "description": "Add important information as a structured memory to working memory. Use this to store user preferences, trip details, requirements, or other important facts that should be remembered. The memory server will automatically promote important memories to long-term storage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory content to store",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "message"],
                            "description": "Type of memory: 'episodic' (events/experiences), 'semantic' (facts/preferences), 'message' (conversation snippets)",
                        },
                        "topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional topics for categorization (e.g., ['preferences', 'budget', 'destinations'])",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional entities mentioned (e.g., ['Paris', 'hotel', 'vegetarian'])",
                        },
                    },
                    "required": ["text", "memory_type"],
                },
            },
        }

    @classmethod
    def get_update_memory_data_tool_schema(cls) -> dict[str, Any]:
        """
        Get OpenAI-compatible tool schema for updating working memory data.

        Returns:
            Tool schema dictionary compatible with OpenAI tool calling format
        """
        return {
            "type": "function",
            "function": {
                "name": "update_working_memory_data",
                "description": "Update or add structured data to working memory. Use this to store session-specific information like current trip plans, preferences, or other structured data that should persist in the session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "JSON data to store or update in working memory",
                        },
                        "merge_strategy": {
                            "type": "string",
                            "enum": ["replace", "merge", "deep_merge"],
                            "default": "merge",
                            "description": "How to handle existing data: 'replace' (overwrite), 'merge' (shallow merge), 'deep_merge' (recursive merge)",
                        },
                    },
                    "required": ["data"],
                },
            },
        }

    @classmethod
    def get_all_memory_tool_schemas(cls) -> list[dict[str, Any]]:
        """
        Get all memory-related tool schemas for easy LLM integration.

        Returns:
            List of all memory tool schemas

        Example:
            ```python
            # Get all memory tools for OpenAI
            tools = MemoryAPIClient.get_all_memory_tool_schemas()

            response = await openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            ```
        """
        return [
            cls.get_memory_search_tool_schema(),
            cls.get_working_memory_tool_schema(),
            cls.get_add_memory_tool_schema(),
            cls.get_update_memory_data_tool_schema(),
        ]

    @classmethod
    def get_all_memory_tool_schemas_anthropic(cls) -> list[dict[str, Any]]:
        """
        Get all memory-related tool schemas in Anthropic format.

        Returns:
            List of all memory tool schemas formatted for Anthropic API

        Example:
            ```python
            # Get all memory tools for Anthropic
            tools = MemoryAPIClient.get_all_memory_tool_schemas_anthropic()

            response = anthropic.messages.create(
                model="claude-3-opus-20240229",
                messages=messages,
                tools=tools,
                max_tokens=1024
            )
            ```
        """
        return [
            cls.get_memory_search_tool_schema_anthropic(),
            cls.get_working_memory_tool_schema_anthropic(),
            cls.get_add_memory_tool_schema_anthropic(),
            cls.get_update_memory_data_tool_schema_anthropic(),
        ]

    @classmethod
    def get_memory_search_tool_schema_anthropic(cls) -> dict[str, Any]:
        """Get memory search tool schema in Anthropic format."""
        openai_schema = cls.get_memory_search_tool_schema()
        return cls._convert_openai_to_anthropic_schema(openai_schema)

    @classmethod
    def get_working_memory_tool_schema_anthropic(cls) -> dict[str, Any]:
        """Get working memory tool schema in Anthropic format."""
        openai_schema = cls.get_working_memory_tool_schema()
        return cls._convert_openai_to_anthropic_schema(openai_schema)

    @classmethod
    def get_add_memory_tool_schema_anthropic(cls) -> dict[str, Any]:
        """Get add memory tool schema in Anthropic format."""
        openai_schema = cls.get_add_memory_tool_schema()
        return cls._convert_openai_to_anthropic_schema(openai_schema)

    @classmethod
    def get_update_memory_data_tool_schema_anthropic(cls) -> dict[str, Any]:
        """Get update memory data tool schema in Anthropic format."""
        openai_schema = cls.get_update_memory_data_tool_schema()
        return cls._convert_openai_to_anthropic_schema(openai_schema)

    @staticmethod
    def _convert_openai_to_anthropic_schema(
        openai_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert OpenAI tool schema to Anthropic format.

        Args:
            openai_schema: Tool schema in OpenAI format

        Returns:
            Tool schema in Anthropic format
        """
        function_def = openai_schema["function"]

        return {
            "name": function_def["name"],
            "description": function_def["description"],
            "input_schema": function_def["parameters"],
        }

    # === Function Call Resolution ===

    @staticmethod
    def parse_openai_function_call(function_call: dict[str, Any]) -> UnifiedToolCall:
        """
        Parse OpenAI legacy function call format to unified format.

        Args:
            function_call: Dict with 'name' and 'arguments' keys

        Returns:
            UnifiedToolCall object
        """
        import json

        name = function_call.get("name", "")
        arguments_str = function_call.get("arguments", "{}")

        try:
            arguments = (
                json.loads(arguments_str)
                if isinstance(arguments_str, str)
                else arguments_str
            )
        except (json.JSONDecodeError, TypeError):
            arguments = {}

        return UnifiedToolCall(
            id=None, name=name, arguments=arguments, provider="openai"
        )

    @staticmethod
    def parse_openai_tool_call(tool_call: dict[str, Any]) -> UnifiedToolCall:
        """
        Parse OpenAI tool call format to unified format.

        Args:
            tool_call: Dict with 'id', 'type', and 'function' keys

        Returns:
            UnifiedToolCall object
        """
        import json

        tool_id = tool_call.get("id", "")
        function_data = tool_call.get("function", {})
        name = function_data.get("name", "")
        arguments_str = function_data.get("arguments", "{}")

        try:
            arguments = (
                json.loads(arguments_str)
                if isinstance(arguments_str, str)
                else arguments_str
            )
        except (json.JSONDecodeError, TypeError):
            arguments = {}

        return UnifiedToolCall(
            id=tool_id, name=name, arguments=arguments, provider="openai"
        )

    @staticmethod
    def parse_anthropic_tool_use(tool_use: dict[str, Any]) -> UnifiedToolCall:
        """
        Parse Anthropic tool use format to unified format.

        Args:
            tool_use: Dict with 'id', 'name', and 'input' keys

        Returns:
            UnifiedToolCall object
        """
        return UnifiedToolCall(
            id=tool_use.get("id", ""),
            name=tool_use.get("name", ""),
            arguments=tool_use.get("input", {}),
            provider="anthropic",
        )

    @staticmethod
    def parse_tool_call(tool_call: dict[str, Any]) -> UnifiedToolCall:
        """
        Parse any tool call format to unified format.

        Auto-detects the format based on the structure and converts accordingly.

        Args:
            tool_call: Tool call in any supported format

        Returns:
            UnifiedToolCall object

        Example:
            ```python
            # OpenAI legacy format
            openai_call = {"name": "search_memory", "arguments": '{"query": "test"}'}
            unified = MemoryAPIClient.parse_tool_call(openai_call)

            # OpenAI current format
            openai_tool = {
                "id": "call_123",
                "type": "function",
                "function": {"name": "search_memory", "arguments": '{"query": "test"}'}
            }
            unified = MemoryAPIClient.parse_tool_call(openai_tool)

            # Anthropic format
            anthropic_tool = {
                "type": "tool_use",
                "id": "tool_123",
                "name": "search_memory",
                "input": {"query": "test"}
            }
            unified = MemoryAPIClient.parse_tool_call(anthropic_tool)
            ```
        """
        # Detect Anthropic format
        if tool_call.get("type") == "tool_use" and "input" in tool_call:
            return MemoryAPIClient.parse_anthropic_tool_use(tool_call)

        # Detect OpenAI current tool call format
        elif tool_call.get("type") == "function" and "function" in tool_call:
            return MemoryAPIClient.parse_openai_tool_call(tool_call)

        # Detect OpenAI legacy function call format
        elif "name" in tool_call and "arguments" in tool_call:
            return MemoryAPIClient.parse_openai_function_call(tool_call)

        # Generic format - assume it's already in a usable format
        else:
            return UnifiedToolCall(
                id=tool_call.get("id"),
                name=tool_call.get("name", ""),
                arguments=tool_call.get("arguments", {}),
                provider="generic",
            )

    async def resolve_tool_call(
        self,
        tool_call: dict[str, Any],
        session_id: str,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> ToolCallResolutionResult:
        """
        Resolve a tool call from any LLM provider format.

        This method automatically detects the tool call format (OpenAI, Anthropic, etc.)
        and resolves it appropriately. This is the recommended method for handling
        tool calls from different LLM providers.

        Args:
            tool_call: Tool call in any supported format
            session_id: Session ID for working memory operations
            namespace: Optional namespace for operations

        Returns:
            ToolCallResolutionResult with standardized response format

        Example:
            ```python
            # Works with any provider format
            result = await client.resolve_tool_call(
                tool_call=provider_tool_call,  # Any format
                session_id="session123",
            )

            if result["success"]:
                print(result["formatted_response"])
            else:
                print(f"Error: {result['error']}")
            ```
        """
        try:
            # Parse to unified format
            unified_call = self.parse_tool_call(tool_call)

            # Resolve using the unified format
            return await self.resolve_function_call(
                function_name=unified_call["name"],
                function_arguments=unified_call["arguments"],
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
            )

        except Exception as e:
            return ToolCallResolutionResult(
                success=False,
                function_name=tool_call.get("name", "unknown"),
                result=None,
                error=str(e),
                formatted_response=f"I encountered an error processing the tool call: {str(e)}",
            )

    async def resolve_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        session_id: str,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> list[ToolCallResolutionResult]:
        """
        Resolve multiple tool calls from any LLM provider format.

        Args:
            tool_calls: List of tool calls in any supported format
            session_id: Session ID for working memory operations
            namespace: Optional namespace for operations
            user_id: Optional user ID for operations

        Returns:
            List of ToolCallResolutionResult objects in the same order as input

        Example:
            ```python
            # Handle batch of tool calls from any provider
            results = await client.resolve_tool_calls(
                tool_calls=provider_tool_calls,
                session_id="session123"
            )

            for result in results:
                if result["success"]:
                    print(f"{result['function_name']}: {result['formatted_response']}")
            ```
        """
        results = []
        for tool_call in tool_calls:
            result = await self.resolve_tool_call(
                tool_call=tool_call,
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
            )
            results.append(result)

        return results

    async def resolve_function_call(
        self,
        function_name: str,
        function_arguments: str | dict[str, Any],
        session_id: str,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> ToolCallResolutionResult:
        """
        Resolve a function call for memory-related tools.

        This utility method handles all memory tool function calls with proper
        error handling, argument parsing, and response formatting. Perfect for
        LLM frameworks that need to handle function calls.

        Args:
            function_name: Name of the function to call
            function_arguments: JSON string or dict of function arguments
            session_id: Session ID for working memory operations
            namespace: Optional namespace for operations

        Returns:
            Dict with standardized response format:
            {
                "success": bool,
                "function_name": str,
                "result": Any,  # The actual function result
                "error": str | None,
                "formatted_response": str,  # Human-readable response for LLM
            }

        Example:
            ```python
            # Handle OpenAI function call
            if hasattr(response, "tool_calls"):
                for tool_call in response.tool_calls:
                    result = await client.resolve_function_call(
                        function_name=tool_call.function.name,
                        function_arguments=tool_call.function.arguments,
                        session_id="current_session"
                    )

                    if result["success"]:
                        print(result["formatted_response"])
                    else:
                        print(f"Error: {result['error']}")
            ```
        """
        import json

        # Parse arguments if they're a JSON string
        try:
            if isinstance(function_arguments, str):
                args = json.loads(function_arguments)
            else:
                args = function_arguments or {}
        except (json.JSONDecodeError, TypeError) as e:
            return ToolCallResolutionResult(
                success=False,
                function_name=function_name,
                result=None,
                error=f"Invalid function arguments: {function_arguments}. JSON decode error: {str(e)}",
                formatted_response="I encountered an error parsing the function arguments. Please try again.",
            )

        # Apply default namespace if not provided
        effective_namespace = namespace or self.config.default_namespace

        try:
            # Route to appropriate function based on name
            if function_name == "search_memory":
                result = await self._resolve_search_memory(args)

            elif function_name == "get_working_memory":
                result = await self._resolve_get_working_memory(
                    session_id, effective_namespace, user_id
                )

            elif function_name == "add_memory_to_working_memory":
                result = await self._resolve_add_memory(
                    args, session_id, effective_namespace, user_id
                )

            elif function_name == "update_working_memory_data":
                result = await self._resolve_update_memory_data(
                    args, session_id, effective_namespace, user_id
                )

            else:
                return ToolCallResolutionResult(
                    success=False,
                    function_name=function_name,
                    result=None,
                    error=f"Unknown function: {function_name}",
                    formatted_response=f"I don't know how to handle the function '{function_name}'. Please check the function name.",
                )

            return ToolCallResolutionResult(
                success=True,
                function_name=function_name,
                result=result,
                error=None,
                formatted_response=result.get("summary", str(result))
                if isinstance(result, dict)
                else str(result),
            )

        except Exception as e:
            return ToolCallResolutionResult(
                success=False,
                function_name=function_name,
                result=None,
                error=str(e),
                formatted_response=f"I encountered an error while executing {function_name}: {str(e)}",
            )

    async def _resolve_search_memory(self, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve search_memory function call."""
        query = args.get("query", "")
        if not query:
            raise ValueError("Query parameter is required for memory search")

        topics = args.get("topics")
        entities = args.get("entities")
        memory_type = args.get("memory_type")
        max_results = args.get("max_results", 5)
        min_relevance = args.get("min_relevance")
        user_id = args.get("user_id")

        return await self.search_memory_tool(
            query=query,
            topics=topics,
            entities=entities,
            memory_type=memory_type,
            max_results=max_results,
            min_relevance=min_relevance,
            user_id=user_id,
        )

    async def _resolve_get_working_memory(
        self, session_id: str, namespace: str | None, user_id: str | None = None
    ) -> dict[str, Any]:
        """Resolve get_working_memory function call."""
        return await self.get_working_memory_tool(
            session_id=session_id,
            namespace=namespace,
            user_id=user_id,
        )

    async def _resolve_add_memory(
        self,
        args: dict[str, Any],
        session_id: str,
        namespace: str | None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve add_memory_to_working_memory function call."""
        text = args.get("text", "")
        if not text:
            raise ValueError("Text parameter is required for adding memory")

        memory_type = args.get("memory_type", "semantic")
        topics = args.get("topics")
        entities = args.get("entities")

        return await self.add_memory_tool(
            session_id=session_id,
            text=text,
            memory_type=memory_type,
            topics=topics,
            entities=entities,
            namespace=namespace,
            user_id=user_id,
        )

    async def _resolve_update_memory_data(
        self,
        args: dict[str, Any],
        session_id: str,
        namespace: str | None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve update_working_memory_data function call."""
        data = args.get("data", {})
        if not data:
            raise ValueError(
                "Data parameter is required for updating working memory data"
            )

        merge_strategy = args.get("merge_strategy", "merge")

        return await self.update_memory_data_tool(
            session_id=session_id,
            data=data,
            merge_strategy=merge_strategy,
            namespace=namespace,
            user_id=user_id,
        )

    async def resolve_function_calls(
        self,
        function_calls: list[dict[str, Any]],
        session_id: str,
        namespace: str | None = None,
        user_id: str | None = None,
    ) -> list[ToolCallResolutionResult]:
        """
        Resolve multiple function calls in batch.

        Args:
            function_calls: List of function call dicts with 'name' and 'arguments' keys
            session_id: Session ID for working memory operations
            namespace: Optional namespace for operations
            user_id: Optional user ID for operations

        Returns:
            List of resolution results in the same order as input

        Example:
            ```python
            # Handle multiple function calls
            calls = [
                {"name": "search_memory", "arguments": {"query": "user preferences"}},
                {"name": "get_working_memory", "arguments": {}},
            ]

            results = await client.resolve_function_calls(calls, "session123")
            for result in results:
                if result["success"]:
                    print(f"{result['function_name']}: {result['formatted_response']}")
            ```
        """
        results = []
        for call in function_calls:
            function_name = call.get("name", "")
            function_arguments = call.get("arguments", {})

            result = await self.resolve_function_call(
                function_name=function_name,
                function_arguments=function_arguments,
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
            )
            results.append(result)

        return results

    # === Memory Lifecycle Management ===

    async def promote_working_memories_to_long_term(
        self,
        session_id: str,
        memory_ids: list[str] | None = None,
        namespace: str | None = None,
    ) -> AckResponse:
        """
        Explicitly promote specific working memories to long-term storage.

        Note: Memory promotion normally happens automatically when working memory
        is saved. This method is for cases where you need manual control over
        the promotion timing or want to promote specific memories immediately.

        Args:
            session_id: The session containing memories to promote
            memory_ids: Specific memory IDs to promote (if None, promotes all unpromoted)
            namespace: Optional namespace filter

        Returns:
            Acknowledgement of promotion operation
        """
        # Get current working memory
        working_memory = await self.get_working_memory(
            session_id=session_id, namespace=namespace
        )

        # Filter memories if specific IDs are requested
        memories_to_promote = working_memory.memories
        if memory_ids is not None:
            memories_to_promote = [
                memory for memory in working_memory.memories if memory.id in memory_ids
            ]

        if not memories_to_promote:
            return AckResponse(status="ok")

        # Create long-term memories
        return await self.create_long_term_memory(memories_to_promote)

    # === Batch Operations ===

    async def bulk_create_long_term_memories(
        self,
        memory_batches: list[list[ClientMemoryRecord | MemoryRecord]],
        batch_size: int = 100,
        delay_between_batches: float = 0.1,
    ) -> list[AckResponse]:
        """
        Create multiple batches of memories with proper rate limiting.

        Args:
            memory_batches: List of memory record batches
            batch_size: Maximum memories per batch request
            delay_between_batches: Delay in seconds between batches

        Returns:
            List of acknowledgement responses for each batch
        """
        results = []

        for batch in memory_batches:
            # Split large batches into smaller chunks
            for i in range(0, len(batch), batch_size):
                chunk = batch[i : i + batch_size]
                response = await self.create_long_term_memory(chunk)
                results.append(response)

                # Rate limiting delay
                if delay_between_batches > 0:
                    await asyncio.sleep(delay_between_batches)

        return results

    # === Pagination Utilities ===

    async def search_all_long_term_memories(
        self,
        text: str,
        session_id: SessionId | dict[str, Any] | None = None,
        namespace: Namespace | dict[str, Any] | None = None,
        topics: Topics | dict[str, Any] | None = None,
        entities: Entities | dict[str, Any] | None = None,
        created_at: CreatedAt | dict[str, Any] | None = None,
        last_accessed: LastAccessed | dict[str, Any] | None = None,
        user_id: UserId | dict[str, Any] | None = None,
        distance_threshold: float | None = None,
        memory_type: MemoryType | dict[str, Any] | None = None,
        batch_size: int = 50,
    ) -> AsyncIterator[MemoryRecord]:
        """
        Auto-paginating search that yields all matching long-term memory results.

        Automatically handles pagination to retrieve all results without
        requiring manual offset management.

        Args:
            text: Search query text
            session_id: Optional session ID filter
            namespace: Optional namespace filter
            topics: Optional topics filter
            entities: Optional entities filter
            created_at: Optional creation date filter
            last_accessed: Optional last accessed date filter
            user_id: Optional user ID filter
            distance_threshold: Optional distance threshold
            memory_type: Optional memory type filter
            batch_size: Number of results to fetch per API call

        Yields:
            Individual memory records from all result pages
        """
        offset = 0
        while True:
            results = await self.search_long_term_memory(
                text=text,
                session_id=session_id,
                namespace=namespace,
                topics=topics,
                entities=entities,
                created_at=created_at,
                last_accessed=last_accessed,
                user_id=user_id,
                distance_threshold=distance_threshold,
                memory_type=memory_type,
                limit=batch_size,
                offset=offset,
            )

            if not results.memories:
                break

            for memory in results.memories:
                yield memory

            # If we got fewer results than batch_size, we've reached the end
            if len(results.memories) < batch_size:
                break

            offset += batch_size

    def validate_memory_record(self, memory: ClientMemoryRecord | MemoryRecord) -> None:
        """
        Validate memory record before sending to server.

        Checks:
        - Required fields are present
        - Memory type is valid
        - Dates are properly formatted
        - Text content is not empty
        - ID format is valid

        Raises:
            MemoryValidationError: If validation fails with descriptive message
        """
        if not memory.text or not memory.text.strip():
            raise MemoryValidationError("Memory text cannot be empty")

        if memory.memory_type not in [
            "episodic",
            "semantic",
            "message",
        ]:
            raise MemoryValidationError(f"Invalid memory type: {memory.memory_type}")

        if memory.id and not self._is_valid_ulid(memory.id):
            raise MemoryValidationError(f"Invalid ID format: {memory.id}")

        # created_at is validated by Pydantic

        # last_accessed is validated by Pydantic

    def validate_search_filters(self, **filters: Any) -> None:
        """Validate search filter parameters before API call."""
        valid_filter_keys = {
            "session_id",
            "namespace",
            "topics",
            "entities",
            "created_at",
            "last_accessed",
            "user_id",
            "distance_threshold",
            "memory_type",
            "limit",
            "offset",
        }

        for key in filters:
            if key not in valid_filter_keys:
                raise MemoryValidationError(f"Invalid filter key: {key}")

        if "limit" in filters and (
            not isinstance(filters["limit"], int) or filters["limit"] <= 0
        ):
            raise MemoryValidationError("Limit must be a positive integer")

        if "offset" in filters and (
            not isinstance(filters["offset"], int) or filters["offset"] < 0
        ):
            raise MemoryValidationError("Offset must be a non-negative integer")

        if "distance_threshold" in filters and (
            not isinstance(filters["distance_threshold"], int | float)
            or filters["distance_threshold"] < 0
        ):
            raise MemoryValidationError(
                "Distance threshold must be a non-negative number"
            )

    _ULID_REGEX = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}")

    def _is_valid_ulid(self, ulid_str: str) -> bool:
        """Return True if a string looks like a valid Crockford-base32 ULID."""
        return bool(self._ULID_REGEX.fullmatch(ulid_str))

    async def update_working_memory_data(
        self,
        session_id: str,
        data_updates: dict[str, Any],
        namespace: str | None = None,
        merge_strategy: Literal["replace", "merge", "deep_merge"] = "merge",
        user_id: str | None = None,
    ) -> WorkingMemoryResponse:
        """
        Update specific data fields in working memory without replacing everything.

        Args:
            session_id: Target session
            data_updates: Dictionary of updates to apply
            namespace: Optional namespace
            merge_strategy: How to handle existing data
            user_id: Optional user ID for the session

        Returns:
            WorkingMemoryResponse with updated memory
        """
        # Get existing memory
        existing_memory = await self.get_working_memory(
            session_id=session_id, namespace=namespace, user_id=user_id
        )

        # Determine final data based on merge strategy
        if existing_memory and existing_memory.data:
            if merge_strategy == "replace":
                final_data = data_updates
            elif merge_strategy == "merge":
                final_data = {**existing_memory.data, **data_updates}
            elif merge_strategy == "deep_merge":
                final_data = self._deep_merge_dicts(existing_memory.data, data_updates)
            else:
                raise MemoryValidationError(f"Invalid merge strategy: {merge_strategy}")
        else:
            final_data = data_updates

        # Create updated working memory
        working_memory = WorkingMemory(
            session_id=session_id,
            namespace=namespace or self.config.default_namespace,
            messages=existing_memory.messages if existing_memory else [],
            memories=existing_memory.memories if existing_memory else [],
            data=final_data,
            context=existing_memory.context if existing_memory else None,
            user_id=existing_memory.user_id if existing_memory else None,
        )

        return await self.put_working_memory(session_id, working_memory)

    async def append_messages_to_working_memory(
        self,
        session_id: str,
        messages: list[dict[str, Any]],  # Expect proper message dicts
        namespace: str | None = None,
        model_name: str | None = None,
        context_window_max: int | None = None,
        user_id: str | None = None,
    ) -> WorkingMemoryResponse:
        """
        Append new messages to existing working memory.

        More efficient than retrieving, modifying, and setting full memory.

        Args:
            session_id: Target session
            messages: List of message dictionaries with 'role' and 'content' keys
            namespace: Optional namespace
            model_name: Optional model name for token-based summarization
            context_window_max: Optional direct specification of context window max tokens

        Returns:
            WorkingMemoryResponse with updated memory (potentially summarized if token limit exceeded)
        """
        # Get existing memory
        existing_memory = await self.get_working_memory(
            session_id=session_id, namespace=namespace, user_id=user_id
        )

        # Validate new messages have required structure
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(
                    "All messages must be dictionaries with 'role' and 'content' keys"
                )

        # Get existing messages (already in proper dict format from get_working_memory)
        existing_messages = []
        if existing_memory and existing_memory.messages:
            existing_messages = existing_memory.messages

        final_messages = existing_messages + messages

        # Create updated working memory
        working_memory = WorkingMemory(
            session_id=session_id,
            namespace=namespace or self.config.default_namespace,
            messages=final_messages,
            memories=existing_memory.memories if existing_memory else [],
            data=existing_memory.data if existing_memory else {},
            context=existing_memory.context if existing_memory else None,
            user_id=existing_memory.user_id if existing_memory else None,
        )

        return await self.put_working_memory(
            session_id,
            working_memory,
            model_name=model_name,
            context_window_max=context_window_max,
        )

    async def memory_prompt(
        self,
        query: str,
        session_id: str | None = None,
        namespace: str | None = None,
        window_size: int | None = None,
        model_name: str | None = None,
        context_window_max: int | None = None,
        long_term_search: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Hydrate a user query with memory context and return a prompt ready to send to an LLM.

        Args:
            query: The input text to find relevant context for
            session_id: Optional session ID to include session messages
            namespace: Optional namespace for the session
            window_size: Optional number of messages to include
            model_name: Optional model name to determine context window size
            context_window_max: Optional direct specification of context window tokens
            long_term_search: Optional search parameters for long-term memory
            user_id: Optional user ID for the session

        Returns:
            Dict with messages hydrated with relevant memory context

        Example:
            ```python
            # Create a prompt with both session and long-term memory context
            prompt = await client.memory_prompt(
                query="What are my UI preferences?",
                session_id="current_session",
                window_size=10,
                long_term_search={
                    "topics": {"any": ["preferences", "ui"]},
                    "limit": 5
                }
            )

            # Send to your LLM
            messages = prompt.get("messages", [])
            # Add the user query and send to OpenAI, Claude, etc.
            ```
        """
        payload: dict[str, Any] = {"query": query}

        # Add session parameters if provided
        if session_id is not None:
            session_params: dict[str, Any] = {"session_id": session_id}
            if namespace is not None:
                session_params["namespace"] = namespace
            elif self.config.default_namespace is not None:
                session_params["namespace"] = self.config.default_namespace
            if window_size is not None:
                session_params["window_size"] = str(window_size)
            # Use provided model_name or fall back to config default
            effective_model_name = model_name or self.config.default_model_name
            if effective_model_name is not None:
                session_params["model_name"] = effective_model_name

            # Use provided context_window_max or fall back to config default
            effective_context_window_max = (
                context_window_max or self.config.default_context_window_max
            )
            if effective_context_window_max is not None:
                session_params["context_window_max"] = str(effective_context_window_max)
            if user_id is not None:
                session_params["user_id"] = user_id
            payload["session"] = session_params

        # Add long-term search parameters if provided
        if long_term_search is not None:
            payload["long_term_search"] = long_term_search

        try:
            response = await self._client.post(
                "/v1/memory/prompt",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict):
                return result
            return {"response": result}
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e.response)
            raise

    async def hydrate_memory_prompt(
        self,
        query: str,
        session_id: dict[str, Any] | None = None,
        namespace: dict[str, Any] | None = None,
        topics: dict[str, Any] | None = None,
        entities: dict[str, Any] | None = None,
        created_at: dict[str, Any] | None = None,
        last_accessed: dict[str, Any] | None = None,
        user_id: dict[str, Any] | None = None,
        distance_threshold: float | None = None,
        memory_type: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Hydrate a user query with long-term memory context using filters.

        This is a convenience method that creates a memory prompt using only
        long-term memory search with the specified filters.

        Args:
            query: The input text to find relevant context for
            session_id: Optional session ID filter (as dict)
            namespace: Optional namespace filter (as dict)
            topics: Optional topics filter (as dict)
            entities: Optional entities filter (as dict)
            created_at: Optional creation date filter (as dict)
            last_accessed: Optional last accessed date filter (as dict)
            user_id: Optional user ID filter (as dict)
            distance_threshold: Optional distance threshold
            memory_type: Optional memory type filter (as dict)
            limit: Maximum number of long-term memories to include

        Returns:
            Dict with messages hydrated with relevant long-term memories
        """
        # Build long-term search parameters
        long_term_search: dict[str, Any] = {"limit": limit}

        if session_id is not None:
            long_term_search["session_id"] = session_id
        if namespace is not None:
            long_term_search["namespace"] = namespace
        elif self.config.default_namespace is not None:
            long_term_search["namespace"] = {"eq": self.config.default_namespace}
        if topics is not None:
            long_term_search["topics"] = topics
        if entities is not None:
            long_term_search["entities"] = entities
        if created_at is not None:
            long_term_search["created_at"] = created_at
        if last_accessed is not None:
            long_term_search["last_accessed"] = last_accessed
        if user_id is not None:
            long_term_search["user_id"] = user_id
        if distance_threshold is not None:
            long_term_search["distance_threshold"] = distance_threshold
        if memory_type is not None:
            long_term_search["memory_type"] = memory_type

        return await self.memory_prompt(
            query=query,
            long_term_search=long_term_search,
        )

    def _deep_merge_dicts(
        self, base: dict[str, Any], updates: dict[str, Any]
    ) -> dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = base.copy()
        for key, value in updates.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result


# Helper function to create a memory client
async def create_memory_client(
    base_url: str,
    timeout: float = 30.0,
    default_namespace: str | None = None,
    default_model_name: str | None = None,
    default_context_window_max: int | None = None,
) -> MemoryAPIClient:
    """
    Create and initialize a Memory API Client.

    Args:
        base_url: Base URL of the memory server (e.g., 'http://localhost:8000')
        timeout: Request timeout in seconds (default: 30.0)
        default_namespace: Optional default namespace to use for operations
        default_model_name: Optional default model name for auto-summarization
        default_context_window_max: Optional default context window limit for auto-summarization

    Returns:
        Initialized MemoryAPIClient instance

    Raises:
        MemoryClientError: If unable to connect to the server

    Example:
        ```python
        # Basic client setup
        client = await create_memory_client("http://localhost:8000")

        # With custom namespace and timeout
        client = await create_memory_client(
            base_url="http://memory-server.example.com",
            timeout=60.0,
            default_namespace="my_app"
        )

        # With model configuration for auto-summarization
        client = await create_memory_client(
            base_url="http://localhost:8000",
            default_model_name="gpt-4o",
            default_namespace="travel_agent"
        )

        # Use as context manager
        async with await create_memory_client("http://localhost:8000") as client:
            results = await client.search_memory_tool(
                query="user preferences",
                topics=["ui", "settings"]
            )
        ```
    """
    config = MemoryClientConfig(
        base_url=base_url,
        timeout=timeout,
        default_namespace=default_namespace,
        default_model_name=default_model_name,
        default_context_window_max=default_context_window_max,
    )
    client = MemoryAPIClient(config)

    # Test connection with a health check
    try:
        await client.health_check()
    except Exception as e:
        await client.close()
        raise MemoryClientError(
            f"Failed to connect to memory server at {base_url}: {e}"
        ) from e

    return client
