import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel, Field
from ulid import ULID

from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    EventDate,
    LastAccessed,
    MemoryType,
    Namespace,
    SessionId,
    Topics,
    UserId,
)


logger = logging.getLogger(__name__)

JSONTypes = str | float | int | bool | list | dict


class MemoryTypeEnum(str, Enum):
    """Enum for memory types with string values"""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    MESSAGE = "message"


# These should match the keys in MODEL_CONFIGS
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


class MemoryMessage(BaseModel):
    """A message in the memory system"""

    role: str
    content: str
    id: str = Field(
        default_factory=lambda: str(ULID()),
        description="Unique identifier for the message (auto-generated if not provided)",
    )
    persisted_at: datetime | None = Field(
        default=None,
        description="Server-assigned timestamp when message was persisted to long-term storage",
    )
    discrete_memory_extracted: Literal["t", "f"] = Field(
        default="f",
        description="Whether memory extraction has run for this message",
    )


class SessionListResponse(BaseModel):
    """Response containing a list of sessions"""

    sessions: list[str]
    total: int


class MemoryRecord(BaseModel):
    """A memory record"""

    id: str = Field(description="Client-provided ID for deduplication and overwrites")
    text: str
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for the memory record",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for the memory record",
    )
    namespace: str | None = Field(
        default=None,
        description="Optional namespace for the memory record",
    )
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the memory was last accessed",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the memory was created",
    )
    updated_at: datetime = Field(
        description="Datetime when the memory was last updated",
        default_factory=lambda: datetime.now(UTC),
    )
    pinned: bool = Field(
        default=False,
        description="Whether this memory is pinned and should not be auto-deleted",
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this memory has been accessed (best-effort, rate-limited)",
    )
    topics: list[str] | None = Field(
        default=None,
        description="Optional topics for the memory record",
    )
    entities: list[str] | None = Field(
        default=None,
        description="Optional entities for the memory record",
    )
    memory_hash: str | None = Field(
        default=None,
        description="Hash representation of the memory for deduplication",
    )
    discrete_memory_extracted: Literal["t", "f"] = Field(
        default="f",
        description="Whether memory extraction has run for this memory",
    )
    memory_type: MemoryTypeEnum = Field(
        default=MemoryTypeEnum.MESSAGE,
        description="Type of memory",
    )
    persisted_at: datetime | None = Field(
        default=None,
        description="Server-assigned timestamp when memory was persisted to long-term storage",
    )
    extracted_from: list[str] | None = Field(
        default=None,
        description="List of message IDs that this memory was extracted from",
    )
    event_date: datetime | None = Field(
        default=None,
        description="Date/time when the event described in this memory occurred (primarily for episodic memories)",
    )


class ExtractedMemoryRecord(MemoryRecord):
    """A memory record that has already been extracted (e.g., explicit memories from API/MCP)"""

    discrete_memory_extracted: Literal["t", "f"] = Field(
        default="t",
        description="Whether memory extraction has run for this memory",
    )
    memory_type: MemoryTypeEnum = Field(
        default=MemoryTypeEnum.SEMANTIC,
        description="Type of memory",
    )


class ClientMemoryRecord(MemoryRecord):
    """A memory record with a client-provided ID"""

    id: str = Field(
        default_factory=lambda: str(ULID()),
        description="Client-provided ID for deduplication and overwrites",
    )


class WorkingMemory(BaseModel):
    """Working memory for a session - contains both messages and structured memory records"""

    # Support both message-based memory (conversation) and structured memory records
    messages: list[MemoryMessage] = Field(
        default_factory=list,
        description="Conversation messages (role/content pairs)",
    )
    memories: list[MemoryRecord | ClientMemoryRecord] = Field(
        default_factory=list,
        description="Structured memory records for promotion to long-term storage",
    )

    # Arbitrary JSON data storage (separate from memories)
    data: dict[str, JSONTypes] | None = Field(
        default=None,
        description="Arbitrary JSON data storage (key-value pairs)",
    )

    # Session context and metadata (moved from SessionMemory)
    context: str | None = Field(
        default=None,
        description="Optional summary of past session messages",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for the working memory",
    )
    tokens: int = Field(
        default=0,
        description="Optional number of tokens in the working memory",
    )

    # Required session scoping
    session_id: str
    namespace: str | None = Field(
        default=None,
        description="Optional namespace for the working memory",
    )

    # TTL and timestamps
    ttl_seconds: int | None = Field(
        default=None,  # Persistent by default
        description="TTL for the working memory in seconds",
    )
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the working memory was last accessed",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the working memory was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Datetime when the working memory was last updated",
    )


class WorkingMemoryResponse(WorkingMemory):
    """Response containing working memory"""

    context_percentage_total_used: float | None = Field(
        default=None,
        description="Percentage of total context window currently used (0-100)",
    )
    context_percentage_until_summarization: float | None = Field(
        default=None,
        description="Percentage until auto-summarization triggers (0-100, reaches 100% at summarization threshold)",
    )


class WorkingMemoryRequest(BaseModel):
    """Request parameters for working memory operations"""

    session_id: str
    namespace: str | None = None
    user_id: str | None = None
    model_name: ModelNameLiteral | None = None
    context_window_max: int | None = None


class AckResponse(BaseModel):
    """Generic acknowledgement response"""

    status: str


class MemoryRecordResult(MemoryRecord):
    """Result from a memory search"""

    dist: float


class MemoryRecordResults(BaseModel):
    """Results from a memory search"""

    memories: list[MemoryRecordResult]
    total: int
    next_offset: int | None = None


class MemoryRecordResultsResponse(MemoryRecordResults):
    """Response containing memory search results"""


class CreateMemoryRecordRequest(BaseModel):
    """Payload for creating memory records"""

    memories: list[ExtractedMemoryRecord]


class GetSessionsQuery(BaseModel):
    """Query parameters for getting sessions"""

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    namespace: str | None = None
    user_id: str | None = None


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""

    now: int


class SearchRequest(BaseModel):
    """Payload for long-term memory search"""

    text: str | None = Field(
        default=None,
        description="Optional text to use for a semantic search",
    )
    session_id: SessionId | None = Field(
        default=None,
        description="Optional session ID to filter by",
    )
    namespace: Namespace | None = Field(
        default=None,
        description="Optional namespace to filter by",
    )
    topics: Topics | None = Field(
        default=None,
        description="Optional topics to filter by",
    )
    entities: Entities | None = Field(
        default=None,
        description="Optional entities to filter by",
    )
    created_at: CreatedAt | None = Field(
        default=None,
        description="Optional created at timestamp to filter by",
    )
    last_accessed: LastAccessed | None = Field(
        default=None,
        description="Optional last accessed timestamp to filter by",
    )
    user_id: UserId | None = Field(
        default=None,
        description="Optional user ID to filter by",
    )
    distance_threshold: float | None = Field(
        default=None,
        description="Optional distance threshold to filter by",
    )
    memory_type: MemoryType | None = Field(
        default=None,
        description="Optional memory type to filter by",
    )
    event_date: EventDate | None = Field(
        default=None,
        description="Optional event date to filter by (for episodic memories)",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Optional limit on the number of results",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Optional offset",
    )

    # Recency re-ranking controls (optional)
    recency_boost: bool | None = Field(
        default=None,
        description="Enable recency-aware re-ranking (defaults to enabled if None)",
    )
    recency_w_sem: float | None = Field(
        default=None, description="Weight for semantic similarity"
    )
    recency_w_recency: float | None = Field(
        default=None, description="Weight for recency score"
    )
    recency_wf: float | None = Field(
        default=None, description="Weight for freshness component"
    )
    recency_wa: float | None = Field(
        default=None, description="Weight for novelty (age) component"
    )
    recency_half_life_last_access_days: float | None = Field(
        default=None, description="Half-life (days) for last_accessed decay"
    )
    recency_half_life_created_days: float | None = Field(
        default=None, description="Half-life (days) for created_at decay"
    )
    # Server-side recency rerank (Redis-only path) toggle
    server_side_recency: bool | None = Field(
        default=None,
        description="If true, attempt server-side recency-aware re-ranking when supported by backend",
    )

    def get_filters(self):
        """Get all filter objects as a dictionary"""
        filters = {}

        if self.session_id is not None:
            filters["session_id"] = self.session_id

        if self.namespace is not None:
            filters["namespace"] = self.namespace

        if self.topics is not None:
            filters["topics"] = self.topics

        if self.entities is not None:
            filters["entities"] = self.entities

        if self.user_id is not None:
            filters["user_id"] = self.user_id

        if self.created_at is not None:
            filters["created_at"] = self.created_at

        if self.last_accessed is not None:
            filters["last_accessed"] = self.last_accessed

        if self.memory_type is not None:
            filters["memory_type"] = self.memory_type

        if self.event_date is not None:
            filters["event_date"] = self.event_date

        return filters


class MemoryPromptRequest(BaseModel):
    query: str
    session: WorkingMemoryRequest | None = None
    long_term_search: SearchRequest | bool | None = None


class SystemMessage(base.Message):
    """A system message"""

    role: Literal["system"] = "system"


class UserMessage(base.Message):
    """A user message"""

    role: Literal["user"] = "user"


class MemoryPromptResponse(BaseModel):
    messages: list[base.Message | SystemMessage]


class LenientMemoryRecord(ExtractedMemoryRecord):
    """A memory record that can be created without an ID"""

    id: str | None = Field(default_factory=lambda: str(ULID()))


class DeleteMemoryRecordRequest(BaseModel):
    """Payload for deleting memory records"""

    ids: list[str]
