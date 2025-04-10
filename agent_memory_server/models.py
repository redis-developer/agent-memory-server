import logging

from pydantic import BaseModel, Field

from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    LastAccessed,
    Namespace,
    SessionId,
    Topics,
    UserId,
)


logger = logging.getLogger(__name__)

JSONTypes = str | float | int | bool | list | dict


class MemoryMessage(BaseModel):
    """A message in the memory system"""

    role: str
    content: str


class SessionMemory(BaseModel):
    """A session's memory"""

    messages: list[MemoryMessage]
    session_id: str | None = None
    context: str | None = None
    user_id: str | None = None
    last_accessed: int | None = None
    created_at: int | None = None
    tokens: int | None = 0
    namespace: str | None = None


class LongTermMemory(BaseModel):
    """A long-term memory"""

    text: str
    id_: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    last_accessed: int | None = None
    created_at: int | None = None
    namespace: str | None = None
    topics: list[str] | None = None
    entities: list[str] | None = None


class SessionMemoryResponse(SessionMemory):
    """Response containing a session's memory"""


class SessionListResponse(BaseModel):
    """Response containing a list of sessions"""

    sessions: list[str]
    total: int


class SearchPayload(BaseModel):
    """Payload for long-term memory search"""

    text: str
    session_id: SessionId | None = None
    namespace: Namespace | None = None
    topics: Topics | None = None
    entities: Entities | None = None
    created_at: CreatedAt | None = None
    last_accessed: LastAccessed | None = None
    user_id: UserId | None = None
    distance_threshold: float | None = None
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @classmethod
    def create_with_primitives(
        cls,
        text: str,
        # Session ID filters
        session_id: str | None = None,
        session_id_ne: str | None = None,
        session_ids_any: list[str] | None = None,
        session_ids_all: list[str] | None = None,
        # Namespace filters
        namespace: str | None = None,
        namespace_ne: str | None = None,
        namespaces_any: list[str] | None = None,
        namespaces_all: list[str] | None = None,
        # Topics filters
        topics: list[str] | None = None,
        topics_ne: str | None = None,
        topics_any: list[str] | None = None,
        topics_all: list[str] | None = None,
        # Entities filters
        entities: list[str] | None = None,
        entities_ne: str | None = None,
        entities_any: list[str] | None = None,
        entities_all: list[str] | None = None,
        # User ID filters
        user_id: str | None = None,
        user_id_ne: str | None = None,
        user_ids_any: list[str] | None = None,
        user_ids_all: list[str] | None = None,
        # Created at filters
        created_at_gt: int | None = None,
        created_at_lt: int | None = None,
        created_at_gte: int | None = None,
        created_at_lte: int | None = None,
        created_at_eq: int | None = None,
        created_at_ne: int | None = None,
        created_at_between: list[float] | None = None,
        # Last accessed filters
        last_accessed_gt: int | None = None,
        last_accessed_lt: int | None = None,
        last_accessed_gte: int | None = None,
        last_accessed_lte: int | None = None,
        last_accessed_eq: int | None = None,
        last_accessed_ne: int | None = None,
        last_accessed_between: list[float] | None = None,
        # Other filters
        distance_threshold: float | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> "SearchPayload":
        """Create a SearchPayload from primitive values

        All tag filters (session_id, namespace, topics, entities, user_id) support:
        - eq: Equals this value
        - ne: Not equals this value
        - any: Contains any of these values
        - all: Contains all of these values

        All numeric filters (created_at, last_accessed) support:
        - gt: Greater than
        - lt: Less than
        - gte: Greater than or equal
        - lte: Less than or equal
        - eq: Equals
        - ne: Not equals
        - between: Between two values (inclusive)
        """
        payload = {
            "text": text,
            "distance_threshold": distance_threshold,
            "limit": limit,
            "offset": offset,
        }

        # Handle SessionId filter
        session_id_args = {}
        if session_id is not None:
            session_id_args["eq"] = session_id
        if session_id_ne is not None:
            session_id_args["ne"] = session_id_ne
        if session_ids_any is not None:
            session_id_args["any"] = session_ids_any
        if session_ids_all is not None:
            session_id_args["all"] = session_ids_all
        if session_id_args:
            payload["session_id"] = SessionId(**session_id_args)

        # Handle Namespace filter
        namespace_args = {}
        if namespace is not None:
            namespace_args["eq"] = namespace
        if namespace_ne is not None:
            namespace_args["ne"] = namespace_ne
        if namespaces_any is not None:
            namespace_args["any"] = namespaces_any
        if namespaces_all is not None:
            namespace_args["all"] = namespaces_all
        if namespace_args:
            payload["namespace"] = Namespace(**namespace_args)

        # Handle Topics filter
        topics_args = {}
        if topics is not None:
            topics_args["any"] = topics  # Legacy support
        if topics_ne is not None:
            topics_args["ne"] = topics_ne
        if topics_any is not None:
            topics_args["any"] = topics_any
        if topics_all is not None:
            topics_args["all"] = topics_all
        if topics_args:
            payload["topics"] = Topics(**topics_args)

        # Handle Entities filter
        entities_args = {}
        if entities is not None:
            entities_args["any"] = entities  # Legacy support
        if entities_ne is not None:
            entities_args["ne"] = entities_ne
        if entities_any is not None:
            entities_args["any"] = entities_any
        if entities_all is not None:
            entities_args["all"] = entities_all
        if entities_args:
            payload["entities"] = Entities(**entities_args)

        # Handle UserId filter
        user_id_args = {}
        if user_id is not None:
            user_id_args["eq"] = user_id
        if user_id_ne is not None:
            user_id_args["ne"] = user_id_ne
        if user_ids_any is not None:
            user_id_args["any"] = user_ids_any
        if user_ids_all is not None:
            user_id_args["all"] = user_ids_all
        if user_id_args:
            payload["user_id"] = UserId(**user_id_args)

        # Handle CreatedAt filter
        created_at_args = {}
        if created_at_gt is not None:
            created_at_args["gt"] = created_at_gt
        if created_at_lt is not None:
            created_at_args["lt"] = created_at_lt
        if created_at_gte is not None:
            created_at_args["gte"] = created_at_gte
        if created_at_lte is not None:
            created_at_args["lte"] = created_at_lte
        if created_at_eq is not None:
            created_at_args["eq"] = created_at_eq
        if created_at_ne is not None:
            created_at_args["ne"] = created_at_ne
        if created_at_between is not None:
            created_at_args["between"] = created_at_between
        if created_at_args:
            payload["created_at"] = CreatedAt(**created_at_args)

        # Handle LastAccessed filter
        last_accessed_args = {}
        if last_accessed_gt is not None:
            last_accessed_args["gt"] = last_accessed_gt
        if last_accessed_lt is not None:
            last_accessed_args["lt"] = last_accessed_lt
        if last_accessed_gte is not None:
            last_accessed_args["gte"] = last_accessed_gte
        if last_accessed_lte is not None:
            last_accessed_args["lte"] = last_accessed_lte
        if last_accessed_eq is not None:
            last_accessed_args["eq"] = last_accessed_eq
        if last_accessed_ne is not None:
            last_accessed_args["ne"] = last_accessed_ne
        if last_accessed_between is not None:
            last_accessed_args["between"] = last_accessed_between
        if last_accessed_args:
            payload["last_accessed"] = LastAccessed(**last_accessed_args)

        return cls(**payload)

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

        return filters


class HealthCheckResponse(BaseModel):
    """Response for health check endpoint"""

    now: int


class AckResponse(BaseModel):
    """Generic acknowledgement response"""

    status: str


class LongTermMemoryResult(LongTermMemory):
    """Result from a long-term memory search"""

    dist: float


class LongTermMemoryResults(BaseModel):
    """Results from a long-term memory search"""

    memories: list[LongTermMemoryResult]
    total: int
    next_offset: int | None = None


class LongTermMemoryResultsResponse(LongTermMemoryResults):
    """Response containing long-term memory search results"""


class CreateLongTermMemoryPayload(BaseModel):
    """Payload for creating a long-term memory"""

    memories: list[LongTermMemory]


class GetSessionsQuery(BaseModel):
    """Query parameters for getting sessions"""

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    namespace: str | None = None
