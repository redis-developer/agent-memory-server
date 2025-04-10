from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    LastAccessed,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from agent_memory_server.models import (
    LongTermMemoryResult,
    MemoryMessage,
    SearchPayload,
    SessionMemory,
    SessionMemoryResponse,
)


class TestModels:
    def test_memory_message(self):
        """Test MemoryMessage model"""
        msg = MemoryMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_session_memory(self):
        """Test SessionMemory model"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        # Test without any optional fields
        payload = SessionMemory(messages=messages)
        assert payload.messages == messages
        assert payload.context is None
        assert payload.user_id is None
        assert payload.session_id is None
        assert payload.namespace is None
        assert payload.tokens == 0
        assert payload.last_accessed is None
        assert payload.created_at is None

        # Test with all fields
        payload = SessionMemory(
            messages=messages,
            context="Previous conversation summary",
            user_id="user_id",
            session_id="session_id",
            namespace="namespace",
            tokens=100,
            last_accessed=100,
            created_at=100,
        )
        assert payload.messages == messages
        assert payload.context == "Previous conversation summary"
        assert payload.user_id == "user_id"
        assert payload.session_id == "session_id"
        assert payload.namespace == "namespace"
        assert payload.tokens == 100
        assert payload.last_accessed == 100
        assert payload.created_at == 100

    def test_memory_response(self):
        """Test SessionMemoryResponse model"""
        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        # Test without any optional fields
        response = SessionMemoryResponse(messages=messages)
        assert response.messages == messages
        assert response.context is None
        assert response.tokens == 0
        assert response.user_id is None
        assert response.session_id is None
        assert response.namespace is None
        assert response.last_accessed is None
        assert response.created_at is None

        # Test with all fields
        response = SessionMemoryResponse(
            messages=messages,
            context="Conversation summary",
            tokens=150,
            user_id="user_id",
            session_id="session_id",
            namespace="namespace",
            last_accessed=100,
            created_at=100,
        )
        assert response.messages == messages
        assert response.context == "Conversation summary"
        assert response.tokens == 150
        assert response.user_id == "user_id"
        assert response.session_id == "session_id"
        assert response.namespace == "namespace"
        assert response.last_accessed == 100
        assert response.created_at == 100

    def test_search_payload(self):
        """Test SearchPayload model"""

        # Test default pagination
        payload = SearchPayload(text="What is the capital of France?")
        assert payload.text == "What is the capital of France?"
        assert payload.limit == 10
        assert payload.offset == 0

        # Test with all fields using create_with_primitives
        payload = SearchPayload.create_with_primitives(
            text="What is the capital of France?",
            session_id="session_id",
            namespace="namespace",
            topics=["France", "Paris"],
            entities=["France", "Paris"],
            distance_threshold=0.5,
        )
        assert payload.text == "What is the capital of France?"
        assert payload.session_id is not None
        assert payload.session_id.field == "session_id"
        assert payload.session_id.eq == "session_id"
        assert payload.namespace is not None
        assert payload.namespace.field == "namespace"
        assert payload.namespace.eq == "namespace"
        assert payload.topics is not None
        assert payload.topics.field == "topics"
        assert payload.topics.any == ["France", "Paris"]
        assert payload.entities is not None
        assert payload.entities.field == "entities"
        assert payload.entities.any == ["France", "Paris"]
        assert payload.distance_threshold == 0.5
        assert payload.limit == 10
        assert payload.offset == 0

    def test_long_term_memory_result(self):
        """Test LongTermMemoryResult model"""
        result = LongTermMemoryResult(
            text="Paris is the capital of France",
            dist=0.75,
            id_="123",
            session_id="session_id",
            user_id="user_id",
            last_accessed=100,
            created_at=100,
            namespace="namespace",
        )
        assert result.text == "Paris is the capital of France"
        assert result.dist == 0.75

    def test_search_payload_with_filter_objects(self):
        """Test SearchPayload model with filter objects"""

        # Create filter objects directly
        session_id = SessionId(eq="test-session")
        namespace = Namespace(eq="test-namespace")
        topics = Topics(any=["topic1", "topic2"])
        entities = Entities(any=["entity1", "entity2"])
        created_at = CreatedAt(gt=1000, lt=2000)
        last_accessed = LastAccessed(gt=3000, lt=4000)
        user_id = UserId(eq="test-user")

        # Create payload with filter objects
        payload = SearchPayload(
            text="Test query",
            session_id=session_id,
            namespace=namespace,
            topics=topics,
            entities=entities,
            created_at=created_at,
            last_accessed=last_accessed,
            user_id=user_id,
            distance_threshold=0.7,
            limit=15,
            offset=5,
        )

        # Check if payload contains filter objects
        assert payload.text == "Test query"
        assert payload.session_id == session_id
        assert payload.namespace == namespace
        assert payload.topics == topics
        assert payload.entities == entities
        assert payload.created_at == created_at
        assert payload.last_accessed == last_accessed
        assert payload.user_id == user_id
        assert payload.distance_threshold == 0.7
        assert payload.limit == 15
        assert payload.offset == 5

        # Test get_filters method
        filters = payload.get_filters()
        assert filters["session_id"] == session_id
        assert filters["namespace"] == namespace
        assert filters["topics"] == topics
        assert filters["entities"] == entities
        assert filters["created_at"] == created_at
        assert filters["last_accessed"] == last_accessed
        assert filters["user_id"] == user_id

    def test_search_payload_with_expanded_primitives(self):
        """Test SearchPayload.create_with_primitives with expanded filter options"""

        # Test with equality filters
        payload = SearchPayload.create_with_primitives(
            text="Testing expanded filters",
            session_id="session-123",
            namespace="test-namespace",
            user_id="user-456",
            created_at_eq=1000,
            last_accessed_eq=2000,
        )

        assert payload.session_id is not None
        assert payload.session_id.eq == "session-123"
        assert payload.namespace is not None
        assert payload.namespace.eq == "test-namespace"
        assert payload.user_id is not None
        assert payload.user_id.eq == "user-456"
        assert payload.created_at is not None
        assert payload.created_at.eq == 1000
        assert payload.last_accessed is not None
        assert payload.last_accessed.eq == 2000

        # Test with negation filters
        payload = SearchPayload.create_with_primitives(
            text="Testing negation filters",
            session_id_ne="not-this-session",
            namespace_ne="not-this-namespace",
            user_id_ne="not-this-user",
            topics_ne="excluded-topic",
            entities_ne="excluded-entity",
            created_at_ne=500,
            last_accessed_ne=600,
        )

        assert payload.session_id is not None
        assert payload.session_id.ne == "not-this-session"
        assert payload.namespace is not None
        assert payload.namespace.ne == "not-this-namespace"
        assert payload.user_id is not None
        assert payload.user_id.ne == "not-this-user"
        assert payload.topics is not None
        assert payload.topics.ne == "excluded-topic"
        assert payload.entities is not None
        assert payload.entities.ne == "excluded-entity"
        assert payload.created_at is not None
        assert payload.created_at.ne == 500
        assert payload.last_accessed is not None
        assert payload.last_accessed.ne == 600

        # Test with list filters
        payload = SearchPayload.create_with_primitives(
            text="Testing list filters",
            session_ids_any=["session-1", "session-2"],
            namespaces_all=["namespace-1", "namespace-2"],
            topics_any=["topic-1", "topic-2"],
            entities_all=["entity-1", "entity-2"],
            user_ids_any=["user-1", "user-2"],
        )

        assert payload.session_id is not None
        assert payload.session_id.any == ["session-1", "session-2"]
        assert payload.namespace is not None
        assert payload.namespace.all == ["namespace-1", "namespace-2"]
        assert payload.topics is not None
        assert payload.topics.any == ["topic-1", "topic-2"]
        assert payload.entities is not None
        assert payload.entities.all == ["entity-1", "entity-2"]
        assert payload.user_id is not None
        assert payload.user_id.any == ["user-1", "user-2"]

        # Test with range comparison filters
        payload = SearchPayload.create_with_primitives(
            text="Testing range comparison filters",
            created_at_gt=1000,
            created_at_lt=2000,
            last_accessed_gte=3000,
            last_accessed_lte=4000,
        )

        assert payload.created_at is not None
        assert payload.created_at.gt == 1000
        assert payload.created_at.lt == 2000
        assert payload.last_accessed is not None
        assert payload.last_accessed.gte == 3000
        assert payload.last_accessed.lte == 4000

        # Test with between range filters
        payload = SearchPayload.create_with_primitives(
            text="Testing between range filters",
            created_at_between=[1500, 1800],
            last_accessed_between=[3500, 3800],
        )

        assert payload.created_at is not None
        assert payload.created_at.between == [1500, 1800]
        assert payload.last_accessed is not None
        assert payload.last_accessed.between == [3500, 3800]
