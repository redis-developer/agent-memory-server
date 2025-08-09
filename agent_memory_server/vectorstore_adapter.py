"""
This module provides an abstraction layer between the agent memory server
and LangChain VectorStore implementations, allowing for pluggable backends.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_redis.vectorstores import RedisVectorStore
from redisvl.query import RangeQuery, VectorQuery

from agent_memory_server.filters import (
    CreatedAt,
    DiscreteMemoryExtracted,
    Entities,
    EventDate,
    Id,
    LastAccessed,
    MemoryHash,
    MemoryType,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from agent_memory_server.models import (
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
)


logger = logging.getLogger(__name__)

# Type variable for VectorStore implementations
VectorStoreType = TypeVar("VectorStoreType", bound=VectorStore)


class MemoryRedisVectorStore(RedisVectorStore):
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select the relevance score function based on the distance."""

        def relevance_score_fn(distance: float) -> float:
            # Ensure score is between 0 and 1
            score = (2 - distance) / 2
            return max(min(score, 1.0), 0.0)

        return relevance_score_fn


class LangChainFilterProcessor:
    """Utility class for processing and converting filter objects to LangChain backend formats."""

    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    @staticmethod
    def process_tag_filter(
        tag_filter, field_name: str, filter_dict: dict[str, Any]
    ) -> None:
        """Process a tag/string filter and add it to filter_dict if valid."""
        if not tag_filter:
            return

        if tag_filter.eq:
            filter_dict[field_name] = {"$eq": tag_filter.eq}
        elif tag_filter.ne:
            filter_dict[field_name] = {"$ne": tag_filter.ne}
        elif tag_filter.any:
            filter_dict[field_name] = {"$in": tag_filter.any}

    def process_datetime_filter(
        self, dt_filter, field_name: str, filter_dict: dict[str, Any]
    ) -> None:
        """Process a datetime filter and add it to filter_dict if valid."""
        if not dt_filter:
            return

        dt_filter_dict = {}

        if dt_filter.eq:
            dt_filter_dict["$eq"] = self._format_datetime(dt_filter.eq)
        elif dt_filter.ne:
            dt_filter_dict["$ne"] = self._format_datetime(dt_filter.ne)
        elif dt_filter.gt:
            dt_filter_dict["$gt"] = self._format_datetime(dt_filter.gt)
        elif dt_filter.gte:
            dt_filter_dict["$gte"] = self._format_datetime(dt_filter.gte)
        elif dt_filter.lt:
            dt_filter_dict["$lt"] = self._format_datetime(dt_filter.lt)
        elif dt_filter.lte:
            dt_filter_dict["$lte"] = self._format_datetime(dt_filter.lte)
        elif dt_filter.between:
            dt_filter_dict["$between"] = [
                self._format_datetime(dt) for dt in dt_filter.between
            ]

        if dt_filter_dict:
            filter_dict[field_name] = dt_filter_dict

    def _format_datetime(self, dt: datetime) -> str | float:
        """Format datetime for the specific backend."""
        vectorstore_type = str(type(self.vectorstore)).lower()

        # Pinecone requires Unix timestamps for datetime comparisons
        if "pinecone" in vectorstore_type:
            return dt.timestamp()
        # Most other backends use ISO strings
        return dt.isoformat()

    def convert_filters_to_backend_format(
        self,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        event_date: EventDate | None = None,
        memory_hash: MemoryHash | None = None,
        id: Id | None = None,
        discrete_memory_extracted: DiscreteMemoryExtracted | None = None,
    ) -> dict[str, Any] | None:
        """Convert filter objects to backend format for LangChain vectorstores."""
        filter_dict: dict[str, Any] = {}

        # TODO: Seems like we could take *args filters and decide what to do based on type.
        # Apply tag/string filters using the helper function
        self.process_tag_filter(session_id, "session_id", filter_dict)
        self.process_tag_filter(user_id, "user_id", filter_dict)
        self.process_tag_filter(namespace, "namespace", filter_dict)
        self.process_tag_filter(memory_type, "memory_type", filter_dict)
        self.process_tag_filter(topics, "topics", filter_dict)
        self.process_tag_filter(entities, "entities", filter_dict)
        self.process_tag_filter(memory_hash, "memory_hash", filter_dict)
        self.process_tag_filter(id, "id_", filter_dict)
        self.process_tag_filter(
            discrete_memory_extracted, "discrete_memory_extracted", filter_dict
        )

        # Apply datetime filters using the helper function (uses instance method for backend-specific formatting)
        self.process_datetime_filter(created_at, "created_at", filter_dict)
        self.process_datetime_filter(last_accessed, "last_accessed", filter_dict)
        self.process_datetime_filter(event_date, "event_date", filter_dict)

        return filter_dict if filter_dict else None


class VectorStoreAdapter(ABC):
    """Abstract base class for VectorStore adapters."""

    def __init__(self, vectorstore: VectorStore, embeddings: Embeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    @abstractmethod
    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memory records to the vector store.

        Args:
            memories: List of MemoryRecord objects to add

        Returns:
            List of document IDs that were added
        """
        pass

    @abstractmethod
    async def search_memories(
        self,
        query: str,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        memory_hash: MemoryHash | None = None,
        id: Id | None = None,
        discrete_memory_extracted: DiscreteMemoryExtracted | None = None,
        distance_threshold: float | None = None,
        server_side_recency: bool | None = None,
        recency_params: dict | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """Search memories in the vector store.

        Args:
            query: Text query for semantic search
            session_id: Optional session ID filter
            user_id: Optional user ID filter
            namespace: Optional namespace filter
            created_at: Optional created at filter
            last_accessed: Optional last accessed filter
            topics: Optional topics filter
            entities: Optional entities filter
            memory_type: Optional memory type filter
            event_date: Optional event date filter
            memory_hash: Optional memory hash filter
            distance_threshold: Optional similarity threshold
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            MemoryRecordResults containing matching memories
        """
        pass

    @abstractmethod
    async def delete_memories(self, memory_ids: list[str]) -> int:
        """Delete memories by their IDs.

        Args:
            memory_ids: List of memory IDs to delete

        Returns:
            Number of memories deleted
        """
        pass

    @abstractmethod
    async def update_memories(self, memories: list[MemoryRecord]) -> int:
        """Update memory records in the vector store.

        Args:
            memories: List of MemoryRecord objects to update

        Returns:
            Number of memories updated
        """
        pass

    @abstractmethod
    async def count_memories(
        self,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memories matching the given filters.

        Args:
            namespace: Optional namespace filter
            user_id: Optional user ID filter
            session_id: Optional session ID filter

        Returns:
            Number of matching memories
        """
        pass

    def _parse_list_field(self, field_value):
        """Parse a field that might be a list, comma-separated string, or None.

        Centralized here so both LangChain and Redis adapters can normalize
        metadata fields like topics/entities/extracted_from.
        """
        if not field_value:
            return []
        if isinstance(field_value, list):
            return field_value
        if isinstance(field_value, str):
            return field_value.split(",") if field_value else []
        return []

    def memory_to_document(self, memory: MemoryRecord) -> Document:
        """Convert a MemoryRecord to a LangChain Document.

        Args:
            memory: MemoryRecord to convert

        Returns:
            LangChain Document with metadata
        """
        # Use ISO strings for datetime fields (standard format for most backends)
        created_at_val = memory.created_at.isoformat() if memory.created_at else None
        last_accessed_val = (
            memory.last_accessed.isoformat() if memory.last_accessed else None
        )
        updated_at_val = memory.updated_at.isoformat() if memory.updated_at else None
        persisted_at_val = (
            memory.persisted_at.isoformat() if memory.persisted_at else None
        )
        event_date_val = memory.event_date.isoformat() if memory.event_date else None

        pinned_int = 1 if getattr(memory, "pinned", False) else 0
        access_count_int = int(getattr(memory, "access_count", 0) or 0)

        metadata = {
            "id_": memory.id,
            "session_id": memory.session_id,
            "user_id": memory.user_id,
            "namespace": memory.namespace,
            "created_at": created_at_val,
            "last_accessed": last_accessed_val,
            "updated_at": updated_at_val,
            "pinned": pinned_int,
            "access_count": access_count_int,
            "topics": memory.topics,
            "entities": memory.entities,
            "memory_hash": memory.memory_hash,
            "discrete_memory_extracted": memory.discrete_memory_extracted,
            "memory_type": memory.memory_type.value,
            "id": memory.id,
            "persisted_at": persisted_at_val,
            "extracted_from": memory.extracted_from,
            "event_date": event_date_val,
        }

        # Remove None values to keep metadata clean
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Document(
            page_content=memory.text,
            metadata=metadata,
        )

    def document_to_memory(
        self, doc: Document, score: float = 0.0
    ) -> MemoryRecordResult:
        """Convert a LangChain Document to a MemoryRecordResult.

        Args:
            doc: LangChain Document to convert
            score: Similarity score for the document

        Returns:
            MemoryRecordResult with converted data
        """
        metadata = doc.metadata

        # Parse datetime values back to datetime objects (handle both timestamp and ISO string formats)
        def parse_datetime(dt_val: str | float | None) -> datetime | None:
            if dt_val is None:
                return None
            if isinstance(dt_val, int | float):
                # Unix timestamp from Redis
                return datetime.fromtimestamp(dt_val, tz=UTC)
            if isinstance(dt_val, str):
                # ISO string from other backends
                return datetime.fromisoformat(dt_val)
            return None

        created_at = parse_datetime(metadata.get("created_at"))
        last_accessed = parse_datetime(metadata.get("last_accessed"))
        updated_at = parse_datetime(metadata.get("updated_at"))
        persisted_at = parse_datetime(metadata.get("persisted_at"))
        event_date = parse_datetime(metadata.get("event_date"))

        # Provide defaults for required fields
        if not created_at:
            created_at = datetime.now(UTC)
        if not last_accessed:
            last_accessed = datetime.now(UTC)
        if not updated_at:
            updated_at = datetime.now(UTC)

        # Normalize pinned/access_count from metadata
        pinned_meta = metadata.get("pinned", 0)
        try:
            pinned_bool = bool(int(pinned_meta))
        except Exception:
            pinned_bool = bool(pinned_meta)
        access_count_meta = metadata.get("access_count", 0)
        try:
            access_count_val = int(access_count_meta or 0)
        except Exception:
            access_count_val = 0

        return MemoryRecordResult(
            text=doc.page_content,
            id=metadata.get("id") or metadata.get("id_") or "",
            session_id=metadata.get("session_id"),
            user_id=metadata.get("user_id"),
            namespace=metadata.get("namespace"),
            created_at=created_at,
            last_accessed=last_accessed,
            updated_at=updated_at,
            pinned=pinned_bool,
            access_count=access_count_val,
            topics=self._parse_list_field(metadata.get("topics")),
            entities=self._parse_list_field(metadata.get("entities")),
            memory_hash=metadata.get("memory_hash"),
            discrete_memory_extracted=metadata.get("discrete_memory_extracted", "f"),
            memory_type=metadata.get("memory_type", "message"),
            persisted_at=persisted_at,
            extracted_from=self._parse_list_field(metadata.get("extracted_from")),
            event_date=event_date,
            dist=score,
        )

    def generate_memory_hash(self, memory: MemoryRecord) -> str:
        """Generate a stable hash for a memory based on text, user_id, and session_id.

        Args:
            memory: MemoryRecord to hash

        Returns:
            A stable hash string
        """
        # Use the same hash logic as long_term_memory.py for consistency
        from agent_memory_server.long_term_memory import generate_memory_hash

        return generate_memory_hash(memory)

    def _convert_filters_to_backend_format(
        self,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        event_date: EventDate | None = None,
        memory_hash: MemoryHash | None = None,
        id: Id | None = None,
        discrete_memory_extracted: DiscreteMemoryExtracted | None = None,
    ) -> dict[str, Any] | None:
        """Convert filter objects to standard LangChain dictionary format.

        Uses the PGVector/Pinecone style dictionary format with operators like $eq, $in, etc.
        This works with most standard LangChain VectorStore implementations.

        Backend-specific datetime handling:
        - Pinecone: Uses Unix timestamps (numbers)
        - Others: Use ISO strings

        Args:
            Filter objects from filters.py

        Returns:
            Dictionary filter in format: {"field": {"$eq": "value"}} or None
        """
        processor = LangChainFilterProcessor(self.vectorstore)
        # TODO: Seems like we could take *args and pass them to the processor
        filter_dict = processor.convert_filters_to_backend_format(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            topics=topics,
            entities=entities,
            memory_type=memory_type,
            created_at=created_at,
            last_accessed=last_accessed,
            event_date=event_date,
            memory_hash=memory_hash,
            id=id,
        )

        logger.debug(f"Converted to LangChain filter format: {filter_dict}")
        return filter_dict


class LangChainVectorStoreAdapter(VectorStoreAdapter):
    """Generic adapter for any LangChain VectorStore implementation."""

    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memory records to the vector store."""
        if not memories:
            return []

        # Convert MemoryRecords to Documents
        documents = []
        for memory in memories:
            # Generate hash if not provided
            if not memory.memory_hash:
                memory.memory_hash = self.generate_memory_hash(memory)

            doc = self.memory_to_document(memory)
            logger.info(
                f"Converting memory to document: {memory.id} -> metadata: {doc.metadata}"
            )
            documents.append(doc)

        # Add documents to the vector store
        try:
            # Extract IDs from memory records to prevent ULID generation
            memory_ids = [memory.id for memory in memories]

            # Standard LangChain VectorStore implementation
            if hasattr(self.vectorstore, "aadd_documents"):
                ids = await self.vectorstore.aadd_documents(documents, ids=memory_ids)
            elif hasattr(self.vectorstore, "add_documents"):
                ids = self.vectorstore.add_documents(documents, ids=memory_ids)
            else:
                # Fallback to add_texts
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                if hasattr(self.vectorstore, "aadd_texts"):
                    ids = await self.vectorstore.aadd_texts(
                        texts, metadatas=metadatas, ids=memory_ids
                    )
                else:
                    ids = self.vectorstore.add_texts(
                        texts, metadatas=metadatas, ids=memory_ids
                    )

            return ids or memory_ids
        except Exception as e:
            logger.error(f"Error adding memories to vector store: {e}")
            raise

    async def search_memories(
        self,
        query: str,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        memory_hash: MemoryHash | None = None,
        id: Id | None = None,
        distance_threshold: float | None = None,
        discrete_memory_extracted: DiscreteMemoryExtracted | None = None,
        server_side_recency: bool | None = None,
        recency_params: dict | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """Search memories using the LangChain MemoryRedisVectorStore."""
        try:
            # Convert filters to LangChain format
            filter_dict = self._convert_filters_to_backend_format(
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                topics=topics,
                entities=entities,
                memory_type=memory_type,
                created_at=created_at,
                last_accessed=last_accessed,
                event_date=event_date,
                memory_hash=memory_hash,
                id=id,
                discrete_memory_extracted=discrete_memory_extracted,
            )

            # Use LangChain's similarity search with filters
            search_kwargs: dict[str, Any] = {"k": limit + offset}
            if filter_dict:
                search_kwargs["filter"] = filter_dict

            # Perform similarity search
            logger.info(f"Searching for memories with filters: {search_kwargs}")

            docs_with_scores = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **search_kwargs
                )
            )

            # Apply distance threshold if specified
            if distance_threshold is not None:
                docs_with_scores = [
                    (doc, score)
                    for doc, score in docs_with_scores
                    if score
                    >= (1.0 - distance_threshold)  # Convert distance to similarity
                ]

            # Apply offset
            docs_with_scores = docs_with_scores[offset:]

            # Convert to MemoryRecordResult objects
            memory_results = []
            for doc, score in docs_with_scores:
                memory_result = self.document_to_memory(doc, score)
                memory_results.append(memory_result)

            # If recency requested but backend does not support DB-level, rerank here as a fallback
            if server_side_recency and memory_results:
                try:
                    from datetime import UTC as _UTC, datetime as _dt

                    from agent_memory_server.long_term_memory import rerank_with_recency

                    now = _dt.now(_UTC)
                    params = {
                        "w_sem": float(recency_params.get("w_sem", 0.8))
                        if recency_params
                        else 0.8,
                        "w_recency": float(recency_params.get("w_recency", 0.2))
                        if recency_params
                        else 0.2,
                        "wf": float(recency_params.get("wf", 0.6))
                        if recency_params
                        else 0.6,
                        "wa": float(recency_params.get("wa", 0.4))
                        if recency_params
                        else 0.4,
                        "half_life_last_access_days": float(
                            recency_params.get("half_life_last_access_days", 7.0)
                        )
                        if recency_params
                        else 7.0,
                        "half_life_created_days": float(
                            recency_params.get("half_life_created_days", 30.0)
                        )
                        if recency_params
                        else 30.0,
                    }
                    memory_results = rerank_with_recency(
                        memory_results, now=now, params=params
                    )
                except Exception:
                    pass

            # Calculate next offset
            next_offset = offset + limit if len(docs_with_scores) > limit else None

            return MemoryRecordResults(
                memories=memory_results[:limit],  # Limit results after offset
                total=len(docs_with_scores) + offset,  # Approximate total
                next_offset=next_offset,
            )

        except Exception as e:
            logger.error(f"Error searching memories in Redis vectorstore: {e}")
            raise

    async def delete_memories(self, memory_ids: list[str]) -> int:
        """Delete memories by their IDs."""
        if not memory_ids:
            return 0

        try:
            if hasattr(self.vectorstore, "adelete"):
                deleted = await self.vectorstore.adelete(memory_ids)
            elif hasattr(self.vectorstore, "delete"):
                deleted = self.vectorstore.delete(memory_ids)
            else:
                logger.warning("Vector store does not support delete operation")
                return 0

            return len(memory_ids) if deleted else 0

        except Exception as e:
            logger.error(f"Error deleting memories from vector store: {e}")
            raise

    async def count_memories(
        self,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memories in the vector store using LangChain."""
        try:
            # Convert basic filters to our filter objects, then to backend format
            from agent_memory_server.filters import Namespace, SessionId, UserId

            namespace_filter = Namespace(eq=namespace) if namespace else None
            user_id_filter = UserId(eq=user_id) if user_id else None
            session_id_filter = SessionId(eq=session_id) if session_id else None

            # Most vector stores don't have a direct count method
            # We'll use a large similarity search and count results
            # This is not optimal but works as a fallback
            search_kwargs: dict[str, Any] = {
                "k": 10000
            }  # Large number to get all results

            # Apply filters using the proper method signature
            backend_filter = self._convert_filters_to_backend_format(
                namespace=namespace_filter,
                user_id=user_id_filter,
                session_id=session_id_filter,
            )
            if backend_filter:
                search_kwargs["filter"] = backend_filter

            if hasattr(self.vectorstore, "asimilarity_search"):
                docs = await self.vectorstore.asimilarity_search("", **search_kwargs)
            elif hasattr(self.vectorstore, "similarity_search"):
                docs = self.vectorstore.similarity_search("", **search_kwargs)
            else:
                logger.warning("Vector store does not support similarity_search")
                return 0

            # The vectorstore should have already applied the filters
            return len(docs)

        except Exception as e:
            logger.error(f"Error counting memories in vector store: {e}")
            return 0

    async def update_memories(self, memories: list[MemoryRecord]) -> int:
        """Update memory records in the vector store."""
        if not memories:
            return 0

        # This is less than ideal, but the VectorStore interface lacks an update method.
        try:
            await self.delete_memories([memory.id for memory in memories])
            await self.add_memories(memories)
            return len(memories)
        except Exception as e:
            logger.error(f"Error updating memories in vector store: {e}")
            return 0


class RedisVectorStoreAdapter(VectorStoreAdapter):
    """Redis adapter that uses LangChain's RedisVectorStore with Redis-specific optimizations."""

    vectorstore: RedisVectorStore

    def __init__(self, vectorstore: VectorStore, embeddings: Embeddings):
        """Initialize Redis adapter.

        Args:
            vectorstore: Redis VectorStore instance from LangChain
            embeddings: Embeddings instance
        """
        super().__init__(vectorstore, embeddings)

    def memory_to_document(self, memory: MemoryRecord) -> Document:
        """Convert a MemoryRecord to a LangChain Document with Redis timestamp format.

        Args:
            memory: MemoryRecord to convert

        Returns:
            LangChain Document with metadata optimized for Redis
        """
        # For Redis backends, use Unix timestamps for NUMERIC fields
        created_at_val = memory.created_at.timestamp() if memory.created_at else None
        last_accessed_val = (
            memory.last_accessed.timestamp() if memory.last_accessed else None
        )
        updated_at_val = memory.updated_at.timestamp() if memory.updated_at else None
        persisted_at_val = (
            memory.persisted_at.timestamp() if memory.persisted_at else None
        )
        event_date_val = memory.event_date.timestamp() if memory.event_date else None

        pinned_int = 1 if memory.pinned else 0
        access_count_int = int(memory.access_count or 0)

        metadata = {
            "id_": memory.id,  # The client-generated ID
            "session_id": memory.session_id,
            "user_id": memory.user_id,
            "namespace": memory.namespace,
            "created_at": created_at_val,
            "last_accessed": last_accessed_val,
            "updated_at": updated_at_val,
            "pinned": pinned_int,
            "access_count": access_count_int,
            "topics": memory.topics,
            "entities": memory.entities,
            "memory_hash": memory.memory_hash,
            "discrete_memory_extracted": memory.discrete_memory_extracted,
            "memory_type": memory.memory_type.value,
            "persisted_at": persisted_at_val,
            "extracted_from": memory.extracted_from,
            "event_date": event_date_val,
        }

        # Remove None values to keep metadata clean
        metadata = {k: v for k, v in metadata.items() if v is not None}

        # NOTE: We don't get back Document.id from RedisVL (because RedisVectorStore
        # doesn't return it). Instead, we get our client-generated ID back as the "id_"
        # metadata field.However, LangChain docs say "id" here will become a required
        # field in the future, so we're setting it now.
        return Document(
            id=memory.id,
            page_content=memory.text,
            metadata=metadata,
        )

    async def add_memories(self, memories: list[MemoryRecord]) -> list[str]:
        """Add memories using the LangChain RedisVectorStore."""
        if not memories:
            return []

        try:
            # Convert memories to LangChain Documents
            documents = []

            for memory in memories:
                if not memory.memory_hash:
                    memory.memory_hash = self.generate_memory_hash(memory)
                now_timestamp = datetime.now(UTC)
                if not memory.created_at:
                    memory.created_at = now_timestamp
                if not memory.last_accessed:
                    memory.last_accessed = now_timestamp
                if not memory.updated_at:
                    memory.updated_at = now_timestamp

                # Convert memory to document using the parent class method
                doc = self.memory_to_document(memory)
                documents.append(doc)

            # Use the LangChain RedisVectorStore to add documents.
            #
            # NOTE: We pass our client-generated IDs as "keys" as an
            # optimization -- without a key, RedisVL will generate a ULID to use
            # as part of the key. However, either way, RedisVL will later
            # returns a prefixed Redis key instead of our precise ID value, e.g. if
            # we give it "id", we'll get back "my-prefix:<id>". This means if we want
            # to query on ID with Redis Query Engine or get back the ID without
            # parsing that prefixed key, we need to send it as a metadata field.
            # That is exactly what we do, also sending it as "id_" separately from
            # this use for keys.
            keys = [doc.metadata.get("id_", "") for doc in documents]
            return await self.vectorstore.aadd_documents(documents, keys=keys)

        except Exception as e:
            logger.error(f"Error adding memories to Redis vectorstore: {e}")
            raise

    async def update_memories(self, memories: list[MemoryRecord]) -> int:
        """Update memory records in the vector store."""
        if not memories:
            return 0

        added = await self.add_memories(memories)
        return len(added)

    async def search_memories(
        self,
        query: str,
        session_id: SessionId | None = None,
        user_id: UserId | None = None,
        namespace: Namespace | None = None,
        created_at: CreatedAt | None = None,
        last_accessed: LastAccessed | None = None,
        topics: Topics | None = None,
        entities: Entities | None = None,
        memory_type: MemoryType | None = None,
        event_date: EventDate | None = None,
        memory_hash: MemoryHash | None = None,
        id: Id | None = None,
        discrete_memory_extracted: DiscreteMemoryExtracted | None = None,
        distance_threshold: float | None = None,
        server_side_recency: bool | None = None,
        recency_params: dict | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> MemoryRecordResults:
        """Search memories RedisVectorStore."""
        filters = []

        # Add individual filters using the .to_filter() methods from filters.py
        if session_id:
            filters.append(session_id.to_filter())
        if user_id:
            filters.append(user_id.to_filter())
        if namespace:
            filters.append(namespace.to_filter())
        if memory_type:
            filters.append(memory_type.to_filter())
        if topics:
            filters.append(topics.to_filter())
        if entities:
            filters.append(entities.to_filter())
        if created_at:
            filters.append(created_at.to_filter())
        if last_accessed:
            filters.append(last_accessed.to_filter())
        if event_date:
            filters.append(event_date.to_filter())
        if memory_hash:
            filters.append(memory_hash.to_filter())
        if id:
            filters.append(id.to_filter())
        if discrete_memory_extracted:
            filters.append(discrete_memory_extracted.to_filter())

        # Combine filters with AND logic
        redis_filter = None
        if filters:
            if len(filters) == 1:
                redis_filter = filters[0]
            else:
                from functools import reduce

                redis_filter = reduce(lambda x, y: x & y, filters)

        # If server-side recency is requested, attempt RedisVL query first (DB-level path)
        if server_side_recency:
            try:
                index = getattr(self.vectorstore, "_index", None)
                if index is not None:
                    # Embed the query text to vector
                    embedding_vector = self.embeddings.embed_query(query)

                    # Build base KNN query (hybrid)
                    if distance_threshold is not None:
                        knn = RangeQuery(
                            vector=embedding_vector,
                            vector_field_name="vector",
                            filter_expression=redis_filter,
                            distance_threshold=float(distance_threshold),
                            num_results=limit,
                        )
                    else:
                        knn = VectorQuery(
                            vector=embedding_vector,
                            vector_field_name="vector",
                            filter_expression=redis_filter,
                            num_results=limit,
                        )

                    # Aggregate with APPLY/SORTBY boosted score via helper
                    from datetime import UTC as _UTC, datetime as _dt

                    from agent_memory_server.utils.redis_query import (
                        RecencyAggregationQuery,
                    )

                    now_ts = int(_dt.now(_UTC).timestamp())
                    agg = (
                        RecencyAggregationQuery.from_vector_query(
                            knn, filter_expression=redis_filter
                        )
                        .load_default_fields()
                        .apply_recency(now_ts=now_ts, params=recency_params or {})
                        .sort_by_boosted_desc()
                        .paginate(offset, limit)
                    )

                    raw = (
                        await index.aaggregate(agg)
                        if hasattr(index, "aaggregate")
                        else index.aggregate(agg)  # type: ignore
                    )

                    rows = getattr(raw, "rows", raw) or []
                    memory_results: list[MemoryRecordResult] = []
                    for row in rows:
                        fields = getattr(row, "__dict__", None) or row
                        metadata = {
                            k: fields.get(k)
                            for k in [
                                "id_",
                                "session_id",
                                "user_id",
                                "namespace",
                                "created_at",
                                "last_accessed",
                                "updated_at",
                                "pinned",
                                "access_count",
                                "topics",
                                "entities",
                                "memory_hash",
                                "discrete_memory_extracted",
                                "memory_type",
                                "persisted_at",
                                "extracted_from",
                                "event_date",
                            ]
                            if k in fields
                        }
                        text_val = fields.get("text", "")
                        score = fields.get("__vector_score", 1.0) or 1.0
                        doc_obj = Document(page_content=text_val, metadata=metadata)
                        memory_results.append(
                            self.document_to_memory(doc_obj, float(score))
                        )

                    next_offset = (
                        offset + limit if len(memory_results) == limit else None
                    )
                    return MemoryRecordResults(
                        memories=memory_results[:limit],
                        total=offset + len(memory_results),
                        next_offset=next_offset,
                    )
            except Exception as e:
                logger.warning(
                    f"RedisVL DB-level recency search failed; falling back to client-side path: {e}"
                )

        # Prepare search kwargs (standard LangChain path)
        search_kwargs = {
            "query": query,
            "filter": redis_filter,
            "k": limit + offset,
        }

        # Use score_threshold if distance_threshold is provided
        if distance_threshold is not None:
            # Convert distance threshold to score threshold
            # Distance 0 = perfect match, Score 1 = perfect match
            score_threshold = 1.0 - distance_threshold
            search_kwargs["score_threshold"] = score_threshold

        logger.debug(f"[search_memories] Search kwargs: {search_kwargs}")
        search_results = (
            await self.vectorstore.asimilarity_search_with_relevance_scores(
                **search_kwargs
            )
        )

        logger.debug(f"[search_memories] Search results: {search_results}")
        # Convert results to MemoryRecordResult objects
        memory_results = []
        for i, (doc, score) in enumerate(search_results):
            # Apply offset - VectorStore doesn't support pagination...
            # TODO: Implement pagination in RedisVectorStore as a kwarg.
            if i < offset:
                continue

            # Clamp score to valid range [0, 1] to avoid floating-point precision issues
            clamped_score = max(0.0, min(1.0, score))

            # Convert relevance score to distance for the result
            distance = 1.0 - clamped_score

            # Helper function to parse timestamp to datetime
            def parse_timestamp_to_datetime(timestamp_val):
                if not timestamp_val:
                    return datetime.now(UTC)
                if isinstance(timestamp_val, int | float):
                    return datetime.fromtimestamp(timestamp_val, tz=UTC)
                return datetime.now(UTC)

            # Extract memory data
            memory_result = MemoryRecordResult(
                id=doc.metadata.get("id_", ""),  # Get our client-generated ID
                text=doc.page_content,
                dist=distance,
                created_at=parse_timestamp_to_datetime(doc.metadata.get("created_at")),
                updated_at=parse_timestamp_to_datetime(doc.metadata.get("updated_at")),
                last_accessed=parse_timestamp_to_datetime(
                    doc.metadata.get("last_accessed")
                ),
                user_id=doc.metadata.get("user_id"),
                session_id=doc.metadata.get("session_id"),
                namespace=doc.metadata.get("namespace"),
                pinned=doc.metadata.get("pinned", False),
                access_count=int(doc.metadata.get("access_count", 0) or 0),
                topics=self._parse_list_field(doc.metadata.get("topics")),
                entities=self._parse_list_field(doc.metadata.get("entities")),
                memory_hash=doc.metadata.get("memory_hash", ""),
                discrete_memory_extracted=doc.metadata.get(
                    "discrete_memory_extracted", "f"
                ),
                memory_type=doc.metadata.get("memory_type", "message"),
                persisted_at=doc.metadata.get("persisted_at"),
                extracted_from=self._parse_list_field(
                    doc.metadata.get("extracted_from")
                ),
                event_date=doc.metadata.get("event_date"),
            )

            memory_results.append(memory_result)

            # Stop if we have enough results
            if len(memory_results) >= limit:
                break

        # Optional server-side recency-aware rerank (adapter-level fallback)
        # If requested, re-rank using the same logic as server API's local reranking.
        if server_side_recency:
            try:
                from datetime import UTC as _UTC, datetime as _dt

                from agent_memory_server.long_term_memory import rerank_with_recency

                now = _dt.now(_UTC)
                params = {
                    "w_sem": float(recency_params.get("w_sem", 0.8))
                    if recency_params
                    else 0.8,
                    "w_recency": float(recency_params.get("w_recency", 0.2))
                    if recency_params
                    else 0.2,
                    "wf": float(recency_params.get("wf", 0.6))
                    if recency_params
                    else 0.6,
                    "wa": float(recency_params.get("wa", 0.4))
                    if recency_params
                    else 0.4,
                    "half_life_last_access_days": float(
                        recency_params.get("half_life_last_access_days", 7.0)
                    )
                    if recency_params
                    else 7.0,
                    "half_life_created_days": float(
                        recency_params.get("half_life_created_days", 30.0)
                    )
                    if recency_params
                    else 30.0,
                }
                memory_results = rerank_with_recency(
                    memory_results, now=now, params=params
                )
            except Exception as e:
                logger.warning(
                    f"server_side_recency fallback rerank failed, returning base order: {e}"
                )

        next_offset = offset + limit if len(search_results) > offset + limit else None

        return MemoryRecordResults(
            memories=memory_results[:limit],
            total=len(search_results),
            next_offset=next_offset,
        )

    def _parse_list_field(self, field_value):
        """Parse a field that might be a list, comma-separated string, or None."""
        if not field_value:
            return []
        if isinstance(field_value, list):
            return field_value
        if isinstance(field_value, str):
            return field_value.split(",") if field_value else []
        return []

    async def delete_memories(self, memory_ids: list[str]) -> int:
        """Delete memories by their IDs using LangChain's RedisVectorStore."""
        if not memory_ids:
            return 0

        try:
            if hasattr(self.vectorstore, "adelete"):
                deleted = await self.vectorstore.adelete(memory_ids)
            elif hasattr(self.vectorstore, "delete"):
                deleted = self.vectorstore.delete(memory_ids)
            else:
                logger.warning("Redis vectorstore does not support delete operation")
                return 0

            return len(memory_ids) if deleted else 0

        except Exception as e:
            logger.error(f"Error deleting memories from Redis vectorstore: {e}")
            raise

    async def count_memories(
        self,
        namespace: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Count memories using the same approach as search_memories for consistency."""
        try:
            # Use the same filter approach as search_memories
            filters = []

            if namespace:
                from agent_memory_server.filters import Namespace

                namespace_filter = Namespace(eq=namespace).to_filter()
                filters.append(namespace_filter)
            if user_id:
                from agent_memory_server.filters import UserId

                user_filter = UserId(eq=user_id).to_filter()
                filters.append(user_filter)
            if session_id:
                from agent_memory_server.filters import SessionId

                session_filter = SessionId(eq=session_id).to_filter()
                filters.append(session_filter)

            # Combine filters with AND logic
            redis_filter = None
            if filters:
                if len(filters) == 1:
                    redis_filter = filters[0]
                else:
                    from functools import reduce

                    redis_filter = reduce(lambda x, y: x & y, filters)

            # Use the same search method as search_memories but for counting
            # We use the same query that would match the indexed content
            search_results = await self.vectorstore.asimilarity_search(
                query="duplicate",  # Use a query that should match test content
                filter=redis_filter,
                k=10000,  # Large number to get all results
            )

            return len(search_results)

        except Exception as e:
            logger.error(
                f"Error counting memories in Redis vectorstore: {e}", exc_info=True
            )
            return 0
