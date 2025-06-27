import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis
from ulid import ULID

from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.extraction import extract_discrete_memories, handle_extraction
from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    EventDate,
    LastAccessed,
    MemoryHash,
    MemoryType,
    Namespace,
    SessionId,
    Topics,
    UserId,
)
from agent_memory_server.llms import (
    AnthropicClientWrapper,
    OpenAIClientWrapper,
    get_model_client,
)
from agent_memory_server.models import (
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
)
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.redis import (
    ensure_search_index_exists,
    get_redis_conn,
)
from agent_memory_server.vectorstore_factory import get_vectorstore_adapter


# Prompt for extracting memories from messages in working memory context
WORKING_MEMORY_EXTRACTION_PROMPT = """
You are a memory extraction assistant. Your job is to analyze conversation
messages and extract information that might be useful in future conversations.

Extract two types of memories from the following message:
1. EPISODIC: Experiences or events that have a time dimension.
   (They MUST have a time dimension to be "episodic.")
   Example: "User mentioned they visited Paris last month" or "User had trouble with the login process"

2. SEMANTIC: User preferences, facts, or general knowledge that would be useful long-term.
   Example: "User prefers dark mode UI" or "User works as a data scientist"

For each memory, return a JSON object with the following fields:
- type: str -- The memory type, either "episodic" or "semantic"
- text: str -- The actual information to store
- topics: list[str] -- Relevant topics for this memory
- entities: list[str] -- Named entities mentioned
- event_date: str | null -- For episodic memories, the date/time when the event occurred (ISO 8601 format), null for semantic memories

IMPORTANT RULES:
1. Only extract information that would be genuinely useful for future interactions.
2. Do not extract procedural knowledge or instructions.
3. If given `user_id`, focus on user-specific information, preferences, and facts.
4. Return an empty list if no useful memories can be extracted.

Message: {message}

Return format:
{{
    "memories": [
        {{
            "type": "episodic",
            "text": "...",
            "topics": ["..."],
            "entities": ["..."],
            "event_date": "2024-01-15T14:30:00Z"
        }},
        {{
            "type": "semantic",
            "text": "...",
            "topics": ["..."],
            "entities": ["..."],
            "event_date": null
        }}
    ]
}}

Extracted memories:
"""


logger = logging.getLogger(__name__)


async def extract_memory_structure(_id: str, text: str, namespace: str | None):
    redis = await get_redis_conn()

    # Process messages for topic/entity extraction
    topics, entities = await handle_extraction(text)

    # Convert lists to comma-separated strings for TAG fields
    topics_joined = ",".join(topics) if topics else ""
    entities_joined = ",".join(entities) if entities else ""

    await redis.hset(
        Keys.memory_key(_id, namespace),
        mapping={
            "topics": topics_joined,
            "entities": entities_joined,
        },
    )  # type: ignore


def generate_memory_hash(memory: dict) -> str:
    """
    Generate a stable hash for a memory based on text, user_id, and session_id.

    Args:
        memory: Dictionary containing memory data

    Returns:
        A stable hash string
    """
    # Create a deterministic string representation of the key fields
    text = memory.get("text", "")
    user_id = memory.get("user_id", "") or ""
    session_id = memory.get("session_id", "") or ""

    # Combine the fields in a predictable order
    hash_content = f"{text}|{user_id}|{session_id}"

    # Create a stable hash
    return hashlib.sha256(hash_content.encode()).hexdigest()


async def merge_memories_with_llm(memories: list[dict], llm_client: Any = None) -> dict:
    """
    Use an LLM to merge similar or duplicate memories.

    Args:
        memories: List of memory dictionaries to merge
        llm_client: Optional LLM client to use for merging

    Returns:
        A merged memory dictionary
    """
    # If there's only one memory, just return it
    if len(memories) == 1:
        return memories[0]

    # Create a unified set of topics and entities
    all_topics = set()
    all_entities = set()

    for memory in memories:
        if memory.get("topics"):
            if isinstance(memory["topics"], str):
                all_topics.update(memory["topics"].split(","))
            else:
                all_topics.update(memory["topics"])

        if memory.get("entities"):
            if isinstance(memory["entities"], str):
                all_entities.update(memory["entities"].split(","))
            else:
                all_entities.update(memory["entities"])

    # Get the memory texts for LLM prompt
    memory_texts = [m["text"] for m in memories]

    # Construct the LLM prompt
    instruction = "Merge these similar memories into a single, coherent memory:"

    prompt = f"{instruction}\n\n"
    for i, text in enumerate(memory_texts, 1):
        prompt += f"Memory {i}: {text}\n\n"

    prompt += "\nMerged memory:"

    model_name = "gpt-4o-mini"

    if not llm_client:
        model_client: (
            OpenAIClientWrapper | AnthropicClientWrapper
        ) = await get_model_client(model_name)
    else:
        model_client = llm_client

    response = await model_client.create_chat_completion(
        model=model_name,
        prompt=prompt,  # type: ignore
    )

    # Extract the merged content
    merged_text = ""
    if response.choices and len(response.choices) > 0:
        # Handle different response formats
        if hasattr(response.choices[0], "message"):
            merged_text = response.choices[0].message.content
        elif hasattr(response.choices[0], "text"):
            merged_text = response.choices[0].text
        else:
            # Fallback if the structure is different
            merged_text = str(response.choices[0])

    def float_or_datetime(m: dict, key: str) -> float:
        val = m.get(key, time.time())
        if val is None:
            return time.time()
        if isinstance(val, datetime):
            return int(val.timestamp())
        return float(val)

    # Use the earliest creation timestamp
    created_at = min(float_or_datetime(m, "created_at") for m in memories)

    # Use the most recent last_accessed timestamp
    last_accessed = max(float_or_datetime(m, "last_accessed") for m in memories)

    # Prefer non-empty namespace, user_id, session_id from memories
    namespace = next((m["namespace"] for m in memories if m.get("namespace")), None)
    user_id = next((m["user_id"] for m in memories if m.get("user_id")), None)
    session_id = next((m["session_id"] for m in memories if m.get("session_id")), None)

    # Get the memory type from the first memory
    memory_type = next(
        (m["memory_type"] for m in memories if m.get("memory_type")), "semantic"
    )

    # Get the discrete_memory_extracted from the first memory
    discrete_memory_extracted = next(
        (
            m["discrete_memory_extracted"]
            for m in memories
            if m.get("discrete_memory_extracted")
        ),
        "t",
    )

    # Create the merged memory
    merged_memory = {
        "text": merged_text.strip(),
        "id_": str(ULID()),
        "user_id": user_id,
        "session_id": session_id,
        "namespace": namespace,
        "created_at": created_at,
        "last_accessed": last_accessed,
        "updated_at": int(datetime.now(UTC).timestamp()),
        "topics": list(all_topics) if all_topics else None,
        "entities": list(all_entities) if all_entities else None,
        "memory_type": memory_type,
        "discrete_memory_extracted": discrete_memory_extracted,
    }

    # Generate a new hash for the merged memory
    merged_memory["memory_hash"] = generate_memory_hash(merged_memory)

    return merged_memory


async def compact_long_term_memories(
    limit: int = 1000,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    llm_client: OpenAIClientWrapper | AnthropicClientWrapper | None = None,
    redis_client: Redis | None = None,
    vector_distance_threshold: float = 0.12,
    compact_hash_duplicates: bool = True,
    compact_semantic_duplicates: bool = True,
) -> int:
    """
    Compact long-term memories by merging duplicates and semantically similar memories.

    This function can identify and merge two types of duplicate memories:
    1. Hash-based duplicates: Memories with identical content (using memory_hash)
    2. Semantic duplicates: Memories with similar meaning but different text

    Returns the count of remaining memories after compaction.
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    if not llm_client:
        llm_client = await get_model_client(model_name="gpt-4o-mini")

    logger.info(
        f"Starting memory compaction: namespace={namespace}, "
        f"user_id={user_id}, session_id={session_id}, "
        f"hash_duplicates={compact_hash_duplicates}, "
        f"semantic_duplicates={compact_semantic_duplicates}"
    )

    # Build filters for memory queries
    filters = []
    if namespace:
        filters.append(f"@namespace:{{{namespace}}}")
    if user_id:
        filters.append(f"@user_id:{{{user_id}}}")
    if session_id:
        filters.append(f"@session_id:{{{session_id}}}")

    filter_str = " ".join(filters) if filters else "*"

    # Track metrics
    memories_merged = 0
    start_time = time.time()

    # Step 1: Compact hash-based duplicates using Redis aggregation
    if compact_hash_duplicates:
        logger.info("Starting hash-based duplicate compaction")
        try:
            index_name = Keys.search_index_name()

            # Create aggregation query to group by memory_hash and find duplicates
            agg_query = (
                f"FT.AGGREGATE {index_name} {filter_str} "
                "GROUPBY 1 @memory_hash "
                "REDUCE COUNT 0 AS count "
                'FILTER "@count>1" '  # Only groups with more than 1 memory
                "SORTBY 2 @count DESC "
                f"LIMIT 0 {limit}"
            )

            # Execute aggregation to find duplicate groups
            duplicate_groups = await redis_client.execute_command(agg_query)

            if duplicate_groups and duplicate_groups[0] > 0:
                num_groups = duplicate_groups[0]
                logger.info(f"Found {num_groups} groups of hash-based duplicates")

                # Process each group of duplicates
                for i in range(1, len(duplicate_groups), 2):
                    try:
                        # Get the hash and count from aggregation results
                        group_data = duplicate_groups[i]
                        memory_hash = None
                        count = 0

                        for j in range(0, len(group_data), 2):
                            if group_data[j] == b"memory_hash":
                                memory_hash = group_data[j + 1].decode()
                            elif group_data[j] == b"count":
                                count = int(group_data[j + 1])

                        if not memory_hash or count <= 1:
                            continue

                        # Find all memories with this hash
                        # Use FT.SEARCH to find the actual memories with this hash
                        # TODO: Use RedisVL index
                        if filters:
                            # Combine hash query with filters using boolean AND
                            query_expr = f"(@memory_hash:{{{memory_hash}}}) ({' '.join(filters)})"
                        else:
                            query_expr = f"@memory_hash:{{{memory_hash}}}"

                        search_results = await redis_client.execute_command(
                            "FT.SEARCH",
                            index_name,
                            f"'{query_expr}'",
                            "RETURN",
                            "6",
                            "id_",
                            "text",
                            "last_accessed",
                            "created_at",
                            "user_id",
                            "session_id",
                            "SORTBY",
                            "last_accessed",
                            "ASC",
                        )

                        if search_results and search_results[0] > 1:
                            num_duplicates = search_results[0]

                            # Keep the newest memory (last in sorted results)
                            # and delete the rest
                            memories_to_delete = []

                            for j in range(1, len(search_results), 2):
                                # Skip the last item (newest) which we'll keep
                                if j < (int(num_duplicates) - 1) * 2 + 1:
                                    key = search_results[j].decode()
                                    memories_to_delete.append(key)

                            # Delete older duplicates
                            if memories_to_delete:
                                pipeline = redis_client.pipeline()
                                for key in memories_to_delete:
                                    pipeline.delete(key)

                                await pipeline.execute()
                                memories_merged += len(memories_to_delete)
                                logger.info(
                                    f"Deleted {len(memories_to_delete)} hash-based duplicates "
                                    f"with hash {memory_hash}"
                                )
                    except Exception as e:
                        logger.error(f"Error processing duplicate group: {e}")

            logger.info(
                f"Completed hash-based deduplication. Merged {memories_merged} memories."
            )
        except Exception as e:
            logger.error(f"Error during hash-based duplicate compaction: {e}")

    # Step 2: Compact semantic duplicates using vector search
    semantic_memories_merged = 0
    if compact_semantic_duplicates:
        logger.info("Starting semantic duplicate compaction")
        # Get the correct index name
        index_name = Keys.search_index_name()
        logger.info(f"Using index '{index_name}' for semantic duplicate compaction.")

        # Check if the index exists before proceeding
        try:
            await redis_client.execute_command(f"FT.INFO {index_name}")
        except Exception as info_e:
            if "unknown index name" in str(info_e).lower():
                logger.info(f"Search index {index_name} doesn't exist, creating it")
                # Ensure 'get_search_index' is called with the correct name to create it if needed
                await ensure_search_index_exists(redis_client, index_name=index_name)
            else:
                logger.warning(
                    f"Error checking index '{index_name}': {info_e} - attempting to proceed."
                )

        # Get all memories using the vector store adapter
        try:
            # Convert filters to adapter format
            namespace_filter = None
            user_id_filter = None
            session_id_filter = None

            if namespace:
                from agent_memory_server.filters import Namespace

                namespace_filter = Namespace(eq=namespace)
            if user_id:
                from agent_memory_server.filters import UserId

                user_id_filter = UserId(eq=user_id)
            if session_id:
                from agent_memory_server.filters import SessionId

                session_id_filter = SessionId(eq=session_id)

            # Use vectorstore adapter to get all memories
            adapter = await get_vectorstore_adapter()
            search_result = await adapter.search_memories(
                query="",  # Empty query to get all matching filter criteria
                namespace=namespace_filter,
                user_id=user_id_filter,
                session_id=session_id_filter,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Error searching for memories: {e}")
            search_result = None

        if search_result and search_result.memories:
            logger.info(
                f"Found {search_result.total} memories to check for semantic duplicates"
            )

            # Process memories in batches to avoid overloading
            batch_size = 50
            processed_ids = set()  # Track which memories have been processed

            memories_list = search_result.memories
            for i in range(0, len(memories_list), batch_size):
                batch = memories_list[i : i + batch_size]

                for memory_result in batch:
                    memory_id = memory_result.id

                    # Skip if already processed
                    if memory_id in processed_ids:
                        continue

                    # Convert MemoryRecordResult to MemoryRecord for deduplication
                    memory_obj = MemoryRecord(
                        id=memory_result.id,
                        text=memory_result.text,
                        user_id=memory_result.user_id,
                        session_id=memory_result.session_id,
                        namespace=memory_result.namespace,
                        created_at=memory_result.created_at,
                        last_accessed=memory_result.last_accessed,
                        topics=memory_result.topics or [],
                        entities=memory_result.entities or [],
                        memory_type=memory_result.memory_type,  # type: ignore
                        discrete_memory_extracted=memory_result.discrete_memory_extracted,  # type: ignore
                    )

                    # Add this memory to processed list
                    processed_ids.add(memory_id)

                    # Check for semantic duplicates
                    (
                        merged_memory,
                        was_merged,
                    ) = await deduplicate_by_semantic_search(
                        memory=memory_obj,
                        redis_client=redis_client,
                        llm_client=llm_client,
                        namespace=namespace,
                        user_id=user_id,
                        session_id=session_id,
                        vector_distance_threshold=vector_distance_threshold,
                    )

                    if was_merged:
                        semantic_memories_merged += 1
                        # Delete the original memory using the adapter
                        await adapter.delete_memories([memory_id])

                        # Re-index the merged memory
                        if merged_memory:
                            await index_long_term_memories(
                                [merged_memory],
                                redis_client=redis_client,
                                deduplicate=False,  # Already deduplicated
                            )
        logger.info(
            f"Completed semantic deduplication. Merged {semantic_memories_merged} memories."
        )

    # Get the count of remaining memories
    total_memories = await count_long_term_memories(
        namespace=namespace,
        user_id=user_id,
        session_id=session_id,
        redis_client=redis_client,
    )

    end_time = time.time()
    total_merged = memories_merged + semantic_memories_merged

    logger.info(
        f"Memory compaction completed in {end_time - start_time:.2f}s. "
        f"Merged {total_merged} memories. "
        f"{total_memories} memories remain."
    )

    return total_memories


async def index_long_term_memories(
    memories: list[MemoryRecord],
    redis_client: Redis | None = None,
    deduplicate: bool = False,
    vector_distance_threshold: float = 0.12,
    llm_client: Any = None,
) -> None:
    """
    Index long-term memories using the pluggable VectorStore adapter.

    Args:
        memories: List of long-term memories to index
        redis_client: Optional Redis client (kept for compatibility, may be unused depending on backend)
        deduplicate: Whether to deduplicate memories before indexing
        vector_distance_threshold: Threshold for semantic similarity
        llm_client: Optional LLM client for semantic merging
    """
    background_tasks = get_background_tasks()

    # Process memories for deduplication if requested
    processed_memories = []
    if deduplicate:
        # Get Redis client for deduplication operations (still needed for existing dedup logic)
        redis = redis_client or await get_redis_conn()
        model_client = llm_client or await get_model_client(
            model_name=settings.generation_model
        )

        for memory in memories:
            current_memory = memory
            was_deduplicated = False

            # Check for id-based duplicates FIRST (Stage 2 requirement)
            if not was_deduplicated:
                deduped_memory, was_overwrite = await deduplicate_by_id(
                    memory=current_memory,
                    redis_client=redis,
                )
                if was_overwrite:
                    # This overwrote an existing memory with the same id
                    current_memory = deduped_memory or current_memory
                    logger.info(f"Overwrote memory with id {memory.id}")
                else:
                    current_memory = deduped_memory or current_memory

            # Check for hash-based duplicates
            if not was_deduplicated:
                deduped_memory, was_dup = await deduplicate_by_hash(
                    memory=current_memory,
                    redis_client=redis,
                )
                if was_dup:
                    # This is a duplicate, skip it
                    was_deduplicated = True
                else:
                    current_memory = deduped_memory or current_memory

            # Check for semantic duplicates
            if not was_deduplicated:
                deduped_memory, was_merged = await deduplicate_by_semantic_search(
                    memory=current_memory,
                    redis_client=redis,
                    llm_client=model_client,
                    vector_distance_threshold=vector_distance_threshold,
                )
                if was_merged:
                    current_memory = deduped_memory or current_memory

            # Add the memory to be indexed if not a pure duplicate
            if not was_deduplicated:
                current_memory.discrete_memory_extracted = "t"
                processed_memories.append(current_memory)
    else:
        processed_memories = memories

    # If all memories were duplicates, we're done
    if not processed_memories:
        logger.info("All memories were duplicates, nothing to index")
        return

    # Get the VectorStore adapter and add memories
    adapter = await get_vectorstore_adapter()

    # Add memories to the vector store
    try:
        ids = await adapter.add_memories(processed_memories)
        logger.info(f"Indexed {len(processed_memories)} memories with IDs: {ids}")
    except Exception as e:
        logger.error(f"Error indexing memories: {e}")
        raise

    # Schedule background tasks for topic/entity extraction
    for memory in processed_memories:
        memory_id = memory.id or str(ULID())
        await background_tasks.add_task(
            extract_memory_structure, memory_id, memory.text, memory.namespace
        )

    if settings.enable_discrete_memory_extraction:
        # Extract discrete memories from the indexed messages and persist
        # them as separate long-term memory records. This process also
        # runs deduplication if requested.
        await background_tasks.add_task(
            extract_discrete_memories,
            deduplicate=deduplicate,
        )


async def search_long_term_memories(
    text: str,
    session_id: SessionId | None = None,
    user_id: UserId | None = None,
    namespace: Namespace | None = None,
    created_at: CreatedAt | None = None,
    last_accessed: LastAccessed | None = None,
    topics: Topics | None = None,
    entities: Entities | None = None,
    distance_threshold: float | None = None,
    memory_type: MemoryType | None = None,
    event_date: EventDate | None = None,
    memory_hash: MemoryHash | None = None,
    limit: int = 10,
    offset: int = 0,
) -> MemoryRecordResults:
    """
    Search for long-term memories using the pluggable VectorStore adapter.

    Args:
        text: Search query text
        redis: Redis client (kept for compatibility but may be unused depending on backend)
        session_id: Optional session ID filter
        user_id: Optional user ID filter
        namespace: Optional namespace filter
        created_at: Optional created at filter
        last_accessed: Optional last accessed filter
        topics: Optional topics filter
        entities: Optional entities filter
        distance_threshold: Optional similarity threshold
        memory_type: Optional memory type filter
        event_date: Optional event date filter
        memory_hash: Optional memory hash filter
        limit: Maximum number of results
        offset: Offset for pagination

    Returns:
        MemoryRecordResults containing matching memories
    """
    # Get the VectorStore adapter
    adapter = await get_vectorstore_adapter()

    # Delegate search to the adapter
    return await adapter.search_memories(
        query=text,
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
        created_at=created_at,
        last_accessed=last_accessed,
        topics=topics,
        entities=entities,
        memory_type=memory_type,
        event_date=event_date,
        memory_hash=memory_hash,
        distance_threshold=distance_threshold,
        limit=limit,
        offset=offset,
    )


async def search_memories(
    text: str,
    redis: Redis,
    session_id: SessionId | None = None,
    user_id: UserId | None = None,
    namespace: Namespace | None = None,
    created_at: CreatedAt | None = None,
    last_accessed: LastAccessed | None = None,
    topics: Topics | None = None,
    entities: Entities | None = None,
    distance_threshold: float | None = None,
    memory_type: MemoryType | None = None,
    event_date: EventDate | None = None,
    limit: int = 10,
    offset: int = 0,
    include_working_memory: bool = True,
    include_long_term_memory: bool = True,
) -> MemoryRecordResults:
    """
    Search for memories across both working memory and long-term storage.

    This provides a search interface that spans all memory types and locations.

    Args:
        text: Search query text
        redis: Redis client
        session_id: Filter by session ID
        user_id: Filter by user ID
        namespace: Filter by namespace
        created_at: Filter by creation date
        last_accessed: Filter by last access date
        topics: Filter by topics
        entities: Filter by entities
        distance_threshold: Distance threshold for semantic search
        memory_type: Filter by memory type
        limit: Maximum number of results to return
        offset: Offset for pagination
        include_working_memory: Whether to include working memory in search
        include_long_term_memory: Whether to include long-term memory in search

    Returns:
        Combined search results from both working and long-term memory
    """
    from agent_memory_server import working_memory

    all_results = []
    total_count = 0

    # Search long-term memory if enabled
    if include_long_term_memory and settings.long_term_memory:
        try:
            long_term_results = await search_long_term_memories(
                text=text,
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                created_at=created_at,
                last_accessed=last_accessed,
                topics=topics,
                entities=entities,
                distance_threshold=distance_threshold,
                memory_type=memory_type,
                event_date=event_date,
                limit=limit,
                offset=offset,
            )
            all_results.extend(long_term_results.memories)
            total_count += long_term_results.total

            logger.info(
                f"Found {len(long_term_results.memories)} long-term memory results"
            )
        except Exception as e:
            logger.error(f"Error searching long-term memory: {e}")

    # Search working memory if enabled
    if include_working_memory:
        try:
            # Get all working memory sessions if no specific session filter
            if session_id and hasattr(session_id, "eq") and session_id.eq:
                # Search specific session
                session_ids_to_search = [session_id.eq]
            else:
                # Get all sessions for broader search
                from agent_memory_server import working_memory

                namespace_value = None
                if namespace and hasattr(namespace, "eq"):
                    namespace_value = namespace.eq

                _, session_ids_to_search = await working_memory.list_sessions(
                    redis=redis,
                    limit=1000,  # Get a reasonable number of sessions
                    offset=0,
                    namespace=namespace_value,
                )

            # Search working memory in relevant sessions
            working_memory_results = []
            for session_id_str in session_ids_to_search:
                try:
                    working_mem = await working_memory.get_working_memory(
                        session_id=session_id_str,
                        namespace=namespace_value if namespace else None,
                        redis_client=redis,
                    )

                    if working_mem and working_mem.memories:
                        # Filter memories based on criteria
                        filtered_memories = working_mem.memories

                        # Apply memory_type filter
                        if memory_type:
                            if hasattr(memory_type, "eq") and memory_type.eq:
                                filtered_memories = [
                                    mem
                                    for mem in filtered_memories
                                    if mem.memory_type == memory_type.eq
                                ]
                            elif hasattr(memory_type, "any") and memory_type.any:
                                filtered_memories = [
                                    mem
                                    for mem in filtered_memories
                                    if mem.memory_type in memory_type.any
                                ]

                        # Apply user_id filter
                        if user_id and hasattr(user_id, "eq") and user_id.eq:
                            filtered_memories = [
                                mem
                                for mem in filtered_memories
                                if mem.user_id == user_id.eq
                            ]

                        # Convert to MemoryRecordResult format and add to results
                        for memory in filtered_memories:
                            # Simple text matching for working memory (no vector search)
                            if text.lower() in memory.text.lower():
                                working_memory_results.append(
                                    MemoryRecordResult(
                                        id=memory.id or "",  # Use id instead of id_
                                        text=memory.text,
                                        dist=0.0,  # No vector distance for working memory
                                        created_at=memory.created_at or 0,
                                        updated_at=memory.updated_at or 0,
                                        last_accessed=memory.last_accessed or 0,
                                        user_id=memory.user_id,
                                        session_id=session_id_str,
                                        namespace=memory.namespace,
                                        topics=memory.topics or [],
                                        entities=memory.entities or [],
                                        memory_hash="",  # Working memory doesn't have hash
                                        memory_type=memory.memory_type,
                                        persisted_at=memory.persisted_at,
                                        event_date=memory.event_date,
                                    )
                                )

                except Exception as e:
                    logger.warning(
                        f"Error searching working memory for session {session_id_str}: {e}"
                    )
                    continue

            all_results.extend(working_memory_results)
            total_count += len(working_memory_results)

            logger.info(f"Found {len(working_memory_results)} working memory results")

        except Exception as e:
            logger.error(f"Error searching working memory: {e}")

    # Sort combined results by relevance (distance for long-term, text match quality for working)
    # For simplicity, put working memory results first (distance 0.0), then long-term by distance
    all_results.sort(key=lambda x: (x.dist, x.created_at))

    # Apply pagination to combined results
    paginated_results = all_results[offset : offset + limit] if all_results else []

    logger.info(
        f"Memory search found {len(all_results)} total results, returning {len(paginated_results)}"
    )

    return MemoryRecordResults(
        total=total_count,
        memories=paginated_results,
        next_offset=offset + limit if offset + limit < len(all_results) else None,
    )


async def count_long_term_memories(
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    redis_client: Redis | None = None,
) -> int:
    """
    Count the total number of long-term memories matching the given filters.

    Uses the pluggable VectorStore adapter instead of direct Redis calls.

    Args:
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        redis_client: Optional Redis client (for compatibility - not used by adapter)

    Returns:
        Total count of memories matching filters
    """
    # Get the VectorStore adapter
    adapter = await get_vectorstore_adapter()

    # Delegate to the adapter
    return await adapter.count_memories(
        namespace=namespace,
        user_id=user_id,
        session_id=session_id,
    )


async def deduplicate_by_hash(
    memory: MemoryRecord,
    redis_client: Redis | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> tuple[MemoryRecord | None, bool]:
    """
    Check if a memory has hash-based duplicates and handle accordingly.

    Args:
        memory: The memory to check for duplicates
        redis_client: Optional Redis client
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter

    Returns:
        Tuple of (memory to save (if any), was_duplicate)
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    # Generate hash for the memory
    memory_hash = generate_memory_hash(
        {
            "text": memory.text,
            "user_id": memory.user_id or "",
            "session_id": memory.session_id or "",
        }
    )

    # Use vectorstore adapter to search for memories with the same hash
    # Build filter objects
    namespace_filter = None
    if namespace or memory.namespace:
        namespace_filter = Namespace(eq=namespace or memory.namespace)

    user_id_filter = None
    if user_id or memory.user_id:
        user_id_filter = UserId(eq=user_id or memory.user_id)

    session_id_filter = None
    if session_id or memory.session_id:
        session_id_filter = SessionId(eq=session_id or memory.session_id)

    # Create memory hash filter
    memory_hash_filter = MemoryHash(eq=memory_hash)

    # Use vectorstore adapter to search for memories with the same hash
    adapter = await get_vectorstore_adapter()

    # Search for existing memories with the same hash
    # Use a dummy query since we're filtering by hash, not doing semantic search
    results = await adapter.search_memories(
        query="",  # Empty query since we're filtering by hash
        session_id=session_id_filter,
        user_id=user_id_filter,
        namespace=namespace_filter,
        memory_hash=memory_hash_filter,
        limit=1,  # We only need to know if one exists
    )

    if results.memories and len(results.memories) > 0:
        # Found existing memory with the same hash
        logger.info(f"Found existing memory with hash {memory_hash}")

        # Update the last_accessed timestamp of the existing memory
        existing_memory = results.memories[0]
        if existing_memory.id:
            # Use the memory key format to update last_accessed
            existing_key = Keys.memory_key(
                existing_memory.id, existing_memory.namespace
            )
            await redis_client.hset(
                existing_key,
                "last_accessed",
                str(int(datetime.now(UTC).timestamp())),
            )  # type: ignore

            # Don't save this memory, it's a duplicate
            return None, True
    # No duplicates found, return the original memory
    return memory, False


async def deduplicate_by_id(
    memory: MemoryRecord,
    redis_client: Redis | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> tuple[MemoryRecord | None, bool]:
    """
    Check if a memory with the same id exists and handle accordingly.
    This implements Stage 2 requirement: use id as the basis for deduplication and overwrites.

    Args:
        memory: The memory to check for id duplicates
        redis_client: Optional Redis client
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter

    Returns:
        Tuple of (memory to save (potentially updated), was_overwrite)
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    # If no id, can't deduplicate by id
    if not memory.id:
        return memory, False

    # Build filters for the search
    filters = []
    if namespace or memory.namespace:
        ns = namespace or memory.namespace
        filters.append(f"@namespace:{{{ns}}}")
    if user_id or memory.user_id:
        uid = user_id or memory.user_id
        filters.append(f"@user_id:{{{uid}}}")
    if session_id or memory.session_id:
        sid = session_id or memory.session_id
        filters.append(f"@session_id:{{{sid}}}")

    filter_str = " ".join(filters) if filters else ""

    # Search for existing memories with the same id
    index_name = Keys.search_index_name()

    # Use FT.SEARCH to find memories with this id
    # TODO: Use RedisVL
    if filter_str:
        # Combine the id query with filters - Redis FT.SEARCH uses implicit AND between terms
        query_expr = f"@id:{{{memory.id}}} {filter_str}"
    else:
        query_expr = f"@id:{{{memory.id}}}"

    search_results = await redis_client.execute_command(
        "FT.SEARCH",
        index_name,
        f"'{query_expr}'",
        "RETURN",
        "2",
        "id_",
        "persisted_at",
        "SORTBY",
        "last_accessed",
        "DESC",
    )

    if search_results and search_results[0] > 0:
        # Found existing memory with the same id
        logger.info(f"Found existing memory with id {memory.id}, will overwrite")

        # Get the existing memory key and persisted_at
        existing_key = search_results[1]
        if isinstance(existing_key, bytes):
            existing_key = existing_key.decode()

        existing_persisted_at = "0"
        if len(search_results) > 2:
            existing_persisted_at = search_results[2]
            if isinstance(existing_persisted_at, bytes):
                existing_persisted_at = existing_persisted_at.decode()

        # Delete the existing memory
        await redis_client.delete(existing_key)

        # If the existing memory was already persisted, preserve that timestamp
        if existing_persisted_at != "0":
            memory.persisted_at = datetime.fromtimestamp(int(existing_persisted_at))

        # Return the memory to be saved (overwriting the existing one)
        return memory, True

    # No existing memory with this id found
    return memory, False


async def deduplicate_by_semantic_search(
    memory: MemoryRecord,
    redis_client: Redis | None = None,
    llm_client: Any = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    vector_distance_threshold: float = 0.12,
) -> tuple[MemoryRecord | None, bool]:
    """
    Check if a memory has semantic duplicates and merge if found.

    Args:
        memory: The memory to check for semantic duplicates
        redis_client: Optional Redis client
        llm_client: Optional LLM client for merging
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        vector_distance_threshold: Distance threshold for semantic similarity

    Returns:
        Tuple of (memory to save (potentially merged), was_merged)
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    if not llm_client:
        llm_client = await get_model_client(model_name="gpt-4o-mini")

    # Use vector store adapter to find semantically similar memories
    adapter = await get_vectorstore_adapter()

    # Convert filters to adapter format
    namespace_filter = None
    user_id_filter = None
    session_id_filter = None

    # TODO: Refactor to avoid inline imports (fix circular imports)
    if namespace or memory.namespace:
        from agent_memory_server.filters import Namespace

        namespace_filter = Namespace(eq=namespace or memory.namespace)
    if user_id or memory.user_id:
        from agent_memory_server.filters import UserId

        user_id_filter = UserId(eq=user_id or memory.user_id)
    if session_id or memory.session_id:
        from agent_memory_server.filters import SessionId

        session_id_filter = SessionId(eq=session_id or memory.session_id)

    # Use the vectorstore adapter for semantic search
    search_result = await adapter.search_memories(
        query=memory.text,  # Use memory text for semantic search
        namespace=namespace_filter,
        user_id=user_id_filter,
        session_id=session_id_filter,
        distance_threshold=vector_distance_threshold,
        limit=5,
    )

    vector_search_result = search_result.memories if search_result else []

    if vector_search_result and len(vector_search_result) > 0:
        # Found semantically similar memories
        similar_memory_ids = []
        similar_memories_data = []

        for similar_memory_result in vector_search_result:
            similar_memory_ids.append(similar_memory_result.id)

            # Convert MemoryRecordResult to dict format for merge_memories_with_llm
            similar_memory_dict = {
                "id_": similar_memory_result.id,
                "text": similar_memory_result.text,
                "user_id": similar_memory_result.user_id,
                "session_id": similar_memory_result.session_id,
                "namespace": similar_memory_result.namespace,
                "created_at": int(similar_memory_result.created_at.timestamp()),
                "last_accessed": int(similar_memory_result.last_accessed.timestamp()),
                "topics": similar_memory_result.topics or [],
                "entities": similar_memory_result.entities or [],
                "memory_type": similar_memory_result.memory_type,
                "discrete_memory_extracted": similar_memory_result.discrete_memory_extracted,
            }
            similar_memories_data.append(similar_memory_dict)

        # Merge the memories
        merged_memory = await merge_memories_with_llm(
            [memory.model_dump()] + similar_memories_data,
            llm_client=llm_client,
        )

        # Convert back to MemoryRecord
        merged_memory_obj = MemoryRecord(
            id=memory.id or str(ULID()),
            text=merged_memory["text"],
            user_id=merged_memory["user_id"],
            session_id=merged_memory["session_id"],
            namespace=merged_memory["namespace"],
            created_at=merged_memory["created_at"],
            last_accessed=merged_memory["last_accessed"],
            topics=merged_memory.get("topics", []),
            entities=merged_memory.get("entities", []),
            memory_type=merged_memory.get("memory_type", "semantic"),
            discrete_memory_extracted=merged_memory.get(
                "discrete_memory_extracted", "t"
            ),
        )

        # Delete the similar memories using the adapter
        if similar_memory_ids:
            await adapter.delete_memories(similar_memory_ids)

        logger.info(
            f"Merged new memory with {len(similar_memory_ids)} semantic duplicates"
        )
        return merged_memory_obj, True

    # No similar memories found or error occurred
    return memory, False


async def promote_working_memory_to_long_term(
    session_id: str,
    namespace: str | None = None,
    redis_client: Redis | None = None,
) -> int:
    """
    Promote eligible working memory records to long-term storage.

    This function:
    1. Identifies memory records with no persisted_at from working memory
    2. For message records, runs extraction to generate semantic/episodic memories
    3. Uses id to detect and replace duplicates in long-term memory
    4. Persists the record and stamps it with persisted_at = now()
    5. Updates the working memory session store to reflect new timestamps

    Args:
        session_id: The session ID to promote memories from
        namespace: Optional namespace for the session
        redis_client: Optional Redis client to use

    Returns:
        Number of memories promoted to long-term storage
    """

    from agent_memory_server import working_memory
    from agent_memory_server.utils.redis import get_redis_conn

    redis = redis_client or await get_redis_conn()

    # Get current working memory
    current_working_memory = await working_memory.get_working_memory(
        session_id=session_id,
        namespace=namespace,
        redis_client=redis,
    )

    if not current_working_memory:
        logger.debug(f"No working memory found for session {session_id}")
        return 0

    # Find memories with no persisted_at (eligible for promotion)
    unpersisted_memories = [
        memory
        for memory in current_working_memory.memories
        if memory.persisted_at is None
    ]

    if not unpersisted_memories:
        logger.debug(f"No unpersisted memories found in session {session_id}")
        return 0

    logger.info(
        f"Promoting {len(unpersisted_memories)} memories from session {session_id}"
    )

    promoted_count = 0
    updated_memories = []
    extracted_memories = []

    # Stage 7: Extract memories from message records if enabled
    if settings.enable_discrete_memory_extraction:
        message_memories = [
            memory
            for memory in unpersisted_memories
            if memory.memory_type == MemoryTypeEnum.MESSAGE
            and memory.discrete_memory_extracted == "f"
        ]

        if message_memories:
            logger.info(
                f"Extracting memories from {len(message_memories)} message records"
            )
            extracted_memories = await extract_memories_from_messages(message_memories)

            # Mark message memories as extracted
            for message_memory in message_memories:
                message_memory.discrete_memory_extracted = "t"

    for memory in current_working_memory.memories:
        if memory.persisted_at is None:
            # This memory needs to be promoted

            # Check for id-based duplicates and handle accordingly
            deduped_memory, was_overwrite = await deduplicate_by_id(
                memory=memory,
                redis_client=redis,
            )

            # Set persisted_at timestamp
            current_memory = deduped_memory or memory
            current_memory.persisted_at = datetime.now(UTC)

            # Index the memory in long-term storage
            await index_long_term_memories(
                [current_memory],
                redis_client=redis,
                deduplicate=False,  # Already deduplicated by id
            )

            promoted_count += 1
            updated_memories.append(current_memory)

            if was_overwrite:
                logger.info(f"Overwrote existing memory with id {memory.id}")
            else:
                logger.info(f"Promoted new memory with id {memory.id}")
        else:
            # This memory is already persisted, keep as-is
            updated_memories.append(memory)

    # Add extracted memories to working memory for future promotion
    if extracted_memories:
        logger.info(
            f"Adding {len(extracted_memories)} extracted memories to working memory"
        )
        updated_memories.extend(extracted_memories)

    # Update working memory with the new persisted_at timestamps and extracted memories
    if promoted_count > 0 or extracted_memories:
        updated_working_memory = current_working_memory.model_copy()
        updated_working_memory.memories = updated_memories
        updated_working_memory.updated_at = datetime.now(UTC)

        await working_memory.set_working_memory(
            working_memory=updated_working_memory,
            redis_client=redis,
        )

        logger.info(
            f"Successfully promoted {promoted_count} memories to long-term storage"
            + (
                f" and extracted {len(extracted_memories)} new memories"
                if extracted_memories
                else ""
            )
        )

    return promoted_count


async def extract_memories_from_messages(
    message_records: list[MemoryRecord],
    llm_client: OpenAIClientWrapper | AnthropicClientWrapper | None = None,
) -> list[MemoryRecord]:
    """
    Extract semantic and episodic memories from message records.

    Args:
        message_records: List of message-type memory records to extract from
        llm_client: Optional LLM client for extraction

    Returns:
        List of extracted memory records with extracted_from field populated
    """
    if not message_records:
        return []

    client = llm_client or await get_model_client(settings.generation_model)
    extracted_memories = []

    for message_record in message_records:
        if message_record.memory_type != MemoryTypeEnum.MESSAGE:
            continue

        try:
            # Use LLM to extract memories from the message
            response = await client.create_chat_completion(
                model=settings.generation_model,
                prompt=WORKING_MEMORY_EXTRACTION_PROMPT.format(
                    message=message_record.text
                ),
                response_format={"type": "json_object"},
            )

            extraction_result = json.loads(response.choices[0].message.content)

            if "memories" in extraction_result and extraction_result["memories"]:
                for memory_data in extraction_result["memories"]:
                    # Parse event_date if provided
                    event_date = None
                    if memory_data.get("event_date"):
                        try:
                            event_date = datetime.fromisoformat(
                                memory_data["event_date"].replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Could not parse event_date '{memory_data.get('event_date')}': {e}"
                            )

                    # Create a new memory record from the extraction
                    extracted_memory = MemoryRecord(
                        id=str(ULID()),  # Server-generated ID
                        text=memory_data["text"],
                        memory_type=memory_data.get("type", "semantic"),
                        topics=memory_data.get("topics", []),
                        entities=memory_data.get("entities", []),
                        extracted_from=[message_record.id] if message_record.id else [],
                        event_date=event_date,
                        # Inherit context from the source message
                        session_id=message_record.session_id,
                        user_id=message_record.user_id,
                        namespace=message_record.namespace,
                        persisted_at=None,  # Will be set during promotion
                        discrete_memory_extracted="t",
                    )
                    extracted_memories.append(extracted_memory)

                logger.info(
                    f"Extracted {len(extraction_result['memories'])} memories from message {message_record.id}"
                )

        except Exception as e:
            logger.error(
                f"Error extracting memories from message {message_record.id}: {e}"
            )
            continue

    return extracted_memories
