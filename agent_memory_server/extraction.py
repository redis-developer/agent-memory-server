import json
import os
from typing import TYPE_CHECKING, Any

import ulid
from redis.asyncio.client import Redis
from tenacity.asyncio import AsyncRetrying
from tenacity.stop import stop_after_attempt
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from agent_memory_server.config import settings
from agent_memory_server.filters import DiscreteMemoryExtracted
from agent_memory_server.llms import (
    AnthropicClientWrapper,
    OpenAIClientWrapper,
    get_model_client,
)
from agent_memory_server.logging import get_logger
from agent_memory_server.models import MemoryRecord
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.redis import get_redis_conn


if TYPE_CHECKING:
    from bertopic import BERTopic


logger = get_logger(__name__)

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global model instances
_topic_model: "BERTopic | None" = None
_ner_model: Any | None = None
_ner_tokenizer: Any | None = None


def get_topic_model() -> "BERTopic":
    """
    Get or initialize the BERTopic model.

    Returns:
        The BERTopic model instance
    """
    from bertopic import BERTopic

    global _topic_model
    if _topic_model is None:
        # TODO: Expose this as a config option
        _topic_model = BERTopic.load(
            settings.topic_model, embedding_model="all-MiniLM-L6-v2"
        )
    return _topic_model  # type: ignore


def get_ner_model() -> Any:
    """
    Get or initialize the NER model and tokenizer.

    Returns:
        The NER pipeline instance
    """
    global _ner_model, _ner_tokenizer
    if _ner_model is None:
        _ner_tokenizer = AutoTokenizer.from_pretrained(settings.ner_model)
        _ner_model = AutoModelForTokenClassification.from_pretrained(settings.ner_model)
    return pipeline("ner", model=_ner_model, tokenizer=_ner_tokenizer)


def extract_entities(text: str) -> list[str]:
    """
    Extract named entities from text using the NER model.

    TODO: Cache this output.

    Args:
        text: The text to extract entities from

    Returns:
        List of unique entity names
    """
    try:
        ner = get_ner_model()
        results = ner(text)

        # Group tokens by entity
        current_entity = []
        entities = []

        for result in results:
            if result["word"].startswith("##"):
                # This is a continuation of the previous entity
                current_entity.append(result["word"][2:])
            else:
                # This is a new entity
                if current_entity:
                    entities.append("".join(current_entity))
                current_entity = [result["word"]]

        # Add the last entity if exists
        if current_entity:
            entities.append("".join(current_entity))

        return list(set(entities))  # Remove duplicates

    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return []


async def extract_topics_llm(
    text: str,
    num_topics: int | None = None,
    client: OpenAIClientWrapper | AnthropicClientWrapper | None = None,
) -> list[str]:
    """
    Extract topics from text using the LLM model.
    """
    _client = client or await get_model_client(settings.topic_model)
    _num_topics = num_topics if num_topics is not None else settings.top_k_topics

    prompt = f"""
    Extract the topic {_num_topics} topics from the following text:
    {text}

    Return a list of topics in JSON format, for example:
    {{
        "topics": ["topic1", "topic2", "topic3"]
    }}
    """
    topics = []

    async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
        with attempt:
            response = await _client.create_chat_completion(
                model=settings.generation_model,
                prompt=prompt,
                response_format={"type": "json_object"},
            )
            try:
                topics = json.loads(response.choices[0].message.content)["topics"]
            except (json.JSONDecodeError, KeyError):
                logger.error(
                    f"Error decoding JSON: {response.choices[0].message.content}"
                )
                topics = []
            if topics:
                topics = topics[:_num_topics]

    return topics


def extract_topics_bertopic(text: str, num_topics: int | None = None) -> list[str]:
    """
    Extract topics from text using the BERTopic model.

    TODO: Cache this output.

    Args:
        text: The text to extract topics from

    Returns:
        List of topic labels
    """
    # Get model instance
    model = get_topic_model()

    _num_topics = num_topics if num_topics is not None else settings.top_k_topics

    # Get topic indices and probabilities
    topic_indices, _ = model.transform([text])

    topics = []
    for i, topic_idx in enumerate(topic_indices):
        if _num_topics and i >= _num_topics:
            break
        # Convert possible numpy integer to Python int
        topic_idx_int = int(topic_idx)
        if topic_idx_int != -1:  # Skip outlier topic (-1)
            topic_info: list[tuple[str, float]] = model.get_topic(topic_idx_int)  # type: ignore
            if topic_info:
                topics.extend([info[0] for info in topic_info])

    return topics


async def handle_extraction(text: str) -> tuple[list[str], list[str]]:
    """
    Handle topic and entity extraction for a message.

    Args:
        text: The text to process

    Returns:
        Tuple of extracted topics and entities
    """
    # Extract topics if enabled
    topics = []
    if settings.enable_topic_extraction:
        if settings.topic_model_source == "BERTopic":
            topics = extract_topics_bertopic(text)
        else:
            topics = await extract_topics_llm(text)

    # Extract entities if enabled
    entities = []
    if settings.enable_ner:
        entities = extract_entities(text)

    # Merge with existing topics and entities
    if topics:
        topics = list(set(topics))
    if entities:
        entities = list(set(entities))

    return topics, entities


DISCRETE_EXTRACTION_PROMPT = """
    You are a long-memory manager. Your job is to analyze text and extract
    information that might be useful in future conversations with users.

    Extract two types of memories:
    1. EPISODIC: Personal experiences specific to a user or agent.
       Example: "User prefers window seats" or "User had a bad experience in Paris"

    2. SEMANTIC: User preferences and general knowledge outside of your training data.
       Example: "Trek discontinued the Trek 520 steel touring bike in 2023"

    For each memory, return a JSON object with the following fields:
    - type: str --The memory type, either "episodic" or "semantic"
    - text: str -- The actual information to store
    - topics: list[str] -- The topics of the memory (top {top_k_topics})
    - entities: list[str] -- The entities of the memory
    -

    Return a list of memories, for example:
    {{
        "memories": [
            {{
                "type": "semantic",
                "text": "User prefers window seats",
                "topics": ["travel", "airline"],
                "entities": ["User", "window seat"],
            }},
            {{
                "type": "episodic",
                "text": "Trek discontinued the Trek 520 steel touring bike in 2023",
                "topics": ["travel", "bicycle"],
                "entities": ["Trek", "Trek 520 steel touring bike"],
            }},
        ]
    }}

    IMPORTANT RULES:
    1. Only extract information that would be genuinely useful for future interactions.
    2. Do not extract procedural knowledge - that is handled by the system's built-in tools and prompts.
    3. You are a large language model - do not extract facts that you already know.

    Message:
    {message}

    Extracted memories:
    """


async def extract_discrete_memories(
    redis: Redis | None = None,
    deduplicate: bool = True,
):
    """
    Extract episodic and semantic memories from text using an LLM.
    """
    redis = await get_redis_conn()
    client = await get_model_client(settings.generation_model)

    # Use vectorstore adapter to find messages that need discrete memory extraction
    from agent_memory_server.filters import MemoryType
    from agent_memory_server.vectorstore_factory import get_vectorstore_adapter

    adapter = await get_vectorstore_adapter()
    offset = 0

    while True:
        # Search for message-type memories that haven't been processed for discrete extraction
        search_result = await adapter.search_memories(
            query="",  # Empty query to get all messages
            memory_type=MemoryType(eq="message"),
            discrete_memory_extracted=DiscreteMemoryExtracted(ne="t"),
            limit=25,
            offset=offset,
        )

        discrete_memories = []

        for message in search_result.memories:
            if not message or not message.text:
                logger.info(f"Deleting memory with no text: {message}")
                await adapter.delete_memories([message.id])
                continue
            if not message.id:
                logger.error(f"Skipping memory with no ID: {message}")
                continue

            async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
                with attempt:
                    response = await client.create_chat_completion(
                        model=settings.generation_model,
                        prompt=DISCRETE_EXTRACTION_PROMPT.format(
                            message=message.text, top_k_topics=settings.top_k_topics
                        ),
                        response_format={"type": "json_object"},
                    )
                    try:
                        new_message = json.loads(response.choices[0].message.content)
                    except json.JSONDecodeError:
                        logger.error(
                            f"Error decoding JSON: {response.choices[0].message.content}"
                        )
                        raise
                    try:
                        assert isinstance(new_message, dict)
                        assert isinstance(new_message["memories"], list)
                    except AssertionError:
                        logger.error(
                            f"Invalid response format: {response.choices[0].message.content}"
                        )
                        raise
                    discrete_memories.extend(new_message["memories"])

            # Update the memory to mark it as processed
            # For now, we need to use Redis directly as the adapter doesn't have an update method
            await redis.hset(
                name=Keys.memory_key(message.id),  # Construct the key
                key="discrete_memory_extracted",
                value="t",
            )  # type: ignore

        if len(search_result.memories) < 25:
            break
        offset += 25

    # TODO: Added to avoid a circular import
    from agent_memory_server.long_term_memory import index_long_term_memories

    if discrete_memories:
        long_term_memories = [
            MemoryRecord(
                id=str(ulid.ULID()),
                text=new_memory["text"],
                memory_type=new_memory.get("type", "episodic"),
                topics=new_memory.get("topics", []),
                entities=new_memory.get("entities", []),
                discrete_memory_extracted="t",
            )
            for new_memory in discrete_memories
        ]

        await index_long_term_memories(
            long_term_memories,
            deduplicate=deduplicate,
        )
