import asyncio
import logging
import sys

from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent

from agent_memory_server.api import (
    create_long_term_memory as core_create_long_term_memory,
    get_session_memory as core_get_session_memory,
    search_long_term_memory as core_search_long_term_memory,
)
from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.models import (
    AckResponse,
    CreateLongTermMemoryPayload,
    LongTermMemoryResults,
    SearchPayload,
)


logger = logging.getLogger(__name__)
mcp_app = FastMCP(
    "Redis Agent Memory Server - ALWAYS check memory for user information",
    port=settings.mcp_port,
    instructions="When responding to user queries, ALWAYS check memory first before answering questions about user preferences, history, or personal information.",
)


@mcp_app.tool()
async def create_long_term_memories(
    payload: CreateLongTermMemoryPayload,
) -> AckResponse:
    """
    Create long-term memories that can be searched later.

    This tool saves memories contained in the payload for future retrieval.

    IMPORTANT NOTES ON SESSION IDs:
    - When including a session_id, use the EXACT session identifier from the current conversation
    - NEVER invent or guess a session ID - if you don't know it, omit the field
    - If you want memories accessible across all sessions, omit the session_id field

    Each memory in the payload should include:
    - text: The content of the memory (required)
    - id_: Optional unique identifier (auto-generated if not provided)
    - session_id: Optional conversation session identifier
    - user_id: Optional user identifier
    - namespace: Optional grouping namespace
    - topics: Optional list of topics for better searchability
    - entities: Optional list of entities mentioned in the text

    Example JSON input:
    ```json
    {
      "payload": {
        "memories": [
          {
            "text": "The user prefers dark mode in all applications",
            "session_id": "session_12345",
            "user_id": "user_789",
            "topics": ["preferences", "ui"]
          }
        ]
      }
    }
    ```

    Args:
        payload: A CreateLongTermMemoryPayload containing a list of memories to create

    Returns:
        An acknowledgement response indicating success
    """
    return await core_create_long_term_memory(
        payload, background_tasks=get_background_tasks()
    )


@mcp_app.tool()
async def search_long_term_memory(
    payload: SearchPayload,
) -> LongTermMemoryResults:
    """
    Search for memories related to a text query.

    Finds memories based on a combination of semantic similarity and input filters.

    This tool performs a semantic search on stored memories using the query text and filters
    in the payload. Results are ranked by relevance.

    IMPORTANT NOTES ON SESSION IDs:
    - When including a session_id filter, use the EXACT session identifier
    - NEVER invent or guess a session ID - if you don't know it, omit this filter
    - If you want to search across all sessions, don't include a session_id filter
    - Session IDs from examples will NOT work with real data

    COMMON USAGE PATTERNS:

    1. Basic search with just query text:
    ```json
    {
      "payload": {
        "text": "user's favorite color"
      }
    }
    ```

    2. Search with simple session filter:
    ```json
    {
      "payload": {
        "text": "user's favorite color",
        "session_id": {
          "eq": "session_12345"
        }
      }
    }
    ```

    3. Search with complex filters:
    ```json
    {
      "payload": {
        "text": "user preferences",
        "topics": {
          "any": ["preferences", "settings"]
        },
        "created_at": {
          "gt": 1640995200
        },
        "limit": 5
      }
    }
    ```

    Args:
        payload: A SearchPayload object containing:
            - text: The semantic search query text (required)
            - Filter objects for various fields (session_id, namespace, topics, etc.)
            - Pagination options (limit, offset)
            - Other search options (distance_threshold)

    Returns:
        LongTermMemoryResults containing matched memories sorted by relevance
    """
    return await core_search_long_term_memory(payload)


# NOTE: Prompts don't support search filters in FastMCP, so we need to use a
# tool instead.
@mcp_app.tool()
async def hydrate_memory_prompt(
    payload: SearchPayload,
) -> list[base.Message]:
    """
    Hydrate a user prompt with relevant session history and long-term memories.

    CRITICAL: Use this tool for EVERY question that might benefit from memory context,
    especially when you don't have sufficient information to answer confidently.

    This tool enriches the user's query by retrieving:
    1. Context from the current conversation session
    2. Relevant long-term memories related to the query

    ALWAYS use this tool when:
    - The user references past conversations
    - The question is about user preferences or personal information
    - You need additional context to provide a complete answer
    - The question seems to assume information you don't have in current context

    The function uses the text field from the payload as the user's query,
    and any filters to retrieve relevant memories.

    IMPORTANT NOTES ON SESSION IDs:
    - When filtering by session_id, you must provide the EXACT session identifier
    - NEVER invent or guess a session ID - if you don't know it, omit this filter
    - Session IDs from examples will NOT work with real data

    Example JSON input:
    ```json
    {
      "payload": {
        "text": "What was my favorite color?",
        "session_id": {
          "eq": "session_12345"
        },
        "namespace": {
          "eq": "user_preferences"
        }
      }
    }
    ```

    Args:
        payload: A SearchPayload containing:
            - text: The user's query/message (required)
            - Filter objects for retrieving relevant session and memories

    Returns:
        A list of messages forming a prompt, including context and the user's query
    """
    messages = []
    if payload.session_id and payload.session_id.eq:
        try:
            session_memory = await core_get_session_memory(payload.session_id.eq)
        except HTTPException:
            session_memory = None
    else:
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
        long_term_memories = await core_search_long_term_memory(payload)
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
            content=TextContent(type="text", text=payload.text),
        )
    )

    return messages


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        asyncio.run(mcp_app.run_sse_async())
    else:
        asyncio.run(mcp_app.run_stdio_async())
