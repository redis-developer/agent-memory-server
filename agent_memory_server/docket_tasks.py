"""
Background task management using Docket.
"""

import logging

from docket import Docket

from agent_memory_server.config import settings
from agent_memory_server.extraction import extract_discrete_memories
from agent_memory_server.long_term_memory import (
    compact_long_term_memories,
    delete_long_term_memories,
    extract_memory_structure,
    forget_long_term_memories,
    index_long_term_memories,
    periodic_forget_long_term_memories,
    promote_working_memory_to_long_term,
)
from agent_memory_server.summarization import summarize_session


logger = logging.getLogger(__name__)


# Register functions in the task collection for the CLI worker
task_collection = [
    extract_memory_structure,
    summarize_session,
    index_long_term_memories,
    compact_long_term_memories,
    extract_discrete_memories,
    promote_working_memory_to_long_term,
    delete_long_term_memories,
    forget_long_term_memories,
    periodic_forget_long_term_memories,
]


async def register_tasks() -> None:
    """Register all task functions with Docket."""
    if not settings.use_docket:
        logger.info("Docket is disabled, skipping task registration")
        return

    # Initialize Docket client
    async with Docket(
        name=settings.docket_name,
        url=settings.redis_url,
    ) as docket:
        # Register all tasks
        for task in task_collection:
            docket.register(task)

        logger.info(f"Registered {len(task_collection)} background tasks with Docket")
