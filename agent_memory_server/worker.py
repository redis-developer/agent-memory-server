"""
Run the Docket worker directly from Python.

This module provides a way to run the background task worker in-process
instead of using the CLI command.

Usage:
    python -m agent_memory_server.worker
"""

import asyncio
import signal
import sys
from datetime import timedelta

from docket import Docket, Worker

from agent_memory_server.config import settings
from agent_memory_server.docket_tasks import task_collection
from agent_memory_server.logging import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)


async def run_worker(concurrency: int = 10, redelivery_timeout: int = 30):
    """
    Run the Docket worker in Python.

    Args:
        concurrency: Number of tasks to process concurrently
        redelivery_timeout: Seconds to wait before redelivering a task to another worker
    """
    if not settings.use_docket:
        logger.error("Docket is disabled in settings. Cannot run worker.")
        return None

    logger.info(f"Starting Docket worker for {settings.docket_name}")
    logger.info(
        f"Concurrency: {concurrency}, Redelivery timeout: {redelivery_timeout}s"
    )

    # Create a signal handler to gracefully shut down
    shutdown_event = asyncio.Event()

    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        # Initialize Docket client
        async with Docket(
            name=settings.docket_name,
            url=settings.redis_url,
        ) as docket:
            # Register all tasks
            for task in task_collection:
                docket.register(task)

            logger.info(f"Registered {len(task_collection)} tasks")

            # Create and run the worker
            async with Worker(
                docket,
                concurrency=concurrency,
                redelivery_timeout=timedelta(seconds=redelivery_timeout),
            ) as worker:
                # Run until shutdown is requested
                await worker.run_forever()

    except Exception as e:
        logger.error(f"Error running worker: {e}")
        return 1

    logger.info("Worker shut down gracefully")
    return 0


def main():
    """Command line entry point"""
    # Parse command line arguments
    concurrency = 10
    redelivery_timeout = 30

    args = sys.argv[1:]
    if "--concurrency" in args:
        try:
            idx = args.index("--concurrency")
            concurrency = int(args[idx + 1])
        except (ValueError, IndexError):
            pass

    if "--redelivery-timeout" in args:
        try:
            idx = args.index("--redelivery-timeout")
            redelivery_timeout = int(args[idx + 1])
        except (ValueError, IndexError):
            pass

    return asyncio.run(
        run_worker(concurrency=concurrency, redelivery_timeout=redelivery_timeout)
    )


if __name__ == "__main__":
    sys.exit(main())
