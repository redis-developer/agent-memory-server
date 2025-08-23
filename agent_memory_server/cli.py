"""
Command-line interface for agent-memory-server.
"""

import importlib
import sys
from datetime import UTC, datetime, timedelta

import click
import uvicorn

from agent_memory_server import __version__
from agent_memory_server.config import settings
from agent_memory_server.logging import (
    configure_logging,
    configure_mcp_logging,
    get_logger,
)
from agent_memory_server.migrations import (
    migrate_add_discrete_memory_extracted_2,
    migrate_add_memory_hashes_1,
    migrate_add_memory_type_3,
)
from agent_memory_server.utils.redis import ensure_search_index_exists, get_redis_conn


logger = get_logger(__name__)

VERSION = __version__


@click.group()
def cli():
    """Command-line interface for agent-memory-server."""
    pass


@cli.command()
def version():
    """Show the version of agent-memory-server."""
    click.echo(f"agent-memory-server version {VERSION}")


@cli.command()
def rebuild_index():
    """Rebuild the search index."""
    import asyncio

    configure_logging()

    async def setup_and_run():
        redis = await get_redis_conn()
        await ensure_search_index_exists(redis, overwrite=True)

    asyncio.run(setup_and_run())


@cli.command()
def migrate_memories():
    """Migrate memories from the old format to the new format."""
    import asyncio

    configure_logging()
    click.echo("Starting memory migrations...")

    async def run_migrations():
        redis = await get_redis_conn()
        migrations = [
            migrate_add_memory_hashes_1,
            migrate_add_discrete_memory_extracted_2,
            migrate_add_memory_type_3,
        ]
        for migration in migrations:
            await migration(redis=redis)

    asyncio.run(run_migrations())

    click.echo("Memory migrations completed successfully.")


@cli.command()
@click.option("--port", default=settings.port, help="Port to run the server on")
@click.option("--host", default="0.0.0.0", help="Host to run the server on")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def api(port: int, host: str, reload: bool):
    """Run the REST API server."""
    from agent_memory_server.main import on_start_logger

    configure_logging()
    on_start_logger(port)
    uvicorn.run(
        "agent_memory_server.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.option("--port", default=settings.mcp_port, help="Port to run the MCP server on")
@click.option(
    "--mode",
    default="stdio",
    help="Run the MCP server in SSE or stdio mode",
    type=click.Choice(["stdio", "sse"]),
)
def mcp(port: int, mode: str):
    """Run the MCP server."""
    import asyncio

    # Configure MCP-specific logging BEFORE any imports to avoid stdout contamination
    if mode == "stdio":
        configure_mcp_logging()
    else:
        configure_logging()

    # Update the port in settings FIRST
    settings.mcp_port = port

    # Import mcp_app AFTER settings have been updated
    from agent_memory_server.mcp import mcp_app

    async def setup_and_run():
        # Redis setup is handled by the MCP app before it starts

        # Run the MCP server
        if mode == "sse":
            logger.info(f"Starting MCP server on port {port}\n")
            await mcp_app.run_sse_async()
        elif mode == "stdio":
            # Don't run a task worker in stdio mode.
            # TODO: Make configurable with a CLI flag?
            settings.use_docket = False
            await mcp_app.run_stdio_async()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # TODO: Do we really need to update the port again?
    # Update the port in settings
    settings.mcp_port = port

    asyncio.run(setup_and_run())


@cli.command()
@click.argument("task_path")
@click.option(
    "--args",
    "-a",
    multiple=True,
    help="Arguments to pass to the task in the format key=value",
)
def schedule_task(task_path: str, args: list[str]):
    """
    Schedule a background task by path.

    TASK_PATH is the import path to the task function, e.g.,
    "agent_memory_server.long_term_memory.compact_long_term_memories"
    """
    import asyncio

    from docket import Docket

    configure_logging()

    # Parse the arguments
    task_args = {}
    for arg in args:
        try:
            key, value = arg.split("=", 1)
            # Try to convert to appropriate type
            if value.lower() == "true":
                task_args[key] = True
            elif value.lower() == "false":
                task_args[key] = False
            elif value.isdigit():
                task_args[key] = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") <= 1:
                task_args[key] = float(value)
            else:
                task_args[key] = value
        except ValueError:
            click.echo(f"Invalid argument format: {arg}. Use key=value format.")
            sys.exit(1)

    async def setup_and_run_task():
        redis = await get_redis_conn()
        await ensure_search_index_exists(redis)

        # Import the task function
        module_path, function_name = task_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            task_func = getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            click.echo(f"Error importing task: {e}")
            sys.exit(1)

        # Initialize Docket client
        async with Docket(
            name=settings.docket_name,
            url=settings.redis_url,
        ) as docket:
            click.echo(f"Scheduling task {task_path} with arguments: {task_args}")
            await docket.add(task_func)(**task_args)
            click.echo("Task scheduled successfully")

    asyncio.run(setup_and_run_task())


@cli.command()
@click.option(
    "--concurrency", default=10, help="Number of tasks to process concurrently"
)
@click.option(
    "--redelivery-timeout",
    default=30,
    help="Seconds to wait before redelivering a task to another worker",
)
def task_worker(concurrency: int, redelivery_timeout: int):
    """
    Start a Docket worker using the Docket name from settings.

    This command starts a worker that processes background tasks registered
    with Docket. The worker uses the Docket name from settings.
    """
    import asyncio

    from docket import Worker

    configure_logging()

    if not settings.use_docket:
        click.echo("Docket is disabled in settings. Cannot run worker.")
        sys.exit(1)

    async def _ensure_stream_and_group():
        """Ensure the Docket stream and consumer group exist to avoid NOGROUP errors."""
        from redis.exceptions import ResponseError

        redis = await get_redis_conn()
        stream_key = f"{settings.docket_name}:stream"
        group_name = "docket-workers"

        try:
            # Create consumer group, auto-create stream if missing
            await redis.xgroup_create(
                name=stream_key, groupname=group_name, id="$", mkstream=True
            )
        except ResponseError as e:
            # BUSYGROUP means it already exists; safe to ignore
            if "BUSYGROUP" not in str(e).upper():
                raise

    async def _run_worker():
        # Ensure Redis stream/consumer group and search index exist before starting worker
        await _ensure_stream_and_group()
        try:
            redis = await get_redis_conn()
            # Don't overwrite if an index already exists; just ensure it's present
            await ensure_search_index_exists(redis, overwrite=False)
        except Exception as e:
            logger.warning(f"Failed to ensure search index exists: {e}")
        await Worker.run(
            docket_name=settings.docket_name,
            url=settings.redis_url,
            concurrency=concurrency,
            redelivery_timeout=timedelta(seconds=redelivery_timeout),
            tasks=["agent_memory_server.docket_tasks:task_collection"],
        )

    asyncio.run(_run_worker())


@cli.group()
def token():
    """Manage authentication tokens."""
    pass


@token.command()
@click.option("--description", "-d", required=True, help="Token description")
@click.option("--expires-days", "-e", type=int, help="Token expiration in days")
def add(description: str, expires_days: int | None):
    """Add a new authentication token."""
    import asyncio

    from agent_memory_server.auth import TokenInfo, generate_token, hash_token
    from agent_memory_server.utils.keys import Keys

    async def create_token():
        redis = await get_redis_conn()

        # Generate token
        token = generate_token()
        token_hash = hash_token(token)

        # Calculate expiration
        now = datetime.now(UTC)
        expires_at = now + timedelta(days=expires_days) if expires_days else None

        # Create token info
        token_info = TokenInfo(
            description=description,
            created_at=now,
            expires_at=expires_at,
            token_hash=token_hash,
        )

        # Store in Redis
        key = Keys.auth_token_key(token_hash)
        await redis.set(key, token_info.model_dump_json())

        # Set TTL if expiration is set
        if expires_at:
            ttl_seconds = int((expires_at - now).total_seconds())
            await redis.expire(key, ttl_seconds)

        # Add to tokens list (for listing purposes)
        list_key = Keys.auth_tokens_list_key()
        await redis.sadd(list_key, token_hash)

        click.echo("Token created successfully!")
        click.echo(f"Token: {token}")
        click.echo(f"Description: {description}")
        if expires_at:
            click.echo(f"Expires: {expires_at.isoformat()}")
        else:
            click.echo("Expires: Never")
        click.echo("\nWARNING: Save this token securely. It will not be shown again.")

    asyncio.run(create_token())


@token.command()
def list():
    """List all authentication tokens."""
    import asyncio

    from agent_memory_server.auth import TokenInfo
    from agent_memory_server.utils.keys import Keys

    async def list_tokens():
        redis = await get_redis_conn()

        # Get all token hashes
        list_key = Keys.auth_tokens_list_key()
        token_hashes = await redis.smembers(list_key)

        if not token_hashes:
            click.echo("No tokens found.")
            return

        click.echo("Authentication Tokens:")
        click.echo("=" * 50)

        for token_hash in token_hashes:
            key = Keys.auth_token_key(token_hash)
            token_data = await redis.get(key)

            if not token_data:
                # Token expired or deleted, remove from list
                await redis.srem(list_key, token_hash)
                continue

            try:
                token_info = TokenInfo.model_validate_json(token_data)

                # Mask the token hash for display
                masked_hash = token_hash[:8] + "..." + token_hash[-8:]

                click.echo(f"Token: {masked_hash}")
                click.echo(f"Description: {token_info.description}")
                click.echo(f"Created: {token_info.created_at.isoformat()}")
                if token_info.expires_at:
                    click.echo(f"Expires: {token_info.expires_at.isoformat()}")
                else:
                    click.echo("Expires: Never")
                click.echo("-" * 30)

            except Exception as e:
                click.echo(f"Error processing token {token_hash}: {e}")

    asyncio.run(list_tokens())


@token.command()
@click.argument("token_hash")
def show(token_hash: str):
    """Show details for a specific token."""
    import asyncio

    from agent_memory_server.auth import TokenInfo
    from agent_memory_server.utils.keys import Keys

    async def show_token():
        nonlocal token_hash
        redis = await get_redis_conn()

        # Try to find the token by partial hash
        if len(token_hash) < 16:
            # If partial hash provided, find the full hash
            list_key = Keys.auth_tokens_list_key()
            token_hashes = await redis.smembers(list_key)

            matching_hashes = [h for h in token_hashes if h.startswith(token_hash)]

            if not matching_hashes:
                click.echo(f"No token found matching '{token_hash}'")
                return
            if len(matching_hashes) > 1:
                click.echo(f"Multiple tokens match '{token_hash}':")
                for h in matching_hashes:
                    click.echo(f"  {h[:8]}...{h[-8:]}")
                return
            token_hash = matching_hashes[0]

        key = Keys.auth_token_key(token_hash)
        token_data = await redis.get(key)

        if not token_data:
            click.echo(f"Token not found: {token_hash}")
            return

        try:
            token_info = TokenInfo.model_validate_json(token_data)

            click.echo("Token Details:")
            click.echo("=" * 30)
            click.echo(f"Hash: {token_hash}")
            click.echo(f"Description: {token_info.description}")
            click.echo(f"Created: {token_info.created_at.isoformat()}")
            if token_info.expires_at:
                click.echo(f"Expires: {token_info.expires_at.isoformat()}")
                # Check if expired
                if datetime.now(UTC) > token_info.expires_at:
                    click.echo("Status: EXPIRED")
                else:
                    click.echo("Status: Active")
            else:
                click.echo("Expires: Never")
                click.echo("Status: Active")

        except Exception as e:
            click.echo(f"Error processing token: {e}")

    asyncio.run(show_token())


@token.command()
@click.argument("token_hash")
@click.option("--force", "-f", is_flag=True, help="Force removal without confirmation")
def remove(token_hash: str, force: bool):
    """Remove an authentication token."""
    import asyncio

    from agent_memory_server.auth import TokenInfo
    from agent_memory_server.utils.keys import Keys

    async def remove_token():
        nonlocal token_hash
        redis = await get_redis_conn()

        # Try to find the token by partial hash
        if len(token_hash) < 16:
            # If partial hash provided, find the full hash
            list_key = Keys.auth_tokens_list_key()
            token_hashes = await redis.smembers(list_key)

            matching_hashes = [h for h in token_hashes if h.startswith(token_hash)]

            if not matching_hashes:
                click.echo(f"No token found matching '{token_hash}'")
                return
            if len(matching_hashes) > 1:
                click.echo(f"Multiple tokens match '{token_hash}':")
                for h in matching_hashes:
                    click.echo(f"  {h[:8]}...{h[-8:]}")
                return
            token_hash = matching_hashes[0]

        key = Keys.auth_token_key(token_hash)
        token_data = await redis.get(key)

        if not token_data:
            click.echo(f"Token not found: {token_hash}")
            return

        try:
            token_info = TokenInfo.model_validate_json(token_data)

            # Show token info before removal
            click.echo("Token to remove:")
            click.echo(f"  Description: {token_info.description}")
            click.echo(f"  Created: {token_info.created_at.isoformat()}")

            # Confirm removal
            if not force and not click.confirm(
                "Are you sure you want to remove this token?"
            ):
                click.echo("Token removal cancelled.")
                return

            # Remove from Redis
            await redis.delete(key)

            # Remove from tokens list
            list_key = Keys.auth_tokens_list_key()
            await redis.srem(list_key, token_hash)

            click.echo("Token removed successfully.")

        except Exception as e:
            click.echo(f"Error processing token: {e}")

    asyncio.run(remove_token())


if __name__ == "__main__":
    cli()
