import contextlib
import os
import time
from unittest import mock
from unittest.mock import AsyncMock, patch

import docket
import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from testcontainers.compose import DockerCompose

from agent_memory_server.api import router as memory_router
from agent_memory_server.config import settings
from agent_memory_server.dependencies import DocketBackgroundTasks, get_background_tasks
from agent_memory_server.healthcheck import router as health_router
from agent_memory_server.llms import OpenAIClientWrapper
from agent_memory_server.models import (
    MemoryMessage,
)

# Import the module to access its global for resetting
from agent_memory_server.utils import redis as redis_utils_module
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.redis import ensure_search_index_exists


load_dotenv()


@pytest.fixture()
def memory_message():
    """Create a sample memory message"""
    return MemoryMessage(role="user", content="Hello, world!")


@pytest.fixture()
def memory_messages():
    """Create a list of sample memory messages"""
    return [
        MemoryMessage(role="user", content="What is the capital of France?"),
        MemoryMessage(role="assistant", content="The capital of France is Paris."),
        MemoryMessage(role="user", content="And what about Germany?"),
        MemoryMessage(role="assistant", content="The capital of Germany is Berlin."),
    ]


@pytest.fixture()
def mock_openai_client():
    """Create a mock OpenAI client"""
    return AsyncMock(spec=OpenAIClientWrapper)

    # We won't set default side effects here, allowing tests to set their own mocks
    # This prevents conflicts with tests that need specific return values


@pytest.fixture(autouse=True)
async def search_index(async_redis_client):
    """Create a Redis connection pool for testing"""
    # Reset the cached index in redis_utils_module
    redis_utils_module._index = None

    await async_redis_client.flushdb()

    try:
        try:
            await async_redis_client.execute_command(
                "FT.INFO", settings.redisvl_index_name
            )
            await async_redis_client.execute_command(
                "FT.DROPINDEX", settings.redisvl_index_name
            )
        except Exception as e:
            if "unknown index name".lower() not in str(e).lower():
                pass

        await ensure_search_index_exists(async_redis_client)

    except Exception:
        raise

    yield

    # Clean up after tests
    await async_redis_client.flushdb()
    with contextlib.suppress(Exception):
        await async_redis_client.execute_command(
            "FT.DROPINDEX", settings.redisvl_index_name
        )


@pytest.fixture()
async def session(use_test_redis_connection, async_redis_client):
    """Set up a test session with Redis data for testing"""
    import logging

    logging.getLogger(__name__)

    try:
        session_id = "test-session"
        namespace = "test-namespace"

        # Create working memory data
        from agent_memory_server.models import MemoryMessage, WorkingMemory

        messages = [
            MemoryMessage(role="user", content="Hello"),
            MemoryMessage(role="assistant", content="Hi there"),
        ]

        working_memory = WorkingMemory(
            messages=messages,
            memories=[],  # No structured memories for this test
            context="Sample context",
            user_id="test-user",
            tokens=150,
            session_id=session_id,
            namespace=namespace,
        )

        # Store in unified working memory format
        from agent_memory_server.working_memory import set_working_memory

        await set_working_memory(
            working_memory=working_memory,
            redis_client=use_test_redis_connection,
        )

        # Also add session to sessions list for compatibility
        sessions_key = Keys.sessions_key(namespace=namespace)
        current_time = int(time.time())
        await use_test_redis_connection.zadd(sessions_key, {session_id: current_time})

        # Index the messages as long-term memories directly without background tasks
        import ulid
        from redisvl.utils.vectorize import OpenAITextVectorizer

        from agent_memory_server.models import MemoryRecord

        # Create MemoryRecord objects for each message
        long_term_memories = []
        for msg in messages:
            memory = MemoryRecord(
                id=str(ulid.new()),
                text=f"{msg.role}: {msg.content}",
                session_id=session_id,
                namespace=namespace,
                user_id="test-user",
            )
            long_term_memories.append(memory)

        # Index the memories directly
        vectorizer = OpenAITextVectorizer()
        embeddings = await vectorizer.aembed_many(
            [memory.text for memory in long_term_memories],
            batch_size=20,
            as_buffer=True,
        )

        async with use_test_redis_connection.pipeline(transaction=False) as pipe:
            for idx, vector in enumerate(embeddings):
                memory = long_term_memories[idx]
                id_ = memory.id if memory.id else str(ulid.new())
                key = Keys.memory_key(id_, memory.namespace)

                # Generate memory hash for the memory
                from agent_memory_server.long_term_memory import (
                    generate_memory_hash,
                )

                memory_hash = generate_memory_hash(
                    {
                        "text": memory.text,
                        "user_id": memory.user_id or "",
                        "session_id": memory.session_id or "",
                    }
                )

                await pipe.hset(  # type: ignore
                    key,
                    mapping={
                        "text": memory.text,
                        "id_": id_,
                        "session_id": memory.session_id or "",
                        "user_id": memory.user_id or "",
                        "last_accessed": int(memory.last_accessed.timestamp())
                        if memory.last_accessed
                        else int(time.time()),
                        "created_at": int(memory.created_at.timestamp())
                        if memory.created_at
                        else int(time.time()),
                        "namespace": memory.namespace or "",
                        "memory_hash": memory_hash,
                        "vector": vector,
                        "topics": "",
                        "entities": "",
                    },
                )

            await pipe.execute()

        return session_id
    except Exception:
        raise


@pytest.fixture(scope="session", autouse=True)
def redis_container(request):
    """
    If using xdist, create a unique Compose project for each xdist worker by
    setting COMPOSE_PROJECT_NAME. That prevents collisions on container/volume
    names.
    """
    # In xdist, the config has "workerid" in workerinput
    workerinput = getattr(request.config, "workerinput", {})
    worker_id = workerinput.get("workerid", "master")

    # Set the Compose project name so containers do not clash across workers
    os.environ["COMPOSE_PROJECT_NAME"] = f"redis_test_{worker_id}"
    os.environ.setdefault("REDIS_IMAGE", "redis/redis-stack-server:latest")

    compose = DockerCompose(
        context="tests",
        compose_file_name="docker-compose.yml",
        pull=True,
    )
    compose.start()

    yield compose

    compose.stop()


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """
    Use the `DockerCompose` fixture to get host/port of the 'redis' service
    on container port 6379 (mapped to an ephemeral port on the host).
    """
    host, port = redis_container.get_service_host_and_port("redis", 6379)
    return f"redis://{host}:{port}"


@pytest.fixture()
def async_redis_client(redis_url):
    """
    An async Redis client that uses the dynamic `redis_url`.
    """
    return AsyncRedis.from_url(redis_url)


@pytest.fixture()
def mock_async_redis_client():
    """Create a mock async Redis client"""
    return AsyncMock(spec=AsyncRedis)


@pytest.fixture()
def redis_client(redis_url):
    """
    A sync Redis client that uses the dynamic `redis_url`.
    """
    return Redis.from_url(redis_url)


@pytest.fixture()
def use_test_redis_connection(redis_url: str):
    """Replace the Redis connection with a test one"""
    replacement_redis = AsyncRedis.from_url(redis_url)

    # Create a mock get_redis_conn function that always returns the replacement_redis
    async def mock_get_redis_conn(*args, **kwargs):
        # Ignore any URL parameter and always return the replacement_redis
        return replacement_redis

    # Create a patched Docket class that uses the test Redis URL
    original_docket_init = docket.Docket.__init__

    def patched_docket_init(self, name, url=None, *args, **kwargs):
        # Use the test Redis URL instead of the default one
        return original_docket_init(self, name, *args, url=redis_url, **kwargs)

    with (
        patch("agent_memory_server.utils.redis.get_redis_conn", mock_get_redis_conn),
        patch("agent_memory_server.utils.redis.get_redis_conn", mock_get_redis_conn),
        patch("docket.docket.Docket.__init__", patched_docket_init),
    ):
        yield replacement_redis


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API keys",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-api-tests"):
        return

    # Otherwise skip all tests requiring an API key
    skip_api = pytest.mark.skip(
        reason="""
        Skipping test because API keys are not provided.
        "Use --run-api-tests to run these tests.
        """
    )
    for item in items:
        if item.get_closest_marker("requires_api_keys"):
            item.add_marker(skip_api)


@pytest.fixture()
def mock_background_tasks():
    """Create a mock DocketBackgroundTasks instance"""
    return mock.Mock(name="DocketBackgroundTasks", spec=DocketBackgroundTasks)


@pytest.fixture(autouse=True)
def setup_redis_pool(use_test_redis_connection):
    """Set up the global Redis pool for all tests"""
    # Set the global _redis_pool variable to ensure that direct calls to get_redis_conn work
    import agent_memory_server.utils.redis

    agent_memory_server.utils.redis._redis_pool = use_test_redis_connection

    yield

    # Reset the global _redis_pool variable after the test
    agent_memory_server.utils.redis._redis_pool = None


@pytest.fixture()
def app(use_test_redis_connection):
    """Create a test FastAPI app with routers"""
    app = FastAPI()

    # Include routers
    app.include_router(health_router)
    app.include_router(memory_router)

    # Override the get_redis_conn function to return the test Redis connection
    async def mock_get_redis_conn(*args, **kwargs):
        return use_test_redis_connection

    # Override the dependency
    from agent_memory_server.utils.redis import get_redis_conn

    app.dependency_overrides[get_redis_conn] = mock_get_redis_conn

    return app


@pytest.fixture()
def app_with_mock_background_tasks(use_test_redis_connection, mock_background_tasks):
    """Create a test FastAPI app with routers"""
    app = FastAPI()

    # Include routers
    app.include_router(health_router)
    app.include_router(memory_router)

    # Override the get_redis_conn function to return the test Redis connection
    async def mock_get_redis_conn(*args, **kwargs):
        return use_test_redis_connection

    # Override the dependencies
    from agent_memory_server.utils.redis import get_redis_conn

    app.dependency_overrides[get_redis_conn] = mock_get_redis_conn
    app.dependency_overrides[get_background_tasks] = lambda: mock_background_tasks

    return app


@pytest.fixture(autouse=True)
def disable_auth_for_tests():
    """Disable authentication for all tests"""
    original_value = settings.disable_auth
    settings.disable_auth = True
    yield
    settings.disable_auth = original_value


@pytest.fixture()
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture()
async def client_with_mock_background_tasks(app_with_mock_background_tasks):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mock_background_tasks),
        base_url="http://test",
    ) as client:
        yield client
