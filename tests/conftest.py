import asyncio
import contextlib
import os
from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis import Redis
from redis.asyncio import ConnectionPool, Redis as AsyncRedis
from testcontainers.compose import DockerCompose

from agent_memory_server.api import router as memory_router
from agent_memory_server.config import settings
from agent_memory_server.dependencies import DocketBackgroundTasks, get_background_tasks
from agent_memory_server.healthcheck import router as health_router
from agent_memory_server.llms import OpenAIClientWrapper
from agent_memory_server.messages import (
    MemoryMessage,
    index_long_term_memories,
    set_session_memory,
)
from agent_memory_server.models import LongTermMemory, SessionMemory
from agent_memory_server.utils import (
    ensure_redisearch_index,
)


load_dotenv()


@pytest.fixture(scope="session")
def event_loop(request):
    return asyncio.get_event_loop()


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
    # TODO: Replace with RedisVL index.

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
                print(f"Error checking index: {e}")

        await ensure_redisearch_index(async_redis_client)

    except Exception as e:
        print(f"ERROR: Failed to create RediSearch index: {str(e)}")
        print("This might indicate that Redis is not running with RediSearch module")
        print("Make sure you're using redis-stack, not standard redis")
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

    session_id = "test-session"

    await index_long_term_memories(
        [
            LongTermMemory(
                session_id=session_id,
                text="User: Hello",
                namespace="test-namespace",
            ),
            LongTermMemory(
                session_id=session_id,
                text="Assistant: Hi there",
                namespace="test-namespace",
            ),
        ],
    )

    # Add messages to session memory
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    await set_session_memory(
        async_redis_client,
        session_id,
        SessionMemory(
            messages=[MemoryMessage(**msg) for msg in messages],
            context="Sample context",
            user_id="test-user",
            tokens=150,
            namespace="test-namespace",
        ),
        background_tasks=DocketBackgroundTasks(),
    )

    return session_id


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
    """Replace the Redis connection pool with a test one"""
    replacement_pool = ConnectionPool.from_url(redis_url)
    with patch("agent_memory_server.utils._redis_pool", new=replacement_pool):
        yield replacement_pool


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


@pytest.fixture()
def app(use_test_redis_connection):
    """Create a test FastAPI app with routers"""
    app = FastAPI()

    # Include routers
    app.include_router(health_router)
    app.include_router(memory_router)

    return app


@pytest.fixture()
def app_with_mock_background_tasks(use_test_redis_connection, mock_background_tasks):
    """Create a test FastAPI app with routers"""
    app = FastAPI()

    # Include routers
    app.include_router(health_router)
    app.include_router(memory_router)

    app.dependency_overrides[get_background_tasks] = lambda: mock_background_tasks

    return app


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
