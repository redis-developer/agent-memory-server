import asyncio
import os
from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import BackgroundTasks, FastAPI
from httpx import ASGITransport, AsyncClient
from redis import Redis
from redis.asyncio import ConnectionPool
from redis.asyncio import Redis as AsyncRedis
from testcontainers.compose import DockerCompose

from healthcheck import router as health_router
from memory import router as memory_router
from models import (
    MemoryMessage,
    OpenAIClientWrapper,
)
from retrieval import router as retrieval_router
from utils import REDIS_INDEX_NAME, Keys, ensure_redisearch_index


@pytest.fixture(scope="session")
def event_loop(request):
    loop = asyncio.get_event_loop()
    yield loop


@pytest.fixture
def memory_message():
    """Create a sample memory message"""
    return MemoryMessage(role="user", content="Hello, world!")


@pytest.fixture
def memory_messages():
    """Create a list of sample memory messages"""
    return [
        MemoryMessage(role="user", content="What is the capital of France?"),
        MemoryMessage(role="assistant", content="The capital of France is Paris."),
        MemoryMessage(role="user", content="And what about Germany?"),
        MemoryMessage(role="assistant", content="The capital of Germany is Berlin."),
    ]


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client"""
    client = AsyncMock(spec=OpenAIClientWrapper)

    # We won't set default side effects here, allowing tests to set their own mocks
    # This prevents conflicts with tests that need specific return values

    return client


@pytest.fixture(autouse=True)
async def search_index(async_redis_client):
    """Create a real Redis connection pool for testing"""
    # TODO: Replace with RedisVL index.

    await async_redis_client.flushdb()

    vector_dimensions = 1536
    distance_metric = "COSINE"
    index_name = REDIS_INDEX_NAME

    try:
        try:
            await async_redis_client.execute_command("FT.INFO", index_name)
            await async_redis_client.execute_command("FT.DROPINDEX", index_name)
        except Exception as e:
            if "unknown index name".lower() not in str(e).lower():
                print(f"Error checking index: {e}")

        await ensure_redisearch_index(
            async_redis_client, vector_dimensions, distance_metric, index_name
        )

    except Exception as e:
        print(f"ERROR: Failed to create RediSearch index: {str(e)}")
        print("This might indicate that Redis is not running with RediSearch module")
        print("Make sure you're using redis-stack, not standard redis")
        raise

    yield

    # Clean up after tests
    await async_redis_client.flushdb()
    try:
        await async_redis_client.execute_command("FT.DROPINDEX", index_name)
    except Exception:
        pass


@pytest.fixture
async def test_session_setup(async_redis_client):
    """Set up a test session with Redis data for testing"""
    import json
    import time

    session_id = "test-session"

    # Add session to sorted set of sessions
    await async_redis_client.zadd(Keys.sessions_key(), {session_id: int(time.time())})

    # Add messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    key = Keys.messages_key(session_id)
    for msg in messages:
        await async_redis_client.lpush(key, json.dumps(msg))

    # Add context
    await async_redis_client.set(Keys.context_key(session_id), "Sample context")
    await async_redis_client.set(Keys.token_count_key(session_id), "150")

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


@pytest.fixture
def async_redis_client(redis_url):
    """
    An async Redis client that uses the dynamic `redis_url`.
    """
    return AsyncRedis.from_url(redis_url)


@pytest.fixture
def mock_async_redis_client():
    """Create a mock async Redis client"""
    client = AsyncMock(spec=AsyncRedis)
    return client


@pytest.fixture
def redis_client(redis_url):
    """
    A sync Redis client that uses the dynamic `redis_url`.
    """
    return Redis.from_url(redis_url)


@pytest.fixture
def use_test_redis_connection(redis_url: str):
    replacement_pool = ConnectionPool.from_url(redis_url)
    with patch("utils._redis_pool", new=replacement_pool):
        yield


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
        reason="Skipping test because API keys are not provided. Use --run-api-tests to run these tests."
    )
    for item in items:
        if item.get_closest_marker("requires_api_keys"):
            item.add_marker(skip_api)


MockBackgroundTasks = mock.Mock(name="BackgroundTasks", spec=BackgroundTasks)


@pytest.fixture
def app(use_test_redis_connection):
    """Create a test FastAPI app with routers"""
    app = FastAPI()

    # Include routers
    app.include_router(health_router)
    app.include_router(memory_router)
    app.include_router(retrieval_router)

    return app


@pytest.fixture
def app_with_mock_background_tasks(use_test_redis_connection):
    """Create a test FastAPI app with routers"""
    app = FastAPI()

    # Include routers
    app.include_router(health_router)
    app.include_router(memory_router)
    app.include_router(retrieval_router)

    mock_background_tasks = MockBackgroundTasks()
    app.dependency_overrides[BackgroundTasks] = lambda: mock_background_tasks

    return app


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
async def client_with_mock_background_tasks(app_with_mock_background_tasks):
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mock_background_tasks),
        base_url="http://test",
    ) as client:
        yield client
