# Agent Memory Client

A Python client library for the [Agent Memory Server](https://github.com/your-org/agent-memory-server) REST API, providing comprehensive memory management capabilities for AI agents and applications.

## Features

- **Complete API Coverage**: Full support for all Agent Memory Server endpoints
- **Memory Lifecycle Management**: Explicit control over working → long-term memory promotion
- **Batch Operations**: Efficient bulk operations with built-in rate limiting
- **Auto-Pagination**: Seamless iteration over large result sets
- **Client-Side Validation**: Pre-flight validation to catch errors early
- **Enhanced Convenience Methods**: Simplified APIs for common operations
- **Type Safety**: Full type hints for better development experience
- **Async-First**: Built for modern async Python applications

## Installation

```bash
pip install agent-memory-client
```

## Quick Start

```python
import asyncio
from agent_memory_client import create_memory_client, ClientMemoryRecord, MemoryTypeEnum

async def main():
    # Create a client instance
    client = await create_memory_client(
        base_url="http://localhost:8000",
        default_namespace="my-app"
    )

    try:
        # Create some memories
        memories = [
            ClientMemoryRecord(
                text="User prefers dark mode",
                memory_type=MemoryTypeEnum.SEMANTIC,
                topics=["preferences", "ui"]
            ),
            ClientMemoryRecord(
                text="User completed onboarding on 2024-01-15",
                memory_type=MemoryTypeEnum.EPISODIC,
                topics=["onboarding", "milestones"]
            )
        ]

        # Store in long-term memory
        await client.create_long_term_memory(memories)

        # Search memories
        results = await client.search_long_term_memory(
            text="user interface preferences",
            limit=10
        )

        print(f"Found {len(results.memories)} relevant memories")
        for memory in results.memories:
            print(f"- {memory.text} (distance: {memory.dist})")

    finally:
        await client.close()

# Run the example
asyncio.run(main())
```

## Core API

### Client Setup

```python
from agent_memory_client import MemoryAPIClient, MemoryClientConfig

# Manual configuration
config = MemoryClientConfig(
    base_url="http://localhost:8000",
    timeout=30.0,
    default_namespace="my-app"
)
client = MemoryAPIClient(config)

# Or use the helper function
client = await create_memory_client(
    base_url="http://localhost:8000",
    default_namespace="my-app"
)
```

### Working Memory Operations

```python
from agent_memory_client import WorkingMemory, MemoryMessage

# Create working memory with messages
working_memory = WorkingMemory(
    session_id="user-session-123",
    messages=[
        MemoryMessage(role="user", content="Hello!"),
        MemoryMessage(role="assistant", content="Hi there! How can I help?")
    ],
    namespace="chat-app"
)

# Store working memory
response = await client.put_session_memory("user-session-123", working_memory)

# Retrieve working memory
memory = await client.get_session_memory("user-session-123")

# Convenience method for data storage
await client.set_working_memory_data(
    session_id="user-session-123",
    data={"user_preferences": {"theme": "dark", "language": "en"}}
)
```

### Long-Term Memory Operations

```python
from agent_memory_client import ClientMemoryRecord, MemoryTypeEnum

# Create memories
memories = [
    ClientMemoryRecord(
        text="User enjoys science fiction books",
        memory_type=MemoryTypeEnum.SEMANTIC,
        topics=["books", "preferences"],
        user_id="user-123"
    )
]

# Store memories
await client.create_long_term_memory(memories)

# Search with filters
from agent_memory_client.filters import Topics, UserId

results = await client.search_long_term_memory(
    text="science fiction",
    topics=Topics(any=["books", "entertainment"]),
    user_id=UserId(eq="user-123"),
    limit=20
)
```

## Enhanced Features

### Memory Lifecycle Management

```python
# Explicitly promote working memories to long-term storage
await client.promote_working_memories_to_long_term(
    session_id="user-session-123",
    memory_ids=["memory-1", "memory-2"]  # Optional: specific memories
)
```

### Batch Operations

```python
# Bulk create with rate limiting
memory_batches = [batch1, batch2, batch3]
results = await client.bulk_create_long_term_memories(
    memory_batches=memory_batches,
    batch_size=50,
    delay_between_batches=0.1
)
```

### Auto-Pagination

```python
# Iterate through all results automatically
async for memory in client.search_all_long_term_memories(
    text="user preferences",
    batch_size=100
):
    print(f"Memory: {memory.text}")
```

### Client-Side Validation

```python
from agent_memory_client.exceptions import MemoryValidationError

try:
    # Validate before sending
    client.validate_memory_record(memory)
    client.validate_search_filters(limit=10, offset=0)
except MemoryValidationError as e:
    print(f"Validation error: {e}")
```

### Enhanced Convenience Methods

```python
# Update working memory data with merge strategies
await client.update_working_memory_data(
    session_id="user-session-123",
    data_updates={"new_setting": "value"},
    merge_strategy="deep_merge"  # "replace", "merge", or "deep_merge"
)

# Append messages efficiently
new_messages = [
    MemoryMessage(role="user", content="What's the weather?"),
    MemoryMessage(role="assistant", content="It's sunny today!")
]

await client.append_messages_to_working_memory(
    session_id="user-session-123",
    messages=new_messages
)
```

## Advanced Filtering

```python
from agent_memory_client.filters import (
    SessionId, Namespace, Topics, Entities,
    CreatedAt, LastAccessed, UserId, MemoryType
)
from datetime import datetime, timezone

# Complex search with multiple filters
results = await client.search_long_term_memory(
    text="machine learning",
    session_id=SessionId(in_=["session-1", "session-2"]),
    namespace=Namespace(eq="ai-research"),
    topics=Topics(any=["ml", "ai"], none=["deprecated"]),
    entities=Entities(all=["tensorflow", "python"]),
    created_at=CreatedAt(gte=datetime(2024, 1, 1, tzinfo=timezone.utc)),
    user_id=UserId(eq="researcher-123"),
    memory_type=MemoryType(eq="semantic"),
    distance_threshold=0.8,
    limit=50
)
```

## Error Handling

```python
from agent_memory_client.exceptions import (
    MemoryClientError,
    MemoryValidationError,
    MemoryNotFoundError,
    MemoryServerError
)

try:
    memory = await client.get_session_memory("nonexistent-session")
except MemoryNotFoundError:
    print("Session not found")
except MemoryServerError as e:
    print(f"Server error {e.status_code}: {e}")
except MemoryClientError as e:
    print(f"Client error: {e}")
```

## Context Manager Usage

```python
async with create_memory_client("http://localhost:8000") as client:
    # Client will be automatically closed when exiting the context
    results = await client.search_long_term_memory("search query")
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=agent_memory_client
```

### Code Quality

```bash
# Format code
black agent_memory_client/

# Lint code
ruff check agent_memory_client/

# Type checking
mypy agent_memory_client/
```

## Requirements

- Python 3.10+
- httpx >= 0.25.0
- pydantic >= 2.0.0
- ulid-py >= 1.1.0

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see the [main repository](https://github.com/your-org/agent-memory-server) for contribution guidelines.

## Links

- [Agent Memory Server](https://github.com/your-org/agent-memory-server) - The server this client connects to
- [Documentation](https://agent-memory-client.readthedocs.io) - Full API documentation
- [Issues](https://github.com/your-org/agent-memory-client/issues) - Bug reports and feature requests
