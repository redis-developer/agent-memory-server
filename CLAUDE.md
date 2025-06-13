# CLAUDE.md - Redis Agent Memory Server Project Context

## Frequently Used Commands
Get started in a new environment by installing `uv`:
```bash
pip install uv
```

```bash
# Development workflow
uv venv                      # Create a virtualenv (once)
source .venv/bin/activate    # Activate the virtualenv (start of terminal session)
uv install --all-extras      # Install dependencies
uv sync --all-extras         # Sync latest dependencies
uv run ruff check            # Run linting
uv run ruff format           # Format code
uv run pytest                # Run tests
uv run pytest tests/         # Run specific test directory

# Server commands
uv run agent-memory api      # Start REST API server (default port 8000)
uv run agent-memory mcp      # Start MCP server (stdio mode)
uv run agent-memory mcp --mode sse --port 9000  # Start MCP server (SSE mode)

# Database/Redis operations
uv run agent-memory rebuild-index     # Rebuild Redis search index
uv run agent-memory migrate-memories  # Run memory migrations

# Background task management
uv run agent-memory task-worker       # Start background task worker
uv run agent-memory schedule-task "agent_memory_server.long_term_memory.compact_long_term_memories"

# Docker development
docker-compose up            # Start full stack (API, MCP, Redis)
docker-compose up redis      # Start only Redis Stack
docker-compose down          # Stop all services
```

IMPORTANT: This project uses `pre-commit`. You should run `pre-commit`
before committing:
```bash
uv run pre-commit run --all-files
```

## Important Architectural Patterns

### Dual Interface Design (REST + MCP)
- **REST API**: Traditional HTTP endpoints for web applications (`api.py`)
- **MCP Server**: Model Context Protocol for AI agent integration (`mcp.py`)
- Both interfaces share the same core memory management logic

### Memory Architecture
```python
# Two-tier memory system
Working Memory (Session-scoped)  →  Long-term Memory (Persistent)
    ↓                                      ↓
- Messages                          - Semantic search
- Context                          - Topic modeling
- Structured memories              - Entity recognition
- Metadata                         - Deduplication
```

### RedisVL Integration
**CRITICAL**: Always use RedisVL query types instead of direct redis-py client access for searches:
```python
# Correct - Use RedisVL queries
from redisvl.query import VectorQuery, FilterQuery
query = VectorQuery(vector=embedding, vector_field_name="embedding", return_fields=["text"])

# Avoid - Direct redis client searches
# redis.ft().search(...)  # Don't do this
```

### Async-First Design
- All core operations are async
- Background task processing with Docket
- Async Redis connections throughout

## Critical Rules

### Authentication
- **PRODUCTION**: Never set `DISABLE_AUTH=true` in production
- **DEVELOPMENT**: Use `DISABLE_AUTH=true` for local testing only
- JWT/OAuth2 authentication required for all endpoints except `/health`, `/docs`, `/openapi.json`

### Memory Management
- Working memory automatically promotes structured memories to long-term storage
- Conversations are summarized when exceeding window size
- Use model-aware token limits for context window management

### RedisVL Usage (Required)
Always use RedisVL query types for any search operations. This is a project requirement.

## Testing Notes

The project uses `pytest` with `testcontainers` for Redis integration testing:

- `uv run pytest` - Run all tests
- `uv run pytest tests/unit/` - Unit tests only
- `uv run pytest tests/integration/` - Integration tests (require Redis)
- `uv run pytest -v` - Verbose output
- `uv run pytest --cov` - With coverage

## Project Structure

```
agent_memory_server/
├── main.py              # FastAPI application entry point
├── api.py               # REST API endpoints
├── mcp.py               # MCP server implementation
├── config.py            # Configuration management
├── auth.py              # OAuth2/JWT authentication
├── models.py            # Pydantic data models
├── working_memory.py    # Session-scoped memory management
├── long_term_memory.py  # Persistent memory with semantic search
├── messages.py          # Message handling and formatting
├── summarization.py     # Conversation summarization
├── extraction.py        # Topic and entity extraction
├── filters.py           # Search filtering logic
├── llms.py              # LLM provider integrations
├── migrations.py        # Database schema migrations
├── docket_tasks.py      # Background task definitions
├── cli.py               # Command-line interface
├── dependencies.py      # FastAPI dependency injection
├── healthcheck.py       # Health check endpoint
├── logging.py           # Structured logging setup
├── client/              # Client libraries
└── utils/               # Utility modules
    ├── redis.py         # Redis connection and setup
    ├── keys.py          # Redis key management
    └── api_keys.py      # API key utilities
```

## Core Components

### 1. Memory Management
- **Working Memory**: Session-scoped storage with automatic summarization
- **Long-term Memory**: Persistent storage with semantic search capabilities
- **Memory Promotion**: Automatic migration from working to long-term memory
- **Deduplication**: Prevents duplicate memories using content hashing

### 2. Search and Retrieval
- **Semantic Search**: Vector-based similarity search using embeddings
- **Filtering System**: Advanced filtering by session, namespace, topics, entities, timestamps
- **Hybrid Search**: Combines semantic similarity with metadata filtering
- **RedisVL Integration**: All search operations use RedisVL query builders

### 3. AI Integration
- **Topic Modeling**: Automatic topic extraction using BERTopic or LLM
- **Entity Recognition**: BERT-based named entity recognition
- **Summarization**: Conversation summarization when context window exceeded
- **Multi-LLM Support**: OpenAI, Anthropic, and other providers

### 4. Authentication & Security
- **OAuth2/JWT**: Industry-standard authentication with JWKS validation
- **Multi-Provider**: Auth0, AWS Cognito, Okta, Azure AD support
- **Role-Based Access**: Fine-grained permissions using JWT claims
- **Development Mode**: `DISABLE_AUTH` for local development

### 5. Background Processing
- **Docket Tasks**: Redis-based task queue for background operations
- **Memory Indexing**: Asynchronous embedding generation and indexing
- **Compaction**: Periodic cleanup and optimization of stored memories

## Environment Configuration

Key environment variables:
```bash
# Redis
REDIS_URL=redis://localhost:6379

# Authentication (Production)
OAUTH2_ISSUER_URL=https://your-auth-provider.com
OAUTH2_AUDIENCE=your-api-audience
DISABLE_AUTH=false  # Never true in production

# Development
DISABLE_AUTH=true   # Local development only
LOG_LEVEL=DEBUG

# AI Services
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
GENERATION_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Memory Configuration
LONG_TERM_MEMORY=true
WINDOW_SIZE=20
ENABLE_TOPIC_EXTRACTION=true
ENABLE_NER=true
```

## API Interfaces

### REST API (Port 8000)
- Session management (`/v1/working-memory/`)
- Working memory operations (`/v1/working-memory/{id}`)
- Long-term memory search (`/v1/long-term-memory/search`)
- Memory hydration (`/v1/memory/prompt`)

### MCP Server (Port 9000)
- `create_long_term_memories` - Store persistent memories
- `search_long_term_memory` - Semantic search with filtering
- `memory_prompt` - Hydrate queries with relevant context
- `set_working_memory` - Manage session memory

## Development Workflow

0. **Install uv**: `pip install uv` to get started with uv
1. **Setup**: `uv install` to install dependencies
2. **Redis**: Start Redis Stack via `docker-compose up redis`
3. **Development**: Use `DISABLE_AUTH=true` for local testing
4. **Testing**: Run `uv run pytest` before committing
5. **Linting**: Pre-commit hooks handle code formatting
6. **Background Tasks**: Start worker with `uv run agent-memory task-worker`

## Documentation
- API docs available at `/docs` when server is running
- OpenAPI spec at `/openapi.json`
- Authentication examples in README.md
- System architecture diagram in `diagram.png`
