# 🔮 Redis Agent Memory Server

A Redis-powered memory server built for AI agents and applications. It manages both conversational context and long-term memories, offering semantic search, automatic summarization, and flexible APIs through both REST and MCP interfaces.

## Features

- **Working Memory**

  - Session-scoped storage for messages, structured memories, context, and metadata
  - Automatically summarizes conversations when they exceed the window size
  - Token limit management (adapts to the context window of the client's LLM)
  - Supports all major OpenAI and Anthropic models
  - Automatic promotion of structured memories to long-term storage

- **Long-Term Memory**

  - Persistent storage for memories across sessions
  - Semantic search to retrieve memories with advanced filtering system
  - Filter by session, namespace, topics, entities, timestamps, and more
  - Supports both exact match and semantic similarity search
  - Automatic topic modeling for stored memories with BERTopic or configured LLM
  - Automatic Entity Recognition using BERT
  - Memory deduplication and compaction

- **Other Features**
  - Namespace support for session and working memory isolation
  - Both a REST interface and MCP server
  - Background task processing for memory indexing and promotion
  - Unified search across working memory and long-term memory

For detailed information about memory types, their differences, and when to use each, see the [Memory Types Guide](docs/memory-types.md).

## Authentication

The Redis Agent Memory Server supports OAuth2/JWT Bearer token authentication for secure API access. It's compatible with Auth0, AWS Cognito, Okta, Azure AD, and other standard OAuth2 providers.

For complete authentication setup, configuration, and usage examples, see [Authentication Documentation](docs/authentication.md).

For manual Auth0 testing, see the [manual OAuth testing guide](manual_oauth_qa/README.md).

## System Diagram

![System Diagram](diagram.png)

## Project Status and Roadmap

### Project Status: In Development, Pre-Release

This project is under active development and is **pre-release** software. Think of it as an early beta!

### Roadmap

- [x] Long-term memory deduplication and compaction
- [x] Use a background task system instead of `BackgroundTask`
- [x] Authentication/authorization hooks (OAuth2/JWT support)
- [ ] Configurable strategy for moving working memory to long-term memory
- [ ] Separate Redis connections for long-term and working memory

## REST API Endpoints

The server provides REST endpoints for managing working memory, long-term memory, and memory search. Key endpoints include session management, memory storage/retrieval, semantic search, and memory-enriched prompts.

For complete API documentation with examples, see [REST API Documentation](docs/api.md).

## MCP Server Interface

Agent Memory Server offers an MCP (Model Context Protocol) server interface powered by FastMCP, providing tool-based memory management for LLMs and agents. Includes tools for working memory, long-term memory, semantic search, and memory-enriched prompts.

For complete MCP setup and usage examples, see [MCP Documentation](docs/mcp.md).

## Command Line Interface

The `agent-memory-server` provides a comprehensive CLI for managing servers and tasks. Key commands include starting API/MCP servers, scheduling background tasks, running workers, and managing migrations.

For complete CLI documentation and examples, see [CLI Documentation](docs/cli.md).

## Getting Started

For complete setup instructions, see [Getting Started Guide](docs/getting-started.md).

## Configuration

Configure servers and workers using environment variables. Includes background task management, memory compaction, and data migrations.

For complete configuration details, see [Configuration Guide](docs/configuration.md).

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

## Development

For development setup, testing, and contributing guidelines, see [Development Guide](docs/development.md).
