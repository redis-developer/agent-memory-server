# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2025-06-13

*Changes from the initial release:*

### Architecture Evolution
- **Working Memory (formerly Short-term Memory)**:
  - Renamed from "short-term memory" to "working memory" to better reflect its purpose
  - Enhanced with automatic promotion system that moves structured memories to long-term storage in background
  - Added support for arbitrary JSON data storage alongside memory structures
  - Improved automatic conversation summarization in working memory

- **Long-term Memory Promotion**:
  - Implemented seamless flow from working memory to long-term memory via background task processing
  - Agent only has to think about working memory, long-term memory is managed automatically (but can be managed manually, too)
  - Use any LangChain `VectorStore` subclass for long-term storage, defaults to `RedisVectorStore`
  - Structured memories automatically promoted with vector embeddings and metadata indexing
  - Deduplication and compaction systems for long-term memory management
  - Background task worker system using Docket for reliable memory processing

### Client SDK and Tooling
  - Both working and long-term memory available as tools for LLM integration (LLM can choose to persist a long-term memory or search for long-term memories)
  - Memory-enriched prompt generation via `/v1/memory/prompt` endpoint
  - Unified search across both memory types with metadata filtering
  - Support for namespace isolation and session management

### Search and Retrieval
  - Vector-based similarity search using OpenAI embeddings
  - Rich filtering system by session, namespace, topics, entities, timestamps
  - Hybrid search combining semantic similarity with metadata filtering
  - RedisVL integration for high-performance vector operations with Redis

### Enhanced Memory Classification:
  - Semantic memories for facts and preferences
  - Episodic memories for time-bound events with event dates (requires a timeframe)
  - Message memories for conversation records
  - Automatic topic modeling and entity recognition either using BERTopic or a configured LLM
  - Rich metadata extraction and indexing

### Authentication and Security
  - OAuth2/JWT Bearer token authentication with JWKS validation
  - Multi-provider support (Auth0, AWS Cognito, Okta, Azure AD)
  - Role-based access control using JWT claims
  - Development mode with configurable auth bypass

### Operational Features
- **Comprehensive CLI Interface**:
  - Commands for server management (`api`, `mcp`, `task-worker`)
  - Database operations (`rebuild-index`)
  - Background task scheduling and management
  - Health monitoring and diagnostics


## [0.0.1]

### Initial Release - 2025-04-07
- Initial release with basic short-term and long-term memory functionality
