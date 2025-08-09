# REST API Endpoints

The following endpoints are available:

- **GET /v1/health**
  A simple health check endpoint returning the current server time.
  Example Response:

  ```json
  { "now": 1616173200 }
  ```

- **GET /v1/working-memory/**
  Retrieves a paginated list of session IDs.
  _Query Parameters:_

  - `limit` (int): Number of sessions per page (default: 10)
  - `offset` (int): Number of sessions to skip (default: 0)
  - `namespace` (string, optional): Filter sessions by namespace.

- **GET /v1/working-memory/{session_id}**
  Retrieves working memory for a session, including messages, structured memories,
  context, and metadata.
  _Query Parameters:_

  - `namespace` (string, optional): The namespace to use for the session
  - `model_name` (string, optional): The client's LLM model name to determine appropriate context window size
  - `context_window_max` (int, optional): Direct specification of max context window tokens (overrides model_name)

- **PUT /v1/working-memory/{session_id}**
  Sets working memory for a session, replacing any existing memory.
  Automatically summarizes conversations that exceed the token limit.
  _Request Body Example:_

  ```json
  {
    "messages": [
      { "role": "user", "content": "Hello" },
      { "role": "assistant", "content": "Hi there" }
    ],
    "memories": [
      {
        "id": "mem-123",
        "text": "User prefers direct communication",
        "memory_type": "semantic"
      }
    ],
    "context": "Previous conversation summary...",
    "session_id": "session-123",
    "namespace": "default"
  }
  ```

- **DELETE /v1/working-memory/{session_id}**
  Deletes all working memory (messages, context, structured memories, metadata) for a session.

- **POST /v1/long-term-memory/**
  Creates long-term memories directly, bypassing working memory.
  _Request Body Example:_

  ```json
  {
    "memories": [
      {
        "id": "mem-456",
        "text": "User is interested in AI and machine learning",
        "memory_type": "semantic",
        "session_id": "session-123",
        "namespace": "default"
      }
    ]
  }
  ```

- **POST /v1/long-term-memory/search**
  Performs vector search on long-term memories with advanced filtering options.
  _Request Body Example:_

  ```json
  {
    "text": "Search query text",
    "limit": 10,
    "offset": 0,
    "session_id": { "eq": "session-123" },
    "namespace": { "eq": "default" },
    "topics": { "any": ["AI", "Machine Learning"] },
    "entities": { "all": ["OpenAI", "Claude"] },
    "created_at": { "gte": 1672527600, "lte": 1704063599 },
    "last_accessed": { "gt": 1704063600 },
    "user_id": { "eq": "user-456" },
    "recency_boost": true,
    "recency_w_sem": 0.8,
    "recency_w_recency": 0.2,
    "recency_wf": 0.6,
    "recency_wa": 0.4,
    "recency_half_life_last_access_days": 7.0,
    "recency_half_life_created_days": 30.0
  }
  ```

  When `recency_boost` is enabled (default), results are re-ranked using a combined score of semantic similarity and a recency score computed from `last_accessed` and `created_at`. The optional fields adjust weighting and half-lives. The server rate-limits updates to `last_accessed` in the background when results are returned.

- **POST /v1/long-term-memory/forget**
  Trigger a forgetting pass (admin/maintenance).

  _Request Body Example:_

  ```json
  {
    "policy": {
      "max_age_days": 30,
      "max_inactive_days": 30,
      "budget": null,
      "memory_type_allowlist": null
    },
    "namespace": "ns1",
    "user_id": "u1",
    "session_id": null,
    "limit": 1000,
    "dry_run": true
  }
  ```

  _Response Example:_
  ```json
  {
    "scanned": 123,
    "deleted": 5,
    "deleted_ids": ["id1", "id2"],
    "dry_run": true
  }
  ```

  Notes:
  - Uses the vector store adapter (RedisVL) to select candidates via filters, applies the policy locally, then deletes via the adapter (unless `dry_run=true`).
  - A periodic variant can be scheduled via Docket when enabled in settings.

- **POST /v1/memory/prompt**
  Generates prompts enriched with relevant memory context from both working
  memory and long-term memory. Useful for retrieving context before answering questions.
  _Request Body Example:_

  ```json
  {
    "query": "What did we discuss about AI?",
    "session": {
      "session_id": "session-123",
      "namespace": "default",
      "model_name": "gpt-4o",
      "context_window_max": 4000
    },
    "long_term_search": {
      "text": "AI discussion",
      "limit": 5,
      "namespace": { "eq": "default" }
    }
  }
  ```

## Filter Options

_Filter options for search endpoints:_

- Tag filters (session_id, namespace, topics, entities, user_id):

  - `eq`: Equals this value
  - `ne`: Not equals this value
  - `any`: Contains any of these values
  - `all`: Contains all of these values

- Numeric filters (created_at, last_accessed):
  - `gt`: Greater than
  - `lt`: Less than
  - `gte`: Greater than or equal
  - `lte`: Less than or equal
  - `eq`: Equals
  - `ne`: Not equals
  - `between`: Between two values
