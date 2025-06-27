# Vector Store Backends

The Redis Agent Memory Server supports any vector store backend through a flexible factory system. Instead of maintaining database-specific code, you simply specify a Python function that creates and returns your vectorstore.

## Configuration

Set the `VECTORSTORE_FACTORY` environment variable to point to your factory function:

```bash
# Use the default Redis factory
VECTORSTORE_FACTORY="agent_memory_server.vectorstore_factory.create_redis_vectorstore"

# Use a custom Chroma factory
VECTORSTORE_FACTORY="my_vectorstores.create_chroma"

# Use a custom adapter directly
VECTORSTORE_FACTORY="my_package.adapters.CustomMemoryAdapter.create"
```

## Factory Function Requirements

Your factory function must:

1. **Accept an `embeddings` parameter**: `(embeddings: Embeddings) -> Union[VectorStore, VectorStoreAdapter]`
2. **Return either**:
   - A `VectorStore` instance (will be wrapped in `LangChainVectorStoreAdapter`)
   - A `VectorStoreAdapter` instance (used directly for full customization)

## Complete Working Example

Here's a complete example you can use to test:

```python
# my_simple_vectorstore.py
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List, Optional

class SimpleMemoryVectorStore(VectorStore):
    """A simple in-memory vector store for testing/development."""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.docs = []
        self.vectors = []

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs):
        """Add texts to the store."""
        if metadatas is None:
            metadatas = [{}] * len(texts)

        ids = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            doc_id = metadata.get('id', f"doc_{len(self.docs)}")
            doc = Document(page_content=text, metadata=metadata)
            self.docs.append(doc)
            ids.append(doc_id)

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Simple similarity search (returns all docs for demo)."""
        return self.docs[:k]

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs):
        """Create vectorstore from texts."""
        instance = cls(embedding)
        instance.add_texts(texts, metadatas)
        return instance

def create_simple_vectorstore(embeddings: Embeddings) -> SimpleMemoryVectorStore:
    """Factory function that creates a simple in-memory vectorstore."""
    return SimpleMemoryVectorStore(embeddings)
```

Then configure it:
```bash
# Set the factory to your custom function
VECTORSTORE_FACTORY="my_simple_vectorstore.create_simple_vectorstore"

# Start the server - it will use your custom vectorstore!
python -m agent_memory_server
```

## Examples

### Basic Chroma Factory

```python
# my_vectorstores.py
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

def create_chroma(embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name="memory_records",
        persist_directory="./chroma_data",
        embedding_function=embeddings
    )
```

### Pinecone Factory with Configuration

```python
# my_vectorstores.py
import os
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore

def create_pinecone(embeddings: Embeddings) -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "memory-index"),
        embedding=embeddings,
        api_key=os.getenv("PINECONE_API_KEY")
    )
```

### Custom Adapter Factory

```python
# my_adapters.py
from langchain_core.embeddings import Embeddings
from agent_memory_server.vectorstore_adapter import VectorStoreAdapter
from your_custom_vectorstore import YourVectorStore

class CustomVectorStoreAdapter(VectorStoreAdapter):
    """Custom adapter with specialized memory operations."""

    def __init__(self, vectorstore: YourVectorStore, embeddings: Embeddings):
        super().__init__(vectorstore, embeddings)
        # Custom initialization

    # Override methods as needed...

def create_custom_adapter(embeddings: Embeddings) -> CustomVectorStoreAdapter:
    vectorstore = YourVectorStore(
        host="localhost",
        port=6333,
        collection_name="memories"
    )
    return CustomVectorStoreAdapter(vectorstore, embeddings)
```

### Advanced Configuration Pattern

For complex configuration, you can read from environment variables or config files:

```python
# my_vectorstores.py
import os
import json
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

def create_qdrant(embeddings: Embeddings) -> QdrantVectorStore:
    # Read configuration from environment
    config = json.loads(os.getenv("QDRANT_CONFIG", "{}"))

    return QdrantVectorStore(
        host=config.get("host", "localhost"),
        port=config.get("port", 6333),
        collection_name=config.get("collection_name", "memory_records"),
        embeddings=embeddings,
        **config.get("extra_params", {})
    )
```

Then set:
```bash
VECTORSTORE_FACTORY="my_vectorstores.create_qdrant"
QDRANT_CONFIG='{"host": "my-qdrant.com", "port": 443, "extra_params": {"https": true}}'
```

## Error Handling

The factory system provides clear error messages:

- **Import errors**: Missing dependencies or incorrect module paths
- **Function not found**: Function doesn't exist in the specified module
- **Invalid return type**: Function must return `VectorStore` or `VectorStoreAdapter`
- **Runtime errors**: Issues during vectorstore creation

## Default Redis Factory

The built-in Redis factory is available at:
```
agent_memory_server.vectorstore_factory.create_redis_vectorstore
```

This creates a Redis vectorstore using the configured `redis_url` and `redisvl_index_name` settings.

## Benefits

✅ **Zero database-specific code** in the core system
✅ **Complete flexibility** - configure any vectorstore
✅ **Dynamic imports** - only load what you need
✅ **Custom adapters** - full control over memory operations
✅ **Environment-based config** - no code changes needed

## Supported Backends

| Backend | Type | Installation | Best For |
|---------|------|-------------|----------|
| **Redis** (default) | Self-hosted | Built-in | Development, existing Redis infrastructure |
| **Chroma** | Self-hosted/Cloud | `pip install chromadb` | Local development, prototyping |
| **Pinecone** | Managed Cloud | `pip install pinecone-client` | Production, managed service |
| **Weaviate** | Self-hosted/Cloud | `pip install weaviate-client` | Production, advanced features |
| **Qdrant** | Self-hosted/Cloud | `pip install qdrant-client` | Production, high performance |
| **Milvus** | Self-hosted/Cloud | `pip install pymilvus` | Large scale, enterprise |
| **PostgreSQL/PGVector** | Self-hosted | `pip install langchain-postgres psycopg2-binary` | Existing PostgreSQL infrastructure |
| **LanceDB** | Embedded | `pip install lancedb` | Embedded applications |
| **OpenSearch** | Self-hosted/Cloud | `pip install opensearch-py` | Existing OpenSearch infrastructure |

## Configuration

### Backend Selection

Set the backend using the `LONG_TERM_MEMORY_BACKEND` environment variable:

```bash
# Choose your backend
LONG_TERM_MEMORY_BACKEND=redis  # Default
LONG_TERM_MEMORY_BACKEND=chroma
LONG_TERM_MEMORY_BACKEND=pinecone
LONG_TERM_MEMORY_BACKEND=weaviate
LONG_TERM_MEMORY_BACKEND=qdrant
LONG_TERM_MEMORY_BACKEND=milvus
LONG_TERM_MEMORY_BACKEND=pgvector  # or 'postgres'
LONG_TERM_MEMORY_BACKEND=lancedb
LONG_TERM_MEMORY_BACKEND=opensearch
```

### Installation

Install the memory server with your chosen backend:

```bash
# Install with specific backend
pip install agent-memory-server[redis]      # Default
pip install agent-memory-server[chroma]
pip install agent-memory-server[pinecone]
pip install agent-memory-server[weaviate]
pip install agent-memory-server[qdrant]
pip install agent-memory-server[milvus]
pip install agent-memory-server[pgvector]
pip install agent-memory-server[lancedb]
pip install agent-memory-server[opensearch]

# Install with all backends
pip install agent-memory-server[all]
```

## Backend-Specific Configuration

### Redis (Default)

**Installation:**
```bash
pip install agent-memory-server[redis]
```

**Configuration:**
```bash
LONG_TERM_MEMORY_BACKEND=redis
REDIS_URL=redis://localhost:6379

# RedisVL settings (optional, for compatibility)
REDISVL_DISTANCE_METRIC=COSINE
REDISVL_VECTOR_DIMENSIONS=1536
REDISVL_INDEX_NAME=memory
REDISVL_INDEX_PREFIX=memory
```

**Setup:**
- Requires Redis with RediSearch module (RedisStack recommended)
- Default choice, no additional setup needed if Redis is running

---

### Chroma

**Installation:**
```bash
pip install agent-memory-server[chroma]
```

**Configuration:**
```bash
LONG_TERM_MEMORY_BACKEND=chroma

# For HTTP client mode
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=agent_memory

# For persistent storage mode (alternative)
CHROMA_PERSIST_DIRECTORY=/path/to/chroma/data
```

**Setup:**
- For HTTP mode: Run Chroma server on specified host/port
- For persistent mode: Specify a directory for local storage
- Great for development and prototyping

---

### Pinecone

**Installation:**
```bash
pip install agent-memory-server[pinecone]
```

**Configuration:**
```bash
LONG_TERM_MEMORY_BACKEND=pinecone
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=agent-memory
```

**Setup:**
1. Create a Pinecone account and get API key
2. Create an index in the Pinecone console
3. Set environment and index name in configuration
- Fully managed service, excellent for production

---

### Weaviate

**Installation:**
```bash
pip install agent-memory-server[weaviate]
```

**Configuration:**
```bash
LONG_TERM_MEMORY_BACKEND=weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_weaviate_api_key_here  # Optional for local
WEAVIATE_CLASS_NAME=AgentMemory
```

**Setup:**
- For local: Run Weaviate with Docker
- For cloud: Use Weaviate Cloud Services (WCS)
- Advanced features like hybrid search available

---

### Qdrant

**Installation:**
```bash
pip install agent-memory-server[qdrant]
```

**Configuration:**
```bash
LONG_TERM_MEMORY_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here  # Optional for local
QDRANT_COLLECTION_NAME=agent_memory
```

**Setup:**
- For local: Run Qdrant with Docker
- For cloud: Use Qdrant Cloud
- High performance with excellent filtering capabilities

---

### Milvus

**Installation:**
```bash
pip install agent-memory-server[milvus]
```

**Configuration:**
```bash
LONG_TERM_MEMORY_BACKEND=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=agent_memory
MILVUS_USER=your_milvus_username     # Optional
MILVUS_PASSWORD=your_milvus_password # Optional
```

**Setup:**
- For local: Run Milvus standalone with Docker
- For production: Use Milvus cluster or Zilliz Cloud
- Excellent for large-scale applications

---

### PostgreSQL/PGVector

**Installation:**
```bash
pip install agent-memory-server[pgvector]
```

**Configuration:**
```bash
LONG_TERM_MEMORY_BACKEND=pgvector  # or 'postgres'
POSTGRES_URL=postgresql://user:password@localhost:5432/agent_memory
POSTGRES_TABLE_NAME=agent_memory
```

**Setup:**
1. Install PostgreSQL with pgvector extension
2. Create database and enable pgvector extension:
   ```sql
   CREATE EXTENSION vector;
   ```
- Great for existing PostgreSQL infrastructure

---

### LanceDB

**Installation:**
```bash
pip install agent-memory-server[lancedb]
```

**Configuration:**
```bash
LONG_TERM_MEMORY_BACKEND=lancedb
LANCEDB_URI=./lancedb               # Local directory
LANCEDB_TABLE_NAME=agent_memory
```

**Setup:**
- Embedded database, no separate server needed
- Just specify a local directory for storage
- Good for applications that need embedded vector storage

---

### OpenSearch

**Installation:**
```bash
pip install agent-memory-server[opensearch]
```

**Configuration:**
```bash
LONG_TERM_MEMORY_BACKEND=opensearch
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_USERNAME=your_opensearch_username  # Optional
OPENSEARCH_PASSWORD=your_opensearch_password  # Optional
OPENSEARCH_INDEX_NAME=agent-memory
```

**Setup:**
- For local: Run OpenSearch with Docker
- For cloud: Use Amazon OpenSearch Service or self-hosted
- Good for existing Elasticsearch/OpenSearch infrastructure

## Feature Support Matrix

| Backend | Similarity Search | Metadata Filtering | Hybrid Search | Distance Functions |
|---------|------------------|-------------------|---------------|-------------------|
| Redis | ✅ | ✅ | ❌ | COSINE, L2, IP |
| Chroma | ✅ | ✅ | ❌ | COSINE, L2, IP |
| Pinecone | ✅ | ✅ | ✅ | COSINE, EUCLIDEAN, DOTPRODUCT |
| Weaviate | ✅ | ✅ | ✅ | COSINE, DOT, L2, HAMMING, MANHATTAN |
| Qdrant | ✅ | ✅ | ❌ | COSINE, EUCLIDEAN, DOT |
| Milvus | ✅ | ✅ | ❌ | L2, IP, COSINE, HAMMING, JACCARD |
| PGVector | ✅ | ✅ | ❌ | L2, COSINE, IP |
| LanceDB | ✅ | ✅ | ❌ | L2, COSINE |
| OpenSearch | ✅ | ✅ | ✅ | COSINE, L2 |

## Migration Between Backends

Currently, there is no automated migration tool between backends. To switch backends:

1. Export your data from the current backend (if needed)
2. Change the `LONG_TERM_MEMORY_BACKEND` configuration
3. Install the new backend dependencies
4. Configure the new backend settings
5. Restart the server (it will start with an empty index)
6. Re-index your data (if you have an export)

## Performance Considerations

- **Redis**: Fast for small to medium datasets, good for development
- **Chroma**: Good for prototyping, reasonable performance for small datasets
- **Pinecone**: Excellent performance and scalability, optimized for production
- **Weaviate**: Good performance with advanced features, scales well
- **Qdrant**: High performance, excellent for production workloads
- **Milvus**: Excellent for large-scale deployments, horizontal scaling
- **PGVector**: Good for existing PostgreSQL deployments, limited scale
- **LanceDB**: Good performance for embedded use cases
- **OpenSearch**: Good for existing OpenSearch infrastructure, handles large datasets

## Troubleshooting

### Common Issues

1. **Backend dependencies not installed**: Install with the correct extras: `pip install agent-memory-server[backend_name]`

2. **Connection errors**: Check that your backend service is running and configuration is correct

3. **Authentication failures**: Verify API keys and credentials are correct

4. **Index/Collection doesn't exist**: The system will try to create indexes automatically, but some backends may require manual setup

5. **Performance issues**: Check your vector dimensions match the embedding model (default: 1536 for OpenAI text-embedding-3-small)

### Backend-Specific Troubleshooting

**Redis**: Ensure RediSearch module is loaded (`MODULE LIST` in redis-cli)
**Chroma**: Check if Chroma server is running on the correct port
**Pinecone**: Verify index exists and environment is correct
**Weaviate**: Ensure Weaviate is running and accessible
**Qdrant**: Check Qdrant service status and collection configuration
**Milvus**: Verify Milvus is running and collection exists
**PGVector**: Ensure pgvector extension is installed and enabled
**LanceDB**: Check directory permissions and disk space
**OpenSearch**: Verify OpenSearch is running and index settings are correct

## Next Steps

- See [Configuration Guide](configuration.md) for complete configuration options
- See [API Documentation](api.md) for usage examples
- See [Development Guide](development.md) for setting up a development environment
