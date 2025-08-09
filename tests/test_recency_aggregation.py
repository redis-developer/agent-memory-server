from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory_server.utils.redis_query import RecencyAggregationQuery
from agent_memory_server.vectorstore_adapter import RedisVectorStoreAdapter


@pytest.mark.asyncio
async def test_recency_aggregation_query_builds_and_paginates():
    # Build a VectorQuery without touching Redis (pure construction)
    from redisvl.query import VectorQuery

    dummy_vec = [0.0, 0.0, 0.0]
    vq = VectorQuery(vector=dummy_vec, vector_field_name="vector", num_results=10)

    # Build aggregation
    agg = (
        RecencyAggregationQuery.from_vector_query(vq)
        .load_default_fields()
        .apply_recency(
            now_ts=1_700_000_000,
            params={
                "w_sem": 0.7,
                "w_recency": 0.3,
                "wf": 0.5,
                "wa": 0.5,
                "half_life_last_access_days": 5.0,
                "half_life_created_days": 20.0,
            },
        )
        .sort_by_boosted_desc()
        .paginate(5, 7)
    )

    # Validate the aggregate request contains APPLY, SORTBY, and LIMIT via build_args
    args = agg.build_args()
    args_str = " ".join(map(str, args))
    assert "APPLY" in args_str
    assert "boosted_score" in args_str
    assert "SORTBY" in args_str
    assert "LIMIT" in args_str


@pytest.mark.asyncio
async def test_redis_adapter_uses_aggregation_when_server_side_recency():
    # Mock vectorstore and its underlying RedisVL index
    mock_index = MagicMock()

    class Rows:
        def __init__(self, rows):
            self.rows = rows

    # Simulate aaggregate returning rows from FT.AGGREGATE
    mock_index.aaggregate = AsyncMock(
        return_value=Rows(
            [
                {
                    "id_": "m1",
                    "namespace": "ns",
                    "session_id": "s1",
                    "user_id": "u1",
                    "created_at": 1_700_000_000,
                    "last_accessed": 1_700_000_000,
                    "updated_at": 1_700_000_000,
                    "pinned": 0,
                    "access_count": 1,
                    "topics": "",
                    "entities": "",
                    "memory_hash": "h",
                    "discrete_memory_extracted": "t",
                    "memory_type": "semantic",
                    "persisted_at": None,
                    "extracted_from": "",
                    "event_date": None,
                    "text": "hello",
                    "__vector_score": 0.9,
                }
            ]
        )
    )

    mock_vectorstore = MagicMock()
    mock_vectorstore._index = mock_index
    # If the adapter falls back, ensure awaited LC call is defined
    mock_vectorstore.asimilarity_search_with_relevance_scores = AsyncMock(
        return_value=[]
    )

    # Mock embeddings
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.0, 0.0, 0.0]

    adapter = RedisVectorStoreAdapter(mock_vectorstore, mock_embeddings)

    results = await adapter.search_memories(
        query="hello",
        server_side_recency=True,
        namespace=None,
        limit=5,
        offset=0,
    )

    # Ensure we went through aggregate path
    assert mock_index.aaggregate.await_count == 1
    assert len(results.memories) == 1
    assert results.memories[0].id == "m1"
    assert results.memories[0].text == "hello"
