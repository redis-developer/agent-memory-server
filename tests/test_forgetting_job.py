from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from agent_memory_server.models import (
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
)


def _mk_result(id: str, created_days: int, accessed_days: int, dist: float = 0.3):
    now = datetime.now(UTC)
    return MemoryRecordResult(
        id=id,
        text=f"mem-{id}",
        dist=dist,
        created_at=now - timedelta(days=created_days),
        updated_at=now - timedelta(days=created_days),
        last_accessed=now - timedelta(days=accessed_days),
        user_id="u1",
        session_id=None,
        namespace="ns1",
        topics=[],
        entities=[],
        memory_hash="",
        memory_type=MemoryTypeEnum.SEMANTIC,
        persisted_at=None,
        extracted_from=[],
        event_date=None,
    )


@pytest.mark.asyncio
async def test_forget_long_term_memories_dry_run_selection():
    # Candidates: keep1 (recent), del1 (old+inactive), del2 (very old)
    results = [
        _mk_result("keep1", created_days=5, accessed_days=2),
        _mk_result("del1", created_days=60, accessed_days=45),
        _mk_result("del2", created_days=400, accessed_days=5),
    ]

    mock_adapter = AsyncMock()
    mock_adapter.search_memories.return_value = MemoryRecordResults(
        memories=results, total=len(results), next_offset=None
    )

    with patch(
        "agent_memory_server.long_term_memory.get_vectorstore_adapter",
        return_value=mock_adapter,
    ):
        from agent_memory_server.long_term_memory import forget_long_term_memories

        policy = {
            "max_age_days": 30,
            "max_inactive_days": 30,
            "budget": None,
            "memory_type_allowlist": None,
        }

        resp = await forget_long_term_memories(
            policy,
            namespace="ns1",
            user_id="u1",
            limit=100,
            dry_run=True,
            pinned_ids=["del1"],
        )

        # No deletes should occur in dry run
        mock_adapter.delete_memories.assert_not_called()
        # Expect only del2 to be selected because del1 is pinned
        assert set(resp["deleted_ids"]) == {"del2"}
        assert resp["deleted"] == 1
        assert resp["scanned"] == 3


@pytest.mark.asyncio
async def test_forget_long_term_memories_executes_deletes_when_not_dry_run():
    results = [
        _mk_result("keep1", created_days=1, accessed_days=1),
        _mk_result("del_old", created_days=365, accessed_days=10),
    ]

    mock_adapter = AsyncMock()
    mock_adapter.search_memories.return_value = MemoryRecordResults(
        memories=results, total=len(results), next_offset=None
    )
    mock_adapter.delete_memories.return_value = 1

    with patch(
        "agent_memory_server.long_term_memory.get_vectorstore_adapter",
        return_value=mock_adapter,
    ):
        from agent_memory_server.long_term_memory import forget_long_term_memories

        policy = {
            "max_age_days": 180,
            "max_inactive_days": None,
            "budget": None,
            "memory_type_allowlist": None,
        }

        resp = await forget_long_term_memories(
            policy, namespace="ns1", user_id="u1", limit=100, dry_run=False
        )

        mock_adapter.delete_memories.assert_called_once_with(["del_old"])
        assert resp["deleted"] == 1
        assert resp["deleted_ids"] == ["del_old"]
