#!/usr/bin/env python3
"""
Meeting Memory Orchestrator (Episodic Memories)

This example demonstrates managing meeting knowledge using episodic memories:

1) Ingest meeting transcripts and extract action items and decisions
2) Store each item as a long-term EPISODIC memory with event_date/topics/entities
3) Query decisions and open action items with time/topic filters
4) Mark tasks done by editing memories

Two modes:
- Interactive (default): simple REPL with commands
- Demo (--demo): automated run with two synthetic meetings

Environment variables:
- MEMORY_SERVER_URL (default: http://localhost:8000)

You can enable smarter extraction and query intent parsing with an LLM by setting
OPENAI_API_KEY. Without it, the script falls back to deterministic parsing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from agent_memory_client import MemoryAPIClient, create_memory_client
from agent_memory_client.filters import (
    CreatedAt,
    MemoryType,
    Namespace,
    Topics,
)
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


DEFAULT_USER = "demo_user"
DEFAULT_SESSION = "meeting_memory_demo"
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")


def _namespace(user_id: str) -> str:
    return f"meeting_memory:{user_id}"


@dataclass
class MeetingItem:
    kind: str  # "action" | "decision"
    text: str
    owner: str | None = None
    due: str | None = None
    topic: str | None = None


ACTION_RE = re.compile(r"^\s*(?:Action|ACTION)\s*:\s*(.+?)\s*$")
DECISION_RE = re.compile(r"^\s*(?:Decision|DECISION)\s*:\s*(.+?)\s*$")
OWNER_RE = re.compile(r"\b(?:Owner|owner)\s*:\s*([A-Za-z0-9_\- ]+)\b")
DUE_RE = re.compile(r"\b(?:Due|due)\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\b")
TOPIC_RE = re.compile(r"\b(?:Topic|topic)\s*:\s*([A-Za-z0-9_\- ]+)\b")


def extract_items_from_transcript(text: str) -> list[MeetingItem]:
    items: list[MeetingItem] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m_action = ACTION_RE.search(line)
        m_decision = DECISION_RE.search(line)
        if not m_action and not m_decision:
            continue

        kind = "action" if m_action else "decision"
        body = (m_action or m_decision).group(1)  # type: ignore

        owner = _first_group_or_none(OWNER_RE.search(line))
        due = _first_group_or_none(DUE_RE.search(line))
        topic = _first_group_or_none(TOPIC_RE.search(line))

        items.append(
            MeetingItem(kind=kind, text=body, owner=owner, due=due, topic=topic)
        )
    return items


def _first_group_or_none(match: re.Match[str] | None) -> str | None:
    return match.group(1).strip() if match else None


async def _get_client() -> MemoryAPIClient:
    return await create_memory_client(base_url=MEMORY_SERVER_URL, timeout=30.0)


def _get_llm() -> ChatOpenAI | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    # Provide function-calling capable model
    return ChatOpenAI(model="gpt-4o", temperature=0)


EXTRACT_ITEMS_FN = {
    "name": "extract_meeting_items",
    "description": "Extract structured meeting items from a transcript.",
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["action", "decision"]},
                        "text": {"type": "string"},
                        "owner": {"type": "string"},
                        "due": {
                            "type": "string",
                            "description": "YYYY-MM-DD if present",
                        },
                        "topic": {"type": "string"},
                    },
                    "required": ["kind", "text"],
                },
            }
        },
        "required": ["items"],
    },
}


TRANSLATE_QUERY_FN = {
    "name": "translate_meeting_query",
    "description": "Translate a natural language meeting question into filters.",
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["action", "decision", "any"]},
            "topic": {"type": "string"},
            "since_days": {"type": "integer", "minimum": 0},
            "query_text": {
                "type": "string",
                "description": "fallback semantic search text",
            },
        },
    },
}


def _llm_bind(*functions: dict) -> ChatOpenAI | None:
    llm = _get_llm()
    if not llm:
        return None
    return llm.bind_functions(list(functions))


def extract_items_via_llm(transcript: str) -> list[MeetingItem] | None:
    llm = _llm_bind(EXTRACT_ITEMS_FN)
    if not llm:
        return None
    system = {
        "role": "system",
        "content": "You extract meeting action items and decisions and return them via the function call.",
    }
    user = {
        "role": "user",
        "content": f"Extract items from this transcript. Use extract_meeting_items.\n\n{transcript}",
    }
    resp = llm.invoke([system, user])
    fn = getattr(resp, "additional_kwargs", {}).get("function_call")
    if not fn:
        return None
    try:
        args = (
            json.loads(fn["arguments"])
            if isinstance(fn.get("arguments"), str)
            else fn.get("arguments", {})
        )
    except json.JSONDecodeError:
        return None
    items_payload = args.get("items", [])
    items: list[MeetingItem] = []
    for it in items_payload:
        try:
            items.append(
                MeetingItem(
                    kind=it.get("kind", "").strip(),
                    text=it.get("text", "").strip(),
                    owner=(it.get("owner") or None),
                    due=(it.get("due") or None),
                    topic=(it.get("topic") or None),
                )
            )
        except Exception:
            continue
    return items


def translate_query_via_llm(question: str) -> dict[str, Any] | None:
    llm = _llm_bind(TRANSLATE_QUERY_FN)
    if not llm:
        return None
    system = {
        "role": "system",
        "content": "Translate user questions about meetings into simple filters via function call.",
    }
    user = {"role": "user", "content": question}
    resp = llm.invoke([system, user])
    fn = getattr(resp, "additional_kwargs", {}).get("function_call")
    if not fn:
        return None
    try:
        args = (
            json.loads(fn["arguments"])
            if isinstance(fn.get("arguments"), str)
            else fn.get("arguments", {})
        )
    except json.JSONDecodeError:
        return None
    return args


async def store_meeting_items(
    items: Iterable[MeetingItem], *, user_id: str, event_date: datetime
) -> None:
    client = await _get_client()
    ns = _namespace(user_id)
    records: list[ClientMemoryRecord] = []

    for item in items:
        topics: list[str] = ["meeting", item.kind]
        if item.topic:
            topics.append(item.topic)

        entities: list[str] = []
        if item.owner:
            entities.append(item.owner)

        text_parts = [f"{item.kind.title()}: {item.text}"]
        if item.owner:
            text_parts.append(f"Owner: {item.owner}")
        if item.due:
            text_parts.append(f"Due: {item.due}")
        text_parts.append("Status: open")

        record = ClientMemoryRecord(
            text=" | ".join(text_parts),
            memory_type=MemoryTypeEnum.EPISODIC,
            topics=topics,
            entities=entities or None,
            namespace=ns,
            user_id=user_id,
            event_date=event_date,
        )
        records.append(record)

    if records:
        await client.create_long_term_memory(records)


async def list_items(
    *,
    user_id: str,
    since_days: int | None = None,
    topic: str | None = None,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    client = await _get_client()
    ns = _namespace(user_id)

    created_at = None
    if since_days is not None and since_days > 0:
        created_at = CreatedAt(gte=(datetime.now(UTC) - timedelta(days=since_days)))

    topics_filter = None
    if topic and kind:
        topics_filter = Topics(all=["meeting", topic, kind])
    elif topic:
        topics_filter = Topics(all=["meeting", topic])
    elif kind:
        topics_filter = Topics(all=["meeting", kind])
    else:
        topics_filter = Topics(all=["meeting"])  # default to all meeting items

    results = await client.search_long_term_memory(
        text="meeting items",
        namespace=Namespace(eq=ns),
        topics=topics_filter,
        created_at=created_at,
        memory_type=MemoryType(eq="episodic"),
        limit=100,
        optimize_query=False,
    )
    # Return as dicts for easier display
    return [m.model_dump() for m in results.memories]


async def mark_done(*, memory_id: str) -> dict[str, Any]:
    client = await _get_client()
    # Fetch, update text status
    mem = await client.get_long_term_memory(memory_id)
    text = mem.text
    if "Status:" in text:
        new_text = re.sub(r"Status:\s*\w+", "Status: done", text)
    else:
        new_text = text + " | Status: done"
    updated = await client.edit_long_term_memory(memory_id, {"text": new_text})
    return updated.model_dump()


async def search_items(*, user_id: str, query: str) -> list[dict[str, Any]]:
    client = await _get_client()
    ns = _namespace(user_id)
    results = await client.search_long_term_memory(
        text=query,
        namespace=Namespace(eq=ns),
        topics=Topics(any=["meeting", "action", "decision"]),
        memory_type=MemoryType(eq="episodic"),
        limit=50,
    )
    return [m.model_dump() for m in results.memories]


DEMO_MEETING_1 = """
Topic: CI
Decision: Adopt GitHub Actions for CI | Owner: Team Infra
Action: Create base CI workflow file | Owner: Priya | Due: 2025-08-20
Action: Add test matrix for Python versions | Owner: Marco
""".strip()

DEMO_MEETING_2 = """
Topic: Hiring
Decision: Proceed with offer for Backend Engineer | Owner: Hiring
Action: Draft offer letter | Owner: Sam | Due: 2025-08-25
Action: Schedule onboarding plan | Owner: Lee
""".strip()


async def run_demo(user_id: str, session_id: str) -> None:
    print("🗂️  Meeting Memory Orchestrator Demo")
    print("This demo ingests two meetings and shows queries.")

    # Ingest
    for idx, (txt, event_date) in enumerate(
        [
            (DEMO_MEETING_1, datetime.now(UTC) - timedelta(days=7)),
            (DEMO_MEETING_2, datetime.now(UTC)),
        ],
        start=1,
    ):
        items = extract_items_from_transcript(txt)
        await store_meeting_items(items, user_id=user_id, event_date=event_date)
        print(f"✅ Ingested meeting {idx} with {len(items)} items")

    # Queries
    decisions = await list_items(user_id=user_id, kind="decision")
    print(f"\nDecisions ({len(decisions)}):")
    for m in decisions:
        print(f"- {m['text']}")

    open_actions = [
        m
        for m in await list_items(user_id=user_id, kind="action")
        if "Status: open" in m["text"]
    ]
    print(f"\nOpen Actions ({len(open_actions)}):")
    for m in open_actions:
        print(f"- {m['id']}: {m['text']}")

    # Mark first open action done
    if open_actions:
        updated = await mark_done(memory_id=open_actions[0]["id"])
        print(f"\n✅ Marked done: {updated['id']}")

    # Search by topic
    hiring = await list_items(user_id=user_id, topic="Hiring")
    print(f"\nItems with topic 'Hiring' ({len(hiring)}):")
    for m in hiring:
        print(f"- {m['text']}")


async def run_interactive(user_id: str, session_id: str) -> None:
    print("🗂️  Meeting Memory Orchestrator - Interactive Mode")
    print(
        "Commands:\n  ingest            (paste transcript, end with a single '.' line)\n  ingest <path>     (load transcript from file)\n  list [--days N] [--topic T] [--kind action|decision]\n  decisions         (list decisions)\n  open-tasks        (list open action items)\n  done <id>         (mark task done)\n  search <query>    (semantic search)\n  exit"
    )

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            return
        if not raw:
            continue
        if raw.lower() in {"exit", "quit"}:
            print("Bye")
            return

        cmd, *rest = raw.split()
        try:
            if cmd == "ingest":
                if rest:
                    path = rest[0]
                    with open(path, encoding="utf-8") as f:
                        text = f.read()
                else:
                    print(
                        "Paste transcript lines; finish with a single '.' on a new line:"
                    )
                    lines: list[str] = []
                    while True:
                        line = input()
                        if line.strip() == ".":
                            break
                        lines.append(line)
                    text = "\n".join(lines)

                items = extract_items_via_llm(text) or extract_items_from_transcript(
                    text
                )
                await store_meeting_items(
                    items, user_id=user_id, event_date=datetime.now(UTC)
                )
                print(f"Stored {len(items)} items.")

            elif cmd == "list":
                days = None
                topic = None
                kind = None
                # naive arg parsing
                if "--days" in rest:
                    i = rest.index("--days")
                    if i + 1 < len(rest):
                        days = int(rest[i + 1])
                if "--topic" in rest:
                    i = rest.index("--topic")
                    if i + 1 < len(rest):
                        topic = rest[i + 1]
                if "--kind" in rest:
                    i = rest.index("--kind")
                    if i + 1 < len(rest):
                        kind = rest[i + 1]
                items = await list_items(
                    user_id=user_id, since_days=days, topic=topic, kind=kind
                )
                for m in items:
                    print(f"- {m['id']}: {m['text']}")

            elif cmd == "decisions":
                for m in await list_items(user_id=user_id, kind="decision"):
                    print(f"- {m['id']}: {m['text']}")

            elif cmd == "open-tasks":
                items = await list_items(user_id=user_id, kind="action")
                for m in items:
                    if "Status: open" in m["text"]:
                        print(f"- {m['id']}: {m['text']}")

            elif cmd == "done" and rest:
                updated = await mark_done(memory_id=rest[0])
                print(f"Updated: {updated['id']}")

            elif cmd == "search" and rest:
                items = await search_items(user_id=user_id, query=" ".join(rest))
                for m in items:
                    print(f"- {m['id']}: {m['text']}")

            elif cmd == "ask" and rest:
                q = " ".join(rest)
                params = translate_query_via_llm(q) or {}
                kind = params.get("kind")
                topic = params.get("topic")
                since_days = params.get("since_days")
                query_text = params.get("query_text")
                if kind or topic or since_days:
                    results = await list_items(
                        user_id=user_id,
                        since_days=since_days,
                        topic=topic,
                        kind=(None if kind == "any" else kind),
                    )
                elif query_text:
                    results = await search_items(user_id=user_id, query=query_text)
                else:
                    results = []
                for m in results:
                    print(f"- {m['id']}: {m['text']}")

            else:
                print("Unknown command")

        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting Memory Orchestrator")
    parser.add_argument("--user-id", default=DEFAULT_USER)
    parser.add_argument("--session-id", default=DEFAULT_SESSION)
    parser.add_argument("--memory-server-url", default=MEMORY_SERVER_URL)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.memory_server_url:
        os.environ["MEMORY_SERVER_URL"] = args.memory_server_url

    if args.demo:
        asyncio.run(run_demo(args.user_id, args.session_id))
    else:
        asyncio.run(run_interactive(args.user_id, args.session_id))


if __name__ == "__main__":
    main()
