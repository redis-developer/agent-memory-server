#!/usr/bin/env python3
"""
E-commerce Shopping Assistant (Preference Memory)

Demonstrates storing, updating, and using user preferences as long-term memories,
and a session cart stored in working memory data.

Two modes:
- Interactive (default): REPL commands
- Demo (--demo): seeds preferences, recommends, updates, and shows recall

Environment variables:
- MEMORY_SERVER_URL (default: http://localhost:8000)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from agent_memory_client import MemoryAPIClient, create_memory_client
from agent_memory_client.filters import MemoryType, Namespace, Topics, UserId
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum, WorkingMemory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


DEFAULT_USER = "shopper"
DEFAULT_SESSION = "shopping_session"
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")


def _namespace(user_id: str) -> str:
    return f"shopping_assistant:{user_id}"


async def _get_client() -> MemoryAPIClient:
    return await create_memory_client(base_url=MEMORY_SERVER_URL, timeout=30.0)


def _get_llm() -> ChatOpenAI | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return ChatOpenAI(model="gpt-4o", temperature=0)


EXTRACT_PREFS_FN = {
    "name": "extract_preferences",
    "description": "Extract normalized user preferences from an utterance.",
    "parameters": {
        "type": "object",
        "properties": {
            "preferences": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Key-value preferences like size, brand, color, budget",
            }
        },
        "required": ["preferences"],
    },
}


def _llm_bind(functions: list[dict]) -> ChatOpenAI | None:
    llm = _get_llm()
    if not llm:
        return None
    return llm.bind_functions(functions)


async def set_preferences_from_utterance(
    user_id: str, utterance: str
) -> dict[str, str] | None:
    llm = _llm_bind([EXTRACT_PREFS_FN])
    if not llm:
        return None
    system = {
        "role": "system",
        "content": "Extract preferences via the function call only.",
    }
    user = {"role": "user", "content": utterance}
    resp = llm.invoke([system, user])
    fn = getattr(resp, "additional_kwargs", {}).get("function_call")
    if not fn:
        return None
    import json as _json

    try:
        args = (
            _json.loads(fn["arguments"])
            if isinstance(fn.get("arguments"), str)
            else fn.get("arguments", {})
        )
    except Exception:
        return None
    prefs = args.get("preferences", {})
    # Persist each as semantic preference
    for k, v in prefs.items():
        await set_preference(user_id, k, str(v))
    return {k: str(v) for k, v in prefs.items()}


async def set_preference(user_id: str, key: str, value: str) -> None:
    client = await _get_client()
    ns = _namespace(user_id)
    record = ClientMemoryRecord(
        text=f"Preference {key} = {value}",
        memory_type=MemoryTypeEnum.SEMANTIC,
        topics=["preferences"],
        entities=[key, value],
        namespace=ns,
        user_id=user_id,
    )
    await client.create_long_term_memory([record])


async def list_preferences(user_id: str) -> list[dict[str, Any]]:
    client = await _get_client()
    ns = _namespace(user_id)
    # Empty-text search pattern for "what do you remember about me?"
    results = await client.search_long_term_memory(
        text="",
        namespace=Namespace(eq=ns),
        topics=Topics(any=["preferences"]),
        user_id=UserId(eq=user_id),
        memory_type=MemoryType(eq="semantic"),
        limit=50,
        optimize_query=False,
    )
    return [m.model_dump() for m in results.memories]


async def recommend(
    user_id: str, occasion: str, budget: int | None, color: str | None
) -> str:
    # Let the LLM compose the recommendation text using remembered prefs
    prefs = await list_preferences(user_id)
    pref_map: dict[str, str] = {}
    for m in prefs:
        text = m["text"]
        if text.startswith("Preference ") and " = " in text:
            k, v = text[len("Preference ") :].split(" = ", 1)
            pref_map[k.strip()] = v.strip()
    llm = _get_llm()
    if not llm:
        # Fallback if no LLM
        size = pref_map.get("size", "M")
        brand = pref_map.get("brand", "Acme")
        base_color = color or pref_map.get("color", "navy")
        price = budget or int(pref_map.get("budget", "150"))
        return f"Suggested outfit for {occasion}: {brand} {base_color} blazer, size {size}, around ${price}."
    messages = [
        {
            "role": "system",
            "content": "Compose a concise recommendation using the preferences.",
        },
        {
            "role": "user",
            "content": f"Occasion: {occasion}. Constraints: budget={budget}, color={color}. Preferences: {pref_map}",
        },
    ]
    return str(llm.invoke(messages).content)


async def _get_working_memory(user_id: str, session_id: str) -> WorkingMemory:
    client = await _get_client()
    ns = _namespace(user_id)
    wm = await client.get_working_memory(session_id=session_id, namespace=ns)
    return WorkingMemory(**wm.model_dump())


async def add_to_cart(user_id: str, session_id: str, item: dict[str, Any]) -> None:
    client = await _get_client()
    ns = _namespace(user_id)
    wm = await _get_working_memory(user_id, session_id)
    data = wm.data or {}
    cart = data.get("cart", [])
    if not isinstance(cart, list):
        cart = []
    cart.append(item)
    data["cart"] = cart
    await client.update_working_memory_data(
        session_id=session_id, data_updates=data, namespace=ns
    )


async def show_cart(user_id: str, session_id: str) -> list[dict[str, Any]]:
    wm = await _get_working_memory(user_id, session_id)
    cart = wm.data.get("cart", []) if wm.data else []
    return cart if isinstance(cart, list) else []


async def clear_cart(user_id: str, session_id: str) -> None:
    client = await _get_client()
    ns = _namespace(user_id)
    await client.update_working_memory_data(
        session_id=session_id,
        data_updates={"cart": []},
        namespace=ns,
        merge_strategy="replace",
    )


DEMO_STEPS = [
    ("set", {"key": "size", "value": "L"}),
    ("set", {"key": "brand", "value": "TailorCo"}),
    ("set", {"key": "color", "value": "charcoal"}),
    ("set", {"key": "budget", "value": "200"}),
    ("recommend", {"occasion": "wedding", "budget": 200, "color": None}),
    ("add", {"item": {"sku": "TC-CHA-BLAZER", "price": 199}}),
    ("cart", {}),
    ("set", {"key": "size", "value": "XL"}),
    ("recommend", {"occasion": "reception", "budget": None, "color": "navy"}),
    ("remember", {}),
]


async def run_demo(user_id: str, session_id: str) -> None:
    print("🛍️  Shopping Assistant Demo")
    for cmd, args in DEMO_STEPS:
        if cmd == "set":
            await set_preference(user_id, args["key"], args["value"])
            print(f"Set {args['key']}={args['value']}")
        elif cmd == "recommend":
            rec = await recommend(
                user_id, args["occasion"], args["budget"], args["color"]
            )
            print(f"Recommendation: {rec}")
        elif cmd == "add":
            await add_to_cart(user_id, session_id, args["item"])
            print(f"Added to cart: {json.dumps(args['item'])}")
        elif cmd == "cart":
            print(f"Cart: {json.dumps(await show_cart(user_id, session_id))}")
        elif cmd == "remember":
            prefs = await list_preferences(user_id)
            print("Preferences:")
            for m in prefs:
                print(f"- {m['text']}")


async def run_interactive(user_id: str, session_id: str) -> None:
    print("🛍️  Shopping Assistant - Interactive Mode")
    print(
        'Commands:\n  set key=value\n  set-from "utterance" (LLM extraction)\n  show-prefs\n  recommend <occasion> [--budget B] [--color C]\n  add {json_item}\n  cart\n  clear-cart\n  remember\n  exit'
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

        try:
            if raw.startswith("set ") and "=" in raw:
                _, pair = raw.split(" ", 1)
                key, value = pair.split("=", 1)
                await set_preference(user_id, key.strip(), value.strip())
                print("OK")

            elif raw == "show-prefs" or raw == "remember":
                prefs = await list_preferences(user_id)
                for m in prefs:
                    print(f"- {m['text']}")

            elif raw.startswith("set-from "):
                utterance = raw[len("set-from ") :].strip().strip('"')
                extracted = await set_preferences_from_utterance(user_id, utterance)
                if extracted:
                    print(f"Set: {extracted}")
                else:
                    print("No preferences extracted or LLM not configured")

            elif raw.startswith("recommend "):
                parts = raw.split()
                occasion = parts[1]
                budget = None
                color = None
                if "--budget" in parts:
                    i = parts.index("--budget")
                    if i + 1 < len(parts):
                        budget = int(parts[i + 1])
                if "--color" in parts:
                    i = parts.index("--color")
                    if i + 1 < len(parts):
                        color = parts[i + 1]
                print(await recommend(user_id, occasion, budget, color))

            elif raw.startswith("add "):
                _, json_str = raw.split(" ", 1)
                item = json.loads(json_str)
                await add_to_cart(user_id, session_id, item)
                print("OK")

            elif raw == "cart":
                print(json.dumps(await show_cart(user_id, session_id)))

            elif raw == "clear-cart":
                await clear_cart(user_id, session_id)
                print("OK")

            else:
                print("Unknown command")

        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shopping Assistant")
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
