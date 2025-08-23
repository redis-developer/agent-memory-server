#!/usr/bin/env python3
"""
Memory Editing Agent Example

This example demonstrates how to use the Agent Memory Server's memory editing capabilities
through tool calls in a conversational AI scenario. The agent can:

1. Create and store memories about user preferences and information
2. Search for existing memories to review and update
3. Edit memories when new information is provided or corrections are needed
4. Delete memories that are no longer relevant
5. Retrieve specific memories by ID for detailed review

This showcases a realistic workflow where an AI assistant manages and updates
user information over time through natural conversation.

Environment variables:
- OPENAI_API_KEY: Required for OpenAI ChatGPT
- MEMORY_SERVER_URL: Memory server URL (default: http://localhost:8000)
"""

import asyncio
import json
import logging
import os

from agent_memory_client import (
    MemoryAPIClient,
    create_memory_client,
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce third-party logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Environment setup
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")
DEFAULT_USER = "demo_user"

SYSTEM_PROMPT = {
    "role": "system",
    "content": """
    You are a helpful personal assistant that learns about the user over time.
    You can search, store, update, and remove information using memory tools as needed.

    Principles:
    - Be natural and conversational; focus on helping the user with their goals.
    - Keep what you know about the user accurate and up to date.
    - When updating or deleting stored information, first find the relevant
    memory and use its exact id for changes. If uncertain, ask a brief
    clarifying question.
    - Avoid surfacing implementation details (e.g., tool names) to the user.
    Summarize outcomes succinctly.
    - Do not create duplicate memories if an equivalent one already exists.

    Time and date grounding rules:
    - When users mention relative dates ("today", "yesterday", "last week"),
      call get_current_datetime to ground to an absolute date/time.
    - For episodic updates, ALWAYS set event_date and also include the grounded,
      human-readable date in the text (e.g., "on August 14, 2025").
    - Do not guess dates. If unsure, ask or omit the date phrase in text while
      still setting event_date only when certain.

    Available capabilities (for your use, not to be listed to the user):
    - search previous information, review current session context, add important facts, and edit/delete existing items by id.
    - When you receive paginated search results ('has_more' is true with a 'next_offset'), iterate with the same query and offset to retrieve more results if needed to answer the user.
    """,
}


class MemoryEditingAgent:
    """
    A conversational agent that demonstrates comprehensive memory editing capabilities.

    This agent shows how to manage user information through natural conversation,
    including creating, searching, editing, and deleting memories as needed.
    """

    def __init__(self):
        self._memory_client: MemoryAPIClient | None = None
        self._setup_llm()

    def _get_namespace(self, user_id: str) -> str:
        """Generate consistent namespace for a user."""
        return f"memory_editing_agent:{user_id}"

    async def get_client(self) -> MemoryAPIClient:
        """Get the memory client, initializing it if needed."""
        if not self._memory_client:
            self._memory_client = await create_memory_client(
                base_url=MEMORY_SERVER_URL,
                timeout=30.0,
                default_model_name="gpt-4o",
            )
        return self._memory_client

    def _setup_llm(self):
        """Set up the LLM with all memory tools."""
        # Get all available memory tool schemas
        memory_tool_schemas = MemoryAPIClient.get_all_memory_tool_schemas()

        # Extract function schemas for OpenAI
        available_functions = [tool["function"] for tool in memory_tool_schemas]

        logger.info(
            f"Available memory tools: {[func['name'] for func in available_functions]}"
        )

        # Set up LLM with function calling - force tool usage more aggressively
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3).bind_tools(
            memory_tool_schemas,  # Use full tool schemas, not just functions
            tool_choice="auto",  # Let the model choose when to use tools
        )

    async def cleanup(self):
        """Clean up resources."""
        if self._memory_client:
            await self._memory_client.close()

    async def _add_message_to_working_memory(
        self, session_id: str, user_id: str, role: str, content: str
    ) -> None:
        """Add a message to working memory."""
        client = await self.get_client()
        await client.append_messages_to_working_memory(
            session_id=session_id,
            messages=[{"role": role, "content": content}],
            namespace=self._get_namespace(user_id),
            user_id=user_id,
        )

    async def _handle_multiple_function_calls(
        self,
        tool_calls: list,
        context_messages: list,
        session_id: str,
        user_id: str,
    ) -> str:
        """Handle multiple function calls sequentially."""
        client = await self.get_client()

        all_results = []
        successful_calls = []

        print(f"🔧 Processing {len(tool_calls)} tool calls...")

        # Execute all tool calls
        for i, tool_call in enumerate(tool_calls):
            function_name = tool_call.get("name", "unknown")
            print(f"🔧 Using {function_name} tool ({i + 1}/{len(tool_calls)})...")

            # Use the client's unified tool call resolver
            result = await client.resolve_tool_call(
                tool_call=tool_call,
                session_id=session_id,
                namespace=self._get_namespace(user_id),
                user_id=user_id,
            )

            all_results.append(result)

            if result["success"]:
                successful_calls.append(
                    {"name": function_name, "result": result["formatted_response"]}
                )
                print(f"   ✅ {function_name}: {result['formatted_response'][:100]}...")

                # Show memories when search_memory tool is used (print contents in demo output)
                if function_name == "search_memory" and "memories" in result.get(
                    "result", {}
                ):
                    memories = result["result"]["memories"]
                    if memories:
                        print(f"   🧠 Found {len(memories)} memories:")
                        for j, memory in enumerate(memories[:10], 1):  # Show first 10
                            memory_text = (memory.get("text", "") or "").strip()
                            topics = memory.get("topics", [])
                            score = memory.get("relevance_score")
                            mem_id = memory.get("id")
                            preview = (
                                (memory_text[:160] + "...")
                                if len(memory_text) > 160
                                else memory_text
                            )
                            print(
                                f"     [{j}] id={mem_id} :: {preview} (topics: {topics}, score: {score})"
                            )
                        if len(memories) > 10:
                            print(f"     ... and {len(memories) - 10} more memories")
                        # Duplicate check summary (by text)
                        texts = [(m.get("text", "") or "").strip() for m in memories]
                        unique_texts = {t for t in texts if t}
                        from collections import Counter as _Counter

                        c = _Counter([t for t in texts if t])
                        dup_texts = [t for t, n in c.items() if n > 1]
                        print(
                            f"   🧾 Text summary: total={len(texts)}, unique={len(unique_texts)}, duplicates={len(dup_texts)}"
                        )
                        if dup_texts:
                            sample = [
                                ((t[:80] + "...") if len(t) > 80 else t)
                                for t in dup_texts[:3]
                            ]
                            print(
                                f"   ⚠️ Duplicate texts (sample): {sample}{' ...' if len(dup_texts) > 3 else ''}"
                            )
                    else:
                        print("   🧠 No memories found for this search")
            else:
                logger.error(f"Function call failed: {result['error']}")
                print(f"   ❌ {function_name}: {result['error']}")

        # Normalize tool calls to OpenAI-style for the assistant echo message
        normalized_tool_calls: list[dict] = []
        for idx, tc in enumerate(tool_calls):
            # If already in OpenAI format, keep as-is
            if tc.get("type") == "function" and "function" in tc:
                norm = {
                    "id": tc.get("id", f"tool_call_{idx}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", tc.get("name", "")),
                        "arguments": tc.get("function", {}).get(
                            "arguments",
                            tc.get("arguments", json.dumps(tc.get("args", {}))),
                        ),
                    },
                }
            else:
                # Convert LangChain-style {name, args} or legacy {name, arguments}
                name = tc.get("name", "")
                args_value = tc.get("arguments", tc.get("args", {}))
                if not isinstance(args_value, str):
                    try:
                        args_value = json.dumps(args_value)
                    except Exception:
                        args_value = "{}"
                norm = {
                    "id": tc.get("id", f"tool_call_{idx}"),
                    "type": "function",
                    "function": {"name": name, "arguments": args_value},
                }
            normalized_tool_calls.append(norm)

        # Build assistant echo message that initiated the tool calls
        assistant_tools_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": normalized_tool_calls,
        }

        # Build per-call tool messages with proper tool_call_id threading
        tool_result_messages: list[dict] = []
        for i, (tc, res) in enumerate(
            zip(normalized_tool_calls, all_results, strict=False)
        ):
            function_name = tc.get("function", {}).get("name", "")
            if not res.get("success", False):
                logger.error(
                    f"Tool '{function_name}' failed; suppressing user-visible error. {res.get('error')}"
                )
                continue
            # Prefer structured JSON result so the model sees IDs (e.g., for edit/delete)
            result_payload = res.get("result")
            try:
                content_str = (
                    json.dumps(result_payload)
                    if isinstance(result_payload, dict | list)
                    else str(res.get("formatted_response", ""))
                )
            except Exception:
                content_str = str(res.get("formatted_response", ""))
            tool_result_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"tool_call_{i}"),
                    "name": function_name,
                    "content": content_str,
                }
            )

        # Re-invoke the same tool-enabled model with tool results so it can chain reasoning
        messages = context_messages + [assistant_tools_message] + tool_result_messages

        # Allow the model to request follow-up tool calls (e.g., edit/delete) up to 2 rounds
        max_follow_ups = 2
        rounds = 0
        final_response = self.llm.invoke(messages)
        while (
            rounds < max_follow_ups
            and hasattr(final_response, "tool_calls")
            and final_response.tool_calls
        ):
            rounds += 1
            followup_calls = final_response.tool_calls
            print(
                f"🔁 Follow-up: processing {len(followup_calls)} additional tool call(s)..."
            )

            # Resolve follow-up tool calls
            followup_results = []
            for i, tool_call in enumerate(followup_calls):
                fname = tool_call.get("name", "unknown")
                print(
                    f"   🔧 Follow-up using {fname} tool ({i + 1}/{len(followup_calls)})..."
                )
                res = await client.resolve_tool_call(
                    tool_call=tool_call,
                    session_id=session_id,
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                )
                followup_results.append(res)

            # Echo assistant tool calls and provide tool results back to the model
            normalized_followups = []
            for idx, tc in enumerate(followup_calls):
                if tc.get("type") == "function" and "function" in tc:
                    normalized_followups.append(tc)
                else:
                    name = tc.get("name", "")
                    args_value = tc.get("arguments", tc.get("args", {}))
                    if not isinstance(args_value, str):
                        try:
                            args_value = json.dumps(args_value)
                        except Exception:
                            args_value = "{}"
                    normalized_followups.append(
                        {
                            "id": tc.get("id", f"tool_call_followup_{rounds}_{idx}"),
                            "type": "function",
                            "function": {"name": name, "arguments": args_value},
                        }
                    )

            assistant_followup_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": normalized_followups,
            }
            messages.append(assistant_followup_msg)

            for i, (tc, res) in enumerate(
                zip(normalized_followups, followup_results, strict=False)
            ):
                if not res.get("success", False):
                    logger.error(
                        f"Follow-up tool '{tc.get('function', {}).get('name', '')}' failed; suppressing user-visible error. {res.get('error')}"
                    )
                    continue
                result_payload = res.get("result")
                try:
                    content_str = (
                        json.dumps(result_payload)
                        if isinstance(result_payload, dict | list)
                        else str(res.get("formatted_response", ""))
                    )
                except Exception:
                    content_str = str(res.get("formatted_response", ""))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get(
                            "id", f"tool_call_followup_{rounds}_{i}"
                        ),
                        "name": tc.get("function", {}).get("name", ""),
                        "content": content_str,
                    }
                )

            final_response = self.llm.invoke(messages)

        response_content = str(final_response.content).strip()
        if not response_content:
            response_content = (
                f"I've completed {len(successful_calls)} action(s)."
                if successful_calls
                else "I attempted actions but encountered issues."
            )
        return response_content

    async def _handle_function_call(
        self,
        function_call: dict,
        context_messages: list,
        session_id: str,
        user_id: str,
    ) -> str:
        """Handle function calls using the client's unified resolver."""
        function_name = function_call["name"]
        client = await self.get_client()

        print(f"🔧 Using {function_name} tool...")

        # Use the client's unified tool call resolver
        result = await client.resolve_tool_call(
            tool_call=function_call,
            session_id=session_id,
            namespace=self._get_namespace(user_id),
            user_id=user_id,
        )

        if not result["success"]:
            logger.error(f"Function call failed: {result['error']}")
            return result["formatted_response"]

        # Show memories when search_memory tool is used
        if function_name == "search_memory" and "memories" in result.get(
            "raw_result", {}
        ):
            memories = result["raw_result"]["memories"]
            if memories:
                print(f"   🧠 Found {len(memories)} memories:")
                for i, memory in enumerate(memories[:3], 1):  # Show first 3
                    memory_text = memory.get("text", "")[:80]
                    topics = memory.get("topics", [])
                    print(f"     [{i}] {memory_text}... (topics: {topics})")
                if len(memories) > 3:
                    print(f"     ... and {len(memories) - 3} more memories")
            else:
                print("   🧠 No memories found for this search")

        # Generate a follow-up response with the function result
        follow_up_messages = context_messages + [
            {
                "role": "assistant",
                "content": f"Let me {function_name.replace('_', ' ')}...",
            },
            {
                "role": "function",
                "name": function_name,
                "content": result["formatted_response"],
            },
            {
                "role": "user",
                "content": "Please provide a helpful response based on this information.",
            },
        ]

        final_response = self.llm.invoke(follow_up_messages)
        return str(final_response.content)

    async def _generate_response(
        self, session_id: str, user_id: str, user_input: str
    ) -> str:
        """Generate a response using the LLM with conversation context."""
        # Get working memory for context
        client = await self.get_client()
        working_memory = await client.get_working_memory(
            session_id=session_id,
            namespace=self._get_namespace(user_id),
            model_name="gpt-4o-mini",
            user_id=user_id,
        )

        context_messages = working_memory.messages

        # Convert MemoryMessage objects to dict format for LLM
        context_messages_dicts = []
        for msg in context_messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                # MemoryMessage object - convert to dict
                msg_dict = {"role": msg.role, "content": msg.content}
                context_messages_dicts.append(msg_dict)
            else:
                # Already a dict
                context_messages_dicts.append(msg)

        # Ensure system prompt is at the beginning
        context_messages_dicts = [
            msg for msg in context_messages_dicts if msg.get("role") != "system"
        ]
        context_messages_dicts.insert(0, SYSTEM_PROMPT)

        try:
            response = self.llm.invoke(context_messages_dicts)

            # Handle tool calls (modern format)
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Process ALL tool calls, not just the first one
                return await self._handle_multiple_function_calls(
                    response.tool_calls,
                    context_messages_dicts,
                    session_id,
                    user_id,
                )

            # Handle legacy function calls
            if (
                hasattr(response, "additional_kwargs")
                and "function_call" in response.additional_kwargs
            ):
                return await self._handle_function_call(
                    response.additional_kwargs["function_call"],
                    context_messages_dicts,
                    session_id,
                    user_id,
                )

            response_content = str(response.content).strip()
            # Ensure we have a non-empty response
            if not response_content:
                response_content = (
                    "I'm sorry, I encountered an error processing your request."
                )
            return response_content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def process_user_input(
        self, user_input: str, session_id: str, user_id: str
    ) -> str:
        """Process user input and return assistant response."""
        try:
            # Add user message to working memory
            await self._add_message_to_working_memory(
                session_id, user_id, "user", user_input
            )

            # Generate response
            response = await self._generate_response(session_id, user_id, user_input)

            # Add assistant response to working memory
            await self._add_message_to_working_memory(
                session_id, user_id, "assistant", response
            )

            return response

        except Exception as e:
            logger.exception(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def run_demo_conversation(
        self, session_id: str = "memory_editing_demo", user_id: str = DEFAULT_USER
    ):
        """Run a demonstration conversation showing memory editing capabilities."""
        print("🧠 Memory Editing Agent Demo")
        print("=" * 50)
        print(
            "This demo shows how the agent manages and edits memories through conversation."
        )
        print(
            "Watch for 🧠 indicators showing retrieved memories from the agent's tools."
        )
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print()

        # Demo conversation scenarios
        demo_inputs = [
            "Hi! I'm Alice. I love coffee and I work as a software engineer at Google.",
            "Actually, I need to correct something - I work at Microsoft, not Google.",
            "Oh, and I just got promoted to Senior Software Engineer last week!",
            "I forgot to mention, I moved to Seattle last month and I actually prefer tea over coffee now.",
            "Can you tell me what you remember about me?",
            "I want to update my job information - I just started as a Principal Engineer.",
            "Can you show me the specific memory about my job and then delete the old Google one if it still exists?",
        ]

        try:
            for user_input in demo_inputs:
                print(f"👤 User: {user_input}")
                print("🤔 Assistant is thinking...")

                response = await self.process_user_input(
                    user_input, session_id, user_id
                )
                print(f"🤖 Assistant: {response}")
                print("-" * 70)
                print()

                # Add a small delay for better demo flow
                await asyncio.sleep(1)

        finally:
            await self.cleanup()

    async def run_interactive(
        self, session_id: str = "memory_editing_session", user_id: str = DEFAULT_USER
    ):
        """Run interactive session with the memory editing agent."""
        print("🧠 Memory Editing Agent - Interactive Mode")
        print("=" * 50)
        print("I can help you manage your personal information through conversation.")
        print("Try things like:")
        print("- 'I love pizza and work as a teacher'")
        print("- 'Actually, I work as a professor, not a teacher'")
        print("- 'What do you remember about me?'")
        print("- 'Delete the old information about my job'")
        print()
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print("Type 'exit' to quit")
        print()

        try:
            while True:
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("👋 Thanks for trying the Memory Editing Agent!")
                    break

                print("🤔 Thinking...")
                response = await self.process_user_input(
                    user_input, session_id, user_id
                )
                print(f"🤖 Assistant: {response}")
                print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        finally:
            await self.cleanup()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Memory Editing Agent Example")
    parser.add_argument("--user-id", default=DEFAULT_USER, help="User ID")
    parser.add_argument(
        "--session-id", default="demo_memory_editing", help="Session ID"
    )
    parser.add_argument(
        "--memory-server-url", default="http://localhost:8000", help="Memory server URL"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run automated demo conversation"
    )

    args = parser.parse_args()

    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        return

    # Set memory server URL from argument if provided
    if args.memory_server_url:
        os.environ["MEMORY_SERVER_URL"] = args.memory_server_url

    try:
        agent = MemoryEditingAgent()

        if args.demo:
            # Run automated demo
            asyncio.run(
                agent.run_demo_conversation(
                    session_id=args.session_id, user_id=args.user_id
                )
            )
        else:
            # Run interactive session
            asyncio.run(
                agent.run_interactive(session_id=args.session_id, user_id=args.user_id)
            )

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Error running memory editing agent: {e}")
        raise


if __name__ == "__main__":
    main()
