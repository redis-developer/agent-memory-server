#!/usr/bin/env python3
"""
Memory Prompt Agent Example

This example demonstrates how to use the memory prompt feature of the Agent Memory Server.
Instead of manually managing conversation history, this agent:

1. Stores all messages in working memory
2. For each turn, uses the memory prompt to get relevant memories
3. Combines the memory prompt results with the system prompt
4. Sends the enriched context to the LLM

The memory prompt feature automatically:
- Retrieves relevant memories based on the current input
- Formats them into a context-aware prompt
- Provides both recent conversation history and relevant long-term memories

Environment variables:
- OPENAI_API_KEY: Required for OpenAI ChatGPT
- MEMORY_SERVER_URL: Memory server URL (default: http://localhost:8000)
"""

import asyncio
import logging
import os
import textwrap
from typing import Any

from agent_memory_client import (
    MemoryAPIClient,
    create_memory_client,
)
from agent_memory_client.filters import Namespace, UserId
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

SYSTEM_PROMPT = textwrap.dedent("""
    You are a helpful and knowledgeable personal assistant. You have access to memories
    from previous conversations and can provide personalized responses based on what
    you know about the user.

    You are skilled at:
    - Remembering user preferences and past conversations
    - Providing contextual responses based on conversation history
    - Being helpful, friendly, and engaging
    - Asking relevant follow-up questions

    When you receive context from memories, use them naturally in your responses.
    If memories contain information about the user's preferences, interests, or past
    conversations, incorporate that knowledge to provide more personalized assistance.

    Be conversational and natural - don't explicitly mention that you're using memories
    unless it's relevant to the conversation.
""").strip()


class MemoryPromptAgent:
    """
    A conversational agent that demonstrates the memory prompt feature.

    This agent uses the memory prompt to automatically retrieve and format
    relevant memories for each conversation turn, providing rich context
    to the LLM without manual history management.
    """

    def __init__(self):
        self._memory_client: MemoryAPIClient | None = None
        self._setup_llm()

    def _get_namespace(self, user_id: str) -> str:
        """Generate consistent namespace for a user."""
        return f"memory_prompt_agent:{user_id}"

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
        """Set up the LLM instance."""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

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

    async def _get_memory_prompt(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        show_memories: bool = False,
    ) -> list[dict[str, Any]]:
        """Get memory prompt with relevant context for the current input."""
        client = await self.get_client()

        # Use the memory prompt feature to get relevant memories
        result = await client.memory_prompt(
            session_id=session_id,
            query=user_input,
            namespace=self._get_namespace(user_id),
            # Optional parameters to control memory retrieval
            model_name="gpt-4o-mini",  # Controls token-based truncation
            long_term_search={
                "limit": 30,
                # More permissive distance threshold (relevance ~= 1 - distance)
                # 0.7 distance ≈ 30% min relevance, suitable for generic demo queries
                "distance_threshold": 0.7,
                # Let the server optimize vague queries for better recall
                "optimize_query": True,
            },
            user_id=user_id,
        )

        # Show retrieved memories if requested
        if show_memories and "messages" in result:
            # Look for system message containing long-term memories
            for msg in result["messages"]:
                if msg.get("role") == "system":
                    content = msg.get("content", {})
                    if isinstance(content, dict):
                        text = content.get("text", "")
                    else:
                        text = str(content)

                    if "Long term memories related to" in text:
                        # Parse the memory lines
                        lines = text.split("\n")
                        memory_lines = [
                            line.strip()
                            for line in lines
                            if line.strip().startswith("- ")
                        ]

                        if memory_lines:
                            print(
                                f"🧠 Retrieved {len(memory_lines)} relevant memories:"
                            )
                            ids: list[str] = []
                            for i, memory_line in enumerate(
                                memory_lines[:5], 1
                            ):  # Show first 5
                                # Extract memory text and optional ID
                                memory_text = memory_line[2:]  # Remove "- "
                                mem_id = None
                                if "(ID:" in memory_text and ")" in memory_text:
                                    try:
                                        mem_id = (
                                            memory_text.split("(ID:", 1)[1]
                                            .split(")", 1)[0]
                                            .strip()
                                        )
                                        ids.append(mem_id)
                                    except Exception:
                                        pass
                                    memory_text = memory_text.split("(ID:")[0].strip()
                                print(f"   [{i}] id={mem_id} :: {memory_text}")
                            # Duplicate/uniqueness summary
                            unique_ids = {i for i in ids if i}
                            from collections import Counter

                            c = Counter([i for i in ids if i])
                            duplicates = [i for i, n in c.items() if n > 1]
                            print(
                                f"🧾 ID summary: total_shown={len(ids)}, unique={len(unique_ids)}, duplicates={len(duplicates)}"
                            )
                            if duplicates:
                                print(
                                    f"⚠️ Duplicate IDs among shown: {duplicates[:5]}{' ...' if len(duplicates) > 5 else ''}"
                                )
                            if len(memory_lines) > 5:
                                print(
                                    f"   ... and {len(memory_lines) - 5} more memories"
                                )
                            print()
                        else:
                            print(
                                "🧠 No relevant long-term memories found for this query"
                            )
                            print()
                        break

        return result["messages"]

    async def _generate_response(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        show_memories: bool = False,
    ) -> str:
        """Generate a response using the LLM with memory-enriched context."""
        # Get memory prompt with relevant context
        memory_messages = await self._get_memory_prompt(
            session_id, user_id, user_input, show_memories
        )

        # Add system prompt to the beginning
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add memory messages (these already include context and the user query)
        for msg in memory_messages:
            # Extract text content from the message structure
            content = msg.get("content", "")
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            messages.append({"role": msg["role"], "content": str(content)})

        # Generate response
        response = self.llm.invoke(messages)
        return str(response.content)

    async def process_user_input(
        self, user_input: str, session_id: str, user_id: str
    ) -> str:
        """Process user input and return assistant response."""
        try:
            # Add user message to working memory first
            await self._add_message_to_working_memory(
                session_id, user_id, "user", user_input
            )

            # Generate response using memory prompt (with memory visibility in demo mode)
            response = await self._generate_response(
                session_id, user_id, user_input, show_memories=True
            )

            # Add assistant response to working memory
            await self._add_message_to_working_memory(
                session_id, user_id, "assistant", response
            )

            return response

        except Exception as e:
            logger.exception(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def run_demo_conversation(
        self, session_id: str = "memory_prompt_demo", user_id: str = DEFAULT_USER
    ):
        """Run a demonstration conversation showing memory prompt capabilities."""
        print("🧠 Memory Prompt Agent Demo")
        print("=" * 50)
        print("This demo shows how the memory prompt feature automatically retrieves")
        print("relevant memories to provide contextual responses.")
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print()

        # First, we need to create some long-term memories to demonstrate the feature
        print("🔧 Setting up demo by checking for existing background memories...")

        client = await self.get_client()

        # Check if we already have demo memories for this user
        should_create_memories = True
        try:
            existing_memories = await client.search_long_term_memory(
                text="Alice",
                namespace=Namespace(eq=self._get_namespace(user_id)),
                user_id=UserId(eq=user_id),
                limit=10,
            )

            if existing_memories and len(existing_memories.memories) >= 5:
                print("✅ Found existing background memories about Alice")
                print()
                should_create_memories = False
        except Exception:
            # Search failed, proceed with memory creation
            pass

        if should_create_memories:
            print("🔧 Creating new background memories...")
            from agent_memory_client.models import ClientMemoryRecord

            # Create some background memories that the prompt agent can use
            demo_memories = [
                ClientMemoryRecord(
                    text="User Alice loves Italian food, especially pasta and pizza",
                    memory_type="semantic",
                    topics=["food", "preferences"],
                    entities=["Alice", "Italian food", "pasta", "pizza"],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
                ClientMemoryRecord(
                    text="Alice works as a software engineer at a tech startup in San Francisco",
                    memory_type="semantic",
                    topics=["work", "job", "location"],
                    entities=[
                        "Alice",
                        "software engineer",
                        "tech startup",
                        "San Francisco",
                    ],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
                ClientMemoryRecord(
                    text="Alice enjoys hiking on weekends and has climbed Mount Tamalpais several times",
                    memory_type="semantic",
                    topics=["hobbies", "outdoors", "hiking"],
                    entities=["Alice", "hiking", "weekends", "Mount Tamalpais"],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
                # This is actually an episodic memory because it has a time, right?
                ClientMemoryRecord(
                    text="Alice is planning a trip to Italy next summer to visit Rome and Florence",
                    memory_type="semantic",
                    topics=["travel", "plans", "Italy"],
                    entities=["Alice", "Italy", "Rome", "Florence", "summer"],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
                # TODO: Episodic memories require dates/times
                ClientMemoryRecord(
                    text="Alice mentioned she's learning Italian using Duolingo and taking evening classes",
                    memory_type="episodic",
                    topics=["learning", "languages", "education"],
                    entities=["Alice", "Italian", "Duolingo", "classes"],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
            ]

            await client.create_long_term_memory(demo_memories)
            print("✅ Created background memories about Alice")
            print()

        # Demo conversation scenarios that should trigger memory retrieval
        demo_inputs = [
            "I love Italian food. What's a good Italian restaurant recommendation?",
            "I'm planning a trip to Italy next summer to visit Rome and Florence. Any tips?",
            "I enjoy hiking on weekends. What should I do this weekend for some outdoor activity?",
            "I'm learning Italian. Any suggestions to speed up my progress?",
            "I'm a software engineer in San Francisco. Can you suggest some programming projects?",
            "What do you know about me from our previous conversations?",
        ]

        try:
            for user_input in demo_inputs:
                print(f"👤 User: {user_input}")
                print("🤔 Assistant is thinking... (retrieving relevant memories)")

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
        self, session_id: str = "memory_prompt_session", user_id: str = DEFAULT_USER
    ):
        """Main async interaction loop for the memory prompt agent."""
        print("🧠 Memory Prompt Agent - Interactive Mode")
        print("=" * 50)
        print("This agent uses memory prompts to provide contextual responses.")
        print("Try mentioning your preferences, interests, or past conversations!")
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print("Type 'exit' to quit")
        print()

        try:
            while True:
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("👋 Thank you for using the Memory Prompt Agent!")
                    break

                # Process input and get response
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

    def run_demo(
        self, session_id: str = "memory_prompt_demo", user_id: str = DEFAULT_USER
    ):
        """Synchronous wrapper for the async demo method."""
        asyncio.run(self.run_demo_conversation(session_id, user_id))

    def run(
        self, session_id: str = "memory_prompt_session", user_id: str = DEFAULT_USER
    ):
        """Synchronous wrapper for the async interactive method."""
        asyncio.run(self.run_interactive(session_id, user_id))


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Memory Prompt Agent Example")
    parser.add_argument("--user-id", default=DEFAULT_USER, help="User ID")
    parser.add_argument(
        "--session-id", default="demo_memory_prompt_session", help="Session ID"
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
        agent = MemoryPromptAgent()

        if args.demo:
            # Run automated demo
            agent.run_demo(session_id=args.session_id, user_id=args.user_id)
        else:
            # Run interactive session
            agent.run(session_id=args.session_id, user_id=args.user_id)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Error running memory prompt agent: {e}")
        raise


if __name__ == "__main__":
    main()
