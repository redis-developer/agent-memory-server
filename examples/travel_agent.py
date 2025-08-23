#!/usr/bin/env python3
"""
Travel Agent using Agent Memory Server with Web Search

This script demonstrates how to manage both short-term and long-term agent memory
using the Agent Memory Server with optional web search capabilities. The agent manages:

1. Working memory via the memory server (session messages and data)
2. Long-term memory storage and retrieval via the memory server
3. Memory extraction and contextual retrieval
4. Conversation flow without LangGraph dependencies
5. Optional cached web search using Tavily API with Redis caching
6. Automatic discovery and use of all available memory client tools

Web search features:
- Cached web search results using Redis to avoid rate limiting
- Function calling integration with OpenAI for intelligent search decisions
- Automatic fallback when Tavily API key or Redis are not available

Environment variables:
- OPENAI_API_KEY: Required for OpenAI ChatGPT
- TAVILY_API_KEY: Optional for web search functionality
- MEMORY_SERVER_URL: Memory server URL (default: http://localhost:8000)
- REDIS_URL: Redis URL for caching (default: redis://localhost:6379)
"""

import asyncio
import json
import logging
import os

from agent_memory_client import (
    MemoryAPIClient,
    create_memory_client,
)
from agent_memory_client.filters import Namespace, UserId
from agent_memory_client.models import (
    WorkingMemory,
)
from dotenv import load_dotenv
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI
from redis import Redis


load_dotenv()


try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError as e:
    raise ImportError("Please install langchain-community for this demo.") from e


# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce third-party logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# Environment setup
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DEFAULT_USER = "demo_user"

MAX_WEB_SEARCH_RESULTS = 3

SYSTEM_PROMPT = {
    "role": "system",
    "content": """
    You are a helpful travel assistant. You can help with travel-related questions.
    You have access to conversation history and memory management tools to provide
    personalized responses.

    Available tools:

    1. **web_search** (if available): Search for current travel information, weather,
       events, or other up-to-date data when specifically needed.

    2. **Memory Management Tools** (always available):
       - **search_memory**: Look up previous conversations and stored information
       - **get_working_memory**: Check current session context
       - **add_memory_to_working_memory**: Store important preferences or information
       - **update_working_memory_data**: Save session-specific data

    **Guidelines**:
    - Answer the user's actual question first and directly
    - When someone shares information (like "I like X"), simply acknowledge it naturally - don't immediately give advice or suggestions unless they ask
    - Search memory or web when it would be helpful for the current conversation
    - Don't assume the user is actively planning a trip unless they explicitly say so
    - Be conversational and natural - respond to what the user actually says
    - When sharing memories, simply state what you remember rather than turning it into advice
    - Only offer suggestions, recommendations, or tips if the user explicitly asks for them
    - Store preferences and important details, but don't be overly eager about it
    - If someone shares a preference, respond like a friend would - acknowledge it, maybe ask a follow-up question, but don't launch into advice

    Be helpful, friendly, and responsive. Mirror their conversational style - if they're just chatting, chat back. If they ask for help, then help.
    """,
}


class CachingTavilySearchResults(TavilySearchResults):
    """
    An interface to Tavily search that caches results in Redis.

    Caching the results of the web search allows us to avoid rate limiting,
    improve latency, and reduce costs.
    """

    def __init__(self, redis_client: Redis, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis_client = redis_client

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[list[dict[str, str]] | str, dict]:
        """Use the tool."""
        cache_key = f"tavily_search:{query}"
        cached_result: str | None = self.redis_client.get(cache_key)  # type: ignore
        if cached_result:
            return json.loads(cached_result), {}
        result, raw_results = super()._run(query, run_manager)
        self.redis_client.set(cache_key, json.dumps(result), ex=60 * 60)
        return result, raw_results


class TravelAgent:
    """
    Travel Agent with comprehensive memory management capabilities.

    Uses the Agent Memory Server client with automatic discovery and integration
    of all available memory tools. Supports web search when configured.
    """

    def __init__(self):
        self._memory_client: MemoryAPIClient | None = None
        self._redis_client: Redis | None = None
        self._web_search_tool: CachingTavilySearchResults | None = None

        # Initialize LLMs and tools
        self._setup_llms()
        self._setup_tools()

    def _get_namespace(self, user_id: str) -> str:
        """Generate consistent namespace for a user."""
        return f"travel_agent:{user_id}"

    async def get_client(self) -> MemoryAPIClient:
        """Get the memory client, initializing it if needed."""
        if not self._memory_client:
            self._memory_client = await create_memory_client(
                base_url=MEMORY_SERVER_URL,
                timeout=30.0,
                default_model_name="gpt-4o",  # Configure model for auto-summarization
            )
        return self._memory_client

    def _setup_llms(self):
        """Set up the LLM instances."""
        # Define the web search tool function
        web_search_function = {
            "name": "web_search",
            "description": """
              Search the web for current information about travel destinations,
              requirements, weather, events, or any other travel-related
              queries. Use this when you need up-to-date information that may
              not be in your training data.
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant travel information",
                    }
                },
                "required": ["query"],
            },
        }

        # Set up available functions list
        available_functions = []

        # Web search (optional)
        if os.getenv("TAVILY_API_KEY"):
            available_functions.append(web_search_function)

        # Memory tools (always available) - get all available tool schemas from client
        memory_tool_schemas = MemoryAPIClient.get_all_memory_tool_schemas()

        # Extract function schemas from tool schemas
        for tool_schema in memory_tool_schemas:
            available_functions.append(tool_schema["function"])

        logger.info(
            f"Available memory tools: {[tool['function']['name'] for tool in memory_tool_schemas]}"
        )

        # Log all available tools for debugging
        all_tool_names = []
        if os.getenv("TAVILY_API_KEY"):
            all_tool_names.append("web_search")
        all_tool_names.extend(
            [tool["function"]["name"] for tool in memory_tool_schemas]
        )
        logger.info(f"Total available tools: {all_tool_names}")

        # Set up LLM with function calling
        if available_functions:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7).bind_tools(
                available_functions
            )
        else:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    def _setup_tools(self):
        """Set up tools including web search if Tavily API key is available."""
        # Set up Redis client for caching
        try:
            self._redis_client = Redis.from_url(REDIS_URL)
            self._redis_client.ping()
            logger.info("Redis client connected for caching")
        except Exception as e:
            logger.warning(f"Could not connect to Redis for caching: {e}")
            self._redis_client = None

        # Set up web search tool if Tavily API key is available
        if os.getenv("TAVILY_API_KEY") and self._redis_client:
            try:
                self._web_search_tool = CachingTavilySearchResults(
                    redis_client=self._redis_client, max_results=MAX_WEB_SEARCH_RESULTS
                )
                logger.info("Web search tool with caching enabled")
            except Exception as e:
                logger.warning(f"Could not set up web search tool: {e}")
                self._web_search_tool = None
        else:
            if not os.getenv("TAVILY_API_KEY"):
                logger.info("TAVILY_API_KEY not set, web search disabled")
            if not self._redis_client:
                logger.info("Redis not available, web search caching disabled")

    async def cleanup(self):
        """Clean up resources."""
        if self._memory_client:
            await self._memory_client.close()
            logger.info("Memory client closed")
        if self._redis_client:
            self._redis_client.close()
            logger.info("Redis client closed")

    async def _get_working_memory(self, session_id: str, user_id: str) -> WorkingMemory:
        """Get working memory for a session, creating it if it doesn't exist."""
        client = await self.get_client()
        result = await client.get_working_memory(
            session_id=session_id,
            namespace=self._get_namespace(user_id),
            model_name="gpt-4o-mini",  # Controls token-based truncation
        )
        return WorkingMemory(**result.model_dump())

    async def _search_web(self, query: str) -> str:
        """Perform a web search if the tool is available."""
        if not self._web_search_tool:
            return "Web search is not available. Please ensure TAVILY_API_KEY is set and Redis is connected."

        try:
            results, _ = self._web_search_tool._run(query)
            if isinstance(results, str):
                return results

            # Format the results
            formatted_results = []
            for result in results:
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "No URL")
                formatted_results.append(f"**{title}**\n{content}\nSource: {url}")

            return "\n\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return f"Error performing web search: {str(e)}"

    async def _add_message_to_working_memory(
        self, session_id: str, user_id: str, role: str, content: str
    ) -> WorkingMemory:
        """Add a message to working memory."""
        # Add new message
        new_message = [{"role": role, "content": content}]

        # Get the memory client and save updated working memory
        client = await self.get_client()
        await client.append_messages_to_working_memory(
            session_id=session_id,
            messages=new_message,
            namespace=self._get_namespace(user_id),
        )

    async def _handle_function_call(
        self,
        function_call: dict,
        context_messages: list,
        session_id: str,
        user_id: str,
        show_memories: bool = False,
    ) -> str:
        """Handle function calls for both web search and memory tools."""
        function_name = function_call["name"]

        # Handle web search separately (not a memory function)
        if function_name == "web_search":
            return await self._handle_web_search_call(function_call, context_messages)

        # Handle all memory functions using the client's unified resolver
        return await self._handle_memory_tool_call(
            function_call, context_messages, session_id, user_id, show_memories
        )

    async def _handle_web_search_call(
        self, function_call: dict, context_messages: list
    ) -> str:
        """Handle web search function calls."""
        print("Searching the web...")
        try:
            function_args = json.loads(function_call["arguments"])
            query = function_args.get("query", "")

            # Perform the web search
            search_results = await self._search_web(query)

            # Generate a follow-up response with the search results
            follow_up_messages = context_messages + [
                {
                    "role": "assistant",
                    "content": f"I'll search for that information: {query}",
                },
                {
                    "role": "function",
                    "name": "web_search",
                    "content": search_results,
                },
                {
                    "role": "user",
                    "content": "Please provide a helpful response based on the search results.",
                },
            ]

            final_response = self.llm.invoke(follow_up_messages)
            return str(final_response.content)

        except (json.JSONDecodeError, TypeError):
            logger.error(f"Invalid web search arguments: {function_call['arguments']}")
            return "I'm sorry, I encountered an error processing your web search request. Please try again."

    async def _handle_memory_tool_call(
        self,
        function_call: dict,
        context_messages: list,
        session_id: str,
        user_id: str,
        show_memories: bool = False,
    ) -> str:
        """Handle memory tool function calls using the client's unified resolver."""
        function_name = function_call["name"]
        client = await self.get_client()

        print("Accessing memory...")
        result = await client.resolve_tool_call(
            tool_call=function_call,  # Pass the entire function call object
            session_id=session_id,
            namespace=self._get_namespace(user_id),
        )

        if not result["success"]:
            logger.error(f"Function call failed: {result['error']}")
            return result["formatted_response"]

        # Show memories when search_memory tool is used and in demo mode
        if (
            show_memories
            and function_name == "search_memory"
            and "memories" in result.get("raw_result", {})
        ):
            memories = result["raw_result"]["memories"]
            if memories:
                print(f"🧠 Retrieved {len(memories)} memories:")
                for i, memory in enumerate(memories[:3], 1):  # Show first 3
                    memory_text = memory.get("text", "")[:80]
                    topics = memory.get("topics", [])
                    relevance = memory.get("dist", 0)
                    relevance_score = (
                        max(0, 1 - relevance) if relevance is not None else 0
                    )
                    print(
                        f"   [{i}] {memory_text}... (topics: {topics}, relevance: {relevance_score:.2f})"
                    )
                if len(memories) > 3:
                    print(f"   ... and {len(memories) - 3} more memories")
                print()
            else:
                print("🧠 No relevant memories found for this query")
                print()

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
        response_content = str(final_response.content)

        # Debug logging for empty responses
        if not response_content or not response_content.strip():
            logger.error(
                f"Empty response from LLM in memory tool call handler. Function: {function_name}"
            )
            logger.error(f"Response object: {final_response}")
            logger.error(f"Response content: '{final_response.content}'")
            logger.error(
                f"Response additional_kwargs: {getattr(final_response, 'additional_kwargs', {})}"
            )
            return "I apologize, but I couldn't generate a proper response to your request."

        return response_content

    async def _generate_response(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        show_memories: bool = False,
    ) -> str:
        """Generate a response using the LLM with conversation context."""
        # Manage conversation history
        working_memory = await self._get_working_memory(session_id, user_id)
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

        # Always ensure system prompt is at the beginning
        # Remove any existing system messages and add our current one
        context_messages = [
            msg for msg in context_messages_dicts if msg.get("role") != "system"
        ]
        context_messages.insert(0, SYSTEM_PROMPT)

        # Note: user input has already been added to working memory,
        # so we don't need to add it again here

        try:
            logger.info(f"Context messages: {context_messages}")
            response = self.llm.invoke(context_messages)

            # Handle function calls using unified approach
            if hasattr(response, "additional_kwargs"):
                # Check for OpenAI-style function_call (single call)
                if "function_call" in response.additional_kwargs:
                    return await self._handle_function_call(
                        response.additional_kwargs["function_call"],
                        context_messages,
                        session_id,
                        user_id,
                        show_memories,
                    )
                # Check for LangChain-style tool_calls (array of calls)
                if "tool_calls" in response.additional_kwargs:
                    tool_calls = response.additional_kwargs["tool_calls"]
                    if tool_calls and len(tool_calls) > 0:
                        # Process ALL tool calls, then provide JSON tool messages back to the model
                        client = await self.get_client()

                        # Normalize tool calls to OpenAI current-format
                        normalized_calls: list[dict] = []
                        for idx, tc in enumerate(tool_calls):
                            if tc.get("type") == "function" and "function" in tc:
                                normalized_calls.append(tc)
                            else:
                                name = tc.get("function", {}).get(
                                    "name", tc.get("name", "")
                                )
                                args_value = tc.get("function", {}).get(
                                    "arguments", tc.get("arguments", {})
                                )
                                if not isinstance(args_value, str):
                                    try:
                                        args_value = json.dumps(args_value)
                                    except Exception:
                                        args_value = "{}"
                                normalized_calls.append(
                                    {
                                        "id": tc.get("id", f"tool_call_{idx}"),
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": args_value,
                                        },
                                    }
                                )

                        # Resolve calls sequentially; capture results
                        results = []
                        for call in normalized_calls:
                            fname = call.get("function", {}).get("name", "")
                            try:
                                res = await client.resolve_tool_call(
                                    tool_call={
                                        "name": fname,
                                        "arguments": call.get("function", {}).get(
                                            "arguments", "{}"
                                        ),
                                    },
                                    session_id=session_id,
                                    namespace=self._get_namespace(user_id),
                                    user_id=user_id,
                                )
                            except Exception as e:
                                logger.error(f"Tool '{fname}' failed: {e}")
                                res = {"success": False, "error": str(e)}
                            results.append((call, res))

                        # Build assistant echo plus tool results as JSON content
                        assistant_tools_msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": normalized_calls,
                        }

                        tool_messages: list[dict] = []
                        for i, (tc, res) in enumerate(results):
                            if not res.get("success", False):
                                logger.error(
                                    f"Suppressing user-visible error for tool '{tc.get('function', {}).get('name', '')}': {res.get('error')}"
                                )
                                continue
                            payload = res.get("result")
                            try:
                                content = (
                                    json.dumps(payload)
                                    if isinstance(payload, dict | list)
                                    else str(res.get("formatted_response", ""))
                                )
                            except Exception:
                                content = str(res.get("formatted_response", ""))
                            tool_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id", f"tool_call_{i}"),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "content": content,
                                }
                            )

                        # Give the model one follow-up round to chain further
                        messages = (
                            context_messages + [assistant_tools_msg] + tool_messages
                        )
                        followup = self.llm.invoke(messages)
                        # Optional: one more round if tool_calls requested
                        rounds = 0
                        max_rounds = 1
                        while (
                            rounds < max_rounds
                            and hasattr(followup, "tool_calls")
                            and followup.tool_calls
                        ):
                            rounds += 1
                            follow_calls = followup.tool_calls
                            # Resolve
                            follow_results = []
                            for _j, fcall in enumerate(follow_calls):
                                fname = fcall.get("name", "")
                                try:
                                    fres = await client.resolve_tool_call(
                                        tool_call=fcall,
                                        session_id=session_id,
                                        namespace=self._get_namespace(user_id),
                                        user_id=user_id,
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Follow-up tool '{fname}' failed: {e}"
                                    )
                                    fres = {"success": False, "error": str(e)}
                                follow_results.append((fcall, fres))
                            # Echo
                            norm_follow = []
                            for idx2, fc in enumerate(follow_calls):
                                if fc.get("type") == "function" and "function" in fc:
                                    norm_follow.append(fc)
                                else:
                                    name = fc.get("name", "")
                                    args_value = fc.get("arguments", fc.get("args", {}))
                                    if not isinstance(args_value, str):
                                        try:
                                            args_value = json.dumps(args_value)
                                        except Exception:
                                            args_value = "{}"
                                    norm_follow.append(
                                        {
                                            "id": fc.get(
                                                "id", f"tool_call_follow_{idx2}"
                                            ),
                                            "type": "function",
                                            "function": {
                                                "name": name,
                                                "arguments": args_value,
                                            },
                                        }
                                    )
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": norm_follow,
                                }
                            )
                            for k, (fc, fr) in enumerate(follow_results):
                                if not fr.get("success", False):
                                    logger.error(
                                        f"Suppressing user-visible error for follow-up tool '{fc.get('name', '')}': {fr.get('error')}"
                                    )
                                    continue
                                payload = fr.get("result")
                                try:
                                    content = (
                                        json.dumps(payload)
                                        if isinstance(payload, dict | list)
                                        else str(fr.get("formatted_response", ""))
                                    )
                                except Exception:
                                    content = str(fr.get("formatted_response", ""))
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": fc.get(
                                            "id", f"tool_call_follow_{k}"
                                        ),
                                        "name": fc.get("function", {}).get(
                                            "name", fc.get("name", "")
                                        ),
                                        "content": content,
                                    }
                                )
                            followup = self.llm.invoke(messages)

                        return str(followup.content)

            response_content = str(response.content)

            # Debug logging for empty responses
            if not response_content or not response_content.strip():
                logger.error("Empty response from LLM in main response generation")
                logger.error(f"Response object: {response}")
                logger.error(f"Response content: '{response.content}'")
                logger.error(
                    f"Response additional_kwargs: {getattr(response, 'additional_kwargs', {})}"
                )
                return "I apologize, but I couldn't generate a proper response to your request."

            return response_content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def process_user_input(
        self,
        user_input: str,
        session_id: str,
        user_id: str,
        show_memories: bool = False,
    ) -> str:
        """Process user input and return assistant response."""
        try:
            # Add user message to working memory first
            await self._add_message_to_working_memory(
                session_id, user_id, "user", user_input
            )

            response = await self._generate_response(
                session_id, user_id, user_input, show_memories
            )

            # Validate response before adding to working memory
            if not response or not response.strip():
                logger.error("Generated response is empty, using fallback message")
                response = "I'm sorry, I encountered an error generating a response to your request."

            await self._add_message_to_working_memory(
                session_id, user_id, "assistant", response
            )
            return response

        except Exception as e:
            logger.exception(f"Error processing user input: {e}")
            return "I'm sorry, I encountered an error processing your request."

    async def run_async(
        self, session_id: str = "travel_session", user_id: str = DEFAULT_USER
    ):
        """Main async interaction loop for the travel agent."""
        print("Welcome to the Travel Assistant! (Type 'exit' to quit)")
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print()

        try:
            while True:
                user_input = input("\nYou (type 'quit' to quit): ")

                if not user_input.strip():
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("Thank you for using the Travel Assistant. Goodbye!")
                    break

                # Process input and get response
                response = await self.process_user_input(
                    user_input, session_id, user_id
                )
                print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
        finally:
            await self.cleanup()

    async def run_demo_conversation(
        self, session_id: str = "travel_demo", user_id: str = DEFAULT_USER
    ):
        """Run a demonstration conversation showing travel agent capabilities."""
        print("✈️ Travel Agent Demo")
        print("=" * 50)
        print(
            "This demo shows how the travel agent uses memory and web search capabilities."
        )
        print(
            "Watch for 🧠 indicators showing retrieved memories from previous conversations."
        )
        print(f"Session ID: {session_id}, User ID: {user_id}")
        print()

        # First, create some background memories for the demo
        print(
            "🔧 Setting up demo by checking for existing background travel memories..."
        )

        client = await self.get_client()

        # Check if we already have demo memories for this user
        should_create_memories = True
        try:
            existing_memories = await client.search_long_term_memory(
                text="Sarah",
                namespace=Namespace(eq=self._get_namespace(user_id)),
                user_id=UserId(eq=user_id),
                limit=10,
            )

            if existing_memories and len(existing_memories.memories) >= 5:
                print("✅ Found existing background travel memories about Sarah")
                print()
                should_create_memories = False
        except Exception:
            # Search failed, proceed with memory creation
            pass

        if should_create_memories:
            print("🔧 Creating new background travel memories...")
            from agent_memory_client.models import ClientMemoryRecord

            # Create some background travel memories
            demo_memories = [
                ClientMemoryRecord(
                    text="User Sarah loves beach destinations and prefers warm weather vacations",
                    memory_type="semantic",
                    topics=["travel", "preferences", "beaches"],
                    entities=["Sarah", "beach", "warm weather"],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
                ClientMemoryRecord(
                    text="Sarah has a budget of $3000 for her next vacation and wants to travel in summer",
                    memory_type="semantic",
                    topics=["travel", "budget", "planning"],
                    entities=["Sarah", "$3000", "summer", "vacation"],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
                ClientMemoryRecord(
                    text="Sarah visited Thailand last year and loved the food and culture there",
                    memory_type="episodic",
                    topics=["travel", "experience", "Thailand"],
                    entities=["Sarah", "Thailand", "food", "culture"],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
                ClientMemoryRecord(
                    text="Sarah is interested in learning about local customs and trying authentic cuisine when traveling",
                    memory_type="semantic",
                    topics=["travel", "culture", "food"],
                    entities=["Sarah", "local customs", "authentic cuisine"],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
                ClientMemoryRecord(
                    text="Sarah mentioned she's not a strong swimmer so prefers shallow water activities",
                    memory_type="semantic",
                    topics=["travel", "preferences", "activities"],
                    entities=["Sarah", "swimming", "shallow water"],
                    namespace=self._get_namespace(user_id),
                    user_id=user_id,
                ),
            ]

            await client.create_long_term_memory(demo_memories)
            print("✅ Created background travel memories about Sarah")
            print()

        # Demo conversation scenarios
        demo_inputs = [
            "Hi! I'm thinking about planning a vacation this summer.",
            "I'd like somewhere with beautiful beaches but not too expensive.",
            "What do you remember about my travel preferences?",
            "Can you suggest some destinations that would be good for someone like me?",
            "I'm also interested in experiencing local culture and food.",
            "What's the weather like in Bali during summer?",
        ]

        try:
            for user_input in demo_inputs:
                print(f"👤 User: {user_input}")
                print(
                    "🤔 Assistant is thinking... (checking memories and web if needed)"
                )

                response = await self.process_user_input(
                    user_input, session_id, user_id, show_memories=True
                )
                print(f"🤖 Assistant: {response}")
                print("-" * 70)
                print()

                # Add a small delay for better demo flow
                await asyncio.sleep(1)

        finally:
            await self.cleanup()

    def run(self, session_id: str = "travel_session", user_id: str = DEFAULT_USER):
        """Synchronous wrapper for the async run method."""
        asyncio.run(self.run_async(session_id, user_id))

    def run_demo(self, session_id: str = "travel_demo", user_id: str = DEFAULT_USER):
        """Synchronous wrapper for the async demo method."""
        asyncio.run(self.run_demo_conversation(session_id, user_id))


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Travel Agent with Memory Server")
    parser.add_argument("--user-id", default=DEFAULT_USER, help="User ID")
    parser.add_argument(
        "--session-id", default="demo_travel_session", help="Session ID"
    )
    parser.add_argument(
        "--memory-server-url", default="http://localhost:8000", help="Memory server URL"
    )
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis URL for caching"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run automated demo conversation"
    )

    args = parser.parse_args()

    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        return

    # Check for optional Tavily API key
    if not os.getenv("TAVILY_API_KEY"):
        print(
            "Note: TAVILY_API_KEY not set - web search functionality will be disabled"
        )
        print("To enable web search, set TAVILY_API_KEY environment variable")

    # Check for Redis connection
    redis_url = args.redis_url if hasattr(args, "redis_url") else REDIS_URL
    print(f"Using Redis at: {redis_url} for caching (if available)")

    # Set memory server URL from argument if provided
    if args.memory_server_url:
        os.environ["MEMORY_SERVER_URL"] = args.memory_server_url

    # Set Redis URL from argument if provided
    if args.redis_url:
        os.environ["REDIS_URL"] = args.redis_url

    try:
        agent = TravelAgent()

        if args.demo:
            # Run automated demo
            agent.run_demo(session_id=args.session_id, user_id=args.user_id)
        else:
            # Run interactive session
            agent.run(session_id=args.session_id, user_id=args.user_id)

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Error running travel agent: {e}")
        raise


if __name__ == "__main__":
    main()
