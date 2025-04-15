import json
import logging

import tiktoken
from redis import WatchError

from agent_memory_server.config import settings
from agent_memory_server.llms import (
    AnthropicClientWrapper,
    OpenAIClientWrapper,
    get_model_config,
)
from agent_memory_server.models import MemoryMessage
from agent_memory_server.utils import Keys, get_model_client, get_redis_conn


logger = logging.getLogger(__name__)


async def _incremental_summary(
    model: str,
    client: OpenAIClientWrapper | AnthropicClientWrapper,
    context: str | None,
    messages: list[str],
) -> tuple[str, int]:
    """
    Incrementally summarize messages, building upon a previous summary.

    Args:
        model: The model to use (OpenAI or Anthropic)
        client: The client wrapper (OpenAI or Anthropic)
        context: Previous summary, if any
        messages: New messages to summarize

    Returns:
        Tuple of (summary, tokens_used)
    """
    # Reverse messages to put the most recent ones last
    messages.reverse()
    messages_joined = "\n".join(messages)
    prev_summary = context or ""

    # Prompt template for progressive summarization
    progressive_prompt = f"""
You are a precise summarization assistant. Your task is to progressively
summarize conversation history while maintaining critical context and accuracy.

INSTRUCTIONS:
1. Build upon the previous summary by incorporating new information chronologically
2. Preserve key details: names, technical terms, code references, and important decisions
3. Maintain the temporal sequence of events and discussions
4. For technical discussions, keep specific terms, versions, and implementation details
5. For code-related content, preserve function names, file paths, and important parameters
6. If the new content is irrelevant or doesn't add value, return "NONE"
7. Keep the summary concise but complete - aim for 2-3 sentences unless more detail is crucial
8. Use neutral, factual language

EXAMPLE
Current summary:
The user inquires about retirement investment options, specifically comparing
traditional IRAs and Roth IRAs. The assistant explains the key differences in
tax treatment, with traditional IRAs offering immediate tax deductions and Roth
IRAs providing tax-free withdrawals in retirement.

New lines of conversation:
Human: What factors should I consider when deciding between the two?
Assistant: Several key factors influence this decision: 1) Your current tax
bracket vs. expected retirement tax bracket, 2) Time horizon until retirement,
3) Current income and eligibility for Roth IRA contributions, and 4) Desire for
flexibility in retirement withdrawals. For example, if you expect to be in a
higher tax bracket during retirement, a Roth IRA might be more advantageous
since qualified withdrawals are tax-free. Additionally, Roth IRAs don't have
required minimum distributions (RMDs) during your lifetime, offering more
flexibility in estate planning.

New summary:
The discussion covers retirement investment options, comparing traditional and
Roth IRAs' tax implications, with traditional IRAs offering immediate deductions
and Roth IRAs providing tax-free withdrawals. The conversation expands to cover
decision factors including current vs. future tax brackets, retirement timeline,
income eligibility, and withdrawal flexibility, with specific emphasis on Roth
IRA advantages for those expecting higher retirement tax brackets and the
benefit of no required minimum distributions. END OF EXAMPLE

Current summary:
{prev_summary}

New lines of conversation:
{messages_joined}

New summary:
"""

    try:
        # Get completion from client
        response = await client.create_chat_completion(model, progressive_prompt)

        # Extract completion text
        completion = response.choices[0]["message"]["content"]

        # Get token usage
        tokens_used = response.total_tokens

        logger.info(f"Summarization complete, using {tokens_used} tokens")
        return completion, tokens_used
    except Exception as e:
        logger.error(f"Error in incremental summarization: {e}")
        raise


async def summarize_session(
    session_id: str,
    model: str,
    window_size: int,
) -> None:
    """
    Summarize messages in a session when they exceed the window size.

    This function:
    1. Gets the oldest messages up to window size and current context
    2. Generates a new summary that includes these messages
    3. Removes older, summarized messages and updates the context

    Stop summarizing

    Args:
        session_id: The session ID
        model: The model to use
        window_size: Maximum number of messages to keep
        client: The client wrapper (OpenAI or Anthropic)
        redis_conn: Redis connection
    """
    redis = get_redis_conn()
    client = await get_model_client(settings.generation_model)

    messages_key = Keys.messages_key(session_id)
    metadata_key = Keys.metadata_key(session_id)

    async with redis.pipeline(transaction=False) as pipe:
        await pipe.watch(messages_key, metadata_key)

        num_messages = await pipe.llen(messages_key)  # type: ignore
        if num_messages < window_size:
            logger.info(f"No messages to summarize for session {session_id}")
            return

        messages_raw = await pipe.lrange(messages_key, 0, window_size - 1)  # type: ignore
        metadata = await pipe.hgetall(metadata_key)  # type: ignore
        pipe.multi()

        while True:
            try:
                messages = []
                for msg_raw in messages_raw:
                    if isinstance(msg_raw, bytes):
                        msg_raw = msg_raw.decode("utf-8")
                    msg_dict = json.loads(msg_raw)
                    messages.append(MemoryMessage(**msg_dict))

                model_config = get_model_config(model)
                max_tokens = model_config.max_tokens

                # Token allocation:
                # - For small context (<10k): use 12.5% (min 512)
                # - For medium context (10k-50k): use 10% (min 1024)
                # - For large context (>50k): use 5% (min 2048)
                if max_tokens < 10000:
                    summary_max_tokens = max(512, max_tokens // 8)  # 12.5%
                elif max_tokens < 50000:
                    summary_max_tokens = max(1024, max_tokens // 10)  # 10%
                else:
                    summary_max_tokens = max(2048, max_tokens // 20)  # 5%

                # Scale buffer tokens with context size, but keep reasonable bounds
                buffer_tokens = min(max(230, max_tokens // 100), 1000)

                max_message_tokens = max_tokens - summary_max_tokens - buffer_tokens
                encoding = tiktoken.get_encoding("cl100k_base")
                total_tokens = 0
                messages_to_summarize = []

                for msg in messages:
                    msg_str = f"{msg.role}: {msg.content}"
                    msg_tokens = len(encoding.encode(msg_str))

                    # TODO: Here, we take a partial message if a single message's
                    # total size exceeds the buffer. Should this be configurable
                    # behavior?
                    if msg_tokens > max_message_tokens:
                        msg_str = msg_str[: max_message_tokens // 2]
                        msg_tokens = len(encoding.encode(msg_str))
                        total_tokens += msg_tokens

                    if total_tokens + msg_tokens <= max_message_tokens:
                        total_tokens += msg_tokens
                        messages_to_summarize.append(msg_str)

                if not messages_to_summarize:
                    logger.info(f"No messages to summarize for session {session_id}")
                    return

                context = metadata.get("context", "")

                summary, summary_tokens_used = await _incremental_summary(
                    model,
                    client,
                    context,
                    messages_to_summarize,
                )
                total_tokens += summary_tokens_used

                metadata["context"] = summary
                metadata["tokens"] = str(total_tokens)

                pipe.hmset(metadata_key, mapping=metadata)

                # Messages that were summarized
                num_summarized = len(messages_to_summarize)
                pipe.ltrim(messages_key, 0, num_summarized - 1)

                await pipe.execute()
                break
            except WatchError:
                continue

    logger.info(f"Summarization complete for session {session_id}")
