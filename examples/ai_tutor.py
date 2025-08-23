#!/usr/bin/env python3
"""
AI Tutor / Learning Coach (Functional Demo)

Demonstrates a working tutor that:
- Runs short quizzes by topic
- Stores quiz results as EPISODIC memories with event_date and topics
- Tracks weak concepts as SEMANTIC memories
- Suggests what to practice next based on recent performance
- Provides a recent summary

Two modes:
- Interactive (default): REPL commands
- Demo (--demo): runs a mini sequence across topics and shows suggestions/summary

Environment variables:
- MEMORY_SERVER_URL (default: http://localhost:8000)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from agent_memory_client import MemoryAPIClient, create_memory_client
from agent_memory_client.filters import CreatedAt, MemoryType, Namespace, Topics
from agent_memory_client.models import ClientMemoryRecord, MemoryTypeEnum
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()


DEFAULT_USER = "student"
DEFAULT_SESSION = "tutor_session"
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")


def _namespace(user_id: str) -> str:
    return f"ai_tutor:{user_id}"


async def _get_client() -> MemoryAPIClient:
    return await create_memory_client(base_url=MEMORY_SERVER_URL, timeout=30.0)


def _get_llm() -> ChatOpenAI | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return ChatOpenAI(model="gpt-4o", temperature=0)


GENERATE_QUESTIONS_FN = {
    "name": "generate_quiz",
    "description": "Generate a short quiz for a topic.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "answer": {"type": "string"},
                        "concept": {"type": "string"},
                    },
                    "required": ["prompt", "answer", "concept"],
                },
            }
        },
        "required": ["questions"],
    },
}


GRADE_ANSWER_FN = {
    "name": "grade_answer",
    "description": "Grade a student's answer and provide a brief feedback.",
    "parameters": {
        "type": "object",
        "properties": {
            "correct": {"type": "boolean"},
            "feedback": {"type": "string"},
            "concept": {"type": "string"},
        },
        "required": ["correct", "feedback"],
    },
}


# Quiz generation prompt (agent-first)
QUIZ_GENERATION_SYSTEM_PROMPT = (
    "You are a helpful tutoring agent that designs short, focused quizzes. "
    "Always respond via the generate_quiz tool call with a JSON object that contains a 'questions' array. "
    "Each question item must have: 'prompt' (concise, clear), 'answer' (the expected correct answer), and 'concept' (a short tag). "
    "Guidelines: \n"
    "- Keep prompts 1-2 sentences max.\n"
    "- Prefer single-word/phrase or numeric answers when possible.\n"
    "- Cover diverse sub-concepts of the topic.\n"
    "- Avoid trick questions or ambiguity.\n"
    "- Use the requested difficulty to adjust complexity and vocabulary.\n"
)


def _create_agent_executor(user_id: str) -> AgentExecutor | None:
    """Create an AgentExecutor wired to our server tools, with user_id injected."""
    llm = _get_llm()
    if not llm:
        return None

    @tool(
        "store_quiz_result",
        description="Store a quiz result as an episodic memory for the current user.",
    )
    async def store_quiz_result_tool(topic: str, concept: str, correct: bool) -> str:
        await _tool_store_quiz_result(
            user_id=user_id, topic=topic, concept=concept, correct=correct
        )
        return "ok"

    @tool(
        "search_quiz_results",
        description="Return recent episodic quiz results as JSON for the current user.",
    )
    async def search_quiz_results_tool(since_days: int = 7) -> str:
        results = await _tool_search_quiz_results(
            user_id=user_id, since_days=since_days
        )
        return json.dumps(results)

    @tool(
        "generate_quiz",
        description="Generate a quiz (JSON array of {prompt, answer, concept}) for a topic and difficulty.",
    )
    async def generate_quiz_tool(
        topic: str, num_questions: int = 4, difficulty: str = "mixed"
    ) -> str:
        questions = await _generate_quiz(
            llm, topic=topic, num_questions=num_questions, difficulty=difficulty
        )
        return json.dumps(
            [
                {"prompt": q.prompt, "answer": q.answer, "concept": q.concept}
                for q in questions
            ]
        )

    @tool(
        "grade_answer",
        description="Grade a student's answer; return JSON {correct: bool, feedback: string}.",
    )
    async def grade_answer_tool(prompt: str, expected: str, student: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Return ONLY a JSON object with keys: correct (boolean), feedback (string)."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Grade the student's answer. Provide brief helpful feedback.\n"
                    f"prompt: {json.dumps(prompt)}\n"
                    f"expected: {json.dumps(expected)}\n"
                    f"student: {json.dumps(student)}"
                ),
            },
        ]
        try:
            resp = llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else ""
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("not dict")
            # Ensure keys present
            result = {
                "correct": bool(data.get("correct", False)),
                "feedback": str(data.get("feedback", "")).strip(),
            }
        except Exception:
            # Fallback: strict match check
            result = {
                "correct": (student or "").strip().lower()
                == (expected or "").strip().lower(),
                "feedback": "",
            }
        return json.dumps(result)

    tools = [
        store_quiz_result_tool,
        search_quiz_results_tool,
        generate_quiz_tool,
        grade_answer_tool,
    ]

    system_prompt = (
        "You are a tutoring agent. Use tools for storing quiz results and listing recent quiz events. "
        "When summarizing, always include dates from event_date in '<Mon DD, YYYY>' format."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


async def _generate_quiz(
    llm: ChatOpenAI, topic: str, num_questions: int, difficulty: str
) -> list[Question]:
    # Keep as a utility for the generate_quiz tool; expect JSON array from model
    messages = [
        {
            "role": "system",
            "content": "Return ONLY a JSON array of objects with keys: prompt, answer, concept.",
        },
        {
            "role": "user",
            "content": (
                f"Create a {num_questions}-question quiz on topic '{topic}' at {difficulty} difficulty."
            ),
        },
    ]
    resp = llm.invoke(messages)
    content = resp.content if isinstance(resp.content, str) else ""
    try:
        arr = json.loads(content or "[]")
        if not isinstance(arr, list):
            arr = []
    except Exception:
        arr = []
    cleaned: list[Question] = []
    for q in arr:
        prompt = (q.get("prompt", "") or "").strip()
        answer = (q.get("answer", "") or "").strip()
        concept = (q.get("concept", topic) or topic).strip()
        if prompt and answer:
            cleaned.append(Question(prompt=prompt, answer=answer, concept=concept))
    return cleaned[:num_questions]


# Grading handled via agent tool; removed direct model parsing


def _as_tools(*functions: dict) -> list[dict]:
    """Wrap function schemas for OpenAI tool calling."""
    return [{"type": "function", "function": fn} for fn in functions]


def _llm_bind_tools(*functions: dict) -> Any:
    llm = _get_llm()
    if not llm:
        return None
    return llm.bind_tools(_as_tools(*functions))


# Agent tools for memory operations
STORE_QUIZ_RESULT_TOOL = {
    "name": "store_quiz_result",
    "description": (
        "Store a quiz result as an episodic memory with event_date set to now. "
        "Topics must include ['quiz', <topic>, <concept>, 'correct'|'incorrect'] to avoid parsing text later."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "concept": {"type": "string"},
            "correct": {"type": "boolean"},
        },
        "required": ["topic", "concept", "correct"],
    },
}

SEARCH_QUIZ_RESULTS_TOOL = {
    "name": "search_quiz_results",
    "description": (
        "Search recent episodic quiz results for a user within N days and return JSON array of entries "
        "with fields: topic, concept, correct (bool), event_date (ISO), text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "since_days": {"type": "integer", "minimum": 1, "default": 7},
        },
        "required": [],
    },
}

SEARCH_WEAK_CONCEPTS_TOOL = (
    None  # Deprecated in favor of LLM deriving concepts from raw search
)


async def _tool_store_quiz_result(
    user_id: str, topic: str, concept: str, correct: bool
) -> dict:
    client = await _get_client()
    ns = _namespace(user_id)
    tags = ["quiz", topic, concept, "correct" if correct else "incorrect"]
    record = ClientMemoryRecord(
        text=f"Quiz result: topic={topic}, concept={concept}, correct={correct}",
        memory_type=MemoryTypeEnum.EPISODIC,
        topics=tags,
        namespace=ns,
        user_id=user_id,
        event_date=datetime.now(UTC),
    )
    await client.create_long_term_memory([record])
    return {"status": "ok"}


async def _tool_search_quiz_results(user_id: str, since_days: int = 7) -> list[dict]:
    client = await _get_client()
    ns = _namespace(user_id)
    results = await client.search_long_term_memory(
        text="quiz results",
        namespace=Namespace(eq=ns),
        topics=Topics(any=["quiz"]),
        memory_type=MemoryType(eq="episodic"),
        created_at=CreatedAt(gte=(datetime.now(UTC) - timedelta(days=since_days))),
        limit=100,
    )
    formatted: list[dict] = []
    for m in results.memories:
        event_date = getattr(m, "event_date", None)
        event_iso = None
        if isinstance(event_date, datetime):
            try:
                event_iso = event_date.isoformat()
            except Exception:
                event_iso = None
        formatted.append(
            {
                "id": getattr(m, "id", None),
                "text": getattr(m, "text", None),
                "topics": list(getattr(m, "topics", []) or []),
                "entities": list(getattr(m, "entities", []) or []),
                "event_date": event_iso,
            }
        )
    return formatted


@dataclass
class Question:
    prompt: str
    answer: str
    concept: str


QUIZZES: dict[str, list[Question]] = {
    "algebra": [
        Question("Solve: 2x + 3 = 9. x = ?", "3", "linear_equations"),
        Question("What is the slope in y = 5x + 1?", "5", "slope"),
    ],
    "geometry": [
        Question("Sum of interior angles in a triangle?", "180", "triangles"),
        Question("Area of a circle with r=3?", "28.27", "circle_area"),
    ],
}


async def record_quiz_result(
    user_id: str, topic: str, concept: str, correct: bool
) -> None:
    # Direct tool call is deterministic; no need to route through the agent
    await _tool_store_quiz_result(
        user_id=user_id, topic=topic, concept=concept, correct=correct
    )


async def get_weak_concepts(user_id: str, since_days: int = 30) -> list[str]:
    executor = _create_agent_executor(user_id)
    if not executor:
        raise RuntimeError("OPENAI_API_KEY required for agent operations")
    res = await executor.ainvoke(
        {
            "input": (
                f"Use search_quiz_results(since_days={since_days}) and return ONLY a JSON array of weak concepts (strings) "
                "by selecting entries that were answered incorrectly."
            )
        }
    )
    content = res.get("output", "") if isinstance(res, dict) else ""
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    # Fallback to line parsing if model responds textually
    return [line.strip("- ") for line in (content or "").splitlines() if line.strip()]


async def practice_next(user_id: str) -> str:
    concepts = await get_weak_concepts(user_id, since_days=30)
    if not concepts:
        return "You're doing great! No weak concepts detected recently."
    return f"Focus next on: {', '.join(concepts[:3])}"


async def recent_summary(user_id: str, since_days: int = 7) -> list[str]:
    executor = _create_agent_executor(user_id)
    if not executor:
        raise RuntimeError("OPENAI_API_KEY required for agent operations")
    res = await executor.ainvoke(
        {
            "input": (
                f"Call search_quiz_results(since_days={since_days}) and produce a summary where each line is in the format "
                "'<Mon DD, YYYY> — <Topic Title Case> / <concept>: <correct|incorrect>' and always include the date."
            )
        }
    )
    content = res.get("output", "") if isinstance(res, dict) else ""
    return [line for line in (content or "").splitlines() if line.strip()]


async def run_quiz(
    user_id: str, topic: str, *, num_questions: int = 4, difficulty: str = "mixed"
) -> None:
    questions: list[Question] | None = None
    llm = _llm_bind_tools(GENERATE_QUESTIONS_FN, GRADE_ANSWER_FN)
    executor = _create_agent_executor(user_id)
    if executor:
        res = await executor.ainvoke(
            {
                "input": (
                    f"Generate a {num_questions}-question quiz on topic '{topic}' at {difficulty} "
                    "difficulty using the generate_quiz tool. Return ONLY a JSON array of {prompt, answer, concept}."
                )
            }
        )
        content = res.get("output", "") if isinstance(res, dict) else ""
        try:
            arr = json.loads(content)
        except Exception:
            arr = []
        if not isinstance(arr, list):
            arr = []
        if isinstance(arr, list):
            cleaned: list[Question] = []
            for q in arr:
                prompt = (q.get("prompt", "") or "").strip()
                answer = (q.get("answer", "") or "").strip()
                concept = (q.get("concept", topic) or topic).strip()
                if prompt and answer:
                    cleaned.append(
                        Question(prompt=prompt, answer=answer, concept=concept)
                    )
            questions = cleaned[:num_questions]

    if not questions:
        print("Could not generate a quiz. Try a different topic or difficulty.")
        return
    correct_count = 0
    total = len(questions)
    for q in questions:
        print(q.prompt)
        ans = input("Your answer: ").strip()
        correct = _normalize(ans) == _normalize(q.answer)
        graded_feedback = None
        if llm:
            # Agent-based grading via tool
            executor = _create_agent_executor(user_id)
            if executor:
                res = await executor.ainvoke(
                    {
                        "input": (
                            "Use grade_answer(prompt=..., expected=..., student=...) and return ONLY JSON {correct, feedback}. "
                            f"prompt={json.dumps(q.prompt)}, expected={json.dumps(q.answer)}, student={json.dumps(ans)}"
                        )
                    }
                )
                try:
                    payload = res.get("output", "") if isinstance(res, dict) else ""
                    data = json.loads(payload)
                    if isinstance(data, dict):
                        graded_feedback = data.get("feedback")
                        if "correct" in data:
                            correct = bool(data.get("correct"))
                except Exception:
                    pass
        print("Correct!" if correct else f"Incorrect. Expected {q.answer}")
        if graded_feedback:
            print(f"Feedback: {graded_feedback}")
        await record_quiz_result(user_id, topic, q.concept, correct)
        if correct:
            correct_count += 1
    print(f"Score: {correct_count}/{total}")


def _normalize(s: str) -> str:
    return s.strip().lower()


async def run_demo(user_id: str, session_id: str) -> None:
    print("🎓 AI Tutor Demo (LLM-generated)")
    llm = _llm_bind_tools(GENERATE_QUESTIONS_FN, GRADE_ANSWER_FN)
    if not llm:
        print("OPENAI_API_KEY required for demo.")
        return

    # Single demo quiz
    topic = "algebra"
    num_questions = 4
    difficulty = "mixed"

    # Generate quiz via agent tool (executor)
    executor = _create_agent_executor(user_id)
    questions: list[Question] = []
    if executor:
        res = await executor.ainvoke(
            {
                "input": (
                    f"Use generate_quiz(topic='{topic}', num_questions={num_questions}, difficulty='{difficulty}') "
                    "and return ONLY a JSON array of {prompt, answer, concept}."
                )
            }
        )
        content = res.get("output", "") if isinstance(res, dict) else ""
        try:
            arr = json.loads(content)
            if isinstance(arr, list):
                for q in arr:
                    prompt = (q.get("prompt", "") or "").strip()
                    answer = (q.get("answer", "") or "").strip()
                    concept = (q.get("concept", topic) or topic).strip()
                    if prompt and answer:
                        questions.append(
                            Question(prompt=prompt, answer=answer, concept=concept)
                        )
                questions = questions[:num_questions]
        except Exception:
            questions = []
    if not questions:
        print(f"Could not generate quiz for topic '{topic}'.")
        return

    # Generate student answers via separate LLM call (no tools)
    base_llm = _get_llm()
    if not base_llm:
        print("OPENAI_API_KEY required for demo.")
        return
    answers_system = {
        "role": "system",
        "content": (
            "You are a diligent student. Provide concise answers to the following questions. "
            "Return ONLY a JSON array of strings, one answer per question, in order; no extra text."
        ),
    }
    q_lines = "\n".join([f"{i + 1}. {q.prompt}" for i, q in enumerate(questions)])
    answers_user = {"role": "user", "content": f"Questions:\n{q_lines}\n"}
    ans_resp = base_llm.invoke([answers_system, answers_user])
    ans_content = ans_resp.content if isinstance(ans_resp.content, str) else ""
    try:
        answers = json.loads(ans_content or "[]")
        if not isinstance(answers, list):
            answers = []
        answers = [str(a) for a in answers]
    except Exception:
        answers = []
    if len(answers) < len(questions):
        answers.extend([""] * (len(questions) - len(answers)))
    answers = answers[: len(questions)]

    print(f"\nTopic: {topic}")
    correct_count = 0
    for i, q in enumerate(questions):
        student_answer = answers[i]
        executor = _create_agent_executor(user_id)
        is_correct = _normalize(student_answer) == _normalize(q.answer)
        feedback = None
        if executor:
            res_g = await executor.ainvoke(
                {
                    "input": (
                        "Use grade_answer(prompt=..., expected=..., student=...) and return ONLY JSON {correct, feedback}. "
                        f"prompt={json.dumps(q.prompt)}, expected={json.dumps(q.answer)}, student={json.dumps(student_answer)}"
                    )
                }
            )
            try:
                payload = res_g.get("output", "") if isinstance(res_g, dict) else ""
                data = json.loads(payload)
                if isinstance(data, dict):
                    feedback = data.get("feedback")
                    if "correct" in data:
                        is_correct = bool(data.get("correct"))
            except Exception:
                pass

        print(f"Q: {q.prompt}")
        print(f"A: {student_answer}")
        print(
            "Result: "
            + ("Correct" if is_correct else f"Incorrect (expected {q.answer})")
        )
        if feedback:
            print(f"Feedback: {feedback}")

        await record_quiz_result(user_id, topic, q.concept, is_correct)
        if is_correct:
            correct_count += 1

    print(f"Score: {correct_count}/{len(questions)}")

    print("\nWeak concepts:")
    for c in await get_weak_concepts(user_id):
        print(f"- {c}")

    print("\nPractice next:")
    print(await practice_next(user_id))

    print("\nRecent summary:")
    for line in await recent_summary(user_id):
        print(f"- {line}")


async def run_interactive(user_id: str, session_id: str) -> None:
    print("🎓 AI Tutor - Interactive Mode")
    print(
        "Commands:\n"
        "  quiz [<topic>]  # prompts for topic, count (1-25), difficulty (easy|medium|hard|mixed)\n"
        "  practice-next\n"
        "  weak-concepts\n"
        "  summary [--days N]\n"
        "  exit"
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

        parts = raw.split()
        cmd = parts[0]
        try:
            if cmd == "quiz":
                # Ask interactively for quiz parameters
                try:
                    topic = parts[1] if len(parts) > 1 else input("Topic: ").strip()
                except IndexError:
                    topic = input("Topic: ").strip()
                try:
                    n_raw = input("Number of questions (default 4, max 25): ").strip()
                    num_q = int(n_raw) if n_raw else 4
                except Exception:
                    num_q = 4
                if num_q < 1:
                    num_q = 1
                if num_q > 25:
                    num_q = 25
                difficulty = (
                    input("Difficulty (easy|medium|hard|mixed) [mixed]: ").strip()
                    or "mixed"
                )
                await run_quiz(
                    user_id, topic, num_questions=num_q, difficulty=difficulty
                )
            elif cmd == "practice-next":
                print(await practice_next(user_id))
            elif cmd == "weak-concepts":
                for c in await get_weak_concepts(user_id):
                    print(f"- {c}")
            elif cmd == "summary":
                days = 7
                if "--days" in parts:
                    i = parts.index("--days")
                    if i + 1 < len(parts):
                        days = int(parts[i + 1])
                for line in await recent_summary(user_id, days):
                    print(f"- {line}")
            else:
                print("Unknown command")
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Tutor")
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
