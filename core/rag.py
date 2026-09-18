"""The end-to-end question pipeline.

    question
      -> guardrails (length, prompt-injection patterns)
      -> follow-up rewriting (only when there is chat history)
      -> hybrid retrieval (optionally filtered by category)
      -> grounded answer from Claude (JSON: status, answer, citations)
      -> escalation to the right office when the answer is not in the documents
      -> log to SQLite for analytics
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from core import config, db, llm
from core.ingest import Chunk
from core.retriever import Hit, HybridRetriever

INJECTION_PATTERNS = [
    r"ignore (all|any|the|your) (previous|prior|above) (instructions|rules)",
    r"disregard (the|your) (system|previous) (prompt|instructions)",
    r"you are now",
    r"reveal (your|the) (system )?prompt",
]


@dataclass
class Result:
    status: str
    answer: str
    sources: list[Chunk] = field(default_factory=list)
    category: str = "general"
    follow_ups: list[str] = field(default_factory=list)
    escalation: dict | None = None
    search_query: str = ""
    latency_ms: int = 0
    interaction_id: int | None = None


def load_offices() -> dict:
    if config.OFFICES_PATH.exists():
        return json.loads(config.OFFICES_PATH.read_text(encoding="utf-8"))
    return {}


def check_guardrails(question: str) -> str | None:
    """Return a user-facing message if the question should not be processed."""
    if not question.strip():
        return "Please type a question."
    if len(question) > config.MAX_QUESTION_CHARS:
        return f"Please keep questions under {config.MAX_QUESTION_CHARS} characters."
    lowered = question.lower()
    if any(re.search(p, lowered) for p in INJECTION_PATTERNS):
        return "I can only help with questions about NIT Rourkela rules and student processes."
    return None


class CampusCopilot:
    def __init__(self, chunks: list[Chunk], retriever: HybridRetriever | None = None, log: bool = True):
        self.chunks = chunks
        self.retriever = retriever or HybridRetriever(chunks)
        self.offices = load_offices()
        self.log = log

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        category: str | None = None,
        session_id: str = "cli",
    ) -> Result:
        started = time.perf_counter()
        history = history or []

        blocked = check_guardrails(question)
        if blocked:
            return self._finish(Result(status="blocked", answer=blocked), question, session_id, started)

        try:
            search_query = llm.rewrite_query(question, history)
            hits: list[Hit] = self.retriever.search(search_query, k=config.TOP_K, category=category)
            if not hits:
                result = Result(status="not_found", answer="I couldn't find this in the official documents I have.",
                                search_query=search_query, category=category or "general")
            else:
                data = llm.answer(question, hits)
                cited = [hits[i - 1].chunk for i in data["citations"]]
                result = Result(
                    status=data["status"],
                    answer=data["answer"],
                    sources=cited,
                    category=data["category"],
                    follow_ups=data["follow_ups"],
                    search_query=search_query,
                )
        except Exception as exc:  # surface a friendly error, keep the details in the log
            result = Result(status="error", answer="Something went wrong while answering. Please try again.",
                            search_query=f"error: {exc}")

        if result.status == "not_found":
            result.escalation = self.offices.get(result.category) or self.offices.get("general")
        return self._finish(result, question, session_id, started)

    def _finish(self, result: Result, question: str, session_id: str, started: float) -> Result:
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        if self.log:
            try:
                result.interaction_id = db.log_interaction(
                    session_id=session_id,
                    question=question,
                    search_query=result.search_query,
                    status=result.status,
                    answer=result.answer,
                    category=result.category,
                    sources=[{"source": c.source, "page": c.page} for c in result.sources],
                    retrieval_mode=self.retriever.mode,
                    latency_ms=result.latency_ms,
                )
            except Exception:
                pass  # logging must never break the student experience
        return result
