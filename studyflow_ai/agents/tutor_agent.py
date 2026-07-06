"""
tutor_agent.py
--------------
Socratic Tutoring Agent for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Receive validated TutoringRequest objects from the Coordinator Agent.
  - Retrieve relevant course material via the vector_search_tool.
  - Build a grounded prompt context from retrieved chunks.
  - Generate a Socratic, step-by-step tutoring response.
  - Never answer from unsupported assumptions; explicitly state when
    insufficient course material is available.
  - Return a structured TutoringResponse with answer text and citations.
  - Never call VectorService or any repository directly.
  - Never implement retrieval or embedding logic internally.

References:
  docs/architecture/multi_agent_architecture.md     §3.4 TutorAgent
  docs/architecture/agent_responsibility_matrix.md  §TutorAgent
  docs/architecture/interface_contracts.md          §TutoringRequest / TutoringResponse
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Google ADK core
from google.adk import Agent

# Tool Layer – the only entry point this agent is permitted to use
from studyflow_ai.tools.vector_search_tool import query_academic_chunks

# Approved request / response / state contracts
from studyflow_ai.models.requests import TutoringRequest
from studyflow_ai.models.responses import TutoringResponse
from studyflow_ai.models.state import ConversationState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K: int = 5
_MIN_RELEVANCE_SCORE: float = 0.3
_NO_CONTEXT_ANSWER: str = (
    "I don't have enough course material to answer that question accurately. "
    "Please upload the relevant syllabus or lecture notes so I can help you."
)


# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    """
    Load a plain-text system prompt from the prompts/ package.

    Args:
        filename: Name of the .txt file inside studyflow_ai/prompts/.

    Returns:
        The prompt string, or a safe fallback if the file is missing.
    """
    prompt_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "prompts", filename)
    )
    try:
        with open(prompt_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        logger.warning("Prompt file '%s' not found. Using inline fallback.", filename)
        return (
            "You are the Tutoring Agent for StudyFlow AI. "
            "Help the student solve academic problems step-by-step using "
            "a Socratic method. Never give direct homework answers."
        )


# ---------------------------------------------------------------------------
# TutorAgent
# ---------------------------------------------------------------------------

class TutorAgent:
    """
    Provides grounded, Socratic academic tutoring for StudyFlow AI.

    Design contract (Architecture Freeze §3.4):
      - Stateless except for the ConversationState passed into each turn.
      - All retrieval delegated to the Tool Layer (query_academic_chunks).
      - Response generation currently uses a deterministic template; Phase 2
        will integrate Gemini reasoning for true Socratic dialogue.
      - Never fabricates academic facts; explicitly declines when context
        is insufficient.

    The underlying ADK agent is registered with query_academic_chunks so it
    can operate in multi-turn agentic mode in the future.
    """

    def __init__(self) -> None:
        self._adk_agent: Agent = Agent(
            name="TutorAgent",
            instruction=_load_prompt("tutoring.txt"),
            tools=[query_academic_chunks],
        )
        logger.info("TutorAgent initialised (ADK agent: %s).", self._adk_agent.name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def handle(
        self,
        request: TutoringRequest,
        state: ConversationState,
    ) -> TutoringResponse:
        """
        Process a single academic question.

        Workflow (per Execution Workflow doc §Ask Academic Question):
          1. Validate the incoming request.
          2. Read ConversationState for chat context.
          3. Retrieve relevant chunks via vector_search_tool.
          4. Build a grounded prompt context from retrieval results.
          5. Generate the tutoring response.
          6. Return a TutoringResponse with answer and citations.

        Args:
            request: Validated payload containing user_id, course_id,
                     question, and optional chat_history.
            state:   Mutable session state for recording this turn.

        Returns:
            TutoringResponse with the grounded answer and source citations.
        """
        logger.info(
            "Handling tutoring request | user_id=%s course_id=%s",
            request.user_id,
            request.course_id,
        )

        # Step 1 – validate inputs.
        self._validate_request(request)

        # Step 2 – read conversation context.
        history_summary = self._summarise_history(request.chat_history)

        # Step 3 – retrieve relevant academic context via the Tool Layer.
        retrieved_chunks: list[dict] = await self._retrieve_context(
            question=request.question,
            course_id=request.course_id,
        )

        # Step 4 – build grounded prompt context.
        grounded_context, citations = self._build_context(retrieved_chunks)

        # Step 5 – generate the tutoring response.
        answer: str = self._generate_answer(
            question=request.question,
            grounded_context=grounded_context,
            history_summary=history_summary,
        )

        # Record the turn in session state.
        self._append_turn(state, question=request.question, answer=answer)

        # Step 6 – return structured response.
        logger.info(
            "Tutoring response ready | user_id=%s citations=%d",
            request.user_id,
            len(citations),
        )
        return TutoringResponse(answer=answer, citations=citations)

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(request: TutoringRequest) -> None:
        """
        Validate required fields before processing.

        Args:
            request: The incoming TutoringRequest.

        Raises:
            ValueError: On empty required fields.
        """
        if not request.user_id or not request.user_id.strip():
            raise ValueError("TutoringRequest.user_id must not be empty.")

        if not request.course_id or not request.course_id.strip():
            raise ValueError("TutoringRequest.course_id must not be empty.")

        if not request.question or not request.question.strip():
            raise ValueError("TutoringRequest.question must not be empty.")

        logger.debug(
            "TutoringRequest validated for user_id=%s course_id=%s.",
            request.user_id,
            request.course_id,
        )

    async def _retrieve_context(
        self,
        question: str,
        course_id: str,
    ) -> list[dict]:
        """
        Retrieve relevant academic chunks via the vector_search_tool.

        All retrieval and embedding logic lives in the Tool Layer; this
        method is a thin orchestration wrapper.

        Args:
            question:  The student's academic question.
            course_id: Scope filter for the vector search.

        Returns:
            List of chunk dicts from the vector store (empty on failure).
        """
        logger.debug(
            "Invoking query_academic_chunks | course_id=%s top_k=%d",
            course_id,
            _DEFAULT_TOP_K,
        )
        try:
            chunks: list[dict] = await query_academic_chunks(
                query=question,
                course_id=course_id,
                top_k=_DEFAULT_TOP_K,
            )
            logger.info(
                "query_academic_chunks returned %d chunk(s) for course_id=%s.",
                len(chunks),
                course_id,
            )
            return chunks
        except Exception as exc:
            logger.error(
                "query_academic_chunks failed for course_id=%s: %s",
                course_id,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(
        chunks: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        Build a grounded text context and citation list from retrieved chunks.

        Each chunk dict is expected to have at least a ``text`` key. Optional
        ``source``, ``page``, and ``score`` keys are preserved as citation
        metadata when present.

        Chunks below _MIN_RELEVANCE_SCORE (when a score is provided) are
        discarded to avoid low-confidence grounding.

        Args:
            chunks: Raw dicts returned by query_academic_chunks.

        Returns:
            A (grounded_context_string, citations_list) tuple.
        """
        if not chunks:
            return "", []

        context_parts: list[str] = []
        citations: list[dict] = []

        for idx, chunk in enumerate(chunks, start=1):
            # Relevance gate.
            score = chunk.get("score")
            if score is not None and float(score) < _MIN_RELEVANCE_SCORE:
                continue

            text = chunk.get("text", "").strip()
            if not text:
                continue

            context_parts.append(f"[{idx}] {text}")

            citation: dict = {"chunk_index": idx, "text_preview": text[:120]}
            if chunk.get("source"):
                citation["source"] = chunk["source"]
            if chunk.get("page"):
                citation["page"] = chunk["page"]
            if score is not None:
                citation["score"] = float(score)
            citations.append(citation)

        grounded_context = "\n\n".join(context_parts)
        return grounded_context, citations

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_answer(
        question: str,
        grounded_context: str,
        history_summary: str,
    ) -> str:
        """
        Produce the tutoring answer from the grounded context.

        Phase 1 implementation:
          - If context is available, compose a structured answer referencing
            the retrieved material.
          - If no context was retrieved, return the explicit
            _NO_CONTEXT_ANSWER rather than inventing facts.

        TODO(Phase 2): Replace this template with an ADK structured-output
        call to Gemini, passing grounded_context and history_summary as
        prompt context so the LLM generates a true Socratic dialogue with
        follow-up questions, hints, and step-by-step explanations.

        Args:
            question:         The student's original question.
            grounded_context: Concatenated text from retrieved chunks.
            history_summary:  Condensed prior conversation turns.

        Returns:
            The answer string.
        """
        if not grounded_context:
            return _NO_CONTEXT_ANSWER

        # TODO(Phase 2): Replace static template with Gemini reasoning call:
        #   response = await self._adk_agent.chat(
        #       prompt=_build_socratic_prompt(question, grounded_context, history_summary),
        #       output_schema=SocraticAnswerSchema,
        #   )
        #   return response.answer

        answer_parts: list[str] = [
            f"Based on your course material, here is guidance on your question:\n",
            f"> **Your question:** {question}\n",
        ]

        if history_summary:
            answer_parts.append(
                f"_Continuing from our earlier discussion:_ {history_summary}\n"
            )

        answer_parts.append(
            "**Relevant course context:**\n\n" + grounded_context + "\n"
        )
        answer_parts.append(
            "Think about how the concepts above relate to your question. "
            "What connections can you identify?"
        )

        return "\n".join(answer_parts)

    # ------------------------------------------------------------------
    # Conversation history helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_history(chat_history: list[dict]) -> str:
        """
        Condense prior conversation turns into a short summary string.

        Only the most recent turns are included to keep prompt context
        bounded. Full history is preserved in ConversationState.

        TODO(Phase 2): Replace naive truncation with LLM-based
        summarisation for long multi-turn sessions.

        Args:
            chat_history: List of message dicts with ``role`` and ``content``.

        Returns:
            A brief summary string, or empty if no history.
        """
        if not chat_history:
            return ""

        max_recent: int = 4
        recent = chat_history[-max_recent:]

        parts: list[str] = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate individual messages to keep the summary compact.
            preview = content[:200] + "…" if len(content) > 200 else content
            parts.append(f"{role}: {preview}")

        return " | ".join(parts)

    @staticmethod
    def _append_turn(
        state: ConversationState,
        question: str,
        answer: str,
    ) -> None:
        """
        Record the Q&A turn in the session's message history.

        Args:
            state:    Mutable session state.
            question: The student's question.
            answer:   The generated tutoring answer.
        """
        state.messages.append({"role": "user", "content": question})
        state.messages.append({"role": "assistant", "content": answer})

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def adk_agent(self) -> Agent:
        """The underlying Google ADK Agent instance."""
        return self._adk_agent
