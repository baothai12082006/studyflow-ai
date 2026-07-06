"""
ingestion_agent.py
------------------
Ingestion & Parsing Agent for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Receive validated IngestionRequest objects from the Coordinator Agent.
  - Orchestrate the syllabus ingestion workflow across the Tool Layer.
  - Invoke parse_syllabus_document tool to extract structured course data.
  - Invoke index_academic_document tool to embed content into the vector DB.
  - Return a structured IngestionResponse to the caller.
  - Update AcademicState with extracted course information.
  - Never call StorageService or VectorService directly.
  - Never implement parsing or indexing logic internally.

References:
  docs/architecture/multi_agent_architecture.md     §3.2 IngestionAgent
  docs/architecture/agent_responsibility_matrix.md  §IngestionAgent
  docs/architecture/interface_contracts.md          §IngestionRequest / IngestionResponse
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Google ADK core
from google.adk import Agent

# Tool Layer – the only entry points this agent is permitted to use
from studyflow_ai.tools.syllabus_parser_tool import parse_syllabus_document
from studyflow_ai.tools.vector_search_tool import index_academic_document

# Approved request / response / state contracts
from studyflow_ai.models.requests import IngestionRequest
from studyflow_ai.models.responses import IngestionResponse
from studyflow_ai.models.state import AcademicState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported MIME types
# ---------------------------------------------------------------------------

SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "image/png",
        "image/jpeg",
    }
)

# Minimum non-empty text length required to attempt vector indexing
_MIN_INDEXABLE_CHARS: int = 50


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
            "You are the Ingestion & Parsing Agent for StudyFlow AI. "
            "Extract course metadata, deadlines, and grading schema from syllabus documents."
        )


# ---------------------------------------------------------------------------
# IngestionAgent
# ---------------------------------------------------------------------------

class IngestionAgent:
    """
    Orchestrates syllabus ingestion for StudyFlow AI.

    Design contract (Architecture Freeze §3.2):
      - Stateless except for AcademicState produced during ingestion.
      - All tool calls are delegated to the Tool Layer; no direct service access.
      - Returned IngestionResponse is consumed by the Coordinator Agent.

    The underlying ADK agent is registered with both tools so the LLM can
    independently decide when to call them if the agent is later used in an
    agentic (multi-turn) mode rather than a direct ``handle()`` call.
    """

    def __init__(self) -> None:
        # Initialise the ADK agent with the ingestion system prompt and tool registrations.
        self._adk_agent: Agent = Agent(
            name="IngestionAgent",
            instruction=_load_prompt("ingestion.txt"),
            tools=[parse_syllabus_document, index_academic_document],
        )
        logger.info("IngestionAgent initialised (ADK agent: %s).", self._adk_agent.name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def handle(self, request: IngestionRequest) -> IngestionResponse:
        """
        Execute the complete syllabus ingestion workflow.

        Workflow (per Execution Workflow doc §Upload Syllabus):
          1. Validate the incoming request.
          2. Parse the document via parse_syllabus_document tool.
          3. Index the extracted content via index_academic_document tool.
          4. Build and return a structured IngestionResponse.

        Args:
            request: Validated payload containing file_uri, mime_type,
                     and course_id.

        Returns:
            IngestionResponse with extraction_status, confidence_score,
            and the extracted course data as extracted_json.

        Raises:
            ValueError: If the request payload fails validation.
            RuntimeError: If the parsing step fails unrecoverably.
        """
        logger.info(
            "Starting ingestion | course_id=%s file_uri=%s mime_type=%s",
            request.course_id,
            request.file_uri,
            request.mime_type,
        )

        # Step 1 – validate inputs before touching any tool.
        self._validate_request(request)

        # Step 2 – parse the document through the Tool Layer.
        academic_state: AcademicState = await self._run_parse(request)

        # Step 3 – index the extracted content through the Tool Layer.
        indexed: bool = await self._run_index(request.course_id, academic_state)

        # Step 4 – compose the IngestionResponse for the Coordinator.
        response = self._build_response(academic_state, indexed)

        logger.info(
            "Ingestion complete | course_id=%s status=%s indexed=%s",
            request.course_id,
            response.extraction_status,
            indexed,
        )
        return response

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _validate_request(self, request: IngestionRequest) -> None:
        """
        Validate required fields and supported MIME types.

        Args:
            request: The IngestionRequest to validate.

        Raises:
            ValueError: On empty required fields or unsupported MIME type.
        """
        if not request.file_uri or not request.file_uri.strip():
            raise ValueError("IngestionRequest.file_uri must not be empty.")

        if not request.course_id or not request.course_id.strip():
            raise ValueError("IngestionRequest.course_id must not be empty.")

        if request.mime_type not in SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Unsupported MIME type '{request.mime_type}'. "
                f"Supported types: {sorted(SUPPORTED_MIME_TYPES)}"
            )

        logger.debug("IngestionRequest validated for course_id=%s.", request.course_id)

    async def _run_parse(self, request: IngestionRequest) -> AcademicState:
        """
        Invoke the syllabus_parser_tool and return the structured AcademicState.

        Delegates all parsing and OCR logic to the Tool Layer; this agent
        never inspects raw document bytes or text directly.

        Args:
            request: The validated ingestion request.

        Returns:
            AcademicState populated by the parser tool.

        Raises:
            RuntimeError: Propagated from the tool if parsing fails.
        """
        logger.debug(
            "Invoking parse_syllabus_document | file_uri=%s course_id=%s",
            request.file_uri,
            request.course_id,
        )

        # IngestionRequest does not carry user_id; derive a placeholder until
        # the Coordinator passes it through (tracked in Interface Contracts §TODO-ING-01).
        # TODO(Phase 2): Extend IngestionRequest with user_id so the tool
        # receives the correct value instead of a sentinel string.
        user_id_placeholder: str = "unknown"

        academic_state: AcademicState = await parse_syllabus_document(
            file_uri=request.file_uri,
            user_id=user_id_placeholder,
            course_id=request.course_id,
        )

        logger.info(
            "parse_syllabus_document returned %d deadline(s) for course_id=%s.",
            len(academic_state.deadlines),
            request.course_id,
        )
        return academic_state

    async def _run_index(
        self,
        course_id: str,
        academic_state: AcademicState,
    ) -> bool:
        """
        Invoke the vector_search_tool to index extractable content.

        Skips indexing if there is no usable text in the parsed state to
        prevent empty embedding calls to the vector database.

        Args:
            course_id:      The course UUID used as the vector namespace.
            academic_state: The AcademicState returned from the parse step.

        Returns:
            True if indexing succeeded or was not required; False on tool failure.
        """
        # Build indexable text chunks from the deadline titles and the course title.
        # The tool layer handles chunking strategy; we provide logical units here.
        text_chunks: list[str] = self._build_text_chunks(academic_state)

        if not text_chunks:
            logger.info(
                "No indexable content found for course_id=%s. Skipping vector indexing.",
                course_id,
            )
            return True  # Not a failure; simply nothing to index yet.

        logger.debug(
            "Invoking index_academic_document | course_id=%s chunks=%d",
            course_id,
            len(text_chunks),
        )

        indexed: bool = await index_academic_document(
            course_id=course_id,
            text_chunks=text_chunks,
        )

        if not indexed:
            logger.warning(
                "index_academic_document returned False for course_id=%s. "
                "Content was parsed but not indexed.",
                course_id,
            )
        return indexed

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response(
        academic_state: AcademicState,
        indexed: bool,
    ) -> IngestionResponse:
        """
        Construct the IngestionResponse from a completed AcademicState.

        Confidence score heuristic:
          - 1.0 if both parsed and indexed successfully.
          - 0.7 if parsed but indexing failed or was skipped.
          - 0.0 placeholder; actual confidence will be supplied by the
            parser tool in a future iteration (Interface Contracts §TODO-ING-02).

        TODO(Phase 2): Replace heuristic confidence score with the value
        returned directly by parse_syllabus_document once that tool exposes
        an extraction confidence metric.

        Args:
            academic_state: The AcademicState produced by the parse step.
            indexed:        Whether vector indexing completed successfully.

        Returns:
            A fully populated IngestionResponse.
        """
        confidence: float = 1.0 if indexed else 0.7

        extracted_json: dict = {
            "course_id": academic_state.course_id,
            "title": academic_state.title,
            "deadline_count": len(academic_state.deadlines),
            "deadlines": academic_state.deadlines,
            "indexed": indexed,
        }

        return IngestionResponse(
            extraction_status="SUCCESS",
            confidence_score=confidence,
            extracted_json=extracted_json,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_text_chunks(academic_state: AcademicState) -> list[str]:
        """
        Derive indexable text units from an AcademicState.

        Produces one chunk per deadline entry, prefixed with the course title,
        to give the vector store meaningful semantic context per item.

        Args:
            academic_state: The populated course state from the parser tool.

        Returns:
            List of non-empty strings ready for embedding. Empty list if
            there is nothing meaningful to index.
        """
        chunks: list[str] = []

        for deadline in academic_state.deadlines:
            title = deadline.get("title", "").strip()
            dtype = deadline.get("type", "").strip()
            date = deadline.get("date", "").strip()

            if not title:
                continue

            chunk = (
                f"Course: {academic_state.title} | "
                f"Type: {dtype} | "
                f"Title: {title} | "
                f"Date: {date}"
            )

            if len(chunk) >= _MIN_INDEXABLE_CHARS:
                chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def adk_agent(self) -> Agent:
        """The underlying Google ADK Agent instance."""
        return self._adk_agent
