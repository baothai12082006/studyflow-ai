"""
coordinator_agent.py
--------------------
Entry-point agent for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Receive every user request from the client layer.
  - Maintain session-scoped ConversationState.
  - Classify user intent (UPLOAD | PLAN | QNA | PROGRESS).
  - Delegate to the appropriate sub-agent via dependency injection.
  - Never call external services or tools directly.
  - Never contain business logic.

Sub-agent routing will be wired in Phase 2 where each TODO is marked.

References:
  docs/architecture/multi_agent_architecture.md
  docs/architecture/agent_responsibility_matrix.md
  docs/architecture/interface_contracts.md
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Google ADK core
from google.adk import Agent

# Approved request / response / state contracts
from studyflow_ai.models.requests import CoordinatorRequest
from studyflow_ai.models.responses import CoordinatorResponse
from studyflow_ai.models.state import ConversationState

logger = logging.getLogger(__name__)


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
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", filename
    )
    try:
        with open(os.path.normpath(prompt_path), "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        logger.warning(
            "Prompt file '%s' not found. Using inline fallback.", filename
        )
        return (
            "You are the Coordinator Agent for StudyFlow AI. "
            "Classify user intent and delegate to the correct sub-agent."
        )


# ---------------------------------------------------------------------------
# CoordinatorAgent
# ---------------------------------------------------------------------------

class CoordinatorAgent:
    """
    Orchestrates all StudyFlow AI interactions.

    Design contract (Architecture Freeze §2.1):
      - Stateless except for the session-scoped ConversationState.
      - All sub-agents are injected; none are instantiated internally.
      - No direct calls to external APIs or database layers.

    Args:
        ingestion_agent:  Sub-agent responsible for parsing uploaded syllabi.
        planning_agent:   Sub-agent responsible for building study schedules.
        tutoring_agent:   Sub-agent responsible for Socratic Q&A sessions.
        monitoring_agent: Sub-agent responsible for progress tracking and alerts.
    """

    # Intent labels produced by _classify_intent()
    INTENT_UPLOAD: str = "UPLOAD"
    INTENT_PLAN: str = "PLAN"
    INTENT_QNA: str = "QNA"
    INTENT_PROGRESS: str = "PROGRESS"

    def __init__(
        self,
        ingestion_agent: Optional[Agent] = None,
        planning_agent: Optional[Agent] = None,
        tutoring_agent: Optional[Agent] = None,
        monitoring_agent: Optional[Agent] = None,
    ) -> None:
        # Store injected sub-agents; None is acceptable during scaffold phase.
        self._ingestion_agent = ingestion_agent
        self._planning_agent = planning_agent
        self._tutoring_agent = tutoring_agent
        self._monitoring_agent = monitoring_agent

        # Initialise the underlying ADK agent with the coordinator system prompt.
        self._adk_agent = Agent(
            name="CoordinatorAgent",
            instruction=_load_prompt("coordinator.txt"),
            # TODO(Phase 2): register sub_agents=[...] once each is implemented.
        )

        logger.info("CoordinatorAgent initialised (ADK agent: %s).", self._adk_agent.name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def handle(
        self,
        request: CoordinatorRequest,
        state: ConversationState,
    ) -> CoordinatorResponse:
        """
        Process a single user turn.

        Steps:
          1. Append the user message to the session state.
          2. Classify the intent.
          3. Delegate to the appropriate sub-agent.
          4. Append the assistant reply to session state.
          5. Return a CoordinatorResponse to the client layer.

        Args:
            request: Validated payload from the client (user_id, session_id,
                     prompt, optional attachments).
            state:   Mutable session state scoped to this conversation.

        Returns:
            CoordinatorResponse with status, display_text, and optional
            ui_action hint for the frontend.
        """
        logger.info(
            "Handling request | user=%s session=%s",
            request.user_id,
            request.session_id,
        )

        # 1. Record incoming user turn in session state.
        self._append_message(state, role="user", content=request.prompt)

        # 2. Classify intent.
        intent = self._classify_intent(request)
        state.active_intent = intent
        logger.debug("Resolved intent: %s", intent)

        # 3. Delegate to sub-agent.
        reply_text, ui_action = await self._delegate(intent, request, state)

        # 4. Record assistant reply in session state.
        self._append_message(state, role="assistant", content=reply_text)

        # 5. Return structured response.
        return CoordinatorResponse(
            status="SUCCESS",
            display_text=reply_text,
            ui_action=ui_action,
        )

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, request: CoordinatorRequest) -> str:
        """
        Resolve user intent from the prompt text and presence of attachments.

        Classification order (highest to lowest priority):
          UPLOAD   → attachment present, or explicit upload/syllabus keywords.
          PLAN     → scheduling / calendar / study plan keywords.
          PROGRESS → streak / progress / milestone keywords.
          QNA      → default fall-through to Tutoring Agent.

        TODO(Phase 2): Replace keyword heuristics with a structured ADK
        output call (e.g. self._adk_agent.chat(prompt, output_schema=IntentSchema))
        so the LLM performs intent resolution instead of keyword matching.

        Args:
            request: The incoming coordinator request.

        Returns:
            One of the INTENT_* class constants.
        """
        if request.attachments:
            return self.INTENT_UPLOAD

        text = request.prompt.lower()

        upload_keywords = {"upload", "syllabus", "document", "pdf", "parse"}
        plan_keywords = {"plan", "schedule", "calendar", "reschedule", "block", "session"}
        progress_keywords = {"progress", "streak", "milestone", "metric", "alert", "performance"}

        if upload_keywords & set(text.split()):
            return self.INTENT_UPLOAD
        if plan_keywords & set(text.split()):
            return self.INTENT_PLAN
        if progress_keywords & set(text.split()):
            return self.INTENT_PROGRESS

        return self.INTENT_QNA

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    async def _delegate(
        self,
        intent: str,
        request: CoordinatorRequest,
        state: ConversationState,
    ) -> tuple[str, Optional[str]]:
        """
        Route execution to the correct sub-agent based on resolved intent.

        Each branch is a stub today.  Phase 2 will replace the stub bodies
        with real sub-agent calls using the ADK runner or sub_agents pattern.

        Args:
            intent:  One of the INTENT_* constants.
            request: The original coordinator request.
            state:   Current session state (read-only in delegation).

        Returns:
            A (reply_text, ui_action) tuple.  ui_action may be None.
        """

        if intent == self.INTENT_UPLOAD:
            # TODO(Phase 2): Invoke self._ingestion_agent with an IngestionRequest.
            # Example:
            #   ingestion_req = IngestionRequest(
            #       file_uri=request.attachments[0]["file_uri"],
            #       mime_type=request.attachments[0]["mime_type"],
            #       course_id=...,
            #   )
            #   result: IngestionResponse = await self._ingestion_agent.handle(ingestion_req)
            #   return result.extracted_json summary, "SHOW_SYLLABUS_PREVIEW"
            logger.debug("Delegating to IngestionAgent (stub).")
            return "I received your document and will begin parsing it now.", "SHOW_UPLOAD_PROGRESS"

        if intent == self.INTENT_PLAN:
            # TODO(Phase 2): Invoke self._planning_agent with a PlanningRequest.
            # Example:
            #   planning_req = PlanningRequest(user_id=request.user_id, course_id=...)
            #   result: PlanningResponse = await self._planning_agent.handle(planning_req)
            #   return f"{result.tasks_created} study blocks added to your calendar.", "OPEN_CALENDAR"
            logger.debug("Delegating to PlanningAgent (stub).")
            return "I am building your personalised study schedule.", "OPEN_CALENDAR"

        if intent == self.INTENT_PROGRESS:
            # TODO(Phase 2): Invoke self._monitoring_agent with a MonitoringRequest.
            # Example:
            #   result: MonitoringResponse = await self._monitoring_agent.handle(request.user_id)
            #   return result.summary_text, "SHOW_PROGRESS_DASHBOARD"
            logger.debug("Delegating to MonitoringAgent (stub).")
            return "Here is a summary of your study progress and streaks.", "SHOW_PROGRESS_DASHBOARD"

        # Default: QNA intent → Tutoring Agent
        # TODO(Phase 2): Invoke self._tutoring_agent with a TutoringRequest.
        # Example:
        #   tutoring_req = TutoringRequest(
        #       user_id=request.user_id,
        #       course_id=...,
        #       question=request.prompt,
        #       chat_history=state.messages,
        #   )
        #   result: TutoringResponse = await self._tutoring_agent.handle(tutoring_req)
        #   return result.answer, None
        logger.debug("Delegating to TutoringAgent (stub).")
        return "Let me help you understand that topic.", None

    # ------------------------------------------------------------------
    # Session state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_message(
        state: ConversationState,
        role: str,
        content: str,
    ) -> None:
        """
        Append a chat turn to the session message history.

        Args:
            state:   Mutable session state.
            role:    Either "user" or "assistant".
            content: The message text for this turn.
        """
        state.messages.append({"role": role, "content": content})

    # ------------------------------------------------------------------
    # Properties (read-only access to injected agents for testing)
    # ------------------------------------------------------------------

    @property
    def adk_agent(self) -> Agent:
        """The underlying Google ADK Agent instance."""
        return self._adk_agent

    @property
    def ingestion_agent(self) -> Optional[Agent]:
        """Injected IngestionAgent (None until Phase 2 wiring)."""
        return self._ingestion_agent

    @property
    def planning_agent(self) -> Optional[Agent]:
        """Injected PlanningAgent (None until Phase 2 wiring)."""
        return self._planning_agent

    @property
    def tutoring_agent(self) -> Optional[Agent]:
        """Injected TutoringAgent (None until Phase 2 wiring)."""
        return self._tutoring_agent

    @property
    def monitoring_agent(self) -> Optional[Agent]:
        """Injected MonitoringAgent (None until Phase 2 wiring)."""
        return self._monitoring_agent
