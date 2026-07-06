"""
studyflow_agent.py
------------------
Root Agent for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Act as the single entry point for the entire agent system.
  - Own and orchestrate every sub-agent (Coordinator, Ingestion, Planner, Tutor, Monitoring).
  - Contain no business logic directly.
  - Expose the StudyFlowAgent interface for system interaction.

References:
  docs/design/multi_agent_architecture.md
  docs/design/agent_responsibility_matrix.md
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Google ADK core
from google.adk import Agent

# Sub-agents
from studyflow_ai.agents.coordinator_agent import CoordinatorAgent
from studyflow_ai.agents.ingestion_agent import IngestionAgent
from studyflow_ai.agents.planner_agent import PlannerAgent
from studyflow_ai.agents.tutor_agent import TutorAgent
from studyflow_ai.agents.monitoring_agent import MonitoringAgent

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
        os.path.dirname(__file__), "prompts", filename
    )
    try:
        with open(os.path.normpath(prompt_path), "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        logger.warning(
            "Prompt file '%s' not found. Using inline fallback.", filename
        )
        return (
            "You are the Root Agent for StudyFlow AI. "
            "Coordinate academic workflows, ingestion, planning, tutoring, and monitoring."
        )


# ---------------------------------------------------------------------------
# StudyFlowAgent
# ---------------------------------------------------------------------------

class StudyFlowAgent:
    """
    Root Agent for the StudyFlow AI platform.

    Exposes the system entry point, manages instantiation of sub-agents,
    wires the dependencies, and presents the central ADK Agent wrapper.

    Future Compatibility (Documentation Only):
      - TODO: Runner: Prepare the Root Agent to be run by the ADK Runner.
      - TODO: Session: Manage conversational session state across runs.
      - TODO: Memory: Integrate persistent memory for long-term user preferences.
      - TODO: ArtifactService: Enable storage and retrieval of course artifacts.
    """

    def __init__(self) -> None:
        """
        Initialise the StudyFlowAgent, instantiating and wiring all sub-agents.
        """
        logger.info("Initializing StudyFlowAgent and constructing sub-agents...")

        # 1. Instantiate the sub-agents
        self._ingestion_agent = IngestionAgent()
        self._planner_agent = PlannerAgent()
        self._tutor_agent = TutorAgent()
        self._monitoring_agent = MonitoringAgent()

        # 2. Inject sub-agents into the CoordinatorAgent (primary orchestrator)
        self._coordinator_agent = CoordinatorAgent(
            ingestion_agent=self._ingestion_agent,
            planning_agent=self._planner_agent,
            tutoring_agent=self._tutor_agent,
            monitoring_agent=self._monitoring_agent,
        )

        # 3. Create the root ADK Agent instance
        self.create_root_agent()

        logger.info("StudyFlowAgent initialised successfully.")

    def create_root_agent(self) -> Agent:
        """
        Create and configure the root ADK Agent representing StudyFlow AI.

        Returns:
            The root ADK Agent instance.
        """
        # Load the global system prompt from prompts/root.txt
        instruction = _load_prompt("root.txt")

        # Create one ADK Agent instance representing StudyFlow AI.
        # Registers the Coordinator's ADK agent as the primary sub-agent.
        self._adk_agent = Agent(
            name="StudyFlowAgent",
            instruction=instruction,
            sub_agents=[self._coordinator_agent.adk_agent],
        )

        logger.info(
            "Root ADK agent '%s' created (Sub-agents: %s).",
            self._adk_agent.name,
            [sa.name for sa in self._adk_agent.sub_agents] if hasattr(self._adk_agent, "sub_agents") else []
        )
        return self._adk_agent

    def coordinator(self) -> CoordinatorAgent:
        """
        Get the Coordinator Agent instance.

        Returns:
            The primary CoordinatorAgent orchestrator.
        """
        return self._coordinator_agent

    def adk_agent(self) -> Agent:
        """
        Get the underlying Google ADK Agent instance representing StudyFlow AI.

        Returns:
            The root ADK Agent.
        """
        return self._adk_agent

    # ------------------------------------------------------------------
    # Properties (read-only access to sub-agents)
    # ------------------------------------------------------------------

    @property
    def ingestion_agent(self) -> IngestionAgent:
        """The Ingestion & Parsing Agent instance."""
        return self._ingestion_agent

    @property
    def planning_agent(self) -> PlannerAgent:
        """The Planning & Scheduling Agent instance."""
        return self._planner_agent

    @property
    def tutoring_agent(self) -> TutorAgent:
        """The Tutoring & Explainer Agent instance."""
        return self._tutoring_agent

    @property
    def monitoring_agent(self) -> MonitoringAgent:
        """The Monitoring & Streak Agent instance."""
        return self._monitoring_agent
