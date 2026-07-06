"""
planner_agent.py
----------------
Study Planning Agent for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Receive validated PlanningRequest objects from the Coordinator Agent.
  - Read AcademicState (deadlines, course metadata) to understand workload.
  - Read ProgressState (streaks, struggle topics) to personalise study intensity.
  - Read existing calendar events to avoid scheduling conflicts.
  - Generate a personalised study plan as a list of StudyTaskSchema objects.
  - Create study sessions through the calendar_tool.
  - Return a structured PlanningResponse to the caller.
  - Never implement scheduling algorithms inside the agent.
  - Never call CalendarService or repositories directly.
  - Delegate all calendar and progress work through the Tool Layer.

References:
  docs/architecture/multi_agent_architecture.md     §3.3 PlannerAgent
  docs/architecture/agent_responsibility_matrix.md  §PlannerAgent
  docs/architecture/interface_contracts.md          §PlanningRequest / PlanningResponse
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

# Google ADK core
from google.adk import Agent

# Tool Layer – the only entry points this agent is permitted to use
from studyflow_ai.tools.calendar_tool import (
    create_calendar_session,
    query_calendar_events,
)
from studyflow_ai.tools.progress_tool import fetch_user_progress

# Approved request / response / state contracts
from studyflow_ai.models.requests import PlanningRequest
from studyflow_ai.models.responses import PlanningResponse
from studyflow_ai.models.schemas import StudyTaskSchema
from studyflow_ai.models.state import AcademicState, ProgressState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Planning defaults
# ---------------------------------------------------------------------------

_DEFAULT_SESSION_MINUTES: int = 60
_PLANNING_WINDOW_DAYS: int = 14
_MAX_SESSIONS_PER_DAY: int = 3
_DAY_START_HOUR: int = 9
_DAY_END_HOUR: int = 21


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
            "You are the Planning Agent for StudyFlow AI. "
            "Generate optimised study plans considering course timelines, "
            "upcoming exams, and external calendar gaps."
        )


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """
    Generates personalised study schedules for StudyFlow AI.

    Design contract (Architecture Freeze §3.3):
      - Stateless except for data read from the Tool Layer during a single
        planning invocation.
      - All calendar and progress operations delegated through the Tool Layer.
      - Scheduling heuristics live in this agent only as a scaffold; they will
        be replaced by LLM-driven optimisation in Phase 2.

    The underlying ADK agent is registered with the calendar and progress tools
    so it can be used in a multi-turn agentic mode in the future.
    """

    def __init__(
        self,
        academic_state_provider: Optional[object] = None,
    ) -> None:
        """
        Initialise the PlannerAgent.

        Args:
            academic_state_provider: Injected dependency for resolving
                AcademicState by course_id. In Phase 1 this is provided by
                the Coordinator Agent or passed directly; in Phase 2 it will
                be replaced by a repository lookup tool.
        """
        self._academic_state_provider = academic_state_provider

        self._adk_agent: Agent = Agent(
            name="PlannerAgent",
            instruction=_load_prompt("planning.txt"),
            tools=[
                query_calendar_events,
                create_calendar_session,
                fetch_user_progress,
            ],
        )

        logger.info("PlannerAgent initialised (ADK agent: %s).", self._adk_agent.name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def handle(
        self,
        request: PlanningRequest,
        academic_state: AcademicState,
    ) -> PlanningResponse:
        """
        Execute the complete study-planning workflow.

        Workflow (per Execution Workflow doc §Create Study Plan):
          1. Validate the incoming request.
          2. Read AcademicState (passed by Coordinator).
          3. Read learning progress via progress_tool.
          4. Read existing calendar events via calendar_tool.
          5. Generate study tasks from deadlines and availability.
          6. Create study sessions via calendar_tool.
          7. Return PlanningResponse.

        Args:
            request:        Validated payload containing user_id and course_id.
            academic_state: The AcademicState for the target course, supplied
                            by the Coordinator Agent after ingestion.

        Returns:
            PlanningResponse with tasks_created count and calendar_sync_status.
        """
        logger.info(
            "Starting planning | user_id=%s course_id=%s",
            request.user_id,
            request.course_id,
        )

        # Step 1 – validate inputs.
        self._validate_request(request, academic_state)

        # Step 2 – AcademicState already supplied by the caller.
        deadlines = academic_state.deadlines
        logger.debug(
            "AcademicState loaded | course=%s deadlines=%d",
            academic_state.title,
            len(deadlines),
        )

        # Step 3 – read progress through the Tool Layer.
        progress: ProgressState = await self._read_progress(request.user_id)

        # Step 4 – read existing calendar events through the Tool Layer.
        now = datetime.utcnow()
        window_end = now + timedelta(days=_PLANNING_WINDOW_DAYS)
        busy_slots: list[dict] = await self._read_calendar(
            user_id=request.user_id,
            start=now,
            end=window_end,
        )

        # Step 5 – generate study tasks.
        tasks: list[StudyTaskSchema] = self._generate_study_tasks(
            course_id=request.course_id,
            course_title=academic_state.title,
            deadlines=deadlines,
            busy_slots=busy_slots,
            progress=progress,
            window_start=now,
            window_end=window_end,
        )

        # Step 6 – persist sessions through calendar_tool.
        sync_ok: bool = await self._create_sessions(request.user_id, tasks)

        # Step 7 – compose response.
        response = PlanningResponse(
            tasks_created=len(tasks),
            calendar_sync_status=sync_ok,
            error_msg=None if sync_ok else "Some study sessions failed to sync.",
        )

        logger.info(
            "Planning complete | user_id=%s tasks_created=%d synced=%s",
            request.user_id,
            response.tasks_created,
            response.calendar_sync_status,
        )
        return response

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(
        request: PlanningRequest,
        academic_state: AcademicState,
    ) -> None:
        """
        Validate required fields before running the planning pipeline.

        Args:
            request:        The incoming PlanningRequest.
            academic_state: The AcademicState to plan against.

        Raises:
            ValueError: On missing or mismatched fields.
        """
        if not request.user_id or not request.user_id.strip():
            raise ValueError("PlanningRequest.user_id must not be empty.")

        if not request.course_id or not request.course_id.strip():
            raise ValueError("PlanningRequest.course_id must not be empty.")

        if request.course_id != academic_state.course_id:
            raise ValueError(
                f"course_id mismatch: request has '{request.course_id}' "
                f"but AcademicState has '{academic_state.course_id}'."
            )

        logger.debug("PlanningRequest validated for user_id=%s.", request.user_id)

    async def _read_progress(self, user_id: str) -> ProgressState:
        """
        Fetch the student's learning progress via the progress_tool.

        Args:
            user_id: Student UUID.

        Returns:
            ProgressState (a default state if the tool returns empty).
        """
        logger.debug("Invoking fetch_user_progress | user_id=%s", user_id)
        try:
            return await fetch_user_progress(user_id=user_id)
        except Exception as exc:
            logger.warning(
                "fetch_user_progress failed for user_id=%s: %s. "
                "Proceeding with default progress.",
                user_id,
                exc,
            )
            return ProgressState(user_id=user_id)

    async def _read_calendar(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """
        Read existing calendar events via the calendar_tool.

        Args:
            user_id: Student UUID.
            start:   Planning window start (UTC).
            end:     Planning window end (UTC).

        Returns:
            List of busy-slot dicts (empty list on failure).
        """
        logger.debug(
            "Invoking query_calendar_events | user_id=%s window=%s→%s",
            user_id,
            start.isoformat(),
            end.isoformat(),
        )
        try:
            return await query_calendar_events(
                user_id=user_id,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            )
        except Exception as exc:
            logger.warning(
                "query_calendar_events failed for user_id=%s: %s. "
                "Proceeding without busy-slot filtering.",
                user_id,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Study task generation
    # ------------------------------------------------------------------

    def _generate_study_tasks(
        self,
        course_id: str,
        course_title: str,
        deadlines: list[dict],
        busy_slots: list[dict],
        progress: ProgressState,
        window_start: datetime,
        window_end: datetime,
    ) -> list[StudyTaskSchema]:
        """
        Generate StudyTaskSchema objects for each deadline in the planning window.

        Current heuristic (Phase 1):
          - One study session per deadline, scheduled on the day before the
            deadline (or today if the deadline is tomorrow or sooner).
          - Session duration adjusted by struggle-topic overlap: longer sessions
            for topics the student has struggled with.
          - Busy-slot conflict resolution is basic (shift by one hour).

        TODO(Phase 2): Replace this heuristic with an LLM-driven planner
        that receives deadlines, busy slots, and progress context and returns
        an optimised multi-day study schedule via ADK structured output.

        Args:
            course_id:    UUID of the course.
            course_title: Human-readable course title (for calendar entries).
            deadlines:    Deadline dicts from AcademicState.
            busy_slots:   Busy-slot dicts from calendar_tool.
            progress:     Student progress state.
            window_start: Planning window start.
            window_end:   Planning window end.

        Returns:
            List of StudyTaskSchema objects ready for calendar insertion.
        """
        tasks: list[StudyTaskSchema] = []
        busy_times: set[str] = self._extract_busy_hours(busy_slots)
        struggle_set: set[str] = set(progress.struggle_topics)

        for deadline in deadlines:
            deadline_date_str = deadline.get("date", "")
            deadline_title = deadline.get("title", "Untitled")
            deadline_type = deadline.get("type", "ASSIGNMENT")

            # Parse deadline date; skip if unparseable.
            deadline_dt = self._parse_iso(deadline_date_str)
            if deadline_dt is None:
                logger.warning(
                    "Skipping deadline '%s': unparseable date '%s'.",
                    deadline_title,
                    deadline_date_str,
                )
                continue

            # Only plan within the active window.
            if deadline_dt < window_start or deadline_dt > window_end:
                continue

            # Determine session duration – extend for struggle topics.
            duration_minutes = _DEFAULT_SESSION_MINUTES
            if deadline_title in struggle_set or deadline_type == "EXAM":
                duration_minutes = int(_DEFAULT_SESSION_MINUTES * 1.5)

            # Target the day before the deadline, defaulting to today.
            target_day = max(
                (deadline_dt - timedelta(days=1)).replace(
                    hour=_DAY_START_HOUR, minute=0, second=0, microsecond=0
                ),
                window_start.replace(
                    hour=_DAY_START_HOUR, minute=0, second=0, microsecond=0
                ),
            )

            # Find an available slot (basic conflict avoidance).
            session_start = self._find_open_slot(
                target_day, duration_minutes, busy_times
            )
            session_end = session_start + timedelta(minutes=duration_minutes)

            task = StudyTaskSchema(
                task_id=str(uuid.uuid4()),
                course_id=course_id,
                title=f"Study: {course_title} – {deadline_title}",
                start_time=session_start,
                end_time=session_end,
            )
            tasks.append(task)

            # Mark the slot as busy so subsequent tasks don't overlap.
            busy_times.add(session_start.isoformat())

        logger.info("Generated %d study task(s) for course_id=%s.", len(tasks), course_id)
        return tasks

    # ------------------------------------------------------------------
    # Calendar session creation
    # ------------------------------------------------------------------

    async def _create_sessions(
        self,
        user_id: str,
        tasks: list[StudyTaskSchema],
    ) -> bool:
        """
        Persist study tasks as calendar events via the calendar_tool.

        Args:
            user_id: Student UUID.
            tasks:   List of StudyTaskSchema objects to sync.

        Returns:
            True if all sessions were created successfully; False if any failed.
        """
        if not tasks:
            logger.debug("No tasks to sync – returning True.")
            return True

        all_ok = True
        for task in tasks:
            try:
                event_id: str = await create_calendar_session(
                    user_id=user_id,
                    title=task.title,
                    start_time=task.start_time.isoformat(),
                    end_time=task.end_time.isoformat(),
                )
                logger.debug(
                    "Calendar session created | task_id=%s event_id=%s",
                    task.task_id,
                    event_id,
                )
            except RuntimeError as exc:
                logger.error(
                    "Failed to create calendar session for task_id=%s: %s",
                    task.task_id,
                    exc,
                )
                all_ok = False

        return all_ok

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_iso(date_str: str) -> Optional[datetime]:
        """
        Parse an ISO 8601 datetime string tolerantly.

        Args:
            date_str: Raw date string from AcademicState deadlines.

        Returns:
            datetime or None if parsing fails.
        """
        if not date_str:
            return None
        try:
            # Handle trailing Z for UTC.
            cleaned = date_str.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_busy_hours(busy_slots: list[dict]) -> set[str]:
        """
        Extract ISO start-time strings from busy-slot dicts for fast lookup.

        Args:
            busy_slots: Raw dicts returned by query_calendar_events.

        Returns:
            Set of ISO datetime strings representing occupied hours.
        """
        times: set[str] = set()
        for slot in busy_slots:
            start = slot.get("start_time") or slot.get("start", "")
            if start:
                times.add(start)
        return times

    @staticmethod
    def _find_open_slot(
        target_day: datetime,
        duration_minutes: int,
        busy_times: set[str],
    ) -> datetime:
        """
        Find the earliest open slot on the target day that does not overlap
        with any entries in busy_times.

        TODO(Phase 2): Implement proper interval-tree conflict detection
        rather than hourly string comparison.

        Args:
            target_day:       Start of the target day (at DAY_START_HOUR).
            duration_minutes: Length of the session in minutes.
            busy_times:       Set of ISO strings marking occupied slots.

        Returns:
            datetime for the session start.
        """
        candidate = target_day
        attempts = 0
        max_attempts = _MAX_SESSIONS_PER_DAY * 4  # safety bound

        while attempts < max_attempts:
            if candidate.hour >= _DAY_END_HOUR:
                break
            if candidate.isoformat() not in busy_times:
                return candidate
            candidate += timedelta(hours=1)
            attempts += 1

        # Fallback: return the original target if nothing is free.
        return target_day

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def adk_agent(self) -> Agent:
        """The underlying Google ADK Agent instance."""
        return self._adk_agent
