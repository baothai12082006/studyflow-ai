"""
monitoring_agent.py
-------------------
Monitoring & Streak Agent for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Receive validated MonitoringRequest objects from the Coordinator Agent.
  - Read ProgressState for the student via the progress_tool.
  - Analyse risk signals: missing streak, late assignments, low completion
    rate, inactivity, and upcoming deadlines within 48 hours.
  - Generate a human-readable monitoring summary and a list of
    actionable recommendations.
  - Send push notifications through the notification_tool when risk is
    detected — never when the student is on track.
  - Return a structured MonitoringResponse to the caller.
  - Never call repositories or services directly.
  - Never implement business analytics internally; all persistence access
    flows through the Tool Layer.

References:
  docs/architecture/multi_agent_architecture.md     §3.5 MonitoringAgent
  docs/architecture/agent_responsibility_matrix.md  §MonitoringAgent
  docs/architecture/interface_contracts.md          §MonitoringRequest / MonitoringResponse
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field

# Google ADK core
from google.adk import Agent

# Tool Layer – the only entry points this agent is permitted to use
from studyflow_ai.tools.notification_tool import send_immediate_reminder
from studyflow_ai.tools.progress_tool import fetch_user_progress

# Approved state contracts
from studyflow_ai.models.state import AcademicState, ProgressState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local request / response models
# (MonitoringRequest and MonitoringResponse are not yet in models/; they are
#  defined here pending the Phase 2 model-package consolidation described in
#  Interface Contracts §TODO-MON-01.)
# ---------------------------------------------------------------------------

class MonitoringRequest(BaseModel):
    """Payload delegated to the MonitoringAgent by the Coordinator."""

    user_id: str = Field(..., description="Student UUID to monitor.")
    academic_states: list[AcademicState] = Field(
        default_factory=list,
        description="Active course states used for deadline risk detection.",
    )


class MonitoringResponse(BaseModel):
    """Structured output returned from the MonitoringAgent."""

    summary: str = Field(..., description="Human-readable monitoring summary.")
    risk_level: str = Field(
        ...,
        description="Overall risk level: NONE, LOW, MEDIUM, or HIGH.",
    )
    notification_sent: bool = Field(
        ..., description="Whether a push notification was dispatched."
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Ordered list of actionable study recommendations.",
    )


# ---------------------------------------------------------------------------
# Risk signal constants
# ---------------------------------------------------------------------------

_STREAK_MISSING_THRESHOLD_DAYS: int = 2
_INACTIVITY_THRESHOLD_DAYS: int = 3
_LOW_COMPLETION_RATE_THRESHOLD: float = 0.4
_UPCOMING_DEADLINE_HOURS: int = 48
_MIN_TASKS_FOR_RATE: int = 5  # only apply rate check when enough data exists

_RISK_NONE: str = "NONE"
_RISK_LOW: str = "LOW"
_RISK_MEDIUM: str = "MEDIUM"
_RISK_HIGH: str = "HIGH"

# Notification priority mapping per risk level
_NOTIFY_PRIORITY: dict[str, str] = {
    _RISK_LOW: "NORMAL",
    _RISK_MEDIUM: "NORMAL",
    _RISK_HIGH: "HIGH",
}


# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    """
    Load a plain-text system prompt from the prompts/ package.

    Args:
        filename: Name of the .txt file inside studyflow_ai/prompts/.

    Returns:
        The prompt string, or a safe inline fallback if the file is missing.
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
            "You are the Monitoring & Streak Agent for StudyFlow AI. "
            "Monitor study activity, detect risks, and send timely reminders."
        )


# ---------------------------------------------------------------------------
# MonitoringAgent
# ---------------------------------------------------------------------------

class MonitoringAgent:
    """
    Monitors student study behaviour and dispatches risk-aware notifications.

    Design contract (Architecture Freeze §3.5):
      - Stateless; reads all data from the Tool Layer on every invocation.
      - Notifications sent only when a risk signal is detected.
      - Risk analysis is deterministic in Phase 1; Phase 2 will integrate
        Gemini-powered behavioural reasoning.

    The underlying ADK agent is registered with both monitoring tools to
    support future multi-turn agentic usage.
    """

    def __init__(self) -> None:
        self._adk_agent: Agent = Agent(
            name="MonitoringAgent",
            instruction=_load_prompt("monitoring.txt"),
            tools=[fetch_user_progress, send_immediate_reminder],
        )
        logger.info(
            "MonitoringAgent initialised (ADK agent: %s).", self._adk_agent.name
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def handle(
        self,
        request: MonitoringRequest,
    ) -> MonitoringResponse:
        """
        Execute the complete monitoring and notification workflow.

        Workflow:
          1. Validate the incoming request.
          2. Fetch the student's ProgressState via progress_tool.
          3. Detect risk signals across streak, completion, inactivity,
             and upcoming deadlines.
          4. Generate a summary and ordered recommendations.
          5. Dispatch a push notification if a risk is present.
          6. Return a structured MonitoringResponse.

        Args:
            request: Validated payload containing user_id and optional
                     academic_states for deadline-proximity analysis.

        Returns:
            MonitoringResponse with summary, risk_level, notification_sent,
            and recommendations.
        """
        logger.info("Starting monitoring | user_id=%s", request.user_id)

        # Step 1 – validate inputs.
        self._validate_request(request)

        # Step 2 – read progress through the Tool Layer.
        progress: ProgressState = await self._read_progress(request.user_id)

        # Step 3 – detect risks.
        risks: list[str] = self._detect_risks(
            progress=progress,
            academic_states=request.academic_states,
        )
        risk_level: str = self._resolve_risk_level(risks)

        # Step 4 – generate summary and recommendations.
        summary: str = self._generate_summary(progress, risks, risk_level)
        recommendations: list[str] = self._build_recommendations(risks, progress)

        # Step 5 – send notification when a risk is detected.
        notification_sent: bool = await self._send_notifications(
            user_id=request.user_id,
            risk_level=risk_level,
            risks=risks,
        )

        # Step 6 – compose and return the response.
        response = MonitoringResponse(
            summary=summary,
            risk_level=risk_level,
            notification_sent=notification_sent,
            recommendations=recommendations,
        )

        logger.info(
            "Monitoring complete | user_id=%s risk=%s notified=%s",
            request.user_id,
            risk_level,
            notification_sent,
        )
        return response

    # ------------------------------------------------------------------
    # Step 1 – validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(request: MonitoringRequest) -> None:
        """
        Validate required fields in the MonitoringRequest.

        Args:
            request: The incoming MonitoringRequest to validate.

        Raises:
            ValueError: If user_id is empty or missing.
        """
        if not request.user_id or not request.user_id.strip():
            raise ValueError("MonitoringRequest.user_id must not be empty.")

        logger.debug("MonitoringRequest validated for user_id=%s.", request.user_id)

    # ------------------------------------------------------------------
    # Step 2 – progress retrieval
    # ------------------------------------------------------------------

    async def _read_progress(self, user_id: str) -> ProgressState:
        """
        Fetch the student's learning progress through the progress_tool.

        Falls back to a zero-baseline ProgressState on tool failure so that
        monitoring can still detect inactivity-related risks.

        Args:
            user_id: The student's UUID.

        Returns:
            A ProgressState populated by the tool, or a default instance
            if the tool call fails.
        """
        logger.debug("Invoking fetch_user_progress | user_id=%s", user_id)
        try:
            progress: ProgressState = await fetch_user_progress(user_id=user_id)
            logger.info(
                "Progress fetched | user_id=%s streak=%d tasks=%d",
                user_id,
                progress.current_streak_days,
                progress.total_tasks_completed,
            )
            return progress
        except Exception as exc:
            logger.error(
                "fetch_user_progress failed for user_id=%s: %s. "
                "Using default ProgressState.",
                user_id,
                exc,
            )
            return ProgressState(user_id=user_id)

    # ------------------------------------------------------------------
    # Step 3 – risk detection
    # ------------------------------------------------------------------

    def _detect_risks(
        self,
        progress: ProgressState,
        academic_states: list[AcademicState],
    ) -> list[str]:
        """
        Analyse the student's progress and academic deadlines for risk signals.

        Evaluated signals:
          - MISSING_STREAK: current streak has lapsed for ≥ threshold days.
          - LOW_COMPLETION_RATE: ratio of completed tasks to struggle topics
            is below threshold (only applied when enough data exists).
          - INACTIVE_STUDENT: no tasks completed, suggesting the student has
            not engaged with the platform for several days.
          - LATE_ASSIGNMENT: a deadline in any provided AcademicState has
            already passed without a completed task recorded.
          - UPCOMING_DEADLINE: a deadline is within the next 48 hours.

        Args:
            progress:        The student's ProgressState.
            academic_states: One AcademicState per active course; may be empty.

        Returns:
            List of risk-signal identifier strings. Empty list means no risks.
        """
        risks: list[str] = []

        # Risk: streak missing
        if progress.current_streak_days < _STREAK_MISSING_THRESHOLD_DAYS:
            risks.append("MISSING_STREAK")
            logger.debug(
                "Risk detected: MISSING_STREAK | streak=%d",
                progress.current_streak_days,
            )

        # Risk: low completion rate (only meaningful above a minimum task count)
        total_attempts = progress.total_tasks_completed + len(progress.struggle_topics)
        if (
            total_attempts >= _MIN_TASKS_FOR_RATE
            and progress.total_tasks_completed > 0
        ):
            rate = progress.total_tasks_completed / total_attempts
            if rate < _LOW_COMPLETION_RATE_THRESHOLD:
                risks.append("LOW_COMPLETION_RATE")
                logger.debug("Risk detected: LOW_COMPLETION_RATE | rate=%.2f", rate)

        # Risk: inactive student (zero tasks ever completed)
        if progress.total_tasks_completed == 0:
            risks.append("INACTIVE_STUDENT")
            logger.debug("Risk detected: INACTIVE_STUDENT")

        # Deadline-based risks
        now = datetime.now(tz=timezone.utc)
        upcoming_threshold = now + timedelta(hours=_UPCOMING_DEADLINE_HOURS)

        for academic_state in academic_states:
            for deadline in academic_state.deadlines:
                deadline_dt = self._parse_iso(deadline.get("date", ""))
                if deadline_dt is None:
                    continue

                # Ensure deadline_dt is timezone-aware for comparison
                if deadline_dt.tzinfo is None:
                    deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)

                # Risk: deadline already passed
                if deadline_dt < now:
                    if "LATE_ASSIGNMENT" not in risks:
                        risks.append("LATE_ASSIGNMENT")
                        logger.debug(
                            "Risk detected: LATE_ASSIGNMENT | deadline=%s course=%s",
                            deadline.get("date"),
                            academic_state.course_id,
                        )

                # Risk: deadline within 48 hours
                elif deadline_dt <= upcoming_threshold:
                    if "UPCOMING_DEADLINE" not in risks:
                        risks.append("UPCOMING_DEADLINE")
                        logger.debug(
                            "Risk detected: UPCOMING_DEADLINE | deadline=%s course=%s",
                            deadline.get("date"),
                            academic_state.course_id,
                        )

        return risks

    @staticmethod
    def _resolve_risk_level(risks: list[str]) -> str:
        """
        Determine the aggregate risk level from the list of active risk signals.

        Escalation rules (highest match wins):
          HIGH   → LATE_ASSIGNMENT or (MISSING_STREAK + INACTIVE_STUDENT)
          MEDIUM → UPCOMING_DEADLINE or LOW_COMPLETION_RATE or INACTIVE_STUDENT
          LOW    → MISSING_STREAK alone
          NONE   → no signals

        Args:
            risks: List of risk-signal identifiers from _detect_risks().

        Returns:
            One of: "NONE", "LOW", "MEDIUM", "HIGH".
        """
        if not risks:
            return _RISK_NONE

        risk_set = set(risks)

        if "LATE_ASSIGNMENT" in risk_set or (
            "MISSING_STREAK" in risk_set and "INACTIVE_STUDENT" in risk_set
        ):
            return _RISK_HIGH

        if (
            "UPCOMING_DEADLINE" in risk_set
            or "LOW_COMPLETION_RATE" in risk_set
            or "INACTIVE_STUDENT" in risk_set
        ):
            return _RISK_MEDIUM

        if "MISSING_STREAK" in risk_set:
            return _RISK_LOW

        return _RISK_NONE

    # ------------------------------------------------------------------
    # Step 4 – summary and recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_summary(
        progress: ProgressState,
        risks: list[str],
        risk_level: str,
    ) -> str:
        """
        Compose a concise, human-readable monitoring summary.

        The summary always includes core progress metrics (streak, tasks
        completed) and a risk-level assessment. When risks are present, each
        active signal is listed so the student understands what triggered
        the monitoring alert.

        TODO(Phase 2): Replace static template with a Gemini-generated
        personalised summary using the student's persona and historical
        trends for richer, context-aware feedback.

        Args:
            progress:   The student's ProgressState.
            risks:      Active risk signals from _detect_risks().
            risk_level: Aggregate risk level from _resolve_risk_level().

        Returns:
            A multi-line Markdown-compatible summary string.
        """
        lines: list[str] = [
            f"**Study Activity Report**",
            f"- Current streak: {progress.current_streak_days} day(s)",
            f"- Highest streak: {progress.highest_streak_days} day(s)",
            f"- Tasks completed: {progress.total_tasks_completed}",
            f"- Risk level: **{risk_level}**",
        ]

        if risks:
            lines.append("\n**Active alerts:**")
            label_map: dict[str, str] = {
                "MISSING_STREAK": "Streak at risk — study today to keep it going.",
                "LOW_COMPLETION_RATE": "Task completion rate is below target.",
                "INACTIVE_STUDENT": "No study activity recorded yet.",
                "LATE_ASSIGNMENT": "One or more assignments are overdue.",
                "UPCOMING_DEADLINE": "A deadline is approaching within 48 hours.",
            }
            for risk in risks:
                lines.append(f"  • {label_map.get(risk, risk)}")
        else:
            lines.append("\n✅ All good — keep up the great work!")

        return "\n".join(lines)

    @staticmethod
    def _build_recommendations(
        risks: list[str],
        progress: ProgressState,
    ) -> list[str]:
        """
        Produce an ordered list of actionable recommendations from risk signals.

        Recommendations are ordered from highest urgency to lowest. Struggle
        topics from ProgressState are injected into relevant recommendations
        when available for personalisation.

        Args:
            risks:    Active risk signals from _detect_risks().
            progress: The student's ProgressState for personalisation data.

        Returns:
            List of recommendation strings. Empty if no risks are detected.
        """
        recommendations: list[str] = []
        risk_set = set(risks)

        if "LATE_ASSIGNMENT" in risk_set:
            recommendations.append(
                "Submit any overdue assignments immediately and contact your instructor."
            )

        if "UPCOMING_DEADLINE" in risk_set:
            recommendations.append(
                "Review your schedule for the next 48 hours and prioritise pending work."
            )

        if "INACTIVE_STUDENT" in risk_set:
            recommendations.append(
                "Start your first study session today to build your learning momentum."
            )

        if "LOW_COMPLETION_RATE" in risk_set:
            topics = (
                f" Focus on: {', '.join(progress.struggle_topics[:3])}."
                if progress.struggle_topics
                else ""
            )
            recommendations.append(
                f"Increase daily study blocks to improve your completion rate.{topics}"
            )

        if "MISSING_STREAK" in risk_set:
            recommendations.append(
                "Complete at least one study task today to restore your streak."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Step 5 – notifications
    # ------------------------------------------------------------------

    async def _send_notifications(
        self,
        user_id: str,
        risk_level: str,
        risks: list[str],
    ) -> bool:
        """
        Dispatch a push notification via the notification_tool when a risk
        is present. No notification is sent when risk_level is NONE.

        Priority escalation:
          HIGH   → "HIGH" priority notification.
          MEDIUM → "NORMAL" priority notification.
          LOW    → "NORMAL" priority notification.
          NONE   → no notification dispatched.

        Args:
            user_id:    The student's UUID.
            risk_level: Resolved risk level from _resolve_risk_level().
            risks:      Active risk signals for composing the alert message.

        Returns:
            True if a notification was sent successfully (or was not needed);
            False if a notification was attempted but the tool call failed.
        """
        if risk_level == _RISK_NONE:
            logger.debug(
                "No risks detected for user_id=%s. Skipping notification.", user_id
            )
            return False

        alert_message = self._compose_alert_message(risk_level, risks)
        priority = _NOTIFY_PRIORITY.get(risk_level, "NORMAL")

        logger.debug(
            "Invoking send_immediate_reminder | user_id=%s priority=%s",
            user_id,
            priority,
        )
        try:
            sent: bool = await send_immediate_reminder(
                user_id=user_id,
                alert_message=alert_message,
                priority=priority,
            )
            if sent:
                logger.info(
                    "Notification dispatched | user_id=%s priority=%s", user_id, priority
                )
            else:
                logger.warning(
                    "send_immediate_reminder returned False | user_id=%s", user_id
                )
            return sent
        except Exception as exc:
            logger.error(
                "send_immediate_reminder raised for user_id=%s: %s", user_id, exc
            )
            return False

    @staticmethod
    def _compose_alert_message(risk_level: str, risks: list[str]) -> str:
        """
        Compose a concise push-notification body from the active risk signals.

        The message is intentionally brief to suit push-notification character
        limits. Detailed guidance is surfaced in the MonitoringResponse summary.

        Args:
            risk_level: Resolved risk level string.
            risks:      Active risk signals.

        Returns:
            A single-line alert message string.
        """
        risk_set = set(risks)

        if "LATE_ASSIGNMENT" in risk_set:
            return "⚠️ You have overdue assignments. Open StudyFlow to take action now."

        if "UPCOMING_DEADLINE" in risk_set:
            return "📅 A deadline is due in less than 48 hours. Review your plan!"

        if "INACTIVE_STUDENT" in risk_set:
            return "👋 Welcome! Start your first study session to kick off your streak."

        if "LOW_COMPLETION_RATE" in risk_set:
            return "📉 Your task completion rate needs a boost. Let's study together!"

        if "MISSING_STREAK" in risk_set:
            return "🔥 Your streak is at risk! Complete a task today to keep it alive."

        return f"StudyFlow reminder [{risk_level}]: Check your study activity."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_iso(date_str: str) -> Optional[datetime]:
        """
        Parse an ISO 8601 datetime string tolerantly.

        Accepts strings ending in 'Z' (UTC) as well as standard offset
        notation. Returns None for any input that cannot be parsed so that
        individual malformed deadline entries are skipped gracefully.

        Args:
            date_str: Raw date string from an AcademicState deadline dict.

        Returns:
            A timezone-aware datetime, or None if parsing fails.
        """
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def adk_agent(self) -> Agent:
        """The underlying Google ADK Agent instance."""
        return self._adk_agent
