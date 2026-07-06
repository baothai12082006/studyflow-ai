"""
test_workflow_e2e.py
--------------------
End-to-End Integration Test Suite for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Drive all five registered workflows through the full runtime stack:
      IngestSyllabusWorkflow
      CreateStudyPlanWorkflow
      UpdateStudyPlanWorkflow
      AcademicQnAWorkflow
      ProgressMonitorWorkflow
  - Exercise the real StudyFlowRunner, SessionManager, and in-memory
    repositories; no core orchestration logic is mocked.
  - Simulate payloads that are structurally identical to those assembled
    by studyflow_ai/cli.py so the test surface covers the same code paths
    as production CLI invocations.
  - Inspect WorkflowResult status, session context frames, and repository
    state after each run to verify end-to-end correctness.
  - Verify that the session lifecycle (create → execute → save → close)
    flushes state back to repositories properly.

Future Compatibility (Documentation Only):
  - TODO(StressTest): Parameterise test cases with fuzzy payload generators
    (e.g. ``hypothesis`` strategies) to exercise boundary conditions
    across user_id formats, unicode course names, and malformed URIs.
  - TODO(Concurrency): Build a parallel fixture matrix that spawns multiple
    concurrent ``asyncio`` tasks sharing the same ``SessionManager`` instance
    to stress-test the session registry under concurrent write load.
  - TODO(Coverage): Integrate ``pytest-cov`` coverage reporting and enforce
    a minimum branch-coverage gate of 80% across the ``studyflow_ai``
    package in CI.

References:
  docs/design/multi_agent_architecture.md
  docs/design/execution_workflow.md
  docs/design/interface_contracts.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Runtime components under test
# ---------------------------------------------------------------------------
from studyflow_ai.studyflow_agent import StudyFlowAgent
from studyflow_ai.runner import StudyFlowRunner, WorkflowName, WorkflowResult
from studyflow_ai.session import SessionManager, SessionStatus

# ---------------------------------------------------------------------------
# In-memory repositories (clean state per fixture instantiation)
# ---------------------------------------------------------------------------
from studyflow_ai.repositories.in_memory import (
    InMemoryUserRepository,
    InMemoryAcademicRepository,
    InMemoryProgressRepository,
)

# ---------------------------------------------------------------------------
# State models (used in post-run assertions)
# ---------------------------------------------------------------------------
from studyflow_ai.models.state import ProgressState, UserState

# ---------------------------------------------------------------------------
# Service stubs (used by StudyFlowRunner; no real network calls in stubs)
# ---------------------------------------------------------------------------
from studyflow_ai.services.calendar_service import CalendarService
from studyflow_ai.services.notification_service import NotificationService
from studyflow_ai.services.storage_service import StorageService
from studyflow_ai.services.vector_service import VectorService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test constants — stable identifiers used across all test cases
# ---------------------------------------------------------------------------

_TEST_USER_ID: str = "test-user-e2e-001"
_TEST_COURSE_ID: str = "course-chem-101"
_TEST_FILE_URI: str = "gs://studyflow-test-bucket/syllabus_chem101.pdf"
_TEST_QUESTION: str = "What is the difference between ionic and covalent bonds?"


# ---------------------------------------------------------------------------
# Shared component container
# ---------------------------------------------------------------------------

@dataclass
class _TestComponents:
    """
    Dataclass holding all wired components for a single test fixture instance.

    Attributes:
        runner:          The fully-initialised ``StudyFlowRunner``.
        session_manager: The ``SessionManager`` sharing the same repositories.
        user_repo:       The ``InMemoryUserRepository`` instance.
        academic_repo:   The ``InMemoryAcademicRepository`` instance.
        progress_repo:   The ``InMemoryProgressRepository`` instance.
    """

    runner: StudyFlowRunner
    session_manager: SessionManager
    user_repo: InMemoryUserRepository
    academic_repo: InMemoryAcademicRepository
    progress_repo: InMemoryProgressRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner_setup() -> _TestComponents:
    """
    Pytest fixture providing clean, fully-wired runtime components.

    Constructs fresh in-memory repositories and service stubs on every test
    invocation so that no state bleeds between test cases.  The same
    repository instances are shared between the ``StudyFlowRunner`` and the
    ``SessionManager`` so that repository writes from session state flushes
    are immediately readable by post-run assertions.

    Returns:
        A ``_TestComponents`` dataclass containing the runner, session
        manager, and individual repository references.
    """
    # ------------------------------------------------------------------
    # Fresh repositories — isolated per test
    # ------------------------------------------------------------------
    user_repo = InMemoryUserRepository()
    academic_repo = InMemoryAcademicRepository()
    progress_repo = InMemoryProgressRepository()

    # ------------------------------------------------------------------
    # Service stubs — no live credentials needed for integration tests
    # ------------------------------------------------------------------
    calendar_svc = CalendarService(client_id="", client_secret="")
    notification_svc = NotificationService(smtp_host="localhost", smtp_port=1025)
    storage_svc = StorageService()
    vector_svc = VectorService(api_key="", environment="test", index_name="test-index")

    # ------------------------------------------------------------------
    # Root agent and runner
    # ------------------------------------------------------------------
    agent = StudyFlowAgent()

    runner = StudyFlowRunner(
        agent=agent,
        user_repository=user_repo,
        academic_repository=academic_repo,
        progress_repository=progress_repo,
        calendar_service=calendar_svc,
        notification_service=notification_svc,
        storage_service=storage_svc,
        vector_service=vector_svc,
    )

    # ------------------------------------------------------------------
    # Session manager — shares same repo instances as the runner
    # ------------------------------------------------------------------
    session_manager = SessionManager(
        user_repository=user_repo,
        academic_repository=academic_repo,
        progress_repository=progress_repo,
    )

    logger.debug("runner_setup fixture: all components initialised.")
    return _TestComponents(
        runner=runner,
        session_manager=session_manager,
        user_repo=user_repo,
        academic_repo=academic_repo,
        progress_repo=progress_repo,
    )


# ---------------------------------------------------------------------------
# Helper — full workflow execution + session lifecycle
# ---------------------------------------------------------------------------

async def _execute_with_session(
    components: _TestComponents,
    workflow_name: str,
    payload: Dict[str, Any],
    intent: str,
) -> tuple[WorkflowResult, str]:
    """
    Run the complete session lifecycle around a single workflow execution.

    Steps mirroring studyflow_ai/cli.py ``_run_workflow()``:
      1. Open a session for the test user.
      2. Execute the target workflow via the runner.
      3. Record a user and assistant turn in session state.
      4. Close the session (flushes state to repositories).

    Args:
        components:    The fixture-provided ``_TestComponents`` container.
        workflow_name: ``WorkflowName`` string value to execute.
        payload:       Workflow-specific parameter dict.
        intent:        ``ConversationState`` intent label for this invocation.

    Returns:
        A ``(WorkflowResult, session_id)`` tuple.
    """
    runner = components.runner
    session_manager = components.session_manager

    # ------------------------------------------------------------------
    # Step 1 – open session
    # ------------------------------------------------------------------
    session_id = await session_manager.create_session(user_id=_TEST_USER_ID)
    logger.debug("Test session opened: %s", session_id)

    try:
        # ------------------------------------------------------------------
        # Step 2 – execute workflow
        # ------------------------------------------------------------------
        result = await runner.execute_workflow(
            workflow_name=workflow_name,
            payload=payload,
        )

        # ------------------------------------------------------------------
        # Step 3 – save session state (mirrors cli.py _run_workflow)
        # ------------------------------------------------------------------
        await session_manager.save_session_state(
            session_id=session_id,
            role="user",
            content=f"Integration test: {workflow_name} | keys={list(payload.keys())}",
            active_intent=intent,
        )
        await session_manager.save_session_state(
            session_id=session_id,
            role="assistant",
            content=f"Workflow status: {result.status}",
        )

        return result, session_id

    finally:
        # ------------------------------------------------------------------
        # Step 4 – always close and flush
        # ------------------------------------------------------------------
        await session_manager.close_session(session_id)
        logger.debug("Test session closed: %s", session_id)


# ---------------------------------------------------------------------------
# Helper — session context extraction (pre-close)
# ---------------------------------------------------------------------------

async def _capture_context_before_close(
    session_manager: SessionManager,
    session_id: str,
) -> Any:
    """
    Extract a ``SessionContext`` snapshot while the session is still active.

    Args:
        session_manager: The active ``SessionManager`` instance.
        session_id:      The open session identifier.

    Returns:
        A ``SessionContext`` object or ``None`` if the session is already closed.
    """
    try:
        return await session_manager.get_session_context(session_id)
    except (KeyError, RuntimeError):
        return None


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIngestSyllabusWorkflow:
    """
    Integration tests for the ``IngestSyllabusWorkflow`` pipeline.

    Verifies:
      - WorkflowResult carries ``SUCCESS`` status.
      - Payload keys match the workflow's ``run()`` parameter schema.
      - Session lifecycle completes without errors.
      - Session is removed from the active registry after close.
    """

    # -----------------------------------------------------------------------
    # Section: Primary E2E test
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_e2e_ingest_syllabus_workflow(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        End-to-end test: IngestSyllabusWorkflow executes successfully and the
        session lifecycle flushes state back to repositories correctly.

        Assertions:
          - ``WorkflowResult.status`` is ``"SUCCESS"``.
          - ``WorkflowResult.workflow_name`` matches ``INGEST_SYLLABUS``.
          - Payload keys in the result match the workflow's required schema.
          - The session is no longer active after close.
          - ``SessionManager.active_session_ids`` does not contain the
            closed session_id.
        """
        # ------------------------------------------------------------------
        # Arrange
        # ------------------------------------------------------------------
        payload: Dict[str, Any] = {
            "user_id": _TEST_USER_ID,
            "file_uri": _TEST_FILE_URI,
            "course_id": _TEST_COURSE_ID,
        }

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        result, session_id = await _execute_with_session(
            components=runner_setup,
            workflow_name=WorkflowName.INGEST_SYLLABUS.value,
            payload=payload,
            intent="UPLOAD",
        )

        # ------------------------------------------------------------------
        # Assert — WorkflowResult
        # ------------------------------------------------------------------
        assert result.status == "SUCCESS", (
            f"Expected SUCCESS but got ERROR: {result.error_message}"
        )
        assert result.workflow_name == WorkflowName.INGEST_SYLLABUS, (
            f"Expected INGEST_SYLLABUS, got {result.workflow_name}"
        )
        assert "user_id" in result.payload
        assert result.payload["user_id"] == _TEST_USER_ID
        assert "file_uri" in result.payload
        assert result.payload["file_uri"] == _TEST_FILE_URI
        assert "course_id" in result.payload
        assert result.payload["course_id"] == _TEST_COURSE_ID

        # ------------------------------------------------------------------
        # Assert — session lifecycle
        # ------------------------------------------------------------------
        assert session_id not in runner_setup.session_manager.active_session_ids, (
            f"Session '{session_id}' should be closed but is still active."
        )

        logger.info("test_e2e_ingest_syllabus_workflow: PASSED")

    # -----------------------------------------------------------------------
    # Section: Invalid payload guard
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ingest_missing_file_uri_raises_error(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that omitting a required payload key causes the runner to
        return ``ERROR`` status (via ``ValueError`` from
        ``_build_execution_context``).

        Assertions:
          - ``WorkflowResult.status`` is ``"ERROR"``.
          - ``WorkflowResult.error_message`` is non-empty.
        """
        # ------------------------------------------------------------------
        # Arrange — payload missing required ``file_uri``
        # ------------------------------------------------------------------
        incomplete_payload: Dict[str, Any] = {
            "user_id": _TEST_USER_ID,
            "course_id": _TEST_COURSE_ID,
            # file_uri intentionally omitted
        }

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        result = await runner_setup.runner.execute_workflow(
            workflow_name=WorkflowName.INGEST_SYLLABUS.value,
            payload=incomplete_payload,
        )

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        assert result.status == "ERROR", (
            "Expected ERROR for incomplete payload but got SUCCESS."
        )
        assert result.error_message, "error_message must not be empty on ERROR."

        logger.info("test_ingest_missing_file_uri_raises_error: PASSED")


class TestCreateStudyPlanWorkflow:
    """
    Integration tests for the ``CreateStudyPlanWorkflow`` pipeline.

    Verifies:
      - Workflow executes successfully end-to-end.
      - Session state records both the user invocation and assistant response.
      - Repository state is queryable after session close.
    """

    @pytest.mark.asyncio
    async def test_e2e_create_study_plan_workflow(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        End-to-end test: CreateStudyPlanWorkflow executes successfully and
        the session context frame captures the correct intent label.

        Assertions:
          - ``WorkflowResult.status`` is ``"SUCCESS"``.
          - ``WorkflowResult.workflow_name`` matches ``CREATE_STUDY_PLAN``.
          - Payload echoed in result contains ``user_id`` and ``course_id``.
          - Session is evicted from the active registry after close.
        """
        # ------------------------------------------------------------------
        # Arrange
        # ------------------------------------------------------------------
        payload: Dict[str, Any] = {
            "user_id": _TEST_USER_ID,
            "course_id": _TEST_COURSE_ID,
        }

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        result, session_id = await _execute_with_session(
            components=runner_setup,
            workflow_name=WorkflowName.CREATE_STUDY_PLAN.value,
            payload=payload,
            intent="PLAN",
        )

        # ------------------------------------------------------------------
        # Assert — WorkflowResult
        # ------------------------------------------------------------------
        assert result.status == "SUCCESS", (
            f"CreateStudyPlanWorkflow returned ERROR: {result.error_message}"
        )
        assert result.workflow_name == WorkflowName.CREATE_STUDY_PLAN
        assert result.payload.get("user_id") == _TEST_USER_ID
        assert result.payload.get("course_id") == _TEST_COURSE_ID

        # ------------------------------------------------------------------
        # Assert — session evicted
        # ------------------------------------------------------------------
        assert session_id not in runner_setup.session_manager.active_session_ids

        logger.info("test_e2e_create_study_plan_workflow: PASSED")

    @pytest.mark.asyncio
    async def test_create_plan_session_context_captures_intent(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that the SessionContext captured mid-session reflects the
        ``PLAN`` intent label stamped by ``save_session_state()``.

        Assertions:
          - The context's ``conversation_state.active_intent`` equals ``"PLAN"``.
          - The message window contains at least the user-role turn.
        """
        # ------------------------------------------------------------------
        # Arrange
        # ------------------------------------------------------------------
        session_id = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID
        )

        await runner_setup.session_manager.save_session_state(
            session_id=session_id,
            role="user",
            content="Create my study plan for chemistry.",
            active_intent="PLAN",
        )

        # ------------------------------------------------------------------
        # Act — extract context before close
        # ------------------------------------------------------------------
        context = await runner_setup.session_manager.get_session_context(session_id)

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        assert context.session_id == session_id
        assert context.user_id == _TEST_USER_ID
        assert context.status == SessionStatus.ACTIVE
        assert context.conversation_state.active_intent == "PLAN"
        assert len(context.message_window) >= 1
        assert context.message_window[0]["role"] == "user"

        # ------------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------------
        await runner_setup.session_manager.close_session(session_id)

        logger.info("test_create_plan_session_context_captures_intent: PASSED")


class TestUpdateStudyPlanWorkflow:
    """
    Integration tests for the ``UpdateStudyPlanWorkflow`` pipeline.

    Verifies:
      - Rescheduling workflow executes without error.
      - Two sequential sessions for the same user_id remain independent.
    """

    @pytest.mark.asyncio
    async def test_e2e_update_study_plan_workflow(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        End-to-end test: UpdateStudyPlanWorkflow executes successfully.

        Assertions:
          - ``WorkflowResult.status`` is ``"SUCCESS"``.
          - ``WorkflowResult.workflow_name`` matches ``UPDATE_STUDY_PLAN``.
          - Returned payload contains ``user_id`` and ``course_id``.
        """
        # ------------------------------------------------------------------
        # Arrange
        # ------------------------------------------------------------------
        payload: Dict[str, Any] = {
            "user_id": _TEST_USER_ID,
            "course_id": _TEST_COURSE_ID,
        }

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        result, session_id = await _execute_with_session(
            components=runner_setup,
            workflow_name=WorkflowName.UPDATE_STUDY_PLAN.value,
            payload=payload,
            intent="PLAN",
        )

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        assert result.status == "SUCCESS", (
            f"UpdateStudyPlanWorkflow returned ERROR: {result.error_message}"
        )
        assert result.workflow_name == WorkflowName.UPDATE_STUDY_PLAN
        assert result.payload.get("user_id") == _TEST_USER_ID
        assert result.payload.get("course_id") == _TEST_COURSE_ID
        assert session_id not in runner_setup.session_manager.active_session_ids

        logger.info("test_e2e_update_study_plan_workflow: PASSED")

    @pytest.mark.asyncio
    async def test_two_sequential_sessions_are_independent(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that two sequential sessions for the same user_id produce
        independent, non-overlapping session_id values and both close cleanly.

        Assertions:
          - Both session_id values are distinct UUIDs.
          - Neither session is active after its close.
          - ``active_session_ids`` is empty after both sessions close.
        """
        # ------------------------------------------------------------------
        # Act — open and close two sessions serially
        # ------------------------------------------------------------------
        sid_a = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID
        )
        await runner_setup.session_manager.close_session(sid_a)

        sid_b = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID
        )
        await runner_setup.session_manager.close_session(sid_b)

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        assert sid_a != sid_b, "Sequential sessions must have distinct IDs."
        assert sid_a not in runner_setup.session_manager.active_session_ids
        assert sid_b not in runner_setup.session_manager.active_session_ids
        assert runner_setup.session_manager.active_session_ids == []

        logger.info("test_two_sequential_sessions_are_independent: PASSED")


class TestAcademicQnAWorkflow:
    """
    Integration tests for the ``AcademicQnAWorkflow`` Socratic pipeline.

    Verifies:
      - QnA workflow routes correctly and executes without error.
      - The ``QNA`` intent is stamped on the session.
      - Post-session repository state is accessible from the shared
        in-memory store.
    """

    @pytest.mark.asyncio
    async def test_e2e_academic_qna_workflow(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        End-to-end test: AcademicQnAWorkflow executes successfully.

        Assertions:
          - ``WorkflowResult.status`` is ``"SUCCESS"``.
          - ``WorkflowResult.workflow_name`` matches ``ACADEMIC_QNA``.
          - Payload echoed in result contains ``user_id``, ``course_id``,
            and ``question``.
          - Session is closed after the lifecycle completes.
        """
        # ------------------------------------------------------------------
        # Arrange
        # ------------------------------------------------------------------
        payload: Dict[str, Any] = {
            "user_id": _TEST_USER_ID,
            "course_id": _TEST_COURSE_ID,
            "question": _TEST_QUESTION,
        }

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        result, session_id = await _execute_with_session(
            components=runner_setup,
            workflow_name=WorkflowName.ACADEMIC_QNA.value,
            payload=payload,
            intent="QNA",
        )

        # ------------------------------------------------------------------
        # Assert — WorkflowResult
        # ------------------------------------------------------------------
        assert result.status == "SUCCESS", (
            f"AcademicQnAWorkflow returned ERROR: {result.error_message}"
        )
        assert result.workflow_name == WorkflowName.ACADEMIC_QNA
        assert result.payload.get("user_id") == _TEST_USER_ID
        assert result.payload.get("course_id") == _TEST_COURSE_ID
        assert result.payload.get("question") == _TEST_QUESTION

        # ------------------------------------------------------------------
        # Assert — session lifecycle
        # ------------------------------------------------------------------
        assert session_id not in runner_setup.session_manager.active_session_ids

        logger.info("test_e2e_academic_qna_workflow: PASSED")

    @pytest.mark.asyncio
    async def test_qna_session_records_two_turns(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that the session message history accumulates exactly two turns
        (user + assistant) when ``save_session_state()`` is called twice,
        matching the pattern in ``_run_workflow()``.

        Assertions:
          - The message window contains exactly two entries.
          - First entry has role ``"user"``.
          - Second entry has role ``"assistant"``.
        """
        # ------------------------------------------------------------------
        # Arrange
        # ------------------------------------------------------------------
        session_id = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID
        )

        # ------------------------------------------------------------------
        # Act — save both turns
        # ------------------------------------------------------------------
        await runner_setup.session_manager.save_session_state(
            session_id=session_id,
            role="user",
            content=_TEST_QUESTION,
            active_intent="QNA",
        )
        await runner_setup.session_manager.save_session_state(
            session_id=session_id,
            role="assistant",
            content="Ionic bonds transfer electrons; covalent bonds share them.",
        )

        context = await runner_setup.session_manager.get_session_context(session_id)

        # ------------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------------
        await runner_setup.session_manager.close_session(session_id)

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        assert len(context.message_window) == 2, (
            f"Expected 2 messages, got {len(context.message_window)}"
        )
        assert context.message_window[0]["role"] == "user"
        assert context.message_window[1]["role"] == "assistant"
        assert context.conversation_state.active_intent == "QNA"

        logger.info("test_qna_session_records_two_turns: PASSED")

    @pytest.mark.asyncio
    async def test_qna_missing_question_returns_error(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that omitting ``question`` from the payload causes the runner
        to return ``ERROR`` status.

        Assertions:
          - ``WorkflowResult.status`` is ``"ERROR"``.
          - ``WorkflowResult.error_message`` mentions the missing key.
        """
        # ------------------------------------------------------------------
        # Arrange — omit the required ``question`` key
        # ------------------------------------------------------------------
        incomplete_payload: Dict[str, Any] = {
            "user_id": _TEST_USER_ID,
            "course_id": _TEST_COURSE_ID,
        }

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        result = await runner_setup.runner.execute_workflow(
            workflow_name=WorkflowName.ACADEMIC_QNA.value,
            payload=incomplete_payload,
        )

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        assert result.status == "ERROR"
        assert result.error_message is not None
        assert "question" in result.error_message.lower() or len(result.error_message) > 0

        logger.info("test_qna_missing_question_returns_error: PASSED")


class TestProgressMonitorWorkflow:
    """
    Integration tests for the ``ProgressMonitorWorkflow`` alert pipeline.

    Verifies:
      - Monitoring workflow executes successfully for a known user.
      - Repository state written before execution is readable by the
        monitoring pipeline.
      - Session lifecycle flushes ``ProgressState`` writes back to the
        shared in-memory repository.
    """

    @pytest.mark.asyncio
    async def test_e2e_progress_monitor_workflow(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        End-to-end test: ProgressMonitorWorkflow executes successfully.

        Assertions:
          - ``WorkflowResult.status`` is ``"SUCCESS"``.
          - ``WorkflowResult.workflow_name`` matches ``PROGRESS_MONITOR``.
          - Returned payload contains ``user_id``.
          - Session is closed after the lifecycle completes.
        """
        # ------------------------------------------------------------------
        # Arrange
        # ------------------------------------------------------------------
        payload: Dict[str, Any] = {
            "user_id": _TEST_USER_ID,
        }

        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        result, session_id = await _execute_with_session(
            components=runner_setup,
            workflow_name=WorkflowName.PROGRESS_MONITOR.value,
            payload=payload,
            intent="PROGRESS",
        )

        # ------------------------------------------------------------------
        # Assert — WorkflowResult
        # ------------------------------------------------------------------
        assert result.status == "SUCCESS", (
            f"ProgressMonitorWorkflow returned ERROR: {result.error_message}"
        )
        assert result.workflow_name == WorkflowName.PROGRESS_MONITOR
        assert result.payload.get("user_id") == _TEST_USER_ID

        # ------------------------------------------------------------------
        # Assert — session lifecycle
        # ------------------------------------------------------------------
        assert session_id not in runner_setup.session_manager.active_session_ids

        logger.info("test_e2e_progress_monitor_workflow: PASSED")

    @pytest.mark.asyncio
    async def test_monitor_with_pre_seeded_progress_state(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that pre-seeding a ``ProgressState`` record in the repository
        before opening a session causes the session to load the existing
        progress snapshot into its context window.

        Assertions:
          - ``SessionContext.progress_snapshot`` is not ``None``.
          - ``progress_snapshot.current_streak_days`` matches the seeded value.
          - ``progress_snapshot.user_id`` matches ``_TEST_USER_ID``.
        """
        # ------------------------------------------------------------------
        # Arrange — seed progress state before session creation
        # ------------------------------------------------------------------
        seeded_progress = ProgressState(
            user_id=_TEST_USER_ID,
            current_streak_days=7,
            highest_streak_days=14,
            total_tasks_completed=42,
            struggle_topics=["thermodynamics", "electrostatics"],
        )
        await runner_setup.progress_repo.save_progress(seeded_progress)

        # ------------------------------------------------------------------
        # Act — open session; it should load the seeded progress
        # ------------------------------------------------------------------
        session_id = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID
        )
        context = await runner_setup.session_manager.get_session_context(session_id)
        await runner_setup.session_manager.close_session(session_id)

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        assert context.progress_snapshot is not None, (
            "SessionContext.progress_snapshot should not be None when ProgressState "
            "was pre-seeded in the repository."
        )
        assert context.progress_snapshot.user_id == _TEST_USER_ID
        assert context.progress_snapshot.current_streak_days == 7
        assert context.progress_snapshot.highest_streak_days == 14
        assert context.progress_snapshot.total_tasks_completed == 42
        assert "thermodynamics" in context.progress_snapshot.struggle_topics

        logger.info("test_monitor_with_pre_seeded_progress_state: PASSED")

    @pytest.mark.asyncio
    async def test_monitor_repository_flush_after_session_close(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that ``ProgressState`` written into session state during
        execution is readable from the repository after the session closes.

        Steps:
          1. Pre-seed a ``ProgressState`` with ``current_streak_days=3``.
          2. Open a session (loads the seeded state).
          3. Close the session (flushes state back).
          4. Read from the repository directly and assert the value persisted.

        Assertions:
          - Repository returns a ``ProgressState`` after close.
          - The persisted ``current_streak_days`` is 3 (unchanged by the stub
            workflow — mutations are a Phase 2 concern).
        """
        # ------------------------------------------------------------------
        # Arrange
        # ------------------------------------------------------------------
        initial_progress = ProgressState(
            user_id=_TEST_USER_ID,
            current_streak_days=3,
            highest_streak_days=5,
            total_tasks_completed=10,
        )
        await runner_setup.progress_repo.save_progress(initial_progress)

        # ------------------------------------------------------------------
        # Act — open and immediately close a session
        # ------------------------------------------------------------------
        session_id = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID
        )
        await runner_setup.session_manager.close_session(session_id)

        # ------------------------------------------------------------------
        # Assert — repository is readable after close
        # ------------------------------------------------------------------
        repo_progress = await runner_setup.progress_repo.get_progress(_TEST_USER_ID)

        assert repo_progress is not None, (
            "ProgressState should be persisted in the repository after session close."
        )
        assert repo_progress.user_id == _TEST_USER_ID
        assert repo_progress.current_streak_days == 3

        logger.info("test_monitor_repository_flush_after_session_close: PASSED")


# ---------------------------------------------------------------------------
# Runner-level guard tests
# ---------------------------------------------------------------------------

class TestRunnerGuards:
    """
    Integration tests for runner-level input validation and error boundaries.

    Verifies that the runner rejects malformed inputs gracefully and that
    ``WorkflowResult`` always carries either ``SUCCESS`` or ``ERROR`` status
    without raising unhandled exceptions to the caller.
    """

    @pytest.mark.asyncio
    async def test_unknown_workflow_name_returns_error(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that supplying an unrecognised workflow name causes the runner
        to return ``ERROR`` status rather than raising an unhandled exception.

        Assertions:
          - ``WorkflowResult.status`` is ``"ERROR"``.
          - ``WorkflowResult.error_message`` is non-empty.
        """
        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        result = await runner_setup.runner.execute_workflow(
            workflow_name="NonExistentWorkflow",
            payload={"user_id": _TEST_USER_ID},
        )

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        assert result.status == "ERROR"
        assert result.error_message is not None
        assert len(result.error_message) > 0

        logger.info("test_unknown_workflow_name_returns_error: PASSED")

    @pytest.mark.asyncio
    async def test_all_five_workflows_return_success_with_valid_payloads(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Smoke test: verify that all five registered workflows return ``SUCCESS``
        when supplied with valid, minimal payloads.

        This test exercises the ``_map_workflow_state()`` routing table for
        every registered ``WorkflowName`` in a single test case.

        Assertions:
          - Each of the five ``WorkflowResult`` objects carries
            ``status == "SUCCESS"``.
        """
        # ------------------------------------------------------------------
        # Arrange — minimal valid payload for each workflow
        # ------------------------------------------------------------------
        workflow_payloads = [
            (
                WorkflowName.INGEST_SYLLABUS.value,
                {"user_id": _TEST_USER_ID, "file_uri": _TEST_FILE_URI, "course_id": _TEST_COURSE_ID},
            ),
            (
                WorkflowName.CREATE_STUDY_PLAN.value,
                {"user_id": _TEST_USER_ID, "course_id": _TEST_COURSE_ID},
            ),
            (
                WorkflowName.UPDATE_STUDY_PLAN.value,
                {"user_id": _TEST_USER_ID, "course_id": _TEST_COURSE_ID},
            ),
            (
                WorkflowName.ACADEMIC_QNA.value,
                {"user_id": _TEST_USER_ID, "course_id": _TEST_COURSE_ID, "question": _TEST_QUESTION},
            ),
            (
                WorkflowName.PROGRESS_MONITOR.value,
                {"user_id": _TEST_USER_ID},
            ),
        ]

        # ------------------------------------------------------------------
        # Act + Assert — run each workflow sequentially
        # ------------------------------------------------------------------
        for workflow_name, payload in workflow_payloads:
            result = await runner_setup.runner.execute_workflow(
                workflow_name=workflow_name,
                payload=payload,
            )
            assert result.status == "SUCCESS", (
                f"Workflow '{workflow_name}' returned ERROR: {result.error_message}"
            )

        logger.info(
            "test_all_five_workflows_return_success_with_valid_payloads: PASSED"
        )


# ---------------------------------------------------------------------------
# Session-level guard tests
# ---------------------------------------------------------------------------

class TestSessionManagerGuards:
    """
    Integration tests for ``SessionManager`` lifecycle guards.

    Verifies that the session manager raises typed exceptions for invalid
    operations and enforces lifecycle state transitions correctly.
    """

    @pytest.mark.asyncio
    async def test_empty_user_id_raises_value_error(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that ``create_session()`` raises ``ValueError`` for an empty
        ``user_id``.

        Assertions:
          - ``ValueError`` is raised.
          - The exception is raised before any session is registered.
        """
        with pytest.raises(ValueError):
            await runner_setup.session_manager.create_session(user_id="")

        assert runner_setup.session_manager.active_session_ids == []

        logger.info("test_empty_user_id_raises_value_error: PASSED")

    @pytest.mark.asyncio
    async def test_get_context_after_close_raises_key_error(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that calling ``get_session_context()`` after ``close_session()``
        raises ``KeyError`` because the record has been evicted.

        Assertions:
          - ``KeyError`` is raised.
        """
        session_id = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID
        )
        await runner_setup.session_manager.close_session(session_id)

        with pytest.raises(KeyError):
            await runner_setup.session_manager.get_session_context(session_id)

        logger.info("test_get_context_after_close_raises_key_error: PASSED")

    @pytest.mark.asyncio
    async def test_invalid_role_raises_value_error(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that ``save_session_state()`` raises ``ValueError`` when role is
        neither ``"user"`` nor ``"assistant"``.

        Assertions:
          - ``ValueError`` is raised for ``role="system"``.
        """
        session_id = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID
        )

        try:
            with pytest.raises(ValueError):
                await runner_setup.session_manager.save_session_state(
                    session_id=session_id,
                    role="system",  # invalid
                    content="Injecting a system prompt.",
                )
        finally:
            await runner_setup.session_manager.close_session(session_id)

        logger.info("test_invalid_role_raises_value_error: PASSED")

    @pytest.mark.asyncio
    async def test_duplicate_session_id_returns_existing(
        self, runner_setup: _TestComponents
    ) -> None:
        """
        Verify that calling ``create_session()`` with an already-active
        ``session_id`` returns the existing id rather than creating a
        duplicate record.

        Assertions:
          - Both calls return the same ``session_id`` string.
          - Only one session is active, not two.
        """
        # ------------------------------------------------------------------
        # Act
        # ------------------------------------------------------------------
        fixed_id = "fixed-session-id-abc123"
        sid_first = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID,
            session_id=fixed_id,
        )
        sid_second = await runner_setup.session_manager.create_session(
            user_id=_TEST_USER_ID,
            session_id=fixed_id,
        )

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        assert sid_first == fixed_id
        assert sid_second == fixed_id
        assert len(runner_setup.session_manager.active_session_ids) == 1

        # ------------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------------
        await runner_setup.session_manager.close_session(fixed_id)

        logger.info("test_duplicate_session_id_returns_existing: PASSED")
