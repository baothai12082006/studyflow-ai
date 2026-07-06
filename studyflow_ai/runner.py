"""
runner.py
---------
ADK Runner for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Act as the single async runtime execution engine for all system workflows.
  - Manage the lifecycle of every registered workflow state machine.
  - Receive a StudyFlowAgent instance and delegate execution through it.
  - Route payloads to the correct workflow via _map_workflow_state().
  - Contain no agent-specific inner business logic.

Registered workflows:
  - IngestSyllabusWorkflow
  - CreateStudyPlanWorkflow
  - UpdateStudyPlanWorkflow
  - AcademicQnAWorkflow
  - ProgressMonitorWorkflow

Future Compatibility (Documentation Only):
  - TODO(Session): Inject a session context object into every workflow execution
    call so agents can share turn-scoped state across async boundaries.
  - TODO(ArtifactService): Verify artifact availability before execution begins;
    fail fast with a descriptive WorkflowError when required artifacts are missing.
  - TODO(Observability): Emit structured telemetry events (start, success, error,
    latency) to a tracing backend (e.g. Cloud Trace / OpenTelemetry) around every
    ``execute_workflow`` call.

References:
  docs/design/multi_agent_architecture.md
  docs/design/execution_workflow.md
  docs/design/interface_contracts.md
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Workflow state machines
# ---------------------------------------------------------------------------
from studyflow_ai.workflows.ingest_syllabus import IngestSyllabusWorkflow
from studyflow_ai.workflows.create_study_plan import CreateStudyPlanWorkflow
from studyflow_ai.workflows.update_study_plan import UpdateStudyPlanWorkflow
from studyflow_ai.workflows.academic_qna import AcademicQnAWorkflow
from studyflow_ai.workflows.progress_monitor import ProgressMonitorWorkflow

# ---------------------------------------------------------------------------
# Root agent (entry point, provides access to every sub-agent)
# ---------------------------------------------------------------------------
from studyflow_ai.studyflow_agent import StudyFlowAgent

# ---------------------------------------------------------------------------
# Repository interfaces (injected, never instantiated here)
# ---------------------------------------------------------------------------
from studyflow_ai.repositories.base import (
    UserRepository,
    AcademicRepository,
    ProgressRepository,
)

# ---------------------------------------------------------------------------
# Service interfaces (injected, never instantiated here)
# ---------------------------------------------------------------------------
from studyflow_ai.services.calendar_service import CalendarService
from studyflow_ai.services.notification_service import NotificationService
from studyflow_ai.services.storage_service import StorageService
from studyflow_ai.services.vector_service import VectorService
from studyflow_ai.utils.telemetry import trace_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workflow name registry
# ---------------------------------------------------------------------------

class WorkflowName(str, Enum):
    """
    Canonical identifiers for every workflow registered with the ADK Runner.

    Each value must match the string passed to ``execute_workflow()`` by the
    caller so that ``_map_workflow_state()`` can route without ambiguity.
    """

    INGEST_SYLLABUS = "IngestSyllabusWorkflow"
    CREATE_STUDY_PLAN = "CreateStudyPlanWorkflow"
    UPDATE_STUDY_PLAN = "UpdateStudyPlanWorkflow"
    ACADEMIC_QNA = "AcademicQnAWorkflow"
    PROGRESS_MONITOR = "ProgressMonitorWorkflow"


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------

class WorkflowResult:
    """
    Lightweight result container returned by ``execute_workflow()``.

    Attributes:
        workflow_name: The canonical ``WorkflowName`` that was executed.
        status:        ``"SUCCESS"`` or ``"ERROR"``.
        payload:       Arbitrary dict of output data produced by the workflow.
        error_message: Human-readable error detail when ``status == "ERROR"``.
    """

    def __init__(
        self,
        workflow_name: WorkflowName,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self.workflow_name = workflow_name
        self.status = status
        self.payload = payload or {}
        self.error_message = error_message

    def __repr__(self) -> str:
        wf_name = self.workflow_name.value if self.workflow_name is not None else "<unknown>"
        return (
            f"WorkflowResult(workflow={wf_name!r}, "
            f"status={self.status!r}, error={self.error_message!r})"
        )


# ---------------------------------------------------------------------------
# StudyFlowRunner
# ---------------------------------------------------------------------------

class StudyFlowRunner:
    """
    Centralised async runtime execution engine for StudyFlow AI workflows.

    Design contract:
      - Stateless between calls; no shared mutable state is stored.
      - All workflow instances are constructed once in ``__init__()`` and
        reused across calls.
      - All repositories and services are injected; none are instantiated here.
      - Sub-agents are accessed exclusively through the injected
        ``StudyFlowAgent`` interface.

    Args:
        agent:                 The fully-wired ``StudyFlowAgent`` root agent.
        user_repository:       Persistence layer for ``UserState`` records.
        academic_repository:   Persistence layer for ``AcademicState`` records.
        progress_repository:   Persistence layer for ``ProgressState`` records.
        calendar_service:      External calendar integration (Google/Outlook).
        notification_service:  Push notification and SMTP dispatch service.
        storage_service:       Cloud file storage and document parser service.
        vector_service:        Vector DB embedding upsert and similarity search.
    """

    def __init__(
        self,
        agent: StudyFlowAgent,
        user_repository: UserRepository,
        academic_repository: AcademicRepository,
        progress_repository: ProgressRepository,
        calendar_service: CalendarService,
        notification_service: NotificationService,
        storage_service: StorageService,
        vector_service: VectorService,
    ) -> None:
        # ------------------------------------------------------------------
        # 1. Store injected dependencies
        # ------------------------------------------------------------------
        self._agent = agent
        self._user_repository = user_repository
        self._academic_repository = academic_repository
        self._progress_repository = progress_repository
        self._calendar_service = calendar_service
        self._notification_service = notification_service
        self._storage_service = storage_service
        self._vector_service = vector_service

        # ------------------------------------------------------------------
        # 2. Register and configure workflow instances
        # ------------------------------------------------------------------
        self._workflows: Dict[WorkflowName, Any] = self._register_workflows()

        logger.info(
            "StudyFlowRunner initialised | workflows=%s | root_agent=%s",
            [wf.value for wf in self._workflows],
            self._agent.adk_agent().name,
        )

    # ------------------------------------------------------------------
    # Workflow registration
    # ------------------------------------------------------------------

    def _register_workflows(self) -> Dict[WorkflowName, Any]:
        """
        Instantiate every registered workflow and return the name→instance map.

        Each workflow receives the runner's injected dependencies so that it
        can interact with the correct repositories and services at execution
        time without accessing them directly from agent code.

        Returns:
            A dict mapping each ``WorkflowName`` to its workflow instance.
        """
        logger.debug("Registering workflow state machines...")

        workflows: Dict[WorkflowName, Any] = {
            WorkflowName.INGEST_SYLLABUS: IngestSyllabusWorkflow(name="ingest_syllabus"),
            WorkflowName.CREATE_STUDY_PLAN: CreateStudyPlanWorkflow(name="create_study_plan"),
            WorkflowName.UPDATE_STUDY_PLAN: UpdateStudyPlanWorkflow(name="update_study_plan"),
            WorkflowName.ACADEMIC_QNA: AcademicQnAWorkflow(name="academic_qna"),
            WorkflowName.PROGRESS_MONITOR: ProgressMonitorWorkflow(name="progress_monitor"),
        }


        logger.debug(
            "Registered %d workflow(s): %s",
            len(workflows),
            [wf.value for wf in workflows],
        )
        return workflows

    # ------------------------------------------------------------------
    # Public execution gateway
    # ------------------------------------------------------------------

    @trace_step("ExecuteWorkflow")
    async def execute_workflow(
        self,
        workflow_name: str,
        payload: Dict[str, Any],
    ) -> WorkflowResult:
        """
        Centralised async execution gateway for all StudyFlow workflows.

        Workflow execution steps:
          1. Resolve the ``WorkflowName`` enum from the caller-supplied string.
          2. Look up the registered workflow instance.
          3. Prepare the execution context from the injected payload.
          4. Dispatch execution to the workflow's ``run()`` coroutine.
          5. Wrap the result in a ``WorkflowResult`` and return it.

        Args:
            workflow_name: One of the ``WorkflowName`` string values (e.g.
                           ``"IngestSyllabusWorkflow"``).
            payload:       Arbitrary key-value dict of parameters required by
                           the target workflow.  Schema is validated inside the
                           workflow's own ``run()`` method.

        Returns:
            A ``WorkflowResult`` describing the outcome of the execution.

        Raises:
            ValueError: If ``workflow_name`` does not match any registered
                        workflow.

        # TODO(Session): Wrap this call with a session context object so that
        # the ADK agent can share turn-scoped state across sub-agent boundaries.
        # TODO(ArtifactService): Before dispatching, verify that any artifact
        # URIs present in ``payload`` exist in the ArtifactService store.
        # TODO(Observability): Emit a span start event here and a span end event
        # in the finally block, tagging with workflow_name, user_id, and status.
        """
        logger.info(
            "Executing workflow | name=%s | payload_keys=%s",
            workflow_name,
            list(payload.keys()),
        )

        try:
            # ------------------------------------------------------------------
            # Step 1 – resolve and validate the workflow name
            # ------------------------------------------------------------------
            resolved_name = self._resolve_workflow_name(workflow_name)

            # ------------------------------------------------------------------
            # Step 2 – look up the workflow instance
            # ------------------------------------------------------------------
            workflow_instance = self._map_workflow_state(resolved_name)

            # ------------------------------------------------------------------
            # Step 3 – build the execution context
            # ------------------------------------------------------------------
            context = self._build_execution_context(resolved_name, payload)

            # ------------------------------------------------------------------
            # Step 4 – dispatch execution
            # ------------------------------------------------------------------
            await workflow_instance.run(**context)

            logger.info(
                "Workflow completed successfully | name=%s",
                resolved_name.value,
            )
            return WorkflowResult(
                workflow_name=resolved_name,
                status="SUCCESS",
                payload=context,
            )

        except ValueError as exc:
            logger.warning(
                "Workflow validation failed | name=%s | error=%s",
                workflow_name,
                exc,
            )
            # Try to resolve to WorkflowName if possible for the result, otherwise use None
            try:
                resolved_name_fallback = self._resolve_workflow_name(workflow_name)
            except Exception:
                resolved_name_fallback = None

            return WorkflowResult(
                workflow_name=resolved_name_fallback,
                status="ERROR",
                error_message=str(exc),
            )

        except Exception as exc:  # noqa: BLE001 – surface all failures uniformly
            logger.exception(
                "Workflow execution failed | name=%s | error=%s",
                workflow_name,
                exc,
            )
            try:
                resolved_name_fallback = self._resolve_workflow_name(workflow_name)
            except Exception:
                resolved_name_fallback = None

            return WorkflowResult(
                workflow_name=resolved_name_fallback,
                status="ERROR",
                error_message=str(exc),
            )


    # ------------------------------------------------------------------
    # Workflow routing
    # ------------------------------------------------------------------

    def _map_workflow_state(self, workflow_name: WorkflowName) -> Any:
        """
        Resolve a ``WorkflowName`` to the corresponding workflow instance.

        This is the central routing table for the ADK Runner.  Every workflow
        registered in ``_register_workflows()`` is reachable through this
        method.

        Args:
            workflow_name: A validated ``WorkflowName`` enum value.

        Returns:
            The workflow instance that owns the ``run()`` coroutine to execute.

        Raises:
            KeyError: If the workflow name is not present in the registry.
                      This should never occur if ``_resolve_workflow_name()``
                      is called first.
        """
        workflow_instance = self._workflows.get(workflow_name)

        if workflow_instance is None:
            raise KeyError(
                f"No workflow registered for name '{workflow_name.value}'. "
                f"Available: {[wf.value for wf in self._workflows]}"
            )

        logger.debug(
            "Mapped workflow name '%s' → %s",
            workflow_name.value,
            type(workflow_instance).__name__,
        )
        return workflow_instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_workflow_name(self, raw_name: str) -> WorkflowName:
        """
        Convert a raw string into a validated ``WorkflowName`` enum value.

        Args:
            raw_name: The caller-supplied workflow identifier string.

        Returns:
            The matching ``WorkflowName`` enum member.

        Raises:
            ValueError: If ``raw_name`` does not match any registered workflow.
        """
        try:
            return WorkflowName(raw_name)
        except ValueError:
            valid_names = [wf.value for wf in WorkflowName]
            raise ValueError(
                f"Unknown workflow name '{raw_name}'. "
                f"Valid options are: {valid_names}"
            )

    def _build_execution_context(
        self,
        workflow_name: WorkflowName,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construct the keyword-argument dict passed to the workflow's ``run()``
        coroutine.

        Each workflow expects a specific set of keyword arguments that map to
        the parameters of its ``run()`` method.  This helper extracts only the
        relevant keys from the caller-supplied payload and validates that
        mandatory keys are present.

        Args:
            workflow_name: The resolved ``WorkflowName`` determining which keys
                           to extract.
            payload:       The caller-supplied parameter dict.

        Returns:
            A filtered dict of keyword arguments ready to be unpacked into the
            workflow's ``run()`` coroutine.

        Raises:
            ValueError: If a required parameter is absent from ``payload``.
        """
        # Per-workflow parameter schemas (required keys only).
        # Optional keys are passed through transparently if present.
        required_keys: Dict[WorkflowName, list[str]] = {
            WorkflowName.INGEST_SYLLABUS: ["user_id", "file_uri", "course_id"],
            WorkflowName.CREATE_STUDY_PLAN: ["user_id", "course_id"],
            WorkflowName.UPDATE_STUDY_PLAN: ["user_id", "course_id"],
            WorkflowName.ACADEMIC_QNA: ["user_id", "question", "course_id"],
            WorkflowName.PROGRESS_MONITOR: ["user_id"],
        }

        required = required_keys.get(workflow_name, [])
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                f"Missing required payload keys for workflow "
                f"'{workflow_name.value}': {missing}"
            )

        # Extract exactly the keys the workflow ``run()`` signature declares.
        context = {key: payload[key] for key in required}

        logger.debug(
            "Built execution context | workflow=%s | context_keys=%s",
            workflow_name.value,
            list(context.keys()),
        )
        return context

    # ------------------------------------------------------------------
    # Resource lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Gracefully tear down all runner-owned resources.

        Performs the following cleanup steps in order:
          1. Flushes any pending in-memory state from registered repositories.
          2. Clears the workflow registry to release workflow instances.
          3. Logs a completion notice.

        This method is safe to call multiple times; all individual cleanup
        steps are wrapped in isolated try/except blocks so that a failure
        in one step never prevents the remaining steps from executing.

        Usage::

            runner = StudyFlowRunner(...)
            try:
                await runner.execute_workflow(...)
            finally:
                await runner.close()
        """
        logger.info("StudyFlowRunner.close() — beginning resource cleanup.")

        # Step 1: Flush repository caches if the repository exposes a flush interface.
        repositories: List[Any] = [
            self._user_repository,
            self._academic_repository,
            self._progress_repository,
        ]
        for repo in repositories:
            try:
                flush = getattr(repo, "flush", None)
                if callable(flush):
                    result = flush()
                    # Support both sync and async flush implementations.
                    if hasattr(result, "__await__"):
                        await result
                    logger.debug(
                        "Flushed repository: %s", type(repo).__name__
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Non-fatal error flushing repository %s: %s",
                    type(repo).__name__,
                    exc,
                )

        # Step 2: Clear the in-memory workflow registry.
        try:
            self._workflows.clear()
            logger.debug("Workflow registry cleared.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Non-fatal error clearing workflow registry: %s", exc)

        logger.info("StudyFlowRunner.close() — resource cleanup complete.")

    # ------------------------------------------------------------------
    # Properties (read-only access to injected dependencies)
    # ------------------------------------------------------------------

    @property
    def agent(self) -> StudyFlowAgent:
        """The root StudyFlowAgent instance owned by this runner."""
        return self._agent

    @property
    def user_repository(self) -> UserRepository:
        """Injected UserRepository for user-state persistence."""
        return self._user_repository

    @property
    def academic_repository(self) -> AcademicRepository:
        """Injected AcademicRepository for course-state persistence."""
        return self._academic_repository

    @property
    def progress_repository(self) -> ProgressRepository:
        """Injected ProgressRepository for progress-state persistence."""
        return self._progress_repository

    @property
    def calendar_service(self) -> CalendarService:
        """Injected CalendarService for external calendar integration."""
        return self._calendar_service

    @property
    def notification_service(self) -> NotificationService:
        """Injected NotificationService for push and email dispatch."""
        return self._notification_service

    @property
    def storage_service(self) -> StorageService:
        """Injected StorageService for cloud file storage and parsing."""
        return self._storage_service

    @property
    def vector_service(self) -> VectorService:
        """Injected VectorService for embedding upsert and similarity search."""
        return self._vector_service
