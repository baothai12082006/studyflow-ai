"""
cli.py
------
Command-Line Interface (CLI) Entrypoint for StudyFlow AI.

Responsibilities (per Architecture doc):
  - Bridge terminal user inputs to the StudyFlowRunner execution pipeline.
  - Parse subcommand arguments cleanly via argparse.
  - Assemble raw Python dict payloads from parsed arguments.
  - Open a SessionManager session for the invocation lifetime.
  - Invoke StudyFlowRunner.execute_workflow() asynchronously.
  - Display a formatted WorkflowResult status report in the terminal.
  - Tear down the session cleanly after execution.

Supported subcommands (maps to WorkflowName values):
  ingest    →  IngestSyllabusWorkflow
  create    →  CreateStudyPlanWorkflow
  update    →  UpdateStudyPlanWorkflow
  qna       →  AcademicQnAWorkflow
  monitor   →  ProgressMonitorWorkflow

Future Compatibility (Documentation Only):
  - TODO(Interactive): Add an ``--interactive`` flag that spawns a continuous
    REPL chat loop powered by the CoordinatorAgent, feeding each user input as a
    QNA workflow invocation and printing the reply in-place.
  - TODO(Rich): Replace all ``print()`` output calls with ``rich.console.Console``
    styled output (panels, status spinners, syntax-highlighted JSON) once the
    ``rich`` library is added to the project dependencies.
  - TODO(ShellCompletion): Register argparse completion scripts via
    ``argcomplete`` so that subcommands and flags complete on <Tab> in bash/zsh.

References:
  docs/design/multi_agent_architecture.md
  docs/design/execution_workflow.md
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import textwrap
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Core runtime components
# ---------------------------------------------------------------------------
from studyflow_ai.studyflow_agent import StudyFlowAgent
from studyflow_ai.runner import StudyFlowRunner, WorkflowName, WorkflowResult
from studyflow_ai.session import SessionManager

# ---------------------------------------------------------------------------
# In-memory repository implementations (used when no persistent DB is wired)
# ---------------------------------------------------------------------------
from studyflow_ai.repositories.in_memory import (
    InMemoryUserRepository,
    InMemoryAcademicRepository,
    InMemoryProgressRepository,
)

# ---------------------------------------------------------------------------
# Service stubs (placeholder until real credentials are configured)
# ---------------------------------------------------------------------------
from studyflow_ai.services.calendar_service import CalendarService
from studyflow_ai.services.notification_service import NotificationService
from studyflow_ai.services.storage_service import StorageService
from studyflow_ai.services.vector_service import VectorService

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from studyflow_ai.config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

# Section separator width used throughout print_* helpers.
_SEPARATOR_WIDTH: int = 60
_SEPARATOR_CHAR: str = "─"


def _print_banner() -> None:
    """
    Print the StudyFlow AI CLI banner to stdout.

    Called once at the start of every CLI invocation so the user always knows
    which system they are interacting with.
    """
    print()
    print("=" * _SEPARATOR_WIDTH)
    print("  StudyFlow AI  │  Command-Line Interface")
    print("=" * _SEPARATOR_WIDTH)
    print()


def _print_section(title: str) -> None:
    """
    Print a labelled section separator to stdout.

    Args:
        title: Short label rendered inline with the separator line.
    """
    line = f" {title} ".center(_SEPARATOR_WIDTH, _SEPARATOR_CHAR)
    print(line)


def _print_result(result: WorkflowResult) -> None:
    """
    Render a ``WorkflowResult`` as a formatted terminal block.

    Prints:
      - Workflow name
      - Status (SUCCESS / ERROR)
      - Payload key-value pairs (for SUCCESS)
      - Error message (for ERROR)

    Args:
        result: The ``WorkflowResult`` returned by ``StudyFlowRunner.execute_workflow()``.
    """
    print()
    _print_section("Workflow Result")
    print(f"  Workflow  : {result.workflow_name.value}")
    print(f"  Status    : {result.status}")

    if result.status == "SUCCESS":
        if result.payload:
            print("  Payload   :")
            for key, value in result.payload.items():
                print(f"    {key}: {value}")
        else:
            print("  Payload   : (empty)")
    else:
        wrapped_error = textwrap.fill(
            result.error_message or "(no detail)",
            width=_SEPARATOR_WIDTH - 14,
            subsequent_indent=" " * 14,
        )
        print(f"  Error     : {wrapped_error}")

    print(_SEPARATOR_CHAR * _SEPARATOR_WIDTH)
    print()


def _print_error(message: str) -> None:
    """
    Print a formatted error message to stderr and exit with code 1.

    Args:
        message: Human-readable description of what went wrong.
    """
    print(f"\n[ERROR] {message}", file=sys.stderr)
    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# Dependency factory
# ---------------------------------------------------------------------------

def _build_runner() -> tuple[StudyFlowRunner, SessionManager]:
    """
    Construct and wire all runtime dependencies needed by the CLI.

    Uses in-memory repositories and stub service implementations so that the
    CLI works out-of-the-box without live database or API credentials.
    Real implementations can be swapped in by replacing the repository and
    service instantiations below.

    Returns:
        A ``(StudyFlowRunner, SessionManager)`` tuple ready for use.
    """
    # Repositories
    user_repo = InMemoryUserRepository()
    academic_repo = InMemoryAcademicRepository()
    progress_repo = InMemoryProgressRepository()

    # Service stubs — credentials pulled from settings (defaults to empty/stub)
    calendar_svc = CalendarService(
        client_id=settings.google_calendar_client_id or "",
        client_secret=settings.google_calendar_client_secret or "",
    )
    notification_svc = NotificationService(
        smtp_host=settings.smtp_host,
        smtp_port=settings.smtp_port,
    )
    storage_svc = StorageService()
    vector_svc = VectorService(
        api_key=settings.pinecone_api_key or "",
        environment=settings.pinecone_environment,
        index_name=settings.pinecone_index_name,
    )

    # Root agent
    agent = StudyFlowAgent()

    # Runner
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

    # Session manager
    session_manager = SessionManager(
        user_repository=user_repo,
        academic_repository=academic_repo,
        progress_repository=progress_repo,
    )

    logger.debug("CLI dependencies wired (in-memory repositories, stub services).")
    return runner, session_manager


# ---------------------------------------------------------------------------
# Async execution core
# ---------------------------------------------------------------------------

async def _run_workflow(
    runner: StudyFlowRunner,
    session_manager: SessionManager,
    workflow_name: str,
    user_id: str,
    payload: Dict[str, Any],
) -> WorkflowResult:
    """
    Open a session, execute the target workflow, and close the session cleanly.

    Steps:
      1. Open a ``SessionManager`` session for the invocation lifetime.
      2. Invoke ``StudyFlowRunner.execute_workflow()`` with the assembled payload.
      3. Save a minimal session state entry recording the CLI invocation.
      4. Close the session and return the result.

    Args:
        runner:         The fully-wired ``StudyFlowRunner`` instance.
        session_manager: The ``SessionManager`` managing this invocation's state.
        workflow_name:  The canonical workflow name string (a ``WorkflowName`` value).
        user_id:        The authenticated student UUID for this invocation.
        payload:        Validated dict of keyword arguments for the target workflow.

    Returns:
        The ``WorkflowResult`` from ``execute_workflow()``.
    """
    # ------------------------------------------------------------------
    # Step 1 – open a session for this CLI invocation
    # ------------------------------------------------------------------
    session_id = await session_manager.create_session(user_id=user_id)
    logger.info(
        "CLI session opened | session_id=%s workflow=%s",
        session_id,
        workflow_name,
    )

    try:
        # ------------------------------------------------------------------
        # Step 2 – execute the workflow
        # ------------------------------------------------------------------
        result = await runner.execute_workflow(
            workflow_name=workflow_name,
            payload=payload,
        )

        # ------------------------------------------------------------------
        # Step 3 – record the invocation in session state
        # ------------------------------------------------------------------
        await session_manager.save_session_state(
            session_id=session_id,
            role="user",
            content=f"CLI invocation: {workflow_name} | payload_keys={list(payload.keys())}",
            active_intent=_infer_intent(workflow_name),
        )
        await session_manager.save_session_state(
            session_id=session_id,
            role="assistant",
            content=f"Workflow completed with status: {result.status}",
        )

        return result

    finally:
        # ------------------------------------------------------------------
        # Step 4 – always close the session to flush state
        # ------------------------------------------------------------------
        await session_manager.close_session(session_id)
        logger.info("CLI session closed | session_id=%s", session_id)


# ---------------------------------------------------------------------------
# Payload builders (one per subcommand)
# ---------------------------------------------------------------------------

def _build_ingest_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Assemble the payload dict for ``IngestSyllabusWorkflow``.

    Args:
        args: Parsed argparse namespace; must contain ``user_id``,
              ``file_uri``, and ``course_id``.

    Returns:
        Dict matching the ``IngestSyllabusWorkflow.run()`` signature.
    """
    return {
        "user_id": args.user_id,
        "file_uri": args.file_uri,
        "course_id": args.course_id,
    }


def _build_create_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Assemble the payload dict for ``CreateStudyPlanWorkflow``.

    Args:
        args: Parsed argparse namespace; must contain ``user_id`` and
              ``course_id``.

    Returns:
        Dict matching the ``CreateStudyPlanWorkflow.run()`` signature.
    """
    return {
        "user_id": args.user_id,
        "course_id": args.course_id,
    }


def _build_update_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Assemble the payload dict for ``UpdateStudyPlanWorkflow``.

    Args:
        args: Parsed argparse namespace; must contain ``user_id`` and
              ``course_id``.

    Returns:
        Dict matching the ``UpdateStudyPlanWorkflow.run()`` signature.
    """
    return {
        "user_id": args.user_id,
        "course_id": args.course_id,
    }


def _build_qna_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Assemble the payload dict for ``AcademicQnAWorkflow``.

    Args:
        args: Parsed argparse namespace; must contain ``user_id``,
              ``question``, and ``course_id``.

    Returns:
        Dict matching the ``AcademicQnAWorkflow.run()`` signature.
    """
    return {
        "user_id": args.user_id,
        "question": args.question,
        "course_id": args.course_id,
    }


def _build_monitor_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Assemble the payload dict for ``ProgressMonitorWorkflow``.

    Args:
        args: Parsed argparse namespace; must contain ``user_id``.

    Returns:
        Dict matching the ``ProgressMonitorWorkflow.run()`` signature.
    """
    return {
        "user_id": args.user_id,
    }


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the top-level argparse parser with all subcommands registered.

    Subcommand map:
      ingest   →  IngestSyllabusWorkflow   (--user-id, --file-uri, --course-id)
      create   →  CreateStudyPlanWorkflow  (--user-id, --course-id)
      update   →  UpdateStudyPlanWorkflow  (--user-id, --course-id)
      qna      →  AcademicQnAWorkflow      (--user-id, --course-id, --question)
      monitor  →  ProgressMonitorWorkflow  (--user-id)

    Returns:
        A fully configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog="studyflow",
        description=(
            "StudyFlow AI — Command-Line Interface\n"
            "Invoke academic workflows directly from your terminal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging verbosity level (default: WARNING).",
    )

    subparsers = parser.add_subparsers(
        dest="subcommand",
        metavar="<subcommand>",
        help="Workflow to execute.",
    )
    subparsers.required = True

    # ------------------------------------------------------------------
    # ingest — IngestSyllabusWorkflow
    # ------------------------------------------------------------------
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Parse and index an uploaded syllabus document.",
        description="Triggers IngestSyllabusWorkflow: upload → parse → vector-index.",
    )
    ingest_parser.add_argument(
        "--user-id",
        required=True,
        metavar="USER_ID",
        help="Authenticated student UUID.",
    )
    ingest_parser.add_argument(
        "--file-uri",
        required=True,
        metavar="FILE_URI",
        help="Local path or cloud URI of the syllabus document (e.g. gs://bucket/syllabus.pdf).",
    )
    ingest_parser.add_argument(
        "--course-id",
        required=True,
        metavar="COURSE_ID",
        help="Destination course UUID to associate the document with.",
    )

    # ------------------------------------------------------------------
    # create — CreateStudyPlanWorkflow
    # ------------------------------------------------------------------
    create_parser = subparsers.add_parser(
        "create",
        help="Generate a personalised study plan for a course.",
        description="Triggers CreateStudyPlanWorkflow: evaluate calendar → prioritise → sync.",
    )
    create_parser.add_argument(
        "--user-id",
        required=True,
        metavar="USER_ID",
        help="Authenticated student UUID.",
    )
    create_parser.add_argument(
        "--course-id",
        required=True,
        metavar="COURSE_ID",
        help="Course UUID to generate the study plan for.",
    )

    # ------------------------------------------------------------------
    # update — UpdateStudyPlanWorkflow
    # ------------------------------------------------------------------
    update_parser = subparsers.add_parser(
        "update",
        help="Reschedule and update an existing study plan.",
        description="Triggers UpdateStudyPlanWorkflow: recalculate → patch calendar.",
    )
    update_parser.add_argument(
        "--user-id",
        required=True,
        metavar="USER_ID",
        help="Authenticated student UUID.",
    )
    update_parser.add_argument(
        "--course-id",
        required=True,
        metavar="COURSE_ID",
        help="Course UUID whose plan should be updated.",
    )

    # ------------------------------------------------------------------
    # qna — AcademicQnAWorkflow
    # ------------------------------------------------------------------
    qna_parser = subparsers.add_parser(
        "qna",
        help="Ask a grounded academic question about a course.",
        description="Triggers AcademicQnAWorkflow: RAG search → Socratic answer.",
    )
    qna_parser.add_argument(
        "--user-id",
        required=True,
        metavar="USER_ID",
        help="Authenticated student UUID.",
    )
    qna_parser.add_argument(
        "--course-id",
        required=True,
        metavar="COURSE_ID",
        help="Course UUID to restrict the RAG search context to.",
    )
    qna_parser.add_argument(
        "--question",
        required=True,
        metavar="QUESTION",
        help="The academic question to answer (wrap in quotes if multi-word).",
    )

    # ------------------------------------------------------------------
    # monitor — ProgressMonitorWorkflow
    # ------------------------------------------------------------------
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Run a progress and streak check for a student.",
        description="Triggers ProgressMonitorWorkflow: streak analysis → risk detection → notify.",
    )
    monitor_parser.add_argument(
        "--user-id",
        required=True,
        metavar="USER_ID",
        help="Authenticated student UUID.",
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------

_SUBCOMMAND_MAP: Dict[str, tuple[str, Any]] = {
    "ingest":   (WorkflowName.INGEST_SYLLABUS.value,   _build_ingest_payload),
    "create":   (WorkflowName.CREATE_STUDY_PLAN.value,  _build_create_payload),
    "update":   (WorkflowName.UPDATE_STUDY_PLAN.value,  _build_update_payload),
    "qna":      (WorkflowName.ACADEMIC_QNA.value,       _build_qna_payload),
    "monitor":  (WorkflowName.PROGRESS_MONITOR.value,   _build_monitor_payload),
}
"""
Routing table mapping each CLI subcommand string to its
(WorkflowName value, payload-builder callable) pair.
"""


def _infer_intent(workflow_name: str) -> str:
    """
    Map a ``WorkflowName`` value string to a ``ConversationState`` intent label.

    Args:
        workflow_name: The canonical workflow name string.

    Returns:
        A CoordinatorAgent intent constant string.
    """
    _intent_map: Dict[str, str] = {
        WorkflowName.INGEST_SYLLABUS.value:   "UPLOAD",
        WorkflowName.CREATE_STUDY_PLAN.value:  "PLAN",
        WorkflowName.UPDATE_STUDY_PLAN.value:  "PLAN",
        WorkflowName.ACADEMIC_QNA.value:       "QNA",
        WorkflowName.PROGRESS_MONITOR.value:   "PROGRESS",
    }
    return _intent_map.get(workflow_name, "NONE")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Primary CLI entry point for StudyFlow AI.

    Execution flow:
      1. Print the banner.
      2. Parse subcommand and arguments via argparse.
      3. Configure logging at the requested verbosity.
      4. Wire all runtime dependencies (agent, runner, session manager).
      5. Resolve the target workflow name and assemble the payload dict.
      6. Execute the workflow asynchronously via ``asyncio.run()``.
      7. Print the ``WorkflowResult`` to the terminal.
      8. Exit with code 0 on SUCCESS or code 1 on ERROR.

    Exit codes:
      0  — Workflow completed with status ``SUCCESS``.
      1  — Workflow completed with status ``ERROR``, argument validation
           failed, or an unhandled exception was raised.
    """
    _print_banner()

    # ------------------------------------------------------------------
    # Step 1 – parse arguments
    # ------------------------------------------------------------------
    parser = _build_argument_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 2 – configure logging
    # ------------------------------------------------------------------
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logger.debug("Log level set to %s.", args.log_level)

    # ------------------------------------------------------------------
    # Step 3 – resolve subcommand routing
    # ------------------------------------------------------------------
    subcommand = args.subcommand
    if subcommand not in _SUBCOMMAND_MAP:
        _print_error(f"Unrecognised subcommand '{subcommand}'.")
        parser.print_help()
        sys.exit(1)

    workflow_name, payload_builder = _SUBCOMMAND_MAP[subcommand]

    # ------------------------------------------------------------------
    # Step 4 – build payload
    # ------------------------------------------------------------------
    try:
        payload = payload_builder(args)
    except AttributeError as exc:
        _print_error(f"Missing required argument: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5 – print pre-execution summary
    # ------------------------------------------------------------------
    _print_section("Execution Plan")
    print(f"  Subcommand : {subcommand}")
    print(f"  Workflow   : {workflow_name}")
    print(f"  User ID    : {payload.get('user_id', '—')}")
    for key, value in payload.items():
        if key != "user_id":
            print(f"  {key.replace('_', ' ').title():<11}: {value}")
    print()

    # ------------------------------------------------------------------
    # Step 6 – wire dependencies
    # ------------------------------------------------------------------
    try:
        runner, session_manager = _build_runner()
    except Exception as exc:
        _print_error(f"Failed to initialise the runtime: {exc}")
        logger.exception("Dependency wiring failed.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 7 – execute the workflow asynchronously
    # ------------------------------------------------------------------
    _print_section("Running")
    print(f"  Invoking {workflow_name} ...\n")

    try:
        result: WorkflowResult = asyncio.run(
            _run_workflow(
                runner=runner,
                session_manager=session_manager,
                workflow_name=workflow_name,
                user_id=payload["user_id"],
                payload=payload,
            )
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Execution cancelled by the user.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        _print_error(f"Unhandled runtime error: {exc}")
        logger.exception("Unhandled exception during workflow execution.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 8 – display result and exit
    # ------------------------------------------------------------------
    _print_result(result)

    exit_code = 0 if result.status == "SUCCESS" else 1
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Module entrypoint guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
