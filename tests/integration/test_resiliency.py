"""
test_resiliency.py
-------------------
Integration tests for Retry and Circuit Breaker resiliency patterns in AcademicQnAWorkflow.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch
import pytest

from studyflow_ai.studyflow_agent import StudyFlowAgent
from studyflow_ai.runner import StudyFlowRunner, WorkflowName, WorkflowResult
from studyflow_ai.repositories.in_memory import (
    InMemoryUserRepository,
    InMemoryAcademicRepository,
    InMemoryProgressRepository,
)
from studyflow_ai.services.calendar_service import CalendarService
from studyflow_ai.services.notification_service import NotificationService
from studyflow_ai.services.storage_service import StorageService
from studyflow_ai.services.vector_service import VectorService
from studyflow_ai.workflows.academic_qna import CircuitBreakerState


# ---------------------------------------------------------------------------
# Custom String Subclass to simulate transient errors twice
# ---------------------------------------------------------------------------

class TransientErrorSimulatorString(str):
    """
    Subclass of str that returns True for 'simulate_transient_error' exactly
    twice, forcing two ConnectionErrors before succeeding on the third attempt.
    """
    def __init__(self, value: str) -> None:
        super().__init__()
        self._failures_delivered = 0

    def __contains__(self, item: object) -> bool:
        if item == "simulate_transient_error":
            if self._failures_delivered < 2:
                self._failures_delivered += 1
                return True
            return False
        return super().__contains__(item)


# ---------------------------------------------------------------------------
# Test Setup Helper & Fixtures
# ---------------------------------------------------------------------------

def _setup_test_runner() -> StudyFlowRunner:
    user_repo = InMemoryUserRepository()
    academic_repo = InMemoryAcademicRepository()
    progress_repo = InMemoryProgressRepository()

    calendar_svc = CalendarService(client_id="test", client_secret="test")
    notification_svc = NotificationService(smtp_host="localhost", smtp_port=25)
    storage_svc = StorageService()
    vector_svc = VectorService(api_key="test", environment="test", index_name="test")

    agent = StudyFlowAgent()
    return StudyFlowRunner(
        agent=agent,
        user_repository=user_repo,
        academic_repository=academic_repo,
        progress_repository=progress_repo,
        calendar_service=calendar_svc,
        notification_service=notification_svc,
        storage_service=storage_svc,
        vector_service=vector_svc,
    )


@pytest.fixture(autouse=True)
def clean_circuit_breaker_state() -> None:
    """
    Autouse fixture to reset the global CircuitBreakerState before and after
    each test execution to prevent test pollution/leakage across suites.
    """
    # Setup
    CircuitBreakerState.circuit_state = "CLOSED"
    CircuitBreakerState.failure_count = 0
    CircuitBreakerState.last_failure_time = 0.0
    yield
    # Teardown
    CircuitBreakerState.circuit_state = "CLOSED"
    CircuitBreakerState.failure_count = 0
    CircuitBreakerState.last_failure_time = 0.0


# ---------------------------------------------------------------------------
# Resiliency Test Suite
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_academic_qna_retry_on_transient_error() -> None:
    """
    Verifies that the AcademicQnAWorkflow retries upon encountering transient
    errors, and succeeds if the error clears before max retries are exceeded.
    """
    runner = _setup_test_runner()

    # Pass the custom string subclass to simulate exactly 2 failures
    question = TransientErrorSimulatorString("Chemistry question")
    payload = {
        "user_id": "test-user-resiliency",
        "course_id": "chem-101",
        "question": question,
    }

    # Patch asyncio.sleep to keep execution speed fast
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await runner.execute_workflow(
            workflow_name=WorkflowName.ACADEMIC_QNA.value,
            payload=payload,
        )

        # Assert workflow succeeded on the 3rd attempt
        assert result.status == "SUCCESS"
        # Since 2 attempts failed, sleep should have been called twice (backoffs for attempt 0 and 1)
        assert mock_sleep.call_count == 2
        # Backoff delay values checks: first attempt backoff = 0.1s, second attempt backoff = 0.2s
        mock_sleep.assert_any_call(0.1)
        mock_sleep.assert_any_call(0.2)


@pytest.mark.asyncio
async def test_academic_qna_circuit_breaker_trips() -> None:
    """
    Verifies that 5 consecutive failures trips the circuit breaker to OPEN,
    and subsequent calls immediately return the fallback WorkflowResult without execution.
    """
    runner = _setup_test_runner()

    payload = {
        "user_id": "test-user-resiliency",
        "course_id": "chem-101",
        "question": "simulate_hard_error", # triggers a hard unrecoverable failure
    }

    # 1. Execute 5 consecutive hard failures
    for i in range(5):
        result = await runner.execute_workflow(
            workflow_name=WorkflowName.ACADEMIC_QNA.value,
            payload=payload,
        )
        assert result.status == "ERROR"
        assert "Simulated LLM API hard unrecoverable failure" in result.error_message

    # Verify circuit breaker transitioned to OPEN
    assert CircuitBreakerState.circuit_state == "OPEN"
    assert CircuitBreakerState.failure_count == 5
    assert CircuitBreakerState.last_failure_time > 0.0

    # 2. Execute 6th call with any normal question; should trip immediately
    normal_payload = {
        "user_id": "test-user-resiliency",
        "course_id": "chem-101",
        "question": "What is water made of?",
    }

    start_time = time.time()
    result_6th = await runner.execute_workflow(
        workflow_name=WorkflowName.ACADEMIC_QNA.value,
        payload=normal_payload,
    )
    end_time = time.time()

    # Assert fast-fail execution speed and exact circuit open fallback message
    assert result_6th.status == "ERROR"
    assert result_6th.error_message == "Circuit open: External LLM API is temporarily unavailable."
    # Fast fail should be near-instantaneous (under 50ms)
    assert (end_time - start_time) < 0.05
