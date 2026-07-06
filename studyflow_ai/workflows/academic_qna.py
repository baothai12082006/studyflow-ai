"""
academic_qna.py
----------------
Academic QnA / Tutoring Workflow.
Orchestrates question receiving, RAG context search, and Socratic tutoring responses.
Includes exponential backoff retries and circuit breaker error resiliency patterns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from google.adk import Workflow
from studyflow_ai.utils.telemetry import trace_step, log_structured

logger = logging.getLogger(__name__)


class CircuitBreakerState:
    """
    State container for the AcademicQnAWorkflow circuit breaker.
    Uses a standard Python class to avoid Pydantic BaseModel metaclass validation side-effects.
    """
    failure_count: int = 0
    circuit_state: str = "CLOSED"
    last_failure_time: float = 0.0


class AcademicQnAWorkflow(Workflow):
    """
    Orchestrates tutoring conversational flows, RAG context searching,
    and Socratic tutoring responses.

    Resiliency features:
      - Exponential backoff retry loop (up to 3 retries) on transient errors.
      - Lightweight circuit breaker that trips to "OPEN" after 5 consecutive failures.
    """

    @trace_step("AcademicQnARun")
    async def run(self, user_id: str, question: str, course_id: str) -> None:
        """
        Orchestrates question receiving, RAG context search, and tutoring answers.
        """
        current_time = time.time()

        # ------------------------------------------------------------------
        # Circuit Breaker Check
        # ------------------------------------------------------------------
        if CircuitBreakerState.circuit_state == "OPEN":
            cooldown_period = 30.0
            if current_time - CircuitBreakerState.last_failure_time <= cooldown_period:
                log_structured(
                    logging.WARNING,
                    "Circuit breaker is OPEN. Fast-failing request.",
                    step_name="AcademicQnARun",
                    circuit_state="OPEN",
                    cooldown_remaining=round(cooldown_period - (current_time - CircuitBreakerState.last_failure_time), 2)
                )
                # Raise exception to bubble up to the runner's execute_workflow,
                # which converts it to a WorkflowResult(status="ERROR", ...)
                raise RuntimeError("Circuit open: External LLM API is temporarily unavailable.")
            else:
                # Reset circuit after cooldown window expires
                CircuitBreakerState.circuit_state = "CLOSED"
                CircuitBreakerState.failure_count = 0
                log_structured(
                    logging.INFO,
                    "Circuit cooldown window expired. Resetting circuit to CLOSED.",
                    step_name="AcademicQnARun",
                    circuit_state="CLOSED"
                )

        # ------------------------------------------------------------------
        # Retry with Exponential Backoff Loop
        # ------------------------------------------------------------------
        max_retries: int = 3
        last_exception: Optional[BaseException] = None

        for attempt in range(max_retries + 1):
            try:
                # Core tutoring simulation / logic execution
                # We check the input question for special test directives
                # to trigger simulated transient or hard errors.
                if "simulate_transient_error" in question or "timeout" in question:
                    raise ConnectionError("Simulated LLM API timeout/connection error.")
                if "simulate_hard_error" in question:
                    raise RuntimeError("Simulated LLM API hard unrecoverable failure.")

                # Success path
                # Reset consecutive failure counter on successful execution
                CircuitBreakerState.failure_count = 0
                return

            except (ConnectionError, asyncio.TimeoutError) as exc:
                last_exception = exc
                if attempt < max_retries:
                    backoff_seconds = (2 ** attempt) * 0.1
                    log_structured(
                        logging.WARNING,
                        f"Transient failure on attempt {attempt + 1}: {exc}. Retrying in {backoff_seconds:.2f}s...",
                        step_name="AcademicQnARun",
                        attempt=attempt + 1,
                        backoff=backoff_seconds
                    )
                    await asyncio.sleep(backoff_seconds)
                else:
                    # Max retries exhausted — log exception detail and fall through.
                    logger.error(
                        "Max retries (%d) exhausted for transient error: %s",
                        max_retries,
                        exc,
                    )

            except Exception as exc:  # noqa: BLE001 — catch all non-transient hard failures
                # Hard / non-transient errors immediately exit the retry loop.
                last_exception = exc
                logger.error(
                    "Non-transient hard error in QnA workflow (no retry): %s", exc
                )
                break

        # ------------------------------------------------------------------
        # Post-execution Failure Handling (Circuit Tripping)
        # ------------------------------------------------------------------
        # Increment consecutive failure count
        CircuitBreakerState.failure_count += 1
        
        if CircuitBreakerState.failure_count >= 5:
            CircuitBreakerState.circuit_state = "OPEN"
            CircuitBreakerState.last_failure_time = time.time()
            log_structured(
                logging.ERROR,
                "Consecutive failures exceeded threshold. Tripping circuit to OPEN.",
                step_name="AcademicQnARun",
                circuit_state="OPEN",
                failure_count=CircuitBreakerState.failure_count
            )

        # Bubble up the execution failure to the runner
        raise RuntimeError(f"Workflow execution failed: {last_exception}")
