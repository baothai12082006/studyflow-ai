"""
telemetry.py
-------------
Structured Logging and Performance Tracing Framework for StudyFlow AI.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

# Type var for return type of decorated functions
T = TypeVar("T")

# Create a telemetry logger
logger = logging.getLogger("studyflow_ai.telemetry")

class StructuredJsonFormatter(logging.Formatter):
    """
    Formats log records as raw JSON strings containing:
    - timestamp
    - level
    - message
    - arbitrary key-value extra_fields
    """
    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Pull extra context fields if present
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            log_payload.update(extra_fields)
            
        return json.dumps(log_payload)

# Initialize standard stream handler with structured JSON formatting
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(StructuredJsonFormatter())
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
# Prevent telemetry logs from propagating to root logger to avoid double logging
logger.propagate = False


def log_structured(level: int, message: str, **kwargs: Any) -> None:
    """
    Safely logs a structured JSON message with additional telemetry context fields.
    """
    try:
        logger.log(level, message, extra={"extra_fields": kwargs})
    except Exception:
        # Telemetry failures must never propagate or crash main application flow
        pass


def trace_step(step_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Asynchronous decorator that measures the execution latency of wrapped coroutines in ms.
    Emits start and completion structured JSON logs containing step_name, duration_ms, and status.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Safe start log
            try:
                log_structured(
                    logging.INFO,
                    f"Starting step: {step_name}",
                    step_name=step_name,
                    status="START"
                )
            except Exception:
                pass

            start_time = time.perf_counter()
            status = "SUCCESS"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "ERROR"
                raise e
            finally:
                # Safe completion log measuring latency in ms
                try:
                    duration_ms = (time.perf_counter() - start_time) * 1000.0
                    log_structured(
                        logging.INFO,
                        f"Completed step: {step_name}",
                        step_name=step_name,
                        status=status,
                        duration_ms=round(duration_ms, 2)
                    )
                except Exception:
                    pass
        return wrapper
    return decorator
