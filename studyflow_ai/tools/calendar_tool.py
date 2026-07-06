"""
Calendar Sync Tool for StudyFlow AI.
Coordinates busy slot queries and task inserts via CalendarService.
"""
import logging
from typing import List
from datetime import datetime
from google.adk import tool
from studyflow_ai.services.calendar_service import CalendarService
from studyflow_ai.models.schemas import StudyTaskSchema

logger = logging.getLogger(__name__)

# Mock configuration
calendar_service = CalendarService(client_id="mock_id", client_secret="mock_secret")

@tool
async def query_calendar_events(user_id: str, start_date: str, end_date: str) -> List[dict]:
    """
    Fetches user busy slots from external calendars.
    
    Args:
        user_id: Unique identifier for the user
        start_date: Start search window boundary (ISO8601)
        end_date: End search window boundary (ISO8601)
    """
    try:
        return await calendar_service.get_busy_slots(user_id, start_date, end_date)
    except Exception as e:
        logger.error(f"Calendar query failed: {e}")
        return []

@tool
async def create_calendar_session(user_id: str, title: str, start_time: str, end_time: str) -> str:
    """
    Creates a new study session in the external calendar.
    
    Args:
        user_id: User UUID
        title: Title of the study block
        start_time: ISO8601 start timestamp
        end_time: ISO8601 end timestamp
    """
    try:
        res = await calendar_service.create_event(user_id, title, start_time, end_time)
        return res.get("event_id", "")
    except Exception as e:
        logger.error(f"Calendar creation failed: {e}")
        raise RuntimeError(f"Failed to create study session: {e}")
